"""ResNet-20 and a small ViT for CIFAR-10, each with explicit activation tap points.

Every place where the project plan wants to round an activation is an `Act` module. By default
an Act is the identity; the quantization code (quant.py) attaches a quantizer to it.
Each Act has a `cat` (category) used by experiment E4:

  ResNet-20 : first (image -> first conv, kept 8-bit), conv_in, last (-> fc, kept 8-bit)
  ViT       : first (image -> patch embed, kept 8-bit), ln_in (LayerNorm inputs = residual stream),
              linear_in (inputs to qkv, proj, fc1), softmax (attention probabilities),
              gelu (GELU outputs = fc2 inputs), matmul (Q, K, V entering the two attention matmuls),
              last (-> classifier head, kept 8-bit)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Act(nn.Module):
    """Activation tap point. mode: 'fp' (identity), 'calib' (record samples), 'quant' (apply quantizer)."""

    def __init__(self, cat, channel_dim=None):
        super().__init__()
        self.cat = cat
        self.channel_dim = channel_dim  # last-dim channels for ln_in (needed by PTF)
        self.mode = "fp"
        self.quantizer = None
        self.samples = []
        self.stats = None  # running min/max during calibration

    def forward(self, x):
        if self.mode == "calib":
            self._record(x.detach())
            return x
        if self.mode == "quant" and self.quantizer is not None:
            return self.quantizer(x)
        return x

    def _record(self, x, max_rows=4096):
        lo, hi = x.min().item(), x.max().item()
        if self.stats is None:
            self.stats = [lo, hi]
        else:
            self.stats = [min(self.stats[0], lo), max(self.stats[1], hi)]
        if self.channel_dim is not None:
            rows = x.reshape(-1, x.shape[-1])
        else:
            rows = x.reshape(-1)
        # keep a random subsample per batch (bounded memory)
        n = rows.shape[0]
        keep = max_rows if self.channel_dim is not None else max_rows * 64
        if n > keep:
            idx = torch.randperm(n, device=rows.device)[:keep]
            rows = rows[idx]
        self.samples.append(rows.float().cpu())
        # optional extra (token x channel) sample for per-channel PTF on non-LayerNorm locations.
        # Uses a private generator so the main sample above is unchanged.
        if getattr(self, "rows_for_ptf", False) and self.channel_dim is None:
            r2 = x.reshape(-1, x.shape[-1])
            g = getattr(self, "_gen", None)
            if g is None:
                g = self._gen = torch.Generator(device=r2.device).manual_seed(1234)
            if r2.shape[0] > 2048:
                r2 = r2[torch.randperm(r2.shape[0], generator=g, device=r2.device)[:2048]]
            self.chan_samples.append(r2.float().cpu())


# --------------------------------------------------------------------------------------------
# ResNet-20 (same parameter names as chenyaofo/pytorch-cifar-models, so its checkpoint loads)
# --------------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, stage=0):
        super().__init__()
        self.act_in = Act("conv_in")       # input to conv1 (and to the downsample conv)
        self.act_in.stage = stage
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act_mid = Act("conv_in")      # input to conv2
        self.act_mid.stage = stage
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                                            nn.BatchNorm2d(planes))

    def forward(self, x):
        xq = self.act_in(x)
        out = F.relu(self.bn1(self.conv1(xq)))
        out = self.bn2(self.conv2(self.act_mid(out)))
        identity = x if self.downsample is None else self.downsample(xq)
        return F.relu(out + identity)


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.act_first = Act("first")
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(*[BasicBlock(16, 16, 1, 1) for _ in range(3)])
        self.layer2 = nn.Sequential(BasicBlock(16, 32, 2, 2), BasicBlock(32, 32, 1, 2), BasicBlock(32, 32, 1, 2))
        self.layer3 = nn.Sequential(BasicBlock(32, 64, 2, 3), BasicBlock(64, 64, 1, 3), BasicBlock(64, 64, 1, 3))
        self.act_last = Act("last")
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(self.act_first(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(self.act_last(x))

    # layers whose weights stay at 8 bits (first and last, standard practice)
    def edge_layers(self):
        return {"conv1", "fc"}


# --------------------------------------------------------------------------------------------
# Small ViT: 4x4 patches (64 tokens + CLS), dim 192, depth 6, 3 heads, MLP 768
# --------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.act_qkv_in = Act("linear_in")
        self.qkv = nn.Linear(dim, dim * 3)
        self.act_q, self.act_k, self.act_v = Act("matmul"), Act("matmul"), Act("matmul")
        self.act_softmax = Act("softmax")
        self.act_proj_in = Act("linear_in")
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.act_qkv_in(x)).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = self.act_q(qkv[0]), self.act_k(qkv[1]), self.act_v(qkv[2])
        attn = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        attn = self.act_softmax(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(self.act_proj_in(out))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.act_ln1 = Act("ln_in", channel_dim=-1)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.act_ln2 = Act("ln_in", channel_dim=-1)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.act_fc1_in = Act("linear_in")
        self.fc1 = nn.Linear(dim, hidden)
        self.act_gelu = Act("gelu")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(self.act_ln1(x)))
        h = self.fc1(self.act_fc1_in(self.norm2(self.act_ln2(x))))
        return x + self.fc2(self.act_gelu(F.gelu(h)))


class ViT(nn.Module):
    def __init__(self, img=32, patch=4, dim=192, depth=6, heads=3, mlp_ratio=4.0, num_classes=10):
        super().__init__()
        self.act_first = Act("first")
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)  # == one linear layer per patch
        n = (img // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.act_ln_final = Act("ln_in", channel_dim=-1)
        self.norm = nn.LayerNorm(dim)
        self.act_last = Act("last")
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(self.act_first(x)).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1) + self.pos_embed
        for b in self.blocks:
            x = b(x)
        x = self.norm(self.act_ln_final(x))[:, 0]
        return self.head(self.act_last(x))

    def edge_layers(self):
        return {"patch_embed", "head"}


def build(name):
    return {"resnet20": ResNet20, "vit": ViT}[name]()


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def acts(model):
    """(name, Act) pairs in forward order."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, Act)]


@torch.no_grad()
def fold_bn(model):
    """Fold every BatchNorm into the conv before it (what an integer deployment actually runs).
    Returns a new model whose convs carry a bias and whose BNs are Identity."""
    import copy
    m = copy.deepcopy(model).eval()

    def fold(conv, bn):
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                        conv.padding, bias=True)
        new.weight.copy_(conv.weight * s.view(-1, 1, 1, 1))
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        new.bias.copy_((b - bn.running_mean) * s + bn.bias)
        return new

    m.conv1 = fold(m.conv1, m.bn1); m.bn1 = nn.Identity()
    for layer in [m.layer1, m.layer2, m.layer3]:
        for blk in layer:
            blk.conv1 = fold(blk.conv1, blk.bn1); blk.bn1 = nn.Identity()
            blk.conv2 = fold(blk.conv2, blk.bn2); blk.bn2 = nn.Identity()
            if blk.downsample is not None:
                blk.downsample = nn.Sequential(fold(blk.downsample[0], blk.downsample[1]), nn.Identity())
    return m
