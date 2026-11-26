"""Post-training fake quantization ("round to the ruler, then back to float").

Core formula from the project plan (symmetric, signed):
    scale = clip / (2^(bits-1) - 1)
    q     = clamp(round(x / scale), -(2^(bits-1)-1), 2^(bits-1)-1)
    x_hat = q * scale
Additions:
  * unsigned grid for tensors that are never negative (post-ReLU, softmax): levels 0..2^bits-1
  * asymmetric grid (zero point) for GELU outputs (min ~ -0.17, max large)
  * Log2 quantizer for softmax (FQ-ViT fix #1)
  * Power-of-Two Factor (PTF) per-channel quantizer for LayerNorm inputs (FQ-ViT fix #2)
  * clip ("ruler length") chosen by MSE search on calibration data (Nagel et al. white paper)
"""
import copy
import math
import torch
import torch.nn as nn

from .models import Act, acts

# ------------------------------------------------------------------ basic fake-quant functions


def fq_sym(x, scale, bits):
    qmax = 2 ** (bits - 1) - 1
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


def fq_unsigned(x, scale, bits):
    qmax = 2 ** bits - 1
    return torch.clamp(torch.round(x / scale), 0, qmax) * scale


def fq_asym(x, scale, zp, bits):
    qmax = 2 ** bits - 1
    return (torch.clamp(torch.round(x / scale) + zp, 0, qmax) - zp) * scale


def fq_log2(x, bits):
    """FQ-ViT log2 quantizer for values in [0, 1]: levels 1, 1/2, 1/4, ..., 2^-(2^bits - 1)."""
    q = torch.clamp(torch.round(-torch.log2(x.clamp_min(1e-30))), 0, 2 ** bits - 1)
    return torch.pow(2.0, -q)


class UniformQ(nn.Module):
    def __init__(self, bits, kind, scale, zp=0.0):
        super().__init__()
        self.bits, self.kind, self.scale, self.zp = bits, kind, float(scale), float(zp)

    def forward(self, x):
        if self.kind == "sym":
            return fq_sym(x, self.scale, self.bits)
        if self.kind == "unsigned":
            return fq_unsigned(x, self.scale, self.bits)
        return fq_asym(x, self.scale, self.zp, self.bits)

    def extra_repr(self):
        return f"{self.kind}, bits={self.bits}, scale={self.scale:.4g}, zp={self.zp:g}"


class Log2Q(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.bits = bits

    def forward(self, x):
        return fq_log2(x, self.bits)


class PTFQ(nn.Module):
    """Per-channel scale = base_scale * 2^alpha_c (alpha_c in {0..K}); channels on the last dim."""

    def __init__(self, bits, base_scale, alpha):
        super().__init__()
        self.bits = bits
        self.register_buffer("scale", base_scale * torch.pow(2.0, alpha.float()))

    def forward(self, x):
        return fq_sym(x, self.scale.to(x.device), self.bits)


# ------------------------------------------------------------------ range ("ruler length") search

RATIOS = torch.cat([torch.logspace(-2.5, -1, 25)[:-1], torch.linspace(0.1, 1.0, 91)])


def _mse(a, b):
    return ((a - b) ** 2).mean().item()


def calibrate_uniform(x, lo, hi, bits, kind, method="mse"):
    """Return a UniformQ. x: 1-D float sample of the tensor; lo/hi: true min/max seen."""
    ratios = RATIOS if method == "mse" else torch.tensor([1.0])
    best = (float("inf"), None)
    for r in ratios.tolist():
        if kind == "sym":
            clip = r * max(abs(lo), abs(hi))
            scale, zp = max(clip, 1e-12) / (2 ** (bits - 1) - 1), 0.0
        elif kind == "unsigned":
            scale, zp = max(r * hi, 1e-12) / (2 ** bits - 1), 0.0
        else:  # asym: keep the (small) negative side, search the positive clip
            top = max(r * hi, 1e-6)
            bot = min(lo, 0.0)
            scale = (top - bot) / (2 ** bits - 1)
            zp = float(round(-bot / scale))
        q = UniformQ(bits, kind, scale, zp)
        e = _mse(q(x), x)
        if e < best[0]:
            best = (e, q)
    return best[1]


def calibrate_ptf(X, bits, K=3):
    """X: (N, C) sample. Search base scale; each channel picks alpha in {0..K} minimizing its MSE."""
    m = X.abs().amax(0)
    qmax = 2 ** (bits - 1) - 1
    best = (float("inf"), None, None)
    for r in torch.linspace(0.05, 1.0, 40).tolist():
        s = r * m.max().item() / (qmax * 2 ** K)
        errs = torch.stack([((fq_sym(X, s * 2 ** a, bits) - X) ** 2).sum(0) for a in range(K + 1)])  # (K+1, C)
        e, alpha = errs.min(0)
        if e.sum().item() < best[0]:
            best = (e.sum().item(), s, alpha)
    return PTFQ(bits, best[1], best[2])


def sqnr_db(x, xq):
    return 10 * math.log10((x ** 2).sum().item() / max(((x - xq) ** 2).sum().item(), 1e-30))


# ------------------------------------------------------------------ weights


@torch.no_grad()
def quantize_weights(model, bits, method="max", edge_bits=8, skip_edges=True):
    """Return a copy of `model` with per-output-channel symmetric fake-quantized weights in every
    Conv2d / Linear. First and last layers use `edge_bits`. bits >= 32 -> unchanged copy."""
    qm = copy.deepcopy(model)
    if bits >= 32:
        return qm
    edges = model.edge_layers() if skip_edges else set()
    for name, mod in qm.named_modules():
        if not isinstance(mod, (nn.Conv2d, nn.Linear)):
            continue
        b = edge_bits if name in edges else bits
        w = mod.weight.data
        flat = w.reshape(w.shape[0], -1)
        maxabs = flat.abs().amax(1, keepdim=True).clamp_min(1e-12)
        qmax = 2 ** (b - 1) - 1
        if method == "max":
            scale = maxabs / qmax
        else:  # per-channel MSE search over clip ratios
            best_e = torch.full_like(maxabs, float("inf"))
            scale = maxabs / qmax
            for r in torch.linspace(0.2, 1.0, 81).tolist():
                s = r * maxabs / qmax
                e = ((fq_sym(flat, s, b) - flat) ** 2).sum(1, keepdim=True)
                better = e < best_e
                best_e = torch.where(better, e, best_e)
                scale = torch.where(better, s, scale)
        mod.weight.data = fq_sym(flat, scale, b).reshape(w.shape)
    return qm


def model_size_bits(model, wbits, edge_bits=8):
    """Weight storage in bits (params x bits; biases/LayerNorm/BN/pos-embed kept at 32 bits)."""
    edges = model.edge_layers()
    total = 0
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            b = 32 if wbits >= 32 else (edge_bits if name in edges else wbits)
            total += mod.weight.numel() * b
            if mod.bias is not None:
                total += mod.bias.numel() * 32
        elif isinstance(mod, (nn.LayerNorm, nn.BatchNorm2d)):
            total += sum(p.numel() for p in mod.parameters()) * 32
    for n, p in model.named_parameters(recurse=False):
        total += p.numel() * 32  # cls token, pos embed
    return total


# ------------------------------------------------------------------ activations


@torch.no_grad()
def collect(model, x_calib, batch=128):
    """Run calibration images through the model, recording samples at every Act."""
    model.eval()
    torch.manual_seed(0)  # the per-batch subsampling in Act._record is random: make it reproducible
    for _, a in acts(model):
        a.samples, a.stats, a.mode, a.quantizer = [], None, "calib", None
        a.chan_samples, a._gen = [], None
    for i in range(0, len(x_calib), batch):
        model(x_calib[i:i + batch])
    for _, a in acts(model):
        a.mode = "fp"
        a.sample = torch.cat(a.samples, 0)
        a.chan_sample = torch.cat(a.chan_samples, 0) if a.chan_samples else None
        a.samples, a.chan_samples = [], []


def kind_for(a, scheme):
    if scheme == "plan":  # exactly the proposal: one symmetric signed formula everywhere
        return "sym"
    lo, _ = a.stats
    if lo >= 0:
        return "unsigned"
    if a.cat == "gelu":
        return "asym"
    return "sym"


@torch.no_grad()
def set_act_quant(model, abits, cats=None, names=None, scheme="improved", softmax_fix=None,
                  ln_fix=None, edge_bits=8, quantize_edges=True, method="mse", cache=None, ptf_cats=()):
    """Attach quantizers to Acts. Must call collect() first.
    cats / names: which locations to quantize (None = all non-edge locations).
    softmax_fix='log2' -> Log2Q at softmax; ln_fix='ptf' -> PTF at LayerNorm inputs.
    Edge locations ('first', 'last') get edge_bits when quantize_edges."""
    for name, a in acts(model):
        a.mode, a.quantizer, a.sqnr = "fp", None, None
        is_edge = a.cat in ("first", "last")
        if is_edge:
            if not quantize_edges or abits >= 32:
                continue
            b = edge_bits
        else:
            if abits >= 32:
                continue
            if cats is not None and a.cat not in cats:
                continue
            if names is not None and name not in names:
                continue
            b = abits
        x = a.sample
        if a.cat == "softmax" and softmax_fix == "log2":
            key = (name, b, "log2")
        elif (a.cat == "ln_in" and ln_fix == "ptf") or a.cat in ptf_cats:
            key = (name, b, "ptf")
        else:
            key = (name, b, kind_for(a, scheme), method)
        if cache is not None and key in cache:
            q, s = cache[key]
        else:
            if key[2] == "log2":
                q = Log2Q(b)
            elif key[2] == "ptf":
                q = calibrate_ptf(x if a.channel_dim is not None else a.chan_sample, b)
            else:
                q = calibrate_uniform(x.reshape(-1), a.stats[0], a.stats[1], b, key[2], method)
            xs = a.chan_sample if (key[2] == "ptf" and a.channel_dim is None) else x
            s = sqnr_db(xs, q(xs))
            if cache is not None:
                cache[key] = (q, s)
        a.quantizer, a.mode, a.sqnr = q, "quant", s


def clear(model):
    for _, a in acts(model):
        a.mode, a.quantizer = "fp", None
