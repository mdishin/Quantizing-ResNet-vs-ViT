"""Full-precision training (experiment E1).

CPU run used here (2 cores, bf16 autocast):
  python train.py --model vit --epochs 45 --out ckpt_vit.pt
Recommended Colab GPU runs:
  python train.py --model vit --epochs 200 --randaug --batch 256 --lr 1e-3 --seed 0
  python train.py --model resnet20 --epochs 160 --opt sgd --lr 0.1 --wd 1e-4 --seed 0
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qproj import data as D, models as M

p = argparse.ArgumentParser()
p.add_argument("--model", default="vit")
p.add_argument("--epochs", type=int, default=45)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--opt", default="adamw")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.05)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--smoothing", type=float, default=0.1)
p.add_argument("--randaug", action="store_true")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default="data")
p.add_argument("--out", default=None)
args = p.parse_args()
out = args.out or f"ckpt_{args.model}_s{args.seed}.pt"

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
amp_dtype = torch.float16 if dev == "cuda" else torch.bfloat16
S = D.splits(D.load_cifar(args.data))
xtr, ytr = S["train"]
model = M.build(args.model).to(dev)
if args.opt == "sgd":
    opt = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
else:
    decay = [p_ for n, p_ in model.named_parameters() if p_.ndim > 1 and "pos_embed" not in n and "cls_token" not in n]
    no_decay = [p_ for n, p_ in model.named_parameters() if not (p_.ndim > 1 and "pos_embed" not in n and "cls_token" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}], args.lr)
steps_per_epoch = len(ytr) // args.batch
total = args.epochs * steps_per_epoch
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / (args.warmup * steps_per_epoch)
                                          if s < args.warmup * steps_per_epoch
                                          else 0.5 * (1 + math.cos(math.pi * (s - args.warmup * steps_per_epoch) / max(1, total - args.warmup * steps_per_epoch))))
scaler = torch.amp.GradScaler(enabled=dev == "cuda")

start, log = 0, []
if os.path.exists(out + ".resume"):
    # load on CPU: load_state_dict moves model/optimizer tensors to the GPU itself, and the RNG state
    # must stay a CPU ByteTensor (loading it straight to CUDA breaks set_rng_state)
    st = torch.load(out + ".resume", map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    start, log = st["epoch"], st["log"]
    torch.set_rng_state(st["rng"].cpu())
    print(f"resumed at epoch {start}")

if args.randaug:
    import torchvision.transforms.v2 as T2
    randaug = T2.RandAugment(num_ops=2, magnitude=9)

xtr_t = torch.from_numpy(xtr).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
ytr_t = torch.from_numpy(ytr)
mean = torch.tensor(D.MEAN, device=dev).view(1, 3, 1, 1)
std = torch.tensor(D.STD, device=dev).view(1, 3, 1, 1)


def augment(xb):
    """Batched random crop (pad 4) + horizontal flip on uint8 NCHW; optional per-sample RandAugment."""
    B = xb.shape[0]
    if args.randaug:
        xb = torch.stack([randaug(img) for img in xb])
    xp = F.pad(xb, (4, 4, 4, 4))
    i = torch.randint(0, 9, (B,)); j = torch.randint(0, 9, (B,))
    rows = (i.view(B, 1) + torch.arange(32)).view(B, 1, 32, 1).expand(B, 3, 32, 40)
    cols = (j.view(B, 1) + torch.arange(32)).view(B, 1, 1, 32).expand(B, 3, 32, 32)
    xb = xp.gather(2, rows).gather(3, cols)
    flip = torch.rand(B) < 0.5
    xb[flip] = xb[flip].flip(3)
    return xb


for ep in range(start, args.epochs):
    model.train(); t0 = time.time(); tot, n = 0.0, 0
    perm = torch.randperm(len(ytr_t))
    for s in range(steps_per_epoch):
        idx = perm[s * args.batch:(s + 1) * args.batch]
        xb = augment(xtr_t[idx]).to(dev).float().div_(255)
        xb = (xb - mean) / std
        yb = ytr_t[idx].to(dev)
        with torch.autocast(dev, dtype=amp_dtype):
            loss = F.cross_entropy(model(xb), yb, label_smoothing=args.smoothing)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * len(idx); n += len(idx)
    rec = {"epoch": ep + 1, "loss": tot / n, "lr": sched.get_last_lr()[0], "sec": time.time() - t0}
    if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
        rec["val"] = float(D.evaluate(model, *S["val"], device=dev))
    log.append(rec); print(json.dumps(rec), flush=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": ep + 1, "log": log, "rng": torch.get_rng_state()}, out + ".resume")

model.eval()
res = {"val": float(D.evaluate(model, *S["val"], device=dev)), "test": float(D.evaluate(model, *S["test"], device=dev)),
       "params": M.n_params(model), "args": vars(args), "log": log}
torch.save(model.state_dict(), out)
json.dump(res, open(out.replace(".pt", "_train.json"), "w"), indent=1)
print("FINAL", json.dumps({k: res[k] for k in ["val", "test", "params"]}))
