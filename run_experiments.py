"""Experiments E1-E4 and E6 (E5 histograms are in analyze.py). Results go to results/<model>.json
(written after every run; re-running skips finished keys).

  python run_experiments.py --model resnet20 --ckpt ckpt_resnet20.pt
  python run_experiments.py --model vit --ckpt ckpt_vit.pt
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qproj import data as D, models as M, quant as Q

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--ckpt", required=True)
p.add_argument("--seeds", type=int, default=3, help="calibration-set seeds")
p.add_argument("--only", default="E1,E2,E3,E4,E6,E6b,E4L")
p.add_argument("--data", default="data")
p.add_argument("--out", default=None)
p.add_argument("--quick", action="store_true", help="smoke test on 1000 test images")
args = p.parse_args()
out = args.out or f"results/{args.model}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

dset = D.load_cifar(args.data)
S = D.splits(dset)
xt, yt = S["test"]
if args.quick:
    xt, yt = xt[:1000], yt[:1000]
model = M.build(args.model)
model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
if args.model == "resnet20":
    model = M.fold_bn(model)  # integer deployment runs folded conv+BN
model = model.to(dev).eval()
is_vit = args.model == "vit"
for _, a in M.acts(model):  # extra per-channel samples for the PTF extension (E6b)
    if a.cat in ("linear_in", "gelu"):
        a.rows_for_ptf = True

res = json.load(open(out)) if os.path.exists(out) else {}


def save():
    json.dump(res, open(out, "w"), indent=1)


_wq = {}
_cur = {"key": None, "model": None, "cache": None}


def wq(wbits, wmethod):
    k = (wbits, wmethod)
    if k not in _wq:
        _wq[k] = Q.quantize_weights(model, wbits, wmethod).to(dev)
    return _wq[k]


def collected(wbits, wmethod, seed):
    k = (wbits, wmethod, seed)
    if _cur["key"] != k:
        qm = wq(wbits, wmethod)
        xc = D.to_tensor(D.splits(dset, calib_seed=seed)["calib"][0]).to(dev)
        Q.collect(qm, xc)
        _cur.update(key=k, model=qm, cache={})
    return _cur["model"], _cur["cache"]


def run(key, wbits=32, abits=32, wmethod="max", seed=0, x=None, y=None, **akw):
    if key in res:
        return res[key]
    t = time.time()
    if abits < 32:
        qm, cache = collected(wbits, wmethod, seed)
        Q.set_act_quant(qm, abits, cache=cache, **akw)
    else:
        qm = wq(wbits, wmethod)
        Q.clear(qm)
    acc = float(D.evaluate(qm, xt if x is None else x, yt if y is None else y, device=dev))
    r = dict(acc=acc, wbits=wbits, abits=abits, wmethod=wmethod, seed=seed,
             **{k: (sorted(v) if isinstance(v, set) else v) for k, v in akw.items() if k != "names"})
    if abits < 32:
        r["sqnr"] = {n: a.sqnr for n, a in M.acts(qm) if a.mode == "quant"}
        Q.clear(qm)
    res[key] = r
    save()
    print(f"{key:45s} acc={acc:6.2f}  ({time.time() - t:.0f}s)", flush=True)
    return r


only = set(args.only.split(","))
seeds = list(range(args.seeds))

# E1: full precision
if "E1" in only:
    run("E1/fp")
    if "E1/val" not in res:
        res["E1/val"] = {"acc": float(D.evaluate(model, *S["val"], device=dev))}; save()

# E2: weights only
if "E2" in only:
    for wm in ["max", "mse"]:
        for b in [8, 6, 4, 2]:
            run(f"E2/{wm}/W{b}", b, 32, wm)

# E3: weights + activations (plan scheme = symmetric everywhere; improved = unsigned/asym where fitting)
if "E3" in only:
    for b in [8, 6, 4]:
        run(f"E3/actonly/W32A{b}/s0", 32, b)
    for s in seeds:
        for b in [8, 6, 4]:
            run(f"E3/improved/W{b}A{b}/s{s}", b, b, seed=s)
    for b in [8, 6, 4]:
        run(f"E3/plan/W{b}A{b}/s0", b, b, scheme="plan")

# E4: where does it break? weights full precision, activations at 4 (and 6) bits
if "E4" in only:
    if is_vit:
        groups = {c: dict(cats={c}) for c in ["linear_in", "ln_in", "softmax", "gelu", "matmul"]}
    else:
        stage_names = {s: {n for n, a in M.acts(model) if getattr(a, "stage", None) == s} for s in [1, 2, 3]}
        groups = {f"stage{s}": dict(names=stage_names[s]) for s in [1, 2, 3]}
    allnames = {n for n, a in M.acts(model) if a.cat not in ("first", "last")}
    for ab in [4, 6]:
        for g, sel in groups.items():
            # only this location quantized (no edges), everything else full precision
            run(f"E4/A{ab}/only/{g}", 32, ab, quantize_edges=False, **sel)
            # everything quantized except this location
            if "cats" in sel:
                keep = {n for n, a in M.acts(model) if a.cat not in ("first", "last") and a.cat not in sel["cats"]}
            else:
                keep = allnames - sel["names"]
            run(f"E4/A{ab}/allbut/{g}", 32, ab, names=keep)

# E4L: per-layer sensitivity at A4 (each single location alone), on the first 2000 test images
if "E4L" in only:
    run("E4L/base", x=xt[:2000], y=yt[:2000])
    for n, a in M.acts(model):
        if a.cat in ("first", "last"):
            continue
        run(f"E4L/A4/{n}", 32, 4, names={n}, quantize_edges=False, x=xt[:2000], y=yt[:2000])

# E6: fixes
if "E6" in only:
    fixes = {"none": {}, "mseW": {}}
    if is_vit:
        fixes.update({"log2": dict(softmax_fix="log2"), "ptf": dict(ln_fix="ptf"),
                      "log2+ptf": dict(softmax_fix="log2", ln_fix="ptf"),
                      "log2+ptf+mseW": dict(softmax_fix="log2", ln_fix="ptf")})
    for (wb, ab) in [(8, 4), (4, 4), (6, 6)]:
        for s in (seeds if (wb, ab) != (6, 6) else [0]):
            for f, kw in fixes.items():
                wm = "mse" if "mseW" in f else "max"
                if wm == "mse" and wb == 8:
                    continue
                run(f"E6/W{wb}A{ab}/{f}/s{s}", wb, ab, wmethod=wm, seed=s, **kw)

# E6b (extension): PTF also on linear-layer inputs and GELU outputs
if "E6b" in only and is_vit:
    for (wb, ab) in [(8, 4), (4, 4)]:
        for s in seeds:
            run(f"E6/W{wb}A{ab}/ptf+linPTF/s{s}", wb, ab, seed=s, ln_fix="ptf", ptf_cats={"linear_in"})
            run(f"E6/W{wb}A{ab}/ptf+linPTF+geluPTF/s{s}", wb, ab, seed=s, ln_fix="ptf", ptf_cats={"linear_in", "gelu"})

print("done")
