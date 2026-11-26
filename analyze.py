"""E5 (look at the values) + all figures and summary tables.
  python analyze.py      # needs results/resnet20.json, results/vit.json, both checkpoints
"""
import json, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qproj import data as D, models as M, quant as Q

os.makedirs("figures", exist_ok=True)
C = {"resnet20": "#2a78d6", "vit": "#eb6834", "aqua": "#1baf7a", "violet": "#6250d6", "muted": "#8a8984"}
NAME = {"resnet20": "ResNet-20", "vit": "ViT (6 blocks)"}
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#b9b8b3",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "text.color": "#0b0b0b", "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False})

R = {m: json.load(open(f"results/{m}.json")) for m in ["resnet20", "vit"]}
FP = {m: R[m]["E1/fp"]["acc"] for m in R}


def seeds_of(m, prefix):
    v = [r["acc"] for k, r in R[m].items() if k.startswith(prefix + "/s")]
    return np.array(v)


summary = {"fp": FP}

# ------------------------------------------------------------- Figure 1: accuracy drop vs bit-width
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for m in R:
    bits = [8, 6, 4, 2]
    drop = [FP[m] - R[m][f"E2/max/W{b}"]["acc"] for b in bits]
    axes[0].plot(bits, drop, "-o", color=C[m], lw=2, ms=8, mec="#fcfcfb", mew=2, label=NAME[m])
    summary.setdefault("E2", {})[m] = dict(zip(bits, [R[m][f"E2/max/W{b}"]["acc"] for b in bits]))
    bits3 = [8, 6, 4]
    mu = np.array([FP[m] - seeds_of(m, f"E3/improved/W{b}A{b}").mean() for b in bits3])
    sd = np.array([seeds_of(m, f"E3/improved/W{b}A{b}").std() for b in bits3])
    axes[1].errorbar(bits3, mu, yerr=sd, fmt="-o", color=C[m], lw=2, ms=8, mec="#fcfcfb", mew=2, capsize=3, label=NAME[m])
    plan = [FP[m] - R[m][f"E3/plan/W{b}A{b}/s0"]["acc"] for b in bits3]
    axes[1].plot(bits3, plan, "--", color=C[m], lw=1.5, alpha=0.7, marker="s", ms=6, label=f"{NAME[m]}, plan's signed grid")
    summary.setdefault("E3", {})[m] = {b: (float(FP[m] - x), float(s)) for b, x, s in zip(bits3, mu, sd)}
for ax, t in zip(axes, ["E2: weights only (activations 32-bit)", "E3: weights + activations (W = A bits)"]):
    ax.set_title(t, loc="left")
    ax.set_xlabel("bits")
    ax.invert_xaxis()
    ax.set_xticks([8, 6, 4, 2] if ax is axes[0] else [8, 6, 4])
axes[0].set_ylabel("accuracy drop from own FP32 (points)")
axes[0].set_yscale("symlog", linthresh=1)
axes[0].set_ylim(-0.3, 100)
axes[0].set_yticks([0, 1, 3, 10, 30, 100], ["0", "1", "3", "10", "30", "100"])
axes[0].legend(loc="upper left")
axes[1].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig1_drop_vs_bits.png", dpi=180)
plt.close(fig)

# ------------------------------------------------------------- Figure 2: where does it break (E4)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [5, 3]})
for ax, m in zip(axes, ["vit", "resnet20"]):
    keys = sorted({k.split("/")[3] for k in R[m] if k.startswith("E4/A4/only/")})
    only = [FP[m] - R[m][f"E4/A4/only/{g}"]["acc"] for g in keys]
    allbut = [FP[m] - R[m][f"E4/A4/allbut/{g}"]["acc"] for g in keys]
    allq = FP[m] - R[m]["E3/actonly/W32A4/s0"]["acc"]
    y = np.arange(len(keys))
    ax.barh(y + 0.2, only, 0.38, color=C[m], label="only this location at 4 bits")
    ax.barh(y - 0.2, allbut, 0.38, color=C["muted"], label="all at 4 bits except this location")
    ax.axvline(allq, color="#0b0b0b", lw=1, ls=":")
    ax.text(allq, len(keys) - 0.45, f" all A4: {allq:.1f}", fontsize=8, va="bottom")
    ax.set_yticks(y, keys)
    ax.set_xlabel("accuracy drop (points), weights FP32")
    ax.set_title(f"E4: {NAME[m]}", loc="left")
    for yy, v in zip(y + 0.2, only):
        ax.text(v, yy, f" {v:.1f}", va="center", fontsize=8, color="#52514e")
    summary.setdefault("E4", {})[m] = {g: {"only": o, "allbut": a} for g, o, a in zip(keys, only, allbut)}
    summary["E4"][m]["all_A4"] = allq
axes[0].legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig2_where_it_breaks.png", dpi=180)
plt.close(fig)

# ------------------------------------------------------------- Figure 3: E6 fixes
m = "vit"
fixes = ["none", "log2", "ptf", "log2+ptf", "ptf+linPTF", "ptf+linPTF+geluPTF", "mseW", "log2+ptf+mseW"]
fig, ax = plt.subplots(figsize=(11, 3.8))
settings = [("W8A4", C["violet"]), ("W4A4", C["vit"])]
summary["E6"] = {}
for i, (st, col) in enumerate(settings):
    vals, errs, labs = [], [], []
    for f in fixes:
        a = seeds_of(m, f"E6/{st}/{f}")
        if len(a) == 0:
            continue
        vals.append(a.mean()); errs.append(a.std()); labs.append(f)
        summary["E6"].setdefault(st, {})[f] = (float(a.mean()), float(a.std()), int(len(a)))
    x = np.array([fixes.index(l) for l in labs]) + (i - 0.5) * 0.38
    ax.bar(x, vals, 0.36, yerr=errs, capsize=3, color=col, label=f"ViT {st}")
    for xx, v in zip(x, vals):
        ax.text(xx, v + 1, f"{v:.1f}", ha="center", fontsize=8, color="#52514e")
ax.axhline(FP[m], color="#0b0b0b", lw=1, ls=":")
ax.text(len(fixes) - 0.6, FP[m] + 1, f"FP32 {FP[m]:.1f}", fontsize=8, ha="right")
ax.set_xticks(np.arange(len(fixes)), [f.replace("+", "+\n") for f in fixes], fontsize=8)
ax.set_ylabel("test accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("E6: fixes for the ViT (mean ± sd over 3 calibration sets)", loc="left")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig4_fixes.png", dpi=180)
plt.close(fig)
for st in ["W8A4", "W4A4"]:
    a = seeds_of("resnet20", f"E6/{st}/none")
    b = seeds_of("resnet20", f"E6/{st}/mseW")
    summary["E6"].setdefault("resnet_" + st, {})["none"] = (float(a.mean()), float(a.std()), int(len(a)))
    if len(b):
        summary["E6"]["resnet_" + st]["mseW"] = (float(b.mean()), float(b.std()), int(len(b)))

# ------------------------------------------------------------- Figure 5: per-layer sensitivity (E4L)
fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), gridspec_kw={"height_ratios": [3, 1.2]})
catcol = {"linear_in": C["resnet20"], "ln_in": C["vit"], "softmax": C["aqua"], "gelu": "#eda100",
          "matmul": C["violet"], "conv_in": C["resnet20"]}
for ax, m in zip(axes, ["vit", "resnet20"]):
    base = R[m].get("E4L/base")
    items = [(k[len("E4L/A4/"):], r) for k, r in R[m].items() if k.startswith("E4L/A4/")]
    ref = FP[m] if base is None else base["acc"]
    names = [n for n, _ in items]
    mm = M.build(m)
    cat = {n: a.cat for n, a in M.acts(mm)}
    d = [ref - r["acc"] for _, r in items]
    ax.bar(range(len(d)), d, color=[catcol[cat[n]] for n in names], width=0.8)
    ax.set_xticks([])
    ax.set_ylabel("drop (pts)")
    if m == "vit":
        ax.set_ylim(-0.9, 3.6)
    ax.set_title(f"Per-location sensitivity at 4 bits, one location at a time: {NAME[m]} (first 2,000 test images)", loc="left", fontsize=10)
    if m == "vit":
        from matplotlib.patches import Patch
        cats_present = [c for c in ["ln_in", "linear_in", "matmul", "softmax", "gelu"]]
        ax.legend(handles=[Patch(color=catcol[c], label=c) for c in cats_present], ncol=5, fontsize=8, loc="upper right", bbox_to_anchor=(1, 1.0))
        # block boundaries
        bidx = [i for i, n in enumerate(names) if n.endswith("act_ln1")]
        for j, i in enumerate(bidx):
            ax.axvline(i - 0.5, color="#e8e7e3", lw=1, zorder=0)
            ax.text(i, ax.get_ylim()[0], f" block {j}", fontsize=7.5, color="#52514e", va="bottom")
        top = sorted(zip(d, names), reverse=True)[:6]
        summary["E4L_top_vit"] = [(n, float(v)) for v, n in top]
    else:
        top = sorted(zip(d, names), reverse=True)[:4]
        summary["E4L_top_resnet"] = [(n, float(v)) for v, n in top]
fig.tight_layout()
fig.savefig("figures/fig5_per_layer.png", dpi=180)
plt.close(fig)

# ------------------------------------------------------------- E5: look at the values
S = D.splits(D.load_cifar("data"))
xc = D.to_tensor(S["calib"][0])
vit = M.ViT(); vit.load_state_dict(torch.load("ckpt_vit.pt", map_location="cpu")); vit.eval()
rn = M.ResNet20(); rn.load_state_dict(torch.load("ckpt_resnet20.pt", map_location="cpu")); rn = M.fold_bn(rn)
Q.collect(vit, xc)

# ResNet conv inputs, per channel (NCHW): hook on conv layers
rn_ranges = {}
hooks = []
for n, mod in rn.named_modules():
    if isinstance(mod, torch.nn.Conv2d) and n != "conv1":
        hooks.append(mod.register_forward_pre_hook(
            lambda mod, inp, n=n: rn_ranges.setdefault(n, []).append(inp[0].abs().amax((0, 2, 3)))))
with torch.no_grad():
    for i in range(0, 512, 128):
        rn(xc[i:i + 128])
for h in hooks:
    h.remove()
rn_ranges = {n: torch.stack(v).amax(0) for n, v in rn_ranges.items()}

e5 = {"vit_ln_in": {}, "resnet_conv_in": {}}
for n, a in M.acts(vit):
    if a.cat == "ln_in":
        r = a.sample.abs().amax(0)
        e5["vit_ln_in"][n] = {"max_over_median": float(r.max() / r.median()), "top_channel": int(r.argmax()),
                              "max": float(r.max()), "median": float(r.median())}
for n, r in rn_ranges.items():
    e5["resnet_conv_in"][n] = {"max_over_median": float(r.max() / r.median().clamp_min(1e-6))}
# softmax: how much attention mass a 4-bit uniform grid zeros out
acts_v = dict(M.acts(vit))
sm = torch.cat([acts_v[f"blocks.{i}.attn.act_softmax"].sample for i in range(6)])
qu = Q.calibrate_uniform(sm, 0.0, 1.0, 4, "unsigned")
e5["softmax"] = {
    "frac_below_half_step_uniform4": float((sm < qu.scale / 2).float().mean()),
    "uniform4_step": qu.scale,
    "frac_below_2^-15": float((sm < 2 ** -15).float().mean()),
    "median": float(sm.median()), "mean": float(sm.mean()),
    "frac_mass_zeroed_uniform4": float((sm * (sm < qu.scale / 2)).sum() / sm.sum()),
}
gel = torch.cat([acts_v[f"blocks.{i}.act_gelu"].sample for i in range(6)])
e5["gelu"] = {"min": float(gel.min()), "max": float(gel.max()), "frac_negative": float((gel < 0).float().mean()),
              "p99": float(gel.quantile(0.99)) if gel.numel() < 16_000_000 else None}
summary["E5"] = e5

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
# (a) per-channel ranges, sorted, normalized by median
ax = axes[0]
worst = max(e5["vit_ln_in"], key=lambda n: e5["vit_ln_in"][n]["max_over_median"])
r = acts_v[worst].sample.abs().amax(0)
r = (r / r.median()).sort(descending=True).values.numpy()
ax.plot(np.arange(len(r)) / (len(r) - 1), r, color=C["vit"], lw=2, label=f"ViT LayerNorm input ({worst.replace('blocks.', 'blk ')})")
wr = max(rn_ranges, key=lambda n: e5["resnet_conv_in"][n]["max_over_median"])
r2 = rn_ranges[wr]
r2 = (r2 / r2.median()).sort(descending=True).values.numpy()
ax.plot(np.arange(len(r2)) / (len(r2) - 1), r2, color=C["resnet20"], lw=2, label=f"ResNet conv input ({wr})")
ax.set_yscale("log")
ax.set_xlabel("channels, sorted by range (fraction)")
ax.set_ylabel("channel max |x| / median channel")
ax.set_title("(a) channel ranges: worst layer of each", loc="left", fontsize=10)
ax.legend(fontsize=7.5)
# (b) softmax values with the two 4-bit grids
ax = axes[1]
v = sm.clamp_min(1e-7).numpy()
ax.hist(np.log2(v), bins=120, color=C["aqua"], alpha=0.9)
ax.axvline(np.log2(qu.scale / 2), color="#0b0b0b", ls=":", lw=1.2)
ax.text(np.log2(qu.scale / 2), ax.get_ylim()[1] * 0.92, " uniform 4-bit:\n below = rounds to 0", fontsize=7.5)
ax.axvline(-15, color=C["violet"], ls="--", lw=1.2)
ax.text(-15, ax.get_ylim()[1] * 0.6, " log2 4-bit\n smallest level", fontsize=7.5, color=C["violet"])
ax.set_xlabel("log2(attention probability)")
ax.set_ylabel("count")
ax.set_title("(b) softmax outputs, all blocks", loc="left", fontsize=10)
# (c) GELU
ax = axes[2]
g = gel.numpy()
ax.hist(g, bins=200, color="#eda100", log=True)
ax.set_xlabel("GELU output")
ax.set_ylabel("count (log)")
ax.set_title("(c) GELU outputs: lopsided range", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig("figures/fig3_values.png", dpi=180)
plt.close(fig)

# ------------------------------------------------------------- model size table
sizes = {}
for m, mod in [("resnet20", M.ResNet20()), ("vit", M.ViT())]:
    sizes[m] = {b: Q.model_size_bits(mod, b) / 8 / 1024 for b in [32, 8, 6, 4, 2]}
summary["size_KiB"] = sizes
json.dump(summary, open("results/summary.json", "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float)[:6000])
