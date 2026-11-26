"""Final-report figures: mean ± sd across 3 trained models of each kind."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

S = json.load(open("final_stats.json"))
C = {"resnet20": "#2a78d6", "vit": "#eb6834", "aqua": "#1baf7a", "violet": "#6250d6", "muted": "#8a8984", "gold": "#eda100"}
NAME = {"resnet20": "ResNet-20", "vit": "ViT"}
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#b9b8b3",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "text.color": "#0b0b0b", "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False})
mk = dict(lw=2, ms=8, mec="#fcfcfb", mew=2, capsize=3)

# ---------------- Figure 1: three panels, weights only / activations only / both
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
for m in ["resnet20", "vit"]:
    b = [8, 6, 4, 2]
    v = [S[f"{m}/drop/E2/max/W{x}"] for x in b]
    axes[0].errorbar(b, [x[0] for x in v], [x[1] for x in v], fmt="-o", color=C[m], label=NAME[m], **mk)
    b = [8, 6, 4]
    v = [S[f"{m}/drop/E3/actonly/W32A{x}/s0"] for x in b]
    axes[1].errorbar(b, [x[0] for x in v], [x[1] for x in v], fmt="-o", color=C[m], label=NAME[m], **mk)
    v = [S[f"{m}/drop/E3/improved/W{x}A{x}"] for x in b]
    axes[2].errorbar(b, [x[0] for x in v], [x[1] for x in v], fmt="-o", color=C[m], label=NAME[m], **mk)
    for ax, x, val in [(axes[1], 4, S[f"{m}/drop/E3/actonly/W32A4/s0"][0]), (axes[2], 4, S[f"{m}/drop/E3/improved/W4A4"][0])]:
        ax.annotate(f"{val:.1f}", (x, val), textcoords="offset points", xytext=(8, -3), fontsize=9, color="#52514e")
axes[0].annotate(f"{S['resnet20/drop/E2/max/W4'][0]:.1f}", (4, S['resnet20/drop/E2/max/W4'][0]), textcoords="offset points", xytext=(8, -3), fontsize=9, color="#52514e")
axes[0].annotate(f"{S['vit/drop/E2/max/W4'][0]:.1f}", (4, S['vit/drop/E2/max/W4'][0]), textcoords="offset points", xytext=(8, -3), fontsize=9, color="#52514e")
titles = ["(a) Weights only (activations 32-bit)", "(b) Activations only (weights 32-bit)", "(c) Weights + activations (W = A)"]
for ax, t in zip(axes, titles):
    ax.set_title(t, loc="left", fontsize=10.5)
    ax.set_xlabel("bits")
    ax.invert_xaxis()
axes[0].set_xticks([8, 6, 4, 2]); axes[1].set_xticks([8, 6, 4]); axes[2].set_xticks([8, 6, 4])
axes[0].set_yscale("symlog", linthresh=1)
axes[0].set_ylim(-0.3, 100)
axes[0].set_yticks([0, 1, 3, 10, 30, 100], ["0", "1", "3", "10", "30", "100"])
axes[0].set_ylabel("accuracy drop from own FP32 (points)")
axes[0].legend(loc="upper left")
fig.text(0.99, 0.01, "mean ± sd over 3 trained models; log scale above 1", ha="right", fontsize=8, color="#8a8984")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("fig1_drop_vs_bits.png", dpi=180); plt.close(fig)

# ---------------- Figure 2: E4 where it breaks
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [5, 3]})
for ax, m, groups in [(axes[0], "vit", ["ln_in", "linear_in", "gelu", "softmax", "matmul"]),
                      (axes[1], "resnet20", ["stage1", "stage2", "stage3"])]:
    labels = {"ln_in": "LayerNorm inputs", "linear_in": "linear inputs", "gelu": "GELU outputs", "softmax": "softmax", "matmul": "Q, K, V",
              "stage1": "stage 1", "stage2": "stage 2", "stage3": "stage 3"}
    y = np.arange(len(groups))[::-1]
    o = [S[f"{m}/E4/A4/only/{g}"] for g in groups]
    a = [S[f"{m}/E4/A4/allbut/{g}"] for g in groups]
    ax.barh(y + 0.2, [x[0] for x in o], 0.38, xerr=[x[1] for x in o], color=C[m], capsize=2, label="only this location at 4 bits")
    ax.barh(y - 0.2, [x[0] for x in a], 0.38, xerr=[x[1] for x in a], color=C["muted"], capsize=2, label="everything at 4 bits except this")
    allq = S[f"{m}/drop/E3/actonly/W32A4/s0"][0]
    ax.axvline(allq, color="#0b0b0b", lw=1, ls=":")
    ax.text(allq, y.max() + 0.55, f" all at 4 bits: {allq:.1f}", fontsize=8, va="bottom")
    for yy, v in zip(y + 0.2, o):
        ax.text(v[0] + v[1] + 0.05 * allq, yy, f"{v[0]:.1f}", va="center", fontsize=8, color="#52514e")
    ax.set_yticks(y, [labels[g] for g in groups])
    ax.set_xlabel("accuracy drop (points), weights FP32")
    ax.set_title(f"{NAME[m]}", loc="left")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("E4: which activations break at 4 bits (mean ± sd over 3 trained models)", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("fig2_where_it_breaks.png", dpi=180); plt.close(fig)

# ---------------- Figure 4: fixes, paired recovery
fixes = ["log2", "ptf", "log2+ptf", "ptf+linPTF", "ptf+linPTF+geluPTF"]
flab = {"log2": "log2 softmax\n(FQ-ViT)", "ptf": "PTF on LN inputs\n(FQ-ViT)", "log2+ptf": "log2 + PTF",
        "ptf+linPTF": "PTF + linear\ninputs (ours)", "ptf+linPTF+geluPTF": "PTF + linear\n+ GELU (ours)"}
fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
x = np.arange(len(fixes))
for i, (st, col) in enumerate([("W8A4", C["violet"]), ("W4A4", C["vit"])]):
    v = [S[f"vit/E6/{st}/{f}"]["recov"] for f in fixes]
    xx = x + (i - 0.5) * 0.38
    ax.bar(xx, [a[0] for a in v], 0.36, yerr=[a[1] for a in v], capsize=3, color=col, label=f"ViT {st}")
    for j, a in enumerate(v):
        ax.scatter([xx[j]] * 3, a[2], s=10, color="#0b0b0b", zorder=3, alpha=0.6)
        ax.text(xx[j], max(a[2]) + 0.35, f"{a[0]:+.1f}", ha="center", fontsize=8, color="#52514e")
ax.axhline(0, color="#52514e", lw=0.8)
ax.set_ylim(-1.5, 10)
ax.set_xticks(x, [flab[f] for f in fixes], fontsize=8)
ax.set_ylabel("accuracy recovered vs no fix (points)")
ax.set_title("(a) Gain from each fix (bars: mean ± sd; dots: each trained ViT)", loc="left", fontsize=10.5)
ax.legend(loc="upper left", fontsize=8)
ax = axes[1]
lab = ["FP32", "no fix", "PTF", "PTF + linear\n+ GELU"]
vals = [S["vit/fp"], S["vit/E6/W4A4/none"]["acc"], S["vit/E6/W4A4/ptf"]["acc"], S["vit/E6/W4A4/ptf+linPTF+geluPTF"]["acc"]]
cols = [C["muted"], C["vit"], C["vit"], C["vit"]]
ax.bar(range(4), [v[0] for v in vals], 0.6, yerr=[v[1] for v in vals], capsize=3, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v[0] + v[1] + 0.6, f"{v[0]:.1f}", ha="center", fontsize=9, color="#52514e")
ax.set_xticks(range(4), lab, fontsize=8)
ax.set_ylim(60, 92)
ax.set_ylabel("ViT test accuracy at W4A4 (%)")
ax.set_title("(b) ViT accuracy at W4A4", loc="left", fontsize=10.5)
fig.tight_layout()
fig.savefig("fig4_fixes.png", dpi=180); plt.close(fig)

# ---------------- Figure 5: per-location sensitivity, ViT, mean over 3 models
import re
d = S["vit/E4L"]
names = list(d.keys())
def cat(n):
    if "softmax" in n: return "softmax"
    if "gelu" in n: return "gelu"
    if "ln" in n: return "ln_in"
    if re.search(r"act_[qkv]$", n): return "matmul"
    return "linear_in"
cc = {"ln_in": C["vit"], "linear_in": C["resnet20"], "matmul": C["violet"], "softmax": C["aqua"], "gelu": C["gold"]}
cl = {"ln_in": "LayerNorm inputs", "linear_in": "linear inputs", "matmul": "Q, K, V", "softmax": "softmax", "gelu": "GELU outputs"}
fig, ax = plt.subplots(figsize=(12, 3.8))
mu = [d[n][0] for n in names]; sd = [d[n][1] for n in names]
ax.bar(range(len(names)), mu, 0.8, yerr=sd, capsize=1.5, error_kw=dict(lw=0.7), color=[cc[cat(n)] for n in names])
bidx = [i for i, n in enumerate(names) if n.endswith("act_ln1")]
for j, i in enumerate(bidx):
    ax.axvline(i - 0.5, color="#d5d4cf", lw=1, zorder=0)
    ax.text(i, -1.05, f" block {j}", fontsize=8, color="#52514e")
ax.set_ylim(-1.2, 4.2)
ax.set_xticks([])
ax.set_ylabel("accuracy drop (points)")
ax.set_title("Per-location sensitivity: each ViT location alone at 4 bits (mean ± sd over 3 trained ViTs, first 2,000 test images)", loc="left", fontsize=10.5)
ax.legend(handles=[Patch(color=cc[c], label=cl[c]) for c in cl], ncol=5, fontsize=8, loc="upper right")
top = sorted(zip(mu, names), reverse=True)[:3]
for v, n in []:
    i = names.index(n)
    ax.annotate(n.replace("blocks.", "blk ").replace("attn.", ""), (i, v + d[n][1]), textcoords="offset points", xytext=(4, 4), fontsize=7.5, color="#52514e")
fig.tight_layout()
fig.savefig("fig5_per_layer.png", dpi=180); plt.close(fig)
print("ok", top)
