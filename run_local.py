"""Run the whole project on your own machine (Linux, Windows or macOS), no notebook needed.

  python run_local.py --smoke                 # ~5 min end-to-end check with tiny settings (separate outputs)
  python run_local.py                         # full run, training seed 0
  python run_local.py --seeds 0 1 2           # final report: 3 trained ViTs
  python run_local.py --seeds 0 1 2 --train-resnet   # also train ResNet-20 on the 45k split

Safe to re-run at any point: finished training is reused, interrupted training resumes,
finished experiment results are skipped.
"""
import argparse, glob, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=[0], help="training seeds")
p.add_argument("--train-resnet", action="store_true", help="train ResNet-20 instead of using the bundled pretrained one")
p.add_argument("--vit-epochs", type=int, default=200)
p.add_argument("--batch", type=int, default=256, help="ViT batch size (use 128 on a small GPU)")
p.add_argument("--lr", type=float, default=1e-3, help="ViT learning rate (use 5e-4 with --batch 128)")
p.add_argument("--no-randaug", action="store_true")
p.add_argument("--smoke", action="store_true", help="tiny end-to-end test; writes to smoke/ only")
p.add_argument("--skip-train", action="store_true")
p.add_argument("--skip-experiments", action="store_true")
args = p.parse_args()


def sh(*cmd):
    print("\n>>", " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable, *cmd])
    if r.returncode != 0:
        sys.exit(f"command failed (exit {r.returncode}); fix the error above and re-run the same command to continue")


import torch  # noqa: E402

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
else:
    print("WARNING: no CUDA GPU visible to PyTorch; everything will run on the CPU (very slow for ViT training).")

out = "smoke" if args.smoke else "."
os.makedirs(os.path.join(out, "results"), exist_ok=True)
ck = lambda name, s: os.path.join(out, f"ckpt_{name}_s{s}.pt")
res = lambda name, s: os.path.join(out, "results", f"{name}_s{s}.json")
seeds = args.seeds[:1] if args.smoke else args.seeds

# ---------------------------------------------------------------- E1: training
if not args.skip_train:
    rn_seeds = seeds if args.train_resnet else [0]
    for s in rn_seeds:
        if os.path.exists(ck("resnet20", s)):
            print(f"ResNet-20 seed {s}: checkpoint exists, skipping")
        elif args.train_resnet:
            sh("train.py", "--model", "resnet20", "--epochs", "1" if args.smoke else "160", "--opt", "sgd",
               "--lr", "0.1", "--wd", "1e-4", "--warmup", "1", "--smoothing", "0", "--batch", "128",
               "--seed", str(s), "--out", ck("resnet20", s))
        else:
            shutil.copy("ckpt_resnet20.pt", ck("resnet20", s))  # bundled public checkpoint (92.60%)
            print("ResNet-20: using the bundled pretrained checkpoint")
    for s in seeds:
        if os.path.exists(ck("vit", s)):
            print(f"ViT seed {s}: checkpoint exists, skipping")
            continue
        cmd = ["train.py", "--model", "vit", "--epochs", "1" if args.smoke else str(args.vit_epochs),
               "--batch", str(args.batch), "--lr", str(args.lr), "--wd", "0.05", "--warmup", "5",
               "--seed", str(s), "--out", ck("vit", s)]
        if not args.no_randaug:
            cmd.append("--randaug")
        sh(*cmd)

# ---------------------------------------------------------------- E2-E4, E6
if not args.skip_experiments:
    extra = ["--quick", "--only", "E1,E2,E3", "--seeds", "1"] if args.smoke else []
    rn_seeds = seeds if args.train_resnet else [0]
    for s in rn_seeds:
        sh("run_experiments.py", "--model", "resnet20", "--ckpt", ck("resnet20", s), "--out", res("resnet20", s), *extra)
    for s in seeds:
        sh("run_experiments.py", "--model", "vit", "--ckpt", ck("vit", s), "--out", res("vit", s), *extra)

if args.smoke:
    print("\nSMOKE TEST PASSED. Outputs are in smoke/ (delete it any time). Now run: python run_local.py")
    sys.exit(0)

# ---------------------------------------------------------------- E5 + figures (training seed 0)
s0 = seeds[0]
if not os.path.exists("results/preliminary_cpu"):  # keep the bundled CPU-run results for reference
    os.makedirs("results/preliminary_cpu")
    for f in ["resnet20.json", "vit.json", "summary.json"]:
        if os.path.exists(f"results/{f}"):
            shutil.copy(f"results/{f}", "results/preliminary_cpu/")
shutil.copy(res("resnet20", 0), "results/resnet20.json")
shutil.copy(res("vit", s0), "results/vit.json")
shutil.copy(ck("vit", s0), "ckpt_vit.pt")
if args.train_resnet:
    shutil.copy(ck("resnet20", 0), "ckpt_resnet20.pt")
sh("analyze.py")
print("Figures written to figures/, all numbers to results/summary.json")

# ---------------------------------------------------------------- seed table
import numpy as np  # noqa: E402

keys = ["E2/max/W8", "E2/max/W4", "E2/max/W2", "E3/actonly/W32A4/s0", "E3/improved/W8A8/s0",
        "E3/improved/W6A6/s0", "E3/improved/W4A4/s0", "E6/W4A4/ptf/s0", "E6/W4A4/ptf+linPTF+geluPTF/s0"]
for m in ["resnet20", "vit"]:
    R = [json.load(open(f)) for f in sorted(glob.glob(f"results/{m}_s*.json"))]
    print(f"\n{m}: accuracy drop from own FP32, mean ± sd over {len(R)} trained model(s)")
    print(f"  {'FP32 accuracy':34s} {np.mean([r['E1/fp']['acc'] for r in R]):6.2f}")
    for k in keys:
        d = [r["E1/fp"]["acc"] - r[k]["acc"] for r in R if k in r]
        if d:
            print(f"  {k:34s} {np.mean(d):6.2f} ± {np.std(d):.2f}")
print("\nDONE")
