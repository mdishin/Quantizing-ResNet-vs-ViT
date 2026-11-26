import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("""# Quantizing ResNet vs ViT (EN.520.638 final project)
Runs the whole pipeline on a Colab GPU: **E1** train both models → **E2/E3** post-training quantization sweeps →
**E4** find the weak spot → **E5** look at the values → **E6** apply the fixes.

Runtime → Change runtime type → **T4 GPU** (or better). Upload `quant_resnet_vs_vit.zip` to the session first
(Files panel), or clone your team repo in the next cell."""),
    code("""USE_DRIVE = True   # keep checkpoints + results on Google Drive so a disconnect loses nothing
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK = '/content/drive/MyDrive/quant_resnet_vs_vit'
else:
    WORK = '/content/qp'
!mkdir -p "{WORK}"
!unzip -nq /content/quant_resnet_vs_vit.zip -d "{WORK}"   # -n: never overwrite files from an earlier run
%cd "{WORK}"
!pip -q install matplotlib
import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU: switch the runtime to T4')"""),
    md("""## E1: full-precision baselines
CIFAR-10 downloads automatically through torchvision (`data/`). Split: 45k train / 5k val / 10k test; calibration =
512 training images.

* **ResNet-20**: either train it (≈91–92.5%, ~25 min on a T4) or load the public checkpoint
  (chenyaofo/pytorch-cifar-models, 92.60%). Note the public checkpoint was trained on all 50k training images, so
  its val score is not held-out; report test accuracy only.
* **ViT**: 200 epochs, AdamW, cosine, label smoothing 0.1, RandAugment. Our CPU run (45 epochs, no RandAugment)
  reached the accuracy in the report; this recipe should land in the plan's 80–85% range.

Use 3 training seeds if time allows (`SEEDS = [0, 1, 2]`)."""),
    code("""SEEDS = [0]          # [0, 1, 2] for the final report
USE_PRETRAINED_RESNET = True"""),
    code("""import os, urllib.request, torch
if USE_PRETRAINED_RESNET:
    urllib.request.urlretrieve('https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt', 'ckpt_resnet20_s0.pt')
else:
    for s in SEEDS:
        !python train.py --model resnet20 --epochs 160 --opt sgd --lr 0.1 --wd 1e-4 --warmup 1 --smoothing 0 --batch 128 --seed {s} --out ckpt_resnet20_s{s}.pt"""),
    code("""for s in SEEDS:
    !python train.py --model vit --epochs 200 --randaug --batch 256 --lr 1e-3 --wd 0.05 --warmup 5 --seed {s} --out ckpt_vit_s{s}.pt"""),
    md("## E2–E4, E6: quantization sweeps (one results file per trained model)"),
    code("""rn_seeds = [0] if USE_PRETRAINED_RESNET else SEEDS
for s in rn_seeds:
    !python run_experiments.py --model resnet20 --ckpt ckpt_resnet20_s{s}.pt --out results/resnet20_s{s}.json
for s in SEEDS:
    !python run_experiments.py --model vit --ckpt ckpt_vit_s{s}.pt --out results/vit_s{s}.json"""),
    md("## E5 + figures (uses training seed 0)"),
    code("""import shutil
shutil.copy('results/resnet20_s0.json', 'results/resnet20.json'); shutil.copy('results/vit_s0.json', 'results/vit.json')
shutil.copy('ckpt_resnet20_s0.pt', 'ckpt_resnet20.pt'); shutil.copy('ckpt_vit_s0.pt', 'ckpt_vit.pt')
!python analyze.py > /dev/null
from IPython.display import Image, display
for f in ['fig1_drop_vs_bits', 'fig2_where_it_breaks', 'fig3_values', 'fig4_fixes', 'fig5_per_layer']:
    display(Image(f'figures/{f}.png'))"""),
    md("## Main table across training seeds (mean ± sd of the accuracy drop)"),
    code("""import json, glob, numpy as np
for m in ['resnet20', 'vit']:
    files = sorted(glob.glob(f'results/{m}_s*.json'))
    R = [json.load(open(f)) for f in files]
    print(m, f'({len(R)} trained models)')
    for key in ['E2/max/W8', 'E2/max/W4', 'E2/max/W2', 'E3/actonly/W32A4/s0', 'E3/improved/W8A8/s0', 'E3/improved/W6A6/s0',
                'E3/improved/W4A4/s0', 'E6/W4A4/ptf/s0', 'E6/W4A4/ptf+linPTF+geluPTF/s0']:
        d = [r['E1/fp']['acc'] - r[key]['acc'] for r in R if key in r]
        if d: print(f'  {key:28s} drop {np.mean(d):6.2f} ± {np.std(d):.2f}')"""),
]
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nbf.write(nb, "Quantizing_ResNet_vs_ViT.ipynb")
print("ok")
