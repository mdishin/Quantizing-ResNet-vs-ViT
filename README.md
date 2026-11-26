<<<<<<< HEAD
# Quantizing-ResNet-vs-ViT
When we shrink a trained CNN and a trained Vision Transformer from 32-bit numbers to 8, 4 and 2bits, which one loses more accuracy, where exactly does it break, and can a known fix recover it?
=======
# Quantizing ResNet vs ViT: code

EN.520.638 Deep Learning final project. Post-training quantization (PTQ) of ResNet-20 and a 6-block ViT on CIFAR-10.

```
qproj/data.py        CIFAR-10 loading, 45k/5k/10k split, 512-image calibration set, evaluation
qproj/models.py      ResNet-20 and ViT with labelled activation tap points (Act), BatchNorm folding
qproj/quant.py       the shared rounding functions, MSE range search, weight quantization, log2 + PTF fixes
train.py             E1: full-precision training (GPU: fp16 AMP; CPU: bf16 autocast)
run_experiments.py   E1-E4, E4L (per-location), E6; resumable, writes results/<model>.json
analyze.py           E5 value statistics + all figures + results/summary.json
Quantizing_ResNet_vs_ViT.ipynb   Colab driver for the whole pipeline
```

## Quick start (your own machine)
```
python run_local.py --smoke      # ~5-10 min end-to-end check, writes to smoke/ only
python run_local.py              # full run, training seed 0 (add --seeds 0 1 2 for the report)
```
Needs an NVIDIA GPU with 4 GB+ and PyTorch with CUDA. Re-run the same command to resume after any interruption.

## Quick start (Colab GPU)
Open the notebook, upload this folder as a zip, run all. Or by hand:
```
python train.py --model vit --epochs 200 --randaug --batch 256 --seed 0 --out ckpt_vit.pt
python run_experiments.py --model vit --ckpt ckpt_vit.pt
python run_experiments.py --model resnet20 --ckpt ckpt_resnet20.pt
python analyze.py
```

## What each location means
| category | ResNet-20 | ViT |
|---|---|---|
| first / last (always 8-bit) | image into conv1; pooled features into fc | image into patch embedding; CLS feature into head |
| conv_in | input of every other conv (post-ReLU) | |
| ln_in | | LayerNorm inputs = the residual stream (13 places) |
| linear_in | | inputs to qkv, attention proj, fc1 (= LayerNorm outputs / attention output) |
| softmax | | attention probabilities |
| gelu | | GELU outputs (= fc2 inputs) |
| matmul | | Q, K, V entering the two attention matmuls (added; needed for a fully integer attention) |

## Quantizer choices
* Weights: symmetric, per output channel, clip = channel max (`--wmethod max`) or per-channel MSE search (`mse`).
  First and last layers are kept at 8 bits. BatchNorm is folded into the conv before quantizing.
* Activations: one scale per tensor, clip chosen by MSE search over 115 candidate ratios on the 512 calibration images.
  `scheme="plan"` uses the plan's single signed symmetric grid everywhere. `scheme="improved"` (default) uses an
  unsigned grid for never-negative tensors (post-ReLU, softmax) and a zero-point grid for GELU outputs.
* Fixes: `softmax_fix="log2"` (FQ-ViT log2 quantizer), `ln_fix="ptf"` (FQ-ViT Power-of-Two Factor, K=3).
>>>>>>> e75f72c (Initial commit)
