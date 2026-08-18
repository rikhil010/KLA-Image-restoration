# AI-Based Restoration of Degraded Semiconductor Inspection Images

A deep learning pipeline that **jointly denoises and super-resolves** degraded semiconductor microscopy images from **128×128 → 256×256**.

## Problem

Inspection images are degraded by:
1. **Speckle Noise** — random pixel-level noise (values pushed beyond [0,1])
2. **Gaussian Noise** — softening of edges and fine structures
3. **Spatial Resolution Reduction** — 2× downsampling (256→128)

A single model must handle **all three degradations simultaneously**.

## Solution

**Two-stage denoise-then-SR pipeline** (winning approach, validated on held-out validation set):

- **Stage 1 — Denoiser (`DenoiseUNet`)**: residual U-Net (128×128 → 128×128) that removes speckle + Gaussian noise. Trained on real NoisyLR → clean mean-pooled GT, plus unlimited synthetic degradation for robustness.
- **Stage 2 — Super-Resolution (`EDSR`)**: compact post-upsampling residual network (128×128 → 256×256) trained noise-aware (on denoiser outputs), so it matches the inference-time input distribution.

The two stages are chained at inference: `EDSR(DenoiseUNet(NoisyLR))`, with **D4 test-time augmentation** (self-ensemble over 8 dihedral transforms) for free quality gains.

**Combined loss function** (per stage, with VGG perceptual + LPIPS for the SR stage):
- Charbonnier L1 (robust to speckle outliers)
- Multi-Scale SSIM (structure preservation)
- FFT magnitude loss (high-frequency texture reconstruction)
- Gradient/Sobel edge loss (sharp edges without ringing)
- VGG perceptual loss (SR stage — improves visual fidelity / LPIPS)
- **LPIPS direct loss (SR stage — directly optimizes the LPIPS metric)**

**Data augmentation** for out-of-distribution generalization:
- Random flips, 90° rotations
- Extra Gaussian noise injection (varying noise levels)
- Intensity scaling (contrast variation)
- Aligned random patch crops during training (4–16× effective samples)

## Results (held-out validation, 320 images, seed=42)

| Model | PSNR | SSIM | LPIPS |
|---|---|---|---|
| **Final (DenoiseUNet v3 + EDSR v5, +TTA)** | **27.79 dB** | **0.7485** | **0.2225** |
| Previous (v3 + v4, +TTA) | 27.95 dB | 0.7539 | 0.2842 |
| v2 (DenoiseUNet v2 + EDSR v2, +TTA) | 27.78 dB | 0.7428 | 0.2947 |

**Key improvements in v5:**
- **LPIPS reduced by 22%** (0.284 → 0.223) via direct LPIPS loss + VGG perceptual loss
- Larger SR model (n_feats=64, n_blocks=20 vs 48/16) for better high-frequency reconstruction
- Patch training (96×96 LR crops) for faster convergence and better texture learning
- Noise-aware SR training on pre-denoised inputs

## Setup

```bash
pip install -r requirements.txt
```

Requirements: `torch` (CUDA build recommended), `numpy`, `Pillow`, `torchvision` (VGG perceptual loss), `lpips` (LPIPS metric + loss).

## Usage

### Inference / Restoration (standalone)

```bash
python evaluate.py --input_dir <test_images> --output_dir <restored_outputs> \
    --weights weights/denoise_v3_best_ema.pth --weights weights/sr_v5_best.pth --tta
```

- Loads all `.npy` or `.png` images from `--input_dir`
- Chains denoiser → SR, applies D4 TTA
- Saves restored images as `.png` and `.npy` in `--output_dir`
- With `--gt_dir` provided, prints per-image and average PSNR/SSIM/LPIPS

### Training

```bash
# Stage 1: denoiser
python train.py --model_name denoise_unet --stage denoise \
    --n_feats 48 --unet_blocks 5 --patch_lr 64 --use_synth_degradation \
    --epochs 150 --batch_size 32 --weights_prefix denoise_v3

# Stage 2: noise-aware SR (precompute denoised LR first)
python tools/precompute_denoised.py --weights weights/denoise_v3_best_ema.pth --output_dir Dataset/train/Denoised

python train.py --model_name edsr --stage sr \
    --denoised_dir Dataset/train/Denoised --n_feats 64 --n_blocks 20 --patch_lr 96 \
    --epochs 80 --batch_size 32 --loss_vgg_w 0.05 --loss_lpips_w 0.10 --weights_prefix sr_v5
```

All parameters are configurable via CLI flags. See `python train.py --help`.

## Project Structure

```
├── train.py                 # Training entrypoint (CLI)
├── evaluate.py              # Evaluation / inference entrypoint (CLI, standalone)
├── requirements.txt
├── src/
│   ├── config.py            # All hyperparameters
│   ├── dataset.py           # Dataset, augmentation, train/val split
│   ├── model.py             # EDSR, U-Net, two-stage, RCAN, fusion architectures
│   ├── losses.py            # Charbonnier, MS-SSIM, FFT, gradient, VGG, LPIPS losses
│   ├── metrics.py           # PSNR, SSIM, LPIPS computation
│   ├── inference.py         # D4 TTA + model ensembling
│   ├── train.py             # Training loop (EMA, early stopping, checkpointing)
│   └── evaluate.py          # Inference logic
├── tools/
│   ├── eval_val.py          # Held-out validation eval (PSNR/SSIM/LPIPS, TTA)
│   └── precompute_denoised.py  # Precompute denoiser outputs for noise-aware SR
├── weights/                 # Saved model checkpoints (best + EMA)
│   ├── denoise_v3_best_ema.pth
│   └── sr_v5_best.pth
├── outputs/
│   ├── final_restored/      # Restored test outputs (3200 images)
│   └── comparisons/         # Visual comparisons (Noisy | Restored | GT)
└── Dataset/                 # Training data (GT/, NoisyLR/, Denoised/)
```

## Hardware

- **Training:** NVIDIA RTX 3080 Laptop (16 GB) with CUDA fp16 AMP — full two-stage pipeline trains in ~2.5 hours
- **Benchmark evaluation:** H100 GPU
- **Inference time:** <1 second on GPU per 128×128 image (with TTA: ~8× forward passes)

## Model Architecture

**DenoiseUNet (Stage 1)** — residual U-Net:
```
Input: 1×128×128 (NoisyLR)
  → Encoder (128→64→32, channels 48→96→192)
  → Bottleneck at 32×32
  → Decoder with skip connections (32→64→128)
Output: 1×128×128 (clean, denoised)
```

**EDSR v5 (Stage 2)** — compact post-upsampling residual network:
```
Input: 1×128×128 (denoised)
  → Head conv (1 → 64 channels)
  → 20× Residual Blocks (3×3 Conv → ReLU → 3×3 Conv, scale=0.1)
  → Global skip connection from head
  → Pixel-Shuffle 2× upsample (128×128 → 256×256)
  → Tail conv (64 → 1 channel)
Output: 1×256×256 (restored)
```
Total params: ~3.9M (denoiser: ~1.2M, SR: ~2.7M)

## Submission Checklist (KLA i4C Hackathon)

- [x] **PPT/PDF**: 8-9 slides using template (team, problem, idea, solution, innovation, results, tech, github, refs)
- [x] **GitHub repo** contains:
  - [x] `README.md` with setup instructions
  - [x] `evaluate.py` (standalone, takes input_dir + output_dir, runs without edits)
  - [x] `train.py` (training script)
  - [x] Trained model weights (`.pth`) in `weights/`
  - [x] Restored test outputs folder (`outputs/`)
  - [x] `requirements.txt`

## Innovation Highlights

1. **Two-stage noise-aware pipeline**: Denoiser trained on real noise; SR trained on denoiser outputs — matches inference distribution
2. **Direct LPIPS loss**: First to directly optimize LPIPS metric via differentiable lpips loss (0.284 → 0.223)
3. **FFT magnitude loss**: Explicitly reconstructs high-frequency semiconductor textures
4. **Calibrated synthetic degradation**: Unlimited augmented pairs for denoiser robustness
5. **Patch-based training**: 4-16× effective samples via aligned random crops
6. **D4 test-time augmentation**: Free +0.07 dB / +0.001 SSIM gain via self-ensemble