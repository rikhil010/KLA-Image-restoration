# epitax - KLA i4C Hackathon Submission

**Team:** epitax  
**Task:** AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Overview

This solution restores degraded 128×128 semiconductor microscopy images to 256×256 resolution by jointly removing speckle + Gaussian noise and performing 2× super-resolution.

## Architecture

**Two-stage pipeline:**
1. **DenoiseUNet** (residual U-Net): 128×128 → 128×128 — removes speckle and Gaussian noise
2. **EDSR v5** (post-upsampling residual network): 128×128 → 256×256 — super-resolves with noise-aware training

**Inference:** `EDSR(DenoiseUNet(NoisyLR))` with **D4 test-time augmentation** (8 dihedral self-ensemble)

**Loss functions:** Charbonnier L1 + MS-SSIM + FFT magnitude + Gradient/Sobel + VGG perceptual + **Direct LPIPS loss**

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.8+ (tested on RTX 3080 / H100)
- 16 GB VRAM minimum

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- `torch==2.11.0+cu128`
- `numpy==2.5.1`
- `Pillow==12.3.0`
- `torchvision==0.19.0+cu128`
- `lpips==0.1.4`
- `tqdm==4.66.4`

## Usage

```bash
python run.py <input-dir> <output-dir>
```

- **input-dir**: Directory containing degraded `.npy` images (128×128, float32)
- **output-dir**: Output directory (created if not exists), will contain restored `.npy` images (256×256, float32, range [0,1])

### Example

```bash
python run.py ./test_degraded ./restored_outputs
```

## Model Weights

Pre-trained weights included in `models/`:
- `denoise_v3_best_ema.pth` — Denoiser (EMA weights)
- `sr_v5_best.pth` — Super-resolution model

No internet access or additional downloads required at runtime.

## Output Format

Each output `.npy` file:
- Same filename as input
- Shape: (256, 256) or (256, 256, 1)
- dtype: float32
- Range: [0, 1] (clamped)
- No NaN or Inf values

## Results (Validation Set, 320 images)

| Model | PSNR | SSIM | LPIPS |
|-------|------|------|-------|
| epitax (Denoise v3 + EDSR v5 + TTA) | **27.79 dB** | **0.7485** | **0.2225** |

## Reproducibility

To retrain from scratch:
```bash
# Stage 1: Denoiser
python ../train.py --model_name denoise_unet --stage denoise \
    --n_feats 48 --unet_blocks 5 --patch_lr 64 --use_synth_degradation \
    --epochs 150 --batch_size 32 --weights_prefix denoise_v3

# Precompute denoised inputs
python ../tools/precompute_denoised.py --weights weights/denoise_v3_best_ema.pth --output_dir Dataset/train/Denoised

# Stage 2: Noise-aware SR
python ../train.py --model_name edsr --stage sr \
    --denoised_dir Dataset/train/Denoised --n_feats 64 --n_blocks 20 --patch_lr 96 \
    --epochs 80 --batch_size 32 --loss_vgg_w 0.05 --loss_lpips_w 0.10 --weights_prefix sr_v5
```

## License

KLA i4C Hackathon 2026 — Team epitax