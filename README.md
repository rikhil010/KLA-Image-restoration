# AI-Based Restoration of Degraded Semiconductor Inspection Images

A deep learning pipeline that jointly denoises and super-resolves degraded semiconductor microscopy images from 128×128 → 256×256.

## Problem

Inspection images are degraded by:
1. **Speckle Noise** — random pixel-level noise (values pushed beyond [0,1])
2. **Gaussian Noise** — softening of edges and fine structures
3. **Spatial Resolution Reduction** — 2× downsampling (256→128)

A single model must handle **all three degradations simultaneously**.

## Solution

**Compact post-upsampling residual network (EDSR-lite)** — processes the 128×128 degraded image through residual blocks, then upsamples to 256×256 via pixel-shuffle.

**Combined loss function:**
- Charbonnier L1 (robust to speckle outliers)
- Multi-Scale SSIM (structure preservation)
- FFT magnitude loss (high-frequency texture reconstruction)
- Gradient/Sobel edge loss (sharp edges without ringing)

**Data augmentation** for out-of-distribution generalization:
- Random flips, 90° rotations
- Extra Gaussian noise injection (varying noise levels)
- Intensity scaling (contrast variation)

## Setup

```bash
pip install torch numpy Pillow
```

## Training

```bash
python train.py --epochs 200 --batch_size 16 --lr 1e-3
```

All parameters are configurable via CLI flags. See `python train.py --help`.

Options:
- `--model_name unet` to use the U-Net alternative
- `--n_feats 48 --n_blocks 16` for a larger model (faster GPU training)
- `--device auto` auto-detects GPU/CPU

Training saves:
- `weights/best.pth` — best model by validation SSIM
- `weights/epoch_NNN.pth` — periodic checkpoints

## Evaluation / Inference

```bash
python evaluate.py --input_dir <test_images> --output_dir <restored_outputs>
```

With ground truth for metrics:
```bash
python evaluate.py --input_dir <test_images> --output_dir <outputs> --gt_dir <ground_truth>
```

The script:
- Loads all `.npy` or `.png` images from `--input_dir`
- Runs the trained model
- Saves restored images as `.png` and `.npy` in `--output_dir`
- Prints per-image and average PSNR/SSIM if `--gt_dir` is provided

## Project Structure

```
├── train.py              # Training entrypoint
├── evaluate.py           # Evaluation entrypoint
├── requirements.txt
├── src/
│   ├── config.py         # All hyperparameters
│   ├── dataset.py        # Dataset, augmentation, train/val split
│   ├── model.py          # EDSR and U-Net architectures
│   ├── losses.py         # Charbonnier, MS-SSIM, FFT, gradient losses
│   ├── metrics.py        # PSNR and SSIM computation
│   ├── train.py          # Training loop (EMA, early stopping, etc.)
│   └── evaluate.py       # Inference logic
├── weights/              # Saved model weights
├── outputs/              # Restored image outputs
└── Dataset/              # Training data (GT/, NoisyLR/)
```

## Model Architecture

```
Input: 1×128×128 (NoisyLR)
  → Head conv (1 → 32 channels)
  → 12× Residual Blocks (3×3 Conv → ReLU → 3×3 Conv, scale=0.1)
  → Global skip connection from head
  → Pixel-Shuffle 2× upsample (128×128 → 256×256)
  → Tail conv (32 → 1 channel)
Output: 1×256×256 (restored)
```

## Hardware

- **Training:** works on CPU (faster on GPU with CUDA)
- **Benchmark evaluation:** runs on H100 GPU
- Model size: ~650K parameters (EDSR, n_feats=32, n_blocks=12)
- Inference time: <1 second on GPU per 128×128 image
