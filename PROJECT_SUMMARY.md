# AI-Based Restoration of Degraded Semiconductor Inspection Images — Project Summary

## ��� Overview

This project implements a deep learning pipeline for **joint denoising and 2× super-resolution** of degraded semiconductor microscopy images. Developed for the **KLA i4C Hackathon**, it transforms 128×128 noisy, low-resolution (NoisyLR) images into 256×256 clean, high-resolution outputs matching ground truth (GT).

**Problem**: Semiconductor inspection images suffer from three simultaneous degradations:
- **Speckle noise** — multiplicative, pushes pixel values beyond [0,1] (observed range: ~[-0.28, 2.16])
- **Gaussian noise** — additive, softens edges and fine structures
- **Spatial resolution reduction** — 2× downsampling (256→128) via mean pooling

**Key Achievement**: Final best model (chained Denoise_v2 EMA + SR_v2 with D4 TTA) on full validation set (320 images):
- **PSNR**: 27.77 dB
- **SSIM**: 0.7428  
- **LPIPS**: 0.2947
- **+0.33 dB PSNR, +0.009 SSIM, -0.061 LPIPS** over previous best ensemble

---

## ������ Architecture

### 1. EDSR (Primary Backbone) — `model_name="edsr"`

**Compact post-upsampling residual network** (Lim et al., NTIRE 2017 winner):

```
Input: [B, 1, 128, 128] (NoisyLR)
  → Head Conv: 1 → n_feats (3×3, padding=1)
  → Body: N × ResidualBlock (Conv3×3 → ReLU → Conv3×3, res_scale=0.1)
  → Global Skip: Body Conv3×3 + Head output
  → PixelShuffle 2×: n_feats channels @ 256×256 (ReLU)
  → Tail Conv: n_feats → 1 (3×3)
Output: [B, 1, 256, 256] (restored)
```

**Why EDSR?**
- Proven SR backbone; won NTIRE 2017
- Post-upsampling: all heavy compute at 128×128 (efficient)
- No Batch Normalization: BN removes useful range info, hurts restoration
- Residual scaling (0.1) stabilizes deep training

**Configurations:**
| Config | n_feats | n_blocks | Params | Use Case |
|--------|---------|----------|--------|----------|
| CPU demo | 24 | 6 | ~89K | Quick CPU runs |
| GPU full | 32 | 12 | ~269K | Standard training |
| v2 large | 48 | 16 | ~9.3M | Best quality (patch training) |

### 2. Residual U-Net — `model_name="unet"`

Alternative architecture with encoder-decoder + skip connections:

```
Encoder: 128→64→32 (MaxPool2d + ResBlocks)
Bottleneck: 32×32 (ResBlocks)
Decoder: 32→64→128 (ConvTranspose2d + concat skip + ResBlocks)
Optional 2× PixelShuffle: 128→256
```

### 3. Two-Stage Pipeline (Denoise → SR) — `model_name="two_stage"`

**Key innovation**: Split the joint task into two specialized stages, then optionally fine-tune end-to-end.

| Stage | Model | Task | Input → Target |
|-------|-------|------|----------------|
| **Denoise** | `DenoiseUNet` (U-Net w/o SR tail) | 128→128 denoising | NoisyLR → mean_pool(GT) |
| **SR** | `EDSR` | 128→256 super-res | mean_pool(GT) → GT |
| **Joint** | `TwoStageModel` | chained + fine-tune | NoisyLR → GT |

**Stage-specific training details:**
- **Denoise stage**: Trained with noise/intensity augmentation; target = clean pooled GT
- **SR stage**: **Noise-aware** — trained on pre-denoised LR (from Denoiser) → GT, matching inference distribution. Uses VGG perceptual loss.
- **Joint stage**: Loads pretrained stages, fine-tunes end-to-end with synthetic degradation

**Checkpoint prefixes** prevent clobbering: `denoise_*`, `sr_*`, `joint_*`, or custom `--weights_prefix`

---

## ��� Loss Functions (src/losses.py)

**CombinedLoss** — weighted sum of five complementary losses (all computed in float32 for AMP stability):

| Loss | Weight | Purpose |
|------|--------|---------|
| **Charbonnier L1** | 1.0 | Smooth L1: `��(x² + ε²)`. Robust to speckle outliers beyond [0,1] |
| **MS-SSIM** | 0.5 | Multi-scale structural similarity (windows 11, 7, 3). Directly optimizes eval metric |
| **FFT Magnitude** | 0.1 | L1 on `|FFT(pred)|` vs `|FFT(target)|`. Forces high-frequency texture reconstruction |
| **Gradient (Sobel)** | 0.1 | L1 on edge magnitudes. Preserves sharp edges without ringing |
| **VGG Perceptual** | 0.05 (v2) | L1 in VGG16 feature space (relu1_2, 2_2, 3_3, 4_3). Improves LPIPS + visual sharpness |

**Implementation notes for fp16 AMP:**
- CombinedLoss casts inputs to float32 internally
- GradientLoss Sobel kernels cast to input device+dtype
- FFTMagnitudeLoss uses `torch.fft.fft2(..., norm='ortho').float()`

---

## ��� Metrics (src/metrics.py)

| Metric | Function | Notes |
|--------|----------|-------|
| **PSNR** | `psnr(pred, target)` | dB, higher = better |
| **SSIM** | `ssim(pred, target)` | [0,1], higher = better. Reuses loss SSIM window |
| **LPIPS** | `lpips_score(pred, target)` | Lower = better. 'alex' net, grayscale→3ch, ImageNet norm |

---

## ��� Dataset & Augmentation (src/dataset.py, src/degradation.py)

### Data Format
- **GT**: 3200 × 256×256 float32 .npy files, range [0,1] exactly
- **NoisyLR**: 3200 × 128×128 float32 .npy files, range ~[-0.28, 2.16]
- **Scale**: Exactly 2× (mean pooling operator)
- **Correlation**: NoisyLR vs mean_pool(GT) ≈ 0.98

### Train/Val Split
- 90% / 10% (320 val images) with fixed seed=42 — reproducible

### Data Augmentation (enabled during training)
- Random horizontal/vertical flips
- Random 90° rotations (k ∈ {0,1,2,3})
- Extra Gaussian noise injection: σ ~ Uniform(0, 0.03)
- Random intensity scaling: × Uniform(0.9, 1.1)
- **Patch/crop training**: `--patch_lr 64` → aligned random 64×64 LR crops → 128×128 GT crops (4-16× effective samples)

### Calibrated Synthetic Degradation (Phase 1 / `--use_synth_degradation`)
Empirically fitted from real data (`src/n_s_real.npy`):
```
clean = mean_pool(GT) * SCALE + OFFSET       # SCALE=1.02, OFFSET=-0.01
speckle = clean * n_s (sampled from real n_s distribution, std��0.17, heavy tails)
noisy = speckle + Gaussian(0, 0.026)
```
- Generates **unlimited perfectly-matched pairs** from 3200 GTs
- Noise-level jitter: `noise_level ~ Uniform(0.8, 1.3)` for OOD robustness
- Validation always uses **real NoisyLR** for honest metrics

### Noise-Aware SR Training (v2)
- Precompute denoised LR: `tools/precompute_denoised.py` → `Dataset/train/Denoised/`
- SR stage trains on (denoised_LR → GT) instead of clean pooled GT
- Fixes the distribution mismatch: at inference, SR receives denoiser output with residual noise

---

## ��� Training Strategy (src/train.py)

| Component | Setting |
|-----------|---------|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| LR Schedule | Cosine annealing to 1e-6 |
| Batch Size | 16 (CPU) / 32 (GPU) |
| Mixed Precision | fp16 AMP (autocast + GradScaler) on GPU |
| Gradient Clipping | max_norm=1.0 |
| EMA | decay=0.999 (saved as `*_ema.pth` for inference) |
| Early Stopping | patience=25 on val SSIM |
| Checkpointing | `best.pth`, `best_ema.pth`, `epoch_NNN.pth` every 10 epochs |
| Seed | 42 (PyTorch + NumPy + CUDA) |

**EMA Fix (critical)**: Original code applied EMA weights too early (epoch 3: ~83% random). Fixed to:
- Validate with **current** model weights
- Only save EMA weights for inference (better generalization)

---

## ��� Inference & Test-Time Augmentation (src/inference.py, evaluate.py)

### D4 Self-Ensemble (TTA)
Applies all 8 dihedral transforms (4 rotations × 2 flips), averages predictions:
```python
# +0.2–0.5 dB PSNR for free
pred = predict_with_tta(model, x)  # 8 forward passes
```

### Multi-Model Ensembling
Average predictions from independently trained models (e.g., two-stage + single EDSR):
```python
pred = predict_ensemble([model1, model2], x, tta=True)
```

### CLI (evaluate.py)
```bash
# Single model
python evaluate.py --input_dir test/ --output_dir out/

# Ensemble + TTA + GT metrics
python evaluate.py --input_dir test/ --output_dir out/ --gt_dir gt/ \
  --weights w1.pth --weights w2.pth --tta
```

---

## ��� Training History & Results

### v1 CPU Demo (30 epochs, n_feats=24, n_blocks=6)
| Metric | Value |
|--------|-------|
| Best val SSIM | 0.7437 (epoch 27) |
| Best val PSNR | 27.48 dB (epoch 26) |
| Full val (320 imgs) | PSNR 27.38±4.56, SSIM 0.7437±0.1692 |
| Training time | 185 min (~3 hrs) on CPU |
| Weights | `best.pth`, `best_ema.pth` (~11 MB) |

### v2 GPU Pipeline (RTX 3080 Laptop, 16GB VRAM, fp16 AMP)

| Stage | Config | Epochs | Time | Key Changes |
|-------|--------|--------|------|-------------|
| Denoise_v2 | U-Net, n_feats=48, blocks=5, patch=64, synth | 150 | ~2h | Wider, patch-trained, unlimited synth data |
| Precompute | Run denoiser on all 3200 NoisyLR | — | ~5 min | Creates Denoised/ for noise-aware SR |
| SR_v2 | EDSR, n_feats=48, blocks=16, patch=64, VGG(0.05) | 150 | ~3h | Noise-aware input, perceptual loss |
| Joint_v2 | TwoStage, pretrained both, VGG(0.05) | 100 | ~2h | End-to-end fine-tune |

**Final Best Model**: **Chained denoise_v2 (EMA) + sr_v2 with D4 TTA** (no joint fine-tune)
- Joint FT degraded to SSIM 0.7312 (synth noise + VGG + low LR destabilized alignment)
- **Chained pretrained stages + TTA wins**

| Model | PSNR (dB) | SSIM | LPIPS |
|-------|-----------|------|-------|
| Previous best (opt1+opt2 ensemble+TTA) | 27.44 | 0.7337 | 0.3562 |
| **denoise_v2 EMA + sr_v2 + D4 TTA** | **27.77** | **0.7428** | **0.2947** |
| **Delta** | **+0.33** | **+0.009** | **-0.061** |

> **LPIPS drop of -0.061 is most meaningful** — indicates significantly better perceptual fidelity on semiconductor textures.

### Training Speed (measured)
- **CPU** (89K params, batch 16): ~1.49 s/batch, ~6 min/epoch
- **GPU RTX 3080** (3.9M params two-stage, batch 32, fp16): **~0.31 s/batch, ~37 s/epoch**
- Full v2 pipeline (denoise 150 + SR 150 + joint 100 ep) ≈ **2–3 hours** on RTX 3080

---

## ������ File Structure

```
Restoration_Semiconductor/
├── CLAUDE.md                    # Project memory & instructions
├── Project details.txt          # Original problem statement
├── README.md                    # Setup + usage instructions
├── ROADMAP.md                   # Development roadmap
├── requirements.txt             # torch, numpy, Pillow
├── train.py                     # Training entrypoint (CLI)
├── evaluate.py                  # Evaluation entrypoint (CLI) — standalone, frozen
├── train_v2_pipeline.sh         # Sequential v2 retrain script
├── src/
│   ├── __init__.py
│   ├── config.py                # All hyperparameters (dataclass)
│   ├── dataset.py               # Dataset + augmentation + patch crop + split
│   ├── degradation.py           # Calibrated synthetic degradation model
│   ├── model.py                 # EDSR + U-Net + TwoStage architectures
│   ├── losses.py                # Charbonnier, MS-SSIM, FFT, Gradient, VGG
│   ├── metrics.py               # PSNR + SSIM + LPIPS
│   ├── inference.py             # D4 TTA self-ensemble + multi-model ensemble
│   ├── train.py                 # Training loop (EMA, early stop, checkpoints)
│   └── evaluate.py              # Inference logic (load_model, run_inference)
├── tools/
│   ├── eval_val.py              # Val eval: PSNR/SSIM/LPIPS, TTA, ensemble
│   └── precompute_denoised.py   # Denoiser → Dataset/train/Denoised/
├── weights/                     # Model checkpoints (.pth)
│   ├── best.pth / best_ema.pth  # v1 CPU best
│   ├── denoise_v2_best.pth      # v2 denoiser
│   ├── sr_v2_best.pth           # v2 SR (noise-aware)
│   └── joint_v2_best.pth        # v2 joint (for reference)
├── outputs/                     # Restored image outputs
��── Dataset/
    └── train/
        ├── GT/                  # 3200 ground truth (256×256)
        ├── NoisyLR/             # 3200 degraded (128×128)
        └── Denoised/            # Generated: denoiser-v2 outputs
```

---

## ��� Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **EDSR over U-Net** | Post-upsampling = efficient; all compute at 128×128; proven SR backbone |
| **No Batch Norm** | BN removes useful dynamic range, hurts restoration (EDSR paper) |
| **Post-upsampling (not pre-)** | 2× SR is mild; heavy compute at LR resolution is optimal |
| **FFT magnitude loss** | Directly constrains frequency content — critical for semiconductor texture |
| **Noise-level augmentation** | Random extra noise teaches model to handle varying noise → better OOD |
| **No GAN** | Only 3200 samples — GAN unstable; PSNR/SSIM focus works better |
| **Raw input (no normalization)** | OOD test images may have different stats; clipping handled by Charbonnier |
| **Calibrated synthesis** | Fitted from real data — unlimited matched pairs, removes data ceiling |
| **Noise-aware SR** | SR stage trains on denoiser output distribution, not clean pooled GT |
| **D4 TTA + Ensemble** | Free accuracy boost at test time (+0.2–0.5 dB PSNR) |
| **EMA for inference** | Better generalization; validate with current weights, save EMA only |

---

## ��� Submission Checklist (KLA Hackathon)

| Requirement | Status |
|-------------|--------|
| **PPT/PDF** (8-9 slides, template) | To be created from this summary |
| **README.md** with setup instructions | �� Complete |
| **evaluate.py** (standalone, no edits needed) | �� Complete |
| **Training script** (train.py + train_v2_pipeline.sh) | �� Complete |
| **Trained model weights** (.pth in weights/) | �� Available (use Git LFS for >100MB) |
| **Restored test outputs** folder | Run `evaluate.py` on test set |
| **requirements.txt** | �� Minimal (torch, numpy, Pillow) + `lpips` for metrics |

**Critical**: `evaluate.py` CLI contract is frozen — KLA benchmarks run it AS-IS on H100.

---

## ��� Quick Start

```bash
# 1. Install dependencies (GPU requires CUDA build)
pip install torch numpy Pillow
# For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install lpips torchvision  # for VGG/LPIPS (v2+)

# 2. Train (CPU demo)
python train.py --epochs 30 --batch_size 16 --lr 1e-3

# 3. Train v2 pipeline (GPU, ~3h on RTX 3080)
bash train_v2_pipeline.sh

# 4. Evaluate on validation
python tools/eval_val.py --weights weights/denoise_v2_best_ema.pth \
                         --weights weights/sr_v2_best_ema.pth --tta

# 5. Inference on test set (submission)
python evaluate.py --input_dir test_images/ --output_dir restored_outputs/ \
  --weights weights/denoise_v2_best_ema.pth --weights weights/sr_v2_best_ema.pth --tta
```

---

## ��� Future Work (Post-Hackathon)

1. **RGB/Color support** — extend to 3-channel inputs (bonus points)
2. **Larger models** — n_feats=64, n_blocks=20+ on A100/H100
3. **Diffusion-based refinement** — for perceptual quality beyond PSNR/SSIM
4. **More aggressive TTA** — 16× (4 rotations × 4 flips) or learned ensemble weights
5. **Domain adaptation** — test-time adaptation for OOD sources
6. **ONNX export** — for production deployment

---

## ��� References

1. **EDSR**: Lim et al., "Enhanced Deep Residual Networks for Single Image Super-Resolution," CVPRW 2017
2. **Charbonnier**: Charbonnier et al., "Deterministic edge-preserving regularization," ICIP 1994
3. **MS-SSIM**: Wang et al., "Multi-scale structural similarity," IEEE TIP 2003
4. **LPIPS**: Zhang et al., "The Unreasonable Effectiveness of Deep Features," CVPR 2018
5. **D4 Self-Ensemble**: Lim et al., "EDSR," CVPRW 2017 (self-ensemble section)
6. **VGG Perceptual**: Johnson et al., "Perceptual Losses for Real-Time Style Transfer," ECCV 2016

---

*Generated: 2026-08-14 | Project: Restoration_Semiconductor | Hackathon: KLA i4C*