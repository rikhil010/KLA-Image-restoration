# Restoration_Semiconductor — Project Memory

## Project Overview
- **Task**: AI-Based Restoration of Degraded Images for Semiconductor Inspection (KLA i4C Hackathon)
- **Goal**: Train a model to jointly denoise (speckle + Gaussian) and super-resolve (2×) degraded semiconductor microscopy images
- **Input**: 128×128 grayscale float32 NoisyLR images (values can exceed [0,1] due to speckle)
- **Output**: 256×256 grayscale float32 GT images (values in [0,1])
- **Dataset**: 3200 paired .npy files in `Dataset/train/` (GT/ and NoisyLR/)
- **Evaluation**: PSNR, SSIM, LPIPS (KLA benchmarks on H100 GPU)
- **Bonus**: RGB/colour image support (future, after grayscale model works)

## Dataset Details
- Format: `.npy` files (numpy arrays), float32
- GT: 256×256, range [0, 1] exactly
- NoisyLR: 128×128, range ~[-0.28, 2.16] (speckle pushes values beyond [0,1])
- Scale factor: exactly 2× (128→256)
- GT/NoisyLR are aligned by filename (000000.npy, 000001.npy, ...)
- Correlation between NoisyLR and mean-pooled GT: ~0.98

## Architecture: Compact EDSR (Post-Upsampling Residual Network)
- **Why EDSR**: Proven SR backbone (won NTIRE 2017), efficient post-upsampling design, no BN (better for restoration)
- **Structure**: Head conv → N residual blocks (3×3 Conv→ReLU→3×3 Conv, scaled by 0.1) → global skip → PixelShuffle 2× → Tail conv
- **Config**: n_feats=24, n_blocks=6 for CPU training; n_feats=32, n_blocks=12 for GPU
- **Params**: ~89K (CPU config) / ~269K (full config)
- **Alternative**: U-Net with pixel-shuffle output also implemented (selectable via `model_name`)

## Architecture: Two-Stage Pipeline (Denoise → SR) — current GPU plan
- **Stage 1 `DenoiseUNet`** (`model_name=denoise_unet`): residual U-Net WITHOUT the SR tail, 128×128→128×128. Trained `NoisyLR → mean_pool(GT)` (clean pooled target). Fixed a latent decoder skip-misalignment bug that would have crashed the U-Net.
- **Stage 2 `EDSR`** (`model_name=edsr`, `stage=sr`): 128×128→256×256. Trained `mean_pool(GT) → GT` (pure SR, clean input).
- **Stage 3 `TwoStageModel`** (`model_name=two_stage`, `stage=joint`): chain `EDSR(DenoiseUNet(x))`, fine-tuned end-to-end on `NoisyLR → GT`; loads stages via `--pretrained_denoiser`/`--pretrained_sr`.
- **Data**: `stage` config re-pairs targets in the dataset via exact 2× mean-pool (`pool2x`). Noise/intensity aug disabled for stage=sr.
- **Config**: n_feats=32, n_blocks=12, unet_blocks=4 → **~3.9M params**
- **Checkpoints**: per-stage prefixes (`denoise_best.pth`, `sr_best.pth`, `best.pth`) so stages don't clobber each other
- **Loss fix**: CombinedLoss computes in float32; GradientLoss Sobel kernels cast to input device+dtype; FFT loss casts to fp32 — all required for fp16 AMP to work on GPU

## Loss Function Design
1. **Charbonnier L1** (weight=1.0): Smooth L1 approx, robust to speckle outliers beyond [0,1]
2. **MS-SSIM** (weight=0.5): Multi-scale structural similarity — directly optimizes evaluation metric
3. **FFT Magnitude Loss** (weight=0.1): Fourier magnitude L1 — forces high-frequency texture reconstruction
4. **Gradient/Sobel Edge Loss** (weight=0.1): Sobel edge magnitude difference — preserves sharp edges without ringing

## Training Strategy
- Adam optimizer, lr=1e-3, cosine annealing to 1e-6
- EMA (decay=0.999) of model weights for better generalization
- Gradient clipping (max_norm=1.0)
- Early stopping on validation SSIM (patience=25)
- Train/val split: 90%/10% with fixed seed=42

## Data Augmentation (for OOD generalization)
- Random horizontal/vertical flips
- Random 90° rotations
- Extra Gaussian noise injection (σ ~ Uniform(0, 0.03))
- Random intensity scaling (× Uniform(0.9, 1.1))

## Environment
- Python 3.12.10 on Windows 11
- **GPU: NVIDIA RTX 3080 Laptop, 16 GB VRAM** (driver 591.74, CUDA 13.1 capable) + AMD iGPU
- **RAM: 32 GB**
- **PyTorch 2.13.0 — currently the CPU-only build (`+cpu`).** GPU training requires the CUDA build:
  `pip install torch --index-url https://download.pytorch.org/whl/cu128` (then `torch.cuda.is_available()` → True)
- numpy 2.5.1, PIL 12.3.0
- torchvision 0.26.0+cu128, lpips 0.1.4 (added Aug 2026 for VGG perceptual loss + LPIPS eval)
- No tensorflow, cv2, skimage, matplotlib
- **Colab is NOT required for training** — local RTX 3080 is sufficient; Colab is only a backup for a bigger A100

## Training Results (CPU, 30 epochs, n_feats=24, n_blocks=6)
- **Total training time**: 185.2 minutes (~3 hours)
- **Best validation SSIM**: 0.7437 (epoch 27)
- **Best validation PSNR**: 27.48 dB (epoch 26)
- **Full validation metrics (320 images)**: PSNR=27.38±4.56 dB, SSIM=0.7437±0.1692
- **Per-sample variation**: Some images as high as PSNR=35.3/SSIM=0.94, some as low as PSNR=21.0/SSIM=0.28
- **Weights saved**: `weights/best.pth` (best by val SSIM) + `weights/best_ema.pth` (EMA weights)
- **Key fix during training**: Original EMA validation applied EMA shadow weights too early (still ~83% random at epoch 3). Fixed to validate with current model weights and only save EMA for inference.

## Training Speed (measured)
- **CPU** (n_feats=24, n_blocks=6): ~1.49 s/batch, ~6 min/epoch (batch=16)
- **GPU — RTX 3080 Laptop, fp16 AMP** (two-stage 3.9M params, batch=32): **~37 s/epoch** full dataset (2880 train + 320 val); ~0.31 s/batch
- **GPU scaling**: denoise 100 ep + SR 100 ep + joint 50 ep ≈ ~2.2 h total on the RTX 3080
- Optional speedup: raise DataLoader `num_workers` (>0) to cut data-loading overhead
- Practical: full three-stage training on GPU is fast; CPU no longer needed for real training

## v2 Quality Improvements (Aug 2026)
- **Diagnostic finding**: SR-from-perfect-clean-LR ceiling ≈ SSIM 0.859 (info bound —
  the 2× mean-pool destroyed GT high-freq detail, even a perfect model can't exceed it).
  The two-stage chained model reached only 0.718 because the SR stage was trained on
  CLEAN pooled GT but at inference receives the denoiser's output with residual noise
  (SR fed raw NoisyLR = 0.50). The fix is noise-aware SR training, not more capacity
  (an 89K-param EDSR already matched the 3.9M two-stage).
- **D4 test-time augmentation** (`src/inference.py`, `--tta` in evaluate.py): average
  predictions over 8 dihedral transforms. +0.07 dB / +0.001 SSIM on existing weights.
- **Multi-model ensembling**: `--weights w1 --weights w2` averages predictions.
- **Patch/crop training**: `patch_lr` (default 64) → aligned random crops multiply
  effective samples 4-16×; val stays full-image.
- **VGG perceptual loss**: `loss_vgg_w` (VGG16 relu1_2/2_2/3_3/4_3, grayscale→3ch
  ImageNet norm). Improves visual sharpness + LPIPS axis of the KLA score.
- **Noise-aware SR**: `denoised_dir` makes the SR stage train on (pre-denoised LR → GT)
  instead of clean pooled GT, matching inference-time input distribution. Precompute via
  `tools/precompute_denoised.py`.
- **LPIPS metric**: `metrics.lpips_score()` (lpips 'alex', grayscale→3ch), used in
  `tools/eval_val.py`. LPIPS baseline (opt1+opt2 ensemble+TTA): 0.356.
- **v2 retrain configs** (see `train_v2_pipeline.sh`): denoiser n_feats=48/unet_blocks=5
  (~10M params), SR n_feats=48/n_blocks=16, both patch-trained; joint fine-tune with
  pretrained stages.
- **Final best model**: **Chained denoise_v2 (EMA) + sr_v2 with D4 TTA** (no joint FT).
  On full val (320 images): **PSNR 27.77 dB | SSIM 0.7428 | LPIPS 0.2947**.
  Previous best (opt1+opt2 ensemble+TTA): 27.44 dB / 0.7337 / 0.3562.
  - **+0.33 dB PSNR, +0.009 SSIM, -0.061 LPIPS** — the LPIPS drop is the most
    meaningful for visual fidelity on semiconductor textures.
  - Joint FT (joint_v2) degraded to 0.7312 — synth noise + VGG + low LR destabilized
    the already-optimal denoise→SR alignment. Chained pre-trained stages + TTA wins.

## File Structure
```
Restoration_Semiconductor/
├── CLAUDE.md                    # This file — project memory
├── Project details.txt          # Original problem statement
├── README.md                    # Setup + usage instructions
├── requirements.txt             # Dependencies
├── train.py                     # Training entrypoint (CLI)
├── evaluate.py                  # Evaluation entrypoint (CLI)
├── train_v2_pipeline.sh         # Sequential v2 retrain script (denoise→SR→joint)
├── src/
│   ├── __init__.py
│   ├── config.py                # All hyperparameters in one dataclass
│   ├── dataset.py               # Dataset + augmentation + patch crop + train/val split
│   ├── degradation.py           # Calibrated synthetic degradation model
│   ├── model.py                 # EDSR + U-Net + two-stage architectures
│   ├── losses.py                # Charbonnier, MS-SSIM, FFT, gradient, VGG perceptual
│   ├── metrics.py               # PSNR + SSIM + LPIPS computation
│   ├── inference.py             # D4 TTA self-ensemble + multi-model ensembling
│   ├── train.py                 # Training loop (EMA, early stop, checkpointing)
│   └── evaluate.py              # Inference logic
├── tools/
│   ├── eval_val.py              # Held-out val eval: PSNR/SSIM/LPIPS, TTA, ensemble
│   └── precompute_denoised.py   # Runs denoiser over NoisyLR → Dataset/train/Denoised/
├── weights/                     # Saved model checkpoints
├── outputs/                     # Restored image outputs
└── Dataset/
    └── train/
        ├── GT/                  # 3200 ground truth images
        ├── NoisyLR/             # 3200 degraded images
        └── Denoised/            # (generated) denoiser-v2 outputs, for noise-aware SR
```

## Submission Requirements (KLA Hackathon)
1. **PPT/PDF**: 8-9 slides using template (team, problem, idea, solution, innovation, results, tech, github, refs)
2. **GitHub repo** must contain:
   - README.md with setup instructions
   - evaluate.py (standalone, takes input_dir + output_dir, runs without edits)
   - Training script (can be .py or notebook)
   - Trained model weights (.pth)
   - Restored test outputs folder
   - requirements.txt

## Key Decisions Log
1. **EDSR over U-Net as default**: Post-upsampling is more efficient; EDSR proven for SR; all compute at 128×128
2. **No batch norm**: BN hurts restoration (removes useful range info), as shown by EDSR paper
3. **Post-upsampling (not pre-upsampling)**: 2× SR is mild; heavy compute at 128×128 is efficient
4. **FFT loss as innovation**: Directly constrains frequency content — critical for semiconductor texture
5. **Noise-level augmentation**: Random extra noise during training teaches model to handle varying noise → better OOD
6. **No GAN**: Too few data samples (3200) for GAN stability; PSNR/SSIM focus better without GAN
7. **Raw input (no normalization)**: OOD test images may have different stats; clipping is handled by Charbonnier loss

## Future Work
- RGB/colour image support (bonus points)
- Larger model (n_feats=48, n_blocks=16) trained on GPU
- Perceptual loss with VGG (requires torchvision)
- GAN-based refinement for perceptual quality
- Test-time augmentation (TTA) for better OOD performance
