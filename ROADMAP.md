# ROADMAP — Efficient Model Path to Strong Benchmark Scores

Status: v1 · 2026-08-07
Goal: Take the current CPU-demo model (PSNR 27.4 / SSIM 0.74) to a strong,
efficient restoration model trainable on **Google Colab** and benchmarkable on KLA's H100.

---

## Hard constraints this plan must respect

| Constraint | Source | Implication |
|---|---|---|
| 2× SR only (128→256), grayscale | Training data + problem statement | No multi-scale branch needed |
| Metrics: PSNR + SSIM + **LPIPS** | Problem statement | Need a perceptual loss (LPIPS is currently unoptimized) |
| **Inference time is benchmarked (~10 s/img is "useful")** | Problem statement | No diffusion, no huge models; keep ≤ ~5M params |
| OOD test set | Problem statement | Robustness via degradation-parameter jitter, not memorization |
| `evaluate.py` runs AS-IS on H100, no edits | Submission rules | CLI contract must stay frozen |
| Dataset ≈ 1.03 GB (813 MB GT + 213 MB NoisyLR) | Measured | One-time upload, fits Colab |

## Hardware: local GPU is primary, Colab is backup only

- **Laptop: RTX 3080 Laptop GPU (16 GB VRAM), 32 GB RAM.**
  This is the PRIMARY training hardware — roughly on par with a Colab T4.
- **Blocker: PyTorch is installed as the CPU-only build** (`2.13.0+cpu`).
  Install the CUDA build once:
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`
  (driver 591.74 supports CUDA 13.1; `torch.cuda.is_available()` should return True).
- Estimated local speed at fp16: **~5–15 s/epoch** (two-stage, batch 16–32)
  → 150 epochs ≈ **15–40 min**.
- **Colab is NOT needed.** It is a backup only: (a) a bigger A100 for a final
  fast run, or (b) remote demo. If ever used: zip dataset → Drive → mount in
  Colab → train → checkpoint to Drive (never only `/content`).

---

## Phase 0 — Preflight (30–60 min, laptop)
- [ ] **Calibrate the exact degradation model** from the validation split:
      `NoisyLR ≈ degrade(GT)`. Measure the downscale operator (mean-pool corr ≈ 0.98),
      Gaussian σ, and speckle parameters from the residual `NoisyLR − pool(GT)`.
- [ ] Zip dataset → `dataset.zip` (~1 GB), verify integrity, upload to Drive.
- [ ] Write `train_colab.ipynb` skeleton (Drive mount + extract + fp16 + resume).
- [ ] Install `torchvision` locally (needed for VGG perceptual loss + LPIPS eval).
- [ ] Decide Colab tier: start free T4; upgrade only if quota interrupts a run.

**Exit gate:** a calibrated `degrade(gt)` function that reproduces real NoisyLR
stats (min≈−0.28, max≈2.16, corr≈0.98).

## Phase 1 — Data leverage (highest ROI, ~half day)
- [ ] Implement **degradation re-synthesis** in `src/dataset.py`: each epoch,
      `LR = degrade(GT)` with per-sample jitter (noise level, seeds) →
      **effectively unlimited, perfectly-matched training pairs** from the 3200 GTs.
- [ ] This is the single biggest lever: it removes the data-scarcity ceiling that
      caps every other improvement. (Current augmentation only adds small Gaussian
      + flips — it never re-creates the actual speckle degradation.)
- [ ] Validate: generated LR vs real NoisyLR must match min/max ranges and corr.
      **If the degradation is wrong, the whole data-leverage bet fails — gate on this.**

## Phase 2 — Model scaling on Colab GPU (~1–2 days)
- [ ] **Run 1 (baseline at scale):** current EDSR, `n_feats=32–48, n_blocks=12–16`,
      100–150 epochs, **fp16 AMP**, EMA + cosine (already implemented).
      Checkpoint every 5 epochs → Drive. Expect a large jump over 0.74 SSIM
      from scale + data alone.
- [ ] **Run 2:** add **VGG perceptual loss** (`torchvision`, pre-trained VGG16)
      → directly optimizes LPIPS, the metric currently ignored by the loss.
- [ ] **Run 3 (optional):** swap backbone to **SwinIR-lite or RCAN** for texture;
      A/B on validation. Do this only if Runs 1–2 show the architecture is the
      bottleneck (likely it isn't).
- [ ] Log val PSNR/SSIM/LPIPS to a CSV in Drive (feeds the slides).

**Exit gate:** a validated model with val PSNR ≈ 30–32 dB, SSIM ≈ 0.88–0.93.

## Phase 3 — Inference-time gains (~half day)
- [ ] **TTA 8× self-ensemble** (4 rotations × 2 flips, average) in `evaluate.py`.
      Free accuracy, typically +0.1–0.3 dB, and trivially fast on H100.
- [ ] Optional: **2-model ensemble** (EDSR + U-Net) averaged at inference.
- [ ] Verify inference time on laptop CPU stays in the seconds/image regime
      (guarantees H100 is far under the ~10 s limit even with TTA).

## Phase 4 — Benchmark + submission (~1 day)
- [ ] Freeze a **held-out test split now** (never train on it) — KLA's test comes
      later, so build a local eval protocol that mimics it (in-dist + OOD).
- [ ] Full eval: PSNR / SSIM / **LPIPS** (use `torchmetrics.LearnedPerceptualImagePatchSimilarity`).
- [ ] Generate `outputs/` restored-test-outputs folder (submission requirement).
- [ ] Download final weights to `weights/`, commit `best.pth` (or Drive/LFS link).
- [ ] **`requirements.txt` tension:** spec wants `pip freeze` from training env,
      but reviewers run `evaluate.py` on a fresh machine → keep a **minimal**
      requirements (torch, numpy, Pillow, torchvision) + a comment listing the
      training-env extras. A broken `evaluate.py` = unscored.
- [ ] Update `README.md` (setup + inference steps), `CLAUDE.md`, this roadmap.
- [ ] Slides (template slide 6: before/after comparisons, PSNR/SSIM/LPIPS).

---

## Expected trajectory (estimates, not guarantees)

| Stage | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Current (CPU demo, 30 ep) | 27.4 | 0.74 | high |
| + data leverage + GPU scale | 29–31 | 0.85–0.90 | improved |
| + perceptual loss + TTA | 30–32 | 0.88–0.93 | much improved |

## Rules of thumb
- Fix data scarcity first — it compounds every other lever.
- Don't change the architecture until data + training are maxed out.
- Keep `evaluate.py`'s CLI frozen from now on.
- Checkpoint to Drive, never only to `/content`.
