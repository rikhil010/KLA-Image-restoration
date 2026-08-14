"""
Configuration for Semiconductor Image Restoration.

All hyperparameters in one place. Adjust here for experiments.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: str = "Dataset/train"
    weights_dir: str = "weights"
    outputs_dir: str = "outputs"

    # ── Data ─────────────────────────────────────────────────────────────
    input_size: int = 128          # NoisyLR spatial dimension
    output_size: int = 256         # GT spatial dimension
    upscale_factor: int = 2        # 128 → 256
    val_split: float = 0.1         # 10 % validation
    seed: int = 42

    # ── Data augmentation ────────────────────────────────────────────────
    aug_flip_h: bool = True
    aug_flip_v: bool = True
    aug_rotate: bool = True        # 90° rotations
    aug_noise_sigma_max: float = 0.03   # extra Gaussian noise (sigma)
    aug_intensity_jitter: float = 0.1   # random brightness scale ±10%

    # ── Patch (crop) training ────────────────────────────────────────────
    # Random aligned crops during training multiply effective sample count
    # (4-16x) and speed convergence. LR crop size; output crop = 2x.
    # Set to >= full input size to disable cropping.
    patch_lr: int = 64             # LR crop side (denoise: 64→64, SR/joint: 64→128)

    # ── Model ────────────────────────────────────────────────────────────
    model_name: str = "edsr"       # "edsr" | "unet" | "denoise_unet" | "two_stage"
    n_feats: int = 32              # feature channels in residual body
    n_blocks: int = 12             # residual blocks (EDSR body / two-stage SR stage)
    unet_blocks: int = 4           # residual blocks per U-Net stage (denoiser)
    res_scale: float = 0.1         # residual scaling factor

    # ── Two-stage pipeline ───────────────────────────────────────────────
    # stage="denoise": train DenoiseUNet (NoisyLR → mean_pool(GT))
    # stage="sr":      train EDSR       (mean_pool(GT) → GT)
    # stage="joint":   train TwoStageModel end-to-end (NoisyLR → GT)
    stage: str = "joint"           # "denoise" | "sr" | "joint"
    pretrained_denoiser: str = ""  # path to pre-trained DenoiseUNet checkpoint
    pretrained_sr: str = ""        # path to pre-trained EDSR checkpoint
    weights_prefix: str = ""       # checkpoint filename prefix ("" → best.pth)
    use_synth_degradation: bool = False  # re-synthesize NoisyLR from GT (Phase 1)
    denoised_dir: str = ""         # if set, SR stage reads pre-denoised LR from here
    num_workers: int = 4           # DataLoader workers (0 = main thread)

    # ── Loss ─────────────────────────────────────────────────────────────
    loss_l1_w: float = 1.0         # Charbonnier L1 weight
    loss_ssim_w: float = 0.5       # MS-SSIM weight (loss = 1 - ssim)
    loss_fft_w: float = 0.1        # FFT magnitude weight
    loss_grad_w: float = 0.1       # gradient (Sobel edge) weight
    loss_vgg_w: float = 0.0        # VGG perceptual weight (0 = disabled)
    charb_eps: float = 1e-3        # Charbonnier epsilon

    # ── Training ─────────────────────────────────────────────────────────
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-3
    lr_min: float = 1e-6
    lr_scheduler: str = "cosine"   # "cosine" | "step"
    lr_step_size: int = 50
    lr_gamma: float = 0.5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # ── EMA ──────────────────────────────────────────────────────────────
    ema_decay: float = 0.999
    use_ema: bool = True

    # ── Early stopping ───────────────────────────────────────────────────
    early_stop_patience: int = 25   # epochs without SSIM improvement
    early_stop_metric: str = "ssim"  # "ssim" or "psnr"

    # ── Checkpointing ────────────────────────────────────────────────────
    save_every: int = 10           # save checkpoint every N epochs

    # ── Device ───────────────────────────────────────────────────────────
    device: str = "auto"           # "auto" | "cuda" | "cpu"

    # ── Evaluation ───────────────────────────────────────────────────────
    eval_batch_size: int = 1       # one image at a time for memory safety

    def __post_init__(self):
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)


def get_config(**overrides) -> Config:
    """Return a Config with any keyword overrides applied."""
    return Config(**overrides)
