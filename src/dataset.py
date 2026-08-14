"""
Dataset and data augmentation for paired image restoration.

Loads .npy files (GT: 256×256 float32 [0,1], NoisyLR: 128×128 float32).
Supports train/val split with fixed seed.
"""

import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .degradation import degrade, mean_pool2x_np


def pool2x(t):
    """Exact 2× mean-pool on a [1,H,W] or [B,1,H,W] tensor (256→128)."""
    if t.dim() == 3:
        return F.avg_pool2d(t.unsqueeze(0), 2).squeeze(0)
    return F.avg_pool2d(t, 2)


class PairDataset(Dataset):
    """
    Paired dataset: (degraded_input, ground_truth_target).
    Input:  NoisyLR 128×128 float32
    Target: GT        256×256 float32 [0,1]
    """

    def __init__(self, gt_paths, lr_paths, cfg, augment=False, synth=False):
        """
        synth=True: regenerate NoisyLR on the fly from GT via the calibrated
        degradation model (unlimited matched pairs). The real NoisyLR files are
        ignored; augmentation then synthesis happen on the GT.
        """
        assert len(gt_paths) == len(lr_paths), "GT and LR count mismatch"
        self.gt_paths = gt_paths
        self.lr_paths = lr_paths
        self.cfg = cfg
        self.augment = augment
        self.synth = synth
        self.rng = np.random.RandomState(cfg.seed)

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        gt = np.load(self.gt_paths[idx])   # [256, 256] float32
        # When denoised_dir is set, the SR stage consumes pre-denoised LR
        # (matched to inference-time denoiser output) instead of raw NoisyLR.
        lr_path = self.lr_paths[idx]
        if self.cfg.denoised_dir:
            lr_path = os.path.join(self.cfg.denoised_dir, os.path.basename(self.lr_paths[idx]))
        lr = np.load(lr_path)   # [128, 128] float32

        # Random augmentation (geometric only for stage="sr")
        if self.augment:
            lr, gt = self._augment(lr, gt)

        # Random aligned patch crop (training only). LR pixel (i,j) corresponds
        # to GT block [2i:2i+2, 2j:2j+2] — exact under the 2x mean-pool operator.
        C = int(getattr(self.cfg, 'patch_lr', 0) or 0)
        if C > 0 and self.augment and C < lr.shape[0]:
            i0 = self.rng.randint(0, lr.shape[0] - C + 1)
            j0 = self.rng.randint(0, lr.shape[1] - C + 1)
            lr = lr[i0:i0 + C, j0:j0 + C]
            gt = gt[2 * i0:2 * i0 + 2 * C, 2 * j0:2 * j0 + 2 * C]

        stage = self.cfg.stage
        if self.synth and stage != "sr":
            # Re-synthesize NoisyLR from GT (clean content = mean_pool(GT))
            clean = mean_pool2x_np(gt)
            noise_level = self.rng.uniform(0.8, 1.3)  # OOD noise-level jitter
            lr = degrade(clean, self.rng, jitter=True, noise_level=noise_level)

        # Add channel dim: [H,W] → [1,H,W]
        lr_t = torch.from_numpy(lr).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt).unsqueeze(0).float()

        # Stage-specific input/target pairing (mean_pool = the 2× downscale operator)
        if stage == "sr":
            if self.cfg.denoised_dir:
                # Super-resolution stage, noise-aware: pre-denoised LR in → GT out
                return lr_t, gt_t
            # Super-resolution stage: clean pooled GT in → GT out
            return pool2x(gt_t), gt_t
        elif stage == "denoise":
            # Denoising stage: NoisyLR in → clean pooled GT out
            return lr_t, pool2x(gt_t)
        else:
            # Joint: NoisyLR in → GT out
            return lr_t, gt_t

    def _augment(self, lr, gt):
        """Apply random augmentation to both LR and GT identically."""
        cfg = self.cfg

        # Horizontal flip
        if cfg.aug_flip_h and self.rng.random() > 0.5:
            lr = lr[:, ::-1].copy()
            gt = gt[:, ::-1].copy()

        # Vertical flip
        if cfg.aug_flip_v and self.rng.random() > 0.5:
            lr = lr[::-1, :].copy()
            gt = gt[::-1, :].copy()

        # 90° rotation (random multiple of 90°)
        if cfg.aug_rotate:
            k = self.rng.randint(0, 4)
            if k > 0:
                lr = np.rot90(lr, k, axes=(0, 1)).copy()
                gt = np.rot90(gt, k, axes=(0, 1)).copy()

        # Extra Gaussian noise injection (helps OOD generalization).
        # Skipped for stage="sr": its input is clean pooled GT (pure SR).
        if cfg.stage != "sr" and cfg.aug_noise_sigma_max > 0:
            sigma = self.rng.uniform(0, cfg.aug_noise_sigma_max)
            lr = lr + self.rng.randn(*lr.shape).astype(np.float32) * sigma

        # Intensity scaling (simulate contrast variation across sources).
        # Skipped for stage="sr": would break input/target correspondence.
        if cfg.stage != "sr" and cfg.aug_intensity_jitter > 0:
            scale = self.rng.uniform(1.0 - cfg.aug_intensity_jitter,
                                     1.0 + cfg.aug_intensity_jitter)
            lr = lr * scale

        return lr, gt


def load_pairs(data_dir):
    """Load all GT/LR file paths from data_dir, sorted by filename."""
    gt_dir = os.path.join(data_dir, "GT")
    lr_dir = os.path.join(data_dir, "NoisyLR")
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))
    assert len(gt_paths) == len(lr_paths), \
        f"Count mismatch: {len(gt_paths)} GT vs {len(lr_paths)} LR"
    return gt_paths, lr_paths


def split_data(gt_paths, lr_paths, val_split, seed):
    """Random train/val split with fixed seed (reproducible)."""
    rng = np.random.RandomState(seed)
    n = len(gt_paths)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_gt = [gt_paths[i] for i in train_idx]
    train_lr = [lr_paths[i] for i in train_idx]
    val_gt = [gt_paths[i] for i in val_idx]
    val_lr = [lr_paths[i] for i in val_idx]
    return (train_gt, train_lr), (val_gt, val_lr)
