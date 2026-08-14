#!/usr/bin/env python3
"""
Precompute denoised LR inputs for noise-aware SR training.

Runs a trained DenoiseUNet over every NoisyLR file and saves the outputs to a
directory. The SR stage is then trained on (denoised → GT), matching the input
distribution the SR will actually receive inside the two-stage pipeline at
inference — this fixes the measured denoise→SR coupling gap.

Usage:
    python tools/precompute_denoised.py --weights weights/denoise_v2_best.pth \
        --input_dir Dataset/train/NoisyLR --output_dir Dataset/train/Denoised
"""

import argparse
import os
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config
from src.evaluate import load_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--input_dir', type=str, default='Dataset/train/NoisyLR')
    p.add_argument('--output_dir', type=str, default='Dataset/train/Denoised')
    p.add_argument('--device', type=str, default='auto')
    args = p.parse_args()

    cfg = get_config(device=args.device)
    model, mcfg = load_model(cfg, args.weights, return_cfg=True)
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, '*.npy')))
    print(f"Denoising {len(files)} images with {mcfg.model_name} on {cfg.device}")

    for i, path in enumerate(files):
        lr = np.load(path).astype(np.float32)   # [128,128]
        t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).float().to(cfg.device)
        out = model(t).squeeze().cpu().numpy()
        name = os.path.basename(path)
        np.save(os.path.join(args.output_dir, name), out.astype(np.float32))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)}")

    print(f"Saved denoised inputs to {args.output_dir}")


if __name__ == '__main__':
    main()
