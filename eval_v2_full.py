#!/usr/bin/env python3
"""
Full validation evaluation for v2 chained model (DenoiseUNet -> EDSR) with D4 TTA and LPIPS.
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config
from src.evaluate import load_model
from src.dataset import load_pairs, split_data
from src.metrics import compute_all_metrics, lpips_score
from src.inference import predict_with_tta


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--denoise_weights', type=str, required=True)
    p.add_argument('--sr_weights', type=str, required=True)
    p.add_argument('--tta', action='store_true')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--n', type=int, default=320, help='images to evaluate')
    p.add_argument('--lolips', action='store_true', help='also compute LPIPS (needs lpips package; slower)')
    args = p.parse_args()

    cfg = get_config(device=args.device)

    # Load validation split
    gt_paths, lr_paths = load_pairs(cfg.data_dir)
    (tr_gt, tr_lr), (val_gt, val_lr) = split_data(gt_paths, lr_paths, 0.1, 42)
    val_gt, val_lr = val_gt[:args.n], val_lr[:args.n]

    # Load models
    print(f"Loading denoiser from {args.denoise_weights}")
    denoiser, denoise_cfg = load_model(cfg, args.denoise_weights, return_cfg=True)
    denoiser.eval()

    print(f"Loading SR from {args.sr_weights}")
    sr_model, sr_cfg = load_model(cfg, args.sr_weights, return_cfg=True)
    sr_model.eval()

    print(f"  denoiser: {denoise_cfg.model_name} (stage={denoise_cfg.stage})")
    print(f"  sr: {sr_cfg.model_name} (stage={sr_cfg.stage})")

    ps, ss, lp = [], [], []

    with torch.no_grad():
        for i in range(len(val_gt)):
            gt = torch.from_numpy(np.load(val_gt[i])).float().to(cfg.device)[None, None]
            lr = torch.from_numpy(np.load(val_lr[i])).float().to(cfg.device)[None, None]

            # Two-stage pipeline with TTA
            if args.tta:
                # Denoiser with TTA
                denoised = predict_with_tta(denoiser, lr)
                # SR with TTA
                pred = predict_with_tta(sr_model, denoised)
            else:
                denoised = denoiser(lr)
                pred = sr_model(denoised)

            m = compute_all_metrics(pred.clamp(0, 1), gt)
            ps.append(m['psnr'].item())
            ss.append(m['ssim'].item())
            if args.lolips:
                lp.append(lpips_score(pred.clamp(0, 1), gt).mean().item())

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(val_gt)}")

    print(f"\n=== Chained Denoise+SR, TTA={args.tta}, n={len(val_gt)} ===")
    print(f"  PSNR  {np.mean(ps):.2f} dB (+/- {np.std(ps):.2f})")
    print(f"  SSIM  {np.mean(ss):.4f} (+/- {np.std(ss):.4f})")
    if lp:
        print(f"  LPIPS {np.mean(lp):.4f} (+/- {np.std(lp):.4f})")

    # Per-sample breakdown
    ps_arr = np.array(ps)
    ss_arr = np.array(ss)
    print(f"\n  Best PSNR: {np.max(ps_arr):.2f} dB")
    print(f"  Worst PSNR: {np.min(ps_arr):.2f} dB")
    print(f"  Best SSIM: {np.max(ss_arr):.4f}")
    print(f"  Worst SSIM: {np.min(ss_arr):.4f}")

    # Quartiles
    print(f"\n  PSNR Quartiles: Q1={np.percentile(ps_arr, 25):.2f}, Med={np.percentile(ps_arr, 50):.2f}, Q3={np.percentile(ps_arr, 75):.2f}")
    print(f"  SSIM Quartiles: Q1={np.percentile(ss_arr, 25):.4f}, Med={np.percentile(ss_arr, 50):.4f}, Q3={np.percentile(ss_arr, 75):.4f}")


if __name__ == '__main__':
    main()