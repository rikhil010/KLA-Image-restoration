#!/usr/bin/env python3
"""
Evaluate checkpoints on the held-out validation split (seed 42) with
PSNR / SSIM / LPIPS, optional D4 TTA and multi-model ensembling.

Usage:
    python tools/eval_val.py --weights weights/denoise_v2_best_ema.pth \
                             --weights weights/sr_v2_best_ema.pth --tta
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config
from src.evaluate import load_model
from src.dataset import load_pairs, split_data
from src.metrics import compute_all_metrics, lpips_score
from src.inference import predict_ensemble


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, action='append', required=True,
                   help='checkpoint path (repeatable for ensembles)')
    p.add_argument('--tta', action='store_true')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--n', type=int, default=320, help='images to evaluate')
    p.add_argument('--lolips', action='store_true',
                   help='also compute LPIPS (needs lpips package; slower)')
    args = p.parse_args()

    cfg = get_config(device=args.device)
    gt_paths, lr_paths = load_pairs(cfg.data_dir)
    (tr_gt, tr_lr), (val_gt, val_lr) = split_data(gt_paths, lr_paths, 0.1, 42)
    val_gt, val_lr = val_gt[:args.n], val_lr[:args.n]

    models = []
    for wp in args.weights:
        m, mcfg = load_model(cfg, wp, return_cfg=True)
        models.append(m)
        print(f"  model {mcfg.model_name} <- {wp}")

    ps, ss, lp = [], [], []
    with torch.no_grad():
        for i in range(len(val_gt)):
            gt = torch.from_numpy(np.load(val_gt[i])).float().cuda()[None, None]
            lr = torch.from_numpy(np.load(val_lr[i])).float().cuda()[None, None]
            pred = predict_ensemble(models, lr, tta=args.tta)
            m = compute_all_metrics(pred.clamp(0, 1), gt)
            ps.append(m['psnr'].item()); ss.append(m['ssim'].item())
            if args.lolips:
                lp.append(lpips_score(pred.clamp(0, 1), gt).mean().item())

    print(f"\n=== {len(models)} model(s), TTA={args.tta}, n={len(val_gt)} ===")
    print(f"  PSNR  {np.mean(ps):.2f} dB")
    print(f"  SSIM  {np.mean(ss):.4f}")
    if lp:
        print(f"  LPIPS {np.mean(lp):.4f}")


if __name__ == '__main__':
    main()
