#!/usr/bin/env python3
"""
Compare all trained models on validation set.
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
from src.inference import predict_with_tta, predict_ensemble


@torch.no_grad()
def eval_model(name, model, val_gt, val_lr, cfg, tta=False, is_two_stage=False, sr_model=None):
    """Evaluate a single model or two-stage pipeline."""
    ps, ss, lp = [], [], []

    for i in range(len(val_gt)):
        gt = torch.from_numpy(np.load(val_gt[i])).float().to(cfg.device)[None, None]
        lr = torch.from_numpy(np.load(val_lr[i])).float().to(cfg.device)[None, None]

        if is_two_stage:
            # Two-stage: denoiser -> SR
            if tta:
                denoised = predict_with_tta(model, lr)
                pred = predict_with_tta(sr_model, denoised)
            else:
                denoised = model(lr)
                pred = sr_model(denoised)
        else:
            # Single model
            if tta:
                pred = predict_with_tta(model, lr)
            else:
                pred = model(lr)

        m = compute_all_metrics(pred.clamp(0, 1), gt)
        ps.append(m['psnr'].item())
        ss.append(m['ssim'].item())
        lp.append(lpips_score(pred.clamp(0, 1), gt).mean().item())

    return np.array(ps), np.array(ss), np.array(lp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=320)
    p.add_argument('--tta', action='store_true')
    p.add_argument('--device', type=str, default='auto')
    args = p.parse_args()

    cfg = get_config(device=args.device)
    gt_paths, lr_paths = load_pairs(cfg.data_dir)
    (_, _), (val_gt, val_lr) = split_data(gt_paths, lr_paths, 0.1, 42)
    val_gt, val_lr = val_gt[:args.n], val_lr[:args.n]

    print(f"Evaluating on {len(val_gt)} validation samples (TTA={args.tta})")
    print("=" * 70)

    # Model configurations to test
    configs = [
        # (name, model_path, is_two_stage, sr_path)
        ("EDSR (v1 CPU, best_ema)", "weights/best_ema.pth", False, None),
        ("EDSR opt1 (v1 GPU, best_ema)", "weights/opt1_best_ema.pth", False, None),
        ("U-Net opt2 (v1 GPU, best_ema)", "weights/opt2_best_ema.pth", False, None),
        ("TwoStage opt1+opt2 (v1, ensemble)", None, True, ("weights/opt1_best_ema.pth", "weights/opt2_best_ema.pth")),
        ("Denoise_v2 + SR_v2 (chained, best)", None, True, ("weights/denoise_v2_best_ema.pth", "weights/sr_v2_best.pth")),
        ("Joint_v2 (end-to-end, best)", "weights/joint_v2_best.pth", False, None),
    ]

    results = {}

    for name, path, is_two_stage, sr_path in configs:
        print(f"\nLoading: {name}")
        try:
            if is_two_stage:
                denoise_path, sr_model_path = sr_path
                denoiser, _ = load_model(cfg, denoise_path, return_cfg=True)
                denoiser.eval()
                sr_model, _ = load_model(cfg, sr_model_path, return_cfg=True)
                sr_model.eval()
                model = denoiser
            else:
                model, mcfg = load_model(cfg, path, return_cfg=True)
                model.eval()
                sr_model = None

            ps, ss, lp = eval_model(name, model, val_gt, val_lr, cfg, tta=args.tta, is_two_stage=is_two_stage, sr_model=sr_model)

            results[name] = {
                'psnr_mean': np.mean(ps), 'psnr_std': np.std(ps),
                'ssim_mean': np.mean(ss), 'ssim_std': np.std(ss),
                'lpips_mean': np.mean(lp), 'lpips_std': np.std(lp),
                'psnr_best': np.max(ps), 'psnr_worst': np.min(ps),
                'ssim_best': np.max(ss), 'ssim_worst': np.min(ss),
            }

            print(f"  PSNR:  {np.mean(ps):.2f} +/- {np.std(ps):.2f} dB (best: {np.max(ps):.2f}, worst: {np.min(ps):.2f})")
            print(f"  SSIM:  {np.mean(ss):.4f} +/- {np.std(ss):.4f} (best: {np.max(ss):.4f}, worst: {np.min(ss):.4f})")
            print(f"  LPIPS: {np.mean(lp):.4f} +/- {np.std(lp):.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {'error': str(e)}

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<40} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}")
    print("-" * 70)
    for name, r in results.items():
        if 'error' not in r:
            print(f"{name:<40} {r['psnr_mean']:>7.2f}   {r['ssim_mean']:>7.4f}   {r['lpips_mean']:>7.4f}")
        else:
            print(f"{name:<40} {'ERROR':>10} {'':>10} {'':>10}")

    # Best model
    valid = {k: v for k, v in results.items() if 'error' not in v}
    if valid:
        best_psnr = max(valid.items(), key=lambda x: x[1]['psnr_mean'])
        best_ssim = max(valid.items(), key=lambda x: x[1]['ssim_mean'])
        best_lpips = min(valid.items(), key=lambda x: x[1]['lpips_mean'])
        print(f"\nBest PSNR:  {best_psnr[0]} ({best_psnr[1]['psnr_mean']:.2f} dB)")
        print(f"Best SSIM:  {best_ssim[0]} ({best_ssim[1]['ssim_mean']:.4f})")
        print(f"Best LPIPS: {best_lpips[0]} ({best_lpips[1]['lpips_mean']:.4f})")


if __name__ == '__main__':
    main()