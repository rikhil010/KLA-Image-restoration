#!/usr/bin/env python3
"""
Evaluation / inference script (standalone, no edits required).

Chains a denoiser (128x128 -> 128x128) into an SR model (128x128 -> 256x256)
when two weights are given, or ensembles them if --ensemble is passed.

Usage:
    python evaluate.py --input_dir <test_images> --output_dir <restored_outputs> \
        --weights weights/denoise_v3_best_ema.pth \
        --weights weights/sr_v3_best_ema.pth --tta
    python evaluate.py --input_dir <test_images> --output_dir <outputs> \
        --gt_dir <ground_truth> --tta
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import get_config
from src.evaluate import load_model, run_inference


def main():
    parser = argparse.ArgumentParser(description='Restore degraded semiconductor images')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing degraded input images (.npy or .png)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for restored output images')
    parser.add_argument('--weights', type=str, action='append', default=None,
                        help='Path to model weights. Repeatable. Default: weights/best_ema.pth')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Optional ground truth directory for computing metrics')
    parser.add_argument('--tta', action='store_true',
                        help='Apply D4 test-time augmentation (self-ensemble)')
    parser.add_argument('--chain', action='store_true', default=True,
                        help='Chain models sequentially (denoiser -> SR). '
                             'Disable with --no-chain to average/ensemble instead.')
    parser.add_argument('--no-chain', dest='chain', action='store_false')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    cfg = get_config(device=args.device)

    weight_paths = args.weights or [None]
    models = []
    for wp in weight_paths:
        m, m_cfg = load_model(cfg, wp, return_cfg=True)
        models.append(m)
        from src.model import count_parameters
        total, _ = count_parameters(m)
        print(f"  Loaded {m_cfg.model_name} ({total:,} params) from {wp or 'default'}")

    run_inference(models, args.input_dir, args.output_dir, cfg,
                  args.gt_dir, tta=args.tta, chain=args.chain)


if __name__ == '__main__':
    main()
