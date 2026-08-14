#!/usr/bin/env python3
"""
Evaluate / inference script.

Usage:
    python evaluate.py --input_dir <test_images> --output_dir <restored_outputs>
    python evaluate.py --input_dir <test_images> --output_dir <outputs> --gt_dir <ground_truth>
    python evaluate.py --input_dir <test_images> --output_dir <outputs> --weights path/to/model.pth
    # Ensemble two models with D4 test-time augmentation:
    python evaluate.py --input_dir <test_images> --output_dir <outputs> \
        --gt_dir <ground_truth> --tta \
        --weights weights/opt1_best_ema.pth --weights weights/opt2_best_ema.pth
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
                        help='Path to model weights. Repeatable to ensemble several '
                             'models. Default: weights/best.pth')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Optional ground truth directory for computing metrics')
    parser.add_argument('--tta', action='store_true',
                        help='Apply D4 test-time augmentation (self-ensemble)')
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

    run_inference(models, args.input_dir, args.output_dir, cfg, args.gt_dir, tta=args.tta)


if __name__ == '__main__':
    main()
