#!/usr/bin/env python3
"""
Train the image restoration model.

Usage:
    python train.py                    # default config
    python train.py --epochs 100       # override epochs
    python train.py --model_name unet  # use U-Net instead
    python train.py --n_feats 48       # wider model
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import get_config
from src.train import run_training


def main():
    parser = argparse.ArgumentParser(description='Train semiconductor image restoration model')
    # Data
    parser.add_argument('--data_dir', type=str, default='Dataset/train')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    # Model
    parser.add_argument('--model_name', type=str, default='edsr',
                        choices=['edsr', 'unet', 'denoise_unet', 'two_stage'])
    parser.add_argument('--n_feats', type=int, default=32)
    parser.add_argument('--n_blocks', type=int, default=12)
    parser.add_argument('--unet_blocks', type=int, default=4)
    parser.add_argument('--res_scale', type=float, default=0.1)
    # Two-stage pipeline
    parser.add_argument('--stage', type=str, default='joint',
                        choices=['denoise', 'sr', 'joint'])
    parser.add_argument('--pretrained_denoiser', type=str, default='')
    parser.add_argument('--pretrained_sr', type=str, default='')
    parser.add_argument('--weights_prefix', type=str, default='')
    parser.add_argument('--use_synth_degradation', action='store_true',
                        help='re-synthesize NoisyLR from GT each epoch (calibrated model)')
    parser.add_argument('--denoised_dir', type=str, default='',
                        help='SR stage reads pre-denoised LR from this dir (noise-aware SR)')
    parser.add_argument('--num_workers', type=int, default=4)
    # Patch training
    parser.add_argument('--patch_lr', type=int, default=64,
                        help='LR crop side for training (output = 2x). >=128 disables cropping.')
    # Loss
    parser.add_argument('--loss_l1_w', type=float, default=1.0)
    parser.add_argument('--loss_ssim_w', type=float, default=0.5)
    parser.add_argument('--loss_fft_w', type=float, default=0.1)
    parser.add_argument('--loss_grad_w', type=float, default=0.1)
    parser.add_argument('--loss_vgg_w', type=float, default=0.0,
                        help='VGG perceptual loss weight (0 = disabled)')
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--no_ema', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=25)
    parser.add_argument('--save_every', type=int, default=10)
    # Device
    parser.add_argument('--device', type=str, default='auto')
    # Paths
    parser.add_argument('--weights_dir', type=str, default='weights')
    parser.add_argument('--outputs_dir', type=str, default='outputs')

    args = parser.parse_args()

    cfg = get_config(
        data_dir=args.data_dir,
        val_split=args.val_split,
        seed=args.seed,
        model_name=args.model_name,
        n_feats=args.n_feats,
        n_blocks=args.n_blocks,
        unet_blocks=args.unet_blocks,
        res_scale=args.res_scale,
        stage=args.stage,
        pretrained_denoiser=args.pretrained_denoiser,
        pretrained_sr=args.pretrained_sr,
        weights_prefix=args.weights_prefix,
        use_synth_degradation=args.use_synth_degradation,
        denoised_dir=args.denoised_dir,
        num_workers=args.num_workers,
        patch_lr=args.patch_lr,
        loss_l1_w=args.loss_l1_w,
        loss_ssim_w=args.loss_ssim_w,
        loss_fft_w=args.loss_fft_w,
        loss_grad_w=args.loss_grad_w,
        loss_vgg_w=args.loss_vgg_w,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min=args.lr_min,
        lr_scheduler=args.lr_scheduler,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        use_ema=not args.no_ema,
        early_stop_patience=args.early_stop_patience,
        save_every=args.save_every,
        device=args.device,
        weights_dir=args.weights_dir,
        outputs_dir=args.outputs_dir,
    )

    run_training(cfg)


if __name__ == '__main__':
    main()
