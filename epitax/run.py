#!/usr/bin/env python3
"""
KLA i4C Hackathon - Semiconductor Image Restoration
Entry point: python run.py <input-dir> <output-dir>

Restores degraded 128x128 semiconductor microscopy images to 256x256
using a two-stage denoise→SR pipeline with D4 test-time augmentation.
"""

import sys
import os
import argparse

# Add epitax/src to path (makes src importable as a package)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import torch

import src.config as config_mod
import src.evaluate as evaluate_mod
import src.inference as inference_mod

get_config = config_mod.get_config
load_model = evaluate_mod.load_model
predict_with_tta = inference_mod.predict_with_tta


def load_image(path):
    """Load image from .npy file. Returns [H, W] float32."""
    return np.load(path).astype(np.float32)


def save_image(arr, path):
    """Save [H, W] float32 array as .npy file."""
    arr = np.clip(arr, 0, 1)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    # Ensure 2D output
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    np.save(path, arr)


def main():
    parser = argparse.ArgumentParser(description='KLA Semiconductor Image Restoration')
    parser.add_argument('input_dir', type=str, help='Directory containing degraded .npy images (128x128)')
    parser.add_argument('output_dir', type=str, help='Directory for restored .npy images (256x256)')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Validate input directory
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    print("Loading models...")
    cfg = get_config(device='auto')

    # Stage 1: Denoiser (128x128 -> 128x128)
    denoiser_path = os.path.join(os.path.dirname(__file__), 'models', 'denoise_v3_best_ema.pth')
    if not os.path.exists(denoiser_path):
        print(f"Error: Denoiser weights not found at {denoiser_path}")
        sys.exit(1)
    denoiser, _ = load_model(cfg, denoiser_path, return_cfg=True)
    denoiser.eval()
    print(f"  Loaded denoiser: {denoiser_path}")

    # Stage 2: SR (128x128 -> 256x256)
    sr_path = os.path.join(os.path.dirname(__file__), 'models', 'sr_v5_best.pth')
    if not os.path.exists(sr_path):
        print(f"Error: SR weights not found at {sr_path}")
        sys.exit(1)
    sr, _ = load_model(cfg, sr_path, return_cfg=True)
    sr.eval()
    print(f"  Loaded SR model: {sr_path}")

    # Find all .npy files
    npy_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith('.npy')]
    if not npy_files:
        print(f"Error: No .npy files found in {input_dir}")
        sys.exit(1)

    print(f"Processing {len(npy_files)} images...")

    # Process each image
    for i, fname in enumerate(npy_files):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        # Load degraded image [H, W]
        lr_img = load_image(input_path)  # [128, 128] float32

        # Convert to tensor: [1, 1, H, W]
        lr_tensor = torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0).float().to(cfg.device)

        # Inference: denoise -> SR with D4 TTA
        with torch.no_grad():
            denoised = denoiser(lr_tensor)                    # [1, 1, 128, 128]
            pred = predict_with_tta(sr, denoised)             # [1, 1, 256, 256]

        # Save restored image
        pred_img = pred.squeeze().cpu().numpy()               # [256, 256]
        save_image(pred_img, output_path)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(npy_files)}] {fname} -> {pred_img.shape}")

    print(f"\nDone! Restored {len(npy_files)} images to {output_dir}")


if __name__ == '__main__':
    main()