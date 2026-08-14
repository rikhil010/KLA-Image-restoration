#!/usr/bin/env python3
"""
Generate submission outputs using chained denoiser + SR with D4 TTA.
"""

import os
import sys
import glob
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import get_config
from src.evaluate import load_model
from src.inference import predict_with_tta


def load_image(path):
    """Load an image from .npy or common image formats. Returns numpy array [H,W] float32."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path).astype(np.float32)
    elif ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'):
        img = Image.open(path)
        if img.mode == 'RGB':
            img = img.convert('L')
        return np.array(img).astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported format: {ext}")


def save_image(arr, path):
    """Save a [H,W] float32 array as a PNG image."""
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(path)


@torch.no_grad()
def run_chained_inference(denoiser, sr_model, input_dir, output_dir, cfg, tta=True):
    """Run chained denoiser -> SR inference on all images in input_dir."""
    extensions = ['*.npy', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    input_files = []
    for ext in extensions:
        input_files.extend(glob.glob(os.path.join(input_dir, ext)))
    input_files.sort()

    if not input_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Running chained inference on {len(input_files)} images (TTA={tta})...")

    for i, path in enumerate(input_files):
        name = os.path.splitext(os.path.basename(path))[0]
        lr_img = load_image(path)  # [H, W] float32

        # Prepare tensor: [1, 1, H, W]
        lr_tensor = torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0).float().to(cfg.device)

        # Stage 1: Denoise (with TTA)
        denoised = predict_with_tta(denoiser, lr_tensor) if tta else denoiser(lr_tensor)

        # Stage 2: SR (with TTA)
        restored = predict_with_tta(sr_model, denoised) if tta else sr_model(denoised)

        restored_np = restored.squeeze().cpu().numpy()
        restored_clamped = np.clip(restored_np, 0, 1)

        # Save output
        save_path_png = os.path.join(output_dir, f"{name}.png")
        save_path_npy = os.path.join(output_dir, f"{name}.npy")
        save_image(restored_clamped, save_path_png)
        np.save(save_path_npy, restored_clamped.astype(np.float32))

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(input_files)}] {name}")

    print(f"\nDone. {len(input_files)} images saved to {output_dir}")


def main():
    cfg = get_config(device='auto')

    # Load models
    print("Loading denoiser...")
    denoiser, _ = load_model(cfg, 'weights/denoise_v2_best_ema.pth', return_cfg=True)
    denoiser.eval()

    print("Loading SR...")
    sr_model, _ = load_model(cfg, 'weights/sr_v2_best.pth', return_cfg=True)
    sr_model.eval()

    # Generate outputs
    run_chained_inference(
        denoiser, sr_model,
        'Dataset/train/NoisyLR',
        'outputs/submission',
        cfg,
        tta=True
    )


if __name__ == '__main__':
    main()