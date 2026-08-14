#!/usr/bin/env python3
"""
Create visual comparisons: NoisyLR (input) | Restored (v2 chained) | Ground Truth | Difference maps
Uses the best v2 models: denoise_v2_best_ema.pth + sr_v2_best_ema.pth with D4 TTA
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import get_config
from src.evaluate import load_model
from src.dataset import load_pairs, split_data
from src.inference import predict_ensemble
from src.metrics import compute_all_metrics


def save_comparison_grid(name, noisy_lr, restored, gt, output_dir, metrics=None):
    """Create a 2x2 grid: NoisyLR (upscaled) | Restored | GT | |Restored-GT|"""
    # Upscale noisy LR to 256x256 for visual comparison (nearest neighbor)
    noisy_up = np.repeat(np.repeat(noisy_lr, 2, axis=0), 2, axis=1)

    # Difference map (absolute error)
    diff = np.abs(restored - gt)

    # Normalize each panel for visualization
    def normalize_for_vis(arr, vmin=0, vmax=1):
        arr = np.clip(arr, vmin, vmax)
        return ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # For diff map, use a different scale (0 to 0.5 typical)
    diff_vis = normalize_for_vis(diff, vmin=0, vmax=0.5)
    noisy_vis = normalize_for_vis(noisy_up, vmin=-0.3, vmax=2.0)  # NoisyLR range
    restored_vis = normalize_for_vis(restored)
    gt_vis = normalize_for_vis(gt)

    # Create grid: 2 rows x 2 cols
    h, w = gt.shape
    grid = np.zeros((2*h, 2*w), dtype=np.uint8)
    grid[0:h, 0:w] = noisy_vis
    grid[0:h, w:2*w] = restored_vis
    grid[h:2*h, 0:w] = gt_vis
    grid[h:2*h, w:2*w] = diff_vis

    # Save grid
    grid_path = os.path.join(output_dir, f"{name}_grid.png")
    Image.fromarray(grid).save(grid_path)

    # Also save individual panels for closer inspection
    Image.fromarray(noisy_vis).save(os.path.join(output_dir, f"{name}_noisy.png"))
    Image.fromarray(restored_vis).save(os.path.join(output_dir, f"{name}_restored.png"))
    Image.fromarray(gt_vis).save(os.path.join(output_dir, f"{name}_gt.png"))
    Image.fromarray(diff_vis).save(os.path.join(output_dir, f"{name}_diff.png"))

    # Save raw arrays too
    np.save(os.path.join(output_dir, f"{name}_restored.npy"), restored.astype(np.float32))
    np.save(os.path.join(output_dir, f"{name}_diff.npy"), diff.astype(np.float32))

    return grid_path


def main():
    # Configuration
    cfg = get_config(device='auto')

    # Load validation split (seed=42, same as training)
    gt_paths, lr_paths = load_pairs(cfg.data_dir)
    (_, _), (val_gt, val_lr) = split_data(gt_paths, lr_paths, 0.1, 42)

    # Load the best v2 models (EMA for denoiser, best for SR - no EMA available)
    denoise_path = "weights/denoise_v2_best_ema.pth"
    sr_path = "weights/sr_v2_best.pth"

    print(f"Loading denoiser from {denoise_path}")
    denoiser, _ = load_model(cfg, denoise_path, return_cfg=True)
    denoiser.eval()

    print(f"Loading SR from {sr_path}")
    sr_model, _ = load_model(cfg, sr_path, return_cfg=True)
    sr_model.eval()

    models = [denoiser, sr_model]  # Will be chained in inference

    # Output directory
    output_dir = "outputs/v2_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate on first N validation images
    n_samples = 10
    print(f"\nGenerating comparisons for {n_samples} validation samples...")

    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for i in range(min(n_samples, len(val_gt))):
            # Load data
            gt = np.load(val_gt[i])  # [256, 256]
            lr = np.load(val_lr[i])  # [128, 128]

            name = os.path.splitext(os.path.basename(val_gt[i]))[0]

            # Prepare tensors
            lr_tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).float().to(cfg.device)
            gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float().to(cfg.device)

            # Run two-stage: denoiser -> SR (with D4 TTA on each)
            # Note: predict_ensemble applies TTA to each model in sequence
            # For true chained TTA, we'd need to apply TTA to the composition
            # Here we use the chained approach: denoise -> SR

            # Stage 1: Denoise (with TTA)
            denoised = predict_ensemble([denoiser], lr_tensor, tta=True)  # [1,1,128,128]

            # Stage 2: SR (with TTA)
            restored = predict_ensemble([sr_model], denoised, tta=True)    # [1,1,256,256]

            # Get numpy arrays
            restored_np = restored.squeeze().cpu().numpy()
            restored_clamped = np.clip(restored_np, 0, 1)

            # Metrics
            m = compute_all_metrics(restored.clamp(0, 1), gt_tensor)
            psnr_val = m['psnr'].item()
            ssim_val = m['ssim'].item()
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            # Create comparison
            save_comparison_grid(name, lr, restored_clamped, gt, output_dir)

            print(f"  [{i+1}/{n_samples}] {name}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}")

    # Summary
    print(f"\n=== Summary ({len(psnr_list)} samples) ===")
    print(f"  Mean PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"  Mean SSIM: {np.mean(ssim_list):.4f}")
    print(f"  Best PSNR: {np.max(psnr_list):.2f} dB")
    print(f"  Worst PSNR: {np.min(psnr_list):.2f} dB")
    print(f"\nComparisons saved to: {output_dir}/")
    print("  *_grid.png     - 2x2 grid (NoisyLR | Restored | GT | |Diff|)")
    print("  *_noisy.png    - Upscaled noisy input")
    print("  *_restored.png - Model output")
    print("  *_gt.png       - Ground truth")
    print("  *_diff.png     - Absolute error map (0-0.5 range)")


if __name__ == "__main__":
    main()