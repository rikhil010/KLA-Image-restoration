"""
Evaluation / inference script.

Accepts a directory of degraded images (.npy or .png) and writes restored outputs.
If ground truth is provided, computes PSNR and SSIM metrics.
"""

import os
import glob
import argparse

import numpy as np
import torch
from PIL import Image

from .config import get_config
from .model import build_model, count_parameters
from .metrics import compute_all_metrics
from .inference import predict_ensemble, predict_with_tta


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


def load_model(cfg, weights_path=None, return_cfg=False):
    """Load a trained model from checkpoint. Prefers EMA weights if available.

    Returns the model (and optionally the model's own cfg).
    """
    if weights_path is None:
        ema_path = os.path.join(cfg.weights_dir, 'best_ema.pth')
        if os.path.exists(ema_path):
            weights_path = ema_path
            print(f"  Using EMA weights: {weights_path}")
        else:
            weights_path = os.path.join(cfg.weights_dir, 'best.pth')
            print(f"  Using standard weights: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=cfg.device, weights_only=False)
    if 'cfg' in checkpoint:
        # Use saved config but override device
        saved_cfg = checkpoint['cfg']
        saved_cfg.device = cfg.device
        cfg = saved_cfg
    model = build_model(cfg)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    if return_cfg:
        return model, cfg
    return model


@torch.no_grad()
def run_inference(models, input_dir, output_dir, cfg, gt_dir=None, tta=False, chain=False):
    """
    Run inference on all images in input_dir, write restored outputs to output_dir.
    If gt_dir is provided, compute PSNR/SSIM per image and print summary.

    Args:
        models: list of loaded eval-mode models.
        tta: apply D4 self-ensemble (flips/rotations) to each model.
        chain: if True, chain models sequentially (e.g. denoiser -> SR);
               if False, ensemble by averaging predictions.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    # Find input images
    extensions = ['*.npy', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    input_files = []
    for ext in extensions:
        input_files.extend(glob.glob(os.path.join(input_dir, ext)))
    input_files.sort()

    if not input_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Optionally load ground truth
    gt_files = None
    if gt_dir and os.path.isdir(gt_dir):
        gt_files = {}
        for ext in ['*.npy', '*.png', '*.jpg', '*.tif']:
            for p in glob.glob(os.path.join(gt_dir, ext)):
                name = os.path.splitext(os.path.basename(p))[0]
                gt_files[name] = p

    psnr_list = []
    ssim_list = []

    tag = "tta" if tta else ""
    print(f"Running inference on {len(input_files)} images "
          f"({len(models)} model(s){', TTA' if tta else ''})...")
    for i, path in enumerate(input_files):
        name = os.path.splitext(os.path.basename(path))[0]
        lr_img = load_image(path)  # [H, W] float32

        # Prepare tensor: [1, 1, H, W]
        lr_tensor = torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0).float().to(cfg.device)

        # Inference: chain models sequentially, or ensemble by averaging
        if chain and len(models) > 1:
            x = lr_tensor
            for m in models:
                pred = predict_with_tta(m, x) if tta else m(x)
                x = pred
            pred = x
        else:
            pred = predict_ensemble(models, lr_tensor, tta=tta)
        pred_img = pred.squeeze().cpu().numpy()

        # Save output
        save_path_png = os.path.join(output_dir, f"{name}.png")
        save_path_npy = os.path.join(output_dir, f"{name}.npy")
        save_image(pred_img, save_path_png)
        np.save(save_path_npy, pred_img)

        # Compute metrics if GT available
        if gt_files and name in gt_files:
            gt_img = load_image(gt_files[name])
            gt_tensor = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0).float().to(cfg.device)
            pred_clamped = pred.clamp(0, 1)
            m = compute_all_metrics(pred_clamped, gt_tensor)
            psnr_list.append(m['psnr'].item())
            ssim_list.append(m['ssim'].item())

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(input_files)}] {name}")

    print(f"\nDone. {len(input_files)} images saved to {output_dir}")

    # Print metrics summary if GT was provided
    if psnr_list:
        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        print(f"\nMetrics (on {len(psnr_list)} images):")
        print(f"  PSNR:  {avg_psnr:.2f} dB")
        print(f"  SSIM:  {avg_ssim:.4f}")

    return psnr_list, ssim_list


def main():
    parser = argparse.ArgumentParser(description='Restore degraded semiconductor images')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing degraded input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for restored output images')
    parser.add_argument('--weights', type=str, action='append', default=None,
                        help='Path to model weights. May be given multiple times '
                             'to ensemble several models. Default: weights/best.pth')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Optional ground truth directory for metrics')
    parser.add_argument('--tta', action='store_true',
                        help='Apply D4 test-time augmentation (self-ensemble)')
    args = parser.parse_args()

    cfg = get_config(device=args.device)

    # Load all models (ensemble members). Each keeps its own saved config.
    weight_paths = args.weights or [None]
    models = []
    for wp in weight_paths:
        m, m_cfg = load_model(cfg, wp, return_cfg=True)
        models.append(m)
        total, _ = count_parameters(m)
        print(f"  Loaded {m_cfg.model_name} ({total:,} params) from {wp or 'default'}")

    run_inference(models, args.input_dir, args.output_dir, cfg, args.gt_dir, tta=args.tta)


if __name__ == '__main__':
    main()
