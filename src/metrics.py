"""
Evaluation metrics: PSNR and SSIM (pure PyTorch, no torchvision needed).
"""

import torch
import torch.nn.functional as F


def psnr(pred, target, data_range=1.0):
    """
    Peak Signal-to-Noise Ratio (dB).
    Higher is better. Standard metric for image restoration.
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10.0 * torch.log10(data_range ** 2 / mse)


def _ssim_single(img1, img2, window_size=11, data_range=1.0):
    """Compute SSIM for a single-channel image pair."""
    from .losses import _create_window, _ssim_map
    channel = 1
    window = _create_window(window_size, channel).to(img1.device, img1.dtype)
    ssim_map = _ssim_map(img1, img2, window, window_size, channel, data_range)
    return ssim_map.mean()


def ssim(pred, target, data_range=1.0):
    """
    Structural Similarity Index (0 to 1).
    Higher is better. Measures structural similarity (edges, textures).
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    channel = pred.size(1)
    results = []
    for i in range(channel):
        results.append(_ssim_single(pred[:, i:i+1], target[:, i:i+1],
                                     window_size=11, data_range=data_range))
    return torch.stack(results).mean()


def compute_all_metrics(pred, target, data_range=1.0):
    """Compute PSNR and SSIM. Returns dict of scalar tensors."""
    return {
        'psnr': psnr(pred, target, data_range),
        'ssim': ssim(pred, target, data_range),
    }


_LPIPS_NET = None


def lpips_score(pred, target, net=None):
    """
    Learned Perceptual Image Patch Similarity (lower is better).

    Lazily loads the official LPIPS 'alex' model on first use. Inputs are
    single-channel [0,1] tensors [B,1,H,W]; replicated to 3 channels in [-1,1].
    Returns a [B] tensor (per-image distance).
    """
    global _LPIPS_NET
    if net is None:
        import lpips
        if _LPIPS_NET is None:
            _LPIPS_NET = lpips.LPIPS(net='alex')
        net = _LPIPS_NET
    net.to(pred.device).eval()
    p = pred.float().repeat(1, 3, 1, 1) * 2.0 - 1.0
    t = target.float().repeat(1, 3, 1, 1) * 2.0 - 1.0
    return net(p, t).squeeze()
