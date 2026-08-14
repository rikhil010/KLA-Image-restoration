"""
Loss functions for image restoration.

Combines four complementary losses:
  1. Charbonnier L1 — robust to speckle outliers (smooth L1 approximation)
  2. MS-SSIM — structure-aware, directly optimizes the evaluation metric
  3. FFT magnitude loss — enforces high-frequency texture reconstruction
  4. Gradient (Sobel edge) loss — preserves sharp edges without ringing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False


# ── Charbonnier L1 ──────────────────────────────────────────────────────────

class CharbonnierL1(nn.Module):
    """
    Smooth L1 approximation: L1(x) ≈ sqrt(x² + ε²).
    More robust than L1 to extreme outliers (speckle pushes values beyond range).
    Avoids the non-differentiable point at 0 that pure L1 has.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


# ── MS-SSIM ─────────────────────────────────────────────────────────────────

def _gaussian_window(size=11, sigma=1.5):
    """Create a 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    return g / g.sum()


def _create_window(window_size=11, channel=1):
    """Create 2D Gaussian window for SSIM computation."""
    _1D = _gaussian_window(window_size).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).unsqueeze(0).unsqueeze(0)
    window = _2D.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim_map(img1, img2, window, window_size, channel, data_range=1.0, k1=0.01, k2=0.03):
    """Compute per-pixel SSIM map between two images."""
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=pad if False else channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map


class MS_SSIM_Loss(nn.Module):
    """
    Multi-Scale SSIM loss.  Loss = 1 - MS_SSIM(pred, target).
    Operates at multiple window sizes (11, 7, 3) to capture structures at different scales.
    """

    def __init__(self, data_range=1.0, window_size=11):
        super().__init__()
        self.data_range = data_range
        self.window_size = window_size

    def forward(self, pred, target):
        channel = pred.size(1)
        ms_ssim_val = 0.0
        weights = [0.5, 0.3, 0.2]
        for w, ws in zip(weights, [11, 7, 3]):
            window = _create_window(ws, channel).to(pred.device, pred.dtype)
            ssim_map = _ssim_map(pred, target, window, ws, channel, self.data_range)
            ms_ssim_val += w * ssim_map.mean()
        return 1.0 - ms_ssim_val


# ── FFT magnitude loss ──────────────────────────────────────────────────────

class FFTMagnitudeLoss(nn.Module):
    """
    Penalize the L1 difference between Fourier magnitudes.
    Forces the model to reconstruct high-frequency patterns (texture, edges)
    that are critical for semiconductor images.
    """

    def forward(self, pred, target):
        # Compute in float32 for numerical stability (avoid ComplexHalf under AMP)
        pred_fft = torch.fft.fft2(pred.float(), norm='ortho')
        target_fft = torch.fft.fft2(target.float(), norm='ortho')
        # L1 on magnitudes
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


# ── Gradient (Sobel edge) loss ──────────────────────────────────────────────

class GradientLoss(nn.Module):
    """
    Compute L1 loss on Sobel edge magnitudes.
    Preserves sharp edges without inducing ringing artifacts.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels (normalized)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _gradient_magnitude(self, x):
        # Cast kernels to input dtype AND device so GPU/fp16 (AMP) training works
        gx = F.conv2d(x, self.sobel_x.to(device=x.device, dtype=x.dtype), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(device=x.device, dtype=x.dtype), padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred, target):
        g_pred = self._gradient_magnitude(pred)
        g_target = self._gradient_magnitude(target)
        return F.l1_loss(g_pred, g_target)


# ── Perceptual (VGG) loss ───────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    L1 distance in VGG16 feature space (relu1_2, relu2_2, relu3_3, relu4_3).

    Aligns the output with human-perceived texture/structure — improves visual
    sharpness and the LPIPS axis of the KLA evaluation. Inputs are single-channel
    [0,1] images; replicated to 3 channels and ImageNet-normalized.
    """

    def __init__(self):
        super().__init__()
        assert _HAS_TORCHVISION, "PerceptualLoss requires torchvision"
        vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = nn.ModuleList([
            vgg[0:4],   # relu1_2
            vgg[4:9],   # relu2_2
            vgg[9:16],  # relu3_3
            vgg[16:23], # relu4_3
        ])
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval()

    def forward(self, pred, target):
        # pred/target: [B,1,H,W] float32 in [0,1]
        pred = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        for layer in self.layers:
            pred = layer(pred)
            target = layer(target)
            loss = loss + F.l1_loss(pred, target)
        return loss


# ── Combined loss ───────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Weighted sum of Charbonnier L1, MS-SSIM, FFT, gradient, and (optional)
    VGG perceptual losses. Weights are configurable via the config object.
    """

    def __init__(self, cfg):
        super().__init__()
        self.charb = CharbonnierL1(eps=cfg.charb_eps)
        self.ssim = MS_SSIM_Loss(data_range=1.0)
        self.fft = FFTMagnitudeLoss()
        self.grad = GradientLoss()
        self.vgg = None
        if getattr(cfg, 'loss_vgg_w', 0.0) > 0 and _HAS_TORCHVISION:
            self.vgg = PerceptualLoss()
        self.w_l1 = cfg.loss_l1_w
        self.w_ssim = cfg.loss_ssim_w
        self.w_fft = cfg.loss_fft_w
        self.w_grad = cfg.loss_grad_w
        self.w_vgg = getattr(cfg, 'loss_vgg_w', 0.0)

    def forward(self, pred, target):
        # Compute all losses in float32 for numerical stability and AMP
        # compatibility (inputs may be fp16 under autocast).
        pred = pred.float()
        target = target.float()
        l1 = self.charb(pred, target)
        ssim = self.ssim(pred, target)
        fft = self.fft(pred, target)
        grad = self.grad(pred, target)
        total = self.w_l1 * l1 + self.w_ssim * ssim + self.w_fft * fft + self.w_grad * grad
        loss_dict = {
            'l1': l1.item(),
            'ssim_loss': ssim.item(),
            'fft': fft.item(),
            'grad': grad.item(),
            'total': total.item(),
        }
        if self.vgg is not None:
            vgg = self.vgg(pred, target)
            total = total + self.w_vgg * vgg
            loss_dict['vgg'] = vgg.item()
            loss_dict['total'] = total.item()
        return total, loss_dict
