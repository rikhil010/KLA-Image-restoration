"""
Inference utilities: D4 test-time augmentation (self-ensemble) and model ensembling.

Self-ensemble (Lim et al., EDSR): apply each of the 8 dihedral transforms to the
input, run the model, invert the transform on the output, and average. This
removes directional bias and typically gains +0.2-0.5 dB PSNR for free.

Model ensemble: average the predictions of several independently-trained models
(e.g. two-stage + single EDSR) — different architectures have complementary
errors, giving another small, reliable boost.
"""

import torch

import torch.nn.functional as F


def _d4_forward(x, k, flip):
    """Apply rot90^k then optional horizontal flip to a [B,C,H,W] tensor."""
    if k:
        x = torch.rot90(x, k, [2, 3])
    if flip:
        x = torch.flip(x, [3])
    return x


def _d4_inverse(y, k, flip):
    """Inverse of _d4_forward: undo flip first, then un-rotate."""
    if flip:
        y = torch.flip(y, [3])
    if k:
        y = torch.rot90(y, -k, [2, 3])
    return y


@torch.no_grad()
def predict_with_tta(model, x):
    """
    Self-ensemble prediction over all 8 D4 transforms.

    Args:
        model: restoration model (128x128 -> 256x256, mean-pool-compatible geometry)
        x: input tensor [B, 1, H, W] on the model's device
    Returns:
        averaged [B, 1, 2H, 2W] prediction (unclamped)
    """
    acc = None
    for k in range(4):
        for flip in (False, True):
            xi = _d4_forward(x, k, flip)
            yi = model(xi)
            yi = _d4_inverse(yi, k, flip)
            acc = yi if acc is None else acc + yi
    return acc / 8.0


@torch.no_grad()
def predict_ensemble(models, x, tta=False):
    """
    Average predictions over a list of models, optionally with self-ensemble.

    Args:
        models: list of loaded eval-mode models (all expecting [B,1,H,W])
        x: input tensor on the models' device
        tta: whether to apply D4 self-ensemble per model
    Returns:
        averaged [B, 1, 2H, 2W] prediction (unclamped)
    """
    acc = None
    for m in models:
        pred = predict_with_tta(m, x) if tta else m(x)
        acc = pred if acc is None else acc + pred
    return acc / float(len(models))
