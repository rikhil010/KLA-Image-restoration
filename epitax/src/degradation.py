"""
Calibrated degradation model — re-synthesizes NoisyLR from GT.

Calibration (Phase 1, measured empirically on Dataset/train):
  - Clean content of NoisyLR  = exact 2x mean-pool of GT (best predictor)
  - Global gain ~1.02, offset ~ -0.01
  - Noise is MULTIPLICATIVE (speckle): residual std grows with signal intensity
      fitted n_s: core Gaussian sigma ~0.16 + rare spikes (sigma ~0.65, ~1.5%
      of pixels) -> reproduces the heavy tails and extreme values (LR up to ~2.2)
  - Plus a small ADDITIVE Gaussian sigma ~0.026
  - n_s overall std ~0.17, mild positive skew, heavy tails (max/std ~9)

Used by the dataset when cfg.use_synth_degradation is on: each epoch generates
fresh matched noisy-LR pairs from the GT (effectively unlimited data, robust to
overfitting to the 3200 fixed NoisyLR files).
"""

import os
import numpy as np

# ── Fitted constants (from calibration) ─────────────────────────────────────
# Clean content of NoisyLR = mean_pool(GT) * SCALE + OFFSET.
SCALE = 1.02              # global gain of clean content
OFFSET = -0.01            # global offset
GAUSS_STD = 0.026         # additive Gaussian (residual floor at zero signal)

# Multiplicative speckle distribution, measured empirically from real NoisyLR
# (n_s = residual / signal for high-signal pixels). Sampling it directly
# reproduces the exact shape: std ~0.17, mild positive skew, heavy tails
# (max ~7 sigma) that generate the extreme NoisyLR values (up to ~2.2).
_NS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "n_s_real.npy")
_N_S = None


def _load_ns():
    global _N_S
    if _N_S is None:
        _N_S = np.load(_NS_PATH).astype(np.float64)
    return _N_S


def mean_pool2x_np(x):
    """Exact 2x mean-pool of a [2H,2W] array -> [H,W] (works for any even size)."""
    h, w = x.shape
    return x.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def degrade(clean, rng, jitter=True, noise_level=1.0):
    """Add calibrated degradation to a clean [128,128] float32 image.

    Args:
        clean: [128,128] float32 clean LR content (typically mean_pool(GT)).
        rng: numpy RandomState (deterministic per dataset).
        jitter: per-sample randomize gain/offset (OOD robustness).
        noise_level: multiplier on noise std (e.g. ~U(0.8, 1.3)) for
                     noise-level augmentation across the dataset.
    Returns:
        [128,128] float32 NoisyLR (values can exceed [0,1]).
    """
    x = clean.astype(np.float64)

    # Global intensity gain/offset
    if jitter:
        scale = SCALE + rng.randn() * 0.02
        offset = OFFSET + rng.randn() * 0.01
    else:
        scale, offset = SCALE, OFFSET
    x = x * scale + offset

    # Multiplicative speckle sampled from the real empirical distribution
    ns = _load_ns()
    idx = rng.randint(0, len(ns), size=x.shape)
    n = ns[idx] * noise_level
    x = x * (1.0 + n)

    # Additive Gaussian
    x = x + rng.randn(*x.shape) * (GAUSS_STD * noise_level)

    return x.astype(np.float32)
