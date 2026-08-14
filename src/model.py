"""
Model architectures for image restoration.

Two options:
  - "edsr": Compact post-upsampling residual network (primary, fast, proven for SR)
  - "unet": Residual U-Net with pixel-shuffle output (alternative, strong denoiser)

Both share: residual learning, no batch norm, pixel-shuffle 2x upsampling at the tail.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Pre-activation residual block (Conv → ReLU → Conv) + skip + res_scale."""

    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.block(x) * self.res_scale


class UpsamplePS(nn.Module):
    """2x upsampling via pixel-shuffle (sub-pixel convolution)."""

    def __init__(self, n_feats, scale=2):
        super().__init__()
        self.body = nn.Conv2d(n_feats, n_feats * scale * scale, 3, padding=1)
        self.ps = nn.PixelShuffle(scale)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.ps(self.body(x)))


# ── EDSR (primary model) ───────────────────────────────────────────────────

class EDSR(nn.Module):
    """
    Compact post-upsampling residual network.

    Architecture:
        Input (1×128×128)
        → Head conv: 1 → n_feats
        → Body: N residual blocks + global skip
        → UpsamplePS (2x pixel-shuffle): n_feats channels @ 256×256
        → Tail conv: n_feats → 1
        Output (1×256×256)

    All compute happens at 128×128 except the final 2x upsample — efficient.
    No batch normalization (BN degrades restoration quality as shown by EDSR).
    """

    def __init__(self, in_channels=1, out_channels=1,
                 n_feats=32, n_blocks=12, res_scale=0.1):
        super().__init__()

        # Head
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)

        # Body: residual blocks + global skip connection
        blocks = [ResidualBlock(n_feats, res_scale) for _ in range(n_blocks)]
        self.body = nn.Sequential(*blocks)
        self.body_conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        # Upsampling tail
        self.upsample = UpsamplePS(n_feats, scale=2)

        # Tail
        self.tail = nn.Conv2d(n_feats, out_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 1, 128, 128]
        head = self.head(x)                      # [B, C, 128, 128]
        body = self.body(head)                    # [B, C, 128, 128]
        body = self.body_conv(body) + head        # global residual from head
        out = self.upsample(body)                 # [B, C, 256, 256]
        out = self.tail(out)                      # [B, 1, 256, 256]
        return out


# ── U-Net (alternative model) ───────────────────────────────────────────────

class UNetResBlock(nn.Module):
    """Residual block for U-Net stages."""

    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.block(x) * self.res_scale


class UNet(nn.Module):
    """
    Residual U-Net with optional pixel-shuffle 2x output.

    Encoder downsamples: 128 → 64 → 32
    Decoder upsamples:   32 → 64 → 128 → (256 via pixel-shuffle if upsample_tail)

    Skip connections between encoder and decoder preserve high-frequency detail.
    Residual blocks in each stage help with denoising.

    With upsample_tail=True: full denoise+SR model (input 128x128 → output 256x256).
    With upsample_tail=False: pure denoiser (input 128x128 → output 128x128),
    used as stage 1 of the two-stage pipeline.
    """

    def __init__(self, in_channels=1, out_channels=1,
                 n_feats=32, n_blocks=4, res_scale=0.1, upsample_tail=True):
        super().__init__()
        self.upsample_tail = upsample_tail

        # Encoder
        self.enc1 = self._stage(in_channels, n_feats, n_blocks, res_scale)  # 128→128
        self.enc2 = self._stage(n_feats, n_feats * 2, n_blocks, res_scale) # 128→64
        self.enc3 = self._stage(n_feats * 2, n_feats * 4, n_blocks, res_scale) # 64→32

        self.pool = nn.MaxPool2d(2)

        # Bottleneck at 32×32
        self.bottleneck = self._stage(n_feats * 4, n_feats * 4, n_blocks, res_scale)

        # Decoder. First skip concats bottleneck (4C) with e3 (4C), both at 32×32.
        self.dec3 = self._stage(n_feats * 8, n_feats * 2, n_blocks, res_scale)  # concat skip

        self.up2 = nn.ConvTranspose2d(n_feats * 2, n_feats * 2, 2, stride=2)
        self.dec2 = self._stage(n_feats * 4, n_feats, n_blocks, res_scale)

        self.up1 = nn.ConvTranspose2d(n_feats, n_feats, 2, stride=2)
        self.dec1 = self._stage(n_feats * 2, n_feats, n_blocks, res_scale)

        # Final 2x upsample via pixel-shuffle (128 → 256); skipped for pure denoiser
        self.upsample = UpsamplePS(n_feats, scale=2) if upsample_tail else None

        # Output
        self.tail = nn.Conv2d(n_feats, out_channels, 3, padding=1)

        self._init_weights()

    @staticmethod
    def _stage(in_ch, out_ch, n_blocks, res_scale):
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [UNetResBlock(out_ch, res_scale) for _ in range(n_blocks)]
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)         # [B, C,   128, 128]
        e2 = self.enc2(self.pool(e1))  # [B, 2C, 64, 64]
        e3 = self.enc3(self.pool(e2))  # [B, 4C, 32, 32]

        # Bottleneck
        b = self.bottleneck(e3)    # [B, 4C, 32, 32]

        # Decoder (bottleneck and e3 are both at 32×32)
        d3 = self.dec3(torch.cat([b, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # 2x pixel-shuffle to 256×256 (skipped for pure denoiser)
        if self.upsample is not None:
            d1 = self.upsample(d1)
        out = self.tail(d1)
        return out


# ── Two-stage pipeline (U-Net denoise → EDSR SR) ───────────────────────────

class DenoiseUNet(UNet):
    """
    Pure denoiser: residual U-Net without the 2x SR tail.
    Input 128×128 → output 128×128. Stage 1 of the two-stage pipeline.
    """

    def __init__(self, in_channels=1, out_channels=1,
                 n_feats=32, n_blocks=4, res_scale=0.1):
        super().__init__(in_channels, out_channels,
                         n_feats, n_blocks, res_scale, upsample_tail=False)


class TwoStageModel(nn.Module):
    """
    Denoise-then-SR pipeline: DenoiseUNet (128→128) → EDSR (128→256).

    Usage:
      - Pre-train DenoiseUNet alone  (model_name="denoise_unet", stage="denoise")
      - Pre-train EDSR alone         (model_name="edsr",          stage="sr")
      - Joint fine-tune both         (model_name="two_stage",     stage="joint",
                                      optionally with --pretrained_denoiser / --pretrained_sr)
    """

    def __init__(self, in_channels=1, out_channels=1,
                 n_feats=32, n_blocks=12, res_scale=0.1, unet_blocks=4):
        super().__init__()
        self.denoiser = DenoiseUNet(in_channels, out_channels,
                                    n_feats, unet_blocks, res_scale)
        self.sr = EDSR(in_channels, out_channels,
                       n_feats, n_blocks, res_scale)

    def forward(self, x):
        return self.sr(self.denoiser(x))

    def load_stage(self, name, path):
        """Load a pre-trained stage ('denoiser' | 'sr') from a standalone checkpoint."""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        getattr(self, name).load_state_dict(sd)
        print(f"  Loaded {name} weights from {path}")


# ── Factory ─────────────────────────────────────────────────────────────────

def build_model(cfg):
    """Build model based on config. Returns model on specified device."""
    if cfg.model_name == "edsr":
        model = EDSR(
            in_channels=1, out_channels=1,
            n_feats=cfg.n_feats, n_blocks=cfg.n_blocks,
            res_scale=cfg.res_scale,
        )
    elif cfg.model_name == "unet":
        model = UNet(
            in_channels=1, out_channels=1,
            n_feats=cfg.n_feats, n_blocks=cfg.n_blocks,
            res_scale=cfg.res_scale,
        )
    elif cfg.model_name == "denoise_unet":
        model = DenoiseUNet(
            in_channels=1, out_channels=1,
            n_feats=cfg.n_feats, n_blocks=cfg.unet_blocks,
            res_scale=cfg.res_scale,
        )
    elif cfg.model_name == "two_stage":
        model = TwoStageModel(
            in_channels=1, out_channels=1,
            n_feats=cfg.n_feats, n_blocks=cfg.n_blocks,
            res_scale=cfg.res_scale, unet_blocks=cfg.unet_blocks,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model_name}")

    model = model.to(cfg.device)
    return model


def count_parameters(model):
    """Return total and trainable parameter count."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
