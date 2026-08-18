"""
Training loop for semiconductor image restoration.

Features:
  - Adam optimizer with cosine annealing LR schedule
  - EMA (Exponential Moving Average) of model weights
  - Gradient clipping for stability
  - Early stopping on validation SSIM
  - Checkpoint saving (best + periodic)
  - Progress logging with per-epoch metrics
"""

import os
import time
import copy

import torch
from torch.utils.data import DataLoader

from .config import get_config
from .dataset import PairDataset, load_pairs, split_data
from .model import build_model, count_parameters
from .losses import CombinedLoss
from .metrics import compute_all_metrics


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        for name, buf in model.named_buffers():
            self.shadow[name] = buf.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        for name, buf in self.model.named_buffers():
            self.shadow[name] = buf.data.clone()

    def apply_shadow(self):
        """Replace model params with EMA shadow values (for eval)."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        for name, buf in model_named_buffers(self.model):
            self.backup[name] = buf.data.clone()
            buf.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model params (after eval)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        for name, buf in model_named_buffers(self.model):
            buf.data.copy_(self.backup[name])


def model_named_buffers(model):
    """Helper to iterate named buffers."""
    return {name: buf for name, buf in model.named_buffers()}


def train_one_epoch(model, loader, criterion, optimizer, ema, cfg, scaler=None):
    """Train for one epoch. Returns average loss and metrics.

    AMP (fp16): when scaler is provided, autocast wraps only the model forward;
    the loss is computed in fp32 (CombinedLoss casts internally) for stability.
    """
    model.train()
    use_amp = scaler is not None
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0

    for batch_idx, (lr_imgs, gt_imgs) in enumerate(loader):
        lr_imgs = lr_imgs.to(cfg.device)
        gt_imgs = gt_imgs.to(cfg.device)

        optimizer.zero_grad()
        if use_amp:
            with torch.autocast('cuda', dtype=torch.float16):
                pred = model(lr_imgs)
        else:
            pred = model(lr_imgs)
        loss, loss_dict = criterion(pred, gt_imgs)

        if use_amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update()

        with torch.no_grad():
            metrics = compute_all_metrics(pred.float().clamp(0, 1), gt_imgs.float())

        total_loss += loss_dict['total']
        total_psnr += metrics['psnr'].item()
        total_ssim += metrics['ssim'].item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_psnr = total_psnr / max(n_batches, 1)
    avg_ssim = total_ssim / max(n_batches, 1)
    return avg_loss, avg_psnr, avg_ssim


@torch.no_grad()
def validate(model, loader, criterion, cfg):
    """Validate on val set. Returns average loss, PSNR, SSIM."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0

    for lr_imgs, gt_imgs in loader:
        lr_imgs = lr_imgs.to(cfg.device)
        gt_imgs = gt_imgs.to(cfg.device)
        pred = model(lr_imgs)
        loss, loss_dict = criterion(pred, gt_imgs)
        metrics = compute_all_metrics(pred.clamp(0, 1), gt_imgs)
        total_loss += loss_dict['total']
        total_psnr += metrics['psnr'].item()
        total_ssim += metrics['ssim'].item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_psnr = total_psnr / max(n_batches, 1)
    avg_ssim = total_ssim / max(n_batches, 1)
    return avg_loss, avg_psnr, avg_ssim


def save_checkpoint(state, path):
    """Save model checkpoint."""
    torch.save(state, path)


def run_training(cfg=None):
    """Main training routine. Call this from train.py entrypoint."""
    if cfg is None:
        cfg = get_config()

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    # ── Data ────────────────────────────────────────────────────────────
    print("Loading dataset...")
    gt_paths, lr_paths = load_pairs(cfg.data_dir)
    (train_gt, train_lr), (val_gt, val_lr) = split_data(gt_paths, lr_paths, cfg.val_split, cfg.seed)
    print(f"  Train: {len(train_gt)} pairs | Val: {len(val_gt)} pairs")

    # Train on synthetic degradation (unlimited matched pairs) when enabled;
    # validation always uses the REAL NoisyLR for a stable, honest metric.
    train_ds = PairDataset(train_gt, train_lr, cfg, augment=True,
                           synth=cfg.use_synth_degradation)
    val_ds = PairDataset(val_gt, val_lr, cfg, augment=False, synth=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    # ── Model ───────────────────────────────────────────────────────────
    model = build_model(cfg)
    total, trainable = count_parameters(model)
    print(f"  Model: {cfg.model_name} (stage={cfg.stage}) | "
          f"Params: {total:,} ({trainable:,} trainable)")

    # Stage-wise pre-training → joint fine-tuning: load pre-trained stages
    if cfg.model_name == "two_stage":
        if cfg.pretrained_denoiser:
            model.load_stage("denoiser", cfg.pretrained_denoiser)
        if cfg.pretrained_sr:
            model.load_stage("sr", cfg.pretrained_sr)

    # Checkpoint filename prefix per stage, so stages don't clobber each other
    prefix = cfg.weights_prefix or (
        "denoise" if cfg.model_name == "denoise_unet" else
        "sr" if cfg.stage == "sr" else "")

    def ckpt_path(name):
        return os.path.join(cfg.weights_dir,
                            f"{prefix}_{name}.pth" if prefix else f"{name}.pth")

    # ── Loss + Optimizer ────────────────────────────────────────────────
    criterion = CombinedLoss(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # fp16 mixed precision (AMP) on GPU
    scaler = (torch.amp.GradScaler('cuda')
              if str(cfg.device).startswith('cuda') else None)
    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

    ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None

    # ── Training loop ───────────────────────────────────────────────────
    best_metric = -1.0
    patience_counter = 0
    best_model_state = None

    print(f"\nStarting training on {cfg.device} for {cfg.epochs} epochs")
    print("-" * 70)
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t_epoch = time.time()

        # Train
        train_loss, train_psnr, train_ssim = train_one_epoch(
            model, train_loader, criterion, optimizer, ema, cfg, scaler)

        # Validate with current model weights (EMA hasn't converged yet in early epochs)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, cfg)
        if ema is not None:
            ema.update()  # update EMA shadow (don't apply for validation)

        scheduler.step()
        elapsed = time.time() - t_epoch
        lr_now = optimizer.param_groups[0]['lr']

        # Select metric for early stopping
        monitor_metric = val_ssim if cfg.early_stop_metric == "ssim" else val_psnr

        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"Train L:{train_loss:.4f} PSNR:{train_psnr:.2f} SSIM:{train_ssim:.4f} | "
              f"Val L:{val_loss:.4f} PSNR:{val_psnr:.2f} SSIM:{val_ssim:.4f} | "
              f"lr:{lr_now:.2e} | {elapsed:.1f}s")

        # Early stopping
        if monitor_metric > best_metric:
            best_metric = monitor_metric
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'cfg': cfg,
            }, ckpt_path('best'))
        else:
            patience_counter += 1

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'cfg': cfg,
            }, ckpt_path(f'epoch_{epoch:03d}'))

        if patience_counter >= cfg.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch} (no {cfg.early_stop_metric} improvement for {cfg.early_stop_patience} epochs)")
            break

    total_time = time.time() - t_start
    print("-" * 70)
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best {cfg.early_stop_metric}: {best_metric:.4f}")

    # Save final best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        save_checkpoint({
            'model_state_dict': model.state_dict(),
            'best_metric': best_metric,
            'cfg': cfg,
        }, ckpt_path('best'))
        print(f"Best model saved to {ckpt_path('best')}")

        # Also save EMA model for inference (better generalization at test time)
        if ema is not None:
            ema.apply_shadow()
            save_checkpoint({
                'model_state_dict': model.state_dict(),
                'best_metric': best_metric,
                'cfg': cfg,
                'ema': True,
            }, ckpt_path('best_ema'))
            ema.restore()
            print(f"EMA model saved to {ckpt_path('best_ema')}")

    return model, best_metric
