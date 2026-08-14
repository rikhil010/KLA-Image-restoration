#!/usr/bin/env bash
# v2 pipeline: improved denoiser -> noise-aware SR -> joint fine-tune.
# Run the stages in sequence. Each writes its own checkpoint prefix.
set -e
cd "$(dirname "$0")"

# 1) Improved denoiser (patch-trained, wider, unlimited synth degradation)
python train.py \
  --model_name denoise_unet --stage denoise \
  --n_feats 48 --unet_blocks 5 --patch_lr 64 --use_synth_degradation \
  --epochs 150 --batch_size 32 --num_workers 4 \
  --weights_prefix denoise_v2 --early_stop_patience 25

# 2) Precompute denoised LR inputs (match inference-time denoiser output)
python tools/precompute_denoised.py \
  --weights weights/denoise_v2_best_ema.pth \
  --input_dir Dataset/train/NoisyLR --output_dir Dataset/train/Denoised

# 3) Noise-aware SR: trained on (denoised -> GT) + VGG perceptual loss
python train.py \
  --model_name edsr --stage sr \
  --denoised_dir Dataset/train/Denoised \
  --n_feats 48 --n_blocks 16 --patch_lr 64 \
  --epochs 150 --batch_size 32 --num_workers 4 \
  --loss_vgg_w 0.05 --weights_prefix sr_v2 --early_stop_patience 30

# 4) Joint fine-tune: denoiser v2 + SR v2, end-to-end, VGG perceptual loss
python train.py \
  --model_name two_stage --stage joint \
  --n_feats 48 --n_blocks 16 --unet_blocks 5 --patch_lr 64 --use_synth_degradation \
  --epochs 100 --batch_size 32 --num_workers 4 \
  --pretrained_denoiser weights/denoise_v2_best.pth \
  --pretrained_sr weights/sr_v2_best.pth \
  --loss_vgg_w 0.05 --weights_prefix joint_v2 --early_stop_patience 25

echo "=== v2 pipeline complete ==="
