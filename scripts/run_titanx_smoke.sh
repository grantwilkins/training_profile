#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment (uncomment if needed)
# conda activate titanx-gpu

python train_gpt.py \
  --model_size 125M \
  --batch_size 1 \
  --sequence_length 512 \
  --gradient_accumulation_steps 1 \
  --precision fp16 \
  --use_flash_attn False \
  --use_fused_optimizer False \
  --max_steps 10 \
  --log_steps 1 \
  --dataset dummy \
  --num_workers 1 \
  --ray_address auto
