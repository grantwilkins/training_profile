#!/usr/bin/env bash
set -euo pipefail

# 2 GPU training on Titan X (no Ray overhead)
# Uses native PyTorch DDP via torchrun

torchrun --nproc_per_node=2 train_gpt_simple.py \
  --model_size 125M \
  --batch_size 1 \
  --sequence_length 512 \
  --gradient_accumulation_steps 1 \
  --precision fp16 \
  --max_steps 100 \
  --log_steps 10 \
  --dataset dummy
