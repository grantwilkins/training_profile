#!/usr/bin/env bash
set -euo pipefail

# Single GPU training on Titan X (no Ray overhead)

python train_gpt_simple.py \
  --model_size 125M \
  --batch_size 1 \
  --sequence_length 512 \
  --gradient_accumulation_steps 1 \
  --precision fp16 \
  --max_steps 100 \
  --log_steps 10 \
  --dataset dummy
