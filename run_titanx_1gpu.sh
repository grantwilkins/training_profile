#!/usr/bin/env bash
set -euo pipefail

# Single GPU training on Titan X (no Ray overhead)

nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -lms 100 >> log.csv &
SMI_PID=$!
python train_gpt_simple.py \
  --model_size 125M \
  --batch_size 1 \
  --sequence_length 512 \
  --gradient_accumulation_steps 1 \
  --precision fp32 \
  --max_steps 200 \
  --save_steps 100 \
  --log_steps 10 \
  --dataset dummy

kill -9 "$SMI_PID"
