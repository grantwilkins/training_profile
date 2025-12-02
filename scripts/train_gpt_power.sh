#!/usr/bin/env bash
# train_gpt_power.sh — power monitoring & fault‑injection wrapper for train_gpt.py
# Patched 2025‑07‑26

set -euo pipefail

DATE_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
LOG=power-trace_${DATE_TIME}.csv

# ----- launch nvidia‑smi sampler (100 ms) -----
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -lms 100 >> "$LOG" &
SMI_PID=$!
echo "NVML logging to $LOG (pid=$SMI_PID)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ----- start training (Ray Train) -----
python3 train_gpt.py \
  --model_size 350M \
  --sequence_length 1024 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --precision bf16 \
  --max_steps 4000 \
  --use_flash_attn \
  --use_fused_optimizer \
  --output_dir /home/dennis/ \
  --log_steps 20 \
  --save_steps 1000 \
  --monitor_memory \
  --log_timing \
  --resources_per_worker '{"GPU": 1, "CPU": 4}' \
  --num_workers 0 \
  --ray_address auto || true

echo "Training finished; stopping NVML logger" >&2
kill -9 "$SMI_PID"
echo "Power trace written to $LOG"
