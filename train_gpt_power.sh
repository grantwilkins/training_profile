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

# ----- start training -----
CMD=(torchrun --standalone --nproc_per_node 8 train_gpt.py \
    --model_size 8B \
    --sequence_length 2048 \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --precision bf16 \
    --max_steps 4000 \
    --use_flash_attn \
    --use_fused_optimizer \
    --output_dir /datadrive/ \
    --log_steps 20 \
    --save_steps 1000 \
    --monitor_memory \
    --log_timing)

"${CMD[@]}" &
TRAIN_PID=$!

# give the job 10 min to warm‑up
sleep 600

# ----- synchronisation stall (SIGSTOP rank 0) -----
echo "Simulating sync‑stall (STOP rank 0)" >&2
RANK0=$(ps -e -o pid,cmd | grep 'train_gpt.py' | grep -v grep | head -n1 | awk '{print $1}')
kill -STOP "$RANK0"
sleep 5
kill -CONT "$RANK0"

echo "Resumed rank 0" >&2
sleep 60

# ----- fail‑stop crash (SIGKILL rank 1) -----
echo "Simulating fail‑stop (KILL rank 1)" >&2
RANK1=$(ps -e -o pid,cmd | grep 'train_gpt.py' | grep -v grep | head -n2 | tail -n1 | awk '{print $1}')
kill -9 "$RANK1"

# wait for torchrun to propagate failure
wait "$TRAIN_PID" || true

sleep 30

echo "Training finished; stopping NVML logger" >&2
kill -9 "$SMI_PID"
echo "Power trace written to $LOG"
