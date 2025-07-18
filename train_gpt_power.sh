#!/bin/bash
# Power monitoring and fault injection script for train_gpt.py
# Reproduces the same power transient scenarios as train_llama3.sh

DATE_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -lms 100 >> train-gpt-8b_${DATE_TIME}.csv &
NVIDIA_SMI_PID=$!

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Start training with optimizations enabled
# Equivalent parameters to original script:
# - 8B model (instead of meta-llama/Llama-3.1-8B-Instruct)
# - allenai/c4 dataset (built-in)
# - sequence_length 1024
# - batch_size 1 per GPU
# - gradient_accumulation_steps 2 (1 * 8 * 2 = 16 global batch size)
# - bf16 precision
# - 2000 training steps
torchrun --standalone --nproc_per_node 8 train_gpt.py \
    --model_size 8B \
    --sequence_length 1024 \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --precision bf16 \
    --max_steps 2000 \
    --use_flash_attn \
    --use_fused_optimizer \
    --output_dir ./ \
    --log_steps 20 \
    --save_steps 500 \
    --monitor_memory \
    --log_timing &
TRAINING_PID=$!

sleep 300 # Allow training to run for 5 minutes before interrupting it

# Simulate synchronization stall (SIGSTOP on rank 0)
echo "Simulating synchronization stall - stopping rank 0..."
pkill -STOP -f "LOCAL_RANK=0"
sleep 5
echo "Resuming rank 0..."
pkill -CONT -f "LOCAL_RANK=0"

sleep 60

# Simulate fail-stop crash (SIGKILL on rank 1)
echo "Simulating fail-stop crash - killing rank 1..."
pkill -9 -f "LOCAL_RANK=1"   # torchrun notices and aborts

wait $TRAINING_PID || true

echo "Training finished, stopping power monitoring..."
kill -9 ${NVIDIA_SMI_PID}

echo "Power trace saved to: train-gpt-8b_${DATE_TIME}.csv"
echo "Training logs and checkpoints saved to current directory" 