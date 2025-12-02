
DATE_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
LOG=../titanx-traces/power-trace_2gpu_${DATE_TIME}.csv

nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -lms 100 >> "$LOG" &
SMI_PID=$!
torchrun --nproc_per_node=2 ../training/train_gpt_simple.py \
  --model_size 350M \
  --batch_size 1 \
  --sequence_length 256 \
  --gradient_accumulation_steps 1 \
  --precision fp32 \
  --max_steps 4000 \
  --save_steps 250 \
  --log_steps 10 \
  --dataset dummy

sleep 60 # wait for the training to stop before killing the SMI process

kill -9 "$SMI_PID"
