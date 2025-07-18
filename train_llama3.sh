DATE_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -lms 250 >> llama-3-8b_${DATE_TIME}.csv &
NVIDIA_SMI_PID=$!

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nproc_per_node 8 train_llama3_8b.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset_path allenai/c4 --sequence_length 1024 --micro_batch_size 1 --global_batch_size 16 --bf16 --output_dir ./ --train_steps 2000 &
TRAINING_PID=$!

sleep 120 # Allow training to run for 2 minutes before interrupting it

pkill -STOP -f "LOCAL_RANK=0"
sleep 5
pkill -CONT -f "LOCAL_RANK=0"

sleep 60
pkill -9 -f "LOCAL_RANK=1"   # torchrun notices and aborts

wait $TRAINING_PID || true

kill -9 ${NVIDIA_SMI_PID}