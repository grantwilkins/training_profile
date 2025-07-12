DATE_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -lms 250 >> llama-3-8b_${DATE_TIME}.csv &
NVIDIA_SMI_PID=$!
torchrun --nproc_per_node 8 train_llama3.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset_path ~/c4.jsonl --sequence_length 2048 --micro_batch_size 4 --global_batch_size 512 --bf16 --output_dir /scratch/ckpt --train_steps 2000
TRAINING_PID=$!
kill -9 ${NVIDIA_SMI_PID}
kill -9 ${TRAINING_PID}