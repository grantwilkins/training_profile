# Ray Multi-Node Training Setup Guide

This guide explains how to set up and run multi-node distributed training with Ray for power profiling experiments.

## Prerequisites

- 2+ nodes with NVIDIA GPUs (tested with 8×H100 per node)
- Network connectivity between nodes
- SSH access between nodes
- Same Python environment on all nodes

## Installation

Install required packages on **all nodes**:

```bash
pip install -U "ray[train]" torch transformers datasets
# Optional: flash-attn (SM80+ only), apex (Ampere+ and CUDA 12 builds recommended)
```

## Method 1: Manual Ray Cluster Setup (Recommended for Power Profiling)

This method gives you direct control over the Ray cluster, which is ideal for power monitoring.

### Step 1: Start Ray Head Node

On your **head node**, run:

```bash
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=2 \
    --num-cpus=16 \
    --object-store-memory=20000000000
```

This will output something like:
```
Ray runtime started.
...
To add another node to this Ray cluster, run
  ray start --address='192.168.1.10:6379'
```

**Note the IP address** (e.g., `192.168.1.10:6379`) - you'll need it for worker nodes.

### Step 2: Start Ray Worker Node(s)

On each **worker node**, run:

```bash
ray start --address='<HEAD_NODE_IP>:6379' \
    --num-gpus=2 \
    --num-cpus=16 \
    --object-store-memory=20000000000
```

Replace `<HEAD_NODE_IP>` with the IP from Step 1.

### Step 3: Verify Cluster

On the head node, verify the cluster:

```bash
ray status
```

You should see both nodes listed with their resources (4 GPUs total for 2 nodes with 2 GPUs each).

### Step 4: Run Training

From the **head node**, launch the training job:

```bash
python train_gpt.py \
    --model_size 350M \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --precision bf16 \
    --max_steps 2000 \
    --num_workers 0 \
    --resources_per_worker '{"GPU": 1, "CPU": 4}' \
    --ray_address auto \
    --save_steps 500
```

## Method 2: Ray Cluster Launcher (Alternative)

For automated setup, you can use Ray's cluster launcher with the provided config file.

### Step 1: Configure Cluster

Edit `ray_cluster_config.yaml` and update:
- `head_ip`: IP address of your head node
- `worker_ips`: IP addresses of your worker nodes
- `ssh_user`: Your SSH username
- `ssh_private_key`: Path to your SSH private key

### Step 2: Start Cluster

```bash
ray up ray_cluster_config.yaml
```

### Step 3: Run Training

```bash
ray submit ray_cluster_config.yaml train_gpt.py \
    --model_size 8B \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --precision bf16 \
    --max_steps 2000 \
    --num_workers 2
```

## Training Configuration

### Key Arguments

- `--num_workers`: Number of Ray workers (typically = number of nodes)
- `--resources_per_worker`: JSON string specifying resources per worker
  - Default: `'{"GPU": 1, "CPU": 4}'` (1 GPU per worker). With 2×12GB Titans per node, prefer 1 GPU/worker.
- `--ray_address`: Ray cluster address
  - `auto`: Connect to local Ray cluster (default)
  - `ray://<HEAD_IP>:10001`: Connect to remote Ray cluster

### Example Configurations

**2 nodes, 2 GPUs each (4 total GPUs):**
```bash
python train_gpt.py \
    --model_size 350M \
    --num_workers 0 \
    --resources_per_worker '{"GPU": 1, "CPU": 4}'
```

**4 nodes, 2 GPUs each (8 total GPUs):**
```bash
python train_gpt.py \
    --model_size 1.3B \
    --num_workers 0 \
    --resources_per_worker '{"GPU": 1, "CPU": 4}'
```

## Power Monitoring

### Monitoring During Training

Ray Train distributes training across all workers. For power monitoring:

1. **On each node**, run your power monitoring tool (e.g., `nvidia-smi dmon`, custom power logger)
2. Training runs synchronously across all nodes
3. The `--barrier_every N` flag can insert synchronization barriers every N steps for cleaner power traces

Example with barriers:
```bash
python train_gpt.py \
    --model_size 350M \
    --num_workers 0 \
    --barrier_every 10  # Sync all nodes every 10 optimizer steps
```

### Checkpoint Locations

Checkpoints are saved asynchronously to: `<output_dir>/gpt_power_profiling/TorchTrainer_*/checkpoint_*/`

## Advanced Features

### 1. Asynchronous Checkpointing

Ray Train automatically handles async checkpointing:
- Checkpoints are saved in the background while training continues
- Only rank 0 saves model state to minimize I/O
- Configure checkpoint retention with `CheckpointConfig` (currently keeping last 3)

### 2. Overlapping Communication & Computation

DDP is configured for optimal async all-reduce:
```python
gradient_as_bucket_view=True  # Optimize for async all-reduce
broadcast_buffers=False        # Reduce sync overhead
bucket_cap_mb=25              # Gradient bucket size
```

This allows gradient communication to overlap with backward computation.

### 3. Monitoring Ray Dashboard

Access the Ray dashboard at: `http://<HEAD_NODE_IP>:8265`

The dashboard shows:
- Cluster resources and utilization
- Task execution timeline
- GPU/CPU usage per node
- Training metrics

## Troubleshooting

### Connection Issues

If workers can't connect to head:
```bash
# Check Ray is running on head
ray status

# Check network connectivity
ping <HEAD_NODE_IP>

# Check port 6379 is open
telnet <HEAD_NODE_IP> 6379
```

### GPU Not Detected

```bash
# Verify GPUs are visible
nvidia-smi

# Check Ray sees GPUs
ray status  # Look for "GPU" resources
```

### Out of Memory

Reduce batch size or enable gradient checkpointing:
```bash
python train_gpt.py \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing
```

## Shutting Down

### Stop Training
Press `Ctrl+C` or let training complete naturally.

### Stop Ray Cluster

On **each node** (head and workers):
```bash
ray stop
```

## Network Requirements

- **Port 6379**: Ray client connections
- **Port 8265**: Ray dashboard (optional)
- **Port 10001**: Ray client server (if using remote connections)
- **Random high ports**: For Ray internal communication (ensure firewall allows)

For production, configure firewall rules to allow these ports between nodes.
