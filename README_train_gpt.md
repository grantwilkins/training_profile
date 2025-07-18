# High-Performance GPT Training Script

A resource-aware, high-performance training script for GPT-style models (1.3B-8B parameters) designed for single-node 8×H100 setups with comprehensive monitoring and power transient capture capabilities.

## Features

- **Memory Accounting**: Automatic calculation and feasibility checking of memory requirements
- **Manual Training Loop**: Fine-grained control without HuggingFace Trainer overhead
- **Multi-Precision Support**: FP32, FP16, and BF16 training with proper AMP implementation
- **Distributed Training**: Efficient DDP with NCCL backend across 8 GPUs
- **Streaming Data**: Memory-efficient C4 dataset loading with streaming
- **Resource Monitoring**: GPU memory usage and training phase timing
- **Power Transient Ready**: Designed for capturing realistic power signatures
- **Multiple Model Sizes**: Pre-configured settings for 1.3B, 3B, 7B, and 8B parameter models
- **⚡ Flash Attention**: Memory-efficient attention computation for improved speed and reduced memory usage
- **⚡ Fused Optimizers**: High-performance fused AdamW variants (Apex FusedAdam or PyTorch fused)
- **⚡ High-Throughput Tokenization**: Optimized batch tokenization with configurable worker processes
- **⚡ Correct LR Scheduling**: Proper learning rate updates only on actual optimizer steps (fixed for gradient accumulation)

## Quick Start

### Basic Usage

```bash
# 7B model with BF16 precision and all optimizations (recommended)
torchrun --nproc_per_node=8 train_gpt.py \
    --model_size 7B \
    --batch_size 4 \
    --precision bf16 \
    --gradient_accumulation_steps 4 \
    --max_steps 1000 \
    --use_flash_attn \
    --use_fused_optimizer \
    --output_dir ./checkpoints/7b_bf16_optimized
```

### Memory Feasibility Check

Before running expensive training, check if your configuration fits in memory:

```bash
python train_gpt.py --model_size 8B --batch_size 2 --precision bf16 --max_steps 1
```

The script will show detailed memory breakdown and feasibility analysis.

## Model Configurations

| Model Size | Parameters | Hidden Size | Layers | Attention Heads | Recommended Batch Size |
|------------|------------|-------------|--------|----------------|----------------------|
| 1.3B       | ~1.3B      | 2048        | 24     | 16             | 16                   |
| 3B         | ~3B        | 2560        | 32     | 20             | 8                    |
| 7B         | ~7B        | 4096        | 32     | 32             | 4                    |
| 8B         | ~8B        | 4096        | 36     | 32             | 2                    |

## Memory Guidelines for H100 80GB

- **1.3B**: `batch_size=16`, no gradient accumulation needed
- **3B**: `batch_size=8`, `gradient_accumulation_steps=2-4`
- **7B**: `batch_size=4`, `gradient_accumulation_steps=4-8`
- **8B**: `batch_size=2`, `gradient_accumulation_steps=8-16`

## Key Arguments

### Model Configuration
- `--model_size`: Choose from 1.3B, 3B, 7B, 8B
- `--base_model`: Base HuggingFace model for architecture (default: facebook/opt-125m)

### Training Settings
- `--batch_size`: Per-GPU batch size
- `--gradient_accumulation_steps`: Steps to accumulate gradients
- `--precision`: fp32, fp16, or bf16 (recommend bf16 for H100)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--max_steps`: Maximum training steps

### Monitoring
- `--monitor_memory`: Enable GPU memory logging
- `--log_timing`: Enable detailed timing of training phases
- `--log_steps`: Frequency of logging (default: 10)

### High-Performance Optimizations
- `--use_flash_attn`: Enable Flash Attention for memory efficiency (default: True)
- `--use_fused_optimizer`: Use fused AdamW variants for speed (default: True)
- `--tokenizer_num_workers`: Parallel tokenization workers (default: auto)
- `--prefetch_factor`: DataLoader prefetch multiplier (default: 4)

## Memory Accounting

The script automatically calculates memory requirements:

```
Memory Requirements:
  Model: 14.50 GB          # Model parameters
  Optimizer: 28.99 GB      # Adam optimizer states
  Activations: 8.19 GB     # Forward pass activations
  Gradients: 14.50 GB      # Gradient storage
  Total: 66.18 GB          # Total memory needed
  Parameters: 7,756,800,000

Feasibility Check:
  Required: 66.18 GB
  Available: 72.00 GB      # 80GB - 10% buffer
  Utilization: 82.7%
  Feasible: ✓
```

## Dataset

Uses the `allenai/c4` dataset with specific shard files:
```python
data_files="en/c4-train.0000*-of-01024.json.gz"
```

The dataset is loaded in streaming mode for memory efficiency and automatically tokenized with proper attention masking for causal language modeling.

## Power Monitoring Integration

The script is designed to work with external power monitoring tools:

1. **Training Phase Timing**: Each forward, backward, and optimizer step is timed
2. **Memory Logging**: Regular GPU memory usage reporting
3. **Step-by-Step Logging**: Detailed per-step metrics

For full power monitoring, integrate with:
- `nvidia-smi dmon` for real-time GPU metrics
- `nsight-systems` for detailed profiling
- Custom power measurement tools

Example power monitoring setup:
```bash
# Terminal 1: Start power monitoring
nvidia-smi dmon -s pucvmet -i 0,1,2,3,4,5,6,7 -f power_log.csv &

# Terminal 2: Run training
torchrun --nproc_per_node=8 train_gpt.py --model_size 7B --batch_size 4 --precision bf16
```

## Output Structure

```
checkpoints/
├── config.json                 # Complete configuration and memory analysis
├── checkpoint-0/
│   ├── pytorch_model.bin       # Model weights
│   ├── config.json             # Model config
│   ├── tokenizer_config.json   # Tokenizer config
│   └── training_state.pt       # Optimizer/scheduler state
├── checkpoint-1000/
└── ...
```

## Advanced Features

### FSDP Support (Experimental)
```bash
torchrun --nproc_per_node=8 train_gpt.py --use_fsdp --model_size 8B
```

### Custom Checkpointing
The script saves complete training state including:
- Model weights
- Optimizer state
- Learning rate scheduler state
- AMP scaler state (if using mixed precision)

### Resume Training
```bash
torchrun --nproc_per_node=8 train_gpt.py \
    --resume_from_checkpoint ./checkpoints/checkpoint-1000 \
    --model_size 7B
```

## Performance Tips

1. **Use BF16**: Better numerical stability than FP16 on H100
2. **Tune Batch Size**: Start with recommended sizes and adjust based on memory
3. **Gradient Checkpointing**: Enabled by default, reduces memory by ~50%
4. **Streaming Dataset**: Prevents memory issues with large datasets
5. **Efficient Data Loading**: 4 workers per GPU with pinned memory

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Increase `--gradient_accumulation_steps` to maintain effective batch size
- Ensure `--gradient_checkpointing` is enabled

### Slow Data Loading
- Check `TOKENIZERS_PARALLELISM=false` is set
- Verify fast storage for dataset cache
- Consider reducing `num_workers` if CPU-bound

### Poor Performance
- Use BF16 on H100 GPUs
- Enable gradient checkpointing
- Check that NCCL is using correct network interfaces

## Performance Optimizations

### Flash Attention
Automatically uses the most efficient attention implementation available:
1. **Flash Attention 2** (if `flash-attn` package installed) - fastest, most memory efficient
2. **PyTorch SDPA** (PyTorch 2.0+) - good fallback with built-in optimizations
3. **Standard attention** - fallback for older PyTorch versions

### Fused Optimizers
Uses the fastest optimizer implementation available:
1. **Apex FusedAdam** (if `apex` installed) - highest performance
2. **PyTorch Fused AdamW** (PyTorch 2.0+) - good built-in option
3. **Standard AdamW** - fallback for compatibility

### Installation for Maximum Performance

```bash
# Essential packages
pip install torch torchvision torchaudio transformers datasets

# For Flash Attention (recommended for H100)
pip install flash-attn --no-build-isolation

# For Apex Fused Optimizers (optional, highest performance)
pip install apex --no-build-isolation

# Alternative: Use PyTorch 2.0+ for built-in fused optimizers and SDPA
```

### Expected Performance Gains
- **Flash Attention**: 15-25% memory reduction, 10-20% speed increase
- **Fused Optimizers**: 5-15% faster optimizer steps
- **High-Throughput Tokenization**: 2-5x faster data preprocessing
- **Corrected LR Scheduling**: Proper convergence with gradient accumulation

## Integration with Existing Workflows

This script is designed to complement the existing `train_llama3_8b.py` but provides:
- More detailed resource management
- Better power monitoring hooks
- Manual training loop control
- Multiple model size support
- High-performance optimizations for maximum throughput

Both scripts can coexist and serve different purposes in your training pipeline. 