#!/usr/bin/env python3
"""
train_gpt.py – High-performance, resource-aware GPT training on 8xH100
=====================================================================

This script provides memory-efficient training of GPT-style models (1.3B-8B parameters)
with comprehensive resource monitoring and power transient capture capabilities.

Features:
- Memory accounting and feasibility checks
- Manual training loop (no Trainer) for fine-grained control
- DDP with gradient checkpointing and bf16/fp16 support
- Streaming C4 dataset with efficient tokenization
- GPU memory and power monitoring hooks
- Support for multiple model sizes with CLI configuration

Usage:
    torchrun --nproc_per_node=8 train_gpt.py --model_size 7B --batch_size 8 --precision bf16
"""

import argparse
import json
import os
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# High-performance imports
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash-attn not available, using standard attention")

try:
    from apex.optimizers import FusedAdam

    FUSED_ADAM_AVAILABLE = True
except ImportError:
    FUSED_ADAM_AVAILABLE = False
    try:
        # PyTorch 2.0+ fused AdamW
        from torch.optim import AdamW

        TORCH_FUSED_AVAILABLE = hasattr(torch.optim.AdamW, "fused")
    except:
        TORCH_FUSED_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuration for different model sizes"""

    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int = 50257
    max_position_embeddings: int = 2048

    @property
    def estimated_params(self) -> int:
        """Estimate total parameter count"""
        # Embedding: vocab_size * hidden_size
        embed_params = self.vocab_size * self.hidden_size

        # Each transformer layer:
        # - Attention: 4 * hidden_size^2 (Q, K, V, O projections)
        # - MLP: 2 * hidden_size * intermediate_size
        # - LayerNorm: 2 * hidden_size (attn + mlp)
        layer_params = (
            4 * self.hidden_size * self.hidden_size  # Attention
            + 2 * self.hidden_size * self.intermediate_size  # MLP
            + 2 * self.hidden_size  # LayerNorms
        )

        total_params = embed_params + self.num_hidden_layers * layer_params
        return total_params


# Model size configurations
MODEL_CONFIGS = {
    "1.3B": ModelConfig(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=24,
        intermediate_size=8192,
    ),
    "3B": ModelConfig(
        hidden_size=2560,
        num_attention_heads=20,
        num_hidden_layers=32,
        intermediate_size=10240,
    ),
    "7B": ModelConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        intermediate_size=11008,
    ),
    "8B": ModelConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=36,
        intermediate_size=11008,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-style model on 8xH100")

    # Model configuration
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["1.3B", "3B", "7B", "8B"],
        default="7B",
        help="Model size configuration",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="facebook/opt-125m",  # Use as base architecture
        help="Base model to use for architecture",
    )

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=10000)

    # Precision and optimization
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Training precision",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Use FSDP instead of DDP"
    )

    # High-performance optimizations
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        default=True,
        help="Use Flash Attention for improved memory and speed",
    )
    parser.add_argument(
        "--use_fused_optimizer",
        action="store_true",
        default=True,
        help="Use fused AdamW optimizer for better performance",
    )
    parser.add_argument(
        "--tokenizer_num_workers",
        type=int,
        default=None,
        help="Number of workers for tokenization (default: min(8, cpu_count))",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Prefetch factor for data loading",
    )

    # Data and I/O
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)

    # Monitoring
    parser.add_argument("--monitor_memory", action="store_true", default=True)
    parser.add_argument("--monitor_power", action="store_true", default=True)
    parser.add_argument("--log_timing", action="store_true", default=True)

    # Distributed training args
    parser.add_argument("--local_rank", type=int, default=0)

    return parser.parse_args()


def calculate_memory_requirements(
    config: ModelConfig,
    batch_size: int,
    sequence_length: int,
    precision: str,
    gradient_checkpointing: bool = True,
) -> Dict[str, float]:
    """Calculate memory requirements in GB"""

    # Bytes per parameter based on precision
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2}[precision]

    # Model parameters
    num_params = config.estimated_params
    model_memory = num_params * bytes_per_param / (1024**3)  # GB

    # Optimizer states (Adam: 2x model params for momentum + variance)
    optimizer_memory = num_params * 2 * 4 / (1024**3)  # Always fp32

    # Activations (approximate)
    # For transformer: roughly batch_size * seq_len * hidden_size * num_layers * 4
    activation_scaling = 0.5 if gradient_checkpointing else 1.0
    activation_memory = (
        batch_size
        * sequence_length
        * config.hidden_size
        * config.num_hidden_layers
        * 4
        * activation_scaling
        * bytes_per_param
        / (1024**3)
    )

    # Gradients (same size as model)
    gradient_memory = model_memory

    total_memory = model_memory + optimizer_memory + activation_memory + gradient_memory

    return {
        "model": model_memory,
        "optimizer": optimizer_memory,
        "activations": activation_memory,
        "gradients": gradient_memory,
        "total": total_memory,
        "estimated_params": num_params,
    }


def check_memory_feasibility(memory_req: Dict[str, float], gpu_memory_gb: float = 80):
    """Check if training is feasible on given GPU memory"""
    total_required = memory_req["total"]
    buffer = gpu_memory_gb * 0.1  # 10% buffer for overhead
    available = gpu_memory_gb - buffer

    feasible = total_required <= available
    utilization = total_required / gpu_memory_gb * 100

    return {
        "feasible": feasible,
        "required_gb": total_required,
        "available_gb": available,
        "utilization_pct": utilization,
        "buffer_gb": buffer,
    }


@contextmanager
def timing_context(name: str, rank: int = 0, log_timing: bool = True):
    """Context manager for timing operations"""
    if not log_timing or rank != 0:
        yield
        return

    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"[TIMING] {name}: {elapsed:.4f}s")


def log_memory_usage(step: int, rank: int = 0):
    """Log current GPU memory usage"""
    if rank != 0:
        return

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(
            f"[MEMORY] Step {step}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB"
        )


def setup_distributed():
    """Initialize distributed training"""
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        torch.cuda.set_device(local_rank)
        return world_size, rank, local_rank
    else:
        return 1, 0, 0


def create_model(
    config: ModelConfig, base_model_name: str, use_flash_attn: bool = True
) -> torch.nn.Module:
    """Create model with specified configuration and optimizations"""
    # Load base config and modify it
    base_config = AutoConfig.from_pretrained(base_model_name)

    # Update with our configuration
    base_config.hidden_size = config.hidden_size
    base_config.num_attention_heads = config.num_attention_heads
    base_config.num_hidden_layers = config.num_hidden_layers
    base_config.intermediate_size = config.intermediate_size
    base_config.max_position_embeddings = config.max_position_embeddings

    # Enable flash attention if available and requested
    if use_flash_attn:
        if FLASH_ATTN_AVAILABLE:
            # Set flash attention config flags
            if hasattr(base_config, "_flash_attn_2_enabled"):
                base_config._flash_attn_2_enabled = True
            elif hasattr(base_config, "use_flash_attention_2"):
                base_config.use_flash_attention_2 = True
        elif hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Fall back to PyTorch 2.0+ SDPA
            if hasattr(base_config, "use_sdpa"):
                base_config.use_sdpa = True

    # Create model from modified config
    model = AutoModelForCausalLM.from_config(base_config)

    return model


def create_dataset(
    tokenizer, sequence_length: int, rank: int = 0, num_workers: Optional[int] = None
):
    """Create streaming C4 dataset with high-throughput tokenization"""
    if rank == 0:
        print("Loading C4 dataset with optimized tokenization...")

    # Determine number of workers for tokenization
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)

    # Load specific shard files as requested
    dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.0000*-of-01024.json.gz",
        split="train",
        streaming=True,  # Enable streaming for memory efficiency
    )

    def high_throughput_tokenize(examples):
        """Optimized tokenization function for high throughput"""
        texts = examples["text"]

        # Batch tokenization with optimizations
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=sequence_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors=None,  # Keep as lists for now
            add_special_tokens=True,
            # Optimization: disable return_offsets_mapping and other unused features
            return_offsets_mapping=False,
            return_length=False,
        )

        # For causal LM, labels are the same as input_ids
        # Create a copy to avoid reference issues
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]

        return tokenized

    # Apply tokenization with high throughput settings
    tokenized_dataset = dataset.map(
        high_throughput_tokenize,
        batched=True,
        batch_size=1000,  # Larger batch size for better throughput
        remove_columns=dataset.column_names,
        num_proc=(
            num_workers if not dataset._is_streaming else None
        ),  # Only for non-streaming
    )

    return tokenized_dataset


def train_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    gradient_accumulation_steps: int,
    step: int,
    rank: int,
    log_timing: bool = True,
) -> Dict[str, float]:
    """Execute one training step with timing and correct LR scheduling"""

    model.train()

    with timing_context("forward_pass", rank, log_timing):
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps

    with timing_context("backward_pass", rank, log_timing):
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    metrics = {"loss": loss.item() * gradient_accumulation_steps}

    # Check if this is an optimizer step (corrected LR scheduling)
    is_optimizer_step = (step + 1) % gradient_accumulation_steps == 0

    # Update weights every gradient_accumulation_steps
    if is_optimizer_step:
        with timing_context("optimizer_step", rank, log_timing):
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # CORRECTED: Only update scheduler when we actually take an optimizer step
            scheduler.step()
            optimizer.zero_grad()

        metrics["learning_rate"] = scheduler.get_last_lr()[0]
        metrics["optimizer_step"] = True
    else:
        metrics["optimizer_step"] = False

    return metrics


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    use_fused: bool = True,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Create optimized optimizer with fused variants when available"""

    # Filter parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]

    if use_fused:
        # Try different fused optimizer options in order of preference
        if FUSED_ADAM_AVAILABLE:
            # Apex FusedAdam (most optimized)
            optimizer = FusedAdam(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
            )
            print("Using Apex FusedAdam optimizer")
            return optimizer
        elif TORCH_FUSED_AVAILABLE:
            # PyTorch 2.0+ fused AdamW
            optimizer = torch.optim.AdamW(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                fused=True,
            )
            print("Using PyTorch fused AdamW optimizer")
            return optimizer
        else:
            print(
                "Warning: Fused optimizers not available, falling back to standard AdamW"
            )

    # Fall back to standard AdamW
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    print("Using standard AdamW optimizer")
    return optimizer


def main():
    args = parse_args()

    # Setup distributed training
    world_size, rank, local_rank = setup_distributed()

    if rank == 0:
        print(f"Training GPT-style model: {args.model_size}")
        print(f"World size: {world_size}, Local rank: {local_rank}")

        # Print optimization status
        print(f"\nOptimization Status:")
        print(
            f"  Flash Attention: {'✓' if args.use_flash_attn and FLASH_ATTN_AVAILABLE else '✗'}"
        )
        print(
            f"  Fused Optimizer: {'✓' if args.use_fused_optimizer and (FUSED_ADAM_AVAILABLE or TORCH_FUSED_AVAILABLE) else '✗'}"
        )
        print(
            f"  Gradient Checkpointing: {'✓' if args.gradient_checkpointing else '✗'}"
        )
        print(f"  Mixed Precision: {args.precision.upper()}")
        print(f"  Tokenizer Workers: {args.tokenizer_num_workers or 'auto'}")
        print(f"  DataLoader Prefetch: {args.prefetch_factor}")

    # Get model configuration
    model_config = MODEL_CONFIGS[args.model_size]

    # Calculate memory requirements
    memory_req = calculate_memory_requirements(
        model_config,
        args.batch_size,
        args.sequence_length,
        args.precision,
        args.gradient_checkpointing,
    )

    # Check feasibility
    feasibility = check_memory_feasibility(memory_req)

    if rank == 0:
        print(f"\nMemory Requirements:")
        print(f"  Model: {memory_req['model']:.2f} GB")
        print(f"  Optimizer: {memory_req['optimizer']:.2f} GB")
        print(f"  Activations: {memory_req['activations']:.2f} GB")
        print(f"  Gradients: {memory_req['gradients']:.2f} GB")
        print(f"  Total: {memory_req['total']:.2f} GB")
        print(f"  Parameters: {memory_req['estimated_params']:,}")

        print(f"\nFeasibility Check:")
        print(f"  Required: {feasibility['required_gb']:.2f} GB")
        print(f"  Available: {feasibility['available_gb']:.2f} GB")
        print(f"  Utilization: {feasibility['utilization_pct']:.1f}%")
        print(f"  Feasible: {'✓' if feasibility['feasible'] else '✗'}")

        if not feasibility["feasible"]:
            warnings.warn(
                f"Training may not fit in GPU memory. "
                f"Consider reducing batch size or using gradient checkpointing."
            )

    # Setup precision
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.precision
    ]
    use_amp = args.precision in ["fp16", "bf16"]

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model with optimizations
    model = create_model(model_config, args.base_model, args.use_flash_attn)
    model = model.to(local_rank, dtype=dtype)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Setup DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Create dataset and dataloader with high-throughput tokenization
    dataset = create_dataset(
        tokenizer, args.sequence_length, rank, args.tokenizer_num_workers
    )

    # For streaming datasets, we can't use DistributedSampler
    # Instead, we'll skip examples based on rank
    def rank_filter(example, idx):
        return idx % world_size == rank

    if world_size > 1:
        dataset = dataset.filter(rank_filter, with_indices=True)

    # Optimized DataLoader configuration
    num_dataloader_workers = min(4, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_dataloader_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if num_dataloader_workers > 0 else False,
    )

    # Setup optimizer with fused variants
    optimizer = create_optimizer(
        model,
        args.learning_rate,
        args.weight_decay,
        use_fused=args.use_fused_optimizer,
        betas=(0.9, 0.95),
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # Setup AMP scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save configuration
        config_dict = {
            "model_config": model_config.__dict__,
            "training_args": vars(args),
            "memory_requirements": memory_req,
            "feasibility": feasibility,
        }

        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

    # Training loop
    if rank == 0:
        print(f"\nStarting training...")
        print(f"Max steps: {args.max_steps}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        print(
            f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}"
        )

    step = 0
    data_iter = iter(dataloader)

    while step < args.max_steps:
        try:
            # Get next batch
            with timing_context("data_loading", rank, args.log_timing):
                batch = next(data_iter)

            # Move batch to device
            batch = {
                k: v.to(local_rank) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Execute training step
            if use_amp:
                with torch.cuda.amp.autocast(dtype=dtype):
                    metrics = train_step(
                        model,
                        batch,
                        optimizer,
                        scheduler,
                        scaler,
                        args.gradient_accumulation_steps,
                        step,
                        rank,
                        args.log_timing,
                    )
            else:
                metrics = train_step(
                    model,
                    batch,
                    optimizer,
                    scheduler,
                    None,
                    args.gradient_accumulation_steps,
                    step,
                    rank,
                    args.log_timing,
                )

            # Logging
            if step % args.log_steps == 0 and rank == 0:
                opt_step_info = " [OPT]" if metrics.get("optimizer_step", False) else ""
                lr_info = (
                    f", LR={metrics.get('learning_rate', 0.0):.2e}"
                    if "learning_rate" in metrics
                    else ""
                )

                print(
                    f"Step {step}{opt_step_info}: Loss={metrics.get('loss', 0.0):.4f}{lr_info}"
                )

                if args.monitor_memory:
                    log_memory_usage(step, rank)

            # Save checkpoint
            if step % args.save_steps == 0 and rank == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                os.makedirs(checkpoint_dir, exist_ok=True)

                if hasattr(model, "module"):
                    model.module.save_pretrained(checkpoint_dir)
                else:
                    model.save_pretrained(checkpoint_dir)

                tokenizer.save_pretrained(checkpoint_dir)

                # Save optimizer state
                torch.save(
                    {
                        "step": step,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if scaler else None,
                    },
                    os.path.join(checkpoint_dir, "training_state.pt"),
                )

                print(f"Saved checkpoint at step {step}")

            step += 1

        except StopIteration:
            if rank == 0:
                print("Dataset exhausted, creating new iterator...")
            data_iter = iter(dataloader)
        except KeyboardInterrupt:
            if rank == 0:
                print("Training interrupted by user")
            break
        except Exception as e:
            if rank == 0:
                print(f"Error during training: {e}")
            raise

    # Final cleanup
    if rank == 0:
        print(f"Training completed after {step} steps")

    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
