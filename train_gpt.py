#!/usr/bin/env python3
"""
train_gpt.py – Multi-node Ray Train GPT training for power profiling
=====================================================================

Ray Train conversion 2025‑10‑21. Key features:
* Multi-node distributed training with Ray Train (2+ nodes)
* Asynchronous checkpointing with Ray's built-in checkpoint mechanism
* Overlapping communication/computation with DDP gradient bucketing
* Optimized for power monitoring during large-scale training
* Flash‑Attention 2 and gradient checkpointing support

Usage:
    python train_gpt.py --model_size 8B --batch_size 1 --gradient_accumulation_steps 2 --precision bf16 --max_steps 2000 --num_workers 2
"""

import argparse
import json
import os

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION_IMPORTS", "1")
import random
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

import ray
import ray.train
import torch
import torch.nn.functional as F
from datasets import load_dataset
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# ———————————————————————————— Optional high‑perf back‑ends ————————————————————————————
try:
    from flash_attn import flash_attn_func  # noqa: F401

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash‑attn not available – falling back to SDPA/eager")

# Hard-disable flash-attn on pre-Ampere GPUs to avoid import/config pain
try:
    major_cc, _ = torch.cuda.get_device_capability(0)
    if major_cc < 8:
        FLASH_ATTN_AVAILABLE = False
except Exception:
    pass

try:
    from apex.optimizers import FusedAdam  # type: ignore

    FUSED_ADAM_AVAILABLE = True
except ImportError:
    FUSED_ADAM_AVAILABLE = False

TORCH_FUSED_AVAILABLE = hasattr(torch.optim.AdamW, "fused")

# ———————————————————————————— Configuration helpers ————————————————————————————


@dataclass
class ModelConfig:
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int = 50257
    max_position_embeddings: int = 2048

    @property
    def estimated_params(self) -> int:
        embed = self.vocab_size * self.hidden_size
        layer = (
            4 * self.hidden_size * self.hidden_size  # attn proj
            + 2 * self.hidden_size * self.intermediate_size  # MLP
            + 2 * self.hidden_size  # LayerNorms
        )
        return embed + self.num_hidden_layers * layer


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "125M": ModelConfig(768, 12, 12, 3072),
    "350M": ModelConfig(1024, 16, 24, 4096),
    "1.3B": ModelConfig(2048, 16, 24, 8192),
    "3B": ModelConfig(2560, 20, 32, 10240),
    "7B": ModelConfig(4096, 32, 32, 11008),
    "8B": ModelConfig(4096, 32, 36, 11008),
}

# ———————————————————————————— Argument parsing ————————————————————————————


def parse_args():
    p = argparse.ArgumentParser("Train GPT‑style model with Ray Train multi-node")
    p.add_argument("--model_size", choices=list(MODEL_CONFIGS), default="350M")
    p.add_argument("--base_model", default="facebook/opt-125m")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--sequence_length", type=int, default=2048)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=10_000)

    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)

    p.add_argument("--use_flash_attn", action="store_true", default=False)
    p.add_argument("--use_fused_optimizer", action="store_true", default=False)

    p.add_argument("--tokenizer_num_workers", type=int)
    p.add_argument("--output_dir", default="/datadrive")
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--log_steps", type=int, default=10)

    p.add_argument("--monitor_memory", action="store_true", default=True)
    p.add_argument("--log_timing", action="store_true", default=True)

    p.add_argument(
        "--barrier_every",
        type=int,
        default=0,
        help="Insert dist.barrier() every N optimiser steps (0=off)",
    )

    # Ray-specific arguments
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of Ray workers (processes). 0=auto (all GPUs)",
    )
    p.add_argument(
        "--use_gpu", action="store_true", default=True, help="Use GPU for training"
    )
    p.add_argument(
        "--resources_per_worker",
        type=str,
        default='{"GPU": 1, "CPU": 4}',
        help="Resources per worker as JSON string (default 1 GPU per worker)",
    )
    p.add_argument(
        "--ray_address",
        type=str,
        default="auto",
        help="Ray cluster address (auto for local, or ray://<head_node_ip>:10001)",
    )

    p.add_argument(
        "--dataset",
        choices=["c4", "wikitext", "dummy"],
        default="dummy",
        help="Dataset to use: c4 (streaming), wikitext-2, or dummy random tokens",
    )

    return p.parse_args()


# ———————————————————————————— Memory estimation utils ————————————————————————————


def calc_mem(
    cfg: ModelConfig, bs: int, seqlen: int, prec: str, ckpt: bool
) -> Dict[str, float]:
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2}[prec]
    n_params = cfg.estimated_params
    model = n_params * bytes_per_param / 1024**3
    optim = n_params * 2 * 4 / 1024**3
    act_scale = 0.5 if ckpt else 1.0
    acts = (
        bs
        * seqlen
        * cfg.hidden_size
        * cfg.num_hidden_layers
        * 4
        * act_scale
        * bytes_per_param
        / 1024**3
    )
    grads = model
    tot = model + optim + acts + grads
    return {
        "model": model,
        "optimizer": optim,
        "activations": acts,
        "gradients": grads,
        "total": tot,
        "params": n_params,
    }


def check_fit(mem: Dict[str, float], gpu_mem: float):
    avail = gpu_mem * 0.9
    util = mem["total"] / gpu_mem * 100
    return {"feasible": mem["total"] <= avail, "avail": avail, "util": util}


# ———————————————————————————— Helpers ————————————————————————————


@contextmanager
def timing(name: str, rank: int = 0, enable: bool = True):
    if not enable or rank != 0:
        yield
        return
    t0 = time.time()
    try:
        yield
    finally:
        print(f"[TIMING] {name}: {time.time() - t0:.4f}s")


def log_mem(step: int):
    alloc = torch.cuda.memory_allocated() / 1024**3
    resv = torch.cuda.memory_reserved() / 1024**3
    print(f"[MEM] step {step}: alloc={alloc:.2f} GB resv={resv:.2f} GB")


# ———————————————————————————— Ray Train helpers ————————————————————————————


def get_ray_train_context():
    """Get distributed training context from Ray Train."""
    import ray.train.torch

    # Ray Train automatically sets up the distributed process group
    world_size = ray.train.get_context().get_world_size()
    rank = ray.train.get_context().get_world_rank()
    local_rank = ray.train.get_context().get_local_rank()

    # Set device
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    return world_size, rank, local_rank, device


# ———————————————————————————— Model creation ————————————————————————————


def build_model(cfg: ModelConfig, base: str, use_flash: bool):
    conf = AutoConfig.from_pretrained(base)
    conf.hidden_size = cfg.hidden_size
    conf.num_attention_heads = cfg.num_attention_heads
    conf.num_hidden_layers = cfg.num_hidden_layers
    conf.intermediate_size = cfg.intermediate_size
    conf.max_position_embeddings = cfg.max_position_embeddings

    # Gate Flash-Attn by GPU capability (requires SM80+)
    can_use_flash = use_flash and FLASH_ATTN_AVAILABLE
    try:
        major_cc, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
        can_use_flash = can_use_flash and major_cc >= 8
    except Exception:
        pass

    if can_use_flash:
        # Use private attribute name in recent Transformers
        setattr(conf, "_attn_implementation", "flash_attention_2")
    else:
        setattr(
            conf,
            "_attn_implementation",
            "sdpa" if hasattr(F, "scaled_dot_product_attention") else "eager",
        )

    return AutoModelForCausalLM.from_config(conf)


# ———————————————————————————— Dataset & loader ————————————————————————————


def build_loader(tokenizer, seqlen: int, bs: int, workers: Optional[int], dataset_name: str = "c4"):
    if dataset_name == "dummy":
        class DummyDataset(IterableDataset):
            def __init__(self, vocab_size, seqlen):
                self.vocab_size = vocab_size
                self.seqlen = seqlen

            def __iter__(self):
                while True:
                    ids = torch.randint(
                        low=0,
                        high=self.vocab_size,
                        size=(self.seqlen,),
                        dtype=torch.long,
                    )
                    yield {
                        "input_ids": ids.clone(),
                        "attention_mask": torch.ones_like(ids),
                        "labels": ids.clone(),
                    }

        ds = DummyDataset(tokenizer.vocab_size, seqlen)

        def collate(examples):
            batch = {k: torch.stack([e[k] for e in examples]) for k in ("input_ids", "attention_mask", "labels")}
            return batch

        return DataLoader(ds, batch_size=bs, collate_fn=collate)

    elif dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        def tok(batch):
            out = tokenizer(
                batch["text"],
                truncation=True,
                max_length=seqlen,
                padding="max_length",
                add_special_tokens=True,
            )
            out["labels"] = [ids[:] for ids in out["input_ids"]]
            return out

        ds = ds.map(tok, batched=True, batch_size=1000, remove_columns=["text"])

        def collate(examples):
            keys = ("input_ids", "attention_mask", "labels")
            return {
                k: torch.tensor([e[k] for e in examples], dtype=torch.long) for k in keys
            }

        return DataLoader(
            ds, batch_size=bs, collate_fn=collate, num_workers=0, pin_memory=True
        )

    else:  # "c4"
        ds = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.0000*-of-01024.json.gz",
            split="train",
            streaming=True,
        )

        def tok(batch):
            out = tokenizer(
                batch["text"],
                truncation=True,
                max_length=seqlen,
                padding="max_length",
                add_special_tokens=True,
            )
            out["labels"] = [ids[:] for ids in out["input_ids"]]
            return out

        # For streaming datasets `column_names` can be `None`; just drop the raw text field.
        ds = ds.map(tok, batched=True, batch_size=1000, remove_columns=["text"])

        def collate(examples):
            # keep only token fields; meta fields (e.g. timestamps) were removed above
            keys = ("input_ids", "attention_mask", "labels")
            return {
                k: torch.tensor([e[k] for e in examples], dtype=torch.long) for k in keys
            }

        return DataLoader(
            ds, batch_size=bs, collate_fn=collate, num_workers=0, pin_memory=True
        )


# ———————————————————————————— Optimiser ————————————————————————————


def make_optim(model, lr, wd, use_fused):
    params = [p for p in model.parameters() if p.requires_grad]
    if use_fused and FUSED_ADAM_AVAILABLE:
        print("Using Apex FusedAdam")
        return FusedAdam(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    if use_fused and TORCH_FUSED_AVAILABLE:
        # Only enable fused AdamW on Ampere (SM80) or newer where kernels are supported
        major_cc, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
        if major_cc >= 8:
            print("Using PyTorch fused AdamW")
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd, fused=True)
        else:
            print(
                "PyTorch fused AdamW not supported on this GPU arch; using standard AdamW"
            )
    print("Using standard AdamW")
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


# ———————————————————————————— Train‑step ————————————————————————————


def step_fn(model, batch, optim, sched, scaler, ga_steps, step, dtype, rank, timing_on):
    model.train()
    with timing("fw", rank, timing_on):
        out = model(**batch)
        loss = out.loss / ga_steps

    fp16_mode = dtype == torch.float16 and scaler is not None  # only valid combo

    with timing("bw", rank, timing_on):
        if fp16_mode:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    metrics = {"loss": loss.item() * ga_steps}

    if (step + 1) % ga_steps == 0:
        with timing("optim", rank, timing_on):
            if fp16_mode:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            optim.zero_grad()
            if optim.param_groups and sched is not None:
                sched.step()
        metrics["lr"] = sched.get_last_lr()[0] if sched else 0.0
    return metrics


# ———————————————————————————— Ray Train function ————————————————————————————


def train_func(config: Dict):
    """
    Main training function executed on each Ray worker.
    Ray Train automatically handles distributed setup.
    """
    import torch.distributed as dist

    # Get Ray Train distributed context
    world_size, rank, local_rank, device = get_ray_train_context()

    if rank == 0:
        print(
            f"Ray Train context: world_size={world_size}, rank={rank}, local_rank={local_rank}"
        )

    # Seed for reproducibility
    set_seed(config.get("seed", 42))

    # Model configuration
    cfg = MODEL_CONFIGS[config["model_size"]]

    # Determine precision with hardware capability fallback
    requested_precision = config["precision"]
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    effective_precision = requested_precision
    if requested_precision == "bf16" and not bf16_supported:
        if rank == 0:
            print("[WARN] bf16 not supported on this GPU, using fp16 instead.")
        effective_precision = "fp16"

    # Memory feasibility check using actual device memory
    gpu_total_gb = (
        torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        / 1024**3
    )
    mem = calc_mem(
        cfg,
        config["batch_size"],
        config["sequence_length"],
        effective_precision,
        config["gradient_checkpointing"],
    )
    fit = check_fit(mem, gpu_total_gb)
    if rank == 0:
        print(json.dumps({"memory": mem, "fit": fit}, indent=2))
        if not fit["feasible"]:
            print("[ERROR] Model+batch+sequence do not fit in GPU memory.")
            print("Try: --model_size 125M --batch_size 1 --sequence_length 512")
            import sys
            sys.exit(1)

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        effective_precision
    ]
    amp = effective_precision in {"fp16", "bf16"}

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Build model
    model = build_model(cfg, config["base_model"], config["use_flash_attn"]).to(
        device, dtype=dtype
    )
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            gradient_as_bucket_view=True,  # Optimize for async all-reduce
            broadcast_buffers=False,  # Reduce sync overhead
            bucket_cap_mb=25,  # Bucket size for gradient bucketing
        )
    if config["gradient_checkpointing"]:
        if hasattr(model, "module"):
            model.module.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()

    loader = build_loader(
        tokenizer,
        config["sequence_length"],
        config["batch_size"],
        config.get("tokenizer_num_workers"),
        config.get("dataset", "dummy"),
    )

    optim = make_optim(
        model,
        config["learning_rate"],
        config["weight_decay"],
        config["use_fused_optimizer"],
    )
    sched = get_cosine_schedule_with_warmup(
        optim, config["warmup_steps"], config["max_steps"]
    )
    scaler = torch.cuda.amp.GradScaler() if config["precision"] == "fp16" else None

    step = 0
    data_iter = iter(loader)

    torch.cuda.synchronize()
    t0 = time.time()

    while step < config["max_steps"]:
        try:
            with timing("data", rank, config["log_timing"]):
                batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}

            if amp:
                with torch.cuda.amp.autocast(dtype=dtype):
                    metrics = step_fn(
                        model,
                        batch,
                        optim,
                        sched,
                        scaler,
                        config["gradient_accumulation_steps"],
                        step,
                        dtype,
                        rank,
                        config["log_timing"],
                    )
            else:
                metrics = step_fn(
                    model,
                    batch,
                    optim,
                    sched,
                    None,
                    config["gradient_accumulation_steps"],
                    step,
                    dtype,
                    rank,
                    config["log_timing"],
                )

            if step % config["log_steps"] == 0 and rank == 0:
                print(
                    f"step {step}: loss={metrics['loss']:.4f} lr={metrics.get('lr', 0):.2e}"
                )
                if config["monitor_memory"]:
                    log_mem(step)

            # Optional barrier for power monitoring
            if config["barrier_every"] and (step + 1) % config["barrier_every"] == 0:
                if dist.is_initialized():
                    dist.barrier()

            # Asynchronous checkpoint with Ray Train
            if step > 0 and step % config["save_steps"] == 0:
                # Prepare checkpoint on rank 0
                if rank == 0:
                    checkpoint_dict = {
                        "step": step,
                        "model_state": (
                            model.module if hasattr(model, "module") else model
                        ).state_dict(),
                        "optim_state": optim.state_dict(),
                        "sched_state": sched.state_dict(),
                        "scaler_state": scaler.state_dict() if scaler else None,
                    }
                    # Ray Train handles async checkpointing to storage
                    ray.train.report(
                        metrics={"step": step, "loss": metrics["loss"]},
                        checkpoint=ray.train.Checkpoint.from_dict(checkpoint_dict),
                    )
                else:
                    # Other ranks just report metrics
                    ray.train.report(metrics={"step": step, "loss": metrics["loss"]})

            step += 1

        except StopIteration:
            data_iter = iter(loader)
        except KeyboardInterrupt:
            if rank == 0:
                print("Interrupted by user")
            break

    torch.cuda.synchronize()
    if rank == 0:
        print(f"Done {step} steps in {time.time() - t0:.1f}s")

    # Final checkpoint
    if rank == 0:
        checkpoint_dict = {
            "step": step,
            "model_state": (
                model.module if hasattr(model, "module") else model
            ).state_dict(),
            "optim_state": optim.state_dict(),
            "sched_state": sched.state_dict(),
            "scaler_state": scaler.state_dict() if scaler else None,
        }
        ray.train.report(
            metrics={"step": step, "loss": metrics.get("loss", 0.0), "done": True},
            checkpoint=ray.train.Checkpoint.from_dict(checkpoint_dict),
        )


# ———————————————————————————— Main ————————————————————————————


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # Initialize Ray
    if not ray.is_initialized():
        if args.ray_address == "auto":
            # Start a local Ray runtime if none is running
            ray.init()
        else:
            ray.init(address=args.ray_address)

    print(f"Ray cluster resources: {ray.cluster_resources()}")

    # Parse resources per worker
    resources_per_worker = json.loads(args.resources_per_worker)

    # Auto-select num_workers if requested
    if args.num_workers <= 0:
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus == 0:
            raise RuntimeError(
                "No GPUs found in Ray cluster. Please check your GPU availability."
            )
        gpus_per_worker = int(resources_per_worker.get("GPU", 1)) or 1
        auto_workers = max(1, total_gpus // gpus_per_worker)
        print(
            f"Auto-selecting num_workers={auto_workers} (GPUs={total_gpus}, per_worker={gpus_per_worker})"
        )
        args.num_workers = auto_workers

    # Titan X-friendly override (12 GB, sm_52)
    if args.model_size == "8B":
        print("[WARN] 8B model too large for Titan X (12GB), using 125M instead")
        args.model_size = "125M"
    if args.batch_size > 1:
        print(f"[WARN] Batch size {args.batch_size} may be too large for Titan X, clamping to 1")
        args.batch_size = 1
    if args.sequence_length > 512:
        print(f"[WARN] Sequence length {args.sequence_length} may be too large for Titan X, clamping to 512")
        args.sequence_length = 512

    # Prepare training config
    train_config = {
        "model_size": args.model_size,
        "base_model": args.base_model,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "precision": args.precision,
        "gradient_checkpointing": args.gradient_checkpointing,
        "use_flash_attn": args.use_flash_attn,
        "use_fused_optimizer": args.use_fused_optimizer,
        "tokenizer_num_workers": args.tokenizer_num_workers,
        "save_steps": args.save_steps,
        "log_steps": args.log_steps,
        "monitor_memory": args.monitor_memory,
        "log_timing": args.log_timing,
        "barrier_every": args.barrier_every,
        "dataset": args.dataset,
        "seed": 42,
    }

    # Configure Ray Train scaling
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        resources_per_worker=resources_per_worker,
    )

    # Configure checkpointing
    checkpoint_config = CheckpointConfig(
        num_to_keep=3,  # Keep last 3 checkpoints
        checkpoint_score_attribute="step",
        checkpoint_score_order="max",
    )

    # Configure run
    run_config = RunConfig(
        name="gpt_power_profiling",
        storage_path=args.output_dir,
        checkpoint_config=checkpoint_config,
    )

    # Create Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    # Start training
    print(f"Starting Ray Train with {args.num_workers} workers...")
    result = trainer.fit()

    print("Training complete!")
    print(f"Final checkpoint: {result.checkpoint}")
    print(f"Metrics: {result.metrics}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
