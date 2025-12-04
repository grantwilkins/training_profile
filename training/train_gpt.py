#!/usr/bin/env python3
"""
train_gpt.py – High‑performance, resource‑aware GPT training on 8×H100
=====================================================================

Patched 2025‑07‑26.  Key fixes:
* Robust streaming‑dataset dataloader with explicit `collate_fn` and `num_workers=0`.
* Correct device placement & Flash‑Attention 2 flag.
* DDP wrapped **before** gradient‑checkpointing so hooks are registered.
* Optional collective barrier after each optimiser step for clean power spikes.
* Safer LR‑scheduler stepping & checkpoint synchronisation.
* Misc. hygiene (seed, AMP guards, skip checkpoint‑0).

Usage (unchanged):
    torchrun --standalone --nproc_per_node 8 train_gpt.py --model_size 8B --batch_size 1 --gradient_accumulation_steps 2 --precision bf16 --max_steps 2000
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
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
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
    "1.3B": ModelConfig(2048, 16, 24, 8192),
    "3B": ModelConfig(2560, 20, 32, 10240),
    "7B": ModelConfig(4096, 32, 32, 11008),
    "8B": ModelConfig(4096, 32, 36, 11008),
}

# ———————————————————————————— Argument parsing ————————————————————————————


def parse_args():
    p = argparse.ArgumentParser("Train GPT‑style model on 8×H100")
    p.add_argument("--model_size", choices=list(MODEL_CONFIGS), default="7B")
    p.add_argument("--base_model", default="facebook/opt-125m")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--sequence_length", type=int, default=2048)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=10_000)

    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)

    p.add_argument("--use_flash_attn", action="store_true", default=True)
    p.add_argument("--use_fused_optimizer", action="store_true", default=True)

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

    p.add_argument("--local_rank", type=int, default=0)
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


def check_fit(mem: Dict[str, float], gpu_mem: float = 80.0):
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


# ———————————————————————————— Distributed setup ————————————————————————————


def setup_dist():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_world_size(), rank, local_rank
    else:
        return 1, 0, 0


# ———————————————————————————— Model creation ————————————————————————————


def build_model(cfg: ModelConfig, base: str, use_flash: bool):
    conf = AutoConfig.from_pretrained(base)
    conf.hidden_size = cfg.hidden_size
    conf.num_attention_heads = cfg.num_attention_heads
    conf.num_hidden_layers = cfg.num_hidden_layers
    conf.intermediate_size = cfg.intermediate_size
    conf.max_position_embeddings = cfg.max_position_embeddings

    if use_flash and FLASH_ATTN_AVAILABLE:
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


def build_loader(tokenizer, seqlen: int, bs: int, workers: Optional[int]):
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


# ———————————————————————————— Optimiz`er ————————————————————————————


def make_optim(model, lr, wd, use_fused):
    params = [p for p in model.parameters() if p.requires_grad]
    if use_fused and FUSED_ADAM_AVAILABLE:
        print("Using Apex FusedAdam")
        return FusedAdam(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    if use_fused and TORCH_FUSED_AVAILABLE:
        print("Using PyTorch fused AdamW")
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, fused=True)
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


# ———————————————————————————— Main ————————————————————————————


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed()

    ws, rank, local_rank = setup_dist()
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"World size: {ws}; device: {device}")

    cfg = MODEL_CONFIGS[args.model_size]
    mem = calc_mem(
        cfg,
        args.batch_size,
        args.sequence_length,
        args.precision,
        args.gradient_checkpointing,
    )
    fit = check_fit(mem)
    if rank == 0:
        print(json.dumps({"memory": mem, "fit": fit}, indent=2))
        if not fit["feasible"]:
            warnings.warn(
                "Model may not fit – consider lowering batch size or enabling checkpointing."
            )

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.precision
    ]
    amp = args.precision in {"fp16", "bf16"}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = build_model(cfg, args.base_model, args.use_flash_attn).to(
        device, dtype=dtype
    )

    if ws > 1:
        model = DDP(model, device_ids=[local_rank])
    if args.gradient_checkpointing:
        model.module.gradient_checkpointing_enable() if hasattr(
            model, "module"
        ) else model.gradient_checkpointing_enable()

    loader = build_loader(
        tokenizer, args.sequence_length, args.batch_size, args.tokenizer_num_workers
    )
    optim = make_optim(
        model, args.learning_rate, args.weight_decay, args.use_fused_optimizer
    )
    sched = get_cosine_schedule_with_warmup(optim, args.warmup_steps, args.max_steps)
    # GradScaler only supports fp16; for bf16 we disable scaling
    scaler = torch.cuda.amp.GradScaler() if args.precision == "fp16" else None

    os.makedirs(args.output_dir, exist_ok=True)

    step = 0
    data_iter = iter(loader)

    torch.cuda.synchronize()
    t0 = time.time()

    while step < args.max_steps:
        try:
            with timing("data", rank, args.log_timing):
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
                        args.gradient_accumulation_steps,
                        step,
                        dtype,
                        rank,
                        args.log_timing,
                    )
            else:
                metrics = step_fn(
                    model,
                    batch,
                    optim,
                    sched,
                    None,
                    args.gradient_accumulation_steps,
                    step,
                    dtype,
                    rank,
                    args.log_timing,
                )

            if step % args.log_steps == 0 and rank == 0:
                print(
                    f"step {step}: loss={metrics['loss']:.4f} lr={metrics.get('lr', 0):.2e}"
                )
                if args.monitor_memory:
                    log_mem(step)

            # barrier for clean power plateau
            if (
                args.barrier_every
                and (step + 1) % args.barrier_every == 0
                and dist.is_initialized()
            ):
                dist.barrier()

            # checkpoint (skip step 0)
            if step and step % args.save_steps == 0 and rank == 0:
                torch.cuda.synchronize()
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                (model.module if hasattr(model, "module") else model).save_pretrained(
                    ckpt_dir
                )
                tokenizer.save_pretrained(ckpt_dir)
                torch.save(
                    {
                        "step": step,
                        "optim": optim.state_dict(),
                        "sched": sched.state_dict(),
                        "scaler": scaler.state_dict() if scaler else None,
                    },
                    os.path.join(ckpt_dir, "state.pt"),
                )
                torch.cuda.synchronize()

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
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
