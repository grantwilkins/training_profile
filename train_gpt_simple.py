#!/usr/bin/env python3
"""
train_gpt_simple.py – Single-node multi-GPU training (no Ray)
==============================================================

For single-node training with 1-2 GPUs on a Titan X box.
Uses native PyTorch DDP via torchrun.

Usage (single GPU):
    python train_gpt_simple.py --model_size 125M --batch_size 1 --max_steps 100

Usage (2 GPUs):
    torchrun --nproc_per_node=2 train_gpt_simple.py --model_size 125M --batch_size 1
"""

import argparse
import json
import os

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION_IMPORTS", "1")
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
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

# Hard-disable flash-attn on pre-Ampere GPUs
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

# ———————————————————————————— Configuration ————————————————————————————


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
            4 * self.hidden_size * self.hidden_size
            + 2 * self.hidden_size * self.intermediate_size
            + 2 * self.hidden_size
        )
        return embed + self.num_hidden_layers * layer


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "125M": ModelConfig(768, 12, 12, 3072),
    "350M": ModelConfig(1024, 16, 24, 4096),
    "1.3B": ModelConfig(2048, 16, 24, 8192),
}


def parse_args():
    p = argparse.ArgumentParser("Train GPT on single node (1-2 GPUs)")
    p.add_argument("--model_size", choices=list(MODEL_CONFIGS), default="125M")
    p.add_argument("--base_model", default="facebook/opt-125m")

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--sequence_length", type=int, default=512)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=1000)

    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)

    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--log_steps", type=int, default=10)

    p.add_argument(
        "--dataset",
        choices=["c4", "wikitext", "dummy"],
        default="dummy",
        help="Dataset to use",
    )
    p.add_argument(
        "--c4_split",
        type=str,
        default="train[:0.01%]",
        help="C4 split (e.g. 'train[:10000]')",
    )

    return p.parse_args()


# ———————————————————————————— Memory utils ————————————————————————————


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
    print(f"[MEM] step {step}: alloc={alloc:.2f} GB resv={resv:.2f} GB")


# ———————————————————————————— Model ————————————————————————————


def build_model(cfg: ModelConfig, base: str):
    conf = AutoConfig.from_pretrained(base)
    conf.hidden_size = cfg.hidden_size
    conf.num_attention_heads = cfg.num_attention_heads
    conf.num_hidden_layers = cfg.num_hidden_layers
    conf.intermediate_size = cfg.intermediate_size
    conf.max_position_embeddings = cfg.max_position_embeddings

    # Use SDPA on Titan X (no flash-attn)
    setattr(
        conf,
        "_attn_implementation",
        "sdpa" if hasattr(F, "scaled_dot_product_attention") else "eager",
    )

    return AutoModelForCausalLM.from_config(conf)


# ———————————————————————————— Dataset ————————————————————————————


def build_loader(
    tokenizer,
    seqlen: int,
    bs: int,
    dataset_name: str = "dummy",
    c4_split: str = "train[:0.01%]",
):
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
            return {
                k: torch.stack([e[k] for e in examples])
                for k in ("input_ids", "attention_mask", "labels")
            }

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
                k: torch.tensor([e[k] for e in examples], dtype=torch.long)
                for k in keys
            }

        return DataLoader(ds, batch_size=bs, collate_fn=collate, num_workers=0)

    else:  # "c4"
        ds = load_dataset("allenai/c4", "en", split=c4_split)

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
                k: torch.tensor([e[k] for e in examples], dtype=torch.long)
                for k in keys
            }

        return DataLoader(ds, batch_size=bs, collate_fn=collate, num_workers=0)


# ———————————————————————————— Training ————————————————————————————


def setup_distributed():
    """Setup DDP if launched with torchrun, else single GPU."""
    if "RANK" in os.environ:
        # torchrun sets these
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")

        return rank, local_rank, world_size
    else:
        # Single GPU
        return 0, 0, 1


def cleanup_distributed():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"Training on {world_size} GPU(s)")

    set_seed(42)

    cfg = MODEL_CONFIGS[args.model_size]

    # Check precision support
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    effective_precision = args.precision
    if args.precision == "bf16" and not bf16_supported:
        if rank == 0:
            print("[WARN] bf16 not supported, using fp16 instead")
        effective_precision = "fp16"

    # Memory check
    gpu_total_gb = (
        torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
    )
    mem = calc_mem(
        cfg, args.batch_size, args.sequence_length, effective_precision, args.gradient_checkpointing
    )
    fit = check_fit(mem, gpu_total_gb)

    if rank == 0:
        print(json.dumps({"memory": mem, "fit": fit}, indent=2))
        if not fit["feasible"]:
            print("[ERROR] Model doesn't fit in GPU memory")
            print("Try: --model_size 125M --batch_size 1 --sequence_length 256")
            cleanup_distributed()
            return

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        effective_precision
    ]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Model
    model = build_model(cfg, args.base_model).to(device, dtype=dtype)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    if args.gradient_checkpointing:
        if hasattr(model, "module"):
            model.module.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()

    # Data
    loader = build_loader(
        tokenizer, args.sequence_length, args.batch_size, args.dataset, args.c4_split
    )

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    sched = get_cosine_schedule_with_warmup(optim, args.warmup_steps, args.max_steps)

    scaler = (
        torch.cuda.amp.GradScaler() if effective_precision == "fp16" else None
    )

    # Training loop
    step = 0
    data_iter = iter(loader)
    model.train()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    torch.cuda.synchronize()
    t0 = time.time()

    while step < args.max_steps:
        try:
            batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward
            if effective_precision in ["fp16", "bf16"]:
                with torch.cuda.amp.autocast(dtype=dtype):
                    out = model(**batch)
                    loss = out.loss / args.gradient_accumulation_steps
            else:
                out = model(**batch)
                loss = out.loss / args.gradient_accumulation_steps

            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()

                optim.zero_grad()
                sched.step()

            # Logging
            if step % args.log_steps == 0 and rank == 0:
                print(
                    f"step {step}: loss={loss.item() * args.gradient_accumulation_steps:.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e}"
                )
                log_mem(step)

            # Checkpointing
            if step > 0 and step % args.save_steps == 0 and rank == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
                torch.save(
                    {
                        "step": step,
                        "model": (
                            model.module if hasattr(model, "module") else model
                        ).state_dict(),
                        "optim": optim.state_dict(),
                        "sched": sched.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1

        except StopIteration:
            data_iter = iter(loader)
        except KeyboardInterrupt:
            if rank == 0:
                print("Interrupted")
            break

    torch.cuda.synchronize()
    if rank == 0:
        print(f"Done {step} steps in {time.time() - t0:.1f}s")

    cleanup_distributed()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
