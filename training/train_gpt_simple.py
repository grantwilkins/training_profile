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
import math
import os

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION_IMPORTS", "1")
import random
import threading
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

    # Power smoothing / synthetic burn
    p.add_argument(
        "--smooth_power",
        action="store_true",
        help="Enable synthetic GPU burn to smooth power ramps and raise dips.",
    )
    p.add_argument(
        "--warmup_total_s",
        type=float,
        default=30.0,
        help="Total duration (seconds) of pre-training startup warmup burn.",
    )
    p.add_argument(
        "--warmup_segments",
        type=int,
        default=60,
        help="Number of segments for warmup ramp shaping (more = smoother).",
    )
    p.add_argument(
        "--cooldown_total_s",
        type=float,
        default=30.0,
        help="Total duration (seconds) of post-training cooldown burn.",
    )
    p.add_argument(
        "--cooldown_segments",
        type=int,
        default=60,
        help="Number of segments for cooldown ramp shaping (more = smoother).",
    )
    p.add_argument(
        "--enable_ckpt_burn",
        action="store_true",
        help="If set, burn GPU during checkpoint I/O/barriers to avoid deep dips.",
    )

    return p.parse_args()


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


class GpuBurner:
    """
    Synthetic GPU load generator.

    - Uses a large matmul working set sized by available GPU memory.
    - Automatically calculates buffer size: 50% of free memory minus 500MB safety margin.
    - Provides blocking burn_for() for warmup/cooldown.
    - Provides a background burn thread for overlapping with checkpoint I/O.
    - Tracks loop counts and durations for all burn operations.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Statistics tracking
        self.warmup_segments: list = []  # [(requested_s, actual_s, loops)]
        self.checkpoint_burns: list = []  # [(requested_s, actual_s, loops)]
        self.cooldown_segments: list = []  # [(requested_s, actual_s, loops)]

        self._init_buffers()

    def _init_buffers(self):
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)

        # Use 20% of free memory minus 1GB safety margin (very conservative)
        safety_margin_bytes = 1024 * 1024 * 1024  # 1 GB
        target_bytes = int(free_bytes * 0.2 - safety_margin_bytes)

        if target_bytes <= 0:
            # Not enough free memory, use minimal buffer
            n = 128
            print(f"[WARN] Very low free memory ({free_bytes / 1024**3:.2f} GB), using minimal burn buffer")
        else:
            # Approximate: 2 * n^2 * bytes_per_elem live in a matmul
            bytes_per_elem = torch.finfo(self.dtype).bits // 8
            n = int((target_bytes / (2 * bytes_per_elem)) ** 0.5)

            # Round to a reasonable multiple to avoid weird tile sizes
            n = max(128, (n // 128) * 128)

        self.n = n
        buffer_size_gb = 2 * n * n * torch.finfo(self.dtype).bits // 8 / 1024**3

        try:
            self.a = torch.randn(n, n, device=self.device, dtype=self.dtype)
            self.b = torch.randn(n, n, device=self.device, dtype=self.dtype)
            print(f"[BURN] Allocated {buffer_size_gb:.2f} GB burn buffer (n={n}, free={free_bytes/1024**3:.2f} GB)")
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Fallback to even smaller buffer
                torch.cuda.empty_cache()
                n = 128
                self.n = n
                self.a = torch.randn(n, n, device=self.device, dtype=self.dtype)
                self.b = torch.randn(n, n, device=self.device, dtype=self.dtype)
                print(f"[WARN] OOM during buffer allocation, using minimal buffer (n={n})")
            else:
                raise

    # ---------- blocking burn for a fixed duration ----------

    def burn_for(self, duration_s: float, phase: str = "unknown"):
        """
        Blocking burn on the default stream for ~duration_s seconds.

        Args:
            duration_s: Target duration in seconds
            phase: One of 'warmup', 'checkpoint', 'cooldown' for tracking
        """
        if duration_s <= 0:
            return

        t0 = time.time()
        loops = 0
        a, b = self.a, self.b
        while True:
            c = a @ b
            a = c
            torch.cuda.synchronize(self.device)
            loops += 1
            if time.time() - t0 >= duration_s:
                break
        self.a = a
        actual_s = time.time() - t0

        # Record stats
        if phase == "warmup":
            self.warmup_segments.append((duration_s, actual_s, loops))
        elif phase == "checkpoint":
            self.checkpoint_burns.append((duration_s, actual_s, loops))
        elif phase == "cooldown":
            self.cooldown_segments.append((duration_s, actual_s, loops))

    # ---------- background thread burn for checkpoints ----------

    def _burn_loop(self):
        """Internal loop: keep GPU busy until stop_event is set."""
        a, b = self.a, self.b
        loops = 0
        while not self._stop_event.is_set():
            c = a @ b
            a = c
            # Chunk work so we can stop promptly
            torch.cuda.synchronize(self.device)
            loops += 1
        self.a = a
        return loops

    def start_burn_thread(self):
        """Start a background burn thread if not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._burn_start_time = time.time()
        self._thread = threading.Thread(target=self._burn_loop)
        self._thread.daemon = True
        self._thread.start()

    def stop_burn_thread(self, join: bool = True):
        """Signal the background burn thread to stop and optionally join."""
        if self._thread is None:
            return
        self._stop_event.set()
        if join:
            actual_s = time.time() - self._burn_start_time
            self._thread.join()
            # Background burns don't have a requested duration, just actual
            self.checkpoint_burns.append((None, actual_s, None))
        self._thread = None

    def print_summary(self, rank: int = 0):
        """Print burn statistics summary (rank 0 only)."""
        if rank != 0:
            return

        print("\n" + "=" * 60)
        print("GPU BURNER SUMMARY")
        print("=" * 60)

        # Warmup
        if self.warmup_segments:
            total_req = sum(r for r, _, _ in self.warmup_segments if r is not None)
            total_act = sum(a for _, a, _ in self.warmup_segments if a is not None)
            total_loops = sum(l for _, _, l in self.warmup_segments if l is not None)
            print(f"\nWarmup ({len(self.warmup_segments)} segments):")
            print(f"  Total requested: {total_req:.2f}s")
            print(f"  Total actual:    {total_act:.2f}s")
            print(
                f"  Overshoot:       {total_act - total_req:.3f}s ({((total_act / total_req - 1) * 100):.2f}%)"
            )
            print(f"  Total loops:     {total_loops}")

        # Checkpoints
        if self.checkpoint_burns:
            total_act = sum(a for _, a, _ in self.checkpoint_burns if a is not None)
            total_loops = sum(
                l for _, _, l in self.checkpoint_burns if l is not None and l > 0
            )
            bg_burns = sum(1 for r, _, _ in self.checkpoint_burns if r is None)
            print(f"\nCheckpoint burns ({len(self.checkpoint_burns)} total):")
            print(f"  Background burns: {bg_burns}")
            print(f"  Total duration:   {total_act:.2f}s")
            if total_loops > 0:
                print(f"  Total loops:      {total_loops}")

        # Cooldown
        if self.cooldown_segments:
            total_req = sum(r for r, _, _ in self.cooldown_segments if r is not None)
            total_act = sum(a for _, a, _ in self.cooldown_segments if a is not None)
            total_loops = sum(l for _, _, l in self.cooldown_segments if l is not None)
            print(f"\nCooldown ({len(self.cooldown_segments)} segments):")
            print(f"  Total requested: {total_req:.2f}s")
            print(f"  Total actual:    {total_act:.2f}s")
            print(
                f"  Overshoot:       {total_act - total_req:.3f}s ({((total_act / total_req - 1) * 100):.2f}%)"
            )
            print(f"  Total loops:     {total_loops}")

        # Grand total
        all_actual = (
            sum(a for _, a, _ in self.warmup_segments if a is not None)
            + sum(a for _, a, _ in self.checkpoint_burns if a is not None)
            + sum(a for _, a, _ in self.cooldown_segments if a is not None)
        )
        all_loops = sum(l for _, _, l in self.warmup_segments if l is not None) + sum(
            l for _, _, l in self.cooldown_segments if l is not None
        )
        print(f"\nGrand Total:")
        print(f"  Total burn time: {all_actual:.2f}s")
        print(f"  Total loops:     {all_loops}")
        print("=" * 60 + "\n")


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


def build_loader(
    tokenizer,
    seqlen: int,
    bs: int,
    dataset_name: str = "dummy",
    c4_split: str = "train[:20000]",
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
    gpu_total_gb = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
    mem = calc_mem(
        cfg,
        args.batch_size,
        args.sequence_length,
        effective_precision,
        args.gradient_checkpointing,
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
    optim = torch.optim.AdamW(
        params, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    sched = get_cosine_schedule_with_warmup(optim, args.warmup_steps, args.max_steps)

    scaler = torch.cuda.amp.GradScaler() if effective_precision == "fp16" else None

    burner: Optional[GpuBurner] = None
    if args.smooth_power:
        burner = GpuBurner(
            device=device,
            dtype=torch.float32,
        )

    # ---- Startup warmup ramp (pre-training) ----
    if args.smooth_power and burner is not None and args.warmup_total_s > 0:
        if rank == 0:
            print(
                f"[BURN] Starting warmup ramp ({args.warmup_total_s}s over {args.warmup_segments} segments)"
            )
        total = args.warmup_total_s
        segments = max(1, args.warmup_segments)
        base_segment = total / segments

        # Cosine ramp from 0 -> 1 for smooth derivative at endpoints
        for i in range(segments):
            x = (i + 0.5) / segments  # 0..1
            intensity = 0.5 * (1.0 - math.cos(math.pi * x))  # 0 → 1
            duration = base_segment * max(0.0, intensity)
            if duration > 0:
                burner.burn_for(duration, phase="warmup")
        if rank == 0:
            print(f"[BURN] Warmup complete")

    # Training loop
    step = 0
    data_iter = iter(loader)
    model.train()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    torch.cuda.synchronize()
    t0 = time.time()

    while step < args.max_steps:
        iter_t0 = time.time()
        is_ckpt_step = step > 0 and step % args.save_steps == 0
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

                optim.zero_grad(set_to_none=True)
                sched.step()

            # Logging
            iter_t1 = time.time()
            iter_time = iter_t1 - iter_t0
            if step % args.log_steps == 0 and rank == 0:
                print(
                    f"step {step}: "
                    f"loss={loss.item() * args.gradient_accumulation_steps:.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e} "
                    f"step_time={iter_time:.3f}s"
                )
                log_mem(step)

            # Checkpointing
            if is_ckpt_step:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")

                if rank == 0:
                    t_ckpt0 = time.time()
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
                    t_ckpt1 = time.time()
                    ckpt_time = t_ckpt1 - t_ckpt0
                    print(f"Saved checkpoint: {ckpt_path} (took {ckpt_time:.2f}s)")

                # Synchronize all ranks
                if world_size > 1:
                    torch.distributed.barrier()

                # Optional: short burn after checkpoint to smooth power dip
                if args.smooth_power and args.enable_ckpt_burn and burner is not None:
                    # Brief 2-second burn to fill power gap after checkpoint I/O
                    burner.burn_for(2.0, phase="checkpoint")

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

    # ---- Cooldown decay ramp (post-training) ----
    if args.smooth_power and burner is not None and args.cooldown_total_s > 0:
        if rank == 0:
            print(
                f"[BURN] Starting cooldown ramp ({args.cooldown_total_s}s over {args.cooldown_segments} segments)"
            )
        total = args.cooldown_total_s
        segments = max(1, args.cooldown_segments)
        base_segment = total / segments

        # Cosine from 1 -> 0 for smooth decay
        for i in range(segments):
            x = (i + 0.5) / segments  # 0..1
            intensity = 0.5 * (1.0 + math.cos(math.pi * x))  # 1 → 0
            duration = base_segment * max(0.0, intensity)
            if duration > 0:
                burner.burn_for(duration, phase="cooldown")
        if rank == 0:
            print(f"[BURN] Cooldown complete")

    # Print burn summary
    if args.smooth_power and burner is not None:
        burner.print_summary(rank)

    cleanup_distributed()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
