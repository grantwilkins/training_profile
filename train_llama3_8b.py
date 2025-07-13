"""
train_llama3.py – Minimal single‑node multi‑GPU trainer for Meta‑Llama‑3‑8B
==========================================================================

This script lets you reproduce three power‑trace scenarios on an 8×H100 box:
  * normal forward/backward training
  * synchronisation stalls (when a rank is SIGSTOP‑ed)
  * fail‑stop crashes (SIGKILL or GPU XID error)

Key design choices
------------------
* **Pure HuggingFace/Transformers** – no Megatron or DeepSpeed build needed.
* **torchrun‑friendly** – just `torchrun --nproc_per_node=8 train_llama3.py …`.
* **Data‑parallel only** – tensor‑parallelism is disabled to keep collectives
  visible (they are handled by PyTorch DDP’s All‑Reduce).
* **Small micro‑batch with gradient accumulation** – amplifies wait time at each
  collective so the power dip is easy to spot.

Example run (BF16)
------------------
```bash
export TOKENIZERS_PARALLELISM=false  # avoids tokenizer spam
GLOBAL_BATCH=512
MICRO_BATCH=4
SEQ_LEN=2048

torchrun --nnodes 1 --nproc_per_node 8 train_llama3.py \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --dataset_path /data/the_pile \
    --sequence_length $SEQ_LEN \
    --micro_batch_size $MICRO_BATCH \
    --global_batch_size $GLOBAL_BATCH \
    --bf16 \
    --output_dir /scratch/llama3_ckpt \
    --train_steps 2000
```
"""

import argparse
import os
from typing import Dict, List

from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Llama‑3 8B on one node (8 GPUs)"
    )

    # Model & data
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="HF dataset name or local path"
    )
    parser.add_argument(
        "--text_column", type=str, default=None, help="Column containing raw text"
    )

    # Sequence & batching
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument(
        "--micro_batch_size", type=int, required=True, help="Per‑GPU batch (examples)"
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        required=True,
        help="Effective batch across GPUs",
    )

    # Optimisation & precision
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true", help="Use BF16 precision")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision (mutually exclusive with --bf16)",
    )

    # Runtime
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


def get_dataset(
    dataset_path: str,
    tokenizer: LlamaTokenizerFast,
    seq_len: int,
    text_column: str = None,
):
    """Load and tokenize dataset into fixed‑length blocks suitable for language modelling."""
    ds = load_dataset(dataset_path, split="train")
    column = text_column or (
        "text" if "text" in ds.column_names else ds.column_names[0]
    )

    def tokenize_fn(batch: Dict[str, List[str]]):
        return tokenizer(batch[column])

    tokenized = ds.map(tokenize_fn, batched=True, num_proc=os.cpu_count())

    # Group into blocks of seq_len tokens
    def group_fn(examples):
        # Concatenate and split
        concatenated = []
        for e in examples["input_ids"]:
            concatenated.extend(e)
        total_length = (len(concatenated) // seq_len) * seq_len
        result = {
            "input_ids": [
                concatenated[i : i + seq_len] for i in range(0, total_length, seq_len)
            ]
        }
        return result

    return tokenized.map(group_fn, batched=True, num_proc=os.cpu_count())


def main():
    import torch.distributed

    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Initialize distributed process group if needed
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # ensure a pad token is defined
    dataset = load_dataset(
        "allenai/c4", data_files="en/c4-train.0000*-of-01024.json.gz"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    model = LlamaForCausalLM.from_pretrained(args.model_name)
    model.gradient_checkpointing_enable()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = args.global_batch_size // (args.micro_batch_size * world_size)
    if (
        grad_accum < 1
        or args.global_batch_size % (args.micro_batch_size * world_size) != 0
    ):
        raise ValueError(
            "global_batch_size must be an integer multiple of micro_batch_size * WORLD_SIZE"
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.train_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        remove_unused_columns=True,
        dataloader_num_workers=4,
        report_to=["none"],  # keep logs simple; integrate with wandb if desired
        ddp_backend="nccl",  # uses NCCL for collectives visible in power trace
        disable_tqdm=local_rank != 0,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if local_rank == 0:
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
