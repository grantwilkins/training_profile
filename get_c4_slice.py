from datasets import load_dataset

ds = load_dataset("c4", "en", split="train[:0.1%]")
ds.to_json("c4_tiny.jsonl")
