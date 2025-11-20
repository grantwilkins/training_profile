from datasets import load_dataset

ds = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.0000*-of-01024.json.gz",
    split="train[:0.1%]",
)

ds.to_json("c4_tiny.jsonl", lines=True)
