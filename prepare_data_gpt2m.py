"""Prepare tokenized data for GPT-2-medium experiments.

Re-tokenizes the same OpenWebText subset used for Pythia experiments,
but with the GPT-2 tokenizer. Output files are saved with _gpt2m suffix.

Usage:
  python prepare_data_gpt2m.py
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "gpt2-medium"
CACHE_DIR = "./models/"
DATA_DIR = "./data/"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Tokenizer: {MODEL_NAME}, vocab_size={tokenizer.vocab_size}")
print("Loading dataset (streaming)...")
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

samples = []
skipped = 0
TARGET = 56000

for i, example in enumerate(ds):
    if len(samples) >= TARGET:
        break
    text = example["text"]
    tokens = tokenizer(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    non_pad = tokens["attention_mask"].sum().item()
    if non_pad >= 256:
        samples.append({
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0)
        })
    else:
        skipped += 1
    if (i + 1) % 10000 == 0:
        print(f"  Processed {i+1} examples, kept {len(samples)}, skipped {skipped}")

print(f"Total processed, kept {len(samples)}, skipped {skipped}")

train_data = samples[:50000]
eval_data = samples[50000:55000]
spectral_data = samples[55000:56000]

os.makedirs(DATA_DIR, exist_ok=True)
torch.save(train_data, os.path.join(DATA_DIR, "train_real_50k_gpt2m.pt"))
torch.save(eval_data, os.path.join(DATA_DIR, "eval_5k_gpt2m.pt"))
torch.save(spectral_data, os.path.join(DATA_DIR, "spectral_1k_gpt2m.pt"))
print(f"Saved: train={len(train_data)}, eval={len(eval_data)}, spectral={len(spectral_data)}")
