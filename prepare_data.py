from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped", cache_dir="./models/")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
    # Only keep samples that actually fill most of the context
    non_pad = tokens["attention_mask"].sum().item()
    if non_pad >= 256:  # at least half-filled
        samples.append({
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0)
        })
    else:
        skipped += 1
    if (i + 1) % 10000 == 0:
        print(f"  Processed {i+1} examples, kept {len(samples)}, skipped {skipped}")

print(f"Total processed, kept {len(samples)}, skipped {skipped}")

# Split
train_data = samples[:50000]
eval_data = samples[50000:55000]
spectral_data = samples[55000:56000]

os.makedirs("./data/", exist_ok=True)
torch.save(train_data, "./data/train_real_50k.pt")
torch.save(eval_data, "./data/eval_5k.pt")
torch.save(spectral_data, "./data/spectral_1k.pt")
print(f"Saved: train={len(train_data)}, eval={len(eval_data)}, spectral={len(spectral_data)}")
