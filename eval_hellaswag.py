"""
HellaSwag 0-shot 4-choice evaluation for Pythia checkpoints.

Usage:
    python eval_hellaswag.py --checkpoint_dir /path/to/checkpoints_410m/ --batch_size 8
    python eval_hellaswag.py --checkpoint_dir /path/to/checkpoints_410m/ --gen 3 --batch_size 16

Input:  checkpoint directory containing gen_N/ subdirectories (HuggingFace format)
Output: JSON array to stdout: [{"gen": N, "hellaswag_acc": float}, ...]

Dependencies: torch, transformers, datasets (or pandas for parquet fallback)
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hellaswag(data_path=None):
    """Load HellaSwag validation split. Tries HF cache first, then parquet fallback."""
    # Try HF datasets cache
    if data_path is None:
        try:
            from datasets import load_dataset
            ds = load_dataset("Rowan/hellaswag", split="validation")
            return ds
        except Exception:
            pass

        # Try known parquet locations
        for candidate in [
            "~/.cache/huggingface/hellaswag_manual/validation.parquet",
            "~/.cache/huggingface/hellaswag_manual/validation-00000-of-00001.parquet",
        ]:
            if os.path.exists(candidate):
                data_path = candidate
                break

    if data_path is not None:
        if data_path.endswith(".parquet"):
            import pandas as pd
            df = pd.read_parquet(data_path)
            return df.to_dict("records")
        elif data_path.endswith(".json") or data_path.endswith(".jsonl"):
            records = []
            with open(data_path) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            return records
        else:
            from datasets import load_dataset
            ds = load_dataset("parquet", data_files={"validation": data_path}, split="validation")
            return ds

    raise FileNotFoundError(
        "Cannot find HellaSwag data. Use --data_path to specify location."
    )


def preprocess_sample(sample):
    """Extract context and 4 endings from a HellaSwag sample."""
    ctx = sample["ctx_a"] + " " + sample["ctx_b"].strip()
    endings = sample["endings"]
    if isinstance(endings, str):
        endings = json.loads(endings)
    label = int(sample["label"])
    return ctx, endings, label


def compute_log_likelihood(model, tokenizer, context, continuation, device, max_length=2048):
    """Compute log-likelihood of continuation given context."""
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    cont_ids = tokenizer.encode(" " + continuation, add_special_tokens=False)

    input_ids = ctx_ids + cont_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        ctx_len = max(0, len(input_ids) - len(cont_ids))
    else:
        ctx_len = len(ctx_ids)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Compute log-prob for each continuation token
    shift_logits = logits[0, ctx_len - 1:-1, :]  # predict positions ctx_len..end
    shift_labels = input_tensor[0, ctx_len:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

    return token_log_probs.sum().item()


def compute_log_likelihood_batch(model, tokenizer, context, endings, device, max_length=2048):
    """Compute log-likelihoods for all 4 endings of one sample in a single batch."""
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)

    all_input_ids = []
    ctx_lens = []
    cont_lens = []

    for ending in endings:
        cont_ids = tokenizer.encode(" " + ending, add_special_tokens=False)
        input_ids = ctx_ids + cont_ids
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            c_len = max(0, len(input_ids) - len(cont_ids))
        else:
            c_len = len(ctx_ids)
        all_input_ids.append(input_ids)
        ctx_lens.append(c_len)
        cont_lens.append(len(input_ids) - c_len)

    # Pad to same length
    max_len = max(len(ids) for ids in all_input_ids)
    padded = []
    attention_masks = []
    for ids in all_input_ids:
        pad_len = max_len - len(ids)
        padded.append([tokenizer.pad_token_id or 0] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))
        # Adjust ctx_lens for left-padding
    for i in range(len(all_input_ids)):
        pad_len = max_len - len(all_input_ids[i])
        ctx_lens[i] += pad_len

    input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
    attn_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_tensor, attention_mask=attn_mask)
        logits = outputs.logits  # (4, seq_len, vocab_size)

    log_likelihoods = []
    for i in range(len(endings)):
        shift_logits = logits[i, ctx_lens[i] - 1:-1, :]
        shift_labels = input_tensor[i, ctx_lens[i]:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        # Only count non-pad positions
        mask = attn_mask[i, ctx_lens[i]:]
        log_likelihoods.append((token_log_probs * mask.float()).sum().item())

    return log_likelihoods


def evaluate_checkpoint(model_path, data, batch_size=8, device="cuda"):
    """Evaluate a single checkpoint on HellaSwag."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    correct = 0
    total = 0

    for i, sample in enumerate(data):
        ctx, endings, label = preprocess_sample(sample)
        log_liks = compute_log_likelihood_batch(model, tokenizer, ctx, endings, device)
        pred = max(range(4), key=lambda j: log_liks[j])
        if pred == label:
            correct += 1
        total += 1

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(data)}] running acc: {correct/total:.4f}", file=sys.stderr)

    del model
    torch.cuda.empty_cache()

    return correct / total if total > 0 else 0.0


def find_generations(checkpoint_dir):
    """Find all gen_N subdirectories, return sorted list of (gen_num, path)."""
    gens = []
    base = Path(checkpoint_dir)
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("gen_"):
            try:
                gen_num = int(d.name.split("_")[1])
                if (d / "config.json").exists():
                    gens.append((gen_num, str(d)))
            except (ValueError, IndexError):
                continue
    return sorted(gens, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(description="HellaSwag 0-shot evaluation for Pythia checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing gen_N/ subdirectories, or a single checkpoint dir")
    parser.add_argument("--gen", type=int, default=None,
                        help="Evaluate only this generation (default: all)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation (controls memory, not speed for per-sample scoring)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to HellaSwag data (parquet/json/jsonl). Auto-detected if not set.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: stdout)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (for debugging)")
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Determine checkpoints to evaluate
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        print(f"Error: {args.checkpoint_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Check if this is a parent dir with gen_N/ subdirs or a single checkpoint
    gens = find_generations(args.checkpoint_dir)
    if not gens and (ckpt_dir / "config.json").exists():
        # Single checkpoint directory
        gens = [(0, str(ckpt_dir))]
    elif not gens:
        print(f"Error: No valid checkpoints found in {args.checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    if args.gen is not None:
        gens = [(g, p) for g, p in gens if g == args.gen]
        if not gens:
            print(f"Error: gen_{args.gen} not found in {args.checkpoint_dir}", file=sys.stderr)
            sys.exit(1)

    # Load data
    print(f"Loading HellaSwag validation data...", file=sys.stderr)
    data = load_hellaswag(args.data_path)
    if hasattr(data, '__len__'):
        print(f"Loaded {len(data)} samples", file=sys.stderr)

    if args.max_samples:
        data = list(data)[:args.max_samples]

    # Evaluate
    results = []
    for gen_num, path in gens:
        print(f"Evaluating gen_{gen_num}: {path}", file=sys.stderr)
        try:
            acc = evaluate_checkpoint(path, data, batch_size=args.batch_size, device=args.device)
            results.append({"gen": gen_num, "hellaswag_acc": round(acc, 6)})
            print(f"  gen_{gen_num}: hellaswag_acc = {acc:.4f}", file=sys.stderr)
        except Exception as e:
            warnings.warn(f"Failed to evaluate gen_{gen_num}: {e}")
            results.append({"gen": gen_num, "hellaswag_acc": None, "error": str(e)})

    # Output
    output_json = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
