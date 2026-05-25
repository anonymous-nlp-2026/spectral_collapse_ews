"""Offline n_sensitivity: uses pre-tokenized eval_5k.pt directly."""
import os, sys, json, time, argparse
import torch
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

EPSILON = 1e-10
BATCH_SIZE = 16
DEFAULT_N_VALUES = [250, 500, 1000, 2000, 5000]

def compute_slv(E):
    sv = torch.linalg.svdvals(E)
    return torch.sum(torch.log(sv[sv > EPSILON])).item()

def compute_effective_rank(E):
    sv = torch.linalg.svdvals(E)
    sv = sv[sv > EPSILON]
    p = sv / sv.sum()
    entropy = -torch.sum(p * torch.log(p)).item()
    return np.exp(entropy)

def extract_embeddings_from_tokens(model, token_data, device, max_n=5000):
    model.eval()
    all_emb = []
    n = min(len(token_data), max_n)
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch_items = token_data[i:i+BATCH_SIZE]
            input_ids = torch.stack([item['input_ids'] for item in batch_items]).to(device)
            attention_mask = torch.stack([item['attention_mask'] for item in batch_items]).to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1)
            mean_emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
            all_emb.append(mean_emb.cpu())
    return torch.cat(all_emb, dim=0).float()

def run_sensitivity(E_pool, n_values, n_repeats, rng):
    pool_n, d = E_pool.shape
    results = []
    for n in n_values:
        if n > pool_n:
            n = pool_n
        slv_vals, reff_vals = [], []
        for _ in range(n_repeats):
            indices = rng.choice(pool_n, size=n, replace=False)
            E_sub = E_pool[indices]
            slv_vals.append(compute_slv(E_sub))
            reff_vals.append(compute_effective_rank(E_sub))
        n_sv = min(n, d)
        results.append({
            "n": n, "n_singular_values": n_sv,
            "slv_mean": float(np.mean(slv_vals)), "slv_std": float(np.std(slv_vals)),
            "slv_values": [float(v) for v in slv_vals],
            "reff_mean": float(np.mean(reff_vals)), "reff_std": float(np.std(reff_vals)),
        })
        cv = abs(np.std(slv_vals) / np.mean(slv_vals) * 100) if np.mean(slv_vals) != 0 else 0
        print(f"  n={n:5d}: SLV = {np.mean(slv_vals):.2f} +/- {np.std(slv_vals):.2f}  (r_eff={np.mean(reff_vals):.1f}, CV={cv:.3f}%)")
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--data_path", default="./data/eval_5k.pt")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_values", type=int, nargs="+", default=DEFAULT_N_VALUES)
    p.add_argument("--n_repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="./results/n_sensitivity_results.json")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Data: {args.data_path}")
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path).to(device)
    d = model.config.hidden_size
    print(f"hidden_size = {d}")

    print(f"\nLoading pre-tokenized data from {args.data_path}")
    token_data = torch.load(args.data_path, weights_only=False)
    print(f"  Loaded {len(token_data)} samples")

    pool_size = max(max(args.n_values), 5000)
    print(f"\nExtracting embeddings (pool_size={min(pool_size, len(token_data))})...")
    t0 = time.time()
    E_pool = extract_embeddings_from_tokens(model, token_data, device, max_n=pool_size)
    print(f"  Embedding pool: {E_pool.shape} ({time.time()-t0:.1f}s)")

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"\nRunning sensitivity analysis (n_repeats={args.n_repeats})...")
    results = run_sensitivity(E_pool, sorted(args.n_values), args.n_repeats, rng)

    ref = next((r for r in results if r["n"] == 1000), results[0])
    print(f"\n{'n':>6s}  {'k':>5s}  {'SLV mean':>10s}  {'SLV std':>9s}  {'Rel vs 1000':>12s}  {'CV%':>7s}")
    print("-" * 60)
    for r in results:
        rel = (r["slv_mean"] - ref["slv_mean"]) / abs(ref["slv_mean"]) * 100
        cv = r["slv_std"] / abs(r["slv_mean"]) * 100 if r["slv_mean"] != 0 else 0
        print(f"{r['n']:6d}  {r['n_singular_values']:5d}  {r['slv_mean']:10.2f}  {r['slv_std']:9.2f}  {rel:+11.2f}%  {cv:7.3f}")

    output = {
        "checkpoint": args.checkpoint_path, "device": device, "seed": args.seed,
        "n_repeats": args.n_repeats, "pool_size": int(E_pool.shape[0]),
        "hidden_size": d, "epsilon": EPSILON, "results": results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
