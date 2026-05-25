import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import json
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from src.rankme import compute_effective_rank

CKPT_BASE = "./checkpoints_410m"
SPECTRAL_PATH = "./data/spectral_1k.pt"
OUTPUT_PATH = "./results/logdet_gen012.json"

GENERATIONS = [0, 1, 2]
DEVICE = torch.device("cpu")

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading spectral data...")
    spectral_samples = torch.load(SPECTRAL_PATH, map_location="cpu", weights_only=False)
    print(f"Loaded {len(spectral_samples)} samples")

    all_results = []

    for gen_idx in GENERATIONS:
        ckpt_path = os.path.join(CKPT_BASE, f"gen_{gen_idx}")
        print(f"\n{'='*50}")
        print(f"[Gen {gen_idx}] Loading model from {ckpt_path}")
        t0 = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        model.eval()
        load_time = time.time() - t0
        print(f"[Gen {gen_idx}] Model loaded in {load_time:.1f}s")

        print(f"[Gen {gen_idx}] Computing effective rank & log-det on CPU...")
        t0 = time.time()
        effective_rank, singular_values, log_det = compute_effective_rank(
            model, spectral_samples, DEVICE, batch_size=16
        )
        compute_time = time.time() - t0
        print(f"[Gen {gen_idx}] Done in {compute_time:.1f}s")

        result = {
            "generation": gen_idx,
            "effective_rank": float(effective_rank),
            "log_det": float(log_det),
            "top_10_sv": singular_values[:10].tolist() if len(singular_values) >= 10 else singular_values.tolist(),
            "compute_time_sec": round(compute_time, 1),
        }
        all_results.append(result)
        print(f"[Gen {gen_idx}] eff_rank={effective_rank:.4f}, log_det={log_det:.4f}")

        del model
        import gc
        gc.collect()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for r in all_results:
        print(f"Gen {r['generation']}: eff_rank={r['effective_rank']:.4f}, log_det={r['log_det']:.4f}")

    if len(all_results) >= 2:
        log_dets = [r["log_det"] for r in all_results]
        monotonic = all(log_dets[i] >= log_dets[i+1] for i in range(len(log_dets)-1))
        print(f"\nlog_det monotonically decreasing: {monotonic}")
        eff_ranks = [r["effective_rank"] for r in all_results]
        mono_er = all(eff_ranks[i] >= eff_ranks[i+1] for i in range(len(eff_ranks)-1))
        print(f"eff_rank monotonically decreasing: {mono_er}")

if __name__ == "__main__":
    main()
