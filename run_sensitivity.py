"""Controller parameter sensitivity grid search.

Sweeps k x tau for the adaptive (SpectralController) condition.

Grid:
  k   in {0.5, 1.0, 2.0}
  tau in {0.002, 0.005, 0.01}
  Total: 9 combinations

Fixed:
  seed=42, condition=adaptive, generations=10
  r_min=0.20, r_max=0.80
  model: EleutherAI/pythia-410m-deduped

Usage:
  python run_sensitivity.py --mode serial --gpu 0
  python run_sensitivity.py --mode single --k 0.5 --tau 0.01 --gpu 1
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import time
import random
import numpy as np
import torch

from src.config import Config
from src.pipeline import run_recursive_training

K_VALUES = [5, 10, 20, 50]
TAU_VALUES = [0.001, 0.002, 0.005]

CHECKPOINT_BASE = "./checkpoints_sensitivity"
RESULTS_DIR = "./results"

FIXED_SEED = 42
FIXED_GENERATIONS = 10
FIXED_R_MIN = 0.20
FIXED_R_MAX = 0.80
MODEL_NAME = "EleutherAI/pythia-410m-deduped"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single(k, tau, gpu, signal="log_det"):
    """Run one (k, tau) combination. Returns wall time in seconds."""
    tag = f"k{k}_tau{tau}"
    ckpt_dir = os.path.join(CHECKPOINT_BASE, tag)
    results_filename = f"sensitivity_{tag}_results.json"

    print("=" * 60)
    print(f"[Sensitivity] k={k}, tau={tau} | gpu={gpu}")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"  Results:    {RESULTS_DIR}/{results_filename}")
    print("=" * 60)

    set_seed(FIXED_SEED)

    config = Config()
    config.model_name = MODEL_NAME
    config.seed = FIXED_SEED
    config.device = "cpu" if gpu < 0 else f"cuda:{gpu}"
    config.num_generations = FIXED_GENERATIONS
    config.real_data_ratio = -1.0
    config.controller_k = k
    config.controller_tau = tau
    config.controller_r_min = FIXED_R_MIN
    config.controller_r_max = FIXED_R_MAX
    config.controller_signal = signal

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    t0 = time.time()
    results = run_recursive_training(
        config,
        results_dir=RESULTS_DIR,
        checkpoint_dir=ckpt_dir,
        results_filename=results_filename,
        ratio_schedule=None,
    )
    wall_time = time.time() - t0

    summary = {
        "experiment": "sensitivity",
        "k": k,
        "tau": tau,
        "seed": FIXED_SEED,
        "generations": FIXED_GENERATIONS,
        "r_min": FIXED_R_MIN,
        "r_max": FIXED_R_MAX,
        "model": MODEL_NAME,
        "gpu": gpu,
        "wall_time_seconds": round(wall_time, 1),
        "results": results,
    }
    summary_path = os.path.join(RESULTS_DIR, f"sensitivity_{tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Sensitivity] k={k}, tau={tau} done in {wall_time:.1f}s")
    print(f"  Summary: {summary_path}")
    return wall_time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controller parameter sensitivity: k x tau grid search"
    )
    parser.add_argument("--mode", type=str, default="serial",
                        choices=["serial", "single"],
                        help="serial: run all 9 combos; single: run one (k, tau)")
    parser.add_argument("--k", type=float, default=None,
                        help="k value (required for --mode single)")
    parser.add_argument("--tau", type=float, default=None,
                        help="tau value (required for --mode single)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index (default: 0)")
    parser.add_argument("--signal", type=str, default="log_det",
                        help="EWS signal type (default: log_det)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "single":
        if args.k is None or args.tau is None:
            print("ERROR: --mode single requires both --k and --tau")
            sys.exit(1)
        run_single(args.k, args.tau, args.gpu, args.signal)
    else:
        grid = [(k, tau) for k in K_VALUES for tau in TAU_VALUES]
        total = len(grid)
        print(f"[Sensitivity] Starting grid search: {total} combinations")
        print(f"  k   = {K_VALUES}")
        print(f"  tau = {TAU_VALUES}")
        print()

        times = []
        for i, (k, tau) in enumerate(grid, 1):
            print(f"\n>>> [{i}/{total}] k={k}, tau={tau}")
            wt = run_single(k, tau, args.gpu, args.signal)
            times.append((k, tau, wt))

        print("\n" + "=" * 60)
        print("[Sensitivity] All done. Wall times:")
        for k, tau, wt in times:
            print(f"  k={k}, tau={tau}: {wt:.1f}s")
        total_time = sum(t for _, _, t in times)
        print(f"  Total: {total_time:.1f}s")


if __name__ == "__main__":
    main()
