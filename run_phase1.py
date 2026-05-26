"""Phase 1: Fixed-budget comparison (4 conditions x 3 seeds x 10 generations).

Conditions:
  (a) pure_collapse: 100% synthetic, 0% real -- control group
  (b) fixed_mix: 50% real + 50% synthetic -- uniform mixing baseline
  (c) linear_increasing: real data ratio linearly increases from r_min to r_max
  (d) adaptive: spectral-guided adaptive mixing -- SpectralController

Usage:
  python run_phase1.py --condition adaptive --seed 42 --tau 0.05 --gpu 0
  python run_phase1.py --condition pure_collapse --seed 42 --gpu 0
  python run_phase1.py --model pythia-1b --condition adaptive --seed 42 --gpu 0
  python run_phase1.py --model pythia-1b --condition fixed_mix --seed 42 --gpu 0

Output:
  results/phase1_{condition}[_k{k}]_seed{seed}_results.json   (410M, default)
  results/phase1_1b_{condition}[_k{k}]_seed{seed}_results.json (1B)
  ./checkpoints_phase1/{condition}[_k{k}]_seed{seed}/  (410M)
  ./checkpoints_phase1_1b/{condition}[_k{k}]_seed{seed}/ (1B)

Dependencies:
  src.pipeline.run_recursive_training, src.config.Config, src.controller.SpectralController
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import random
import numpy as np
import torch

from src.config import Config
from src.pipeline import run_recursive_training


CONDITION_CHOICES = ["pure_collapse", "fixed_mix", "linear_increasing", "adaptive"]
CHECKPOINT_BASE = "./checkpoints_phase1"
RESULTS_DIR = "./results"

# Model registry: short name -> (HuggingFace model ID, checkpoint suffix, results prefix)
MODEL_REGISTRY = {
    "pythia-410m": {
        "model_name": "EleutherAI/pythia-410m-deduped",
        "ckpt_suffix": "",
        "results_prefix": "phase1",
    },
    "pythia-1b": {
        "model_name": "EleutherAI/pythia-1b-deduped",
        "ckpt_suffix": "_1b",
        "results_prefix": "phase1_1b",
    },
}


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: Fixed-budget comparison (4 conditions x 3 seeds x 10 gens)"
    )
    parser.add_argument("--model", type=str, default="pythia-410m",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model variant (default: pythia-410m)")
    parser.add_argument("--condition", type=str, required=True,
                        choices=CONDITION_CHOICES,
                        help="pure_collapse | fixed_mix | linear_increasing | adaptive")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed (42 / 43 / 44)")
    parser.add_argument("--signal", type=str, default="log_det",
                        choices=["log_det", "eff_rank"],
                        help="Spectral signal for controller (adaptive only, default: log_det)")
    parser.add_argument("--tau", type=float, default=0.002,
                        help="Spectral decline threshold (adaptive only, default: 0.002)")
    parser.add_argument("--k", type=float, default=10.0,
                        help="Control gain (adaptive only, default: 10.0)")
    parser.add_argument("--signal_mode", type=str, default="per_gen",
                        choices=["per_gen", "cumulative"],
                        help="Signal delta mode (adaptive only, default: per_gen)")
    parser.add_argument("--r_min", type=float, default=0.20,
                        help="Min real data ratio (adaptive/linear_increasing, default: 0.20)")
    parser.add_argument("--r_max", type=float, default=0.80,
                        help="Max real data ratio (adaptive/linear_increasing, default: 0.80)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (default: 0, use -1 for CPU)")
    parser.add_argument("--generations", type=int, default=10,
                        help="Number of generations (default: 10)")
    parser.add_argument("--real_ratio", type=float, default=0.5,
                        help="Real data ratio for fixed_mix condition (default: 0.5)")
    parser.add_argument("--start_gen", type=int, default=0,
                        help="Generation to resume from (0=fresh start, requires checkpoint and results JSON)")
    return parser.parse_args()


def build_config(args):
    """Build Config and ratio_schedule from CLI args.

    Returns:
        (config, ratio_schedule) where ratio_schedule is a list of per-generation
        real data ratios (length = args.generations), or None for adaptive mode.
    """
    model_info = MODEL_REGISTRY[args.model]

    config = Config()
    config.model_name = model_info["model_name"]
    config.seed = args.seed
    config.device = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
    config.num_generations = args.generations

    T = args.generations
    ratio_schedule = None

    if args.condition == "pure_collapse":
        config.real_data_ratio = 0.0
        ratio_schedule = [0.0] * T
    elif args.condition == "fixed_mix":
        config.real_data_ratio = args.real_ratio
        ratio_schedule = [args.real_ratio] * T
    elif args.condition == "linear_increasing":
        # r_t = r_min + (r_max - r_min) * t / (T-1), t=0..T-1
        config.real_data_ratio = args.r_min
        ratio_schedule = [
            args.r_min + (args.r_max - args.r_min) * t / (T - 1) for t in range(T)
        ] if T > 1 else [args.r_min]
    elif args.condition == "adaptive":
        config.real_data_ratio = -1.0
        config.controller_k = args.k
        config.controller_tau = args.tau
        config.controller_signal = args.signal
        config.controller_r_min = args.r_min
        config.controller_r_max = args.r_max
        config.controller_signal_mode = args.signal_mode
        ratio_schedule = None

    return config, ratio_schedule


def main():
    args = parse_args()
    set_seed(args.seed)

    model_info = MODEL_REGISTRY[args.model]
    ckpt_suffix = model_info["ckpt_suffix"]
    results_prefix = model_info["results_prefix"]

    # Include k in path for adaptive condition to avoid collisions between different k values
    if args.condition == "adaptive":
        k_str = str(int(args.k)) if args.k == int(args.k) else str(args.k)
        run_tag = f"{args.condition}_k{k_str}_seed{args.seed}"
    elif args.condition == "fixed_mix" and args.real_ratio != 0.5:
        r_str = "r" + str(args.real_ratio).replace("0.", "0")
        run_tag = f"{args.condition}_{r_str}_seed{args.seed}"
    else:
        run_tag = f"{args.condition}_seed{args.seed}"

    ckpt_dir = os.path.join(
        CHECKPOINT_BASE + ckpt_suffix, run_tag
    )
    results_filename = f"{results_prefix}_{run_tag}_results.json"

    print("=" * 60)
    print(f"Phase 1: {args.condition} | model={args.model} | seed={args.seed} | gpu={args.gpu}")
    print(f"Model: {model_info['model_name']}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Results: {RESULTS_DIR}/{results_filename}")
    if args.condition == "adaptive":
        print(f"Controller: signal={args.signal}, signal_mode={args.signal_mode}, tau={args.tau}, k={args.k}, "
              f"r_min={args.r_min}, r_max={args.r_max}")
    if args.condition == "linear_increasing":
        print(f"Linear schedule: r_min={args.r_min} -> r_max={args.r_max} "
              f"over {args.generations} generations")
    print("=" * 60)

    config, ratio_schedule = build_config(args)

    results = run_recursive_training(
        config,
        results_dir=RESULTS_DIR,
        checkpoint_dir=ckpt_dir,
        results_filename=results_filename,
        ratio_schedule=ratio_schedule,
        resume_from_gen=args.start_gen,
    )

    # Save summary with full metadata for reproducibility
    summary = {
        "phase": 1,
        "model": args.model,
        "model_name": model_info["model_name"],
        "condition": args.condition,
        "seed": args.seed,
        "num_generations": args.generations,
        "gpu": args.gpu,
        "results": results,
        "config": {
            "model": config.model_name,
            "real_data_ratio": config.real_data_ratio,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_synthetic_samples": config.num_synthetic_samples,
        },
    }
    if args.condition == "adaptive":
        summary["controller_params"] = {
            "k": args.k, "tau": args.tau, "signal": args.signal, "signal_mode": args.signal_mode,
            "r_min": args.r_min, "r_max": args.r_max,
        }
    if args.condition == "linear_increasing":
        summary["ratio_schedule"] = ratio_schedule

    summary_path = os.path.join(
        RESULTS_DIR, f"{results_prefix}_{run_tag}_summary.json"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
