"""Phase 1 runner for GPT-2-medium cross-architecture validation.

Thin wrapper around src.pipeline.run_recursive_training with GPT-2-medium
specific configuration. Does NOT modify run_phase1.py or any existing code.

Usage:
  python run_gpt2m_phase1.py --condition pure_collapse --seed 42 --gpu 0
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


MODEL_NAME = "gpt2-medium"
MODEL_CACHE_DIR = "./models/"
DATA_DIR = "./data/"
CHECKPOINT_BASE = "./checkpoints_phase1_gpt2m"
RESULTS_DIR = "./results"
RESULTS_PREFIX = "phase1_gpt2m"

CONDITION_CHOICES = ["pure_collapse", "fixed_mix", "linear_increasing", "adaptive"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1 GPT-2-medium: cross-architecture validation"
    )
    parser.add_argument("--condition", type=str, required=True,
                        choices=CONDITION_CHOICES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--signal", type=str, default="log_det",
                        choices=["log_det", "eff_rank"])
    parser.add_argument("--tau", type=float, default=0.002)
    parser.add_argument("--k", type=float, default=20.0)
    parser.add_argument("--signal_mode", type=str, default="per_gen",
                        choices=["per_gen", "cumulative"])
    parser.add_argument("--r_min", type=float, default=0.20)
    parser.add_argument("--r_max", type=float, default=0.80)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--real_ratio", type=float, default=0.5)
    parser.add_argument("--start_gen", type=int, default=0)
    return parser.parse_args()


def build_config(args):
    config = Config()
    config.model_name = MODEL_NAME
    config.model_cache_dir = MODEL_CACHE_DIR
    config.seed = args.seed
    config.device = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
    config.num_generations = args.generations

    # GPT-2-medium specific data paths
    config.train_real_path = os.path.join(DATA_DIR, "train_real_50k_gpt2m.pt")
    config.eval_path = os.path.join(DATA_DIR, "eval_5k_gpt2m.pt")
    config.spectral_path = os.path.join(DATA_DIR, "spectral_1k_gpt2m.pt")

    T = args.generations
    ratio_schedule = None

    if args.condition == "pure_collapse":
        config.real_data_ratio = 0.0
        ratio_schedule = [0.0] * T
    elif args.condition == "fixed_mix":
        config.real_data_ratio = args.real_ratio
        ratio_schedule = [args.real_ratio] * T
    elif args.condition == "linear_increasing":
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

    if args.condition == "adaptive":
        k_str = str(int(args.k)) if args.k == int(args.k) else str(args.k)
        run_tag = f"{args.condition}_k{k_str}_seed{args.seed}"
    elif args.condition == "fixed_mix" and args.real_ratio != 0.5:
        r_str = "r" + str(args.real_ratio).replace("0.", "0")
        run_tag = f"{args.condition}_{r_str}_seed{args.seed}"
    else:
        run_tag = f"{args.condition}_seed{args.seed}"

    ckpt_dir = os.path.join(CHECKPOINT_BASE, run_tag)
    results_filename = f"{RESULTS_PREFIX}_{run_tag}_results.json"

    print("=" * 60)
    print(f"Phase 1 GPT-2-medium: {args.condition} | seed={args.seed} | gpu={args.gpu}")
    print(f"Model: {MODEL_NAME}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Results: {RESULTS_DIR}/{results_filename}")
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

    summary = {
        "phase": 1,
        "model": "gpt2-medium",
        "model_name": MODEL_NAME,
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
        RESULTS_DIR, f"{RESULTS_PREFIX}_{run_tag}_summary.json"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
