"""Phase 0: Pure synthetic recursive training to characterize collapse."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.pipeline import run_recursive_training


def plot_results(results, save_path):
    generations = [r["generation"] for r in results]
    eff_ranks = [r["effective_rank"] for r in results]
    perplexities = [r["perplexity"] for r in results]
    distinct_4s = [r["distinct_4"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(generations, eff_ranks, 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Effective Rank')
    axes[0].set_title('Effective Rank vs Generation')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(generations, perplexities, 'r-o', linewidth=2, markersize=6)
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Held-out Perplexity vs Generation')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(generations, distinct_4s, 'g-o', linewidth=2, markersize=6)
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Distinct-4')
    axes[2].set_title('Distinct-4 vs Generation')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to {save_path}")


def compute_decline_rates(results):
    eff_ranks = [r["effective_rank"] for r in results]

    first_half_declines = []
    for i in range(2, 6):
        if i < len(eff_ranks):
            decline = eff_ranks[i-1] - eff_ranks[i]
            first_half_declines.append(decline)

    second_half_declines = []
    for i in range(6, 11):
        if i < len(eff_ranks):
            decline = eff_ranks[i-1] - eff_ranks[i]
            second_half_declines.append(decline)

    avg_first = np.mean(first_half_declines) if first_half_declines else 0
    avg_second = np.mean(second_half_declines) if second_half_declines else 0

    return {
        "first_5_gen_avg_decline": float(avg_first),
        "second_5_gen_avg_decline": float(avg_second),
        "acceleration_ratio": float(avg_second / avg_first) if avg_first > 0 else 0,
        "is_accelerating": bool(avg_second > avg_first)
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Pure Collapse Characterization")
    parser.add_argument("--resume_from_gen", type=int, default=0,
                        help="Resume from generation N (loads gen N-1 checkpoint)")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 0: Pure Collapse Characterization")
    if args.resume_from_gen > 0:
        print(f"  Resuming from generation {args.resume_from_gen}")
    print("=" * 60)

    config = Config()
    config.real_data_ratio = 0.0

    results = run_recursive_training(config, resume_from_gen=args.resume_from_gen)

    plot_path = os.path.join(config.results_dir, "phase0_plots.png")
    plot_results(results, plot_path)

    decline_analysis = compute_decline_rates(results)

    analysis = {
        "results": results,
        "decline_analysis": decline_analysis,
        "config": {
            "model": config.model_name,
            "num_generations": config.num_generations,
            "real_data_ratio": config.real_data_ratio,
            "seed": config.seed,
            "num_synthetic_samples": config.num_synthetic_samples,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
        }
    }

    analysis_path = os.path.join(config.results_dir, "phase0_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o) if isinstance(o, (np.integer, np.floating)) else o.tolist() if isinstance(o, np.ndarray) else o)

    print("\n" + "=" * 60)
    print("DECLINE RATE ANALYSIS")
    print("=" * 60)
    print(f"First 5 generations avg decline: {decline_analysis['first_5_gen_avg_decline']:.4f}")
    print(f"Second 5 generations avg decline: {decline_analysis['second_5_gen_avg_decline']:.4f}")
    print(f"Acceleration ratio: {decline_analysis['acceleration_ratio']:.2f}x")
    print(f"Is accelerating: {decline_analysis['is_accelerating']}")

    print(f"\nFull analysis saved to {analysis_path}")
    print(f"Plots saved to {plot_path}")


if __name__ == "__main__":
    main()
