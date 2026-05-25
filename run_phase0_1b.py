"""Phase 0 (1B): Pure synthetic recursive training on Pythia-1B to verify
cross-scale collapse dynamics.

Usage: python run_phase0_1b.py
Output: results/phase0_1b/phase0_1b_results.json + plots
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.pipeline import run_recursive_training


def plot_results(results, save_path):
    """Plot Phase 0 1B results."""
    generations = [r["generation"] for r in results]
    eff_ranks = [r["effective_rank"] for r in results]
    perplexities = [r["perplexity"] for r in results]
    distinct_4s = [r["distinct_4"] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(generations, eff_ranks, 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Effective Rank')
    axes[0].set_title('Effective Rank vs Generation (1B)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(generations, perplexities, 'r-o', linewidth=2, markersize=6)
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Held-out Perplexity vs Generation (1B)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(generations, distinct_4s, 'g-o', linewidth=2, markersize=6)
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Distinct-4')
    axes[2].set_title('Distinct-4 vs Generation (1B)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to {save_path}")


def compute_decline_rates(results):
    """Compute effective rank decline rate: first 5 vs last 5 generations (Gen 1 baseline)."""
    eff_ranks = [r["effective_rank"] for r in results]
    
    first_half_declines = []
    for i in range(2, min(6, len(eff_ranks))):
        decline = eff_ranks[i-1] - eff_ranks[i]
        first_half_declines.append(decline)
    
    second_half_declines = []
    for i in range(6, min(11, len(eff_ranks))):
        decline = eff_ranks[i-1] - eff_ranks[i]
        second_half_declines.append(decline)
    
    avg_first = np.mean(first_half_declines) if first_half_declines else 0
    avg_second = np.mean(second_half_declines) if second_half_declines else 0
    
    return {
        "first_half_avg_decline": float(avg_first),
        "second_half_avg_decline": float(avg_second),
        "acceleration_ratio": float(avg_second / avg_first) if avg_first > 0 else 0,
        "is_accelerating": avg_second > avg_first,
        "baseline_gen": 1
    }


def main():
    print("="*60)
    print("Phase 0 (1B): Pure Collapse Characterization - Pythia 1B")
    print("="*60)
    
    config = Config()
    config.model_name = "./models/EleutherAI/pythia-1b-deduped"
    config.real_data_ratio = 0.0
    config.device = "cuda:0"
    config.checkpoint_dir = "./checkpoints_1b/"
    config.results_dir = "./results/phase0_1b/"
    
    os.makedirs(config.results_dir, exist_ok=True)
    
    results = run_recursive_training(config)
    
    plot_path = os.path.join(config.results_dir, "phase0_1b_plots.png")
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
    
    analysis_path = os.path.join(config.results_dir, "phase0_1b_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print("\n" + "="*60)
    print("DECLINE RATE ANALYSIS (baseline=Gen 1)")
    print("="*60)
    print(f"First half (Gen 1-5) avg decline: {decline_analysis['first_half_avg_decline']:.4f}")
    print(f"Second half (Gen 5-10) avg decline: {decline_analysis['second_half_avg_decline']:.4f}")
    print(f"Acceleration ratio: {decline_analysis['acceleration_ratio']:.2f}x")
    print(f"Is accelerating: {decline_analysis['is_accelerating']}")
    print(f"\nFull analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
