"""Phase 0 standalone analysis script.
Reads results from phase0_results.json, performs go/no-go decision and tau/k calibration.
Usage: python analyze_phase0.py [--results-path PATH] [--model-name NAME]
"""
import argparse
import json
import os
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(path):
    with open(path) as f:
        return json.load(f)


def compute_decline_rates(results, baseline_gen=1):
    """Compute decline rates from Gen 1 baseline."""
    eff_ranks = [r["effective_rank"] for r in results]
    # first_half: Gen 1->2 through Gen 4->5 (4 deltas), skip Gen 5->6
    first_half_deltas = []
    for i in range(2, 6):  # gen index 2,3,4,5
        if i < len(eff_ranks):
            first_half_deltas.append(eff_ranks[i-1] - eff_ranks[i])
    # second_half: Gen 6->7 through Gen 9->10 (4 deltas)
    second_half_deltas = []
    for i in range(7, 11):  # gen index 7,8,9,10
        if i < len(eff_ranks):
            second_half_deltas.append(eff_ranks[i-1] - eff_ranks[i])

    first_half_rate = np.mean(first_half_deltas) if first_half_deltas else 0.0
    second_half_rate = np.mean(second_half_deltas) if second_half_deltas else 0.0
    acceleration_ratio = second_half_rate / first_half_rate if first_half_rate > 0 else float('inf')

    return {
        "first_half_rate": float(first_half_rate),
        "second_half_rate": float(second_half_rate),
        "acceleration_ratio": float(acceleration_ratio),
        "first_half_deltas": [float(d) for d in first_half_deltas],
        "second_half_deltas": [float(d) for d in second_half_deltas],
    }


def compute_normalized_deltas(results, baseline_gen=1):
    """Compute normalized per-generation decline rate."""
    eff_ranks = [r["effective_rank"] for r in results]
    rank_baseline = abs(eff_ranks[baseline_gen])
    deltas = []
    for t in range(baseline_gen + 1, len(eff_ranks)):
        delta_t = (eff_ranks[t-1] - eff_ranks[t]) / rank_baseline
        deltas.append(float(delta_t))
    return deltas


def check_go_nogo(results, baseline_gen=1):
    """Go/no-go three-condition check."""
    eff_ranks = [r["effective_rank"] for r in results]
    ranks_subset = eff_ranks[baseline_gen:]  # Gen 1-10

    # Condition 1: monotonic decline (allow at most 1 gen increase)
    increases = 0
    for i in range(1, len(ranks_subset)):
        if ranks_subset[i] > ranks_subset[i-1]:
            increases += 1
    cond1_pass = increases <= 1
    cond1_detail = f"rank increases: {increases} (allowed: 1)"

    # Condition 2: second-half decline rate > first-half
    decline = compute_decline_rates(results, baseline_gen)
    cond2_pass = decline["acceleration_ratio"] >= 1.5
    cond2_detail = (f"first_half_rate={decline['first_half_rate']:.4f}, "
                    f"second_half_rate={decline['second_half_rate']:.4f}, "
                    f"ratio={decline['acceleration_ratio']:.4f}")

    # Condition 3: PPL Gen10 vs Gen1 increase > 20%
    ppl_gen1 = results[baseline_gen]["perplexity"]
    ppl_gen10 = results[10]["perplexity"]
    ppl_increase = (ppl_gen10 - ppl_gen1) / ppl_gen1
    cond3_pass = ppl_increase > 0.20
    cond3_detail = f"ppl_gen1={ppl_gen1:.4f}, ppl_gen10={ppl_gen10:.4f}, increase={ppl_increase*100:.2f}%"

    overall = cond1_pass and cond2_pass and cond3_pass

    return {
        "overall": "PASS" if overall else "FAIL",
        "condition_1_monotonic": {"pass": cond1_pass, "detail": cond1_detail},
        "condition_2_acceleration": {"pass": cond2_pass, "detail": cond2_detail},
        "condition_3_perplexity": {"pass": cond3_pass, "detail": cond3_detail},
    }


def calibrate_tau_k(results, baseline_gen=1, r_max=0.80, T=10):
    """Calibrate controller parameters tau and k from Phase 0 data."""
    deltas = compute_normalized_deltas(results, baseline_gen)

    # τ: median of Δ_t where downstream metrics haven't severely degraded
    # Use first half deltas as "healthy" region
    healthy_deltas = deltas[:4]  # Gen 2-5 relative to baseline
    tau = float(np.median(healthy_deltas)) if healthy_deltas else 0.0

    # k: target max allocation ~0.70 (headroom to r_max=0.80)
    max_delta = max(deltas) if deltas else 0.0
    # uniform_share at start of Phase 1: B_rem/remaining = r_base ≈ 0.50
    uniform_share = 0.50  # B_rem/(T-t) when budget is fresh
    target_max = 0.70  # leave headroom below r_max=0.80
    headroom = target_max - uniform_share  # ≈ 0.20
    k = headroom / max(max_delta - tau, 1e-6)

    return {
        "tau": float(tau),
        "k": float(k),
        "max_delta": float(max_delta),
        "healthy_deltas": healthy_deltas,
        "all_deltas": deltas,
        "notes": {
            "tau_method": "median of Gen 2-5 normalized deltas (healthy region)",
            "k_method": f"target max_allocation=0.70, uniform_share={uniform_share:.4f}, r_max={r_max}",
        }
    }


def plot_analysis(results, deltas, tau, save_path):
    """Generate 4-panel analysis figure."""
    gens = [r["generation"] for r in results]
    ranks = [r["effective_rank"] for r in results]
    ppls = [r["perplexity"] for r in results]
    has_hellaswag = "hellaswag_acc_norm" in results[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Phase 0 Spectral Collapse Analysis", fontsize=14, fontweight='bold')

    # Panel 1: Effective rank vs generation
    ax = axes[0, 0]
    ax.plot(gens, ranks, 'o-', color='#2196F3', linewidth=2, markersize=6)
    ax.axhline(ranks[1], color='gray', linestyle='--', alpha=0.5, label=f'Gen 1 baseline ({ranks[1]:.1f})')
    ax.axvspan(1, 5, alpha=0.08, color='green', label='First half (Gen 1-5)')
    ax.axvspan(5, 10, alpha=0.08, color='red', label='Second half (Gen 5-10)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Effective Rank vs Generation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Per-generation normalized delta + τ line
    ax = axes[0, 1]
    delta_gens = list(range(2, 2 + len(deltas)))
    ax.bar(delta_gens, deltas, color='#FF9800', alpha=0.7, edgecolor='#F57C00')
    ax.axhline(tau, color='red', linestyle='--', linewidth=2, label=f'τ = {tau:.4f}')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Normalized Δ_t')
    ax.set_title('Per-Generation Decline Rate (Δ_t)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Perplexity vs generation
    ax = axes[1, 0]
    ax.plot(gens, ppls, 's-', color='#E91E63', linewidth=2, markersize=6)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity vs Generation')
    ax.grid(True, alpha=0.3)

    # Panel 4: HellaSwag or distinct-4
    ax = axes[1, 1]
    if has_hellaswag:
        hellaswag = [r["hellaswag_acc_norm"] for r in results]
        ax.plot(gens, hellaswag, 'D-', color='#4CAF50', linewidth=2, markersize=6)
        ax.set_ylabel('HellaSwag acc_norm')
        ax.set_title('HellaSwag Accuracy vs Generation')
    elif "distinct_4" in results[0]:
        d4 = [r["distinct_4"] for r in results]
        ax.plot(gens, d4, 'D-', color='#9C27B0', linewidth=2, markersize=6)
        ax.set_ylabel('Distinct-4')
        ax.set_title('Distinct-4 vs Generation')
    else:
        ax.text(0.5, 0.5, 'No downstream metric available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('Downstream Metric')
    ax.set_xlabel('Generation')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")


def print_report(model_name, decline, go_nogo, calibration):
    """Print formatted analysis report."""
    print("\n" + "="*60)
    print(f"  PHASE 0 ANALYSIS REPORT — {model_name}")
    print("="*60)

    print("\n[1] DECLINE RATES (baseline: Gen 1)")
    print(f"    First half  (Gen 1→5, skip 5→6): {decline['first_half_rate']:.4f} rank/gen")
    print(f"    Second half (Gen 6→10): {decline['second_half_rate']:.4f} rank/gen")
    print(f"    Acceleration ratio:     {decline['acceleration_ratio']:.4f}x")

    print("\n[2] GO/NO-GO DECISION")
    print(f"    Overall: {'✓ PASS' if go_nogo['overall'] == 'PASS' else '✗ FAIL'}")
    for key in ["condition_1_monotonic", "condition_2_acceleration", "condition_3_perplexity"]:
        cond = go_nogo[key]
        status = "✓" if cond["pass"] else "✗"
        print(f"    {status} {key}: {cond['detail']}")

    print("\n[3] τ/k CALIBRATION")
    print(f"    τ (threshold):  {calibration['tau']:.6f}")
    print(f"    k (gain):       {calibration['k']:.4f}")
    print(f"    max Δ_t:        {calibration['max_delta']:.6f}")
    print(f"    Method: {calibration['notes']['tau_method']}")

    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 0 Analysis")
    parser.add_argument("--results-path", default="results/phase0_results.json")
    parser.add_argument("--model-name", default="410M")
    parser.add_argument("--output-dir", default="results/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    results = load_results(args.results_path)
    print(f"Loaded {len(results)} generations from {args.results_path}")

    # 2. Decline rates
    decline = compute_decline_rates(results, baseline_gen=1)

    # 3. Go/No-go
    go_nogo = check_go_nogo(results, baseline_gen=1)

    # 4. Calibrate tau/k
    calibration = calibrate_tau_k(results, baseline_gen=1)
    deltas = calibration["all_deltas"]
    tau = calibration["tau"]

    # 5. Print report
    print_report(args.model_name, decline, go_nogo, calibration)

    # 6. Plot
    plot_path = os.path.join(args.output_dir, "phase0_analysis_plots.png")
    plot_analysis(results, deltas, tau, plot_path)

    # 7. Save report JSON
    report = {
        "model_name": args.model_name,
        "source_file": args.results_path,
        "num_generations": len(results),
        "decline_rates": decline,
        "go_nogo": go_nogo,
        "calibration": calibration,
    }
    report_path = os.path.join(args.output_dir, "phase0_analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
