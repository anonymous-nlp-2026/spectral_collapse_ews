#!/usr/bin/env python3
"""Compute second-order EWS (Δ²) for spectral collapse analysis."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("./results")
FIGURES_DIR = Path("./figures")

ALPHA1 = 0.5  # EMA for raw SLV
ALPHA2 = 0.3  # EMA for first-order delta

EXPERIMENTS = {
    "r0": {
        "path": RESULTS_DIR / "phase0_results.json",
        "label": "r = 0 (pure collapse)",
        "color": "#c0392b",
    },
    "r02": {
        "path": RESULTS_DIR / "phase1_fixed_mix_r02_seed42_results.json",
        "label": "r = 0.2 (fixed mix)",
        "color": "#e67e22",
    },
    "r05": {
        "path": RESULTS_DIR / "phase1_adaptive_k10_seed42_results.json",
        "label": "r = 0.5 (adaptive)",
        "color": "#2980b9",
    },
}


def load_slv(path):
    with open(path) as f:
        data = json.load(f)
    data.sort(key=lambda x: x["generation"])
    gens = [d["generation"] for d in data]
    slv = [d["log_det"] for d in data]
    return gens, np.array(slv)


def compute_ews(slv, alpha1=ALPHA1, alpha2=ALPHA2):
    n = len(slv)

    # EMA-smoothed SLV
    ema_slv = np.zeros(n)
    ema_slv[0] = slv[0]
    for t in range(1, n):
        ema_slv[t] = alpha1 * slv[t] + (1 - alpha1) * ema_slv[t - 1]

    # First-order delta: Δ_t = S̃_{t-1} - S̃_t
    delta1 = np.zeros(n)
    delta1[0] = 0.0
    for t in range(1, n):
        delta1[t] = ema_slv[t - 1] - ema_slv[t]

    # EMA-smoothed first-order delta
    ema_delta1 = np.zeros(n)
    ema_delta1[0] = delta1[0]
    for t in range(1, n):
        ema_delta1[t] = alpha2 * delta1[t] + (1 - alpha2) * ema_delta1[t - 1]

    # Second-order delta: Δ²_t = Δ̃_{t-1} - Δ̃_t
    delta2 = np.zeros(n)
    delta2[0] = 0.0
    for t in range(1, n):
        delta2[t] = ema_delta1[t - 1] - ema_delta1[t]

    return ema_slv, delta1, ema_delta1, delta2


def main():
    results = {}
    all_data = {}

    for key, exp in EXPERIMENTS.items():
        gens, slv = load_slv(exp["path"])
        ema_slv, delta1, ema_delta1, delta2 = compute_ews(slv)

        results[key] = {
            "generations": gens,
            "raw_slv": slv.tolist(),
            "ema_slv": ema_slv.tolist(),
            "delta1": delta1.tolist(),
            "ema_delta1": ema_delta1.tolist(),
            "delta2": delta2.tolist(),
        }
        all_data[key] = {
            "gens": gens,
            "slv": slv,
            "ema_slv": ema_slv,
            "delta1": delta1,
            "ema_delta1": ema_delta1,
            "delta2": delta2,
            "label": exp["label"],
            "color": exp["color"],
        }

    # Save JSON
    out_path = RESULTS_DIR / "second_order_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    # Print key numerical results
    print("\n" + "=" * 70)
    print(f"SECOND-ORDER EWS ANALYSIS (alpha1={ALPHA1}, alpha2={ALPHA2})")
    print("=" * 70)

    for key in ["r0", "r02", "r05"]:
        d = all_data[key]
        print(f"\n--- {d['label']} (Gen 0-{d['gens'][-1]}) ---")
        print(f"  Raw SLV:     {['%.2f' % v for v in d['slv']]}")
        print(f"  EMA SLV:     {['%.2f' % v for v in d['ema_slv']]}")
        print(f"  Delta1:      {['%.2f' % v for v in d['delta1']]}")
        print(f"  EMA Delta1:  {['%.4f' % v for v in d['ema_delta1']]}")
        print(f"  Delta2:      {['%.4f' % v for v in d['delta2']]}")

        # Analyze patterns
        d2 = d["delta2"]
        neg_streak = 0
        max_neg_streak = 0
        for v in d2[1:]:
            if v > 0:  # positive delta2 means decline is accelerating
                neg_streak += 1
                max_neg_streak = max(max_neg_streak, neg_streak)
            else:
                neg_streak = 0
        
        consecutive_positive = sum(1 for v in d2[1:] if v > 0)
        consecutive_negative = sum(1 for v in d2[1:] if v < 0)
        print(f"  Delta2 > 0 (decline accelerating): {consecutive_positive}/{len(d2)-1}")
        print(f"  Delta2 < 0 (decline decelerating): {consecutive_negative}/{len(d2)-1}")

    # === PLOT ===
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": False,
        "figure.dpi": 150,
    })

    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5))
    fig.suptitle(
        f"Second-Order EWS Analysis ($\\alpha_1$={ALPHA1}, $\\alpha_2$={ALPHA2})",
        fontsize=12, fontweight="bold", y=0.98,
    )

    row_keys = ["r0", "r02", "r05"]
    col_titles = ["Raw SLV (log det)", "First-Order $\\Delta_t$", "Second-Order $\\Delta^2_t$"]

    for i, key in enumerate(row_keys):
        d = all_data[key]
        gens = d["gens"]
        color = d["color"]

        # Column 0: Raw SLV + EMA
        ax = axes[i, 0]
        ax.plot(gens, d["slv"], "o-", color=color, markersize=4, linewidth=1.2, label="Raw")
        ax.plot(gens, d["ema_slv"], "s--", color=color, markersize=3, linewidth=1.0, alpha=0.6, label="EMA")
        ax.set_ylabel(d["label"], fontsize=8, fontweight="bold")
        if i == 0:
            ax.set_title(col_titles[0], fontsize=9)
            ax.legend(fontsize=7, frameon=False)

        # Column 1: First-order delta
        ax = axes[i, 1]
        ax.bar(gens[1:], d["delta1"][1:], color=color, alpha=0.7, width=0.6)
        ax.plot(gens[1:], d["ema_delta1"][1:], "o-", color="black", markersize=3, linewidth=1.0, label="EMA $\\Delta$")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
        if i == 0:
            ax.set_title(col_titles[1], fontsize=9)
            ax.legend(fontsize=7, frameon=False)

        # Column 2: Second-order delta
        ax = axes[i, 2]
        colors_bar = ["#27ae60" if v < 0 else "#e74c3c" for v in d["delta2"][1:]]
        ax.bar(gens[1:], d["delta2"][1:], color=colors_bar, alpha=0.8, width=0.6)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
        if i == 0:
            ax.set_title(col_titles[2], fontsize=9)

        # X-axis label only on bottom row
        if i == 2:
            for j in range(3):
                axes[i, j].set_xlabel("Generation", fontsize=8)

        # Set integer x-ticks
        for j in range(3):
            axes[i, j].set_xticks(gens)
            axes[i, j].tick_params(labelsize=7)

    # Add color legend for delta2
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.8, label="$\\Delta^2 > 0$: decline accelerating"),
        Patch(facecolor="#27ae60", alpha=0.8, label="$\\Delta^2 < 0$: decline decelerating"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    pdf_path = FIGURES_DIR / "second_order_ews_analysis.pdf"
    png_path = FIGURES_DIR / "second_order_ews_analysis.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close()

    # === KEY FINDINGS ===
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Q1: r=0.5 sustained negative delta2?
    d05 = all_data["r05"]
    d2_05 = d05["delta2"]
    gens_05 = d05["gens"]
    print(f"\nQ1: r=0.5 sustained negative Delta2 (growth rate decline)?")
    late_d2 = [(g, v) for g, v in zip(gens_05[1:], d2_05[1:]) if g >= 5]
    if late_d2:
        signs = ["+" if v > 0 else "-" for _, v in late_d2]
        print(f"  Gen 5+: Delta2 signs = {signs}")
        print(f"  Values: {['%.4f' % v for _, v in late_d2]}")
    else:
        print(f"  (No data from Gen 5+)")
    all_signs_05 = [(g, "+" if v > 0 else "-") for g, v in zip(gens_05[1:], d2_05[1:])]
    print(f"  Full sequence: {all_signs_05}")

    # Q2: r=0.2 pattern
    d02 = all_data["r02"]
    d2_02 = d02["delta2"]
    gens_02 = d02["gens"]
    print(f"\nQ2: r=0.2 Delta2 pattern (dip-then-recovery)?")
    all_signs_02 = [(g, "+" if v > 0 else "-", f"{v:.4f}") for g, v in zip(gens_02[1:], d2_02[1:])]
    print(f"  Full sequence: {all_signs_02}")

    # Q3: r=0 pattern
    d0 = all_data["r0"]
    d2_0 = d0["delta2"]
    gens_0 = d0["gens"]
    print(f"\nQ3: r=0 Delta2 pattern?")
    all_signs_0 = [(g, "+" if v > 0 else "-", f"{v:.4f}") for g, v in zip(gens_0[1:], d2_0[1:])]
    print(f"  Full sequence: {all_signs_0}")


if __name__ == "__main__":
    main()
