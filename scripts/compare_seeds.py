"""
Seed comparison script for spectral collapse experiments.

Reads multiple results JSON files (one per seed), computes per-generation
comparison tables, phase boundary consistency checks, and mean+/-std summaries.

Input:  List of JSON file paths, each a list of per-generation dicts with keys
        generation, log_det, effective_rank, perplexity, distinct_4, ...
Output: Formatted tables to stdout + JSON to results/seed_comparison.json.

Usage:
    python compare_seeds.py path1.json path2.json [--labels seed42 seed43]
    python compare_seeds.py results/*.json --labels seed42 seed43 seed44
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path


METRICS = ["log_det", "effective_rank", "perplexity", "distinct_4"]
METRIC_LABELS = {"log_det": "SLV(log_det)", "effective_rank": "eff_rank",
                 "perplexity": "ppl", "distinct_4": "distinct4"}


def load_results(path):
    """Load a results JSON file and return list of per-generation dicts."""
    with open(path) as f:
        data = json.load(f)
    data.sort(key=lambda x: x["generation"])
    return data


def align_seeds(all_data):
    """Truncate all seed data to the shortest generation count."""
    min_len = min(len(d) for d in all_data)
    return [d[:min_len] for d in all_data], min_len


def print_per_gen_table(all_data, labels, n_gen):
    """Print per-generation comparison table with pairwise deltas."""
    n_seeds = len(labels)
    print("=" * 80)
    print("PER-GENERATION COMPARISON")
    print("=" * 80)

    for metric in METRICS:
        ml = METRIC_LABELS[metric]
        print(f"\n--- {ml} ---")

        header = f"{'Gen':>4}"
        for lb in labels:
            header += f"  {lb:>12}"
        if n_seeds >= 2:
            for i in range(n_seeds):
                for j in range(i + 1, n_seeds):
                    header += f"  {'D(' + labels[i][:4] + '-' + labels[j][:4] + ')':>14}"
                    header += f"  {'%diff':>7}"
        print(header)
        print("-" * len(header))

        for g in range(n_gen):
            vals = []
            for sd in all_data:
                vals.append(sd[g].get(metric, float("nan")))
            row = f"{g:>4}"
            for v in vals:
                row += f"  {v:>12.4f}"
            if n_seeds >= 2:
                for i in range(n_seeds):
                    for j in range(i + 1, n_seeds):
                        delta = vals[i] - vals[j]
                        avg = (vals[i] + vals[j]) / 2 if (vals[i] + vals[j]) != 0 else 1e-9
                        pct = delta / abs(avg) * 100
                        row += f"  {delta:>14.4f}"
                        row += f"  {pct:>6.2f}%"
            print(row)


def detect_phase_boundaries(data):
    """
    Detect three phase boundaries from SLV (log_det) trajectory.

    Returns dict with:
      decompression_peak: gen where SLV is maximal
      onset: first gen after peak where SLV decreases
      collapse: gen where SLV starts sustained decrease (3+ consecutive drops)
    """
    slvs = [d.get("log_det", float("nan")) for d in data]
    n = len(slvs)

    peak_gen = max(range(n), key=lambda i: slvs[i])
    peak_val = slvs[peak_gen]

    onset_gen = None
    for i in range(peak_gen + 1, n):
        if slvs[i] < slvs[i - 1]:
            onset_gen = i
            break

    collapse_gen = None
    for i in range(peak_gen, n - 2):
        if slvs[i] > slvs[i + 1] > slvs[i + 2]:
            collapse_gen = i
            break

    return {
        "decompression_peak": {"gen": peak_gen, "slv": peak_val},
        "onset": {"gen": onset_gen, "slv": slvs[onset_gen] if onset_gen is not None else None},
        "collapse": {"gen": collapse_gen, "slv": slvs[collapse_gen] if collapse_gen is not None else None},
    }


def print_phase_boundaries(all_data, labels):
    """Print phase boundary analysis and cross-seed consistency."""
    print("\n" + "=" * 80)
    print("PHASE BOUNDARY ANALYSIS")
    print("=" * 80)

    boundaries = []
    for i, (data, lb) in enumerate(zip(all_data, labels)):
        pb = detect_phase_boundaries(data)
        boundaries.append(pb)
        print(f"\n  [{lb}]")
        peak = pb["decompression_peak"]
        print(f"    Decompression peak : Gen {peak['gen']} (SLV = {peak['slv']:.2f})")
        onset = pb["onset"]
        if onset["gen"] is not None:
            print(f"    Onset (first drop) : Gen {onset['gen']} (SLV = {onset['slv']:.2f})")
        else:
            print(f"    Onset (first drop) : not detected (SLV never decreases after peak)")
        collapse = pb["collapse"]
        if collapse["gen"] is not None:
            print(f"    Collapse (sustained): Gen {collapse['gen']} (SLV = {collapse['slv']:.2f})")
        else:
            print(f"    Collapse (sustained): not detected (no 3+ consecutive drops)")

    if len(labels) >= 2:
        print(f"\n  CONSISTENCY CHECK:")
        for key, desc in [("decompression_peak", "Decompression peak"),
                          ("onset", "Onset"), ("collapse", "Collapse")]:
            gens = [b[key]["gen"] for b in boundaries]
            if all(g is not None for g in gens):
                if len(set(gens)) == 1:
                    print(f"    {desc:25s}: CONSISTENT (all at Gen {gens[0]})")
                else:
                    print(f"    {desc:25s}: DIVERGENT  {dict(zip(labels, gens))}")
            else:
                none_labels = [lb for lb, g in zip(labels, gens) if g is None]
                print(f"    {desc:25s}: INCOMPLETE (not detected for {none_labels})")

    return boundaries


def compute_mean_std(all_data, n_gen):
    """Compute per-gen mean and std across seeds for each metric."""
    summary = []
    for g in range(n_gen):
        entry = {"generation": g}
        for metric in METRICS:
            vals = [sd[g].get(metric, float("nan")) for sd in all_data]
            vals = [v for v in vals if not math.isnan(v)]
            if vals:
                mean = sum(vals) / len(vals)
                if len(vals) > 1:
                    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                    std = math.sqrt(var)
                else:
                    std = 0.0
                entry[metric + "_mean"] = mean
                entry[metric + "_std"] = std
            else:
                entry[metric + "_mean"] = float("nan")
                entry[metric + "_std"] = float("nan")
        summary.append(entry)
    return summary


def print_mean_std_table(summary):
    """Print mean +/- std table in Table 1 format."""
    print("\n" + "=" * 80)
    print("MEAN +/- STD SUMMARY (Table 1 format)")
    print("=" * 80)

    header = f"{'Gen':>4}"
    for metric in METRICS:
        ml = METRIC_LABELS[metric]
        header += f"  {ml:>22}"
    print(header)
    print("-" * len(header))

    for row in summary:
        g = row["generation"]
        line = f"{g:>4}"
        for metric in METRICS:
            m = row[metric + "_mean"]
            s = row[metric + "_std"]
            if math.isnan(m):
                line += f"  {'N/A':>22}"
            elif s == 0.0:
                line += f"  {m:>22.4f}"
            else:
                cell = f"{m:.2f} +/- {s:.2f}"
                line += f"  {cell:>22}"
            
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Compare results across seeds.")
    parser.add_argument("files", nargs="+", help="Paths to results JSON files")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Labels for each seed (default: file basenames)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results/seed_comparison.json)")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.files):
        print(f"Error: {len(args.labels)} labels for {len(args.files)} files", file=sys.stderr)
        sys.exit(1)

    all_data = []
    labels = []
    for i, fp in enumerate(args.files):
        data = load_results(fp)
        all_data.append(data)
        if args.labels:
            labels.append(args.labels[i])
        else:
            labels.append(Path(fp).stem)

    aligned, n_gen = align_seeds(all_data)
    if n_gen < len(all_data[0]):
        lengths = {lb: len(d) for lb, d in zip(labels, all_data)}
        print(f"NOTE: Aligned to {n_gen} generations (shortest). Original lengths: {lengths}\n")

    print_per_gen_table(aligned, labels, n_gen)
    boundaries = print_phase_boundaries(all_data, labels)
    summary = compute_mean_std(aligned, n_gen)
    print_mean_std_table(summary)

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(args.files[0]))),
        "results", "seed_comparison.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "labels": labels,
        "n_gen_aligned": n_gen,
        "per_gen_comparison": [],
        "phase_boundaries": {},
        "mean_std_summary": summary,
    }
    for g in range(n_gen):
        gen_entry = {"generation": g}
        for metric in METRICS:
            gen_entry[metric] = {lb: aligned[i][g].get(metric) for i, lb in enumerate(labels)}
        output["per_gen_comparison"].append(gen_entry)
    for lb, pb in zip(labels, boundaries):
        output["phase_boundaries"][lb] = pb

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON output saved to: {output_path}")


if __name__ == "__main__":
    main()
