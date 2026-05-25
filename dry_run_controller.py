"""Dry-run: simulate the fixed SpectralController over Phase 0 data (Gen 0-8).

Loads phase0_results.json and replays through the controller to verify:
1. Gen 0 baseline correctly initialized
2. Signal triggers at expected generations
3. Budget depletes correctly
4. All ratios are physically realizable (no truncation to bounds)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from src.controller import SpectralController

PHASE0_RESULTS = "./results/phase0_results.json"
TOTAL_GENS = 10
R_BASE = 0.50
R_MIN = 0.20
R_MAX = 0.80
K = 1.0
TAU = 0.005
ALPHA_EMA = 0.5
SIGNAL = "log_det"
TOTAL_PER_GEN = 50000


def main():
    with open(PHASE0_RESULTS) as f:
        phase0_data = json.load(f)

    print(f"Loaded Phase 0 data: {len(phase0_data)} generations (Gen 0-{len(phase0_data)-1})")
    print(f"Controller params: T={TOTAL_GENS}, r_base={R_BASE}, r_min={R_MIN}, r_max={R_MAX}, k={K}, tau={TAU}, alpha={ALPHA_EMA}, signal={SIGNAL}")
    print(f"Total budget: {TOTAL_GENS * R_BASE}")
    print(f"Total per gen: {TOTAL_PER_GEN} (real pool=50K)")
    print()

    total_budget = TOTAL_GENS * R_BASE
    ctrl = SpectralController(
        total_budget=total_budget,
        total_generations=TOTAL_GENS,
        r_base=R_BASE,
        r_min=R_MIN,
        r_max=R_MAX,
        k=K,
        tau=TAU,
        alpha_ema=ALPHA_EMA,
        signal=SIGNAL,
    )

    # Bug #2 fix: set baseline from Gen 0
    gen0 = phase0_data[0]
    ctrl.set_baseline(gen0)
    print(f"Baseline set: S_tilde_0 = {ctrl.S_tilde_0:.4f} (Gen 0 {SIGNAL})")
    print()

    # Gen 1: first gen ratio
    r1 = ctrl.get_first_gen_ratio()
    num_real_1 = int(TOTAL_PER_GEN * r1)
    num_synth_1 = TOTAL_PER_GEN - num_real_1
    actual_r1 = num_real_1 / TOTAL_PER_GEN
    ctrl.report_actual(actual_r1)

    print("=" * 110)
    print(f"{'Gen':<5} {'S_t':<12} {'S_tilde':<12} {'delta':<10} {'triggered':<10} {'r_raw':<8} {'r_clip':<8} {'r_actual':<9} {'B_rem':<8} {'real':<7} {'synth':<7}")
    print("-" * 110)
    print(f"{'1':<5} {'(no input)':<12} {'-':<12} {'-':<10} {'-':<10} {r1:<8.4f} {r1:<8.4f} {actual_r1:<9.4f} {ctrl.B_rem:<8.4f} {num_real_1:<7} {num_synth_1:<7}")

    # Gen 2-10: use update with previous gen's metrics
    for gen in range(2, TOTAL_GENS + 1):
        prev_gen_idx = gen - 1  # metrics from gen-1 to decide gen's ratio
        if prev_gen_idx >= len(phase0_data):
            print(f"\n[!] No Phase 0 data for Gen {prev_gen_idx} -- extrapolating with linear trend")
            last_log_det = phase0_data[-1]["log_det"]
            second_last = phase0_data[-2]["log_det"]
            trend = last_log_det - second_last
            extrapolated = last_log_det + trend * (prev_gen_idx - len(phase0_data) + 1)
            prev_metrics = {"effective_rank": 450.0, "log_det": extrapolated}
        else:
            prev_metrics = phase0_data[prev_gen_idx]

        r = ctrl.update(prev_metrics)
        num_real = int(TOTAL_PER_GEN * r)
        num_synth = TOTAL_PER_GEN - num_real
        actual_r = num_real / TOTAL_PER_GEN
        ctrl.report_actual(actual_r)

        hist = ctrl.history[-1]
        delta_str = f"{hist['delta']:.6f}" if hist['delta'] is not None else "-"
        trig_str = "YES" if hist.get('triggered', False) else "no"
        r_raw_str = f"{hist.get('r_raw', r):.4f}"

        print(f"{gen:<5} {hist['S_t']:<12.4f} {hist['S_tilde']:<12.4f} {delta_str:<10} {trig_str:<10} {r_raw_str:<8} {r:<8.4f} {actual_r:<9.4f} {ctrl.B_rem:<8.4f} {num_real:<7} {num_synth:<7}")

    print("=" * 110)
    print(f"\nFinal B_rem: {ctrl.B_rem:.6f}")
    print(f"Total ratio allocated: {total_budget - ctrl.B_rem:.4f} / {total_budget:.4f}")

    # Check physical realizability
    print("\n--- Physical Realizability Check ---")
    all_ok = True
    for h in ctrl.history:
        r_actual = h.get("r_actual", h["r"])
        num_real = int(TOTAL_PER_GEN * r_actual)
        if num_real > 50000:
            print(f"  [FAIL] Gen {h['t']}: requires {num_real} real samples > pool (50K)")
            all_ok = False
    if all_ok:
        print("  [PASS] All ratios physically realizable (num_real <= 50K for all gens)")

    # Delta distribution for tau selection
    deltas = [h["delta"] for h in ctrl.history if h["delta"] is not None and h["delta"] > 0]
    if deltas:
        print(f"\n--- Positive Delta Distribution (for tau selection) ---")
        print(f"  Count: {len(deltas)}")
        print(f"  Min: {min(deltas):.6f}")
        print(f"  Max: {max(deltas):.6f}")
        print(f"  Mean: {sum(deltas)/len(deltas):.6f}")
        print(f"  Current tau: {TAU}")
        if min(deltas) > TAU:
            print(f"  -> tau={TAU} triggers on ALL positive deltas. Good sensitivity.")
        else:
            print(f"  -> Some deltas below tau={TAU}, not triggering.")


if __name__ == "__main__":
    main()
