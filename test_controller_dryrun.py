#!/usr/bin/env python3
"""Dry-run controller with Phase 0 spectral data to verify D011 fixes."""
import json
import sys
# sys.path adjusted for local use
from src.controller import SpectralController

# Phase 0 spectral metrics (Gen 0-8, from phase0_results.json)
phase0_data = [
    {"generation": 0, "effective_rank": 417.547, "log_det": 2366.684},
    {"generation": 1, "effective_rank": 491.768, "log_det": 2464.762},
    {"generation": 2, "effective_rank": 496.441, "log_det": 2469.928},
    {"generation": 3, "effective_rank": 493.315, "log_det": 2440.566},
    {"generation": 4, "effective_rank": 485.674, "log_det": 2405.050},
    {"generation": 5, "effective_rank": 479.207, "log_det": 2370.173},
    {"generation": 6, "effective_rank": 473.005, "log_det": 2335.165},
    {"generation": 7, "effective_rank": 464.199, "log_det": 2304.190},
    {"generation": 8, "effective_rank": 456.756, "log_det": 2252.119},
]

# Controller params: tau=0.002, k=10.0, r_min=0.2, r_max=0.8, signal=log_det
# total_budget = T * r_base = 10 * 0.5 = 5.0
T = 10
controller = SpectralController(
    total_budget=T * 0.5,
    total_generations=T,
    r_base=0.50,
    r_min=0.20,
    r_max=0.80,
    k=10.0,
    tau=0.002,
    alpha_ema=0.5,
    signal="log_det",
)

total_per_gen = 50000
real_pool = 50000

print(f"{'Gen':<4} {'S_tilde':<12} {'delta':<10} {'triggered':<10} {'r_req':<8} {'r_actual':<10} {'B_rem':<8} {'real':<8} {'synth':<8}")
print("-" * 90)

# Gen 0: set baseline, use get_first_gen_ratio
gen0_metrics = phase0_data[0]
controller.set_baseline(gen0_metrics)
r_first = controller.get_first_gen_ratio()
num_real_0 = int(total_per_gen * r_first)
num_synth_0 = total_per_gen - num_real_0
actual_0 = num_real_0 / total_per_gen
controller.report_actual(actual_0)
print(f"{0:<4} {gen0_metrics['log_det']:<12.3f} {'N/A':<10} {'N/A':<10} {r_first:<8.4f} {actual_0:<10.4f} {controller.B_rem:<8.4f} {num_real_0:<8} {num_synth_0:<8}")

# Gen 1-9: use update with previous gen's metrics
for i in range(1, min(len(phase0_data), T)):
    metrics = phase0_data[i]
    r_req = controller.update(metrics)
    num_real = min(int(total_per_gen * r_req), real_pool)
    num_synth = total_per_gen - num_real
    actual_ratio = num_real / total_per_gen
    controller.report_actual(actual_ratio)
    
    last_h = controller.history[-1]
    delta = last_h.get("delta", "N/A")
    triggered = last_h.get("triggered", False)
    s_tilde = last_h.get("S_tilde", "N/A")
    
    delta_str = f"{delta:.6f}" if isinstance(delta, float) else str(delta)
    s_tilde_str = f"{s_tilde:.3f}" if isinstance(s_tilde, float) else str(s_tilde)
    
    print(f"{i:<4} {s_tilde_str:<12} {delta_str:<10} {str(triggered):<10} {r_req:<8.4f} {actual_ratio:<10.4f} {controller.B_rem:<8.4f} {num_real:<8} {num_synth:<8}")

print("\n--- Controller final state ---")
state = controller.get_state()
print(f"t={state['t']}, B_rem={state['B_rem']:.4f}, S_tilde_0={state['S_tilde_0']:.3f}")
print(f"\nDelta distribution (excluding Gen 0):")
deltas = [h["delta"] for h in state["history"] if h["delta"] is not None]
if deltas:
    print(f"  min={min(deltas):.6f}, max={max(deltas):.6f}, mean={sum(deltas)/len(deltas):.6f}")
    print(f"  Threshold tau=0.002")
    triggered_count = sum(1 for d in deltas if d > 0.002)
    print(f"  Triggered: {triggered_count}/{len(deltas)} generations")
