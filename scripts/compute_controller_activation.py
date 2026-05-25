import json
import os

RESULTS_DIR = "./results"

def load_log_det(path):
    with open(path) as f:
        data = json.load(f)
    return [(d["generation"], d["log_det"]) for d in data]

def compute_activation(gen_slv, alpha=0.5, tau=0.002):
    raw_slv = [s for _, s in gen_slv]
    gens = [g for g, _ in gen_slv]

    ema = [raw_slv[0]]
    for i in range(1, len(raw_slv)):
        ema.append(alpha * raw_slv[i] + (1 - alpha) * ema[-1])

    s0_abs = abs(ema[0])
    delta_t = [None]
    active = [None]
    for t in range(1, len(ema)):
        d = (ema[t - 1] - ema[t]) / s0_abs
        delta_t.append(d)
        active.append(d > tau)

    return {
        "generations": gens,
        "raw_slv": raw_slv,
        "ema_slv": ema,
        "delta_t": delta_t,
        "active": active,
    }

r0_data = load_log_det(os.path.join(RESULTS_DIR, "phase0_results.json"))
r02_data = load_log_det(os.path.join(RESULTS_DIR, "phase1_fixed_mix_r02_seed42_results.json"))
r05_data = load_log_det(os.path.join(RESULTS_DIR, "phase1_adaptive_k10_seed42_results.json"))

alpha = 0.5
tau = 0.002

r0 = compute_activation(r0_data, alpha, tau)
r02 = compute_activation(r02_data, alpha, tau)
r05 = compute_activation(r05_data, alpha, tau)

output = {
    "tau": tau,
    "alpha": alpha,
    "r0": {k: v for k, v in r0.items()},
    "r02": {k: v for k, v in r02.items()},
    "r05": {k: v for k, v in r05.items()},
}

out_path = os.path.join(RESULTS_DIR, "controller_activation_analysis.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"JSON saved to {out_path}\n")

# Print table
max_gen = max(len(r0["generations"]), len(r02["generations"]), len(r05["generations"]))

def fmt_delta(val):
    if val is None:
        return "   ---   "
    return f"{val:+.6f}"

def fmt_active(val):
    if val is None:
        return " --- "
    return "  ✓  " if val else "  ✗  "

header = f"{'Gen':>3} | {'r=0 Δ_t':>11} | {'act?':>5} | {'r=0.2 Δ_t':>11} | {'act?':>5} | {'r=0.5 Δ_t':>11} | {'act?':>5}"
sep = "-" * len(header)
print(sep)
print(header)
print(sep)

for i in range(max_gen):
    r0_d = r0["delta_t"][i] if i < len(r0["delta_t"]) else None
    r0_a = r0["active"][i] if i < len(r0["active"]) else None
    r02_d = r02["delta_t"][i] if i < len(r02["delta_t"]) else None
    r02_a = r02["active"][i] if i < len(r02["active"]) else None
    r05_d = r05["delta_t"][i] if i < len(r05["delta_t"]) else None
    r05_a = r05["active"][i] if i < len(r05["active"]) else None

    gen = i
    print(f"{gen:>3} | {fmt_delta(r0_d):>11} | {fmt_active(r0_a):>5} | {fmt_delta(r02_d):>11} | {fmt_active(r02_a):>5} | {fmt_delta(r05_d):>11} | {fmt_active(r05_a):>5}")

print(sep)

# Summary
def summarize(name, res):
    active_gens = [res["generations"][i] for i in range(len(res["active"])) if res["active"][i] is True]
    total = len([x for x in res["active"] if x is not None])
    n_active = len(active_gens)
    print(f"\n{name}: {n_active}/{total} generations active")
    if active_gens:
        print(f"  Active at: Gen {active_gens}")
    else:
        print(f"  Never activated")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
summarize("r=0   (pure collapse)", r0)
summarize("r=0.2 (fixed mix)", r02)
summarize("r=0.5 (adaptive)", r05)

r0_active = sum(1 for x in r0["active"] if x is True)
r02_active = sum(1 for x in r02["active"] if x is True)
r05_active = sum(1 for x in r05["active"] if x is True)

print("\n" + "=" * 50)
print("SIGNAL ATTENUATION PARADOX ASSESSMENT")
print("=" * 50)
print(f"r=0:   {r0_active} gens active  (expected: sustained activation)")
print(f"r=0.2: {r02_active} gens active  (expected: brief/transient)")
print(f"r=0.5: {r05_active} gens active  (expected: never)")

if r0_active > r02_active >= r05_active:
    print("\n→ Pattern SUPPORTS signal attenuation paradox:")
    print("  Higher mixing ratio → fewer activations → controller becomes blind")
    if r05_active == 0:
        print("  r=0.5 never triggers → complete signal attenuation confirmed")
else:
    print("\n→ Pattern does NOT cleanly support the paradox. Review thresholds.")
