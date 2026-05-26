"""Budget-Constrained Spectral Controller.

Adaptive real-data ratio controller based on spectral health signals.
Core formula: r_{t+1}^raw = B_rem/(T-t) + k·max(0, Δ_t - τ)

signal_mode:
  per_gen:    Δ_t = (S̃_{t-1} - S̃_t) / |S̃_0|  (EMA per-generation change)
  cumulative: Δ_t = max(0, (S_0 - S_t) / |S_0|) (total drift from baseline)
"""


class SpectralController:

    def __init__(self, total_budget, total_generations, r_base=0.50,
                 r_min=0.20, r_max=0.80, k=10.0, tau=0.002, alpha_ema=0.5,
                 signal="log_det", signal_mode="per_gen"):
        self.B_rem = total_budget
        self.T = total_generations
        self.r_base = r_base
        self.r_min = r_min
        self.r_max = r_max
        self.k = k
        self.tau = tau
        self.alpha_ema = alpha_ema
        self.signal = signal
        self.signal_mode = signal_mode
        self.S_tilde_prev = None
        self.S_tilde_0 = None
        self.t = 0
        self.history = []

    def set_baseline(self, spectral_metrics):
        key = {"eff_rank": "effective_rank"}.get(self.signal, self.signal)
        S_0 = spectral_metrics[key]
        self.S_tilde_prev = S_0
        self.S_tilde_0 = S_0

    def update(self, spectral_metrics):
        key = {"eff_rank": "effective_rank"}.get(self.signal, self.signal)
        S_t = spectral_metrics[key]

        if self.signal_mode == "cumulative":
            S_tilde = S_t
        else:
            if self.S_tilde_prev is None:
                S_tilde = S_t
                self.S_tilde_0 = S_t
            else:
                S_tilde = self.alpha_ema * S_t + (1 - self.alpha_ema) * self.S_tilde_prev

        self.t += 1
        remaining_gens = self.T - self.t

        if remaining_gens <= 0:
            r = max(0, min(self.r_max, self.B_rem))
            self.S_tilde_prev = S_tilde
            self.history.append({
                "t": self.t, "S_t": S_t, "S_tilde": S_tilde,
                "delta": None, "r": r, "B_rem": self.B_rem - r,
                "triggered": False
            })
            self.B_rem = max(0, self.B_rem - r)
            return r

        if self.signal_mode == "cumulative":
            if self.S_tilde_0 is not None and self.S_tilde_0 != 0:
                delta = max(0, (self.S_tilde_0 - S_t) / abs(self.S_tilde_0))
            else:
                delta = 0.0
        else:
            if self.S_tilde_prev is not None and self.S_tilde_0 != 0:
                delta = (self.S_tilde_prev - S_tilde) / abs(self.S_tilde_0)
            else:
                delta = 0.0

        triggered = delta > self.tau

        r_raw = self.B_rem / remaining_gens + self.k * max(0, delta - self.tau)

        upper_clip = min(self.r_max, self.B_rem - (remaining_gens - 1) * self.r_min)
        r = max(self.r_min, min(r_raw, upper_clip))

        self.S_tilde_prev = S_tilde
        self.history.append({
            "t": self.t, "S_t": S_t, "S_tilde": S_tilde,
            "delta": delta, "r_raw": r_raw, "r": r,
            "B_rem": self.B_rem - r, "triggered": triggered
        })
        self.B_rem -= r
        return r

    def report_actual(self, actual_ratio):
        if not self.history:
            return
        last = self.history[-1]
        allocated = last["r"]
        correction = allocated - actual_ratio
        self.B_rem += correction
        last["r_actual"] = actual_ratio
        last["B_rem"] = self.B_rem

    def get_first_gen_ratio(self):
        r = self.r_base
        self.B_rem -= r
        self.history.append({
            "t": 0, "S_t": None, "S_tilde": None,
            "delta": None, "r_raw": r, "r": r,
            "B_rem": self.B_rem, "triggered": False
        })
        return r

    def get_state(self):
        return {
            "t": self.t,
            "B_rem": self.B_rem,
            "S_tilde_prev": self.S_tilde_prev,
            "S_tilde_0": self.S_tilde_0,
            "signal": self.signal,
            "signal_mode": self.signal_mode,
            "history": self.history
        }
