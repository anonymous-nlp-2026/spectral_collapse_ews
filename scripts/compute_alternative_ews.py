"""Compute alternative EWS metrics across gen_0..gen_10 checkpoints.

Input:  Pythia-410M checkpoints in ./checkpoints/gen_{0..10}/
        Probe data in ./data/eval_5k.pt (first 1000 samples)
Output: ./results/alternative_ews_comparison.json
        Comparison table printed to stdout
"""
import argparse, json, time, os, sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig

CKPT_DIR = "./checkpoints"
DATA_PATH = "./data/eval_5k.pt"
OUT_PATH = "./results/alternative_ews_comparison.json"
NUM_GENS = 11
PROBE_N = 1000
EMB_SAMPLE = 500
GRAD_BATCH = 32

# SLV baseline from exp_001
SLV_BASELINE = {
    0: 2366.68, 1: None, 2: 2469.57, 3: 2440.57, 4: None,
    5: None, 6: None, 7: None, 8: None, 9: None, 10: 2178.95
}


def load_probe_data(n=PROBE_N):
    data = torch.load(DATA_PATH, map_location="cpu")
    return data[:n]


def compute_isotropy(model, probe_data, device, batch_size=16):
    """Avg pairwise cosine similarity of last-layer hidden states (N=500 random tokens)."""
    model.eval()
    all_hidden = []
    for i in range(0, len(probe_data), batch_size):
        batch = probe_data[i:i+batch_size]
        input_ids = torch.stack([s["input_ids"] for s in batch]).to(device)
        attn_mask = torch.stack([s["attention_mask"] for s in batch]).to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        last_h = out.hidden_states[-1]  # (B, seq_len, hidden)
        mask = attn_mask.unsqueeze(-1).float()
        tokens = (last_h * mask).view(-1, last_h.shape[-1])
        valid = attn_mask.view(-1).bool()
        all_hidden.append(tokens[valid].cpu())
        if sum(h.shape[0] for h in all_hidden) >= EMB_SAMPLE:
            break
    all_hidden = torch.cat(all_hidden, dim=0)
    idx = torch.randperm(all_hidden.shape[0])[:EMB_SAMPLE]
    sample = all_hidden[idx]
    sample_norm = F.normalize(sample, dim=1)
    cos_sim = sample_norm @ sample_norm.T
    n = cos_sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    avg_cos = cos_sim[mask].mean().item()
    return avg_cos


def compute_token_entropy(model, probe_data, device, batch_size=16):
    """Shannon entropy of predicted token distribution (argmax over logits on probe data)."""
    model.eval()
    all_preds = []
    for i in range(0, min(len(probe_data), 200), batch_size):
        batch = probe_data[i:i+batch_size]
        input_ids = torch.stack([s["input_ids"] for s in batch]).to(device)
        attn_mask = torch.stack([s["attention_mask"] for s in batch]).to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        preds = logits.argmax(dim=-1)
        valid = attn_mask.bool()
        all_preds.append(preds[valid].cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    counts = np.bincount(all_preds, minlength=model.config.vocab_size)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def compute_spectral_norm(model):
    """Max singular value of last-layer attention.dense and mlp.dense_4h_to_h."""
    layers = model.gpt_neox.layers
    last = layers[-1]
    w_attn = last.attention.dense.weight.data.float()
    w_ffn = last.mlp.dense_4h_to_h.weight.data.float()
    sv_attn = torch.linalg.svdvals(w_attn)[0].item()
    sv_ffn = torch.linalg.svdvals(w_ffn)[0].item()
    return {"attn_dense_sigma1": sv_attn, "ffn_down_sigma1": sv_ffn}


def compute_stable_rank(model):
    """Stable rank = ||W||_F^2 / sigma_1^2 for last-layer weights."""
    layers = model.gpt_neox.layers
    last = layers[-1]
    results = {}
    for name, param in [("attn_dense", last.attention.dense.weight),
                         ("ffn_down", last.mlp.dense_4h_to_h.weight)]:
        w = param.data.float()
        svs = torch.linalg.svdvals(w)
        frob_sq = (svs ** 2).sum().item()
        sigma1_sq = (svs[0] ** 2).item()
        results[f"{name}_stable_rank"] = frob_sq / sigma1_sq
    return results


def compute_gradient_norm(model, probe_data, device):
    """L2 norm of all-parameter gradients on a small batch."""
    model.train()
    batch = probe_data[:GRAD_BATCH]
    input_ids = torch.stack([s["input_ids"] for s in batch]).to(device)
    attn_mask = torch.stack([s["attention_mask"] for s in batch]).to(device)
    labels = input_ids.clone()
    model.zero_grad()
    out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
    out.loss.backward()
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.float().norm(2).item() ** 2
    return total_norm ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ckpt_dir", default=CKPT_DIR, help="Checkpoint directory")
    parser.add_argument("--out_path", default=OUT_PATH, help="Output JSON path")
    parser.add_argument("--skip-gradient", action="store_true")
    parser.add_argument("--skip-forward", action="store_true",
                        help="Skip forward-pass metrics (isotropy, entropy)")
    args = parser.parse_args()
    device = torch.device(args.device)

    probe_data = load_probe_data()
    print(f"Loaded {len(probe_data)} probe samples, device={device}")

    results = {}
    for gen in range(NUM_GENS):
        ckpt_path = os.path.join(args.ckpt_dir, f"gen_{gen}")
        if not os.path.isdir(ckpt_path):
            print(f"SKIP gen_{gen}: not found")
            continue
        print(f"\n{'='*60}")
        print(f"Processing gen_{gen}")
        print(f"{'='*60}")

        gen_results = {"gen": gen}
        timings = {}

        # Load model
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, dtype=torch.float32, device_map=device,
        )
        model.eval()
        timings["load"] = time.time() - t0
        print(f"  Model loaded in {timings['load']:.1f}s")

        # 1) Spectral norm (no forward pass needed)
        t0 = time.time()
        sn = compute_spectral_norm(model)
        timings["spectral_norm"] = time.time() - t0
        gen_results.update(sn)
        print(f"  Spectral norm: attn={sn['attn_dense_sigma1']:.4f}, ffn={sn['ffn_down_sigma1']:.4f} ({timings['spectral_norm']:.1f}s)")

        # 2) Stable rank
        t0 = time.time()
        sr = compute_stable_rank(model)
        timings["stable_rank"] = time.time() - t0
        gen_results.update(sr)
        print(f"  Stable rank: attn={sr['attn_dense_stable_rank']:.2f}, ffn={sr['ffn_down_stable_rank']:.2f} ({timings['stable_rank']:.1f}s)")

        if not args.skip_forward:
            # 3) Isotropy
            t0 = time.time()
            iso = compute_isotropy(model, probe_data, device)
            timings["isotropy"] = time.time() - t0
            gen_results["isotropy"] = iso
            print(f"  Isotropy (avg cos sim): {iso:.4f} ({timings['isotropy']:.1f}s)")

            # 4) Token frequency entropy
            t0 = time.time()
            ent = compute_token_entropy(model, probe_data, device)
            timings["token_entropy"] = time.time() - t0
            gen_results["token_entropy"] = ent
            print(f"  Token entropy: {ent:.4f} ({timings['token_entropy']:.1f}s)")

        # 5) Gradient norm
        if not args.skip_gradient:
            try:
                t0 = time.time()
                gn = compute_gradient_norm(model, probe_data, device)
                timings["gradient_norm"] = time.time() - t0
                gen_results["gradient_norm"] = gn
                print(f"  Gradient norm: {gn:.4f} ({timings['gradient_norm']:.1f}s)")
            except Exception as e:
                gen_results["gradient_norm"] = None
                gen_results["gradient_norm_error"] = str(e)
                print(f"  Gradient norm FAILED: {e}")
        else:
            gen_results["gradient_norm"] = None
            gen_results["gradient_norm_skipped"] = True

        gen_results["timings"] = timings
        results[f"gen_{gen}"] = gen_results

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  Total time: {sum(timings.values()):.1f}s")

    # Save JSON
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")

    # Print comparison table
    print_comparison_table(results)


def print_comparison_table(results):
    gens = sorted([int(k.split("_")[1]) for k in results.keys()])
    metrics = [
        ("isotropy", "Isotropy (avg cos)", True),   # higher = more collapsed
        ("token_entropy", "Token entropy", False),    # lower = more collapsed
        ("attn_dense_sigma1", "Spectral norm (attn)", True),
        ("ffn_down_sigma1", "Spectral norm (ffn)", True),
        ("attn_dense_stable_rank", "Stable rank (attn)", False),  # lower = more collapsed
        ("ffn_down_stable_rank", "Stable rank (ffn)", False),
        ("gradient_norm", "Gradient norm", None),
    ]

    print(f"\n{'='*120}")
    print("COMPARISON TABLE: Alternative EWS Metrics")
    print(f"{'='*120}")

    # Raw values table
    header = f"{'Metric':<25}" + "".join(f"{'Gen '+str(g):>10}" for g in gens)
    print(header)
    print("-" * len(header))

    for key, label, higher_is_collapse in metrics:
        vals = []
        for g in gens:
            v = results.get(f"gen_{g}", {}).get(key)
            vals.append(v)
        row = f"{label:<25}"
        for v in vals:
            if v is None:
                row += f"{'N/A':>10}"
            elif abs(v) > 100:
                row += f"{v:>10.2f}"
            else:
                row += f"{v:>10.4f}"
        print(row)

    # SLV baseline
    row = f"{'SLV (log_det)':<25}"
    for g in gens:
        v = SLV_BASELINE.get(g)
        row += f"{v:>10.2f}" if v else f"{'N/A':>10}"
    print(row)

    # Analysis summary
    print(f"\n{'='*120}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*120}")
    print(f"{'Metric':<25} {'Peak Gen':>10} {'Mono post-peak?':>16} {'Detection lag':>14} {'Rel change pk->G10':>20} {'Avg s/ckpt':>12}")
    print("-" * 97)

    for key, label, higher_is_collapse in metrics:
        vals = {}
        times = {}
        for g in gens:
            v = results.get(f"gen_{g}", {}).get(key)
            if v is not None:
                vals[g] = v
            t = results.get(f"gen_{g}", {}).get("timings", {})
            relevant_time = t.get(key.replace("attn_dense_sigma1", "spectral_norm")
                                   .replace("ffn_down_sigma1", "spectral_norm")
                                   .replace("attn_dense_stable_rank", "stable_rank")
                                   .replace("ffn_down_stable_rank", "stable_rank"), 0)
            times[g] = relevant_time

        if not vals:
            print(f"{label:<25} {'N/A':>10} {'N/A':>16} {'N/A':>14} {'N/A':>20} {'N/A':>12}")
            continue

        if higher_is_collapse is True:
            peak_gen = max(vals, key=vals.get)
        elif higher_is_collapse is False:
            peak_gen = min(vals, key=vals.get)
        else:
            peak_gen = max(vals, key=vals.get)

        peak_val = vals[peak_gen]
        g10_val = vals.get(10)

        # Monotonicity post-peak
        post_peak = [(g, vals[g]) for g in sorted(vals) if g > peak_gen]
        if len(post_peak) >= 2:
            if higher_is_collapse is True:
                mono_violations = sum(1 for i in range(1, len(post_peak))
                                     if post_peak[i][1] > post_peak[i-1][1])
            elif higher_is_collapse is False:
                mono_violations = sum(1 for i in range(1, len(post_peak))
                                     if post_peak[i][1] < post_peak[i-1][1])
            else:
                mono_violations = 0
            mono = "Yes" if mono_violations <= 1 else f"No ({mono_violations} viol)"
        else:
            mono = "N/A"

        # Detection lag: first gen after gen 2 where direction changes
        det_lag = "N/A"
        for g in sorted(vals):
            if g <= 2:
                continue
            prev_g = g - 1
            if prev_g in vals:
                if higher_is_collapse is True and vals[g] < vals[prev_g]:
                    det_lag = str(g)
                    break
                elif higher_is_collapse is False and vals[g] > vals[prev_g]:
                    det_lag = str(g)
                    break

        # Rel change
        if g10_val is not None and peak_val != 0:
            rel_change = abs(peak_val - g10_val) / abs(peak_val)
            rel_str = f"{rel_change:.4f}"
        else:
            rel_str = "N/A"

        avg_time = np.mean(list(times.values())) if times else 0
        print(f"{label:<25} {'Gen '+str(peak_gen):>10} {mono:>16} {det_lag:>14} {rel_str:>20} {avg_time:>11.1f}s")

    # SLV baseline analysis
    slv_known = {k: v for k, v in SLV_BASELINE.items() if v is not None}
    slv_peak = max(slv_known, key=slv_known.get)
    slv_rel = abs(slv_known[slv_peak] - slv_known[10]) / abs(slv_known[slv_peak])
    print(f"{'SLV (log_det)':<25} {'Gen '+str(slv_peak):>10} {'Yes':>16} {'3':>14} {slv_rel:>20.4f} {'~2.0':>12}s")


if __name__ == "__main__":
    main()
