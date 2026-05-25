"""Compute alternative EWS metrics across gen_0..gen_10 checkpoints (v2).

Changes from v1:
- Dynamic SLV baseline from --results_file instead of hardcoded
- RankMe on embeddings (embedding-level spectral metric)
- PPL growth rate from results JSON
"""
import argparse, json, time, os, sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig

DATA_PATH = "./data/eval_5k.pt"
NUM_GENS = 11
PROBE_N = 1000
EMB_SAMPLE = 500
RANKME_SAMPLE = 2000
GRAD_BATCH = 32


def load_results_baseline(results_file):
    """Load SLV (log_det) and PPL from results JSON."""
    with open(results_file) as f:
        data = json.load(f)
    slv = {}
    ppl = {}
    for entry in data:
        g = entry["generation"]
        slv[g] = entry.get("log_det")
        ppl[g] = entry.get("perplexity")
    return slv, ppl


def compute_ppl_growth_rate(ppl_by_gen):
    """d(ppl)/dg = ppl[g] - ppl[g-1]."""
    rates = {}
    gens = sorted(ppl_by_gen.keys())
    for i, g in enumerate(gens):
        if i == 0:
            rates[g] = 0.0
        else:
            prev = gens[i - 1]
            if ppl_by_gen[g] is not None and ppl_by_gen[prev] is not None:
                rates[g] = ppl_by_gen[g] - ppl_by_gen[prev]
            else:
                rates[g] = None
    return rates


def load_probe_data(n=PROBE_N):
    data = torch.load(DATA_PATH, map_location="cpu")
    return data[:n]


def collect_embeddings(model, probe_data, device, n_tokens, batch_size=16):
    """Forward pass to collect last-layer hidden states."""
    model.eval()
    all_hidden = []
    for i in range(0, len(probe_data), batch_size):
        batch = probe_data[i:i+batch_size]
        input_ids = torch.stack([s["input_ids"] for s in batch]).to(device)
        attn_mask = torch.stack([s["attention_mask"] for s in batch]).to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        last_h = out.hidden_states[-1]
        mask = attn_mask.unsqueeze(-1).float()
        tokens = (last_h * mask).view(-1, last_h.shape[-1])
        valid = attn_mask.view(-1).bool()
        all_hidden.append(tokens[valid].cpu())
        if sum(h.shape[0] for h in all_hidden) >= n_tokens:
            break
    all_hidden = torch.cat(all_hidden, dim=0)
    return all_hidden


def compute_isotropy(embeddings, n_sample=EMB_SAMPLE):
    """Avg pairwise cosine similarity."""
    idx = torch.randperm(embeddings.shape[0])[:n_sample]
    sample = embeddings[idx]
    sample_norm = F.normalize(sample, dim=1)
    cos_sim = sample_norm @ sample_norm.T
    n = cos_sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    return cos_sim[mask].mean().item()


def compute_rankme(embeddings, n_sample=RANKME_SAMPLE):
    """RankMe = exp(H(p)) where p_i = sigma_i / sum(sigma_j)."""
    idx = torch.randperm(embeddings.shape[0])[:n_sample]
    E = embeddings[idx].float()
    svs = torch.linalg.svdvals(E)
    svs = svs[svs > 1e-12]
    p = svs / svs.sum()
    entropy = -(p * torch.log(p)).sum().item()
    return float(np.exp(entropy))


def compute_token_entropy(model, probe_data, device, batch_size=16):
    """Shannon entropy of predicted token distribution."""
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
    layers = model.gpt_neox.layers
    last = layers[-1]
    w_attn = last.attention.dense.weight.data.float()
    w_ffn = last.mlp.dense_4h_to_h.weight.data.float()
    sv_attn = torch.linalg.svdvals(w_attn)[0].item()
    sv_ffn = torch.linalg.svdvals(w_ffn)[0].item()
    return {"attn_dense_sigma1": sv_attn, "ffn_down_sigma1": sv_ffn}


def compute_stable_rank(model):
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
    parser.add_argument("--ckpt_dir", required=True, help="Checkpoint directory with gen_0..gen_10")
    parser.add_argument("--results_file", required=True, help="Results JSON for SLV/PPL baseline")
    parser.add_argument("--out_path", required=True, help="Output JSON path")
    parser.add_argument("--skip-gradient", action="store_true")
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)

    slv_baseline, ppl_baseline = load_results_baseline(args.results_file)
    ppl_growth = compute_ppl_growth_rate(ppl_baseline)

    probe_data = load_probe_data()
    print(f"Loaded {len(probe_data)} probe samples, device={device}")
    print(f"SLV baseline loaded: {len(slv_baseline)} generations")
    print(f"PPL growth rates: {json.dumps({k: round(v, 4) if v else v for k, v in sorted(ppl_growth.items())})}")

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

        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, dtype=torch.float32, device_map=device,
        )
        model.eval()
        timings["load"] = time.time() - t0
        print(f"  Model loaded in {timings['load']:.1f}s")

        # 1) Spectral norm
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
            # Collect embeddings once for isotropy + RankMe
            t0 = time.time()
            embs = collect_embeddings(model, probe_data, device, max(EMB_SAMPLE, RANKME_SAMPLE))
            timings["embeddings"] = time.time() - t0
            print(f"  Embeddings collected: {embs.shape[0]} tokens ({timings['embeddings']:.1f}s)")

            # 3) Isotropy
            t0 = time.time()
            iso = compute_isotropy(embs)
            timings["isotropy"] = time.time() - t0
            gen_results["isotropy"] = iso
            print(f"  Isotropy (avg cos sim): {iso:.4f} ({timings['isotropy']:.1f}s)")

            # 4) RankMe on embeddings
            t0 = time.time()
            rm = compute_rankme(embs)
            timings["rankme"] = time.time() - t0
            gen_results["rankme"] = rm
            print(f"  RankMe: {rm:.2f} ({timings['rankme']:.1f}s)")

            # 5) Token frequency entropy
            t0 = time.time()
            ent = compute_token_entropy(model, probe_data, device)
            timings["token_entropy"] = time.time() - t0
            gen_results["token_entropy"] = ent
            print(f"  Token entropy: {ent:.4f} ({timings['token_entropy']:.1f}s)")

        # 6) Gradient norm
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

        # 7) SLV baseline + PPL growth rate (from results file, no computation)
        gen_results["slv_log_det"] = slv_baseline.get(gen)
        gen_results["ppl"] = ppl_baseline.get(gen)
        gen_results["ppl_growth_rate"] = ppl_growth.get(gen)

        gen_results["timings"] = timings
        results[f"gen_{gen}"] = gen_results

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  Total time: {sum(timings.values()):.1f}s")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out_path}")

    print_comparison_table(results)


def print_comparison_table(results):
    gens = sorted([int(k.split("_")[1]) for k in results.keys()])
    metrics = [
        ("isotropy", "Isotropy (avg cos)", True),
        ("rankme", "RankMe (embeddings)", False),
        ("token_entropy", "Token entropy", False),
        ("attn_dense_sigma1", "Spectral norm (attn)", True),
        ("ffn_down_sigma1", "Spectral norm (ffn)", True),
        ("attn_dense_stable_rank", "Stable rank (attn)", False),
        ("ffn_down_stable_rank", "Stable rank (ffn)", False),
        ("gradient_norm", "Gradient norm", None),
        ("slv_log_det", "SLV (log_det)", False),
        ("ppl", "Perplexity", True),
        ("ppl_growth_rate", "PPL growth rate", True),
    ]

    print(f"\n{'='*140}")
    print("COMPARISON TABLE: Alternative EWS Metrics (v2)")
    print(f"{'='*140}")

    header = f"{'Metric':<25}" + "".join(f"{'Gen '+str(g):>10}" for g in gens)
    print(header)
    print("-" * len(header))

    for key, label, _ in metrics:
        row = f"{label:<25}"
        for g in gens:
            v = results.get(f"gen_{g}", {}).get(key)
            if v is None:
                row += f"{'N/A':>10}"
            elif abs(v) > 100:
                row += f"{v:>10.2f}"
            else:
                row += f"{v:>10.4f}"
        print(row)

    print(f"\n{'='*140}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*140}")
    print(f"{'Metric':<25} {'Direction':>12} {'Peak Gen':>10} {'Mono post-G2?':>14} {'Det lag':>8} {'Rel chg pk->G10':>16} {'Avg s/ckpt':>12}")
    print("-" * 97)

    for key, label, higher_is_collapse in metrics:
        vals = {}
        times = {}
        for g in gens:
            v = results.get(f"gen_{g}", {}).get(key)
            if v is not None:
                vals[g] = v
            t = results.get(f"gen_{g}", {}).get("timings", {})
            tk = key.replace("attn_dense_sigma1", "spectral_norm") \
                     .replace("ffn_down_sigma1", "spectral_norm") \
                     .replace("attn_dense_stable_rank", "stable_rank") \
                     .replace("ffn_down_stable_rank", "stable_rank")
            times[g] = t.get(tk, 0)

        if not vals:
            print(f"{label:<25} {'N/A':>12} {'N/A':>10} {'N/A':>14} {'N/A':>8} {'N/A':>16} {'N/A':>12}")
            continue

        # Direction: compare gen_0 vs gen_10
        g0_val = vals.get(0)
        g10_val = vals.get(10)
        if g0_val is not None and g10_val is not None:
            if g10_val > g0_val * 1.01:
                direction = "UP"
            elif g10_val < g0_val * 0.99:
                direction = "DOWN"
            else:
                direction = "FLAT"
        else:
            direction = "N/A"

        if higher_is_collapse is True:
            peak_gen = max(vals, key=vals.get)
        elif higher_is_collapse is False:
            peak_gen = min(vals, key=vals.get)
        else:
            peak_gen = max(vals, key=vals.get)

        peak_val = vals[peak_gen]

        # Monotonicity post-gen2
        post_g2 = [(g, vals[g]) for g in sorted(vals) if g > 2]
        if len(post_g2) >= 2:
            if higher_is_collapse is True:
                mono_viol = sum(1 for i in range(1, len(post_g2))
                                if post_g2[i][1] > post_g2[i-1][1])
            elif higher_is_collapse is False:
                mono_viol = sum(1 for i in range(1, len(post_g2))
                                if post_g2[i][1] < post_g2[i-1][1])
            else:
                mono_viol = 0
            mono = "Yes" if mono_viol <= 1 else f"No({mono_viol}v)"
        else:
            mono = "N/A"

        # Detection lag
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
        print(f"{label:<25} {direction:>12} {'Gen '+str(peak_gen):>10} {mono:>14} {det_lag:>8} {rel_str:>16} {avg_time:>11.1f}s")


if __name__ == "__main__":
    main()
