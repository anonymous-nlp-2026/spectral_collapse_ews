import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(".")

data = {
    'generation': [0, 1, 2, 3, 4, 5, 6, 7],
    'eff_rank': [417.55, 491.77, 496.44, 493.31, 485.67, 479.21, 473.00, 464.20],
    'log_det': [2366.68, 2464.76, 2469.93, 2440.57, 2405.05, 2370.17, 2335.17, 2304.19],
    'ppl': [9.55, 11.38, 13.58, 16.25, 19.80, 24.59, 30.91, 38.89],
    'distinct4': [0.7414, 0.6256, 0.5154, 0.4900, 0.4679, 0.4630, 0.4655, 0.4474],
    'hellaswag': [0.4130, 0.3897, 0.3764, 0.3658, 0.3558, 0.3439],
}

try:
    with open(ROOT / "results" / "phase0_results.json") as f:
        records = json.load(f)
    if len(records) >= 1:
        data['generation'] = [r['generation'] for r in records]
        data['eff_rank'] = [r.get('effective_rank', 0) for r in records]
        data['log_det'] = [r.get('log_det', 0) for r in records]
        data['ppl'] = [r.get('perplexity', 0) for r in records]
        data['distinct4'] = [r.get('distinct_4', 0) for r in records]
except Exception:
    pass

gens = np.array(data['generation'])
eff_rank = np.array(data['eff_rank'])
log_det = np.array(data['log_det'])
ppl = np.array(data['ppl'])
distinct4 = np.array(data['distinct4'])
hellaswag = np.array(data['hellaswag'])
hella_gens = np.arange(len(hellaswag))

COLOR = '#1f77b4'
BASELINE_COLOR = '#888888'
PEAK_COLOR = '#d62728'
MARKER = 'o'
MS = 5
LW = 1.8

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

fig = plt.figure(figsize=(10, 6.5))
gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.38, wspace=0.8)

ax_rank = fig.add_subplot(gs[0, 0:2])
ax_logdet = fig.add_subplot(gs[0, 2:4])
ax_ppl = fig.add_subplot(gs[0, 4:6])
ax_dist = fig.add_subplot(gs[1, 0:3])
ax_hella = fig.add_subplot(gs[1, 3:6])

peak_idx = 2

def style_ax(ax):
    ax.grid(True, alpha=0.3, color='#cccccc')
    ax.set_xticks(gens)
    ax.set_xlabel('Generation')

# Panel (a): Effective Rank
ax_rank.plot(gens, eff_rank, color=COLOR, marker=MARKER, markersize=MS, linewidth=LW, zorder=3)
ax_rank.axvline(x=2, color=PEAK_COLOR, linestyle=':', alpha=0.6, linewidth=1.2)
ax_rank.annotate('decompression\npeak', xy=(2, eff_rank[peak_idx]),
                 xytext=(3.5, eff_rank[peak_idx] + 5),
                 fontsize=7.5, color=PEAK_COLOR, ha='left',
                 arrowprops=dict(arrowstyle='->', color=PEAK_COLOR, lw=1.0))
ax_rank.set_ylabel('Effective Rank (RankMe)')
ax_rank.set_ylim(440, 510)
style_ax(ax_rank)

# Panel (b): Log-Determinant
ax_logdet.plot(gens, log_det, color=COLOR, marker=MARKER, markersize=MS, linewidth=LW, zorder=3)
ax_logdet.axvline(x=2, color=PEAK_COLOR, linestyle=':', alpha=0.6, linewidth=1.2)
ax_logdet.annotate('decompression\npeak', xy=(2, log_det[peak_idx]),
                   xytext=(3.5, log_det[peak_idx] + 5),
                   fontsize=7.5, color=PEAK_COLOR, ha='left',
                   arrowprops=dict(arrowstyle='->', color=PEAK_COLOR, lw=1.0))
ax_logdet.set_ylabel('Log-Determinant')
ax_logdet.set_ylim(2280, 2500)
style_ax(ax_logdet)

# Panel (c): Perplexity
ax_ppl.plot(gens, ppl, color=COLOR, marker=MARKER, markersize=MS, linewidth=LW, zorder=3)
ax_ppl.set_ylabel('Held-out Perplexity')
ax_ppl.set_ylim(7, 42)
style_ax(ax_ppl)
ax_ppl.annotate('', xy=(gens[-1] - 0.2, ppl[-1] - 1), xytext=(0.2, ppl[0] + 1),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.0, linestyle='--'))

# Panel (d): Distinct-4
ax_dist.plot(gens, distinct4, color=COLOR, marker=MARKER, markersize=MS, linewidth=LW, zorder=3)
ax_dist.set_ylabel('Distinct-4 (Lexical Diversity)')
ax_dist.set_ylim(0.40, 0.80)
style_ax(ax_dist)

# Panel (e): HellaSwag (only available generations)
ax_hella.plot(hella_gens, hellaswag, color=COLOR, marker=MARKER, markersize=MS, linewidth=LW, zorder=3)
ax_hella.set_ylabel('HellaSwag Acc (norm)')
ax_hella.set_ylim(0.33, 0.43)
ax_hella.set_xticks(gens)
ax_hella.set_xlabel('Generation')
ax_hella.grid(True, alpha=0.3, color='#cccccc')
decline_pct = (hellaswag[0] - hellaswag[-1]) / hellaswag[0] * 100
ax_hella.annotate(f'{decline_pct:.1f}% decline', xy=(hella_gens[-1], hellaswag[-1]),
                  xytext=(3.5, 0.41),
                  fontsize=8, color=PEAK_COLOR,
                  arrowprops=dict(arrowstyle='->', color=PEAK_COLOR, lw=1.0))

panels = [(ax_rank, '(a)'), (ax_logdet, '(b)'), (ax_ppl, '(c)'),
          (ax_dist, '(d)'), (ax_hella, '(e)')]
for ax, label in panels:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

out_dir = ROOT / "paper" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "spectral_trajectory.pdf", bbox_inches='tight', dpi=300)
fig.savefig(out_dir / "spectral_trajectory.png", bbox_inches='tight', dpi=300)
plt.close()

print("Done. Saved PDF and PNG to", out_dir)
print(f"Data: Gen 0-{int(gens[-1])} ({len(gens)} generations)")
print(f"  eff_rank: {eff_rank.tolist()}")
print(f"  log_det: {log_det.tolist()}")
print(f"  ppl: {ppl.tolist()}")
print(f"  distinct4: {distinct4.tolist()}")
print(f"  hellaswag (Gen 0-{len(hellaswag)-1}): {hellaswag.tolist()}")
