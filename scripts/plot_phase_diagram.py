#!/usr/bin/env python3
"""Phase diagram: SLV trajectories across mixing ratios with type annotations."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGURES_DIR = Path('./figures')

generations = list(range(11))

data = {
    r'$r=0$':    [2366.68, 2464.76, 2469.93, 2440.57, 2405.05, 2370.17, 2335.17, 2304.19, 2252.12, 2207.57, 2178.95],
    r'$r=0.2$':  [2366.68, 2367.46, 2358.31, 2362.52, 2379.19, 2398.93, 2429.83, 2436.35, 2454.30, 2463.14, 2479.70],
    r'$r=0.25$': [2366.68, 2362.13, 2363.10, 2380.88, 2393.37, 2421.22, 2436.38, 2447.08, 2463.81, 2476.58, 2485.01],
    r'$r=0.3$':  [2366.68, 2366.13, 2372.45, 2386.72, 2404.02, 2431.17, 2452.00, 2467.05, 2485.91, 2497.30, 2511.13],
    r'$r=0.5$':  [2366.68, 2366.58, 2395.36, 2425.64, 2451.72, 2486.34, 2507.62, 2525.55, 2544.03, 2562.13, 2573.20],
}

styles = {
    r'$r=0$':    {'color': '#d62728', 'marker': 'o'},
    r'$r=0.2$':  {'color': '#ff7f0e', 'marker': 's'},
    r'$r=0.25$': {'color': '#bcbd22', 'marker': '^'},
    r'$r=0.3$':  {'color': '#2ca02c', 'marker': 'D'},
    r'$r=0.5$':  {'color': '#1f77b4', 'marker': 'v'},
}

S0 = 2366.68

def main():
    plt.rcParams.update({
        'font.size': 8,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.pad': 3,
        'ytick.major.pad': 3,
        'legend.fontsize': 6.5,
        'legend.framealpha': 0.92,
        'legend.edgecolor': '0.75',
        'legend.handlelength': 1.6,
        'legend.handletextpad': 0.5,
        'legend.borderpad': 0.4,
        'legend.labelspacing': 0.35,
    })

    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)

    gens = np.array(generations)

    for label, vals in data.items():
        s = styles[label]
        ax.plot(gens, vals,
                color=s['color'],
                marker=s['marker'],
                markersize=3.5,
                linewidth=1.5,
                markeredgecolor='white',
                markeredgewidth=0.3,
                label=label,
                zorder=3)

    # --- Annotation 1: Type labels ---
    ax.text(7.5, 2195, 'Type I', fontsize=7, fontstyle='italic',
            color='#d62728', ha='center', va='top', zorder=4)

    ax.text(8.5, 2440, 'Type II', fontsize=7, fontstyle='italic',
            color='#ff7f0e', ha='center', va='bottom', zorder=4)

    ax.text(7.0, 2555, 'Type III', fontsize=7, fontstyle='italic',
            color='#1f77b4', ha='center', va='bottom', zorder=4)

    # --- Annotation 2: r_crit shaded band ---
    y_025 = np.array(data[r'$r=0.25$'])
    y_030 = np.array(data[r'$r=0.3$'])
    ax.fill_between(gens, y_025, y_030, alpha=0.18, color='#888888',
                    zorder=1, linewidth=0)
    ax.text(8.0, 0.5*(y_025[8] + y_030[8]),
            r'$r_{\mathrm{crit}}$', fontsize=6.5,
            color='#555555', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7),
            zorder=4)

    # --- Annotation 3: Gen 0 baseline ---
    ax.axhline(y=S0, color='#999999', linestyle='--', linewidth=0.7,
               zorder=1, alpha=0.8)
    ax.text(10.2, S0 + 5, r'$S_0$', fontsize=6.5, color='#777777',
            ha='left', va='bottom', zorder=4)

    ax.set_xlabel('Generation', fontsize=8.5)
    ax.set_ylabel(r'$\mathrm{SLV}\;(\frac{1}{2}\log\det\mathbf{G})$', fontsize=8.5)
    ax.set_xlim(-0.3, 10.8)
    ax.set_xticks(range(0, 11))
    ax.set_ylim(2150, 2600)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='lower left', frameon=True, fancybox=False)

    ax.grid(True, alpha=0.12, linewidth=0.4)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'phase_diagram.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'phase_diagram.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {FIGURES_DIR}/phase_diagram.pdf')
    print(f'Saved: {FIGURES_DIR}/phase_diagram.png')


if __name__ == '__main__':
    main()
