"""
visualization.py — Results Plotting
=====================================
Generates publication-quality figures for all three experiment series.
Falls back gracefully if matplotlib is not installed.
"""
from __future__ import annotations

import os
from typing import Dict, List

PROTOCOLS = ['aodv', 'dsdv', 'ql', 'dqn']
LABELS    = {'aodv': 'AODV', 'dsdv': 'DSDV',
             'ql':  'QL-Routing', 'dqn': 'DQN-Routing'}
COLORS    = {'aodv': '#e74c3c', 'dsdv': '#3498db',
             'ql':   '#27ae60', 'dqn':  '#9b59b6'}
MARKERS   = {'aodv': 'o', 'dsdv': 's', 'ql': '^', 'dqn': 'D'}

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False


# ── Per-metric configuration ──────────────────────────────────────────────────
METRIC_CFG = {
    'pdr':            ('Packet Delivery Ratio (%)',     False),
    'avg_delay_ms':   ('End-to-End Delay (ms)',         False),
    'nro':            ('Normalised Routing Overhead',   False),
    'avg_energy_j':   ('Average Residual Energy (J)',   False),
    'avg_hops':       ('Average Hop Count',             False),
    'throughput_bps': ('Throughput (bit/s)',            False),
}


def _require_mpl(fn_name: str) -> bool:
    if not HAS_MPL:
        print(f'[visualization] matplotlib not available — skipping {fn_name}.')
        return False
    return True


def _plot_metric(ax, data: Dict, xs, metric: str,
                 xlabel: str, ylabel: str, title: str):
    for p in PROTOCOLS:
        ys   = [data[x][p].get(f'{metric}_mean', 0.0) for x in xs]
        errs = [data[x][p].get(f'{metric}_ci',   0.0) for x in xs]
        ax.errorbar(xs, ys, yerr=errs,
                    label=LABELS[p],
                    color=COLORS[p],
                    marker=MARKERS[p],
                    capsize=4, linewidth=2, markersize=7, markeredgewidth=1.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title,   fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3, linestyle='--')


def plot_series(
    data:      Dict,
    x_values:  List,
    xlabel:    str,
    out_dir:   str  = 'results',
    prefix:    str  = 'series',
    metrics:   List[str] = None,
):
    """
    Generate a grid of all metrics and individual per-metric PNGs.

    data     : {x_value: {protocol: {metric_mean/ci, …}}}
    x_values : ordered list of x-axis values
    """
    if not _require_mpl('plot_series'):
        return
    os.makedirs(out_dir, exist_ok=True)
    xs = sorted(x_values)
    if metrics is None:
        metrics = list(METRIC_CFG.keys())

    n    = len(metrics)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.5))
    axes      = axes.flatten()

    for i, m in enumerate(metrics):
        ylabel, _ = METRIC_CFG.get(m, (m, False))
        _plot_metric(axes[i], data, xs, m, xlabel, ylabel, ylabel)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'MANET Routing — {prefix}  ({xlabel})',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{prefix}_grid.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved grid plot  → {path}')

    # Individual plots
    for m in metrics:
        ylabel, _ = METRIC_CFG.get(m, (m, False))
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        _plot_metric(ax2, data, xs, m, xlabel, ylabel, ylabel)
        plt.tight_layout()
        p2 = os.path.join(out_dir, f'{prefix}_{m}.png')
        plt.savefig(p2, dpi=130, bbox_inches='tight')
        plt.close()
    print(f'  Individual plots saved to {out_dir}/')


def plot_convergence(losses: List[float], out_dir: str = 'results',
                     label: str = 'DQN', window: int = 50):
    """Plot training loss / reward convergence curve."""
    if not _require_mpl('plot_convergence') or not HAS_NP or not losses:
        return
    os.makedirs(out_dir, exist_ok=True)
    raw = np.array(losses, dtype=float)
    if len(raw) < window:
        window = max(1, len(raw))
    smooth = np.convolve(raw, np.ones(window) / window, mode='valid')

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(raw,    alpha=0.25, color='grey',      linewidth=0.8, label='raw')
    ax.plot(smooth, color='steelblue', linewidth=2, label=f'MA-{window}')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')
    ax.set_title(f'{label} Training Loss Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    path = os.path.join(out_dir, f'convergence_{label.lower().replace(" ", "_")}.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Convergence plot → {path}')


def plot_comparison_bar(
    data_point:   Dict[str, Dict],
    metric:       str,
    title:        str,
    out_dir:      str = 'results',
    filename:     str = 'bar.png',
):
    """Bar chart comparing all protocols at a single configuration point."""
    if not _require_mpl('plot_comparison_bar'):
        return
    os.makedirs(out_dir, exist_ok=True)
    means = [data_point[p].get(f'{metric}_mean', 0.0) for p in PROTOCOLS]
    cis   = [data_point[p].get(f'{metric}_ci',   0.0) for p in PROTOCOLS]
    cols  = [COLORS[p] for p in PROTOCOLS]
    lbls  = [LABELS[p] for p in PROTOCOLS]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(PROTOCOLS)), means, color=cols,
                  yerr=cis, capsize=5, edgecolor='white', linewidth=1.2)
    ax.set_xticks(range(len(PROTOCOLS)))
    ax.set_xticklabels(lbls, fontsize=11)
    ylabel, _ = METRIC_CFG.get(metric, (metric, False))
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(cis) * 0.1,
                f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Bar chart → {path}')
