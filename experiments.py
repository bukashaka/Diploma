"""
experiments.py — Three-Series Experiment Suite
================================================
Series 1 — Mobility:   vary max_speed 0 → 40 m/s  (Table 3.3–3.5)
Series 2 — Scale:      vary num_nodes 20 → 100     (Table 3.6–3.7)
Series 3 — Load:       vary num_flows 5 → 30       (Section 3.6)

Each (protocol, config) point is averaged over num_seeds independent runs.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import scipy.stats as st

from environment import SimConfig
from simulation  import run_simulation, pretrain_ql, pretrain_dqn

PROTOCOLS = ['aodv', 'dsdv', 'ql', 'dqn']
LABELS    = {'aodv': 'AODV', 'dsdv': 'DSDV',
             'ql':  'QL-Routing', 'dqn': 'DQN-Routing'}


# ── Single (protocol, config, seeds) evaluation ───────────────────────────────

def run_config(
    protocol:   str,
    cfg:        SimConfig,
    num_seeds:  int = 10,
    base_seed:  int = 100,
    ql_router   = None,
    dqn_router  = None,
) -> Dict:
    """
    Evaluate one (protocol, config) point over num_seeds seeds.
    Returns {metric_mean, metric_std, metric_ci, …} for every metric.
    """
    results = []
    for s in range(num_seeds):
        r = run_simulation(
            protocol, cfg, base_seed + s,
            ql_router  = ql_router,
            dqn_router = dqn_router,
        )
        results.append(r)

    keys = results[0].keys()
    out  = {}
    for k in keys:
        vals = [r[k] for r in results]
        mean = float(np.mean(vals))
        std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci   = float(st.t.ppf(0.975, df=max(1, len(vals) - 1)) * std /
                     np.sqrt(len(vals)))
        out[f'{k}_mean'] = mean
        out[f'{k}_std']  = std
        out[f'{k}_ci']   = ci
    return out


def _run_all_protos(cfg, num_seeds, ql_r, dqn_r) -> Dict[str, Dict]:
    out = {}
    for p in PROTOCOLS:
        t0 = time.time()
        out[p] = run_config(p, cfg, num_seeds,
                            ql_router=ql_r, dqn_router=dqn_r)
        dt = time.time() - t0
        print(f'      {LABELS[p]:12s}  '
              f'PDR={out[p]["pdr_mean"]:5.1f}±{out[p]["pdr_ci"]:4.1f}%  '
              f'delay={out[p]["avg_delay_ms_mean"]:6.1f} ms  '
              f'NRO={out[p]["nro_mean"]:4.2f}  '
              f'({dt:.0f}s)')
    return out


# ── Series 1 — Mobility ───────────────────────────────────────────────────────

def run_series_1(
    base_cfg:  SimConfig,
    speeds:    Optional[List[float]] = None,
    num_seeds: int = 10,
    ql_r   = None,
    dqn_r  = None,
) -> Dict:
    """PDR / delay / NRO vs maximum node speed."""
    if speeds is None:
        speeds = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    data = {}
    for v in speeds:
        print(f'  v_max = {v} m/s')
        cfg      = SimConfig(**{**base_cfg.__dict__, 'max_speed': float(v)})
        data[v]  = _run_all_protos(cfg, num_seeds, ql_r, dqn_r)
    return data


# ── Series 2 — Scalability ────────────────────────────────────────────────────

def run_series_2(
    base_cfg:    SimConfig,
    node_counts: Optional[List[int]] = None,
    num_seeds:   int = 10,
    device:      str = 'cpu',
) -> Dict:
    """PDR / delay vs number of nodes (re-trains RL for each size)."""
    if node_counts is None:
        node_counts = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    data = {}
    for N in node_counts:
        print(f'  N = {N} nodes')
        cfg    = SimConfig(**{**base_cfg.__dict__,
                              'num_nodes': N,
                              'max_speed': 10.0})
        # RL agents must be rebuilt for each node count (different state dim)
        print('    pre-training QL …', end=' ', flush=True)
        t0 = time.time()
        ql_r2  = pretrain_ql(cfg, seed=0)
        print(f'{time.time()-t0:.0f}s')

        print('    pre-training DQN …', end=' ', flush=True)
        t0 = time.time()
        dqn_r2 = pretrain_dqn(cfg, seed=0, device=device)
        print(f'{time.time()-t0:.0f}s')

        data[N] = _run_all_protos(cfg, num_seeds, ql_r2, dqn_r2)
    return data


# ── Series 3 — Traffic load ───────────────────────────────────────────────────

def run_series_3(
    base_cfg:    SimConfig,
    flow_counts: Optional[List[int]] = None,
    num_seeds:   int = 10,
    ql_r  = None,
    dqn_r = None,
) -> Dict:
    """PDR / delay / NRO vs number of CBR flows."""
    if flow_counts is None:
        flow_counts = [5, 10, 15, 20, 25, 30]
    data = {}
    for F in flow_counts:
        print(f'  {F} CBR flows')
        cfg    = SimConfig(**{**base_cfg.__dict__,
                              'num_flows': F,
                              'max_speed': 10.0})
        data[F] = _run_all_protos(cfg, num_seeds, ql_r, dqn_r)
    return data


# ── Table formatting ──────────────────────────────────────────────────────────

METRIC_LABELS = {
    'pdr':            ('PDR (%)',            '{:5.1f} ± {:4.1f}'),
    'avg_delay_ms':   ('Delay (ms)',         '{:6.1f} ± {:5.1f}'),
    'nro':            ('NRO',                '{:5.2f} ± {:4.2f}'),
    'avg_hops':       ('Avg Hops',           '{:5.2f} ± {:4.2f}'),
    'throughput_bps': ('Throughput (bit/s)', '{:8.0f} ± {:6.0f}'),
    'avg_energy_j':   ('Energy (J)',         '{:5.1f} ± {:4.1f}'),
}


def print_table(data: Dict, x_values, factor_label: str = 'Factor'):
    """Print a human-readable results table to stdout."""
    col_w = 22
    hdr   = f'  {factor_label:>10}' + ''.join(
        f'  {LABELS[p]:>{col_w}}' for p in PROTOCOLS)

    for metric, (mname, fmt) in METRIC_LABELS.items():
        print(f'\n  ─── {mname} ───')
        print(hdr)
        for x in sorted(x_values):
            row = f'  {str(x):>10}'
            for p in PROTOCOLS:
                mn = data[x][p].get(f'{metric}_mean', 0.0)
                ci = data[x][p].get(f'{metric}_ci',   0.0)
                row += '  ' + fmt.format(mn, ci).rjust(col_w)
            print(row)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_results(data: Dict, path: str):
    def _cvt(o):
        if isinstance(o, dict):
            return {str(k): _cvt(v) for k, v in o.items()}
        if isinstance(o, (float, np.floating)):
            return round(float(o), 5)
        if isinstance(o, (int, np.integer)):
            return int(o)
        return o
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_cvt(data), f, indent=2, ensure_ascii=False)
    print(f'  Saved → {path}')


def load_results(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # Convert string keys back to int/float where appropriate
    def _restore(o):
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                try:
                    nk = int(k)
                except ValueError:
                    try:
                        nk = float(k)
                    except ValueError:
                        nk = k
                out[nk] = _restore(v)
            return out
        return o
    return _restore(raw)
