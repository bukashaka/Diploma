"""
main.py — MANET RL Routing Experiment Entry Point
===================================================
Usage
-----
  python main.py                      # full run (all 3 series, 10 seeds)
  python main.py --quick              # fast demo (3 seeds, fewer points)
  python main.py --series 1           # only Series 1
  python main.py --series 1 3         # Series 1 and 3
  python main.py --no-plots           # skip matplotlib output
  python main.py --out results/       # custom output directory
  python main.py --device cuda        # use GPU for DQN

Quick-mode typical runtime: ~5–15 minutes on a modern CPU (no GPU).
Full-mode typical runtime:  ~2–6 hours.
"""
import argparse
import os
import sys
import time

from environment  import SimConfig
from simulation   import pretrain_ql, pretrain_dqn
from experiments  import (
    run_series_1, run_series_2, run_series_3,
    print_table, save_results, PROTOCOLS, LABELS,
)
from visualization import plot_series, plot_convergence


# ══════════════════════════════════════════════════════════════════════════════
# Configuration helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_cfg(quick: bool = False) -> SimConfig:
    """Return a SimConfig matching Table 3.1 (or a reduced quick version)."""
    return SimConfig(
        num_nodes       = 50,
        tx_range        = 250.0,
        area            = 1000.0,
        initial_energy  = 100.0,
        max_speed       = 20.0,
        pause_max       = 10.0,
        num_flows       = 10,
        flow_rate       = 4.0,
        sim_time        = 600.0 if not quick else 120.0,
        dt              = 0.1,
        max_wait_steps  = 60,
        ql_pretrain_ep  = 15 if not quick else 3,
        dqn_pretrain_ep = 25 if not quick else 5,
        pretrain_ep_s   = 60.0 if not quick else 20.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MANET RL Routing — Experiment Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--quick',   action='store_true',
        help='Fast demo: 3 seeds, fewer speed/node/flow points, shorter sim',
    )
    parser.add_argument(
        '--series',  nargs='+', type=int, default=[1, 2, 3],
        choices=[1, 2, 3], metavar='N',
        help='Which experiment series to run (1, 2, 3)',
    )
    parser.add_argument(
        '--seeds',   type=int, default=None,
        help='Number of random seeds (default: 10, or 3 with --quick)',
    )
    parser.add_argument(
        '--out',     type=str, default='results',
        help='Output directory for JSON results and PNG plots',
    )
    parser.add_argument(
        '--device',  type=str, default='cpu',
        help="PyTorch device for DQN ('cpu' or 'cuda')",
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip matplotlib figure generation',
    )
    args = parser.parse_args()

    quick     = args.quick
    num_seeds = args.seeds or (3 if quick else 10)
    cfg       = make_cfg(quick)
    out_dir   = args.out
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 65)
    print('  MANET RL Routing — Experiment Suite')
    print('=' * 65)
    print(f'  Mode      : {"Quick demo" if quick else "Full run"}')
    print(f'  Seeds     : {num_seeds}')
    print(f'  Nodes     : {cfg.num_nodes}')
    print(f'  Sim time  : {cfg.sim_time}s')
    print(f'  dt        : {cfg.dt}s')
    print(f'  Series    : {args.series}')
    print(f'  Device    : {args.device}')
    print(f'  Output    : {out_dir}/')
    print()

    # speed/node/flow grid (reduced for --quick)
    speeds_s1 = ([0, 10, 20, 30, 40]        if quick else
                 [0, 5, 10, 15, 20, 25, 30, 35, 40])
    nodes_s2  = ([20, 50, 100]               if quick else
                 [20, 30, 40, 50, 60, 70, 80, 90, 100])
    flows_s3  = ([5, 15, 30]                 if quick else
                 [5, 10, 15, 20, 25, 30])

    t_total = time.time()

    # ── Pre-train RL agents (base config) ─────────────────────────────────────
    ql_router = dqn_router = None

    if {1, 3}.intersection(args.series):
        print('── Pre-training RL agents (base config) ─────────────────────')
        print(f'   QL   : {cfg.ql_pretrain_ep} epochs × {cfg.pretrain_ep_s}s …',
              end=' ', flush=True)
        t0 = time.time()
        ql_router = pretrain_ql(cfg, seed=0)
        print(f'done ({time.time()-t0:.0f}s)')

        print(f'   DQN  : {cfg.dqn_pretrain_ep} epochs × {cfg.pretrain_ep_s}s …',
              end=' ', flush=True)
        t0 = time.time()
        dqn_router = pretrain_dqn(cfg, seed=0, device=args.device)
        print(f'done ({time.time()-t0:.0f}s)')

        # Collect DQN convergence data from one representative agent
        rep_agent = next(iter(dqn_router.agents.values()))
        if rep_agent.losses and not args.no_plots:
            plot_convergence(rep_agent.losses, out_dir, 'DQN-Routing')

    # ── Series 1 : Mobility ───────────────────────────────────────────────────
    if 1 in args.series:
        print('\n── Series 1 — Varying Node Speed ────────────────────────────')
        t0 = time.time()
        s1 = run_series_1(cfg, speeds_s1, num_seeds, ql_router, dqn_router)
        save_results(s1, os.path.join(out_dir, 'series1.json'))
        print_table(s1, speeds_s1, 'Speed (m/s)')
        if not args.no_plots:
            plot_series(s1, speeds_s1, 'Max Speed (m/s)',
                        out_dir, 'series1')
        print(f'  Series 1 completed in {time.time()-t0:.0f}s')

    # ── Series 2 : Scalability ────────────────────────────────────────────────
    if 2 in args.series:
        print('\n── Series 2 — Varying Number of Nodes ───────────────────────')
        t0 = time.time()
        s2 = run_series_2(cfg, nodes_s2, num_seeds, device=args.device)
        save_results(s2, os.path.join(out_dir, 'series2.json'))
        print_table(s2, nodes_s2, 'Num Nodes')
        if not args.no_plots:
            plot_series(s2, nodes_s2, 'Number of Nodes',
                        out_dir, 'series2')
        print(f'  Series 2 completed in {time.time()-t0:.0f}s')

    # ── Series 3 : Traffic Load ───────────────────────────────────────────────
    if 3 in args.series:
        print('\n── Series 3 — Varying Traffic Load ──────────────────────────')
        t0 = time.time()
        s3 = run_series_3(cfg, flows_s3, num_seeds, ql_router, dqn_router)
        save_results(s3, os.path.join(out_dir, 'series3.json'))
        print_table(s3, flows_s3, 'Num Flows')
        if not args.no_plots:
            plot_series(s3, flows_s3, 'Number of CBR Flows',
                        out_dir, 'series3')
        print(f'  Series 3 completed in {time.time()-t0:.0f}s')

    print(f'\nAll done in {time.time()-t_total:.0f}s.  Results in: {out_dir}/')


if __name__ == '__main__':
    main()
