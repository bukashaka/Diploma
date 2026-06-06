"""
simulation.py — Single-Run Simulation Engine
=============================================
run_simulation()  — executes one complete (protocol, cfg, seed) experiment.
pretrain_ql()     — pre-trains a QLRouter for cfg.ql_pretrain_ep epochs.
pretrain_dqn()    — pre-trains a DQNRouter for cfg.dqn_pretrain_ep epochs.

Design notes
------------
* Topology is rebuilt every cfg.dt seconds (= 0.1 s by default).
* Each active packet gets ONE routing attempt per simulation step.
* Packet delay (delay_ms) is accumulated hop-by-hop and is independent of the
  simulation time-step size, giving physically accurate end-to-end latencies.
* NRO is computed from the protocol's total_ctrl() counter, which tracks
  AODV RREQ/RREP/RERR, DSDV broadcasts, and RL ACK feedback uniformly.
"""
from __future__ import annotations

from typing import Dict, Optional

from environment import MANETEnvironment, SimConfig, Packet
from protocols   import AODVProtocol, DSDVProtocol
from agents      import (
    QLRouter, DQNRouter,
    make_ql_agents, make_dqn_agents, make_reward_calc,
)


# ══════════════════════════════════════════════════════════════════════════════
# Core runner
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    protocol_name: str,
    cfg:           SimConfig,
    seed:          int,
    ql_router:     Optional[QLRouter]  = None,
    dqn_router:    Optional[DQNRouter] = None,
    training_mode: bool = False,
    verbose:       bool = False,
) -> Dict:
    """
    Run one complete simulation experiment.

    Parameters
    ----------
    protocol_name : 'aodv' | 'dsdv' | 'ql' | 'dqn'
    cfg           : SimConfig with all simulation parameters
    seed          : random seed (topology + traffic)
    ql_router     : pre-trained QLRouter  (created fresh if None)
    dqn_router    : pre-trained DQNRouter (created fresh if None)
    training_mode : if True, RL agents update weights (used during pre-training)
    verbose       : print progress every 1000 steps

    Returns
    -------
    dict — all performance metrics (PDR, delay, NRO, …)
    """
    env = MANETEnvironment(cfg, seed)

    # ── initialise protocol ────────────────────────────────────────────────────
    if protocol_name == 'aodv':
        proto = AODVProtocol()

    elif protocol_name == 'dsdv':
        proto = DSDVProtocol()
        proto.force_recompute(env)      # build initial routing table

    elif protocol_name == 'ql':
        if ql_router is None:
            ql_router = QLRouter(make_ql_agents(cfg), make_reward_calc())
        if not training_mode:
            ql_router.soft_reset_epsilon()
        proto = ql_router

    elif protocol_name == 'dqn':
        if dqn_router is None:
            dqn_router = DQNRouter(make_dqn_agents(cfg), make_reward_calc())
        if not training_mode:
            dqn_router.soft_reset_epsilon()
        proto = dqn_router

    else:
        raise ValueError(f'Unknown protocol: {protocol_name!r}')

    pkt_waits: Dict[int, int] = {}     # pid → consecutive wait-steps counter
    ctrl_start = proto.total_ctrl() if hasattr(proto, 'total_ctrl') else 0.0

    # ── main loop ──────────────────────────────────────────────────────────────
    for step in range(cfg.total_steps):
        # 1. advance topology + generate new packets
        env.step()

        # 2. protocol periodic maintenance (DSDV broadcasts, AODV expiry)
        proto.periodic_update(env, cfg.dt)

        # 3. route all active packets (one hop attempt per packet per step)
        for pid, pkt in list(env.active.items()):
            if pkt.done:
                continue

            nxt, delay_ms, _ = proto.get_next_hop(env, pkt, pkt.current)

            if nxt is None:
                # no route: wait or timeout-drop
                pkt_waits[pid] = pkt_waits.get(pid, 0) + 1
                if pkt_waits[pid] >= cfg.max_wait_steps:
                    env.timeout_drop(pkt)
                    _post_hop(proto, env, pkt, protocol_name, training_mode)
                continue

            prev = pkt.current
            ok   = env.forward(pkt, nxt, delay_ms)
            if ok:
                pkt_waits.pop(pid, None)
                _post_hop(proto, env, pkt, protocol_name, training_mode)
            # If ok is False the link disappeared between get_next_hop and forward;
            # packet stays active and will retry next step.

        if verbose and step % 1000 == 0:
            m = env.metrics()
            print(f'    step {step:5d}/{cfg.total_steps}  '
                  f'PDR={m["pdr"]:.1f}%  delay={m["avg_delay_ms"]:.1f} ms')

    # 4. timeout-drop packets still active at end of simulation
    for pid, pkt in list(env.active.items()):
        env.timeout_drop(pkt)
        _post_hop(proto, env, pkt, protocol_name, training_mode)

    # ── compute metrics ────────────────────────────────────────────────────────
    m = env.metrics()
    n_recv = m['n_recv']
    if hasattr(proto, 'total_ctrl'):
        ctrl_generated = proto.total_ctrl() - ctrl_start
        m['nro'] = ctrl_generated / n_recv if n_recv > 0 else 0.0

    return m


def _post_hop(proto, env, pkt, proto_name, training_mode):
    """Trigger RL reward propagation after a hop (or at packet end)."""
    if training_mode or True:   # always update during evaluation too (online)
        if proto_name == 'ql' and isinstance(proto, QLRouter):
            proto.post_hop(env, pkt)
        elif proto_name == 'dqn' and isinstance(proto, DQNRouter):
            proto.post_hop(env, pkt)


# ══════════════════════════════════════════════════════════════════════════════
# Pre-training
# ══════════════════════════════════════════════════════════════════════════════

def pretrain_ql(cfg: SimConfig, seed: int = 0) -> QLRouter:
    """
    Pre-train QL agents for cfg.ql_pretrain_ep × cfg.pretrain_ep_s simulated
    seconds. Returns a warmed-up QLRouter ready for evaluation.
    """
    router = QLRouter(make_ql_agents(cfg), make_reward_calc())
    _run_pretrain(router, 'ql', cfg, seed)
    return router


def pretrain_dqn(cfg: SimConfig, seed: int = 0,
                 device: str = 'cpu') -> DQNRouter:
    """
    Pre-train DQN agents for cfg.dqn_pretrain_ep × cfg.pretrain_ep_s
    simulated seconds.
    """
    router = DQNRouter(make_dqn_agents(cfg, device=device), make_reward_calc())
    _run_pretrain(router, 'dqn', cfg, seed)
    return router


def _run_pretrain(proto, proto_name: str, cfg: SimConfig, base_seed: int):
    """Inner loop: run N epochs × pretrain_ep_s seconds each."""
    n_epochs = (cfg.ql_pretrain_ep if proto_name == 'ql'
                else cfg.dqn_pretrain_ep)
    ep_cfg   = SimConfig(**{**cfg.__dict__,
                            'sim_time': cfg.pretrain_ep_s,
                            'dt':       cfg.dt})
    for ep in range(n_epochs):
        _run_epoch(proto, proto_name, ep_cfg, base_seed + ep)
    proto.soft_reset_epsilon()   # switch to exploitation for evaluation


def _run_epoch(proto, proto_name: str, cfg: SimConfig, seed: int):
    """One pre-training epoch."""
    env = MANETEnvironment(cfg, seed)
    if hasattr(proto, 'force_recompute'):
        proto.force_recompute(env)
    pkt_waits: Dict[int, int] = {}

    for _ in range(cfg.total_steps):
        env.step()
        proto.periodic_update(env, cfg.dt)
        for pid, pkt in list(env.active.items()):
            if pkt.done:
                continue
            nxt, delay_ms, _ = proto.get_next_hop(env, pkt, pkt.current)
            if nxt is None:
                pkt_waits[pid] = pkt_waits.get(pid, 0) + 1
                if pkt_waits[pid] >= cfg.max_wait_steps:
                    env.timeout_drop(pkt)
                    _post_hop(proto, env, pkt, proto_name, True)
                continue
            ok = env.forward(pkt, nxt, delay_ms)
            if ok:
                pkt_waits.pop(pid, None)
                _post_hop(proto, env, pkt, proto_name, True)

    for pid, pkt in list(env.active.items()):
        env.timeout_drop(pkt)
        _post_hop(proto, env, pkt, proto_name, True)
