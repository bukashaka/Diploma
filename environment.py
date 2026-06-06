"""
environment.py — MANET Simulation Environment
==============================================
Implements:
  • Random Waypoint mobility model
  • Two-ray ground reflection channel model
  • CBR traffic generation
  • Packet-level delivery / energy bookkeeping
  • Topology graph (via NetworkX)

All physical parameters match Table 3.1 of the thesis.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx

# ── Physical / link constants ──────────────────────────────────────────────────
TX_RANGE_M   = 250.0          # m
TX_POWER_W   = 0.66           # W  (transmit)
RX_POWER_W   = 0.395          # W  (receive)
BW_BPS       = 2_000_000      # 2 Mbps
PKT_BYTES    = 512            # bytes
TX_TIME_S    = PKT_BYTES * 8 / BW_BPS        # ≈ 0.00205 s
ENERGY_TX_J  = TX_POWER_W * TX_TIME_S        # J per TX event
ENERGY_RX_J  = RX_POWER_W * TX_TIME_S        # J per RX event
BASE_HOP_MS  = TX_TIME_S * 1000 + 5.0        # ≈ 7 ms base hop delay
QUEUE_MS     = 2.0            # extra ms per queued packet at receiver
MAX_TTL      = 64
FREQ_HZ      = 2.4e9
C_MS         = 3e8
ANT_HT_M     = 1.5           # antenna height


# ── Simulation configuration ───────────────────────────────────────────────────
@dataclass
class SimConfig:
    """All tunable parameters for one simulation run (Table 3.1 / 3.2)."""
    # Network
    num_nodes:       int   = 50
    tx_range:        float = TX_RANGE_M
    area:            float = 1000.0        # m  (square)
    initial_energy:  float = 100.0         # J
    # Mobility
    max_speed:       float = 20.0          # m/s
    pause_max:       float = 10.0          # s
    # Traffic
    num_flows:       int   = 10
    flow_rate:       float = 4.0           # pkts/s per flow
    # Simulation
    sim_time:        float = 600.0         # s
    dt:              float = 0.1           # s per topology step
    max_wait_steps:  int   = 60            # before timeout-drop
    # RL pre-training (epochs × epoch_duration_s)
    ql_pretrain_ep:  int   = 15
    dqn_pretrain_ep: int   = 25
    pretrain_ep_s:   float = 60.0

    @property
    def total_steps(self) -> int:
        return max(1, int(self.sim_time / self.dt))


# ── Packet ─────────────────────────────────────────────────────────────────────
@dataclass
class Packet:
    pid:      int
    src:      int
    dst:      int
    born_t:   float          # simulation time at creation

    # routing state (mutable during routing)
    current:   int   = field(init=False)
    hops:      int   = 0
    ttl:       int   = MAX_TTL
    path:      List[int] = field(default_factory=list)
    visited:   Set[int]  = field(default_factory=set)

    # accumulated delay (milliseconds, independent of sim time-step)
    delay_ms:  float = 0.0
    # wait counter (steps without a forward)
    wait:      int   = 0

    # outcome
    delivered: bool  = False
    dropped:   bool  = False

    def __post_init__(self):
        self.current = self.src
        self.path    = [self.src]
        self.visited = {self.src}

    @property
    def done(self) -> bool:
        return self.delivered or self.dropped


# ── Node ───────────────────────────────────────────────────────────────────────
class Node:
    """Mobile MANET node."""
    __slots__ = ('nid', 'x', 'y', 'energy', 'init_e',
                 'tx', 'ty', 'speed', 'pause',
                 'neighbors', 'queue_len')

    def __init__(self, nid: int, x: float, y: float, e0: float):
        self.nid       = nid
        self.x         = x
        self.y         = y
        self.energy    = e0
        self.init_e    = e0
        self.tx        = x          # waypoint target x
        self.ty        = y          # waypoint target y
        self.speed     = 0.0
        self.pause     = 0.0
        self.neighbors: Set[int] = set()
        self.queue_len: int      = 0

    def dist(self, other: 'Node') -> float:
        return np.hypot(self.x - other.x, self.y - other.y)


# ── Main environment ───────────────────────────────────────────────────────────
class MANETEnvironment:
    """
    Discrete-time MANET simulation.

    Topology is rebuilt every dt seconds.  Packet delay (delay_ms) is tracked
    independently of simulation time — each forwarded hop adds BASE_HOP_MS +
    queuing, giving physically accurate end-to-end delays (~7–15 ms/hop).
    """

    def __init__(self, cfg: SimConfig, seed: int = 42):
        self.cfg   = cfg
        self.rng   = np.random.default_rng(seed)
        self.t     = 0.0                 # simulated time (s)

        self.nodes: List[Node] = []
        self.graph = nx.Graph()

        # link-change sets, populated by step()
        self.failed_links: Set[Tuple[int, int]] = set()
        self.new_links:    Set[Tuple[int, int]] = set()

        self._init_nodes()

        # CBR flows
        self.flows:   List[Tuple[int, int]] = self._make_flows()
        self._ftimer: List[float]           = [0.0] * len(self.flows)

        # packet book-keeping
        self._pid:     int                  = 0
        self.all_pkts: List[Packet]         = []
        self.active:   Dict[int, Packet]    = {}

    # ── initialisation ──────────────────────────────────────────────────────────

    def _init_nodes(self):
        cfg = self.cfg
        for i in range(cfg.num_nodes):
            x  = self.rng.uniform(0, cfg.area)
            y  = self.rng.uniform(0, cfg.area)
            nd = Node(i, x, y, cfg.initial_energy)
            nd.tx    = self.rng.uniform(0, cfg.area)
            nd.ty    = self.rng.uniform(0, cfg.area)
            nd.speed = self.rng.uniform(0.01, max(cfg.max_speed, 0.01))
            nd.pause = self.rng.uniform(0, cfg.pause_max)
            self.nodes.append(nd)
        self._rebuild_graph()

    def _make_flows(self) -> List[Tuple[int, int]]:
        cfg   = self.cfg
        flows: List[Tuple[int, int]] = []
        seen:  Set[Tuple[int, int]]  = set()
        while len(flows) < cfg.num_flows:
            s = int(self.rng.integers(0, cfg.num_nodes))
            d = int(self.rng.integers(0, cfg.num_nodes))
            if s != d and (s, d) not in seen:
                flows.append((s, d))
                seen.add((s, d))
        return flows

    # ── topology ────────────────────────────────────────────────────────────────

    def _rebuild_graph(self):
        cfg = self.cfg
        ns  = self.nodes
        N   = cfg.num_nodes
        for nd in ns:
            nd.neighbors = set()
        self.graph.clear()
        self.graph.add_nodes_from(range(N))

        xs = np.array([n.x for n in ns], dtype=np.float64)
        ys = np.array([n.y for n in ns], dtype=np.float64)
        for i in range(N):
            dx  = xs[i + 1:] - xs[i]
            dy  = ys[i + 1:] - ys[i]
            d   = np.sqrt(dx * dx + dy * dy)
            for k, j in enumerate(range(i + 1, N)):
                if d[k] <= cfg.tx_range:
                    ns[i].neighbors.add(j)
                    ns[j].neighbors.add(i)
                    self.graph.add_edge(i, j)

    def neighbors(self, nid: int) -> Set[int]:
        return self.nodes[nid].neighbors

    def shortest_path(self, src: int, dst: int) -> Optional[List[int]]:
        if src == dst:
            return [src]
        try:
            return nx.shortest_path(self.graph, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def snr_db(self, i: int, j: int) -> float:
        return _snr_db(self.nodes[i].dist(self.nodes[j]))

    # ── mobility ────────────────────────────────────────────────────────────────

    def _move_nodes(self, dt: float):
        cfg = self.cfg
        for nd in self.nodes:
            if nd.pause > 0:
                nd.pause = max(0.0, nd.pause - dt)
                continue
            if cfg.max_speed <= 0:
                continue
            dx, dy = nd.tx - nd.x, nd.ty - nd.y
            d      = np.hypot(dx, dy)
            step   = nd.speed * dt
            if d <= step + 0.01:
                nd.x, nd.y = nd.tx, nd.ty
                nd.pause   = self.rng.uniform(0, cfg.pause_max)
                nd.tx      = self.rng.uniform(0, cfg.area)
                nd.ty      = self.rng.uniform(0, cfg.area)
                nd.speed   = self.rng.uniform(0.01, max(cfg.max_speed, 0.01))
            else:
                nd.x += (dx / d) * step
                nd.y += (dy / d) * step

    # ── time step ────────────────────────────────────────────────────────────────

    def step(self) -> List[Packet]:
        """
        Advance by cfg.dt seconds.
        Returns list of newly created packets.
        """
        dt = self.cfg.dt
        self.t += dt
        self._move_nodes(dt)

        # track link changes for AODV RERR / DSDV incremental updates
        old_nbrs = {i: nd.neighbors.copy() for i, nd in enumerate(self.nodes)}
        self._rebuild_graph()
        self.failed_links = set()
        self.new_links    = set()
        for i, nd in enumerate(self.nodes):
            for j in old_nbrs[i] - nd.neighbors:
                if i < j:
                    self.failed_links.add((i, j))
            for j in nd.neighbors - old_nbrs[i]:
                if i < j:
                    self.new_links.add((i, j))

        return self._gen_packets()

    def _gen_packets(self) -> List[Packet]:
        new = []
        dt  = self.cfg.dt
        for k, (s, d) in enumerate(self.flows):
            self._ftimer[k] -= dt
            while self._ftimer[k] <= 0:
                p = Packet(self._pid, s, d, self.t)
                self._pid += 1
                self.all_pkts.append(p)
                self.active[p.pid] = p
                new.append(p)
                self._ftimer[k] += 1.0 / self.cfg.flow_rate
        return new

    # ── packet operations ────────────────────────────────────────────────────────

    def forward(self, pkt: Packet, nxt: int, delay_ms: float) -> bool:
        """
        Forward pkt one hop to nxt.  Returns False if link vanished.
        Updates energy, queue lengths, packet state.
        """
        cur = pkt.current
        if nxt not in self.nodes[cur].neighbors:
            return False                  # link no longer exists

        # energy
        self.nodes[cur].energy = max(0.0, self.nodes[cur].energy - ENERGY_TX_J)
        self.nodes[nxt].energy = max(0.0, self.nodes[nxt].energy - ENERGY_RX_J)
        # queue bookkeeping (soft model)
        if self.nodes[cur].queue_len > 0:
            self.nodes[cur].queue_len -= 1
        self.nodes[nxt].queue_len = min(50, self.nodes[nxt].queue_len + 1)

        # update packet
        pkt.delay_ms += delay_ms
        pkt.hops     += 1
        pkt.ttl      -= 1
        pkt.path.append(nxt)
        pkt.visited.add(nxt)
        pkt.current   = nxt
        pkt.wait      = 0

        if nxt == pkt.dst:
            self._deliver(pkt)
        elif pkt.ttl <= 0:
            self._drop(pkt)
        return True

    def _deliver(self, pkt: Packet):
        pkt.delivered = True
        self.active.pop(pkt.pid, None)
        # Release the queue slot at the destination
        if self.nodes[pkt.current].queue_len > 0:
            self.nodes[pkt.current].queue_len -= 1

    def _drop(self, pkt: Packet):
        pkt.dropped = True
        self.active.pop(pkt.pid, None)
        # Release the queue slot at the current node
        if self.nodes[pkt.current].queue_len > 0:
            self.nodes[pkt.current].queue_len -= 1

    def timeout_drop(self, pkt: Packet):
        """Drop a packet that has been waiting too long."""
        self._drop(pkt)

    # ── RL state ─────────────────────────────────────────────────────────────────

    def nbr_features(self, nid: int) -> Dict[int, List[float]]:
        """
        Returns {neighbor_id: [queue_util, energy_ratio, snr_norm, dist_norm]}
        for all current neighbors of node nid.
        """
        nd    = self.nodes[nid]
        feats: Dict[int, List[float]] = {}
        for j in nd.neighbors:
            nb   = self.nodes[j]
            dist = nd.dist(nb)
            feats[j] = [
                min(nb.queue_len / 20.0, 1.0),   # queue utilisation
                nb.energy / nb.init_e,             # residual energy (normalised)
                _snr_db(dist) / 40.0,             # SNR (normalised)
                dist / self.cfg.tx_range,         # relative distance
            ]
        return feats

    # ── metrics ───────────────────────────────────────────────────────────────────

    def metrics(self) -> Dict:
        """Compute all performance metrics over completed packets."""
        delivered = [p for p in self.all_pkts if p.delivered]
        n_sent    = len(self.all_pkts)
        n_recv    = len(delivered)
        pdr       = n_recv / n_sent * 100.0 if n_sent else 0.0

        delays = [p.delay_ms for p in delivered]
        avg_d  = float(np.mean(delays)) if delays else 0.0
        std_d  = float(np.std(delays))  if delays else 0.0

        hops   = [p.hops for p in delivered]
        avg_h  = float(np.mean(hops)) if hops else 0.0

        thru   = n_recv * PKT_BYTES * 8 / self.cfg.sim_time if n_recv else 0.0
        avg_e  = float(np.mean([n.energy for n in self.nodes]))

        return dict(
            pdr            = pdr,
            avg_delay_ms   = avg_d,
            std_delay_ms   = std_d,
            avg_hops       = avg_h,
            throughput_bps = thru,
            avg_energy_j   = avg_e,
            n_sent         = n_sent,
            n_recv         = n_recv,
            # nro filled in by simulation runner after adding protocol ctrl overhead
            nro            = 0.0,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _snr_db(dist: float) -> float:
    """Two-ray ground reflection SNR in dB, clamped to [0, 40]."""
    d   = max(1.0, dist)
    Pt  = 0.01              # transmit power (W) — used for SNR only
    lam = C_MS / FREQ_HZ
    ht  = hr = ANT_HT_M
    d_c = 4 * np.pi * ht * hr / lam
    if d <= d_c:
        Pr = Pt * (lam / (4 * np.pi * d)) ** 2
    else:
        Pr = Pt * (ht * hr) ** 2 / d ** 4
    N0  = 1.38e-23 * 290 * BW_BPS      # thermal noise floor
    snr = 10 * np.log10(max(Pr / N0, 1e-12))
    return float(np.clip(snr, 0.0, 40.0))


def hop_delay_ms(env: MANETEnvironment, src: int, dst: int) -> float:
    """Per-hop delay (ms): transmission time + processing + receiver queuing."""
    q = env.nodes[dst].queue_len
    return BASE_HOP_MS + q * QUEUE_MS
