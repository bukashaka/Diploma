"""
protocols.py — Classical MANET Routing Protocols
================================================
Implements simplified but behaviourally accurate versions of:
  • AODV  — reactive, route-on-demand, RREQ/RREP/RERR
  • DSDV  — proactive, periodic table broadcasts (Bellman-Ford)

Both classes implement the same interface used by the simulation runner:
    get_next_hop(env, pkt, node_id) → (next_hop | None, delay_ms, ctrl_pkts)
    periodic_update(env, dt)
    total_ctrl() → float
    reset()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from environment import MANETEnvironment, Packet

from environment import hop_delay_ms

# ── AODV parameters ────────────────────────────────────────────────────────────
AODV_ROUTE_TTL_S  = 20.0   # route lifetime (s) — calibrated to thesis NRO ≈ 0.42
AODV_RREQ_SIZE    = 15     # ctrl packets per RREQ flood (half-network broadcast)
# (one RREQ = AODV_RREQ_SIZE ctrl-pkts; one RREP unicast = path_len ctrl-pkts)

# ── DSDV parameters ────────────────────────────────────────────────────────────
DSDV_UPDATE_S     = 5.0    # full table broadcast interval
DSDV_INCR_FACTOR  = 0.5    # fractional ctrl-pkts per broken link (incremental update)


# ── Route entry (used only by AODV) ───────────────────────────────────────────
@dataclass
class _RouteEntry:
    nxt:    int    # next-hop node ID
    hops:   int    # path length in hops
    seq:    int    # sequence number (unused in this simplified model)
    born_t: float  # simulation time when this entry was installed


# ══════════════════════════════════════════════════════════════════════════════
# AODV
# ══════════════════════════════════════════════════════════════════════════════
ROUTE_DISC_COOLDOWN_S = 1.5   # minimum gap between two discovery attempts for the
                               # same (src, dst) pair — prevents repeated flooding
                               # when destination is momentarily unreachable.

class AODVProtocol:
    """
    Simplified AODV (RFC 3561).

    Route discovery is modelled by BFS on the *current* topology graph,
    but the result is cached with a lifetime of AODV_ROUTE_TTL_S.
    A route re-discovery is triggered when:
      • no entry exists for the destination, or
      • the cached next-hop has left the neighbourhood (link break → RERR), or
      • the entry has expired.

    Control-packet counting (for NRO):
      • Each RREQ flood          → AODV_RREQ_SIZE ctrl-pkts
      • Each RREP unicast        → path_length     ctrl-pkts
      • Each RERR notification   → 1 ctrl-pkt
    """

    def __init__(self):
        # _rt[node_id][destination] = RouteEntry
        self._rt:    Dict[int, Dict[int, _RouteEntry]] = {}
        self._ctrl:  float = 0.0   # cumulative control packets
        # cooldown: (src, dst) → simulation-time of last discovery attempt
        self._disc_t: Dict[tuple, float] = {}

    def reset(self):
        self._rt.clear()
        self._ctrl  = 0.0
        self._disc_t.clear()

    def total_ctrl(self) -> float:
        return self._ctrl

    # ── public interface ────────────────────────────────────────────────────────

    def get_next_hop(
        self,
        env:     'MANETEnvironment',
        pkt:     'Packet',
        node_id: int,
    ) -> Tuple[Optional[int], float, float]:
        """
        Returns (next_hop, delay_ms, ctrl_pkts).
        delay_ms includes per-hop TX/queuing plus any route-discovery overhead.
        """
        dst  = pkt.dst
        nbrs = env.neighbors(node_id)

        if not nbrs:
            return None, 0.0, 0.0

        # Direct delivery
        if dst in nbrs:
            return dst, hop_delay_ms(env, node_id, dst), 0.0

        # Check cache
        entry = self._rt.get(node_id, {}).get(dst)
        extra = 0.0    # discovery delay (ms)
        ctrl  = 0.0

        if entry is not None:
            age = env.t - entry.born_t
            if entry.nxt in nbrs and age <= AODV_ROUTE_TTL_S:
                # Route valid → forward
                return entry.nxt, hop_delay_ms(env, node_id, entry.nxt), 0.0
            else:
                # Route broken / expired → send RERR, then rediscover
                self._rt.get(node_id, {}).pop(dst, None)
                ctrl       += 1.0       # RERR
                self._ctrl += 1.0

        # Cooldown guard: don't flood RREQ again until cooldown expires
        key = (node_id, dst)
        if env.t - self._disc_t.get(key, -999.0) < ROUTE_DISC_COOLDOWN_S:
            return None, 0.0, 0.0   # wait silently; no ctrl pkts

        # Route discovery
        self._disc_t[key] = env.t
        entry, disc_ms, disc_ctrl = self._discover(env, node_id, dst)
        ctrl  += disc_ctrl
        extra += disc_ms

        if entry is None or entry.nxt not in nbrs:
            return None, extra, ctrl

        d = hop_delay_ms(env, node_id, entry.nxt) + extra
        return entry.nxt, d, ctrl

    def periodic_update(self, env: 'MANETEnvironment', dt: float):
        """Expire stale route entries on link breaks; prune old cooldown records."""
        for nid in range(env.cfg.num_nodes):
            table = self._rt.get(nid, {})
            nbrs  = env.neighbors(nid)
            stale = [
                d for d, e in table.items()
                if e.nxt not in nbrs or (env.t - e.born_t) > AODV_ROUTE_TTL_S
            ]
            for d in stale:
                table.pop(d)
                # reset cooldown so a fresh path can be found next attempt
                self._disc_t.pop((nid, d), None)
        # Prune expired cooldown entries (older than cooldown period)
        expired = [k for k, t in self._disc_t.items()
                   if env.t - t >= ROUTE_DISC_COOLDOWN_S]
        for k in expired:
            self._disc_t.pop(k, None)

    # ── private ─────────────────────────────────────────────────────────────────

    def _discover(
        self,
        env:  'MANETEnvironment',
        src:  int,
        dst:  int,
    ) -> Tuple[Optional[_RouteEntry], float, float]:
        """
        Simulate RREQ/RREP exchange.
        Returns (route_entry_at_src, discovery_delay_ms, ctrl_pkts_generated).
        """
        ctrl = float(AODV_RREQ_SIZE)    # RREQ flood
        self._ctrl += ctrl

        path = env.shortest_path(src, dst)
        if path is None:
            return None, 0.0, ctrl      # unreachable

        hops   = len(path) - 1
        t_now  = env.t

        # Install forward route at every node along the path
        for k in range(len(path) - 1):
            u   = path[k]
            v   = path[k + 1]
            e   = _RouteEntry(nxt=v, hops=hops - k, seq=0, born_t=t_now)
            self._rt.setdefault(u, {})[dst] = e
            # Also install reverse (for RREP return path)
            rev = _RouteEntry(nxt=path[k], hops=k + 1, seq=0, born_t=t_now)
            self._rt.setdefault(v, {}).setdefault(src, rev)

        rrep_ctrl   = float(hops)       # unicast RREP hops
        ctrl       += rrep_ctrl
        self._ctrl += rrep_ctrl

        # Discovery delay = RREQ (src→dst) + RREP (dst→src) round-trip
        disc_ms = hops * 4.0            # ~2 ms/hop × 2 directions

        src_entry = self._rt.get(src, {}).get(dst)
        return src_entry, disc_ms, ctrl


# ══════════════════════════════════════════════════════════════════════════════
# DSDV
# ══════════════════════════════════════════════════════════════════════════════
class DSDVProtocol:
    """
    Simplified DSDV (Destination-Sequenced Distance-Vector).

    Routing tables are recomputed via BFS on the current topology graph every
    DSDV_UPDATE_S seconds.  Control-packet counting:
      • Full update broadcast   → 1 packet per neighbour per node
      • Incremental update      → DSDV_INCR_FACTOR ctrl-pkts per broken link
    """

    def __init__(self):
        # _rt[node_id][destination] = next_hop
        self._rt:          Dict[int, Dict[int, int]] = {}
        self._next_full_t: float = 0.0
        self._ctrl:        float = 0.0

    def reset(self):
        self._rt.clear()
        self._next_full_t = 0.0
        self._ctrl        = 0.0

    def total_ctrl(self) -> float:
        return self._ctrl

    # ── public interface ────────────────────────────────────────────────────────

    def get_next_hop(
        self,
        env:     'MANETEnvironment',
        pkt:     'Packet',
        node_id: int,
    ) -> Tuple[Optional[int], float, float]:
        dst  = pkt.dst
        nbrs = env.neighbors(node_id)

        if not nbrs:
            return None, 0.0, 0.0

        if dst in nbrs:
            return dst, hop_delay_ms(env, node_id, dst), 0.0

        table = self._rt.get(node_id, {})
        nxt   = table.get(dst)
        if nxt is None or nxt not in nbrs:
            return None, 0.0, 0.0      # table is stale or unreachable

        return nxt, hop_delay_ms(env, node_id, nxt), 0.0

    def periodic_update(self, env: 'MANETEnvironment', dt: float):
        """Full periodic recompute + incremental updates for link changes."""
        if env.t >= self._next_full_t:
            self._full_recompute(env)
            self._next_full_t = env.t + DSDV_UPDATE_S
            # Control overhead: each node sends 1 routing-update packet per neighbour
            for nd in env.nodes:
                self._ctrl += float(len(nd.neighbors))

        # Incremental overhead for topology changes
        self._ctrl += float(len(env.failed_links)) * DSDV_INCR_FACTOR

    def force_recompute(self, env: 'MANETEnvironment'):
        """Call once at start of simulation to build initial table."""
        self._full_recompute(env)
        self._next_full_t = env.t + DSDV_UPDATE_S

    # ── private ─────────────────────────────────────────────────────────────────

    def _full_recompute(self, env: 'MANETEnvironment'):
        """Rebuild all-pairs routing table using BFS on current graph."""
        import networkx as nx
        g  = env.graph
        N  = env.cfg.num_nodes
        rt: Dict[int, Dict[int, int]] = {}
        for src in range(N):
            rt[src] = {}
            try:
                paths = nx.single_source_shortest_path(g, src)
            except Exception:
                continue
            for dst, path in paths.items():
                if len(path) >= 2:
                    rt[src][dst] = path[1]    # first hop
        self._rt = rt
