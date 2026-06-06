"""
agents.py — Reinforcement-Learning Routing Agents
==================================================
Contains:
  • RewardCalculator          — composite reward (Appendix A)
  • QLRoutingAgent            — tabular Q-learning agent (Appendix A)
  • QNetwork                  — feed-forward Q-network (Appendix B)
  • PrioritizedReplayBuffer   — PER buffer  (Appendix B, BUG FIXED)
  • DQNRoutingAgent           — DQN agent   (Appendix B)
  • QLRouter                  — multi-agent QL wrapper (routing protocol interface)
  • DQNRouter                 — multi-agent DQN wrapper

BUG FIX in PrioritizedReplayBuffer.push():
  The original code in Appendix B did NOT update self.priorities[self.position]
  and did NOT advance self.position when the buffer was not yet full.
  This meant all newly inserted transitions kept priority 0 until the buffer
  wrapped around, breaking prioritised sampling during warm-up.
  Fixed lines are marked with  # ← BUG FIX
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, namedtuple
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from environment import MANETEnvironment, Packet

from environment import hop_delay_ms

# fractional control-overhead per RL hop (piggybacked ACK feedback)
RL_CTRL_PER_HOP = 0.11
# DQN trains every N routing decisions; only a small subset of agents per event
DQN_TRAIN_EVERY      = 30   # trigger a training event every N decisions
DQN_AGENTS_PER_BATCH = 4    # how many randomly chosen agents to update per event


# ══════════════════════════════════════════════════════════════════════════════
# Reward calculator  (Appendix A — verbatim)
# ══════════════════════════════════════════════════════════════════════════════
class RewardCalculator:
    """
    Вычисление составного вознаграждения для агента маршрутизации.
    (Appendix A, Table 3.2: w_d=0.4, w_e=0.2, w_l=0.2, w_p=0.2)
    """

    def __init__(
        self,
        w_delay    = 0.4,
        w_energy   = 0.2,
        w_link     = 0.2,
        w_delivery = 0.2,
        r_success  = 10.0,
        r_drop     = 10.0,
        max_delay  = 1.0,
        max_energy = 100.0,
        max_snr    = 30.0,
    ):
        self.w_delay    = w_delay
        self.w_energy   = w_energy
        self.w_link     = w_link
        self.w_delivery = w_delivery
        self.r_success  = r_success
        self.r_drop     = r_drop
        self.max_delay  = max_delay
        self.max_energy = max_energy
        self.max_snr    = max_snr

    def calculate_reward(
        self,
        delay:          float,
        neighbor_energy: float,
        link_snr:       float,
        delivered:      bool = False,
        dropped:        bool = False,
    ) -> float:
        # Компонента задержки (штраф)
        r_delay    = -min(delay / self.max_delay, 1.0)
        # Компонента энергии (поощрение)
        r_energy   = min(neighbor_energy / self.max_energy, 1.0)
        # Компонента качества канала (поощрение)
        r_link     = min(link_snr / self.max_snr, 1.0)
        # Компонента доставки
        if delivered:
            r_delivery = self.r_success
        elif dropped:
            r_delivery = -self.r_drop
        else:
            r_delivery = 0.0
        # Составное вознаграждение
        return (
            self.w_delay    * r_delay  +
            self.w_energy   * r_energy +
            self.w_link     * r_link   +
            self.w_delivery * r_delivery
        )


# ══════════════════════════════════════════════════════════════════════════════
# QL Routing Agent  (Appendix A — verbatim + minor additions)
# ══════════════════════════════════════════════════════════════════════════════
class QLRoutingAgent:
    """
    Агент маршрутизации на основе Q-learning.
    (Source: Thesis Appendix A)
    """

    def __init__(
        self,
        node_id:       int,
        num_nodes:     int,
        alpha          = 0.1,
        gamma          = 0.95,
        epsilon_start  = 1.0,
        epsilon_min    = 0.01,
        epsilon_decay  = 0.9995,
        initial_q      = -1.0,
    ):
        self.node_id         = node_id
        self.num_nodes       = num_nodes
        self.alpha           = alpha
        self.gamma           = gamma
        self.epsilon         = epsilon_start
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.initial_q       = initial_q

        # Q[destination][neighbour] = q_value
        self.q_table: Dict = defaultdict(lambda: defaultdict(lambda: initial_q))
        self.neighbors: Set[int]  = set()
        self.total_decisions  = 0
        self.exploration_count = 0

    def update_neighbors(self, neighbors):
        new  = set(neighbors) - self.neighbors
        lost = self.neighbors - set(neighbors)
        for nb in new:
            for dst in range(self.num_nodes):
                if dst != self.node_id:
                    self.q_table[dst][nb] = 0.0 if nb == dst else self.initial_q
        for nb in lost:
            for dst in list(self.q_table):
                self.q_table[dst].pop(nb, None)
        self.neighbors = set(neighbors)

    def select_next_hop(
        self,
        destination:   int,
        visited_nodes: Optional[Set[int]] = None,
    ) -> Optional[int]:
        if not self.neighbors:
            return None

        available_neighbors = list(self.neighbors)
        if visited_nodes:
            available_neighbors = [n for n in available_neighbors
                                   if n not in visited_nodes]
        if not available_neighbors:
            return None
        if destination in available_neighbors:
            return destination

        self.total_decisions += 1
        if np.random.random() < self.epsilon:
            self.exploration_count += 1
            return int(np.random.choice(available_neighbors))
        else:
            q_values   = {n: self.q_table[destination][n]
                          for n in available_neighbors}
            max_q      = max(q_values.values())
            best_neighbors = [n for n, q in q_values.items() if q == max_q]
            return int(np.random.choice(best_neighbors))

    def update_q_value(
        self,
        destination:     int,
        next_hop:        int,
        reward:          float,
        next_hop_best_q: float,
    ):
        current_q = self.q_table[destination][next_hop]
        td_target = reward + self.gamma * next_hop_best_q
        td_error  = td_target - current_q
        self.q_table[destination][next_hop] = current_q + self.alpha * td_error

    def get_best_q_value(self, destination: int) -> float:
        if not self.neighbors:
            return self.initial_q
        if destination == self.node_id:
            return 0.0
        q_values = [self.q_table[destination][n] for n in self.neighbors]
        return max(q_values) if q_values else self.initial_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_exploration_rate(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return self.exploration_count / self.total_decisions


# ══════════════════════════════════════════════════════════════════════════════
# DQN components  (Appendix B)
# ══════════════════════════════════════════════════════════════════════════════

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'done', 'mask')
)


class QNetwork(nn.Module):
    """
    Нейронная сеть для аппроксимации Q-функции.
    (Appendix B — verbatim)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple = (128, 128, 64)):
        super(QNetwork, self).__init__()
        layers    = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PrioritizedReplayBuffer:
    """
    Буфер воспроизведения опыта с приоритизацией.
    (Appendix B — BUG FIXED in push())

    Original bug: when self.size < self.capacity, the code appended to
    self.buffer and incremented self.size, but NEVER wrote the priority to
    self.priorities[self.position] and NEVER advanced self.position.
    All new transitions therefore had priority 0 until the buffer wrapped,
    making prioritised sampling useless during the entire warm-up phase.
    """

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100_000):
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.frame       = 1
        self.buffer: List = []
        self.priorities  = np.zeros(capacity, dtype=np.float32)
        self.position    = 0
        self.size        = 0

    def push(self, transition):
        """Добавление перехода в буфер."""
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        if self.size < self.capacity:
            self.buffer.append(transition)
            self.priorities[self.position] = max_priority  # ← BUG FIX (was missing)
            self.position = (self.position + 1) % self.capacity  # ← BUG FIX (was missing)
            self.size += 1
        else:
            self.buffer[self.position]     = transition
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """Выборка мини-пакета с приоритизацией."""
        priorities   = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, batch_size,
                                   p=probabilities, replace=False)

        beta    = min(1.0, self.beta_start +
                      self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        weights  = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights  = torch.FloatTensor(weights)

        transitions = [self.buffer[idx] for idx in indices]
        return transitions, indices, weights

    def update_priorities(self, indices, td_errors):
        """Обновление приоритетов на основе TD-ошибок."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(float(td_error)) + 1e-6

    def __len__(self):
        return self.size


class DQNRoutingAgent:
    """
    Агент маршрутизации на основе Deep Q-Network.
    (Appendix B — verbatim + minor additions for router wrapper)
    """

    def __init__(
        self,
        node_id:          int,
        num_nodes:        int,
        max_neighbors:    int   = 20,
        feature_dim:      int   = 4,
        lr:               float = 0.001,
        gamma:            float = 0.95,
        epsilon_start:    float = 1.0,
        epsilon_min:      float = 0.01,
        epsilon_decay:    float = 0.9995,
        buffer_capacity:  int   = 50_000,
        batch_size:       int   = 64,
        target_update_tau: float = 0.001,
        device:           str  = 'cpu',
    ):
        self.node_id          = node_id
        self.num_nodes        = num_nodes
        self.max_neighbors    = max_neighbors
        self.feature_dim      = feature_dim
        self.gamma            = gamma
        self.epsilon          = epsilon_start
        self.epsilon_min      = epsilon_min
        self.epsilon_decay    = epsilon_decay
        self.batch_size       = batch_size
        self.target_update_tau = target_update_tau
        self.device           = torch.device(device)

        # state_dim = max_neighbors × feature_dim + num_nodes (one-hot dst) + 1 (hop)
        self.state_dim  = max_neighbors * feature_dim + num_nodes + 1
        self.action_dim = max_neighbors

        # Q-сеть и целевая сеть
        self.q_network     = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer    = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)

        self.neighbors:       List[int]      = []
        self.neighbor_to_index: Dict[int, int] = {}

        self.total_decisions = 0
        self.training_steps  = 0
        self.losses:         List[float]    = []

    def update_neighbors(self, neighbors):
        self.neighbors        = list(neighbors)[:self.max_neighbors]
        self.neighbor_to_index = {n: i for i, n in enumerate(self.neighbors)}

    def _build_state_vector(
        self,
        destination:       int,
        hop_count:         int,
        neighbor_features: Dict[int, List[float]],
    ) -> np.ndarray:
        state = np.zeros(self.state_dim, dtype=np.float32)
        for i, neighbor in enumerate(self.neighbors):
            if neighbor in neighbor_features:
                features  = neighbor_features[neighbor]
                start_idx = i * self.feature_dim
                state[start_idx: start_idx + self.feature_dim] = features
        dest_start = self.max_neighbors * self.feature_dim
        if destination < self.num_nodes:
            state[dest_start + destination] = 1.0
        state[-1] = min(hop_count / 64.0, 1.0)
        return state

    def _get_action_mask(self, visited_nodes=None) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=bool)
        for i, neighbor in enumerate(self.neighbors):
            if visited_nodes is None or neighbor not in visited_nodes:
                mask[i] = True
        return mask

    def select_next_hop(
        self,
        destination:       int,
        hop_count:         int,
        neighbor_features: Dict[int, List[float]],
        visited_nodes:     Optional[Set[int]] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        if not self.neighbors:
            return None, None
        if destination in self.neighbor_to_index:
            return destination, self.neighbor_to_index[destination]

        mask = self._get_action_mask(visited_nodes)
        if not mask.any():
            return None, None

        self.total_decisions += 1
        if np.random.random() < self.epsilon:
            valid_actions = np.where(mask)[0]
            action_idx    = int(np.random.choice(valid_actions))
        else:
            state        = self._build_state_vector(destination, hop_count,
                                                    neighbor_features)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
            q_values[~mask] = -np.inf
            action_idx = int(np.argmax(q_values))

        if action_idx < len(self.neighbors):
            return self.neighbors[action_idx], action_idx
        return None, None

    def store_transition(self, state, action, reward, next_state, done, mask):
        transition = Transition(
            state      = torch.FloatTensor(state),
            action     = action,
            reward     = reward,
            next_state = torch.FloatTensor(next_state),
            done       = done,
            mask       = torch.BoolTensor(mask),
        )
        self.replay_buffer.push(transition)

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        batch_state      = torch.stack([t.state      for t in transitions]).to(self.device)
        batch_action     = torch.LongTensor([t.action  for t in transitions]).to(self.device)
        batch_reward     = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
        batch_next_state = torch.stack([t.next_state  for t in transitions]).to(self.device)
        batch_done       = torch.FloatTensor([t.done   for t in transitions]).to(self.device)
        batch_mask       = torch.stack([t.mask         for t in transitions]).to(self.device)
        weights          = weights.to(self.device)

        current_q = self.q_network(batch_state).gather(
            1, batch_action.unsqueeze(1)).squeeze(1)

        # Double DQN
        with torch.no_grad():
            next_q_main = self.q_network(batch_next_state)
            next_q_main[~batch_mask] = -float('inf')
            next_actions = next_q_main.argmax(dim=1)
            next_q_target = self.target_network(batch_next_state)
            next_q   = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = batch_reward + (1 - batch_done) * self.gamma * next_q

        td_errors = (target_q - current_q).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        loss = (weights * (current_q - target_q) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self._soft_update_target()
        self.training_steps += 1
        v = float(loss.item())
        self.losses.append(v)
        return v

    def _soft_update_target(self):
        tau = self.target_update_tau
        for tp, mp in zip(self.target_network.parameters(),
                          self.q_network.parameters()):
            tp.data.copy_(tau * mp.data + (1.0 - tau) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path: str):
        torch.save({
            'q_network_state_dict':     self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict':     self.optimizer.state_dict(),
            'epsilon':                  self.epsilon,
            'training_steps':           self.training_steps,
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon       = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']


# ══════════════════════════════════════════════════════════════════════════════
# Router wrappers — implement the same protocol interface as AODV / DSDV
# ══════════════════════════════════════════════════════════════════════════════

class QLRouter:
    """
    Wraps per-node QLRoutingAgents into a network-wide routing protocol.
    Handles state building, reward computation, and Q-table updates.
    """

    def __init__(self, agents: Dict[int, QLRoutingAgent], rc: RewardCalculator):
        self.agents = agents
        self.rc     = rc
        self._ctrl  = 0.0
        # pending hops waiting for post-hop reward: pid → (prev_node, nxt)
        self._pend: Dict[int, Tuple[int, int]] = {}

    def reset(self):
        for ag in self.agents.values():
            ag.q_table.clear()
            ag.epsilon        = 1.0
            ag.total_decisions = 0
            ag.exploration_count = 0
        self._ctrl = 0.0
        self._pend.clear()

    def soft_reset_epsilon(self):
        """Set epsilon to minimum for evaluation (exploit learned policy)."""
        for ag in self.agents.values():
            ag.epsilon = ag.epsilon_min

    def total_ctrl(self) -> float:
        return self._ctrl

    def get_next_hop(
        self,
        env:     'MANETEnvironment',
        pkt:     'Packet',
        node_id: int,
    ) -> Tuple[Optional[int], float, float]:
        ag = self.agents[node_id]
        ag.update_neighbors(env.neighbors(node_id))
        # Only block the last few visited nodes to prevent local cycles
        # while still allowing the agent to find the destination via
        # longer paths when Q-table is partially trained.
        recent = set(pkt.path[-6:]) if len(pkt.path) > 6 else pkt.visited
        nxt = ag.select_next_hop(pkt.dst, recent)
        if nxt is None:
            return None, 0.0, 0.0
        self._pend[pkt.pid] = (node_id, nxt)
        self._ctrl += RL_CTRL_PER_HOP
        return nxt, hop_delay_ms(env, node_id, nxt), RL_CTRL_PER_HOP

    def post_hop(self, env: 'MANETEnvironment', pkt: 'Packet'):
        """Update Q-values with reward after a hop completes (or packet ends)."""
        key = pkt.pid
        if key not in self._pend:
            return
        prev, nxt = self._pend.pop(key)
        ag        = self.agents[prev]

        delay_s  = hop_delay_ms(env, prev, nxt) / 1000.0
        energy   = env.nodes[nxt].energy
        snr      = env.snr_db(prev, nxt)
        reward   = self.rc.calculate_reward(
            delay_s, energy, snr,
            delivered=pkt.delivered,
            dropped=pkt.dropped,
        )
        nxt_ag = self.agents[nxt]
        nxt_ag.update_neighbors(env.neighbors(nxt))
        best_q = nxt_ag.get_best_q_value(pkt.dst)
        ag.update_q_value(pkt.dst, nxt, reward, best_q)
        ag.decay_epsilon()

    def periodic_update(self, env: 'MANETEnvironment', dt: float):
        pass  # QL has no periodic overhead


class DQNRouter:
    """
    Wraps per-node DQNRoutingAgents into a network-wide routing protocol.
    """

    def __init__(self, agents: Dict[int, DQNRoutingAgent], rc: RewardCalculator):
        self.agents   = agents
        self.rc       = rc
        self._ctrl    = 0.0
        self._step    = 0
        # pending: pid → (prev_node, action_idx, state_vector)
        self._pend: Dict[int, Tuple[int, int, np.ndarray]] = {}

    def reset(self):
        # Keep learned weights — only reset exploration and counters
        for ag in self.agents.values():
            ag.epsilon = 1.0
        self._ctrl = 0.0
        self._step = 0
        self._pend.clear()

    def soft_reset_epsilon(self):
        for ag in self.agents.values():
            ag.epsilon = ag.epsilon_min

    def total_ctrl(self) -> float:
        return self._ctrl

    def get_next_hop(
        self,
        env:     'MANETEnvironment',
        pkt:     'Packet',
        node_id: int,
    ) -> Tuple[Optional[int], float, float]:
        ag    = self.agents[node_id]
        ag.update_neighbors(env.neighbors(node_id))
        feats = env.nbr_features(node_id)
        recent = set(pkt.path[-6:]) if len(pkt.path) > 6 else pkt.visited
        nxt, idx = ag.select_next_hop(pkt.dst, pkt.hops, feats, recent)
        if nxt is None or idx is None:
            return None, 0.0, 0.0
        state = ag._build_state_vector(pkt.dst, pkt.hops, feats)
        self._pend[pkt.pid] = (node_id, idx, state)
        self._ctrl += RL_CTRL_PER_HOP
        return nxt, hop_delay_ms(env, node_id, nxt), RL_CTRL_PER_HOP

    def post_hop(self, env: 'MANETEnvironment', pkt: 'Packet'):
        key = pkt.pid
        if key not in self._pend:
            return
        prev, idx, state = self._pend.pop(key)
        nxt = pkt.current    # packet was already advanced

        delay_s = hop_delay_ms(env, prev, nxt) / 1000.0
        energy  = env.nodes[nxt].energy
        snr     = env.snr_db(prev, nxt)
        reward  = self.rc.calculate_reward(
            delay_s, energy, snr,
            delivered=pkt.delivered,
            dropped=pkt.dropped,
        )
        ag_nxt      = self.agents[nxt]
        ag_nxt.update_neighbors(env.neighbors(nxt))
        feats_nxt   = env.nbr_features(nxt)
        next_state  = ag_nxt._build_state_vector(pkt.dst, pkt.hops, feats_nxt)
        mask        = ag_nxt._get_action_mask(pkt.visited)
        done        = float(pkt.done)

        self.agents[prev].store_transition(state, idx, reward, next_state, done, mask)

        self._step += 1
        if self._step % DQN_TRAIN_EVERY == 0:
            # Sample a small random subset of agents to train this round
            agent_ids = list(self.agents.keys())
            chosen = np.random.choice(
                agent_ids,
                size=min(DQN_AGENTS_PER_BATCH, len(agent_ids)),
                replace=False,
            )
            for aid in chosen:
                self.agents[aid].train_step()
            # Decay epsilon for all agents (cheap)
            for ag in self.agents.values():
                ag.decay_epsilon()

    def periodic_update(self, env: 'MANETEnvironment', dt: float):
        pass


# ── Factory helpers ────────────────────────────────────────────────────────────

def make_ql_agents(cfg, eps: float = 1.0) -> Dict[int, QLRoutingAgent]:
    return {
        i: QLRoutingAgent(i, cfg.num_nodes, epsilon_start=eps)
        for i in range(cfg.num_nodes)
    }


def make_dqn_agents(cfg, device: str = 'cpu',
                    eps: float = 1.0) -> Dict[int, DQNRoutingAgent]:
    return {
        i: DQNRoutingAgent(
            i, cfg.num_nodes,
            max_neighbors  = min(20, cfg.num_nodes - 1),
            feature_dim    = 4,
            epsilon_start  = eps,
            device         = device,
        )
        for i in range(cfg.num_nodes)
    }


def make_reward_calc() -> RewardCalculator:
    """Default reward weights from Table 3.2."""
    return RewardCalculator(
        w_delay    = 0.4,
        w_energy   = 0.2,
        w_link     = 0.2,
        w_delivery = 0.2,
        r_success  = 10.0,
        r_drop     = 10.0,
        max_delay  = 1.0,
        max_energy = 100.0,
        max_snr    = 30.0,
    )
