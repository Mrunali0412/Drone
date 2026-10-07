"""CPU shared DDQN with masked actions and selectable uniform/PER replay."""
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import random

import torch
from torch import nn

from environment.actions import PowerActionSpace, NoFeasibleActionError
from agents.replay import ReplayBuffer, PrioritizedReplayBuffer


@dataclass
class DDQNConfig:
    hidden_sizes: tuple = (64, 64)
    learning_rate: float = 5e-4
    gamma: float = 0.9
    batch_size: int = 32
    memory_size: int = 50000
    target_update: int = 100  # Optimizer updates.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_ttis: int = 1000
    reward_scale: float = 1e6  # Network reward converted from bits/s to Mbps.
    seed: int = 42
    prioritized_replay: bool = True
    per_alpha: float = 0.6
    per_epsilon: float = 1e-6
    per_beta_start: float = 0.4
    per_beta_updates: int = 10000

    def __post_init__(self):
        for value in (self.batch_size, self.memory_size, self.target_update, self.epsilon_decay_ttis, self.per_beta_updates, *self.hidden_sizes):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('Sizes and intervals must be positive integers')
        if self.batch_size > self.memory_size:
            raise ValueError('Batch size exceeds replay capacity')
        for value in (self.learning_rate, self.reward_scale):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Learning rate and reward scale must be positive')
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError('Invalid discount or exploration range')
        if not isinstance(self.prioritized_replay, bool):
            raise ValueError('prioritized_replay must be Boolean')
        if not 0 <= self.per_alpha <= 1 or not 0 <= self.per_beta_start <= 1:
            raise ValueError('Invalid PER alpha or beta')
        if not math.isfinite(self.per_epsilon) or self.per_epsilon <= 0:
            raise ValueError('PER epsilon must be positive')


class QNetwork(nn.Sequential):
    def __init__(self, hidden_sizes, num_actions):
        layers = []
        size = 4
        for width in hidden_sizes:
            layers.extend((nn.Linear(size, width), nn.ReLU()))
            size = width
        layers.append(nn.Linear(size, num_actions))
        super().__init__(*layers)


def encode_observation(observation, max_power):
    """Fixed transforms saved with the policy; no fitted dataset statistics."""
    gain, sinr, rate, power = observation
    if any(not math.isfinite(x) or x < 0 for x in observation):
        raise ValueError('Observation must contain finite nonnegative features')
    return (math.log10(max(gain, 1e-20)) / 20,
            math.log1p(sinr) / 10, rate / 1e6, power / max_power)


def ddqn_targets(online, target, next_states, masks, rewards, discounts):
    """Evaluation network chooses; target network evaluates feasible actions."""
    with torch.no_grad():
        active = discounts > 0
        if torch.any(active & ~masks.any(dim=1)):
            raise NoFeasibleActionError('Nonterminal transition has no feasible next action')
        result = rewards.clone()
        if active.any():
            q = online(next_states[active]).masked_fill(~masks[active], -torch.inf)
            choices = q.argmax(dim=1, keepdim=True)
            values = target(next_states[active]).gather(1, choices).squeeze(1)
            result[active] += discounts[active] * values
        return result


class SharedDDQN:
    def __init__(self, power_levels, config=None):
        self.config = config or DDQNConfig()
        self.power_levels = tuple(power_levels)
        self.action_space = PowerActionSpace(self.power_levels[0], self.power_levels[-1], len(self.power_levels))
        if self.action_space.powers != self.power_levels:
            raise ValueError('Expected the configured linear power grid')
        self.rng = random.Random(self.config.seed)
        with torch.random.fork_rng():
            torch.manual_seed(self.config.seed)
            self.online = QNetwork(self.config.hidden_sizes, len(self.power_levels))
            self.target = QNetwork(self.config.hidden_sizes, len(self.power_levels))
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay = (PrioritizedReplayBuffer(self.config.memory_size, self.config.per_alpha, self.config.per_epsilon)
                       if self.config.prioritized_replay else ReplayBuffer(self.config.memory_size))
        self.updates = 0
        self.training_ttis = 0

    @property
    def beta(self):
        fraction = min(1, self.updates / self.config.per_beta_updates)
        return self.config.per_beta_start + fraction * (1 - self.config.per_beta_start)

    @property
    def epsilon(self):
        fraction = min(1, self.training_ttis / self.config.epsilon_decay_ttis)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def act(self, observation, mask, *, training=False):
        state = encode_observation(observation, self.power_levels[-1])
        with torch.no_grad():
            values = self.online(torch.tensor(state, dtype=torch.float32)).tolist()
        return self.action_space.select(values, mask, epsilon=self.epsilon if training else 0, rng=self.rng)

    def learn(self):
        if len(self.replay) < self.config.batch_size:
            return None
        indices = None
        if self.config.prioritized_replay:
            batch, indices, weights = self.replay.sample(self.config.batch_size, self.rng, self.beta)
        else:
            batch = self.replay.sample(self.config.batch_size, self.rng)
            weights = [1.0] * len(batch)
        states = torch.tensor([x.state for x in batch], dtype=torch.float32)
        next_states = torch.tensor([x.next_state for x in batch], dtype=torch.float32)
        masks = torch.tensor([x.next_mask for x in batch], dtype=torch.bool)
        rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32)
        discounts = torch.tensor([x.discount for x in batch], dtype=torch.float32)
        actions = torch.tensor([x.action for x in batch]).unsqueeze(1)
        expected = ddqn_targets(self.online, self.target, next_states, masks, rewards, discounts)
        predicted = self.online(states).gather(1, actions).squeeze(1)
        td_errors = expected - predicted
        loss = (torch.tensor(weights, dtype=torch.float32) * td_errors.square()).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10)
        self.optimizer.step()
        if indices is not None:
            self.replay.update_priorities(indices, td_errors.detach().tolist())
        self.updates += 1
        if self.updates % self.config.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss.item()

    def save(self, path, metadata=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'version': 1, 'config': asdict(self.config), 'power_levels': self.power_levels,
                    'online': self.online.state_dict(), 'target': self.target.state_dict(),
                    'optimizer': self.optimizer.state_dict(), 'updates': self.updates,
                    'training_ttis': self.training_ttis, 'metadata': metadata or {},
                    'encoding': 'log10gain20_log1psinr10_rateMbps_powerMax'}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['version'] != 1 or data['encoding'] != 'log10gain20_log1psinr10_rateMbps_powerMax':
            raise ValueError('Unsupported checkpoint')
        agent = cls(data['power_levels'], DDQNConfig(**data['config']))
        agent.online.load_state_dict(data['online'])
        agent.target.load_state_dict(data['target'])
        agent.optimizer.load_state_dict(data['optimizer'])
        agent.updates = data['updates']
        agent.training_ttis = data['training_ttis']
        return agent
