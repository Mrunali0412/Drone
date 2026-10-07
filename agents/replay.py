from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Transition:
    state: tuple
    action: int
    reward: float
    next_state: tuple
    next_mask: tuple
    discount: float  # gamma**duration, or zero at the finite episode boundary.


class ReplayBuffer:
    def __init__(self, capacity):
        self.items = deque(maxlen=capacity)

    def __len__(self):
        return len(self.items)

    def add(self, transition):
        self.items.append(transition)

    def sample(self, batch_size, rng):
        return rng.sample(list(self.items), batch_size)


class PrioritizedReplayBuffer:
    """Proportional PER with replacement; ring indices identify stored entries.

    This reference implementation computes probabilities in O(memory size).
    A sum tree can replace sampling later if profiling warrants it.
    """
    def __init__(self, capacity, alpha=0.6, epsilon=1e-6):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError('Capacity must be a positive integer')
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0, 1]')
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('epsilon must be positive')
        self.capacity, self.alpha, self.epsilon = capacity, alpha, epsilon
        self.items, self.priorities = [], []
        self.position = 0

    def __len__(self):
        return len(self.items)

    def add(self, transition):
        # Standard maximal-priority insertion ensures unseen samples are eligible.
        priority = max(self.priorities, default=1.0)
        if len(self.items) < self.capacity:
            self.items.append(transition)
            self.priorities.append(priority)
        else:
            self.items[self.position] = transition
            self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def probabilities(self):
        if not self.items:
            raise ValueError('Cannot sample empty replay')
        maximum = max(self.priorities)
        scaled = [(priority / maximum)**self.alpha for priority in self.priorities]
        total = sum(scaled)
        return [weight / total for weight in scaled]

    def sample(self, batch_size, rng, beta=0.4):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('Batch size must be a positive integer')
        if not math.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError('beta must be in [0, 1]')
        probabilities = self.probabilities()
        indices = rng.choices(range(len(self.items)), weights=probabilities, k=batch_size)
        # Divide (N*P(i))**(-beta) by its maximum over the whole buffer.
        minimum_probability = min(probabilities)
        weights = [(minimum_probability / probabilities[index])**beta for index in indices]
        return [self.items[index] for index in indices], indices, weights

    def update_priorities(self, indices, td_errors):
        if len(indices) != len(td_errors):
            raise ValueError('Indices and TD errors must have equal lengths')
        updates = {}
        for index, error in zip(indices, td_errors):
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.items):
                raise ValueError('Invalid replay index')
            if not math.isfinite(error):
                raise ValueError('TD errors must be finite')
            priority = abs(error) + self.epsilon
            # Repeated draws use the largest absolute error for that entry.
            updates[index] = max(updates.get(index, 0.0), priority)
        for index, priority in updates.items():
            self.priorities[index] = priority
