"""Shared network reward and adaptive minimum-rate penalty."""
import math
from dataclasses import dataclass
from numbers import Real


@dataclass(frozen=True)
class RewardConfig:
    mode: str = 'cooperative'
    initial_multiplier: float = 0.0
    learning_rate: float = 1e-6  # Multiplier increment per bit/s of mean shortfall.
    penalize_unscheduled: bool = True
    adaptive: bool = True

    def __post_init__(self):
        if self.mode not in ('cooperative', 'sum_rate'):
            raise ValueError("Reward mode must be cooperative or sum_rate")
        for value in (self.initial_multiplier, self.learning_rate):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError("Penalty settings must be finite and nonnegative")
        if not isinstance(self.penalize_unscheduled, bool) or not isinstance(self.adaptive, bool):
            raise ValueError("Reward switches must be Boolean")


class CooperativeReward:
    def __init__(self, config=None):
        self.config = config if config is not None else RewardConfig()
        self.multiplier = self.config.initial_multiplier

    def evaluate(self, metrics, min_rate):
        """Pure calculation; reading reward does not update the multiplier."""
        shortfalls = {
            drone: max(0.0, min_rate - result['rate'])
            for drone, result in metrics.items()
            if self.config.penalize_unscheduled or result['scheduled']
        }
        sum_rate = sum(result['rate'] for result in metrics.values())
        total = sum(shortfalls.values())
        mean = total / len(shortfalls) if shortfalls else 0.0
        penalty = self.multiplier * total if self.config.mode == 'cooperative' else 0.0
        next_multiplier = self.multiplier
        if self.config.mode == 'cooperative' and self.config.adaptive:
            next_multiplier = max(0.0, self.multiplier + self.config.learning_rate * mean)
        if not all(math.isfinite(value) for value in (sum_rate, total, penalty, next_multiplier)):
            raise ValueError("Reward calculation overflowed; review penalty scaling")
        return {
            'reward_type': self.config.mode,
            'sum_rate': sum_rate,
            'penalty_shortfalls': shortfalls,
            'total_rate_shortfall': total,
            'mean_rate_shortfall': mean,
            'penalty_multiplier': self.multiplier,
            'next_penalty_multiplier': next_multiplier,
            'rate_penalty': penalty,
            'reward': sum_rate - penalty,
        }

    def update(self, components):
        self.multiplier = components['next_penalty_multiplier']
