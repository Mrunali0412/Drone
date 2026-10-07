"""Discrete power choices and C3 masks, independent of a learning framework."""
import math
from numbers import Integral, Real


class NoFeasibleActionError(ValueError):
    """No configured power satisfies C3; no unsafe fallback is selected."""


class PowerActionSpace:
    def __init__(self, min_power, max_power, count):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 2:
            raise ValueError("At least two power levels are required")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or value < 0 for value in (min_power, max_power)) or min_power > max_power:
            raise ValueError("Power bounds must be finite, nonnegative and ordered")
        # Linear spacing is a documented reproduction assumption.
        self.powers = tuple(
            min_power if index == 0 else max_power if index == count - 1
            else min_power + (max_power - min_power) * index / (count - 1)
            for index in range(count)
        )

    def power(self, index):
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self.powers):
            raise ValueError("Action index must be an integer within the power action space")
        return self.powers[index]

    def mask(self, link_gains, serving_oru, interference_limit):
        worst_gain = max((gain for oru, gain in link_gains.items() if oru != serving_oru), default=0.0)
        return tuple(power * worst_gain <= interference_limit for power in self.powers)

    def select(self, q_values, mask, *, epsilon=0.0, rng=None):
        """Choose a feasible greedy action or explore using an explicit RNG."""
        import math
        from numbers import Real

        if len(q_values) != len(self.powers) or len(mask) != len(self.powers):
            raise ValueError("Q-values and mask must match the number of powers")
        if any(not isinstance(value, Real) or not math.isfinite(value) for value in q_values):
            raise ValueError("Q-values must be finite numbers")
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        feasible = [index for index, allowed in enumerate(mask) if allowed]
        if not feasible:
            raise NoFeasibleActionError("No power satisfies C3; adjust the scenario or define an outage policy")
        if epsilon:
            if rng is None:
                raise ValueError("Exploration requires an explicit random generator")
            if rng.random() < epsilon:
                return rng.choice(feasible)
        return max(feasible, key=lambda index: q_values[index])
