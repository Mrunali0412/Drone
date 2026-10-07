"""Simulation settings; powers are watts and rates are bits/s."""
import math
from dataclasses import dataclass, field
from numbers import Real


@dataclass
class NetworkConfig:
    orus: dict = field(default_factory=lambda: {0: (0, 0, 50), 1: (250, 0, 50)})
    drones: dict = field(default_factory=lambda: {
        0: (50, 0, 60), 1: (80, 20, 60), 2: (200, 0, 60), 3: (220, 20, 60)
    })
    association: dict = field(default_factory=lambda: {0: 0, 1: 0, 2: 1, 3: 1})
    bandwidth: float = 180e3  # Per RRB, Hz.
    blocklength: int = 200
    error_probability: float = 1e-5
    noise_power: float = 1e-9
    initial_power: float = 0.1
    min_power: float = 0.01
    max_power: float = 0.5
    min_rate: float = 1e5
    max_interference: float = 2e-7
    path_loss_exponent: float = 2.5
    num_rrbs: int = 2
    num_power_levels: int = 20

    def __post_init__(self):
        if not self.orus or not self.drones:
            raise ValueError("At least one O-RU and drone are required")
        for positions in (self.orus, self.drones):
            for point in positions.values():
                if len(point) != 3 or any(
                    isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
                    for x in point
                ):
                    raise ValueError("Positions must contain three finite coordinates")
        if set(self.association) != set(self.drones) or any(
            oru not in self.orus for oru in self.association.values()
        ):
            raise ValueError("Every drone must be associated with an existing O-RU")
        for name in ('bandwidth', 'noise_power', 'path_loss_exponent', 'max_power'):
            self._number(name, positive=True)
        for name in ('initial_power', 'min_power', 'min_rate', 'max_interference'):
            self._number(name)
        self._number('error_probability', positive=True)
        if self.error_probability >= 1:
            raise ValueError("error_probability must be in (0, 1)")
        if not self.min_power <= self.initial_power <= self.max_power:
            raise ValueError("initial_power must lie between min_power and max_power")
        for name, minimum in (('blocklength', 1), ('num_rrbs', 0), ('num_power_levels', 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

    def _number(self, name, positive=False):
        value = getattr(self, name)
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value < 0 or (positive and value == 0)):
            raise ValueError(f"Invalid {name}")
