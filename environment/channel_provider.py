"""Channel providers return linear power gains for every drone/O-RU link."""
import math
from numbers import Real
from typing import Protocol

from environment.channel import calculate_distance, calculate_channel_gain


class ChannelProvider(Protocol):
    def get_channel_gains(self, drones, orus, time_slot):
        """Return {drone_id: {receiving_oru_id: nonnegative_power_gain}}."""
        ...


class DistanceChannelProvider:
    def __init__(self, path_loss_exponent=2.5):
        self.path_loss_exponent = path_loss_exponent

    def get_channel_gains(self, drones, orus, time_slot):
        return {
            drone: {
                oru: calculate_channel_gain(calculate_distance(position, receiver), self.path_loss_exponent)
                for oru, receiver in orus.items()
            }
            for drone, position in drones.items()
        }


def validate_channel_snapshot(gains, drones, orus):
    """Validate and copy provider output before using it in the simulation."""
    if set(gains) != set(drones):
        raise ValueError("Channel snapshot must contain every drone")
    snapshot = {}
    for drone, links in gains.items():
        if set(links) != set(orus):
            raise ValueError("Channel snapshot must contain every receiving O-RU")
        if any(isinstance(gain, bool) or not isinstance(gain, Real)
               or not math.isfinite(gain) or gain < 0 for gain in links.values()):
            raise ValueError("Channel gains must be finite, nonnegative linear power gains")
        snapshot[drone] = dict(links)
    return snapshot
