from copy import deepcopy
from collections.abc import Mapping
import math
from numbers import Real

from environment.channel import (
    calculate_distance,
    calculate_received_power,
    calculate_interference,
    calculate_interference_to_neighbors,
    calculate_sinr
)

from environment.rate import calculate_fbl_rate
from environment.scheduler import proportional_fair_scheduler
from environment.mobility import DroneMobility
from environment.config import NetworkConfig
from environment.reward import CooperativeReward
from environment.actions import PowerActionSpace, NoFeasibleActionError
from environment.channel_provider import DistanceChannelProvider, validate_channel_snapshot


class DroneCommunicationEnvironment:

    def __init__(self, waypoints=None, num_ttis_per_ts=3, throughput_smoothing=0.1, *, config=None, channel_provider=None, max_ttis=100, reward_config=None):

        if isinstance(max_ttis, bool) or not isinstance(max_ttis, int) or max_ttis <= 0:
            raise ValueError("max_ttis must be a positive integer")
        self.max_ttis = max_ttis
        self.elapsed_ttis = 0
        self.reward_model = CooperativeReward(reward_config)

        if (isinstance(throughput_smoothing, bool)
                or not isinstance(throughput_smoothing, Real)
                or not math.isfinite(throughput_smoothing)
                or not 0 < throughput_smoothing <= 1):
            raise ValueError("throughput_smoothing must be finite and in (0, 1]")
        self.throughput_smoothing = throughput_smoothing

        settings = deepcopy(config) if config is not None else NetworkConfig()
        settings.__post_init__()
        self.config = settings
        self.action_space = PowerActionSpace(settings.min_power, settings.max_power, settings.num_power_levels)
        self.orus = {key: tuple(value) for key, value in settings.orus.items()}
        self.drones = {key: tuple(value) for key, value in settings.drones.items()}
        self.association = dict(settings.association)
        for name in ('bandwidth', 'blocklength', 'error_probability', 'noise_power',
                     'min_power', 'max_power', 'min_rate', 'max_interference',
                     'path_loss_exponent', 'num_rrbs'):
            setattr(self, name, getattr(settings, name))
        if waypoints is None:
            waypoints = {drone: [position] for drone, position in self.drones.items()}
        if set(waypoints) != set(self.drones):
            raise ValueError('Waypoints must match the configured drone IDs')
        self.mobility = DroneMobility(waypoints, num_ttis_per_ts)
        self.drones = self.mobility.get_all_positions()
        self.drone_powers = dict.fromkeys(self.drones, settings.initial_power)
        self.average_throughput = dict.fromkeys(self.drones, 1.0)
        self.rrb_assignment = None
        self.channel_provider = (channel_provider if channel_provider is not None
                                 else DistanceChannelProvider(self.path_loss_exponent))
        self.channel_gains = self._load_channel_snapshot(0, self.drones)
        self.previous_metrics = {
            drone: {"sinr": 0.0, "rate": 0.0, "power": settings.initial_power}
            for drone in self.drones
        }
        self._reset_config = deepcopy(settings)
        self._reset_waypoints = deepcopy(self.mobility.waypoints)

    def reset(self):
        """Restart the mission; return (local observations, scheduling info).

        Providers are queried again at slot zero. A stochastic provider must
        manage its own reproducibility; this environment has no random sampler.
        """
        self.__init__(
            self._reset_waypoints, self.mobility.num_ttis_per_ts,
            self.throughput_smoothing, config=self._reset_config,
            channel_provider=self.channel_provider, max_ttis=self.max_ttis,
            reward_config=self.reward_model.config,
        )
        return self.get_observations(), self._decision_info()

    def get_observations(self):
        """Raw features: serving gain, previous SINR, rate (bits/s), power (W).

        Observations include all drones; scheduling info identifies which
        drones need actions. An idle drone's previous transmitted power is zero.
        """
        if self.rrb_assignment is None:
            self.schedule_rrbs()
        return {
            drone: (
                self.channel_gains[drone][self.association[drone]],
                self.previous_metrics[drone]["sinr"],
                self.previous_metrics[drone]["rate"],
                self.previous_metrics[drone]["power"],
            )
            for drone in self.drones
        }

    def _decision_info(self):
        return {
            "rrb_assignment": dict(self.rrb_assignment or {}),
            "time_slot": self.time_slot,
            "tti_in_time_slot": self.tti_in_time_slot,
            "elapsed_ttis": self.elapsed_ttis,
            "power_levels": self.action_space.powers,
            "action_masks": self.get_action_masks(),
        }

    def get_action_masks(self):
        """C3 masks for scheduled drones; idle drones have no decision to make."""
        if self.rrb_assignment is None:
            self.schedule_rrbs()
        return {
            drone: self.action_space.mask(
                self.channel_gains[drone], self.association[drone], self.max_interference
            )
            for drone in self.rrb_assignment
        }

    def step_discrete(self, actions):
        """Execute feasible power indices; step() remains the watts baseline API.

        Empty masks raise NoFeasibleActionError. No below-bound silent action
        or minimum-power fallback is introduced into the paper's action space.
        """
        if self.elapsed_ttis >= self.max_ttis:
            raise RuntimeError("Episode finished; call reset() before stepping")
        if not isinstance(actions, Mapping):
            raise ValueError("Actions must map scheduled drone IDs to power indices")
        old_assignment = self.rrb_assignment
        try:
            masks = self.get_action_masks()
            if set(actions) != set(masks):
                raise ValueError("Provide exactly one index for each scheduled drone")
            powers = {}
            for drone, index in actions.items():
                power = self.action_space.power(index)
                if not any(masks[drone]):
                    raise NoFeasibleActionError(f"Drone {drone} has no C3-feasible power")
                if not masks[drone][index]:
                    raise ValueError(f"Drone {drone}: selected power violates C3")
                powers[drone] = power
            result = self.step(powers)
        except Exception:
            self.rrb_assignment = old_assignment
            raise
        result[4]["action_indices"] = dict(actions)
        return result

    def step(self, actions):
        """Transmit once with {scheduled_drone_id: power_in_watts}.

        Returns (observations, global_reward, terminated, truncated, info).
        Reward is sum rate minus the configured minimum-rate penalty. The fixed TTI
        horizon is a truncation; waypoint completion alone does not terminate.
        Actions are continuous powers for now, not discrete action indices.
        """
        if self.elapsed_ttis >= self.max_ttis:
            raise RuntimeError("Episode finished; call reset() before stepping")
        if not isinstance(actions, Mapping):
            raise ValueError("Actions must map scheduled drone IDs to powers")
        old_assignment = self.rrb_assignment
        if self.rrb_assignment is None:
            self.schedule_rrbs()
        try:
            if set(actions) != set(self.rrb_assignment):
                raise ValueError("Provide exactly one power for each scheduled drone")
            for power in actions.values():
                if (isinstance(power, bool) or not isinstance(power, Real)
                        or not math.isfinite(power)
                        or not self.min_power <= power <= self.max_power):
                    raise ValueError("Action powers must be finite and within power bounds")
        except ValueError:
            self.rrb_assignment = old_assignment
            raise

        old_powers = self.drone_powers.copy()
        self.drone_powers.update(actions)
        try:
            metrics = self.calculate_all_metrics()
            constraints = {drone: self.check_constraints(drone, result)
                           for drone, result in metrics.items()}
            info = self._decision_info()
            info.update({
                "metrics": metrics,
                "constraints": constraints,
                "rate_shortfalls": {drone: max(0.0, self.min_rate - result["rate"])
                                    for drone, result in metrics.items()},
            })
            components = self.reward_model.evaluate(metrics, self.min_rate)
            info.update(components)
            reward = components['reward']
            self.advance_tti()
        except Exception:
            self.drone_powers = old_powers
            self.rrb_assignment = old_assignment
            raise
        observations = self.get_observations()
        info["next_rrb_assignment"] = dict(self.rrb_assignment)
        info["next_action_masks"] = self.get_action_masks()
        info["completed_ttis"] = self.elapsed_ttis
        return observations, reward, False, self.elapsed_ttis >= self.max_ttis, info

    def _load_channel_snapshot(self, time_slot, drones):
        gains = self.channel_provider.get_channel_gains(dict(drones), dict(self.orus), time_slot)
        return validate_channel_snapshot(gains, drones, self.orus)

    @property
    def time_slot(self):
        return self.mobility.time_slot

    @property
    def tti_in_time_slot(self):
        return self.mobility.tti_in_time_slot

    def advance_tti(self):
        """Record achieved throughput, then advance to the next TTI.

        History updates once per completed TTI, before movement. Reading
        metrics or scheduling alone does not update history.
        Returns whether a time-slot boundary was crossed.
        """
        if self.elapsed_ttis >= self.max_ttis:
            raise RuntimeError("Episode finished; call reset() before advancing")
        # Load the next snapshot before committing clock/history changes.
        next_gains = None
        if self.tti_in_time_slot + 1 == self.mobility.num_ttis_per_ts:
            next_positions = {
                drone: points[min(self.mobility.current_waypoint[drone] + 1, len(points) - 1)]
                for drone, points in self.mobility.waypoints.items()
            }
            next_gains = self._load_channel_snapshot(self.time_slot + 1, next_positions)
        metrics = self.calculate_all_metrics()
        components = self.reward_model.evaluate(metrics, self.min_rate)
        rho = self.throughput_smoothing
        self.average_throughput = {
            drone_id: (1 - rho) * self.average_throughput[drone_id]
            + rho * result["rate"]
            for drone_id, result in metrics.items()
        }
        self.previous_metrics = {
            drone: {"sinr": result["sinr"], "rate": result["rate"],
                    "power": self.drone_powers[drone] if result["scheduled"] else 0.0}
            for drone, result in metrics.items()
        }
        self.elapsed_ttis += 1
        self.reward_model.update(components)
        slot_changed = self.mobility.advance_tti()
        self.drones = self.mobility.get_all_positions()
        self.rrb_assignment = None
        if slot_changed:
            self.channel_gains = next_gains
        return slot_changed

    def calculate_drone_metrics(self, drone_id):

        if self.rrb_assignment is None:
            self.schedule_rrbs()
        scheduled = drone_id in self.rrb_assignment

        # Which O-RU serves this drone?
        serving_oru = self.association[drone_id]

        drone_position = self.drones[drone_id]
        oru_position = self.orus[serving_oru]

        # --------------------------------
        # Distance
        # --------------------------------
        distance = calculate_distance(
            drone_position,
            oru_position
        )

        # --------------------------------
        # Channel gain
        # --------------------------------
        channel_gain = self.channel_gains[drone_id][serving_oru]

        # --------------------------------
        # Desired signal
        # --------------------------------
        signal_power = calculate_received_power(
            self.drone_powers[drone_id] if scheduled else 0.0,
            channel_gain
        )

        # --------------------------------
        # Interference
        # --------------------------------
        interference = calculate_interference(
            target_oru=serving_oru,
            serving_drone=drone_id,
            drones=self.drones,
            drone_powers=self.drone_powers,
            orus=self.orus,
            association=self.association,
            rrb_assignment=self.rrb_assignment,
            path_loss_exponent=self.path_loss_exponent,
            channel_gains=self.channel_gains
        )

        interference_to_neighbors = calculate_interference_to_neighbors(
            drone_id=drone_id,
            drones=self.drones,
            drone_powers=self.drone_powers,
            orus=self.orus,
            association=self.association,
            path_loss_exponent=self.path_loss_exponent,
            channel_gains=self.channel_gains
        )
        if not scheduled:
            interference_to_neighbors = {
                oru_id: 0.0 for oru_id in interference_to_neighbors
            }

        # --------------------------------
        # SINR
        # --------------------------------
        sinr = calculate_sinr(
            signal_power,
            interference,
            self.noise_power
        )

        # --------------------------------
        # FBL rate
        # --------------------------------
        rate = calculate_fbl_rate(
            sinr=sinr,
            bandwidth=self.bandwidth,
            blocklength=self.blocklength,
            error_probability=self.error_probability
        )

        return {
            "drone_id": drone_id,
            "serving_oru": serving_oru,
            "scheduled": scheduled,
            "rrb_id": self.rrb_assignment.get(drone_id),
            "distance": distance,
            "channel_gain": channel_gain,
            "signal_power": signal_power,
            "interference": interference,
            "interference_to_neighbors": interference_to_neighbors,
            "sinr": sinr,
            "rate": rate
        }

    def schedule_rrbs(self):
        channel_gains = {
            drone: self.channel_gains[drone][self.association[drone]]
            for drone in self.drones
        }

        rrb_assignment = proportional_fair_scheduler(
            drones=self.drones,
            association=self.association,
            channel_gains=channel_gains,
            drone_powers=self.drone_powers,
            noise_power=self.noise_power,
            average_throughput=self.average_throughput,
            num_rrbs=self.num_rrbs,
            bandwidth=self.bandwidth
        )
        # Freeze this TTI's assignment until explicitly rescheduled or advanced.
        self.rrb_assignment = rrb_assignment.copy()
        return rrb_assignment

    def calculate_all_metrics(self):
        results = {}

        for drone_id in self.drones:
            results[drone_id] = self.calculate_drone_metrics(drone_id)

        return results

    def check_constraints(self, drone_id, metrics):
        """
        Check all three optimization constraints
        for a given drone.
        """

        power = self.drone_powers[drone_id]
        rate = metrics["rate"]
        interference = metrics["interference"]

        # --------------------------------
        # C1: Power constraint
        # --------------------------------
        power_constraint = (
            self.min_power <= power <= self.max_power
        )

        # --------------------------------
        # C2: Minimum rate constraint
        # --------------------------------
        rate_constraint = (
            rate >= self.min_rate
        )

        # --------------------------------
        # C3: Interference constraint
        # --------------------------------
        neighbor_interference = max(
            metrics["interference_to_neighbors"].values(),
            default=0.0
        )

        interference_constraint = (
            neighbor_interference <= self.max_interference
        )

        # --------------------------------
        # Overall feasibility
        # --------------------------------
        all_constraints_satisfied = (
            power_constraint
            and rate_constraint
            and interference_constraint
        )

        return {
            "power_constraint": power_constraint,
            "rate_constraint": rate_constraint,
            "interference_constraint": interference_constraint,
            "all_constraints_satisfied": all_constraints_satisfied
        }

