from environment.channel import (
    calculate_distance,
    calculate_channel_gain,
    calculate_channel_gains,
    calculate_received_power,
    calculate_interference,
    calculate_interference_to_neighbors,
    calculate_sinr
)

from environment.rate import calculate_fbl_rate
from environment.scheduler import proportional_fair_scheduler


class DroneCommunicationEnvironment:

    def __init__(self):

        # --------------------------------
        # 1. O-RU positions
        # --------------------------------

        self.orus = {
            0: (0, 0, 50),
            1: (250, 0, 50)
        }

        # --------------------------------
        # 2. Drone positions
        # --------------------------------

        self.drones = {
            0: (50, 0, 60),
            1: (80, 20, 60),
            2: (200, 0, 60),
            3: (220, 20, 60)
        }

        # --------------------------------
        # 3. Drone → O-RU association
        # --------------------------------

        self.association = {
            0: 0,
            1: 0,
            2: 1,
            3: 1
        }

        # --------------------------------
        # 4. Communication parameters
        # --------------------------------

        self.bandwidth = 180e3

        self.blocklength = 200

        self.error_probability = 1e-5

        self.noise_power = 1e-9

        # --------------------------------
        # 5. Initial transmit powers
        # --------------------------------

        self.drone_powers = {
            0: 0.1,
            1: 0.1,
            2: 0.1,
            3: 0.1
        }

        # Path-loss exponent
        self.path_loss_exponent = 2.5

        self.num_rrbs = 2

        self.average_throughput = {
            0: 1.0,
            1: 1.0,
            2: 1.0,
            3: 1.0
        }


    def calculate_drone_metrics(self, drone_id):

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
    
        channel_gain = calculate_channel_gain(
            distance,
            self.path_loss_exponent
        )
        
        # --------------------------------
        # Desired signal
        # --------------------------------

        signal_power = calculate_received_power(
            self.drone_powers[drone_id],
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
            path_loss_exponent=self.path_loss_exponent
        )



        interference_to_neighbors = calculate_interference_to_neighbors(
            drone_id=drone_id,
            drones=self.drones,
            drone_powers=self.drone_powers,
            orus=self.orus,
            association=self.association,
            path_loss_exponent=self.path_loss_exponent
        )
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
            "distance": distance,
            "channel_gain": channel_gain,
            "signal_power": signal_power,
            "interference": interference,
            "interference_to_neighbors": interference_to_neighbors,
            "sinr": sinr,
            "rate": rate
        }

    def schedule_rrbs(self):
        channel_gains = calculate_channel_gains(
            drones=self.drones,
            association=self.association,
            orus=self.orus,
            path_loss_exponent=self.path_loss_exponent
        )

        rrb_assignment = proportional_fair_scheduler(
            drones=self.drones,
            association=self.association,
            channel_gains=channel_gains,
            drone_powers=self.drone_powers,
            noise_power=self.noise_power,
            average_throughput=self.average_throughput,
            num_rrbs=self.num_rrbs
        )
        return rrb_assignment


    def calculate_all_metrics(self):

        results = {}

        for drone_id in self.drones:

            results[drone_id] = (
                self.calculate_drone_metrics(drone_id)
            )

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
    