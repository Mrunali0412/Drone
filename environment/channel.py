import math


def calculate_distance(point1, point2):
    #Calculate 3D Euclidean distance between two points.
    #point = (x, y, z)

    x1, y1, z1 = point1
    x2, y2, z2 = point2

    distance = math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2 +
        (z1 - z2) ** 2
    )

    return distance


def calculate_channel_gain(distance, path_loss_exponent=2.5):
    
    #Simplified channel gain model.
    #G = 1 / d^alpha
  
    distance = max(distance, 1.0)

    gain = 1 / (distance ** path_loss_exponent)

    return gain


def calculate_received_power(transmit_power, channel_gain):
    """
    Received power = transmit power × channel gain
    """

    return transmit_power * channel_gain

def calculate_interference(
    target_oru,
    serving_drone,
    drones,
    drone_powers,
    orus,
    path_loss_exponent=2.5
):
    """
    Calculate interference received at target_oru
    from all drones except the serving drone.
    """

    interference = 0.0

    target_position = orus[target_oru]

    for drone_id, drone_position in drones.items():

        # Do not count the desired drone's signal as interference
        if drone_id == serving_drone:
            continue

        transmit_power = drone_powers[drone_id]

        distance = calculate_distance(
            drone_position,
            target_position
        )

        gain = calculate_channel_gain(
            distance,
            path_loss_exponent
        )

        received_power = calculate_received_power(
            transmit_power,
            gain
        )

        interference += received_power

    return interference

def calculate_sinr(signal_power, interference, noise_power):
    """
    SINR = signal / (interference + noise)
    """

    denominator = interference + noise_power

    if denominator <= 0:
        return 0.0

    sinr = signal_power / denominator

    return sinr

