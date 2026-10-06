import math


def calculate_pf_metric(
    channel_gain,
    transmit_power,
    noise_power,
    average_throughput,
    bandwidth=1.0
):
    """
    Calculate the proportional-fair scheduling metric.

    PF metric = instantaneous achievable rate
                / historical average throughput

    The instantaneous rate used for PF scheduling is:

        bandwidth * log2(1 + P * channel_gain / noise)

    With bandwidth in Hz, use average throughput in bits/s. The default
    bandwidth of 1 retains spectral-efficiency units for standalone callers.

    """

    average_throughput = max(average_throughput, 1e-12)

    snr = (
        transmit_power * channel_gain
    ) / noise_power

    instantaneous_rate = bandwidth * math.log2(1 + snr)

    pf_metric = (
        instantaneous_rate /
        average_throughput
    )

    return pf_metric


def proportional_fair_scheduler(
    drones,
    association,
    channel_gains,
    drone_powers,
    noise_power,
    average_throughput,
    num_rrbs,
    bandwidth=1.0
):
    """
    Allocate RRBs independently inside each O-RU.

    Each O-RU has num_rrbs orthogonal RRBs.

    Within an O-RU:
        - each RRB is assigned to at most one drone
        - drones are ranked according to PF metric

    Returns:

        {
            drone_id: rrb_id
        }
    """

    rrb_assignment = {}

    # Find all O-RUs
    oru_ids = set(association.values())

    for oru_id in oru_ids:

        # Drones associated with this O-RU
        cell_drones = [
            drone_id
            for drone_id in drones
            if association[drone_id] == oru_id
        ]

        # Calculate PF metric for every drone
        pf_metrics = {}

        for drone_id in cell_drones:

            pf_metrics[drone_id] = calculate_pf_metric(
                channel_gain=channel_gains[drone_id],
                transmit_power=drone_powers[drone_id],
                noise_power=noise_power,
                average_throughput=average_throughput[drone_id],
                bandwidth=bandwidth
            )

        # Sort drones from highest PF metric
        sorted_drones = sorted(
            cell_drones,
            key=lambda drone_id: pf_metrics[drone_id],
            reverse=True
        )

        # Assign RRBs
        for rrb_id, drone_id in enumerate(
            sorted_drones[:num_rrbs]
        ):
            rrb_assignment[drone_id] = rrb_id

    return rrb_assignment
