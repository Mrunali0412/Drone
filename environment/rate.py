import math
from statistics import NormalDist


def calculate_channel_dispersion(sinr):
    """
    Channel dispersion:

    V(gamma) = 1 - 1 / (1 + gamma)^2
    """

    if sinr < 0:
        sinr = 0.0

    dispersion = 1 - 1 / ((1 + sinr) ** 2)

    return dispersion


def calculate_fbl_rate(
    sinr,
    bandwidth,
    blocklength,
    error_probability
):
    """
    Calculate finite-blocklength achievable rate.

    R = W [ log2(1+gamma)
            - sqrt(V(gamma)/n) * Q^-1(epsilon)/ln(2) ]
    """

    if sinr <= 0:
        return 0.0

    # Channel capacity in bits/s/Hz
    capacity = math.log2(1 + sinr)

    # Channel dispersion
    dispersion = calculate_channel_dispersion(sinr)

    # Q^-1(epsilon)
    #
    # Q(x) = 1 - Phi(x)
    # Therefore Q^-1(epsilon) = Phi^-1(1-epsilon)
    q_inverse = NormalDist().inv_cdf(
        1 - error_probability
    )

    # Finite blocklength penalty
    penalty = (
        math.sqrt(dispersion / blocklength)
        * q_inverse
        / math.log(2)
    )

    rate_per_hz = capacity - penalty

    # Rate cannot be negative
    rate_per_hz = max(rate_per_hz, 0.0)

    # Total rate
    rate = bandwidth * rate_per_hz

    return rate