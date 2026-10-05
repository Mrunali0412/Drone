
from environment.drone_env import DroneCommunicationEnvironment


env = DroneCommunicationEnvironment()


# ============================================================
# 1. TEST COMMUNICATION METRICS + CONSTRAINTS
# ============================================================

results = env.calculate_all_metrics()

print("=" * 60)
print("COMMUNICATION METRICS + CONSTRAINTS")
print("=" * 60)


for drone_id, metrics in results.items():

    constraints = env.check_constraints(
        drone_id,
        metrics
    )

    print("=" * 60)

    print(f"Drone {drone_id}")

    print(
        f"Serving O-RU: "
        f"{metrics['serving_oru']}"
    )

    print(
        f"Distance: "
        f"{metrics['distance']:.2f} m"
    )

    print(
        f"Channel Gain: "
        f"{metrics['channel_gain']:.6e}"
    )

    print(
        f"Signal Power: "
        f"{metrics['signal_power']:.6e} W"
    )

    print(
        f"Interference: "
        f"{metrics['interference']:.6e} W"
    )

    print("Interference to Neighbors:")
    for oru_id, interference in metrics["interference_to_neighbors"].items():
        print(f"  O-RU {oru_id}: {interference:.6e} W")

    print(
        f"SINR: "
        f"{metrics['sinr']:.6f}"
    )

    print(
        f"FBL Rate: "
        f"{metrics['rate']:.2f} bits/s"
    )

    print()
    print("Constraints:")

    print(
        f"C1 Power Constraint: "
        f"{constraints['power_constraint']}"
    )

    print(
        f"C2 Minimum Rate Constraint: "
        f"{constraints['rate_constraint']}"
    )

    print(
        f"C3 Interference Constraint: "
        f"{constraints['interference_constraint']}"
    )

    print(
        f"Overall Feasible: "
        f"{constraints['all_constraints_satisfied']}"
    )


# ============================================================
# 2. TEST PF RRB SCHEDULING
# ============================================================

rrb_assignment = env.schedule_rrbs()

print()
print("=" * 60)
print("PF RRB SCHEDULING")
print("=" * 60)

for drone_id, rrb in rrb_assignment.items():

    oru = env.association[drone_id]

    print(
        f"Drone {drone_id} "
        f"-> O-RU {oru} "
        f"-> RRB {rrb}"
    )