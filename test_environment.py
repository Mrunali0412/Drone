from environment.drone_env import DroneCommunicationEnvironment


env = DroneCommunicationEnvironment()

results = env.calculate_all_metrics()


for drone_id, metrics in results.items():

    print("=" * 50)

    print(f"Drone {drone_id}")

    print(f"Serving O-RU: {metrics['serving_oru']}")

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

    print(
        f"SINR: "
        f"{metrics['sinr']:.6f}"
    )

    print(
        f"FBL Rate: "
        f"{metrics['rate']:.2f} bits/s"
    )