"""Run from the project root: python -B -m examples.fixed_power_episode."""
from environment.drone_env import DroneCommunicationEnvironment


def main():
    env = DroneCommunicationEnvironment(max_ttis=6)
    observations, info = env.reset()
    for _ in range(env.max_ttis):
        actions = {drone: 0.1 for drone in info['rrb_assignment']}
        observations, reward, terminated, truncated, result = env.step(actions)
        print(f"TTI {result['completed_ttis']}: sum rate {result['sum_rate'] / 1e6:.3f} Mbps; reward {reward:.2f}")
        if terminated or truncated:
            break
        info = {'rrb_assignment': result['next_rrb_assignment']}


if __name__ == '__main__':
    main()
