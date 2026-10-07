"""Run: python -B -m examples.masked_power_episode."""
import random

from environment.drone_env import DroneCommunicationEnvironment


def main():
    env = DroneCommunicationEnvironment(max_ttis=6)
    rng = random.Random(42)
    _, info = env.reset()
    masks = info['action_masks']
    for _ in range(env.max_ttis):
        # Placeholder Q-values prefer larger powers; this is not a trained DDQN.
        actions = {drone: env.action_space.select(
            env.action_space.powers, mask, epsilon=0.2, rng=rng
        ) for drone, mask in masks.items()}
        _, reward, terminated, truncated, info = env.step_discrete(actions)
        print(f"TTI {info['completed_ttis']}: {info['sum_rate'] / 1e6:.3f} Mbps; reward {reward:.2f}; indices {actions}")
        if terminated or truncated:
            break
        masks = info['next_action_masks']


if __name__ == '__main__':
    main()
