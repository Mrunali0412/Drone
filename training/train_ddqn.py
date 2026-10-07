"""python -B -m training.train_ddqn --episodes 5 --ttis 30"""
import argparse
from dataclasses import replace
import torch

from agents.ddqn import SharedDDQN, DDQNConfig
from environment.drone_env import DroneCommunicationEnvironment
from training.loop import TrainingConfig, train
from environment.reward import RewardConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--ttis', type=int, default=100)
    parser.add_argument('--output', default='artifacts/ddqn_training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epsilon-decay-ttis', type=int, default=10000)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--eval-episodes', type=int, default=3)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--uniform-replay', action='store_true', help='Disable PER for a baseline run')
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error('episodes must be positive')
    torch.set_num_threads(1)
    env = DroneCommunicationEnvironment(max_ttis=args.ttis)
    try:
        loop = TrainingConfig(args.episodes, args.eval_interval, args.eval_episodes, args.checkpoint_interval)
        agent = SharedDDQN(env.action_space.powers, DDQNConfig(prioritized_replay=not args.uniform_replay,
            seed=args.seed, batch_size=args.batch_size, epsilon_decay_ttis=args.epsilon_decay_ttis))
        eval_env = DroneCommunicationEnvironment(max_ttis=args.ttis,
            reward_config=replace(RewardConfig(), adaptive=False))
    except ValueError as error:
        parser.error(str(error))
    result = train(env, eval_env, agent, args.output, loop)
    print(f"Saved {args.output}; best policy: {result['best_policy_evaluation']['mean_sum_rate'] / 1e6:.3f} Mbps")


if __name__ == '__main__':
    main()
