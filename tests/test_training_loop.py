import csv
import json
import tempfile
import unittest
from pathlib import Path
import torch

from agents.ddqn import DDQNConfig, SharedDDQN
from environment.drone_env import DroneCommunicationEnvironment
from environment.reward import RewardConfig
from training.loop import train, evaluate, TrainingConfig


class TestTrainingLoop(unittest.TestCase):
    def test_periodic_evaluation_checkpoints_and_logs(self):
        torch.set_num_threads(1)
        env = DroneCommunicationEnvironment(max_ttis=4)
        evaluation = DroneCommunicationEnvironment(max_ttis=4, reward_config=RewardConfig(adaptive=False))
        agent = SharedDDQN(env.action_space.powers, DDQNConfig(batch_size=4))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = train(env, evaluation, agent, output, TrainingConfig(3, 2, 1, 2))
            self.assertEqual(len(result['training']), 3)
            self.assertEqual([row['episode'] for row in result['evaluations']], [2, 3])
            for filename in ('best.pt', 'latest.pt', 'policy.pt', 'metrics.json', 'run_config.json', 'training.csv'):
                self.assertTrue((output / filename).exists())
            with (output / 'training.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            self.assertEqual(json.loads((output / 'metrics.json').read_text())['best_policy_evaluation']['c3_violations'], 0)
            with self.assertRaises(FileExistsError):
                train(env, evaluation, agent, output, TrainingConfig(1, 1, 1, 1))

    def test_evaluation_does_not_modify_training_state(self):
        torch.set_num_threads(1)
        env = DroneCommunicationEnvironment(max_ttis=3, reward_config=RewardConfig(adaptive=False))
        agent = SharedDDQN(env.action_space.powers)
        before = [parameter.clone() for parameter in agent.online.parameters()]
        random_state = agent.rng.getstate()
        result = evaluate(env, agent, 2)
        self.assertEqual(result['episodes'], 2)
        self.assertEqual(len(agent.replay), 0)
        self.assertEqual(agent.updates, 0)
        self.assertEqual(agent.training_ttis, 0)
        self.assertEqual(agent.rng.getstate(), random_state)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, agent.online.parameters())))

    def test_invalid_intervals(self):
        for kwargs in ({'episodes': 0}, {'eval_interval': 0}, {'eval_episodes': True}, {'checkpoint_interval': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
