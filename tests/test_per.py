import random
import unittest
import torch

from agents.replay import PrioritizedReplayBuffer, Transition
from agents.ddqn import SharedDDQN, DDQNConfig
from environment.drone_env import DroneCommunicationEnvironment


class TestPrioritizedReplay(unittest.TestCase):
    def test_probabilities_and_importance_weights(self):
        buffer = PrioritizedReplayBuffer(2, alpha=1, epsilon=1)
        buffer.add('low')
        buffer.add('high')
        buffer.update_priorities([0, 1], [0, 2])
        self.assertEqual(buffer.probabilities(), [0.25, 0.75])
        batch, indices, weights = buffer.sample(1000, random.Random(42), beta=1)
        self.assertGreater(indices.count(1), 650)
        for index, weight in zip(indices, weights):
            self.assertAlmostEqual(weight, 1 if index == 0 else 1 / 3)

    def test_zero_alpha_and_beta(self):
        buffer = PrioritizedReplayBuffer(2, alpha=0)
        buffer.add(1)
        buffer.add(2)
        buffer.update_priorities([0, 1], [0, 100])
        self.assertEqual(buffer.probabilities(), [0.5, 0.5])
        self.assertEqual(buffer.sample(3, random.Random(0), beta=0)[2], [1, 1, 1])

    def test_ring_overwrite_and_duplicate_priority_updates(self):
        buffer = PrioritizedReplayBuffer(2)
        buffer.add('a')
        buffer.add('b')
        buffer.update_priorities([0, 0], [5, 2])
        self.assertAlmostEqual(buffer.priorities[0], 5 + 1e-6)
        buffer.add('c')
        self.assertEqual(buffer.items, ['c', 'b'])
        self.assertAlmostEqual(buffer.priorities[0], 5 + 1e-6)
        buffer.update_priorities([0], [0])
        self.assertGreater(buffer.probabilities()[0], 0)

    def test_invalid_update_is_atomic(self):
        buffer = PrioritizedReplayBuffer(2)
        buffer.add(1)
        buffer.add(2)
        with self.assertRaises(ValueError):
            buffer.update_priorities([0, 1], [3, float('nan')])
        self.assertEqual(buffer.priorities, [1, 1])
        with self.assertRaises(ValueError):
            PrioritizedReplayBuffer(2).sample(1, random.Random())

    def test_learning_updates_priorities_and_beta(self):
        torch.set_num_threads(1)
        env = DroneCommunicationEnvironment()
        agent = SharedDDQN(env.action_space.powers, DDQNConfig(batch_size=2, per_beta_updates=2))
        self.assertEqual(agent.beta, 0.4)
        for reward in (1., 4.):
            agent.replay.add(Transition((0.,) * 4, 0, reward, (0.,) * 4, (False,) * 20, 0.))
        agent.learn()
        self.assertNotEqual(agent.replay.priorities, [1, 1])
        self.assertAlmostEqual(agent.beta, 0.7)
        agent.learn()
        self.assertEqual(agent.beta, 1)
        agent.learn()
        self.assertEqual(agent.beta, 1)

    def test_uniform_baseline_learns(self):
        torch.set_num_threads(1)
        env = DroneCommunicationEnvironment()
        agent = SharedDDQN(env.action_space.powers, DDQNConfig(batch_size=1, prioritized_replay=False))
        agent.replay.add(Transition((0.,) * 4, 0, 1., (0.,) * 4, (False,) * 20, 0.))
        self.assertIsNotNone(agent.learn())


if __name__ == '__main__':
    unittest.main()
