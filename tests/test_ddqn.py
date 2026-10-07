import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from agents.ddqn import SharedDDQN, DDQNConfig, ddqn_targets
from agents.replay import Transition
from environment.actions import NoFeasibleActionError
from environment.config import NetworkConfig
from environment.drone_env import DroneCommunicationEnvironment
from training.runner import run_episode


class ConstantQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values)

    def forward(self, states):
        return self.values.expand(len(states), -1)


class TestDDQN(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_double_q_target_uses_online_choice_and_target_value(self):
        result = ddqn_targets(ConstantQ([1., 3., 100.]), ConstantQ([20., 5., 999.]),
            torch.zeros(2, 4), torch.tensor([[True, True, False], [False, False, False]]),
            torch.tensor([2., 7.]), torch.tensor([0.9, 0.]))
        self.assertAlmostEqual(result[0].item(), 6.5)
        self.assertEqual(result[1].item(), 7)

    def test_empty_nonterminal_mask_is_rejected(self):
        with self.assertRaises(NoFeasibleActionError):
            ddqn_targets(ConstantQ([0., 1.]), ConstantQ([0., 1.]), torch.zeros(1, 4),
                         torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1), torch.ones(1))

    def test_learning_changes_online_and_synchronizes_target(self):
        env = DroneCommunicationEnvironment()
        agent = SharedDDQN(env.action_space.powers, DDQNConfig(batch_size=2, target_update=2))
        initial = [p.clone() for p in agent.online.parameters()]
        for _ in range(2):
            agent.replay.add(Transition((0., 0., 0., 0.), 0, 1., (0., 0., 0., 0.), (False,) * 20, 0.))
        self.assertIsNotNone(agent.learn())
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial, agent.online.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(initial, agent.target.parameters())))
        agent.learn()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(agent.online.parameters(), agent.target.parameters())))

    def test_training_checkpoint_and_inference(self):
        env = DroneCommunicationEnvironment(max_ttis=8)
        agent = SharedDDQN(env.action_space.powers, DDQNConfig(batch_size=4))
        result = run_episode(env, agent, training=True)
        self.assertGreater(agent.updates, 0)
        self.assertEqual(result['c3_violations'], 0)
        observations, info = env.reset()
        expected = {d: agent.act(observations[d], mask) for d, mask in info['action_masks'].items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pt'
            agent.save(path)
            loaded = SharedDDQN.load(path)
            actual = {d: loaded.act(observations[d], mask) for d, mask in info['action_masks'].items()}
            self.assertEqual(actual, expected)
            evaluation = run_episode(env, loaded)
            self.assertEqual(evaluation['c3_violations'], 0)
            self.assertEqual(len(loaded.replay), 0)
            self.assertEqual(loaded.updates, agent.updates)

    def test_waiting_drones_have_discounted_multi_tti_transitions(self):
        env = DroneCommunicationEnvironment(config=NetworkConfig(num_rrbs=1), max_ttis=6)
        agent = SharedDDQN(env.action_space.powers, DDQNConfig(batch_size=32))
        run_episode(env, agent, training=True)
        transitions = list(agent.replay.items)
        self.assertTrue(any(0 < item.discount < agent.config.gamma for item in transitions))
        self.assertTrue(all(item.discount == 0 or any(item.next_mask) for item in transitions))
        self.assertTrue(any(item.discount == 0 for item in transitions))


if __name__ == '__main__':
    unittest.main()
