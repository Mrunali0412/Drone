import random
import unittest

from environment.actions import PowerActionSpace, NoFeasibleActionError
from environment.config import NetworkConfig
from environment.drone_env import DroneCommunicationEnvironment


class TestPowerActions(unittest.TestCase):
    def test_grid_and_exact_threshold(self):
        space = PowerActionSpace(0.1, 0.5, 5)
        self.assertEqual(space.powers[0], 0.1)
        self.assertEqual(space.powers[-1], 0.5)
        self.assertEqual(space.mask({0: 100, 1: 2, 2: 1}, 0, 0.4), (True, True, False, False, False))
        self.assertEqual(space.mask({0: 100}, 0, 0), (True,) * 5)

    def test_greedy_and_exploration_exclude_masked_actions(self):
        space = PowerActionSpace(0.1, 0.5, 5)
        mask = (True, False, True, False, False)
        self.assertEqual(space.select([1, 100, 2, 200, 300], mask), 2)
        rng = random.Random(42)
        chosen = {space.select([1, 100, 2, 200, 300], mask, epsilon=1, rng=rng) for _ in range(100)}
        self.assertEqual(chosen, {0, 2})
        with self.assertRaises(NoFeasibleActionError):
            space.select([0] * 5, (False,) * 5)

    def test_invalid_indices(self):
        space = PowerActionSpace(0.1, 0.5, 5)
        for index in (-1, 5, True, 1.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                space.power(index)

    def test_discrete_episode_satisfies_c3_and_reset(self):
        env = DroneCommunicationEnvironment(max_ttis=3)
        _, info = env.reset()
        initial_masks = info['action_masks']
        for _ in range(3):
            masks = env.get_action_masks()
            actions = {drone: max(index for index, allowed in enumerate(mask) if allowed)
                       for drone, mask in masks.items()}
            _, _, _, _, result = env.step_discrete(actions)
            self.assertTrue(all(check['interference_constraint'] for check in result['constraints'].values()))
            self.assertEqual(result['action_masks'], masks)
            self.assertEqual(result['next_action_masks'], env.get_action_masks())
            self.assertEqual(result['action_indices'], actions)
        self.assertEqual(env.reset()[1]['action_masks'], initial_masks)

    def test_masked_action_rejected_without_state_change(self):
        env = DroneCommunicationEnvironment()
        obs, info = env.reset()
        actions = dict.fromkeys(info['rrb_assignment'], 0)
        actions[1] = len(env.action_space.powers) - 1
        with self.assertRaises(ValueError):
            env.step_discrete(actions)
        self.assertEqual(env.elapsed_ttis, 0)
        self.assertEqual(env.get_observations(), obs)
        self.assertEqual(env.drone_powers, dict.fromkeys(range(4), 0.1))

    def test_no_feasible_power_is_explicit(self):
        env = DroneCommunicationEnvironment(config=NetworkConfig(max_interference=0))
        _, info = env.reset()
        self.assertTrue(all(not any(mask) for mask in info['action_masks'].values()))
        with self.assertRaises(NoFeasibleActionError):
            env.step_discrete(dict.fromkeys(info['rrb_assignment'], 0))
        self.assertEqual(env.elapsed_ttis, 0)

    def test_masks_use_new_slot_cross_links_and_scheduled_only(self):
        class Provider:
            def get_channel_gains(self, drones, orus, slot):
                return {drone: {oru: 0.01 if oru == drone else (1e-8 if slot == 0 else 1)
                                for oru in orus} for drone in drones}
        config = NetworkConfig(orus={0: (0, 0, 0), 1: (10, 0, 0)},
                               drones={0: (1, 0, 0), 1: (9, 0, 0)}, association={0: 0, 1: 1})
        env = DroneCommunicationEnvironment(config=config, num_ttis_per_ts=1, channel_provider=Provider())
        self.assertTrue(all(all(mask) for mask in env.get_action_masks().values()))
        result = env.step_discrete({0: 0, 1: 0})
        self.assertTrue(all(not any(mask) for mask in result[4]['next_action_masks'].values()))
        env.num_rrbs = 0
        env.schedule_rrbs()
        self.assertEqual(env.get_action_masks(), {})


if __name__ == '__main__':
    unittest.main()
