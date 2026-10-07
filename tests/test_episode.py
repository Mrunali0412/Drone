import unittest

from environment.config import NetworkConfig
from environment.drone_env import DroneCommunicationEnvironment


class TestEpisode(unittest.TestCase):
    def test_joint_actions_reward_and_previous_observations(self):
        env = DroneCommunicationEnvironment(max_ttis=2)
        obs, initial = env.reset()
        for features in obs.values():
            self.assertEqual(features[1:], (0.0, 0.0, 0.1))
        actions = {drone: power for drone, power in enumerate((0.02, 0.05, 0.1, 0.3))}
        reference = DroneCommunicationEnvironment()
        reference.schedule_rrbs()
        reference.drone_powers.update(actions)
        expected = reference.calculate_all_metrics()
        obs, reward, terminated, truncated, info = env.step(actions)
        self.assertEqual(info['metrics'], expected)
        self.assertEqual(reward, sum(result['rate'] for result in expected.values()))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['rrb_assignment'], initial['rrb_assignment'])
        self.assertEqual(env.elapsed_ttis, 1)
        for drone in actions:
            self.assertEqual(obs[drone][1:], (expected[drone]['sinr'], expected[drone]['rate'], actions[drone]))
            self.assertAlmostEqual(env.average_throughput[drone], 0.9 + 0.1 * expected[drone]['rate'])

    def test_fixed_power_episode_reset_reproduces_trajectory(self):
        config = NetworkConfig(orus={10: (0, 0, 50)}, drones={7: (20, 0, 60)}, association={7: 10})
        env = DroneCommunicationEnvironment(
            {7: [(20, 0, 60), (80, 0, 60)]}, num_ttis_per_ts=2,
            config=config, max_ttis=4,
        )
        trajectories = []
        for _ in range(2):
            obs, info = env.reset()
            self.assertEqual(env.time_slot, 0)
            self.assertEqual(env.average_throughput, {7: 1.0})
            episode = []
            for index in range(4):
                transition = env.step({7: 0.1})
                self.assertEqual(transition[3], index == 3)
                episode.append(transition)
            with self.assertRaises(RuntimeError):
                env.step({7: 0.1})
            trajectories.append(episode)
        self.assertEqual(trajectories[0], trajectories[1])
        self.assertNotEqual(trajectories[0][0][4]['metrics'][7]['rate'], trajectories[0][2][4]['metrics'][7]['rate'])

    def test_idle_drones_and_next_schedule(self):
        env = DroneCommunicationEnvironment(config=NetworkConfig(num_rrbs=1))
        _, info = env.reset()
        self.assertEqual(set(info['rrb_assignment']), {0, 3})
        obs, _, _, _, info = env.step({0: 0.1, 3: 0.1})
        self.assertEqual(set(info['next_rrb_assignment']), {1, 2})
        for drone in (1, 2):
            self.assertEqual(obs[drone][1:], (0.0, 0.0, 0.0))
            self.assertEqual(info['rate_shortfalls'][drone], env.min_rate)

    def test_invalid_actions_do_not_mutate_episode(self):
        env = DroneCommunicationEnvironment()
        obs, _ = env.reset()
        for actions in ({}, {0: 0.1}, {0: 0.1, 1: 0.1, 2: 0.1, 3: float('nan')},
                        dict.fromkeys(range(4), 0.6), dict.fromkeys(range(4), True), []):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                env.step(actions)
            self.assertEqual(env.elapsed_ttis, 0)
            self.assertEqual(env.get_observations(), obs)
            self.assertEqual(env.drone_powers, dict.fromkeys(range(4), 0.1))
            self.assertEqual(env.average_throughput, dict.fromkeys(range(4), 1.0))

    def test_provider_failure_preserves_transition_state(self):
        class Provider:
            def get_channel_gains(self, drones, orus, slot):
                if slot:
                    raise ValueError('Missing snapshot')
                return {drone: dict.fromkeys(orus, 0.01) for drone in drones}
        env = DroneCommunicationEnvironment(num_ttis_per_ts=1, channel_provider=Provider())
        obs, _ = env.reset()
        with self.assertRaises(ValueError):
            env.step(dict.fromkeys(range(4), 0.2))
        self.assertEqual(env.get_observations(), obs)
        self.assertEqual(env.drone_powers, dict.fromkeys(range(4), 0.1))
        self.assertEqual(env.elapsed_ttis, 0)
        self.assertEqual(env.time_slot, 0)

    def test_empty_schedule_episode(self):
        env = DroneCommunicationEnvironment(config=NetworkConfig(num_rrbs=0), max_ttis=1)
        env.reset()
        _, reward, terminated, truncated, info = env.step({})
        self.assertEqual(reward, 0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(all(result['rate'] == 0 for result in info['metrics'].values()))

    def test_invalid_episode_limit(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DroneCommunicationEnvironment(max_ttis=value)


if __name__ == '__main__':
    unittest.main()
