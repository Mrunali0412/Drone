import unittest

from environment.reward import RewardConfig, CooperativeReward
from environment.config import NetworkConfig
from environment.drone_env import DroneCommunicationEnvironment


class TestReward(unittest.TestCase):
    def test_hand_calculated_reward_and_update(self):
        model = CooperativeReward(RewardConfig(initial_multiplier=2, learning_rate=0.1))
        metrics = {0: {'rate': 150, 'scheduled': True}, 1: {'rate': 50, 'scheduled': True}}
        result = model.evaluate(metrics, 100)
        self.assertEqual(result['sum_rate'], 200)
        self.assertEqual(result['rate_penalty'], 100)
        self.assertEqual(result['reward'], 100)
        self.assertEqual(result['mean_rate_shortfall'], 25)
        self.assertEqual(result['next_penalty_multiplier'], 4.5)
        self.assertEqual(model.multiplier, 2)
        model.update(result)
        self.assertEqual(model.multiplier, 4.5)

    def test_satisfied_rates_keep_multiplier(self):
        model = CooperativeReward(RewardConfig(initial_multiplier=2))
        result = model.evaluate({0: {'rate': 100, 'scheduled': True}}, 100)
        self.assertEqual(result['reward'], 100)
        self.assertEqual(result['next_penalty_multiplier'], 2)

    def test_unscheduled_interpretation_is_explicit(self):
        metrics = {0: {'rate': 150, 'scheduled': True}, 1: {'rate': 0, 'scheduled': False}}
        all_drones = CooperativeReward(RewardConfig(initial_multiplier=2)).evaluate(metrics, 100)
        scheduled = CooperativeReward(RewardConfig(initial_multiplier=2, penalize_unscheduled=False)).evaluate(metrics, 100)
        self.assertEqual(all_drones['reward'], -50)
        self.assertEqual(scheduled['reward'], 150)
        self.assertEqual(scheduled['penalty_shortfalls'], {0: 0})
        self.assertEqual(CooperativeReward(RewardConfig(penalize_unscheduled=False)).evaluate(
            {1: metrics[1]}, 100)['mean_rate_shortfall'], 0)

    def test_sum_rate_baseline_and_frozen_evaluation(self):
        metrics = {0: {'rate': 50, 'scheduled': True}}
        baseline = CooperativeReward(RewardConfig(mode='sum_rate', initial_multiplier=2))
        result = baseline.evaluate(metrics, 100)
        self.assertEqual(result['reward'], 50)
        self.assertEqual(result['next_penalty_multiplier'], 2)
        frozen = CooperativeReward(RewardConfig(initial_multiplier=2, adaptive=False))
        self.assertEqual(frozen.evaluate(metrics, 100)['reward'], -50)
        self.assertEqual(frozen.evaluate(metrics, 100)['next_penalty_multiplier'], 2)

    def test_episode_updates_once_and_resets(self):
        env = DroneCommunicationEnvironment(config=NetworkConfig(num_rrbs=0),
            reward_config=RewardConfig(initial_multiplier=1, learning_rate=1e-5))
        env.reset()
        env.calculate_all_metrics()
        self.assertEqual(env.reward_model.multiplier, 1)
        _, reward, _, _, info = env.step({})
        self.assertEqual(reward, -400000)
        self.assertEqual(info['penalty_multiplier'], 1)
        self.assertEqual(info['next_penalty_multiplier'], 2)
        self.assertEqual(env.reward_model.multiplier, 2)
        self.assertEqual(env.step({})[1], -800000)
        env.reset()
        self.assertEqual(env.reward_model.multiplier, 1)

    def test_failed_transition_does_not_update_multiplier(self):
        class Provider:
            def get_channel_gains(self, drones, orus, slot):
                if slot:
                    raise ValueError('Missing snapshot')
                return {drone: dict.fromkeys(orus, 1e-8) for drone in drones}
        env = DroneCommunicationEnvironment(num_ttis_per_ts=1, channel_provider=Provider(),
                                           reward_config=RewardConfig(initial_multiplier=2))
        env.reset()
        with self.assertRaises(ValueError):
            env.step(dict.fromkeys(range(4), 0.1))
        self.assertEqual(env.reward_model.multiplier, 2)
        self.assertEqual(env.elapsed_ttis, 0)

    def test_invalid_config(self):
        for kwargs in ({'initial_multiplier': -1}, {'learning_rate': float('nan')},
                       {'mode': 'unknown'}, {'adaptive': 1}, {'penalize_unscheduled': None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RewardConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
