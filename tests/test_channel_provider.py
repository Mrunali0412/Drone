import unittest

from environment.config import NetworkConfig
from environment.drone_env import DroneCommunicationEnvironment


class RecordingProvider:
    def __init__(self):
        self.calls = []

    def get_channel_gains(self, drones, orus, time_slot):
        self.calls.append((time_slot, drones.copy()))
        return {drone: {oru: (time_slot + 1) * 0.01 for oru in orus} for drone in drones}


class TestChannelProvider(unittest.TestCase):
    def test_custom_network_and_provider_drive_all_links(self):
        config = NetworkConfig(
            orus={10: (0, 0, 50), 20: (100, 0, 50), 30: (200, 0, 50)},
            drones={7: (20, 0, 60), 8: (120, 0, 60), 9: (220, 0, 60)},
            association={7: 10, 8: 20, 9: 30}, num_rrbs=1,
        )
        env = DroneCommunicationEnvironment(config=config, channel_provider=RecordingProvider())
        result = env.calculate_drone_metrics(7)
        self.assertEqual(result['channel_gain'], 0.01)
        self.assertAlmostEqual(result['signal_power'], 0.001)
        self.assertAlmostEqual(result['interference'], 0.002)
        self.assertEqual(result['interference_to_neighbors'], {20: 0.001, 30: 0.001})
        self.assertEqual(set(env.schedule_rrbs()), {7, 8, 9})
        config.drones.clear()
        self.assertEqual(set(env.drones), {7, 8, 9})

    def test_snapshot_refreshes_only_at_slot_boundary(self):
        provider = RecordingProvider()
        env = DroneCommunicationEnvironment(channel_provider=provider)
        for _ in range(2):
            env.calculate_all_metrics()
            env.schedule_rrbs()
            env.advance_tti()
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(env.channel_gains[0][0], 0.01)
        env.advance_tti()
        self.assertEqual([slot for slot, _ in provider.calls], [0, 1])
        self.assertEqual(env.channel_gains[0][0], 0.02)

    def test_provider_receives_next_waypoint(self):
        config = NetworkConfig(orus={5: (0, 0, 0)}, drones={9: (1, 0, 0)}, association={9: 5})
        provider = RecordingProvider()
        env = DroneCommunicationEnvironment(
            {9: [(1, 0, 0), (2, 0, 0)]}, num_ttis_per_ts=1,
            config=config, channel_provider=provider,
        )
        env.advance_tti()
        self.assertEqual(provider.calls[1], (1, {9: (2, 0, 0)}))

    def test_invalid_snapshot_does_not_advance_state(self):
        provider = RecordingProvider()
        env = DroneCommunicationEnvironment(num_ttis_per_ts=1, channel_provider=provider)
        provider.get_channel_gains = lambda *args: {}
        history = env.average_throughput.copy()
        with self.assertRaises(ValueError):
            env.advance_tti()
        self.assertEqual(env.time_slot, 0)
        self.assertEqual(env.average_throughput, history)

    def test_invalid_config(self):
        for kwargs in ({'bandwidth': 0}, {'noise_power': float('nan')},
                       {'blocklength': 0}, {'error_probability': 1},
                       {'initial_power': 1}, {'association': {0: 999}},
                       {'num_rrbs': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NetworkConfig(**kwargs)

    def test_invalid_link_values(self):
        provider = RecordingProvider()
        for value in (-1, float('nan'), float('inf')):
            provider.get_channel_gains = lambda drones, orus, slot: {
                drone: {oru: value for oru in orus} for drone in drones
            }
            with self.subTest(value=value), self.assertRaises(ValueError):
                DroneCommunicationEnvironment(channel_provider=provider)


if __name__ == '__main__':
    unittest.main()
