import unittest

from environment.drone_env import DroneCommunicationEnvironment
from environment.scheduler import calculate_pf_metric


class TestThroughputHistory(unittest.TestCase):
    def test_completed_tti_records_served_and_waiting_drones_once(self):
        env = DroneCommunicationEnvironment(throughput_smoothing=0.25)
        env.num_rrbs = 1
        metrics = env.calculate_all_metrics()
        for _ in range(3):
            env.calculate_all_metrics()
            env.schedule_rrbs()
        self.assertEqual(env.average_throughput, dict.fromkeys(range(4), 1.0))
        env.advance_tti()
        for drone in range(4):
            self.assertAlmostEqual(env.average_throughput[drone], 0.75 + 0.25 * metrics[drone]['rate'])
        self.assertEqual(env.average_throughput[1], 0.75)
        self.assertEqual(env.average_throughput[2], 0.75)

    def test_waiting_drones_gain_priority_next_tti(self):
        env = DroneCommunicationEnvironment()
        env.num_rrbs = 1
        self.assertEqual(set(env.schedule_rrbs()), {0, 3})
        env.advance_tti()
        self.assertEqual(set(env.schedule_rrbs()), {1, 2})

    def test_slot_boundary_records_rate_before_moving(self):
        missions = {
            0: [(50, 0, 60), (150, 0, 60)],
            1: [(80, 20, 60), (100, 40, 60)],
            2: [(200, 0, 60), (240, 40, 60)],
            3: [(220, 20, 60), (260, 60, 60)],
        }
        env = DroneCommunicationEnvironment(missions, num_ttis_per_ts=1, throughput_smoothing=1.0)
        metrics = env.calculate_all_metrics()
        self.assertTrue(env.advance_tti())
        self.assertEqual(env.average_throughput, {drone: result['rate'] for drone, result in metrics.items()})
        self.assertEqual(env.drones[0], (150, 0, 60))

    def test_empty_schedule_decays_all_history(self):
        env = DroneCommunicationEnvironment(throughput_smoothing=0.2)
        env.num_rrbs = 0
        env.advance_tti()
        self.assertEqual(env.average_throughput, dict.fromkeys(range(4), 0.8))
        env.advance_tti()
        for value in env.average_throughput.values():
            self.assertAlmostEqual(value, 0.64)

    def test_advance_without_prior_metric_read_records_transmission(self):
        env = DroneCommunicationEnvironment()
        reference = DroneCommunicationEnvironment().calculate_all_metrics()
        env.advance_tti()
        for drone in range(4):
            self.assertAlmostEqual(env.average_throughput[drone], 0.9 + 0.1 * reference[drone]['rate'])

    def test_invalid_smoothing(self):
        for value in (0, -0.1, 1.1, True, None, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DroneCommunicationEnvironment(throughput_smoothing=value)

    def test_pf_rate_and_history_use_same_units(self):
        # SNR 3 gives 2 bits/s/Hz; 100 Hz gives 200 bits/s.
        self.assertEqual(calculate_pf_metric(1, 3, 1, 50, bandwidth=100), 4)


if __name__ == '__main__':
    unittest.main()
