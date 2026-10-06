import unittest

from environment.channel import calculate_interference
from environment.drone_env import DroneCommunicationEnvironment


class TestRrbInterference(unittest.TestCase):
    def setUp(self):
        self.drones = {
            0: (1, 0, 0),
            1: (1, 0, 0),  # Same-cell transmission must be excluded.
            2: (2, 0, 0),  # Cross-link gain to O-RU 0 is 1/4.
            3: (4, 0, 0),  # Different RRB must be excluded.
            4: (1, 0, 0),  # Unscheduled transmission must be excluded.
        }
        self.powers = {drone: 0.4 for drone in self.drones}
        self.orus = {0: (0, 0, 0), 1: (10, 0, 0)}
        self.association = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
        self.assignment = {0: 0, 1: 1, 2: 0, 3: 1}

    def interference(self):
        return calculate_interference(
            target_oru=0, serving_drone=0, drones=self.drones,
            drone_powers=self.powers, orus=self.orus,
            association=self.association, rrb_assignment=self.assignment,
            path_loss_exponent=2,
        )

    def test_only_same_rrb_other_cell_contributes(self):
        # 0.4 / 2**2 = 0.1 W, using the link to the receiving O-RU.
        self.assertAlmostEqual(self.interference(), 0.1)
        for drone in (0, 1, 3, 4):
            self.powers[drone] = 100.0
        self.assertAlmostEqual(self.interference(), 0.1)

    def test_different_rrb_has_no_interference(self):
        self.assignment[2] = 2
        self.assertEqual(self.interference(), 0.0)

    def test_unscheduled_interferer_has_no_interference(self):
        del self.assignment[2]
        self.assertEqual(self.interference(), 0.0)

    def test_unscheduled_target_has_no_reception(self):
        del self.assignment[0]
        self.assertEqual(self.interference(), 0.0)


class TestEnvironmentScheduling(unittest.TestCase):
    def test_default_metrics_use_one_shared_assignment(self):
        env = DroneCommunicationEnvironment()
        metrics = env.calculate_all_metrics()
        self.assertEqual(env.rrb_assignment, {0: 0, 1: 1, 3: 0, 2: 1})
        # Drone 0 receives only drone 3's cochannel transmission.
        distance_squared = 220**2 + 20**2 + 10**2
        expected_interference = 0.1 / distance_squared**1.25
        self.assertAlmostEqual(metrics[0]['interference'], expected_interference, places=15)
        expected_signal = 0.1 / (50**2 + 10**2)**1.25
        self.assertAlmostEqual(metrics[0]['sinr'], expected_signal / (expected_interference + env.noise_power))
        for drone, result in metrics.items():
            self.assertTrue(result['scheduled'])
            self.assertEqual(result['rrb_id'], env.rrb_assignment[drone])

    def test_unscheduled_drones_are_silent(self):
        env = DroneCommunicationEnvironment()
        env.num_rrbs = 1
        metrics = env.calculate_all_metrics()
        self.assertEqual(set(env.rrb_assignment), {0, 3})
        for drone in (1, 2):
            result = metrics[drone]
            self.assertFalse(result['scheduled'])
            self.assertIsNone(result['rrb_id'])
            for name in ('signal_power', 'interference', 'sinr', 'rate'):
                self.assertEqual(result[name], 0.0)
            self.assertTrue(all(value == 0.0 for value in result['interference_to_neighbors'].values()))
        before = env.calculate_all_metrics()
        env.drone_powers[1] = 100.0
        env.drone_powers[2] = 100.0
        self.assertEqual(env.calculate_all_metrics(), before)

    def test_empty_schedule_is_preserved(self):
        env = DroneCommunicationEnvironment()
        env.num_rrbs = 0
        self.assertEqual(env.schedule_rrbs(), {})
        self.assertTrue(all(result['rate'] == 0.0 for result in env.calculate_all_metrics().values()))

    def test_next_tti_refreshes_assignment(self):
        env = DroneCommunicationEnvironment()
        env.num_rrbs = 1
        assignment = env.schedule_rrbs()
        assignment.clear()
        self.assertEqual(set(env.rrb_assignment), {0, 3})
        env.drone_powers[1] = 0.5
        self.assertFalse(env.calculate_drone_metrics(1)['scheduled'])
        env.advance_tti()
        self.assertIsNone(env.rrb_assignment)
        self.assertTrue(env.calculate_drone_metrics(1)['scheduled'])


if __name__ == '__main__':
    unittest.main()
