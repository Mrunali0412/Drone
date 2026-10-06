import unittest

from environment.mobility import DroneMobility
from environment.drone_env import DroneCommunicationEnvironment


class TestDroneMobility(unittest.TestCase):

    def setUp(self):

        self.waypoints = {
            0: [
                (50, 0, 60),
                (70, 20, 60),
                (90, 40, 60)
            ],

            1: [
                (80, 20, 60),
                (100, 40, 60),
                (120, 60, 60)
            ],

            2: [
                (200, 0, 60),
                (220, 20, 60),
                (240, 40, 60)
            ],

            3: [
                (220, 20, 60),
                (240, 40, 60),
                (260, 60, 60)
            ]
        }

        self.mobility = DroneMobility(
            waypoints=self.waypoints,
            num_ttis_per_ts=3
        )

    def test_initial_positions(self):

        self.assertEqual(
            self.mobility.get_position(0),
            (50, 0, 60)
        )

        self.assertEqual(
            self.mobility.get_position(1),
            (80, 20, 60)
        )

    def test_drones_stay_during_time_slot(self):

        initial_positions = self.mobility.get_all_positions()

        for _ in range(3):
            current_positions = self.mobility.get_all_positions()

            self.assertEqual(
                current_positions,
                initial_positions
            )
            self.mobility.advance_tti()

        self.assertEqual(self.mobility.get_position(0), (70, 20, 60))
        self.assertEqual(self.mobility.time_slot, 1)
        self.assertEqual(self.mobility.tti_in_time_slot, 0)

    def test_tti_boundary_and_multiple_slots(self):
        for slot in range(2):
            positions = self.mobility.get_all_positions()
            for tti in range(2):
                self.assertFalse(self.mobility.advance_tti())
                self.assertEqual(self.mobility.tti_in_time_slot, tti + 1)
                self.assertEqual(self.mobility.get_all_positions(), positions)
            self.assertTrue(self.mobility.advance_tti())
            self.assertEqual(self.mobility.time_slot, slot + 1)
        self.assertEqual(self.mobility.get_position(0), (90, 40, 60))

    def test_final_waypoint_is_retained(self):
        for _ in range(12):
            self.mobility.advance_tti()
        self.assertEqual(self.mobility.time_slot, 4)
        self.assertEqual(self.mobility.get_waypoint_index(0), 2)
        self.assertEqual(self.mobility.get_position(0), (90, 40, 60))

    def test_invalid_inputs(self):
        for ttis in (0, -1, 1.5, True):
            with self.subTest(ttis=ttis), self.assertRaises(ValueError):
                DroneMobility(self.waypoints, ttis)
        for waypoints in ({}, {0: []}, {0: [(1, 2)]}, {0: [(1, 2, float('nan'))]}):
            with self.subTest(waypoints=waypoints), self.assertRaises(ValueError):
                DroneMobility(waypoints, 3)

    def test_environment_metrics_follow_mobility(self):
        env = DroneCommunicationEnvironment(self.waypoints, num_ttis_per_ts=3)
        initial_metrics = env.calculate_all_metrics()
        for _ in range(2):
            self.assertFalse(env.advance_tti())
            for drone, result in env.calculate_all_metrics().items():
                self.assertEqual(result['distance'], initial_metrics[drone]['distance'])
                self.assertEqual(result['channel_gain'], initial_metrics[drone]['channel_gain'])
        self.assertTrue(env.advance_tti())
        self.assertEqual(env.time_slot, 1)
        self.assertEqual(env.tti_in_time_slot, 0)
        self.assertEqual(env.drones, {drone: points[1] for drone, points in self.waypoints.items()})
        moved_metrics = env.calculate_all_metrics()
        self.assertNotEqual(moved_metrics[0]['distance'], initial_metrics[0]['distance'])
        self.assertNotEqual(moved_metrics[0]['channel_gain'], initial_metrics[0]['channel_gain'])
        self.assertNotEqual(moved_metrics[0]['rate'], initial_metrics[0]['rate'])

    def test_environment_stationary_default(self):
        env = DroneCommunicationEnvironment()
        original = env.drones.copy()
        for _ in range(6):
            env.advance_tti()
        self.assertEqual(env.drones, original)
        self.assertEqual(env.time_slot, 2)

    def test_environment_rejects_missing_drones(self):
        with self.assertRaises(ValueError):
            DroneCommunicationEnvironment({0: [(50, 0, 60)]})

    def test_move_to_next_waypoint(self):

        self.mobility.advance_time_slot()

        self.assertEqual(
            self.mobility.get_position(0),
            (70, 20, 60)
        )

        self.assertEqual(
            self.mobility.get_position(1),
            (100, 40, 60)
        )

    def test_all_drones_move_together(self):

        self.mobility.advance_time_slot()

        self.assertEqual(
            self.mobility.get_waypoint_index(0),
            1
        )

        self.assertEqual(
            self.mobility.get_waypoint_index(1),
            1
        )

        self.assertEqual(
            self.mobility.get_waypoint_index(2),
            1
        )

        self.assertEqual(
            self.mobility.get_waypoint_index(3),
            1
        )

    def test_multiple_time_slots(self):

        self.mobility.advance_time_slot()

        self.mobility.advance_time_slot()

        self.assertEqual(
            self.mobility.get_position(0),
            (90, 40, 60)
        )

        self.assertEqual(
            self.mobility.get_position(3),
            (260, 60, 60)
        )


if __name__ == "__main__":
    unittest.main()
