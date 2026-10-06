import math
from numbers import Real


class DroneMobility:
    """
    Implements the Hover-Transmit-Fly mobility model.

    During one Time Slot (TS), drones remain at their current
    waypoint for all TTIs.

    At the beginning of the next TS, drones move synchronously
    to their next waypoint.
    """

    def __init__(self, waypoints, num_ttis_per_ts):
        """
        Parameters
        ----------
        waypoints : dict
            Dictionary containing waypoint sequences for each drone.

            Example:
            {
                0: [(50, 0, 60), (70, 20, 60), (90, 40, 60)],
                1: [(80, 20, 60), (100, 40, 60), (120, 60, 60)]
            }

        num_ttis_per_ts : int
            Number of TTIs inside one Time Slot.
        """

        if isinstance(num_ttis_per_ts, bool) or not isinstance(num_ttis_per_ts, int) or num_ttis_per_ts <= 0:
            raise ValueError("num_ttis_per_ts must be a positive integer")
        if not waypoints or any(not points for points in waypoints.values()):
            raise ValueError("Each drone must have at least one waypoint")

        # Copy trajectories so later caller edits cannot change the mission.
        self.waypoints = {
            drone_id: tuple(tuple(point) for point in points)
            for drone_id, points in waypoints.items()
        }
        for points in self.waypoints.values():
            for point in points:
                if len(point) != 3:
                    raise ValueError("Waypoints must contain three coordinates")
                if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in point):
                    raise ValueError("Waypoint coordinates must be finite numbers")
        self.num_ttis_per_ts = num_ttis_per_ts
        self.time_slot = 0
        self.tti_in_time_slot = 0

        # Every drone initially starts at waypoint 0
        self.current_waypoint = {
            drone_id: 0
            for drone_id in waypoints
        }

        # Current position of every drone
        self.positions = {
            drone_id: drone_waypoints[0]
            for drone_id, drone_waypoints in self.waypoints.items()
        }

    def get_position(self, drone_id):
        """Return the current position of a drone."""
        return self.positions[drone_id]

    def get_all_positions(self):
        """Return the current positions of all drones."""
        return self.positions.copy()

    def advance_time_slot(self):
        """
        Move every drone to its next waypoint.

        All drones move synchronously at the beginning
        of the next Time Slot.
        """

        self.time_slot += 1
        self.tti_in_time_slot = 0
        for drone_id in self.waypoints:

            current_index = self.current_waypoint[drone_id]

            # Move only if another waypoint exists
            if current_index + 1 < len(self.waypoints[drone_id]):

                next_index = current_index + 1

                self.current_waypoint[drone_id] = next_index

                self.positions[drone_id] = (
                    self.waypoints[drone_id][next_index]
                )

    def advance_tti(self):
        """Finish one TTI; return True when the next time slot begins.

        Read communication metrics before calling this method. Completing
        the last TTI of a slot moves drones ready for the next transmission.
        """
        self.tti_in_time_slot += 1
        if self.tti_in_time_slot == self.num_ttis_per_ts:
            self.advance_time_slot()
            return True
        return False

    def get_waypoint_index(self, drone_id):
        """Return the current waypoint index of a drone."""
        return self.current_waypoint[drone_id]
