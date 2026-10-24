from __future__ import annotations

import unittest

from embodied_v2.domain import AABB, JointTrajectory, Phase
from embodied_v2.state_machine import ManipulationStateMachine


class V2CoreTests(unittest.TestCase):
    def test_aabb_geometry(self) -> None:
        bounds = AABB((-1.0, 0.0, 1.0), (1.0, 2.0, 5.0))
        self.assertEqual(bounds.center, (0.0, 1.0, 3.0))
        self.assertEqual(bounds.size, (2.0, 2.0, 4.0))

    def test_state_machine_rejects_transport_shortcut(self) -> None:
        machine = ManipulationStateMachine()
        machine.transition(Phase.UNDERSTAND_TASK)
        with self.assertRaises(ValueError):
            machine.transition(Phase.TRANSPORT)

    def test_trajectory_requires_aligned_monotonic_samples(self) -> None:
        with self.assertRaises(ValueError):
            JointTrajectory(((0.0,) * 7,), (0.0, 1.0), "bad")
        with self.assertRaises(ValueError):
            JointTrajectory(((0.0,) * 7, (1.0,) * 7), (1.0, 1.0), "bad")


if __name__ == "__main__":
    unittest.main()
