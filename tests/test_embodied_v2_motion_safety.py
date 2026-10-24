from __future__ import annotations

import unittest

import numpy as np

from embodied_v2.adapters import MujocoRuntime
from embodied_v2.domain import AABB, PayloadState, SafetyVerdict
from embodied_v2.motion import (
    EscapePlanner,
    MultiSolutionIK,
    OMPLJointPlanner,
    generate_joint_trajectory,
)
from embodied_v2.perception import MujocoSceneScanner
from embodied_v2.safety import (
    CollisionChecker,
    ExactTrajectoryValidator,
    RuntimeSafetyMonitor,
    RuntimeStabilitySample,
)


class V2MotionSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = MujocoRuntime(width=120, height=90)
        cls.scene = MujocoSceneScanner(cls.runtime).scan()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.runtime.close()

    def payload(self) -> PayloadState:
        entity = next(item for item in self.scene.entities if item.movable)
        return PayloadState(
            entity.entity_id,
            entity.bounds,
            (0.0, 0.0, 0.10),
            (0.0, 0.0, 0.0, 1.0),
            0.05,
            0.001,
            1.0,
            0.008,
        )

    def test_escape_generation_contains_four_geometry_options(self) -> None:
        hand, _ = self.runtime.hand_pose()
        candidates = EscapePlanner().candidates(tuple(hand), self.payload(), self.scene)
        self.assertEqual({item.name for item in candidates}, {
            "vertical_up", "back_up", "left_back_up", "right_back_up"
        })

    def test_multiple_seed_ik_returns_a_valid_solution(self) -> None:
        hand, _ = self.runtime.hand_pose()
        solutions = MultiSolutionIK(self.runtime.model).solve(tuple(hand), self.runtime.arm_q())
        self.assertGreaterEqual(len(solutions), 1)
        self.assertTrue(all(solution.shape == (7,) for solution in solutions))

    def test_validator_checks_generated_execution_samples(self) -> None:
        q = self.runtime.arm_q()
        trajectory = generate_joint_trajectory((q, q + np.array((0.01, 0, 0, 0, 0, 0, 0))))
        report = ExactTrajectoryValidator(CollisionChecker(self.runtime)).validate(
            trajectory, self.scene, None, "approach"
        )
        self.assertEqual(report.verdict, SafetyVerdict.PASS)
        self.assertEqual(report.samples_checked, len(trajectory.positions))
        self.assertEqual(trajectory.source, "v2_quintic_execution_samples")

    def test_runtime_monitor_distinguishes_force_from_instability(self) -> None:
        monitor = RuntimeSafetyMonitor(hard_force_n=10.0)
        stable_force = RuntimeStabilitySample(0.001, 2.0, True, 11.0, False)
        unstable_force = RuntimeStabilitySample(0.006, 5.0, True, 11.0, True)
        self.assertEqual(monitor.evaluate(stable_force), "hold_force_continue_slowly")
        self.assertEqual(monitor.evaluate(unstable_force), "stop_force_and_instability")

    def test_ompl_fallback_uses_shared_collision_contract(self) -> None:
        checker = CollisionChecker(self.runtime)
        start = self.runtime.arm_q()
        goal = start + np.array((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        path = OMPLJointPlanner(self.runtime, checker).plan(
            start, (goal,), payload=None, phase="approach", timeout_s=0.3
        )
        self.assertIsNotNone(path)
        self.assertTrue(all(
            checker.evaluate(tuple(q), None, phase="approach").safe for q in path
        ))


if __name__ == "__main__":
    unittest.main()
