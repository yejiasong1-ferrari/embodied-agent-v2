"""Validate the dense trajectory that will actually be executed."""

from __future__ import annotations

from ..domain import JointTrajectory, PayloadState, SafetyReport, SafetyVerdict
from .collision_checker import CollisionChecker


class ExactTrajectoryValidator:
    def __init__(self, checker: CollisionChecker):
        self.checker = checker

    def validate(
        self,
        trajectory: JointTrajectory,
        scene,
        payload: PayloadState | None,
        phase: str,
        *,
        allowed_support_id: str | None = None,
    ) -> SafetyReport:
        robot_min = float("inf")
        payload_min = float("inf")
        for index, position in enumerate(trajectory.positions):
            sample = self.checker.evaluate(
                position,
                payload,
                phase=phase,
                allowed_support_id=allowed_support_id,
            )
            robot_min = min(robot_min, sample.robot_clearance_m)
            payload_min = min(payload_min, sample.payload_clearance_m)
            if not sample.safe:
                return SafetyReport(
                    SafetyVerdict.FAIL,
                    f"collision at actual sample {index}",
                    robot_min,
                    payload_min,
                    index + 1,
                    sample.collision_pair,
                )
        return SafetyReport(
            SafetyVerdict.PASS,
            "actual execution samples are collision free",
            robot_min,
            payload_min if payload is not None else float("inf"),
            len(trajectory.positions),
        )
