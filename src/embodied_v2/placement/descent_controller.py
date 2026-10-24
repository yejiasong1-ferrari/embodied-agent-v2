"""Millimetre-scale descent with validation at every generated segment."""

from __future__ import annotations

import numpy as np

from ..domain import JointTrajectory, PayloadState, SafetyVerdict, SceneModel
from ..motion.trajectory_generator import generate_joint_trajectory


class ControlledDescent:
    def __init__(self, runtime, ik_solver, validator, step_m: float = 0.004):
        self.runtime = runtime
        self.ik_solver = ik_solver
        self.validator = validator
        self.step_m = float(step_m)
        self.last_rejection: str | None = None

    def plan(
        self,
        start_hand_m: tuple[float, float, float],
        contact_hand_z_m: float,
        payload: PayloadState,
        scene: SceneModel,
        support_id: str,
        *,
        yaw_rad: float = 0.0,
    ) -> tuple[JointTrajectory, ...]:
        self.last_rejection = None
        current_q = self.runtime.arm_q()
        current = np.asarray(start_hand_m, dtype=float)
        _, descent_rotation = self.runtime.hand_pose()
        segments: list[JointTrajectory] = []
        while current[2] - contact_hand_z_m > 1e-5:
            next_z = max(contact_hand_z_m, current[2] - self.step_m)
            target = (float(current[0]), float(current[1]), float(next_z))
            solutions = self.ik_solver.solve(
                target,
                current_q,
                yaw_rad=yaw_rad,
                target_rotation=descent_rotation,
            )
            if not solutions:
                self.last_rejection = f"ik_unreachable:{target}"
                return ()
            selected: tuple[np.ndarray, JointTrajectory] | None = None
            rejections: list[str] = []
            for next_q in sorted(
                solutions, key=lambda q: float(np.linalg.norm(q - current_q))
            ):
                trajectory = generate_joint_trajectory(
                    (current_q, next_q), max_joint_velocity=0.12
                )
                report = self.validator.validate(
                    trajectory,
                    scene,
                    payload,
                    "descent",
                    allowed_support_id=support_id,
                )
                if report.verdict is SafetyVerdict.PASS:
                    selected = (next_q, trajectory)
                    break
                rejections.append(report.collision_pair or report.reason)
            if selected is None:
                self.last_rejection = ",".join(dict.fromkeys(rejections))
                return ()
            next_q, trajectory = selected
            segments.append(trajectory)
            current_q = next_q
            current[2] = next_z
        return tuple(segments)
