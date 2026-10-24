"""Local candidate search joining route intent, multi-IK and final safety."""

from __future__ import annotations

import numpy as np

from ..domain import JointTrajectory, PayloadState, RouteIntent, SafetyVerdict, SceneModel, Vec3
from .corridor_planner import CorridorPlanner
from .escape_planner import EscapePlanner
from .ik_solver import MultiSolutionIK
from .ompl_planner import OMPLJointPlanner
from .trajectory_generator import generate_joint_trajectory


# Local seeds for a symbolic posture. The language model names the posture;
# it does not supply these joint offsets.
_POSTURE_SEED = {
    "keep": np.zeros(7),
    "elbow_up": np.array((0.0, -0.45, 0.0, -0.55, 0.0, 0.35, 0.0)),
    "elbow_left": np.array((0.55, -0.15, 0.35, -0.25, 0.20, 0.15, 0.25)),
    "elbow_right": np.array((-0.55, -0.15, -0.35, -0.25, -0.20, 0.15, -0.25)),
}


class LocalMotionPlanner:
    def __init__(self, runtime, validator):
        self.runtime = runtime
        self.validator = validator
        self.escape = EscapePlanner()
        self.corridors = CorridorPlanner()
        self.ik = MultiSolutionIK(runtime.model)
        self.ompl = OMPLJointPlanner(runtime, validator.checker)
        self.last_report = None
        self.last_rejections: tuple[str, ...] = ()
        self._last_ik_failure_point: tuple[float, float, float] | None = None

    def _cartesian_ik_path(
        self,
        start_position: np.ndarray,
        targets: tuple[Vec3, ...],
        seed: np.ndarray,
        *,
        spacing_m: float = 0.04,
        yaw_rad: float = 0.0,
        posture: str = "keep",
        target_rotation: np.ndarray | None = None,
    ) -> list[np.ndarray] | None:
        self._last_ik_failure_point = None
        path: list[np.ndarray] = [np.asarray(seed, dtype=float)]
        start = np.asarray(start_position, dtype=float)
        current_q = np.asarray(seed, dtype=float) + _POSTURE_SEED.get(posture, _POSTURE_SEED["keep"])
        for target_value in targets:
            target = np.asarray(target_value, dtype=float)
            distance = float(np.linalg.norm(target - start))
            count = max(1, int(np.ceil(distance / spacing_m)))
            for index in range(1, count + 1):
                point = start + (index / count) * (target - start)
                solutions = self.ik.solve(
                    tuple(point),
                    current_q,
                    yaw_rad=float(yaw_rad),
                    target_rotation=target_rotation,
                )
                if not solutions:
                    self._last_ik_failure_point = tuple(float(v) for v in point)
                    return None
                current_q = min(
                    solutions, key=lambda value: float(np.linalg.norm(value - current_q))
                )
                path.append(current_q)
            start = target
        return path

    def plan_to_cartesian(
        self,
        target_m: Vec3,
        scene: SceneModel,
        *,
        yaw_rad: float = 0.0,
        payload: PayloadState | None = None,
        phase: str = "approach",
        max_joint_velocity: float = 0.35,
        target_rotation: np.ndarray | None = None,
    ) -> JointTrajectory | None:
        current_q = self.runtime.arm_q()
        rejections: list[str] = []
        # Once an object is held, a collision-free endpoint is not enough:
        # direct interpolation between two joint solutions may swing the
        # wrist through a large roll/pitch excursion.  Walk short Cartesian
        # segments so every intermediate IK target keeps the grasp attitude.
        if payload is not None:
            hand_position, _ = self.runtime.hand_pose()
            cartesian_path = self._cartesian_ik_path(
                np.asarray(hand_position),
                (target_m,),
                current_q,
                spacing_m=0.008,
                yaw_rad=yaw_rad,
                target_rotation=target_rotation,
            )
            if cartesian_path is not None:
                trajectory = generate_joint_trajectory(
                    cartesian_path, max_joint_velocity=max_joint_velocity
                )
                report = self.validator.validate(
                    trajectory, scene, payload, phase
                )
                self.last_report = report
                if report.verdict is SafetyVerdict.PASS:
                    self.last_rejections = ()
                    return trajectory
                rejections.append(report.collision_pair or report.reason)

        solutions = self.ik.solve(
            target_m,
            current_q,
            yaw_rad=yaw_rad,
            target_rotation=target_rotation,
        )
        if not solutions:
            self.last_rejections = tuple(dict.fromkeys(rejections or ["ik_unreachable"]))
            self.last_report = None
            return None
        for solution in sorted(
            solutions, key=lambda value: float(np.linalg.norm(value - current_q))
        ):
            trajectory = generate_joint_trajectory(
                (current_q, solution), max_joint_velocity=max_joint_velocity
            )
            report = self.validator.validate(trajectory, scene, payload, phase)
            self.last_report = report
            if report.verdict is SafetyVerdict.PASS:
                self.last_rejections = ()
                return trajectory
            rejections.append(report.collision_pair or report.reason)
        self.last_rejections = tuple(dict.fromkeys(rejections))
        return None

    def plan_transport(
        self,
        payload: PayloadState,
        intent: RouteIntent,
        target_m: Vec3,
        scene: SceneModel,
        *,
        yaw_rad: float = 0.0,
        posture: str = "keep",
    ) -> JointTrajectory | None:
        current_q = self.runtime.arm_q()
        hand_position, _ = self.runtime.hand_pose()
        escapes = self.escape.candidates(tuple(hand_position), payload, scene)
        posture_name = posture or intent.posture or "keep"
        ompl_goals: list[np.ndarray] = []
        rejections: list[str] = []
        for escape in escapes:
            cartesian = self.corridors.waypoints(
                escape.position_m, target_m, intent, payload, scene
            )
            joint_path = self._cartesian_ik_path(
                np.asarray(hand_position),
                (escape.position_m,) + cartesian,
                current_q,
                spacing_m=0.018,
                yaw_rad=yaw_rad,
                posture=posture_name,
            )
            if joint_path is None:
                point = self._last_ik_failure_point
                suffix = "" if point is None else ":" + ",".join(
                    f"{value:.3f}" for value in point
                )
                rejections.append(f"{escape.name}:ik_unreachable{suffix}")
                continue
            trajectory = generate_joint_trajectory(
                joint_path,
                max_joint_velocity=0.16,
                minimum_segment_s=0.05,
            )
            report = self.validator.validate(trajectory, scene, payload, "transport")
            self.last_report = report
            if report.verdict is SafetyVerdict.PASS:
                self.last_rejections = ()
                return trajectory
            rejections.append(
                f"{escape.name}:{report.collision_pair or report.reason}"
            )
            ompl_goals.append(joint_path[-1])

        # Exhaust the orientation-preserving Cartesian escapes first. Random
        # joint-space motion is a last resort because a friction grasp cannot
        # tolerate arbitrary wrist attitudes even when the rigid payload clone
        # is collision-free.
        if ompl_goals:
            ompl_path = self.ompl.plan(
                current_q,
                tuple(ompl_goals),
                payload=payload,
                phase="transport",
            )
            if ompl_path is not None:
                trajectory = generate_joint_trajectory(
                    ompl_path, max_joint_velocity=0.16
                )
                report = self.validator.validate(
                    trajectory, scene, payload, "transport"
                )
                self.last_report = report
                if report.verdict is SafetyVerdict.PASS:
                    self.last_rejections = ()
                    return trajectory
                rejections.append(f"ompl:{report.collision_pair or report.reason}")
            else:
                rejections.append("ompl:no_exact_solution")
        self.last_rejections = tuple(dict.fromkeys(rejections))
        return None
