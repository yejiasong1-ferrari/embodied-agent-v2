"""Multiple-seed inverse kinematics; no task or object semantics live here."""

from __future__ import annotations

import numpy as np

from kinematics import solve_hand_ik
from joint_control import LEFT_GRASP


class MultiSolutionIK:
    def __init__(self, model, max_solutions: int = 4):
        self.model = model
        self.max_solutions = int(max_solutions)

    def solve(
        self,
        target_position_m: tuple[float, float, float],
        current_q: np.ndarray,
        *,
        yaw_rad: float = 0.0,
        target_rotation: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...]:
        current = np.asarray(current_q, dtype=float)
        offsets = (
            np.zeros(7),
            np.array((0.35, -0.20, 0.20, 0.15, -0.20, 0.15, 0.30)),
            np.array((-0.35, 0.20, -0.20, -0.15, 0.20, -0.15, -0.30)),
            np.array((0.15, 0.15, -0.35, 0.20, 0.30, -0.20, 0.40)),
            np.array((-0.15, -0.15, 0.35, -0.20, -0.30, 0.20, -0.40)),
        )
        solutions: list[np.ndarray] = []
        seed_candidates = (
            tuple(current + offset for offset in offsets)
            + tuple(LEFT_GRASP + offset for offset in offsets)
            + (None,)
        )
        for candidate_seed in seed_candidates:
            try:
                solution = solve_hand_ik(
                    self.model,
                    np.asarray(target_position_m, dtype=float),
                    # ``None`` invokes the model's canonical downward-palm
                    # seed. It escapes local minima that all perturbations of
                    # the current IK branch share; the exact validator still
                    # vetoes unsafe branch changes.
                    seed=candidate_seed,
                    yaw_rad=float(yaw_rad),
                    target_rotation=target_rotation,
                )
            except RuntimeError:
                continue
            if any(np.linalg.norm(solution - existing) < 0.08 for existing in solutions):
                continue
            solutions.append(solution)
            if len(solutions) >= self.max_solutions:
                break
        return tuple(solutions)
