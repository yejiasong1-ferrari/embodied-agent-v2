"""OMPL-backed joint-space planning with an immutable validated plan result."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Callable, Sequence

import numpy as np
import ruckig
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou


# One process-wide seed must be installed before the first planner is built.
# Re-seeding after an earlier pick plan has started OMPL is explicitly ignored
# by OMPL and would make later transport candidates non-reproducible.
OMPL_RANDOM_SEED = 20261008
ou.RNG.setSeed(OMPL_RANDOM_SEED)


@dataclass(frozen=True)
class JointMotionPlan:
    plan_id: str
    planner: str
    waypoints: tuple[tuple[float, ...], ...]
    path_length_rad: float
    planning_time_s: float
    validated: bool = True

    def as_arrays(self) -> list[np.ndarray]:
        return [np.asarray(point, dtype=float) for point in self.waypoints]


def _path_length(points: Sequence[np.ndarray]) -> float:
    return float(sum(
        np.linalg.norm(np.asarray(b) - np.asarray(a))
        for a, b in zip(points, points[1:])
    ))


def _densely_valid(
    points: Sequence[np.ndarray],
    is_state_valid: Callable[[np.ndarray], bool],
    maximum_step: float,
) -> bool:
    """Recheck every joint segment independently of OMPL's resolution."""
    if not points:
        return False
    for point in points:
        if not is_state_valid(np.asarray(point, dtype=float)):
            return False
    for start, end in zip(points, points[1:]):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        samples = max(2, 1 + int(np.ceil(
            np.max(np.abs(end - start)) / max(maximum_step, 1e-4)
        )))
        for alpha in np.linspace(0.0, 1.0, samples):
            if not is_state_valid(start + float(alpha) * (end - start)):
                return False
    return True


def sample_executed_joint_trajectory(
    waypoints: Sequence[np.ndarray],
    *,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
    minimum_duration: float,
    sample_step_rad: float,
) -> list[np.ndarray] | None:
    """Sample the Ruckig curve that execution will command between waypoints.

    Planning validates straight joint chords. ``move_to_ruckig`` instead tracks
    a per-joint jerk-limited profile that leaves those chords, so the release
    check has to sample this curve. Each segment targets zero velocity, matching
    the carried-transport caller, which does not pass a via velocity.
    """
    points = [np.asarray(point, dtype=float) for point in waypoints]
    if len(points) < 2:
        return points
    samples = [points[0].copy()]
    step = max(float(sample_step_rad), 1e-4)
    for start, target in zip(points, points[1:]):
        inp = ruckig.InputParameter(7)
        trajectory = ruckig.Trajectory(7)
        inp.current_position = np.asarray(start, dtype=float).tolist()
        inp.current_velocity = [0.0] * 7
        inp.current_acceleration = [0.0] * 7
        inp.target_position = np.asarray(target, dtype=float).tolist()
        inp.target_velocity = [0.0] * 7
        inp.target_acceleration = [0.0] * 7
        inp.max_velocity = np.asarray(velocity_limits, dtype=float).tolist()
        inp.max_acceleration = np.asarray(acceleration_limits, dtype=float).tolist()
        inp.max_jerk = np.asarray(jerk_limits, dtype=float).tolist()
        inp.minimum_duration = max(float(minimum_duration), 0.0)
        result = ruckig.Ruckig(7).calculate(inp, trajectory)
        if result not in {ruckig.Result.Working, ruckig.Result.Finished}:
            return None
        span = float(np.max(np.abs(np.asarray(target) - np.asarray(start))))
        count = max(1, int(np.ceil(span / step)))
        for index in range(1, count + 1):
            position, _velocity, _acceleration = trajectory.at_time(
                trajectory.duration * index / count
            )
            samples.append(np.asarray(position, dtype=float))
    return samples


def plan_rrt_connect(
    start: np.ndarray,
    goals: Sequence[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    is_state_valid: Callable[[np.ndarray], bool],
    *,
    timeout_s: float = 1.2,
    interpolation_step_rad: float = 0.08,
    diagnostics: list[dict[str, object]] | None = None,
) -> JointMotionPlan | None:
    """Plan to the best reachable IK goal with OMPL RRT-Connect.

    The caller owns collision semantics.  Every OMPL state and every returned
    interpolated state is checked again before the plan is marked validated.
    """
    start = np.asarray(start, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if start.shape != (7,) or lower.shape != (7,) or upper.shape != (7,):
        raise ValueError("Panda joint vectors must have shape (7,)")

    begun = time.perf_counter()
    best: list[np.ndarray] | None = None
    # RRTConnect and shortcutting are randomized.  A shortcut that fails our
    # stricter dense collision pass is discarded and replanned; executing the
    # unsmoothed tree path made the position controller graze obstacles and
    # miss the Cartesian goal even though its sampled states were nominally
    # valid.
    planning_queries = [goal for goal in goals for _ in range(3)]
    per_goal = max(0.15, float(timeout_s) / max(1, len(planning_queries)))
    for query_index, goal_raw in enumerate(planning_queries):
        goal = np.asarray(goal_raw, dtype=float)
        query = {
            "query_index": query_index,
            "goal_valid": False,
            "ompl_exact": False,
            "chord_dense": False,
        }
        if diagnostics is not None:
            diagnostics.append(query)
        if goal.shape != (7,) or not is_state_valid(goal):
            continue
        query["goal_valid"] = True
        space = ob.RealVectorStateSpace(7)
        bounds = ob.RealVectorBounds(7)
        for index in range(7):
            bounds.setLow(index, float(lower[index]))
            bounds.setHigh(index, float(upper[index]))
        space.setBounds(bounds)
        setup = og.SimpleSetup(space)
        class _ValidityChecker(ob.StateValidityChecker):
            def __init__(self, space_information) -> None:
                super().__init__(space_information)

            def isValid(self, state) -> bool:  # noqa: N802 - OMPL API
                return bool(is_state_valid(np.array([
                    state[index] for index in range(7)
                ], dtype=float)))

        setup.setStateValidityChecker(_ValidityChecker(setup.getSpaceInformation()))
        setup.getSpaceInformation().setStateValidityCheckingResolution(0.004)
        start_state = space.allocState()
        goal_state = space.allocState()
        for index in range(7):
            start_state[index] = float(start[index])
            goal_state[index] = float(goal[index])
        setup.setStartAndGoalStates(start_state, goal_state, 1e-4)
        planner = og.RRTConnect(setup.getSpaceInformation())
        planner.setRange(0.28)
        setup.setPlanner(planner)
        solved = bool(setup.solve(per_goal))
        exact = bool(solved and setup.haveExactSolutionPath())
        query["ompl_exact"] = exact
        if not exact:
            continue
        setup.simplifySolution(min(0.20, per_goal * 0.25))
        path = setup.getSolutionPath()
        simplified = [
            np.array([state[index] for index in range(7)], dtype=float)
            for state in path.getStates()
        ]
        # Shortcut simplification can occasionally graze a collision boundary.
        # Reject that query and let a fresh RRT attempt find a better corridor.
        dense = _densely_valid(simplified, is_state_valid, interpolation_step_rad)
        query["chord_dense"] = dense
        if not dense:
            continue
        coarse = simplified
        if best is None or _path_length(coarse) < _path_length(best):
            best = coarse

    if best is None:
        return None
    digest = hashlib.sha256(np.asarray(best, dtype=np.float64).tobytes()).hexdigest()[:12]
    return JointMotionPlan(
        plan_id=f"ompl_{digest}",
        planner="OMPL RRTConnect",
        waypoints=tuple(tuple(float(value) for value in point) for point in best),
        path_length_rad=_path_length(best),
        planning_time_s=float(time.perf_counter() - begun),
    )
