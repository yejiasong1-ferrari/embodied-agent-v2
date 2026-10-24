"""V2-owned OMPL fallback using the same robot+payload safety contract."""

from __future__ import annotations

import numpy as np
from ompl import base as ob
from ompl import geometric as og

from ..adapters.mujoco_runtime import ARM_JOINTS


class OMPLJointPlanner:
    def __init__(self, runtime, collision_checker):
        self.runtime = runtime
        self.checker = collision_checker
        self.lower = np.array([
            runtime.model.jnt_range[runtime.model.joint(name).id][0]
            for name in ARM_JOINTS
        ], dtype=float)
        self.upper = np.array([
            runtime.model.jnt_range[runtime.model.joint(name).id][1]
            for name in ARM_JOINTS
        ], dtype=float)

    def plan(
        self,
        start: np.ndarray,
        goals: tuple[np.ndarray, ...],
        *,
        payload,
        phase: str,
        timeout_s: float = 2.0,
    ) -> list[np.ndarray] | None:
        # Servo compliance can leave a measured joint a few 1e-4 rad beyond a
        # hard bound. OMPL rejects that start before collision checking and the
        # caller otherwise retries every corridor with the same invalid state.
        margin = 1e-4
        start = np.clip(
            np.asarray(start, dtype=float),
            self.lower + margin,
            self.upper - margin,
        )
        for goal in goals:
            goal = np.clip(
                np.asarray(goal, dtype=float),
                self.lower + margin,
                self.upper - margin,
            )
            if not self.checker.evaluate(tuple(goal), payload, phase=phase).safe:
                continue
            space = ob.RealVectorStateSpace(7)
            bounds = ob.RealVectorBounds(7)
            for index in range(7):
                bounds.setLow(index, float(self.lower[index]))
                bounds.setHigh(index, float(self.upper[index]))
            space.setBounds(bounds)
            setup = og.SimpleSetup(space)
            checker = self.checker

            class _ValidityChecker(ob.StateValidityChecker):
                def __init__(self, information) -> None:
                    super().__init__(information)

                def isValid(self, state) -> bool:  # noqa: N802 - OMPL API
                    q = tuple(float(state[index]) for index in range(7))
                    return checker.evaluate(q, payload, phase=phase).safe

            setup.setStateValidityChecker(_ValidityChecker(setup.getSpaceInformation()))
            setup.getSpaceInformation().setStateValidityCheckingResolution(0.004)
            start_state = space.allocState()
            goal_state = space.allocState()
            for index in range(7):
                start_state[index] = float(start[index])
                goal_state[index] = float(goal[index])
            setup.setStartAndGoalStates(start_state, goal_state, 1e-4)
            planner = og.RRTConnect(setup.getSpaceInformation())
            planner.setRange(0.24)
            setup.setPlanner(planner)
            if not setup.solve(float(timeout_s)):
                continue
            # RRTConnect's first exact path can contain large random detours.
            # Shortcut it under the same validity checker before converting it
            # into the trajectory that the physical gripper must endure.
            setup.simplifySolution()
            path = setup.getSolutionPath()
            path.interpolate(max(2, int(np.ceil(path.length() / 0.06)) + 1))
            result = [
                np.array([state[index] for index in range(7)], dtype=float)
                for state in path.getStates()
            ]
            # RRTConnect may return an approximate path when its time budget
            # expires.  Such a path can be collision-free yet stop near the
            # source, which must never count as a completed transport.
            if not result or float(np.linalg.norm(result[-1] - goal)) > 0.015:
                continue
            if all(
                self.checker.evaluate(tuple(q), payload, phase=phase).safe
                for q in result
            ):
                return result
        return None
