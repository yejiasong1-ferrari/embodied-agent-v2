"""Generate the exact dense joint samples that the executor will consume."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..domain import JointTrajectory


def _smootherstep(value: float) -> float:
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def generate_joint_trajectory(
    waypoints: Sequence[np.ndarray],
    *,
    sample_period_s: float = 0.02,
    max_joint_velocity: float = 0.28,
    minimum_segment_s: float = 0.20,
) -> JointTrajectory:
    if len(waypoints) < 2:
        raise ValueError("at least two joint waypoints are required")
    positions: list[tuple[float, ...]] = []
    times: list[float] = []
    elapsed = 0.0
    for segment_index, (start_value, end_value) in enumerate(zip(waypoints, waypoints[1:])):
        start = np.asarray(start_value, dtype=float)
        end = np.asarray(end_value, dtype=float)
        if start.shape != (7,) or end.shape != (7,):
            raise ValueError("joint waypoints must have shape (7,)")
        span = float(np.max(np.abs(end - start)))
        # Quintic peak speed is 1.875 * span / duration.
        duration = max(
            float(minimum_segment_s),
            1.875 * span / max(float(max_joint_velocity), 1e-6),
        )
        count = max(2, int(math.ceil(duration / sample_period_s)) + 1)
        for sample in range(count):
            if segment_index and sample == 0:
                continue
            fraction = sample / (count - 1)
            q = start + _smootherstep(fraction) * (end - start)
            positions.append(tuple(float(value) for value in q))
            times.append(elapsed + fraction * duration)
        elapsed += duration
    return JointTrajectory(tuple(positions), tuple(times), "v2_quintic_execution_samples")
