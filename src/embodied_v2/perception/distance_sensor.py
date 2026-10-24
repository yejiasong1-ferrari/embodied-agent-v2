"""DistanceSensor implementation and conservative RGB-D/ToF fusion."""

from __future__ import annotations

import numpy as np

from ..adapters.mujoco_runtime import MujocoRuntime
from ..domain import DistanceReading, Vec3


class MujocoRayDistanceSensor:
    def __init__(self, runtime: MujocoRuntime):
        self.runtime = runtime

    def measure(
        self, origin_m: Vec3, direction: Vec3, max_distance_m: float
    ) -> DistanceReading:
        distance, geom_id = self.runtime.raycast(
            origin_m,
            direction,
            max_distance_m=max_distance_m,
            body_exclude=int(self.runtime.model.body("hand").id),
        )
        entity_id = None
        if geom_id is not None:
            body_id = int(self.runtime.model.geom_bodyid[geom_id])
            entity_id = self.runtime.model.body(body_id).name
        return DistanceReading(
            distance_m=distance,
            origin_m=origin_m,
            direction=direction,
            source="mujoco_raycast_tof",
            valid=distance is not None,
            hit_entity_id=entity_id,
        )


def fuse_distance_measurements(
    rgbd_distance_m: float | None,
    tof: DistanceReading,
    *,
    conflict_threshold_m: float = 0.015,
) -> tuple[float | None, str]:
    """Return a fused distance or force re-observation on disagreement."""
    if rgbd_distance_m is None and not tof.valid:
        return None, "unavailable"
    if rgbd_distance_m is None:
        return tof.distance_m, "tof_only"
    if not tof.valid or tof.distance_m is None:
        return float(rgbd_distance_m), "rgbd_only"
    if abs(float(rgbd_distance_m) - float(tof.distance_m)) > float(conflict_threshold_m):
        return None, "sensor_conflict_reobserve"
    # Bias toward the shorter distance: it is the conservative final-approach
    # estimate and does not declare either sensor the sole truth.
    return min(float(rgbd_distance_m), float(tof.distance_m)), "fused"
