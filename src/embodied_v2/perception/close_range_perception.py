"""Fresh local geometry measurement used for manipulation, never wide-view reuse."""

from __future__ import annotations

import math

import numpy as np

from ..adapters.mujoco_runtime import MujocoRuntime
from ..domain import ObjectObservation, SceneModel, vec3


class MujocoCloseRangePerception:
    def __init__(self, runtime: MujocoRuntime, neighborhood_m: float = 0.14):
        self.runtime = runtime
        self.neighborhood_m = float(neighborhood_m)

    def observe(self, entity_id: str, scene: SceneModel) -> ObjectObservation:
        # Capture a fresh wrist frame even though the simulator adapter obtains
        # exact local geometry from MuJoCo. Hardware adapters replace only this
        # class and preserve the same output contract.
        self.runtime.render_rgbd()
        bounds = self.runtime.body_bounds(entity_id)
        center, rotation = self.runtime.body_pose(entity_id)
        yaw = float(math.atan2(rotation[1, 0], rotation[0, 0]))
        size = bounds.size
        nearby = tuple(
            item.entity_id
            for item in scene.obstacles
            if np.linalg.norm(
                np.asarray(item.center_m[:2]) - np.asarray(bounds.center[:2])
            ) <= self.neighborhood_m
        )
        protrusion = None
        geoms = self.runtime.body_geom_ids(entity_id)
        if len(geoms) > 1:
            body_center = np.asarray(bounds.center)
            offsets = [
                self.runtime.data.geom_xpos[geom_id] - body_center
                for geom_id in geoms
            ]
            farthest = max(offsets, key=lambda value: float(np.linalg.norm(value[:2])))
            if float(np.linalg.norm(farthest[:2])) > 0.015:
                protrusion = float(math.atan2(farthest[1], farthest[0]))
        return ObjectObservation(
            entity_id=entity_id,
            center_m=vec3(center),
            bounds=bounds,
            yaw_rad=yaw,
            width_m=float(min(size[0], size[1])),
            height_m=float(size[2]),
            top_z_m=float(bounds.maximum[2]),
            bottom_z_m=float(bounds.minimum[2]),
            protrusion_direction_rad=protrusion,
            local_obstacle_ids=nearby,
            source="close_range_rgbd_geometry",
            confidence=1.0,
        )
