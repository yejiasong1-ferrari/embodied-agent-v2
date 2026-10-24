"""Wide-view RGB-D scan and generic MuJoCo geometry inventory."""

from __future__ import annotations

import time

from ..adapters.mujoco_runtime import MujocoRuntime
from ..domain import SceneEntity, SceneModel, vec3
from .scene_model import build_gap_map, nearest_color, semantic_category


class MujocoSceneScanner:
    def __init__(self, runtime: MujocoRuntime, occupancy_resolution_m: float = 0.01):
        self.runtime = runtime
        self.occupancy_resolution_m = float(occupancy_resolution_m)
        self.last_rgb = None
        self.last_depth = None

    def _entity(self, body_name: str, *, movable: bool) -> SceneEntity:
        bounds = self.runtime.body_bounds(body_name)
        geometry = self.runtime.primary_geometry(body_name)
        rgba = self.runtime.body_rgba(body_name)
        category = semantic_category(geometry, bounds.size, movable)
        support = not movable and category in {"plate", "box"} and bounds.maximum[2] <= 0.30
        position, _ = self.runtime.body_pose(body_name)
        body_id = int(self.runtime.model.body(body_name).id)
        quat_wxyz = self.runtime.data.xquat[body_id]
        quat = (
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
            float(quat_wxyz[0]),
        )
        return SceneEntity(
            entity_id=body_name,
            center_m=vec3(position),
            bounds=bounds,
            color=nearest_color(rgba[:3]),
            geometry=category,
            movable=movable,
            support_candidate=support,
            orientation_xyzw=quat,
            metadata={"raw_geometry": geometry, "rgba": rgba, "quaternion_order": "xyzw"},
        )

    def scan(self) -> SceneModel:
        self.last_rgb, self.last_depth = self.runtime.render_rgbd()
        movable = tuple(
            self._entity(name, movable=True)
            for name in self.runtime.movable_body_names()
        )
        static = tuple(
            self._entity(name, movable=False)
            for name in self.runtime.iter_static_body_names()
        )
        supports = tuple(item for item in static if item.support_candidate)
        # A support is still solid geometry during approach and transport.  The
        # phase-aware validator grants only the target payload/support contact
        # during placement; removing supports here would create a safety hole.
        obstacles = static
        return SceneModel(
            timestamp_s=time.time(),
            entities=movable + static,
            obstacles=obstacles,
            supports=supports,
            occupancy_resolution_m=self.occupancy_resolution_m,
            # Gap semantics and the exact checker see the same solids. Other
            # movable objects are obstacles until one of them is selected as
            # the payload by the manipulation pipeline.
            gap_map=build_gap_map(movable + static),
        )
