"""Generate 2–4 generic grasp candidates from fresh local geometry."""

from __future__ import annotations

import math

import numpy as np

from ..domain import GraspCandidate, ObjectObservation, SceneModel
from .grasp_geometry import GraspGeometryConfig, compute_insertion_depth


class GeometryGraspPlanner:
    def __init__(self, config: GraspGeometryConfig = GraspGeometryConfig()):
        self.config = config

    def candidates(
        self, observation: ObjectObservation, scene: SceneModel
    ) -> tuple[GraspCandidate, ...]:
        insertion = compute_insertion_depth(observation, scene, self.config)
        raw_geometry = str(scene.entity(observation.entity_id).metadata.get(
            "raw_geometry", scene.entity(observation.entity_id).geometry
        ))
        if raw_geometry in {"cylinder", "sphere", "capsule"}:
            anchor = (
                observation.yaw_rad
                if observation.protrusion_direction_rad is None
                else observation.protrusion_direction_rad + math.pi / 2.0
            )
            axes = tuple(anchor + offset for offset in (
                0.0, math.pi / 2.0, math.pi / 4.0, -math.pi / 4.0
            ))
        else:
            axes = (observation.yaw_rad, observation.yaw_rad + math.pi / 2.0)
        center = np.asarray(observation.center_m, dtype=float)
        results: list[GraspCandidate] = []
        for index, yaw in enumerate(axes):
            closing = np.array((-math.sin(yaw), -math.cos(yaw)))
            obstacle_penalty = 0.0
            for obstacle_id in observation.local_obstacle_ids:
                obstacle = scene.entity(obstacle_id)
                delta = np.asarray(obstacle.center_m[:2]) - center[:2]
                distance = float(np.linalg.norm(delta))
                if distance < 1e-6:
                    obstacle_penalty += 1.0
                else:
                    obstacle_penalty += max(0.0, float(abs(delta @ closing) / distance) - 0.6)
            results.append(GraspCandidate(
                candidate_id=f"axis_{index}",
                center_m=(
                    float(center[0]), float(center[1]), insertion.palm_height_m
                ),
                yaw_rad=float((yaw + math.pi) % (2.0 * math.pi) - math.pi),
                approach_direction=(0.0, 0.0, -1.0),
                finger_target_height_m=insertion.finger_target_height_m,
                palm_height_m=insertion.palm_height_m,
                insertion_depth_m=insertion.insertion_depth_m,
                expected_width_m=float(observation.width_m),
                score=float(1.0 - 0.08 * index - 0.20 * obstacle_penalty),
                rationale=("close_range_geometry", "insertion_depth", "local_clearance"),
            ))
        return tuple(sorted(results, key=lambda item: item.score, reverse=True))
