"""Local millimetre grasp geometry, including insertion depth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain import ObjectObservation, SceneModel


@dataclass(frozen=True)
class GraspGeometryConfig:
    usable_finger_length_m: float = 0.058
    palm_to_contact_m: float = 0.147
    minimum_root_clearance_m: float = 0.012
    minimum_bottom_clearance_m: float = 0.012
    maximum_insertion_fraction: float = 0.65


@dataclass(frozen=True)
class InsertionGeometry:
    insertion_depth_m: float
    finger_target_height_m: float
    palm_height_m: float


def compute_insertion_depth(
    observation: ObjectObservation,
    scene: SceneModel,
    config: GraspGeometryConfig = GraspGeometryConfig(),
) -> InsertionGeometry:
    """Compute depth from dimensions and local clearance, never object names."""
    height = max(float(observation.height_m), 1e-4)
    maximum = min(
        config.usable_finger_length_m - config.minimum_root_clearance_m,
        height * config.maximum_insertion_fraction,
    )
    desired = min(maximum, max(0.010, height * 0.42))
    local_obstacles = {
        entity.entity_id: entity
        for entity in scene.obstacles
        if entity.entity_id in observation.local_obstacle_ids
    }
    # A nearby obstacle above the intended contact height limits how deeply
    # the palm may descend. This is purely geometric and applies to any shape.
    obstacle_ceiling = max(
        (entity.bounds.maximum[2] for entity in local_obstacles.values()),
        default=observation.bottom_z_m,
    )
    if obstacle_ceiling > observation.top_z_m:
        desired = min(desired, height * 0.30)
    insertion = float(np.clip(desired, 0.006, max(0.006, maximum)))
    finger_z = max(
        observation.bottom_z_m + config.minimum_bottom_clearance_m,
        observation.top_z_m - insertion,
    )
    finger_z = min(finger_z, observation.top_z_m - 0.006)
    return InsertionGeometry(
        insertion_depth_m=insertion,
        finger_target_height_m=float(finger_z),
        palm_height_m=float(finger_z + config.palm_to_contact_m),
    )
