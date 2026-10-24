from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GraspCandidate:
    id: str
    position: np.ndarray
    orientation: np.ndarray
    approach_vector: np.ndarray
    grasp_width: float
    score: float
    clearance: float
    grasp_type: str
    contact_region: str = "top"
    approach_direction: str = "top"
    contact_point: np.ndarray | None = None
    collision_free: bool | None = None
    reachability: bool | None = None
    stability_score: float = 0.
    geometry_score: float = 0.
    task_score: float = 0.
    source_position: np.ndarray | None = None
