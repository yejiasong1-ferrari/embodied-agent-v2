from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    timestamp: float
    view_id: str
    robot_mask: np.ndarray | None = None

    def __post_init__(self):
        if self.rgb.shape[:2] != self.depth.shape or self.intrinsics.shape != (3, 3) or self.camera_to_world.shape != (4, 4):
            raise ValueError("invalid RGB-D calibration/shape")

    @property
    def extrinsics(self):return self.camera_to_world
    @property
    def camera_pose(self):return self.camera_to_world
    @property
    def camera_name(self):return self.view_id.split(':')[0]


WristCameraObservation = CameraFrame


class CameraInterface(Protocol):
    @property
    def viewpoint_count(self) -> int: ...
    def capture(self) -> CameraFrame: ...
    def move_viewpoint(self, index: int) -> bool: ...
    def set_manipulation_active(self, active: bool) -> None: ...
    def update_collision_world(self, world) -> None: ...
