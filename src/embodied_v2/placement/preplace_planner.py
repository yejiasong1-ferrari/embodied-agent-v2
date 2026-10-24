"""Generate multiple destination-relative pre-place poses."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..domain import ObjectObservation, PayloadState, Vec3


@dataclass(frozen=True)
class PrePlaceCandidate:
    name: str
    hand_position_m: Vec3
    yaw_offset_rad: float
    score: float


class PrePlacePlanner:
    def __init__(self, clearance_m: float = 0.055, offset_m: float = 0.025):
        self.clearance_m = float(clearance_m)
        self.offset_m = float(offset_m)

    def candidates(
        self,
        destination: ObjectObservation,
        payload: PayloadState,
        *,
        payload_world_offset_m: Vec3 = (0.0, 0.0, 0.0),
    ) -> tuple[PrePlaceCandidate, ...]:
        destination_xy = np.asarray(destination.center_m[:2], dtype=float)
        offset_xy = np.asarray(payload_world_offset_m[:2], dtype=float)
        contact_hand_z = destination.top_z_m + payload.bottom_offset_m
        z = contact_hand_z + self.clearance_m

        def hand_target(
            name: str,
            object_delta_xy: tuple[float, float],
            yaw_offset: float,
            height: float,
            score: float,
        ) -> PrePlaceCandidate:
            cosine, sine = math.cos(yaw_offset), math.sin(yaw_offset)
            rotation = np.array(((cosine, -sine), (sine, cosine)))
            rotated_offset = rotation @ offset_xy
            desired_object = destination_xy + np.asarray(object_delta_xy)
            hand_xy = desired_object - rotated_offset
            return PrePlaceCandidate(
                name,
                (float(hand_xy[0]), float(hand_xy[1]), float(height)),
                float(yaw_offset),
                float(score),
            )

        d = self.offset_m
        raw = (
            ("center", (0.0, 0.0), 0.0, z, 1.0),
            ("left_offset", (0.0, d), 0.0, z, 0.94),
            ("right_offset", (0.0, -d), 0.0, z, 0.94),
            ("front_offset", (d, 0.0), 0.0, z, 0.92),
            ("back_offset", (-d, 0.0), 0.0, z, 0.92),
            ("higher", (0.0, 0.0), 0.0, z + self.clearance_m, 0.88),
            ("higher_left", (0.0, d), 0.0, z + self.clearance_m, 0.84),
            ("higher_right", (0.0, -d), 0.0, z + self.clearance_m, 0.84),
            ("wrist_quarter_turn", (0.0, 0.0), math.pi / 2.0, z, 0.80),
        )
        return tuple(hand_target(*item) for item in raw)
