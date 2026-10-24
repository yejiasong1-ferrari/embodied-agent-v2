"""Generate geometry-driven alternatives for leaving the grasp area."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain import PayloadState, SceneModel, Vec3, vec3


@dataclass(frozen=True)
class EscapePose:
    name: str
    position_m: Vec3
    score: float


class EscapePlanner:
    def __init__(self, vertical_m: float = 0.10, lateral_m: float = 0.055):
        self.vertical_m = float(vertical_m)
        self.lateral_m = float(lateral_m)

    def candidates(
        self,
        hand_position_m: Vec3,
        payload: PayloadState,
        scene: SceneModel,
    ) -> tuple[EscapePose, ...]:
        p = np.asarray(hand_position_m, dtype=float)
        # "Back" is radially away from the robot base, not an object-name rule.
        radial = p[:2] / max(float(np.linalg.norm(p[:2])), 1e-6)
        left = np.array((-radial[1], radial[0]))
        # ``bottom_offset_m`` is a transform from hand to payload bottom, not
        # an amount that must be added again. The micro-lift/clearance stage
        # has already separated the payload from its support. Escape only
        # needs a local upward move plus the measured uncertainty envelope.
        rise = max(self.vertical_m, payload.envelope_margin_m + 0.060)
        raw = (
            ("vertical_up", p + (0.0, 0.0, rise)),
            ("back_up", p + (*(-radial * self.lateral_m), rise)),
            ("left_back_up", p + (*(left * self.lateral_m - radial * 0.5 * self.lateral_m), rise)),
            ("right_back_up", p + (*(-left * self.lateral_m - radial * 0.5 * self.lateral_m), rise)),
        )
        result: list[EscapePose] = []
        obstacles = tuple(
            item for item in scene.entities if item.entity_id != payload.entity_id
        )
        for name, target in raw:
            clearance = min(
                (
                    float(np.linalg.norm(target - np.asarray(item.center_m)))
                    - 0.5 * max(item.bounds.size)
                    for item in obstacles
                ),
                default=1.0,
            )
            result.append(EscapePose(name, vec3(target), clearance))
        return tuple(sorted(result, key=lambda item: item.score, reverse=True))
