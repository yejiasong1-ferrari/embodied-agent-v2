"""Geometry-to-semantic helpers for the V2 scene representation."""

from __future__ import annotations

import numpy as np

from ..domain import GapRegion, SceneEntity, SemanticGapMap


_PALETTE = {
    "red": (0.85, 0.08, 0.06),
    "orange": (0.95, 0.35, 0.08),
    "yellow": (0.92, 0.88, 0.25),
    "green": (0.10, 0.70, 0.20),
    "cyan": (0.05, 0.70, 0.75),
    "blue": (0.08, 0.25, 0.90),
    "purple": (0.52, 0.16, 0.72),
    "pink": (0.90, 0.20, 0.55),
    "white": (0.94, 0.94, 0.92),
    "gray": (0.50, 0.50, 0.50),
    "brown": (0.52, 0.32, 0.18),
    "black": (0.12, 0.12, 0.12),
}


def nearest_color(rgb: tuple[float, float, float]) -> str:
    sample = np.asarray(rgb, dtype=float)
    return min(
        _PALETTE,
        key=lambda name: float(np.linalg.norm(sample - np.asarray(_PALETTE[name]))),
    )


def semantic_category(geometry: str, size_m: tuple[float, float, float], movable: bool) -> str:
    width = max(float(size_m[0]), float(size_m[1]))
    height = float(size_m[2])
    if geometry == "cylinder" and not movable and width > 2.5 * max(height, 1e-6):
        return "plate"
    if geometry == "cylinder":
        return "cylinder"
    if geometry == "box":
        return "box"
    return geometry


def build_gap_map(obstacles: tuple[SceneEntity, ...]) -> SemanticGapMap:
    definitions = (
        ("left_outer", (-0.55, -0.18, 0.35)),
        ("center", (-0.18, 0.18, 0.36)),
        ("right_outer", (0.18, 0.55, 0.35)),
    )
    regions: list[GapRegion] = []
    scores: list[tuple[int, str]] = []
    for name, (lateral_low, lateral_high, z) in definitions:
        members = [
            obstacle for obstacle in obstacles
            if lateral_low <= obstacle.center_m[1] <= lateral_high
        ]
        highest = max((item.bounds.maximum[2] for item in members), default=0.0)
        clearance = "open" if not members else "narrow" if highest < 0.12 else "blocked_low"
        description = (
            f"{len(members)} obstacles; highest={highest:.3f} m"
        )
        regions.append(GapRegion(
            name=name,
            description=description,
            clearance=clearance,
            centroid_m=(0.0, (lateral_low + lateral_high) * 0.5, z),
        ))
        scores.append((len(members) + int(highest > 0.12) * 3, name))
    ordered = tuple(name for _, name in sorted(scores))
    blocked = tuple(region.name for region in regions if region.clearance == "blocked_low")
    return SemanticGapMap(tuple(regions), ordered, blocked)
