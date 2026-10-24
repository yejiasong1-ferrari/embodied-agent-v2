"""Single source of truth for tabletop object semantics and grasp priors.

Scene geometry remains in :mod:`scene_layout`; this registry describes how
planning, RGB-D perception and the local execution layer should interpret an
object.  Metric execution values for the original seven scene objects are
copied exactly from the pre-registry tables.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re


HSVRange = tuple[tuple[int, int, int], tuple[int, int, int]]


@dataclass(frozen=True)
class ObjectRecord:
    object_id: str
    body_name: str | None
    display_name: str
    aliases: tuple[str, ...]
    primary_color: str
    color_aliases: tuple[str, ...]
    hsv_ranges: tuple[HSVRange, ...]
    secondary_colors: tuple[str, ...]
    shape_prior: str
    dimensions_m: tuple[float, float, float]
    grasp_height_ratio: float
    grasp_strategy: str
    approach_direction: str
    grip_width_key: str
    default_grip_width: str
    hand_offset_m: float
    fallback_width_m: float
    grasp_depth_m: float
    target_force_n: float
    max_force_n: float
    movable: bool
    support_surfaces: tuple[str, ...]
    place_xy_tolerance_m: float = 0.09
    place_lower_tolerance_m: float = 0.008
    place_upper_tolerance_m: float = 0.025

    @property
    def half_height_m(self) -> float:
        return 0.5 * float(self.dimensions_m[2])


_COMMON_SUPPORTS = ("plate", "table", "tray")


OBJECTS: tuple[ObjectRecord, ...] = (
    ObjectRecord(
        "orange_cube", "cube", "cube", ("cube", "block", "方块", "积木"),
        "orange", ("orange", "橙", "橙色"), (((5, 100, 80), (25, 255, 255)),), (),
        "box", (0.050, 0.050, 0.050), 0.52, "top_grasp", "vertical", "cube", "wide",
        0.082, 0.050, 0.015, 1.80, 2.20, True, _COMMON_SUPPORTS,
    ),
    ObjectRecord(
        "red_cup", "red_cup", "cup", ("cup", "mug", "杯", "杯子", "马克杯"),
        "red", ("red", "红", "红色"),
        (((0, 100, 60), (8, 255, 255)), ((170, 100, 60), (179, 255, 255))), (),
        "handled_cylinder", (0.070, 0.070, 0.100), 0.50, "side_grasp", "vertical", "cup", "wide",
        0.105, 0.070, 0.020, 0.90, 2.00, True, _COMMON_SUPPORTS,
    ),
    ObjectRecord(
        "blue_box", "blue_box", "box", ("box", "block", "盒", "盒子", "方块"),
        "blue", ("blue", "蓝", "蓝色"), (((105, 80, 50), (135, 255, 255)),), (),
        "box", (0.070, 0.060, 0.080), 0.52, "top_grasp", "vertical", "box", "wide",
        0.103, 0.060, 0.018, 2.00, 2.80, True, _COMMON_SUPPORTS,
    ),
    ObjectRecord(
        "green_block", "green_block", "box", ("box", "block", "盒", "盒子", "方块"),
        "green", ("green", "绿", "绿色"), (((35, 70, 45), (82, 255, 255)),), (),
        "box", (0.070, 0.060, 0.080), 0.52, "top_grasp", "vertical", "box", "wide",
        0.103, 0.060, 0.018, 2.00, 2.80, True, _COMMON_SUPPORTS,
    ),
    ObjectRecord(
        "cyan_mug", "cyan_mug", "cup", ("cup", "mug", "杯", "杯子", "马克杯"),
        "cyan", ("cyan", "青", "青色", "蓝绿", "蓝绿色"), (((85, 60, 60), (102, 255, 255)),), (),
        "handled_cylinder", (0.064, 0.064, 0.080), 0.50, "side_grasp", "vertical", "cup", "wide",
        0.105, 0.070, 0.020, 0.90, 2.00, True, _COMMON_SUPPORTS,
    ),
    ObjectRecord(
        "purple_bottle", "purple_bottle", "bottle", ("bottle", "瓶", "瓶子"),
        "purple", ("purple", "紫", "紫色"), (((135, 60, 50), (165, 255, 255)),), (),
        "cylinder", (0.052, 0.052, 0.120), 0.50, "side_grasp", "vertical", "bottle", "wide",
        0.110, 0.052, 0.022, 0.90, 2.00, True, _COMMON_SUPPORTS,
    ),
    ObjectRecord(
        "yellow_plate", "plate", "plate", ("plate", "yellow_plate", "盘", "盘子"),
        "yellow", ("yellow", "黄", "黄色"), (((20, 70, 80), (38, 255, 255)),), (),
        "support_cylinder", (0.240, 0.240, 0.036), 0.50, "edge_grasp", "vertical", "support", "wide",
        0.0, 0.0, 0.0, 0.0, 0.0, False, (),
    ),
    # Demonstrates data-only onboarding after the registry consumers exist.
    ObjectRecord(
        "white_mug", "white_mug", "cup", ("cup", "mug", "杯", "杯子", "马克杯"),
        "white", ("white", "白", "白色"), (((0, 0, 175), (179, 58, 255)),), (),
        "handled_cylinder", (0.064, 0.064, 0.080), 0.50, "side_grasp", "vertical", "cup", "medium",
        0.105, 0.064, 0.020, 0.90, 2.00, True, _COMMON_SUPPORTS,
    ),
)


GENERIC_CYLINDER = ObjectRecord(
    "generic_cylinder", None, "cylinder", ("cylinder", "object", "物体", "圆柱"),
    "unknown", (), (), (), "cylinder", (0.060, 0.060, 0.100), 0.50,
    "side_grasp", "vertical", "cup", "medium", 0.105, 0.060, 0.020,
    0.90, 2.00, True, _COMMON_SUPPORTS,
)


BY_COLOR = {record.primary_color: record for record in OBJECTS}
BY_ID = {record.object_id: record for record in OBJECTS}
BY_BODY = {
    record.body_name: record for record in OBJECTS if record.body_name is not None
}


def record_for_color(color: str | None) -> ObjectRecord | None:
    return BY_COLOR.get(str(color or "").lower())


def record_for_body(body_name: str | None) -> ObjectRecord | None:
    return BY_BODY.get(str(body_name or ""))


def generic_record(color: str, name: str) -> ObjectRecord:
    """Conservative low-confidence template for a table-external target."""
    safe_color = re.sub(r"[^a-z0-9]+", "_", str(color).lower()).strip("_") or "unknown"
    safe_name = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") or "object"
    return replace(
        GENERIC_CYLINDER,
        object_id=f"generic_{safe_color}_{safe_name}",
        display_name=str(name or "cylinder"),
        aliases=(str(name or "cylinder"),),
        primary_color=str(color or "unknown").lower(),
    )


# Shape words are a name-gate only. Color stays the primary key, and a box
# word never matches a cylinder.
SHAPE_WORDS = {
    "圆柱体": "cylinder",
    "圆柱": "cylinder",
    "柱体": "cylinder",
    "cylinder": "cylinder",
}
_CYLINDER_SHAPES = frozenset({"cylinder", "handled_cylinder"})


def shape_query(name: str) -> str | None:
    return SHAPE_WORDS.get(str(name).strip().lower())


def shape_query_in_text(fragment: str) -> str | None:
    text = fragment.lower()
    for word in sorted(SHAPE_WORDS, key=len, reverse=True):
        if word in text:
            return SHAPE_WORDS[word]
    return None


def shapes_compatible(name: str, record: ObjectRecord) -> bool:
    """True when the user's noun agrees with the record's geometry."""
    if name_matches(record, name):
        return True
    kind = shape_query(name)
    if kind == "cylinder" and record.shape_prior in _CYLINDER_SHAPES:
        return True
    if kind == "box" and record.shape_prior == "box":
        return True
    return False


def name_matches(record: ObjectRecord, name: str) -> bool:
    query = str(name).lower()
    aliases = {alias.lower() for alias in record.aliases} | {record.display_name.lower()}
    if query in aliases:
        return True
    if (
        query in {"cube", "box", "block"} and record.display_name in {"cube", "box"}
    ) or (
        query in {"cup", "mug"} and record.display_name == "cup"
    ):
        return True
    kind = shape_query(query)
    return kind == "cylinder" and record.shape_prior in _CYLINDER_SHAPES


def color_tokens() -> dict[str, str]:
    return {
        alias.lower(): record.primary_color
        for record in OBJECTS
        for alias in record.color_aliases
    }


def object_name_from_text(fragment: str, default: str | None = None) -> str | None:
    text = fragment.lower()
    candidates: list[tuple[int, int, str]] = []
    for record in OBJECTS:
        for alias in record.aliases:
            position = text.find(alias.lower())
            if position >= 0:
                candidates.append((position, -len(alias), record.display_name))
    if not candidates:
        return default
    return min(candidates)[2]


def registry_catalog() -> tuple[dict[str, object], ...]:
    """Colors, categories, shapes and aliases for the language planner."""
    return tuple({
        "object_id": record.object_id,
        "color": record.primary_color,
        "category": record.display_name,
        "shape": record.shape_prior,
        "aliases": list(record.aliases[:4]),
    } for record in OBJECTS)


def registry_summary() -> tuple[dict[str, object], ...]:
    """Compact manager-visible description; no low-level trajectories."""
    return tuple({
        "id": record.object_id,
        "name": record.display_name,
        "aliases": list(record.aliases),
        "primary_color": record.primary_color,
        "secondary_colors": list(record.secondary_colors),
        "shape_prior": record.shape_prior,
        "dimensions_m": list(record.dimensions_m),
        "grasp_height_ratio": record.grasp_height_ratio,
        "grasp_template": {
            "strategy": record.grasp_strategy,
            "approach": record.approach_direction,
            "grip_width": record.default_grip_width,
            "force_limit_key": record.grip_width_key,
        },
        "placement": {
            "support_surfaces": list(record.support_surfaces),
            "xy_tolerance_m": record.place_xy_tolerance_m,
            "height_tolerance_m": [
                -record.place_lower_tolerance_m, record.place_upper_tolerance_m,
            ],
        },
    } for record in OBJECTS)
