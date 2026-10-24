"""Open scene inventory. The signature does not depend on the detector.

Cloud VLM is the first backend. Offline, the same function degrades to the
registered-colour detector: known objects can still be found, and a colour
that is not in the registry is simply absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from object_registry import (
    OBJECTS,
    generic_record,
    record_for_color,
    shapes_compatible,
)
from scene_layout import FIXED_POSES, OBJECT_BEARING_DEG
from vision_detector import _backproject_with_saved_pose


# A perceived point may claim a free body only inside this neighbourhood.
# Farther than this, the entry stays unbound and the task reports not-found
# instead of grabbing whichever body is nearest.
BODY_BIND_RADIUS_M = 0.08


@dataclass
class PerceivedObject:
    description: str
    color: str
    shape: str
    view_index: int
    u: float
    v: float
    confidence: str = "low"
    object_id: str | None = None
    body_name: str | None = None
    dimensions_m: tuple[float, float, float] = (0.06, 0.06, 0.10)
    xy_m: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dimensions_m"] = [float(v) for v in self.dimensions_m]
        if self.xy_m is not None:
            payload["xy_m"] = [float(v) for v in self.xy_m]
        return payload


def _canon_color(color: str) -> str:
    text = str(color or "").strip().lower().replace(" ", "")
    return {
        "magenta": "pink", "fuchsia": "pink", "hotpink": "pink",
        "粉色": "pink", "粉红": "pink", "粉": "pink",
    }.get(text, str(color or "").strip().lower())


def _shape_token(shape: str) -> str:
    text = str(shape or "").strip().lower()
    if text in {"cylinder", "bottle", "cup", "mug", "圆柱", "瓶子", "杯子"}:
        return "cylinder"
    if text in {"box", "cube", "block", "方块", "盒子"}:
        return "box"
    return text


def _shape_agrees(shape: str, record) -> bool:
    token = _shape_token(shape)
    if shapes_compatible(token, record) or shapes_compatible(shape, record):
        return True
    if token == "cylinder" and record.shape_prior in {"cylinder", "handled_cylinder"}:
        return True
    if token == "box" and record.shape_prior == "box":
        return True
    return False


def nearest_free_body(xy: np.ndarray, radius_m: float = BODY_BIND_RADIUS_M) -> str | None:
    """Bind a grounded point to a scene body, or to nothing past the radius."""
    point = np.asarray(xy, dtype=float)[:2]
    best_name = None
    best_distance = float(radius_m)
    for name, (x, y, _z, _yaw) in FIXED_POSES.items():
        distance = float(np.hypot(point[0] - x, point[1] - y))
        if distance <= best_distance:
            best_name = name
            best_distance = distance
    return best_name


def _ground_xy(view: dict, u: float, v: float) -> np.ndarray | None:
    depth = np.asarray(view.get("depth"))
    if depth.ndim != 2:
        return None
    height, width = depth.shape
    col = int(np.clip(float(u), 0.0, 1.0) * (width - 1))
    row = int(np.clip(float(v), 0.0, 1.0) * (height - 1))
    z = float(depth[row, col])
    if not np.isfinite(z) or z <= 0.02 or z >= 2.0:
        return None
    pose = (
        view["cam_xpos"], view["cam_xmat"], view["fovy_deg"],
        view["base_xpos"], view["base_xmat"],
    )
    points = _backproject_with_saved_pose(
        np.array([row]), np.array([col]), depth, pose,
    )
    return np.asarray(points[0, :2], dtype=float)


def _measure_patch(view: dict, u: float, v: float) -> tuple[float, float, float]:
    """RGB-D neighbourhood size. Falls back to the conservative cylinder."""
    depth = np.asarray(view.get("depth"))
    if depth.ndim != 2:
        return (0.06, 0.06, 0.10)
    height, width = depth.shape
    col = int(np.clip(float(u), 0.0, 1.0) * (width - 1))
    row = int(np.clip(float(v), 0.0, 1.0) * (height - 1))
    z0 = float(depth[row, col])
    if not np.isfinite(z0) or z0 <= 0.02:
        return (0.06, 0.06, 0.10)
    r0, r1 = max(0, row - 18), min(height, row + 19)
    c0, c1 = max(0, col - 18), min(width, col + 19)
    window = depth[r0:r1, c0:c1]
    rows, cols = np.nonzero(np.isfinite(window) & (np.abs(window - z0) < 0.03) & (window > 0.02))
    if rows.size < 8:
        return (0.06, 0.06, 0.10)
    pose = (
        view["cam_xpos"], view["cam_xmat"], view["fovy_deg"],
        view["base_xpos"], view["base_xmat"],
    )
    points = _backproject_with_saved_pose(rows + r0, cols + c0, depth, pose)
    finite = points[np.all(np.isfinite(points), axis=1)]
    if finite.shape[0] < 8:
        return (0.06, 0.06, 0.10)
    span = np.percentile(finite, 90, axis=0) - np.percentile(finite, 10, axis=0)
    width = float(np.clip(max(span[0], span[1]), 0.03, 0.12))
    height_m = float(np.clip(span[2], 0.04, 0.16))
    return (width, width, height_m)


def reconcile_inventory(items: list[PerceivedObject]) -> list[PerceivedObject]:
    """Attach a registry record on colour+shape agreement; otherwise a temporary entry."""
    reconciled: list[PerceivedObject] = []
    for item in items:
        item = PerceivedObject(
            description=item.description,
            color=_canon_color(item.color),
            shape=item.shape,
            view_index=item.view_index,
            u=item.u, v=item.v,
            confidence=item.confidence,
            object_id=item.object_id,
            body_name=item.body_name,
            dimensions_m=item.dimensions_m,
            xy_m=item.xy_m,
        )
        record = record_for_color(item.color)
        if record is not None and _shape_agrees(item.shape or item.description, record):
            reconciled.append(PerceivedObject(
                description=item.description or record.display_name,
                color=record.primary_color,
                shape=record.shape_prior,
                view_index=item.view_index,
                u=item.u, v=item.v,
                confidence="high",
                object_id=record.object_id,
                body_name=record.body_name,
                dimensions_m=record.dimensions_m,
                xy_m=item.xy_m,
            ))
            continue
        body_name = None
        if item.xy_m is not None:
            body_name = nearest_free_body(np.asarray(item.xy_m, dtype=float))
        template = generic_record(item.color or "unknown", item.description or item.shape or "object")
        reconciled.append(PerceivedObject(
            description=item.description or template.display_name,
            color=str(item.color or "unknown").lower(),
            shape=_shape_token(item.shape) or "cylinder",
            view_index=item.view_index,
            u=item.u, v=item.v,
            confidence="low",
            object_id=template.object_id,
            body_name=body_name,
            dimensions_m=item.dimensions_m,
            xy_m=item.xy_m,
        ))
    return reconciled


def _parse_inventory(raw: object) -> list[dict]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        objects = raw.get("objects", raw.get("items", []))
        if isinstance(objects, list):
            return [item for item in objects if isinstance(item, dict)]
    return []


def perceive_scene(
    views: list[dict],
    *,
    ask_vlm=None,
) -> list[PerceivedObject]:
    """Inventory one fresh set of views. Nothing here is cached across calls.

    ``ask_vlm`` is the cloud backend. Without it, only registered colours that
    the local detector can see are returned.
    """
    if ask_vlm is None:
        return _registered_from_views(views)
    raw = ask_vlm(views)
    drafted: list[PerceivedObject] = []
    for index, item in enumerate(_parse_inventory(raw)):
        view_index = int(item.get("view_index", 0) or 0)
        view_index = min(max(view_index, 0), max(len(views) - 1, 0))
        try:
            u = float(item.get("u", 0.5))
            v = float(item.get("v", 0.5))
        except (TypeError, ValueError):
            u, v = 0.5, 0.5
        xy = None
        dimensions = (0.06, 0.06, 0.10)
        if 0 <= view_index < len(views):
            grounded = _ground_xy(views[view_index], u, v)
            if grounded is not None:
                xy = (float(grounded[0]), float(grounded[1]))
            dimensions = _measure_patch(views[view_index], u, v)
        drafted.append(PerceivedObject(
            description=str(item.get("description", item.get("name", "object"))),
            color=str(item.get("color", "unknown")).lower(),
            shape=str(item.get("shape", "cylinder")),
            view_index=view_index,
            u=u, v=v,
            xy_m=xy,
            dimensions_m=dimensions,
        ))
    vlm_items = reconcile_inventory(drafted)
    # Registered colours the local detector sees stay available even when the
    # cloud list misses them. Unregistered colours come only from the cloud.
    return _merge(vlm_items, _registered_from_views(views))


def _merge(primary: list[PerceivedObject], extra: list[PerceivedObject]) -> list[PerceivedObject]:
    merged = list(primary)
    seen = {(item.color, item.object_id) for item in merged}
    for item in extra:
        key = (item.color, item.object_id)
        if key in seen:
            continue
        if any(existing.color == item.color and existing.confidence == "high" for existing in merged):
            continue
        merged.append(item)
        seen.add(key)
    return merged


def _registered_from_views(views: list[dict]) -> list[PerceivedObject]:
    """Local colour detections. Requires the camera to still be at each view.

    Views captured earlier cannot be re-detected after the arm moves, because
    the colour detector reads the live camera pose. Callers that still have
    the arm parked pass ``live_detections`` instead; an empty view list is a
    valid, empty inventory and must not raise.
    """
    if not views or not all("rgb" in view and "live_detection" in view for view in views):
        found: list[PerceivedObject] = []
        for view in views:
            for item in view.get("registered", ()):
                if isinstance(item, PerceivedObject):
                    found.append(item)
        return reconcile_inventory(found) if found else found
    return []


def registered_detection(color: str, name: str, detection, view_index: int) -> PerceivedObject | None:
    record = record_for_color(color)
    if record is None or detection is None or not record.movable:
        return None
    xy = (float(detection.base_xyz_m[0]), float(detection.base_xyz_m[1]))
    height, width = 1, 1
    u = float(detection.center_uv[0]) / max(width, 1)
    return PerceivedObject(
        description=record.display_name,
        color=record.primary_color,
        shape=record.shape_prior,
        view_index=view_index,
        u=float(detection.center_uv[0]),
        v=float(detection.center_uv[1]),
        confidence="high",
        object_id=record.object_id,
        body_name=record.body_name,
        dimensions_m=record.dimensions_m,
        xy_m=xy,
    )


def collect_open_inventory(robot, model, data, renderer, leader) -> list[PerceivedObject]:
    """Fresh wide views, then one inventory. Not reused for the next instruction."""
    from camera_demo import CAMERA_NAME, render_rgbd
    from joint_control import LEFT_ABOVE
    from vision_detector import detect_colored_object

    angles = []
    for angle in OBJECT_BEARING_DEG.values():
        if not any(abs(float(angle) - existing) < 8.0 for existing in angles):
            angles.append(float(angle))
    views: list[dict] = []
    registered: list[PerceivedObject] = []
    for angle in angles:
        pose = LEFT_ABOVE.copy()
        pose[0] = np.deg2rad(angle)
        if not robot.move_to(pose, duration=0.22):
            break
        rgb, depth = render_rgbd(renderer, data)
        camera_id = model.camera(CAMERA_NAME).id
        base_id = model.body("link0").id
        view = {
            "angle_deg": float(angle),
            "rgb": rgb.copy(),
            "depth": depth.copy(),
            "cam_xpos": np.array(data.cam_xpos[camera_id], dtype=float, copy=True),
            "cam_xmat": np.array(data.cam_xmat[camera_id], dtype=float, copy=True),
            "fovy_deg": float(model.cam_fovy[camera_id]),
            "base_xpos": np.array(data.xpos[base_id], dtype=float, copy=True),
            "base_xmat": np.array(data.xmat[base_id], dtype=float, copy=True),
        }
        views.append(view)
        for record in OBJECTS:
            if not record.movable:
                continue
            detection = detect_colored_object(
                rgb, depth, record.primary_color, record.display_name,
                model, data, CAMERA_NAME,
            )
            item = registered_detection(
                record.primary_color, record.display_name, detection, len(views) - 1,
            )
            if item is not None:
                registered.append(item)
    if not leader.online:
        return _merge([], registered)
    # Local detection already walked every bearing. The cloud call only needs
    # a few wide frames, always including the unregistered fixture's bearing,
    # so the reply stays a short object list.
    vlm_views: list[dict] = []
    for view in views:
        angle = float(view["angle_deg"])
        if not vlm_views or min(
            abs(angle - float(kept["angle_deg"])) for kept in vlm_views
        ) > 70.0:
            vlm_views.append(view)
    pink = OBJECT_BEARING_DEG.get("pink")
    if pink is not None and views:
        closest = min(views, key=lambda view: abs(float(view["angle_deg"]) - float(pink)))
        if closest not in vlm_views:
            vlm_views.append(closest)
    cloud = []
    failure: Exception | None = None
    for _attempt in (1, 2):
        try:
            cloud = perceive_scene(
                vlm_views[:4], ask_vlm=lambda shots: leader.ask_inventory(shots),
            )
            failure = None
            break
        except Exception as error:
            failure = error
            print(f"  开放清点没有返回可解析清单（{type(error).__name__}），再问一次")
    if failure is not None:
        if getattr(leader, "mode", "") == "required":
            raise failure
        cloud = []
    return _merge(cloud, registered)
