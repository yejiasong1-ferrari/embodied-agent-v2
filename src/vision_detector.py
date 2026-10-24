"""Lightweight RGB-D detector with boxes, contours, and 3-D localization.

This runnable simulation backend is intentionally simple.  Its public API can
later be replaced by Grounding DINO + SAM 2 without changing the robot loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from object_registry import OBJECTS, ObjectRecord, generic_record, record_for_color

@dataclass
class Detection:
    label: str
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    contour: np.ndarray
    mask: np.ndarray
    center_uv: tuple[int, int]
    depth_m: float
    camera_xyz_m: np.ndarray
    world_xyz_m: np.ndarray
    base_xyz_m: np.ndarray
    # Who decided which pixel became the object: "color_rgbd" (HSV blob +
    # depth), "llm_point" (VLM pointing + depth), "color_fallback" (the VLM
    # pointing failed and the color path was used instead), or
    # "occluded_support_composite" (a held payload split a support into
    # several colour components during the local placement refresh).
    source: str = "color_rgbd"
    # D-1 record for a support detection. None for grasp targets.
    support_snap: dict | None = None


HSV_RANGES = {
    record.primary_color: list(record.hsv_ranges)
    for record in OBJECTS if record.hsv_ranges
}

# Perceived colours that are not registry entries. The pink bottle fixture
# renders around hue 159 under the scene light; this band is for refinement
# and the offline colour fallback, not for the registry.
EXTRA_HSV = {
    "pink": [((145, 80, 80), (175, 255, 255))],
}


def _color_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ranges = HSV_RANGES.get(color.lower()) or EXTRA_HSV.get(color.lower())
    if not ranges:
        raise ValueError(f"Unsupported fallback color: {color}")
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _pixel_to_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    uv: tuple[int, int],
    depth_m: float,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project one depth pixel through the MuJoCo perspective camera."""
    height, width = image_shape
    camera_id = model.camera(camera_name).id
    fovy = np.deg2rad(model.cam_fovy[camera_id])
    fy = height / (2.0 * np.tan(fovy / 2.0))
    fx = fy
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    u, v = uv

    # MuJoCo camera looks along local -Z; image V points down while local Y is up.
    camera_xyz = np.array(
        ((u - cx) * depth_m / fx, -(v - cy) * depth_m / fy, -depth_m)
    )
    rotation_world_from_camera = data.cam_xmat[camera_id].reshape(3, 3)
    world_xyz = data.cam_xpos[camera_id] + rotation_world_from_camera @ camera_xyz
    return camera_xyz, world_xyz


def _mask_world_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    object_mask: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    """Sample one segmented component as 3-D world points."""
    valid = (object_mask > 0) & np.isfinite(depth) & (depth > 0)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.zeros((0, 3), dtype=float)
    # At most ~2000 samples keeps this inexpensive at interactive frame rates.
    stride = max(1, rows.size // 2000)
    rows = rows[::stride]
    cols = cols[::stride]
    z = depth[rows, cols]
    height, width = depth.shape
    camera_id = model.camera(camera_name).id
    fovy = np.deg2rad(model.cam_fovy[camera_id])
    focal = height / (2.0 * np.tan(fovy / 2.0))
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    camera_points = np.column_stack(
        ((cols - cx) * z / focal, -(rows - cy) * z / focal, -z)
    )
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    return data.cam_xpos[camera_id] + camera_points @ rotation.T


def _mask_median_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    object_mask: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    """Robust 3-D object position from many mask pixels, not one biased pixel.

    A cup handle shifts a 2-D contour centroid toward the handle.  Back-
    projecting a regular sample of the entire segmented object and taking the
    coordinate-wise median is much less sensitive to that shape asymmetry.
    """
    world_points = _mask_world_points(
        model, data, camera_name, object_mask, depth
    )
    if world_points.size == 0:
        raise ValueError("object mask contains no valid depth")
    return np.median(world_points, axis=0)


def _component_matches_object(
    record: ObjectRecord,
    base_points: np.ndarray,
) -> bool:
    """Reject same-colour clutter whose metric footprint cannot be the object.

    Colour only proposes a component.  This geometry gate is deliberately
    broad: it removes the red book stack from red-cup candidates without
    encoding a fixed scene coordinate or pretending to be object recognition.
    """
    if base_points.shape[0] < 20:
        return False
    footprint = np.percentile(base_points[:, :2], 95, axis=0) - np.percentile(
        base_points[:, :2], 5, axis=0
    )
    short, long = sorted(float(value) for value in footprint)
    width, depth, _ = record.dimensions_m
    short_prior, long_prior = sorted((float(width), float(depth)))
    if record.shape_prior == "handled_cylinder":
        # Cup body is about 7 cm across; a handle may extend the long axis.
        return short <= short_prior + 0.025 and long <= long_prior + 0.095
    if record.shape_prior == "cylinder":
        return short <= short_prior + 0.023 and long <= long_prior + 0.048
    if record.shape_prior == "box":
        return short <= short_prior + 0.035 and long <= long_prior + 0.045
    if record.shape_prior == "support_cylinder":
        # A plate is approximately circular in the tabletop frame.  The
        # brown rectangular tray can enter the yellow HSV band from oblique
        # views, but its 0.24 x 0.32 m footprint is clearly anisotropic.
        aspect = long / max(short, 1e-6)
        support_height = float(np.median(base_points[:, 2]))
        return (
            long_prior - 0.10 <= short <= long_prior + 0.05
            and long <= long_prior + 0.06
            and aspect <= 1.24
            and support_height >= 0.030
        )
    return True


def detect_colored_object(
    rgb: np.ndarray,
    depth: np.ndarray,
    color: str,
    object_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str = "gripper_camera",
    near_xy: np.ndarray | None = None,
    near_radius: float = 0.10,
) -> Detection | None:
    """Detect a requested-color object and reject same-color background.

    Every contour is back-projected first.  Workspace/height checks prevent
    the blue floor and Panda base from masquerading as the blue box.
    """
    mask = _color_mask(rgb, color)
    record = record_for_color(color) or generic_record(color, object_name)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 80]
    if not contours:
        return None

    image_height, image_width = mask.shape
    candidates: list[tuple[float, Detection]] = []
    base_id = model.body("link0").id
    rotation_world_from_base = data.xmat[base_id].reshape(3, 3)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, width, height = cv2.boundingRect(contour)
        # Large background components usually touch an image boundary.
        if x <= 1 or y <= 1 or x + width >= image_width - 1 or y + height >= image_height - 1:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        if record.shape_prior == "support_cylinder":
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            circularity = float(
                4.0 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter)
            )
            # Same-colour trays/books are four-sided components.  Keep an
            # ellipse-like plate even when perspective lowers circularity.
            if len(polygon) <= 5 and circularity < 0.84:
                continue
        u = int(moments["m10"] / moments["m00"])
        v = int(moments["m01"] / moments["m00"])
        object_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 255, thickness=cv2.FILLED)
        samples = depth[(object_mask > 0) & np.isfinite(depth) & (depth > 0)]
        if samples.size == 0:
            continue
        depth_m = float(np.median(samples))
        camera_xyz, center_world_xyz = _pixel_to_world(
            model, data, camera_name, (u, v), depth_m, depth.shape
        )
        world_points = _mask_world_points(
            model, data, camera_name, object_mask, depth
        )
        if world_points.size == 0:
            continue
        base_points = (world_points - data.xpos[base_id]) @ rotation_world_from_base
        if not _component_matches_object(record, base_points):
            continue
        world_xyz = np.median(world_points, axis=0)
        if not np.all(np.isfinite(world_xyz)):
            world_xyz = center_world_xyz
        base_xyz = rotation_world_from_base.T @ (world_xyz - data.xpos[base_id])

        radial_distance = float(np.linalg.norm(base_xyz[:2]))
        # All tabletop objects in this scene sit away from the central robot
        # base and expose a surface above z=1.5 cm.  The floor fails the height
        # check; the blue link0 cover fails the radial-distance check.
        if not (0.20 <= radial_distance <= 0.80 and 0.015 <= base_xyz[2] <= 0.55):
            continue
        # Color alone is ambiguous under wide-angle views: the orange cube's
        # bright top can leak into the yellow HSV range.  A plate is a low
        # support surface, so reject a "plate" candidate sitting 30 cm high
        # on the pedestal before either the LLM or motion planner sees it.
        if record.shape_prior == "support_cylinder" and not (
            0.015 <= float(base_xyz[2]) <= 0.10
        ):
            continue

        detection = Detection(
            label=f"{color} {object_name}", score=1.0,
            bbox_xyxy=(x, y, x + width, y + height), contour=contour,
            mask=object_mask, center_uv=(u, v), depth_m=depth_m,
            camera_xyz_m=camera_xyz, world_xyz_m=world_xyz,
            base_xyz_m=base_xyz,
        )
        # Prefer substantial contours near the optical center.  Area helps
        # reject tiny colored reflections; center distance makes active scan
        # lock onto the clearest observation.  near_xy keeps a later overhead
        # view on the object the wide-angle photo already identified.
        if near_xy is not None:
            distance = float(np.linalg.norm(base_xyz[:2] - np.asarray(near_xy, dtype=float)))
            if distance > near_radius:
                continue
            score = -distance
        else:
            center_distance = np.hypot(u - image_width / 2, v - image_height / 2)
            score = float(cv2.contourArea(contour) / (1.0 + 0.02 * center_distance))
        candidates.append((score, detection))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def support_centroid_from_points(
    points: np.ndarray,
    near_xy: np.ndarray,
    *,
    search_radius_m: float = 0.16,
) -> np.ndarray | None:
    """XY centroid of support-height points already in the base frame.

    Returns None when the colour mask does not yield a usable support. The
    caller decides whether a shift of at least 5 mm is worth adopting.
    """
    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or cloud.shape[0] == 0:
        return None
    prior = np.asarray(near_xy, dtype=float)
    valid = (
        np.all(np.isfinite(cloud), axis=1)
        & (cloud[:, 2] >= 0.015)
        & (cloud[:, 2] <= 0.10)
        & (np.linalg.norm(cloud[:, :2] - prior, axis=1) <= float(search_radius_m))
    )
    if int(np.count_nonzero(valid)) < 30:
        return None
    return np.mean(cloud[valid, :2], axis=0)


def support_mask_centroid_xy(
    rgb: np.ndarray,
    depth: np.ndarray,
    color: str,
    near_xy: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData | None = None,
    camera_name: str = "gripper_camera",
    camera_pose: tuple | None = None,
) -> np.ndarray | None:
    """Centroid of the support colour mask on the frame that produced ``near_xy``."""
    record = record_for_color(color)
    if record is None or record.shape_prior != "support_cylinder":
        return None
    mask = _color_mask(rgb, color)
    rows, cols = np.nonzero(mask > 0)
    if rows.size < 30:
        return None
    finite = np.isfinite(depth[rows, cols]) & (depth[rows, cols] > 0.02) & (depth[rows, cols] < 2.0)
    rows, cols = rows[finite], cols[finite]
    if rows.size < 30:
        return None
    if rows.size > 4000:
        choice = np.linspace(0, rows.size - 1, 4000).astype(int)
        rows, cols = rows[choice], cols[choice]
    if camera_pose is None:
        if data is None:
            return None
        points = _backproject_pixels(model, data, camera_name, rows, cols, depth)
    else:
        points = _backproject_with_saved_pose(rows, cols, depth, camera_pose)
    return support_centroid_from_points(points, near_xy)


def _backproject_with_saved_pose(
    rows: np.ndarray,
    cols: np.ndarray,
    depth: np.ndarray,
    camera_pose: tuple,
) -> np.ndarray:
    """Base-frame points using the pose stored with the wide photo."""
    cam_xpos, cam_xmat, fovy_deg, base_xpos, base_xmat = camera_pose
    height, width = depth.shape
    z = depth[rows, cols]
    focal = height / (2.0 * np.tan(np.deg2rad(float(fovy_deg)) / 2.0))
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    camera_points = np.column_stack(
        ((cols - cx) * z / focal, -(rows - cy) * z / focal, -z)
    )
    rotation = np.asarray(cam_xmat, dtype=float).reshape(3, 3)
    world = np.asarray(cam_xpos, dtype=float) + camera_points @ rotation.T
    base_rotation = np.asarray(base_xmat, dtype=float).reshape(3, 3)
    return (world - np.asarray(base_xpos, dtype=float)) @ base_rotation


def _fragmented_support_center(
    point_groups: list[np.ndarray],
    near_xy: np.ndarray,
    diameter_m: float,
) -> np.ndarray | None:
    """Recover the centre of a known support from payload-occluded fragments.

    This is deliberately narrower than ordinary colour detection: it requires
    a prior support coordinate, support-height points, broad two-axis coverage,
    and keeps the existing 6 cm association gate.  It is only used by the
    placement refresh after the normal full-component detector has failed.
    """
    usable: list[np.ndarray] = []
    search_radius = 0.5 * float(diameter_m) + 0.035
    prior = np.asarray(near_xy, dtype=float)
    for group in point_groups:
        points = np.asarray(group, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            continue
        valid = (
            np.all(np.isfinite(points), axis=1)
            & (points[:, 2] >= 0.015)
            & (points[:, 2] <= 0.10)
            & (np.linalg.norm(points[:, :2] - prior, axis=1) <= search_radius)
        )
        if int(np.count_nonzero(valid)) >= 12:
            usable.append(points[valid])
    if not usable:
        return None
    points = np.concatenate(usable, axis=0)
    lower = np.percentile(points[:, :2], 2.5, axis=0)
    upper = np.percentile(points[:, :2], 97.5, axis=0)
    extent = upper - lower
    short, long = sorted(float(value) for value in extent)
    # A few unrelated yellow patches near the old coordinate must not become
    # a plate.  Even with the gripper and payload masking the middle, the
    # visible rim must span both axes of the expected support.
    if (
        short < float(diameter_m) - 0.10
        or long > float(diameter_m) + 0.06
        or long / max(short, 1e-6) > 1.55
    ):
        return None
    center = np.array([
        0.5 * (lower[0] + upper[0]),
        0.5 * (lower[1] + upper[1]),
        float(np.median(points[:, 2])),
    ])
    if float(np.linalg.norm(center[:2] - prior)) >= 0.06:
        return None
    return center


def detect_occluded_support(
    rgb: np.ndarray,
    depth: np.ndarray,
    color: str,
    object_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    near_xy: np.ndarray,
) -> Detection | None:
    """Associate a fragmented support rim during an overhead refresh."""
    record = record_for_color(color)
    if record is None or record.shape_prior != "support_cylinder":
        return None
    color_mask = _color_mask(rgb, color)
    contours, _ = cv2.findContours(
        color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 80]
    if not contours:
        return None
    base_id = model.body("link0").id
    base_rotation = data.xmat[base_id].reshape(3, 3)
    groups: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        image_height, image_width = color_mask.shape
        if (
            x <= 1 or y <= 1
            or x + width >= image_width - 1
            or y + height >= image_height - 1
        ):
            continue
        component_mask = np.zeros(color_mask.shape, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
        world_points = _mask_world_points(
            model, data, camera_name, component_mask, depth
        )
        if world_points.size == 0:
            continue
        base_points = (world_points - data.xpos[base_id]) @ base_rotation
        groups.append(base_points)
        masks.append(component_mask)
    center = _fragmented_support_center(
        groups, np.asarray(near_xy, dtype=float), record.dimensions_m[0]
    )
    if center is None:
        return None

    composite_mask = np.zeros(color_mask.shape, dtype=np.uint8)
    # Keep only fragments that contributed support-height samples near the
    # prior.  This preserves a useful mask for support-span estimation.
    search_radius = 0.5 * record.dimensions_m[0] + 0.035
    for points, component_mask in zip(groups, masks):
        valid = (
            (points[:, 2] >= 0.015)
            & (points[:, 2] <= 0.10)
            & (np.linalg.norm(points[:, :2] - near_xy, axis=1) <= search_radius)
        )
        if int(np.count_nonzero(valid)) >= 12:
            composite_mask |= component_mask
    rows, cols = np.nonzero(composite_mask)
    if rows.size == 0:
        return None
    pixel_points = np.column_stack((cols, rows)).astype(np.int32)
    contour = cv2.convexHull(pixel_points.reshape(-1, 1, 2))
    x, y, width, height = cv2.boundingRect(contour)
    camera_id = model.camera(camera_name).id
    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
    world_xyz = data.xpos[base_id] + base_rotation @ center
    camera_xyz = camera_rotation.T @ (world_xyz - data.cam_xpos[camera_id])
    return Detection(
        label=f"{color} {object_name}",
        score=0.85,
        bbox_xyxy=(x, y, x + width, y + height),
        contour=contour,
        mask=composite_mask,
        center_uv=(int(np.median(cols)), int(np.median(rows))),
        depth_m=float(-camera_xyz[2]),
        camera_xyz_m=camera_xyz,
        world_xyz_m=world_xyz,
        base_xyz_m=center,
        source="occluded_support_composite",
    )


def detection_from_image_point(
    rgb: np.ndarray,
    depth: np.ndarray,
    u_norm: float,
    v_norm: float,
    label: str,
    model: mujoco.MjModel,
    data: mujoco.MjData | None = None,
    camera_name: str = "gripper_camera",
    camera_pose: tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray] | None = None,
) -> Detection | None:
    """Turn one image point into a base-frame detection using local depth.

    The point comes from the wide-angle photo.  Depth around that pixel, not
    the largest same-color blob, supplies the grasp coordinate.  camera_pose is
    (cam_xpos, cam_xmat, fovy_deg, base_xpos, base_xmat) captured with the photo.
    """
    del rgb
    if camera_pose is None:
        if data is None:
            return None
        camera_id = model.camera(camera_name).id
        base_id = model.body("link0").id
        camera_pose = (
            np.array(data.cam_xpos[camera_id], dtype=float, copy=True),
            np.array(data.cam_xmat[camera_id], dtype=float, copy=True),
            float(model.cam_fovy[camera_id]),
            np.array(data.xpos[base_id], dtype=float, copy=True),
            np.array(data.xmat[base_id], dtype=float, copy=True),
        )
    cam_xpos, cam_xmat, fovy_deg, base_xpos, base_xmat = camera_pose
    height, width = depth.shape
    u = int(np.clip(round(float(u_norm) * (width - 1)), 0, width - 1))
    v = int(np.clip(round(float(v_norm) * (height - 1)), 0, height - 1))
    rows = cols = None
    for radius in (8, 18):
        row0, row1 = max(0, v - radius), min(height, v + radius + 1)
        col0, col1 = max(0, u - radius), min(width, u + radius + 1)
        window = depth[row0:row1, col0:col1]
        valid = np.isfinite(window) & (window > 0.02) & (window < 2.0)
        if int(np.count_nonzero(valid)) < 12:
            continue
        seed = float(np.median(window[valid]))
        close = valid & (np.abs(window - seed) < 0.03)
        if int(np.count_nonzero(close)) < 12:
            close = valid
        local_rows, local_cols = np.nonzero(close)
        rows = local_rows + row0
        cols = local_cols + col0
        break
    if rows is None or cols is None:
        return None
    depth_m = float(np.median(depth[rows, cols]))
    focal = height / (2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0))
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    z = depth[rows, cols]
    camera_points = np.column_stack(
        ((cols - cx) * z / focal, -(rows - cy) * z / focal, -z)
    )
    rotation = np.asarray(cam_xmat, dtype=float).reshape(3, 3)
    world_points = np.asarray(cam_xpos, dtype=float) + camera_points @ rotation.T
    base_rotation = np.asarray(base_xmat, dtype=float).reshape(3, 3)
    base_points = (world_points - np.asarray(base_xpos, dtype=float)) @ base_rotation
    world_xyz = np.median(world_points, axis=0)
    base_xyz = np.median(base_points, axis=0)
    radial_distance = float(np.linalg.norm(base_xyz[:2]))
    if not (0.20 <= radial_distance <= 0.80 and 0.015 <= float(base_xyz[2]) <= 0.55):
        return None
    if "plate" in label and not (0.015 <= float(base_xyz[2]) <= 0.10):
        return None
    camera_xyz = np.array(
        ((u - cx) * depth_m / focal, -(v - cy) * depth_m / focal, -depth_m)
    )
    mask = np.zeros(depth.shape, dtype=np.uint8)
    mask[rows, cols] = 255
    x0, y0 = int(cols.min()), int(rows.min())
    x1, y1 = int(cols.max()) + 1, int(rows.max()) + 1
    contour = np.array(
        [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.int32
    )
    return Detection(
        label=label, score=1.0,
        bbox_xyxy=(x0, y0, x1, y1), contour=contour, mask=mask,
        center_uv=(u, v), depth_m=depth_m,
        camera_xyz_m=camera_xyz, world_xyz_m=world_xyz, base_xyz_m=base_xyz,
    )


def _backproject_pixels(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    rows: np.ndarray,
    cols: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    """Back-project pixels into the robot base frame. Rows are empty-safe."""
    if rows.size == 0:
        return np.zeros((0, 3))
    z = depth[rows, cols]
    height, width = depth.shape
    camera_id = model.camera(camera_name).id
    fovy = np.deg2rad(model.cam_fovy[camera_id])
    focal = height / (2.0 * np.tan(fovy / 2.0))
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    camera_points = np.column_stack(
        ((cols - cx) * z / focal, -(rows - cy) * z / focal, -z)
    )
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    world = data.cam_xpos[camera_id] + camera_points @ rotation.T
    base_id = model.body("link0").id
    base_rotation = data.xmat[base_id].reshape(3, 3)
    return (world - data.xpos[base_id]) @ base_rotation


def _fit_section_circle(points_xy: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Algebraic circle. None when the points do not determine one."""
    xy = np.asarray(points_xy, dtype=float)
    if xy.shape[0] < 3:
        return None
    design = np.column_stack((xy[:, 0], xy[:, 1], np.ones(xy.shape[0])))
    target = -(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    radius_sq = float(solution[0] ** 2 + solution[1] ** 2) / 4.0 - float(solution[2])
    if not np.isfinite(radius_sq) or radius_sq <= 0.0:
        return None
    center = np.array((-float(solution[0]) / 2.0, -float(solution[1]) / 2.0))
    if not np.all(np.isfinite(center)):
        return None
    return center, float(np.sqrt(radius_sq))


def geometric_section_center_xy(points_xy: np.ndarray) -> np.ndarray:
    """Center of the body circle after the handle protrusion is left out.

    A handle sits outside the cup wall, so each refit drops points beyond
    the current circle. A symmetric section never loses points and stays
    on the fitted center.
    """
    xy = np.asarray(points_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2 or xy.shape[0] == 0:
        return np.zeros(2)
    xy = xy[np.all(np.isfinite(xy[:, :2]), axis=1), :2]
    if xy.shape[0] == 0:
        return np.zeros(2)
    fallback = np.median(xy, axis=0)
    if xy.shape[0] < 30:
        return fallback
    fitted = _fit_section_circle(xy)
    if fitted is None:
        return fallback
    center, radius = fitted
    for _ in range(3):
        radial = np.linalg.norm(xy - center, axis=1)
        keep = radial <= radius
        if int(np.count_nonzero(keep)) < 15 or int(np.count_nonzero(keep)) == xy.shape[0]:
            break
        refit = _fit_section_circle(xy[keep])
        if refit is None:
            break
        updated, radius = refit
        if float(np.linalg.norm(updated - center)) < 1e-5:
            return updated
        center = updated
    return center


def object_span(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    object_mask: np.ndarray,
    depth: np.ndarray,
) -> dict[str, float] | None:
    """Width, depth and top height of one segmented object, in the base frame.

    Percentiles ignore a few mixed pixels on the object boundary.  The top
    height is what later motions have to clear and where a top grasp starts.
    """
    valid = (object_mask > 0) & np.isfinite(depth) & (depth > 0.02) & (depth < 2.0)
    rows, cols = np.nonzero(valid)
    if rows.size < 40:
        return None
    stride = max(1, rows.size // 1500)
    points = _backproject_pixels(
        model, data, camera_name, rows[::stride], cols[::stride], depth
    )
    finite = points[np.all(np.isfinite(points), axis=1)]
    if finite.shape[0] < 30:
        return None
    xy = finite[:, :2]
    centered = xy - np.median(xy, axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    projected = centered @ axes
    lengths = np.array([
        np.percentile(projected[:, index], 95)
        - np.percentile(projected[:, index], 5)
        for index in range(2)
    ])
    major_index = int(np.argmax(lengths))
    minor_index = 1 - major_index
    major_axis = axes[:, major_index]
    minor_axis = axes[:, minor_index]

    def closing_yaw(axis: np.ndarray) -> float:
        # kinematics.py defines yaw 0 as closing along world Y.  Finger axes
        # are bidirectional, so normalize equivalent angles into ±90 degrees.
        angle = float(np.arctan2(axis[0], axis[1]))
        return float((angle + np.pi / 2.0) % np.pi - np.pi / 2.0)

    radial = np.linalg.norm(centered, axis=1)
    core_radius = float(np.percentile(radial, 65))
    outer_cut = float(np.percentile(radial, 96))
    outer = centered[radial >= outer_cut]
    protrusion = np.median(outer, axis=0) if outer.size else np.zeros(2)
    protrusion_norm = float(np.linalg.norm(protrusion))
    has_handle = bool(
        core_radius > 0.005
        and outer_cut / core_radius > 1.55
        and protrusion_norm > 0.012
    )
    if has_handle:
        handle_axis = protrusion / protrusion_norm
        # Put the two fingertips on faces perpendicular to the handle, so no
        # pad tries to close through it.  closing_yaw consumes the desired
        # gripper closing axis, not an ordinary object yaw.
        avoid_axis = np.array((-handle_axis[1], handle_axis[0]))
        handle_avoid_yaw = closing_yaw(avoid_axis)
        handle_direction = float(np.arctan2(handle_axis[1], handle_axis[0]))
    else:
        handle_avoid_yaw = 0.0
        handle_direction = 0.0

    return {
        "extent_x": float(np.percentile(finite[:, 0], 95) - np.percentile(finite[:, 0], 5)),
        "extent_y": float(np.percentile(finite[:, 1], 95) - np.percentile(finite[:, 1], 5)),
        "top_z": float(np.percentile(finite[:, 2], 95)),
        "major_m": float(lengths[major_index]),
        "minor_m": float(lengths[minor_index]),
        "major_yaw_rad": closing_yaw(major_axis),
        "minor_yaw_rad": closing_yaw(minor_axis),
        "has_handle": float(has_handle),
        "handle_direction_rad": handle_direction,
        "handle_avoid_yaw_rad": handle_avoid_yaw,
        "handle_confidence": float(
            np.clip(outer_cut / max(core_radius, 1e-6) - 1.0, 0.0, 1.0)
        ),
    }


def _workspace_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    depth: np.ndarray,
) -> np.ndarray:
    """Visible tabletop points in the base frame, with the arm itself removed."""
    valid = np.isfinite(depth) & (depth > 0.08) & (depth < 1.6)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.zeros((0, 3))
    stride = max(1, rows.size // 2500)
    points = _backproject_pixels(
        model, data, camera_name, rows[::stride], cols[::stride], depth
    )
    radial = np.linalg.norm(points[:, :2], axis=1)
    keep = (
        np.all(np.isfinite(points), axis=1)
        & (radial > 0.12)
        & (radial < 0.95)
        & (points[:, 2] > 0.02)
        & (points[:, 2] < 0.62)
    )
    # The wrist camera also sees the forearm.  Those points sit next to the
    # hand and would otherwise be treated as a tall obstacle.
    robot_names = ("hand", "left_finger", "right_finger", "link7", "link6", "link5")
    base_id = model.body("link0").id
    base_rotation = data.xmat[base_id].reshape(3, 3)
    centers = np.stack([
        (data.xpos[model.body(name).id] - data.xpos[base_id]) @ base_rotation
        for name in robot_names
    ])
    nearest = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2).min(axis=1)
    keep &= nearest > 0.16
    if not np.any(keep):
        return np.zeros((0, 3))
    return points[keep]


def workspace_ceiling_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    depth: np.ndarray,
) -> float:
    """Highest tabletop surface currently visible to the wrist depth camera.

    The arm's transit height is derived from this value.  Points on the robot
    base, the floor, and the sky are excluded so one bad pixel cannot set it.
    """
    points = _workspace_points(model, data, camera_name, depth)
    if points.shape[0] == 0:
        return 0.0
    # The 99th percentile is the visible top, without one flying depth spike.
    return float(np.percentile(points[:, 2], 99))


def mark_tall_obstacles(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    depth: np.ndarray,
    *,
    min_top_m: float = 0.12,
) -> list[dict]:
    """Cluster tall surfaces seen in this depth frame.

    Short tabletop items stay unmarked.  Pedestals and anything else tall
    enough to meet the forearm are returned as ``{xy, top_z, radius}``.
    """
    points = _workspace_points(model, data, camera_name, depth)
    if points.shape[0] == 0:
        return []
    tall = points[points[:, 2] >= min_top_m]
    if tall.shape[0] < 12:
        return []
    cell = 0.08
    keys = np.floor(tall[:, :2] / cell).astype(int)
    markers: list[dict] = []
    for key in np.unique(keys, axis=0):
        chosen = tall[np.all(keys == key, axis=1)]
        if chosen.shape[0] < 8:
            continue
        markers.append({
            "xy": np.median(chosen[:, :2], axis=0),
            "top_z": float(np.percentile(chosen[:, 2], 95)),
            "radius": 0.14,
        })
    markers.sort(key=lambda item: float(item["top_z"]), reverse=True)
    return merge_tall_obstacles([], markers)


def merge_tall_obstacles(existing: list[dict], fresh: list[dict]) -> list[dict]:
    """Fold new depth markers into ``existing``. Returns only the new ones."""
    added: list[dict] = []
    for marker in fresh:
        xy = np.asarray(marker["xy"], dtype=float)
        top_z = float(marker["top_z"])
        for kept in existing:
            if float(np.linalg.norm(np.asarray(kept["xy"]) - xy)) < 0.12:
                if top_z > float(kept["top_z"]):
                    kept["top_z"] = top_z
                    kept["xy"] = xy.copy()
                break
        else:
            created = {"xy": xy.copy(), "top_z": top_z, "radius": float(marker.get("radius", 0.14))}
            existing.append(created)
            added.append(created)
    return added


def annotate(rgb: np.ndarray, detection: Detection) -> np.ndarray:
    """Draw the target box, exact contour, center, and depth."""
    output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = detection.bbox_xyxy
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.drawContours(output, [detection.contour], -1, (255, 0, 255), 2)
    cv2.drawMarker(output, detection.center_uv, (0, 255, 255), cv2.MARKER_CROSS, 24, 3)
    caption = f"{detection.label}  depth={detection.depth_m:.3f}m"
    cv2.putText(output, caption, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
