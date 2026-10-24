"""Instruction-driven active RGB-D perception with one Panda wrist camera.

Example:
    .venv/bin/mjpython src/camera_demo.py --instruction "拿起红色杯子"
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence

import mujoco
import numpy as np
from PIL import Image

from joint_control import GRIPPER_OPEN, LEFT_ABOVE, PandaController, create_model, scaled_motion_limits
from kinematics import solve_hand_ik
from motion_planning import (
    OMPL_RANDOM_SEED, _densely_valid, plan_rrt_connect,
    sample_executed_joint_trajectory,
)
from object_registry import OBJECTS, generic_record, record_for_body, record_for_color
from scene_layout import OBJECT_BEARING_DEG, place_movable
from embodied_agent import (
    GraspStrategyUpdate, clip_grasp_strategy, offline_grasp_replan,
)
from task_planner import NotFoundReport, TaskSpec
from vision_detector import (
    Detection, annotate, detect_colored_object, detect_occluded_support,
    detection_from_image_point, mark_tall_obstacles, merge_tall_obstacles,
    object_span, support_mask_centroid_xy, workspace_ceiling_m,
    _backproject_pixels,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAMERA_NAME = "gripper_camera"
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
SCAN_ANGLES_DEG = (
    135.0, 112.5, 90.0, 67.5, 45.0, 22.5, 0.0,
    -22.5, -45.0, -67.5, -90.0, -112.5, -135.0,
)


def reset_scene(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Restore the Panda and the designed tabletop."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    place_movable(model, data)
    data.qvel[:] = 0.0
    data.ctrl[-1] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)


def render_rgbd(
    renderer: mujoco.Renderer, data: mujoco.MjData
) -> tuple[np.ndarray, np.ndarray]:
    """Capture aligned RGB uint8 and metric depth float32 arrays."""
    renderer.update_scene(data, camera=CAMERA_NAME)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return rgb, depth


def render_result_overview(
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    support_xy: np.ndarray,
) -> np.ndarray:
    """Render proof of the final placement from a stable external camera."""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = (float(support_xy[0]), float(support_xy[1]), 0.09)
    camera.distance = 0.72
    camera.azimuth = 135.0
    camera.elevation = -42.0
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def _note_ceiling(
    ceiling: list[float] | None,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    depth: np.ndarray,
) -> None:
    if ceiling is None:
        return
    ceiling[0] = max(ceiling[0], workspace_ceiling_m(model, data, CAMERA_NAME, depth))


def _note_obstacles(
    obstacles: list[dict] | None,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    depth: np.ndarray,
) -> None:
    """Record tall surfaces in this frame so later motion can clear them."""
    if obstacles is None:
        return
    fresh = mark_tall_obstacles(model, data, CAMERA_NAME, depth)
    for marker in merge_tall_obstacles(obstacles, fresh):
        xy = marker["xy"]
        print(
            f"  标注高障碍 ({float(xy[0]):+.2f}, {float(xy[1]):+.2f}) "
            f"顶 {marker['top_z']:.3f} m"
        )


def search_for_target(
    task: TaskSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    robot: PandaController,
    ceiling: list[float] | None = None,
    obstacles: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, Detection | None, float | None]:
    """Move the wrist camera until the language-selected object is detected."""
    rgb, depth = render_rgbd(renderer, data)
    _note_ceiling(ceiling, model, data, depth)
    best: tuple[float, float] | None = None  # (distance-to-image-center, angle)
    current_angle: float | None = None
    consecutive_worse = 0
    angles = SCAN_ANGLES_DEG
    # Registered colours keep the original sweep. An unregistered colour starts
    # at the bearing stored for that fixture so the scan does not lock a
    # similarly coloured registry object first.
    if record_for_color(task.object_color) is None:
        preferred = OBJECT_BEARING_DEG.get(task.object_color)
        if preferred is not None:
            angles = (float(preferred),) + tuple(
                angle for angle in SCAN_ANGLES_DEG if abs(float(angle) - float(preferred)) > 8.0
            )
    for angle_deg in angles:
        current_angle = angle_deg
        print(f"→ 搜索 {task.object_color} {task.object_name}: base {angle_deg:+.0f}°")
        pose = LEFT_ABOVE.copy()
        pose[0] = np.deg2rad(angle_deg)
        if not robot.move_to(pose, duration=0.28):
            return rgb, depth, None, None

        rgb, depth = render_rgbd(renderer, data)
        _note_ceiling(ceiling, model, data, depth)
        _note_obstacles(obstacles, model, data, depth)
        detection = detect_colored_object(
            rgb, depth, task.object_color, task.object_name, model, data, CAMERA_NAME
        )
        if detection is not None:
            u, v = detection.center_uv
            center_distance = float(np.hypot(u - IMAGE_WIDTH / 2, v - IMAGE_HEIGHT / 2))
            print(f"  候选目标 @ pixel={detection.center_uv}, center error={center_distance:.1f}px")
            if best is None or center_distance < best[0]:
                best = (center_distance, angle_deg)
                consecutive_worse = 0
            else:
                consecutive_worse += 1
            if center_distance <= 180.0:
                print("  ✓ 目标已锁定，立即结束扫描，到正上方再微调")
                break
            if consecutive_worse >= 2 and best is not None:
                print("  ✓ 视角连续变差，回到已记录的最佳角度")
                break
        else:
            print("  未发现，继续转动")

    if best is None:
        return rgb, depth, None, None

    # Like a person looking back at the clearest view, return to the angle
    # where the target was closest to the optical center.
    best_angle = best[1]
    if current_angle is None or abs(best_angle - current_angle) > 0.1:
        print(f"→ 回到最佳观察角度 {best_angle:+.1f}°")
        best_pose = LEFT_ABOVE.copy()
        best_pose[0] = np.deg2rad(best_angle)
        if not robot.move_to(best_pose, duration=0.28):
            return rgb, depth, None, None
    else:
        print(f"✓ 当前角度 {best_angle:+.1f}° 已是最佳视角")
    rgb, depth = render_rgbd(renderer, data)
    _note_ceiling(ceiling, model, data, depth)
    _note_obstacles(obstacles, model, data, depth)
    detection = detect_colored_object(
        rgb, depth, task.object_color, task.object_name, model, data, CAMERA_NAME
    )
    if detection is not None:
        print(f"✓ 锁定 {detection.label} @ pixel={detection.center_uv}")
        print(f"  depth={detection.depth_m:.3f} m")
        print(f"  world xyz={np.round(detection.world_xyz_m, 3)} m")
    return rgb, depth, detection, best_angle


def open_image(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        Image.open(path).show()


def detection_to_dict(detection: Detection | None) -> dict | None:
    if detection is None:
        return None
    return {
        "label": detection.label,
        "source": detection.source,
        "score": detection.score,
        "bbox_xyxy": list(detection.bbox_xyxy),
        "center_uv": list(detection.center_uv),
        "depth_m": detection.depth_m,
        "camera_xyz_m": detection.camera_xyz_m.tolist(),
        "world_xyz_m": detection.world_xyz_m.tolist(),
        "base_xyz_m": detection.base_xyz_m.tolist(),
        "contour_points": int(len(detection.contour)),
        "mask_pixels": int(np.count_nonzero(detection.mask)),
    }


# Fingertips sit about 11.5 cm below the hand origin.  Transit keeps that
# whole gripper above the tallest surface the depth camera has seen.
FINGER_BELOW_HAND = 0.115
# The rubber pad that actually touches the object hangs this far below the hand origin.
FINGER_PAD_BELOW_HAND = 0.147
TRANSIT_MARGIN = 0.05
RELEASE_CLEARANCE = 0.003
# A grasp only has to leave the support by a few millimetres to prove that it
# is carrying the object.  Longer lifts belong to transport planning, where
# the payload envelope and the complete executed trajectory are validated.
GRASP_VERIFY_LIFT_M = 0.005
GRASP_VERIFY_MIN_RISE_M = 0.003
GRASP_VERIFY_MAX_COMMAND_M = 0.030
# Reuse the established object-motion safety gates for both grasp verification
# and gradual release.  Force balance is deliberately absent: it is evidence,
# not a release gate for asymmetric or curved geometry.
OBJECT_STABILITY_MAX_SHIFT_M = 0.010
OBJECT_STABILITY_MAX_TILT_DEG = 8.0
GRASP_VERIFY_MAX_SLIP_M = 0.010
# Manager's enumerated choices map to these local values.  The medium/wide
# entries reproduce the behaviour before PickStrategy existed.
LIFT_OFFSET_M = {"low": 0.05, "medium": 0.07, "high": 0.09}
PRE_OPEN_MAX_M = 0.078


def _lift_offset_for(lift_height: str) -> float:
    return float(LIFT_OFFSET_M.get(str(lift_height), LIFT_OFFSET_M["medium"]))


def _pre_open_gap(grip_width: str, measured_width_m: float) -> float:
    """Pre-opening before descent.  Never narrower than object + 1 cm."""
    width = float(np.clip(measured_width_m, 0.025, PRE_OPEN_MAX_M))
    if grip_width == "narrow":
        gap = width + 0.010
    elif grip_width == "medium":
        gap = width + 0.020
    else:
        gap = PRE_OPEN_MAX_M
    return float(np.clip(gap, 0.030, PRE_OPEN_MAX_M))


# Largest single pre-release correction; anything bigger means the placement
# geometry is wrong and must be re-planned, not nudged.
PLACE_MICRO_ADJUST_MAX_M = 0.02

# The previous one-shot visual servo accepted up to 3 mm of residual error.
# Online evidence showed a repeatable 2.7 mm under-travel, which passed that
# gate but was large enough to create a one-pad grasp.  Keep the existing
# 3 mm object-motion safety gate and tighten hand/object centering to 1 mm.
GRASP_SERVO_TARGET_M = 0.001
GRASP_SERVO_MAX_STEP_M = 0.006
GRASP_SERVO_MAX_ITERS = 4
GRASP_SERVO_OBJECT_MOTION_LIMIT_M = 0.003

# Inline closure centering is bounded more tightly than the existing 10 mm
# object-displacement abort.  Five millimetres is also the A-3 per-correction
# ceiling; the whole close phase may never accumulate more than that amount.
GRASP_CENTERING_MAX_STEP_M = 0.005
GRASP_CENTERING_TOTAL_BUDGET_M = 0.005
# Batch 5's only new number. Full feedforward of a +2.875 mm white-mug
# delta produced −1.750 mm, a swing about 1.6 times the command, so the
# first sample takes half of δ instead of all of it.
CENTERING_DAMPED_GAIN = 0.5
GRASP_CENTERING_MAX_EVENTS = 2
GRASP_INLINE_CENTERING_MIN_FORCE_N = 0.35
GRASP_INLINE_CENTERING_FORCE_FRACTION = 0.55
# Existing Panda-model calibration: controller aperture is about 6 mm wider
# than the literal inner-pad gap. I-1 records that physical pad aperture so a
# one-sided contact lower bound is not inflated by fixed gripper geometry.
GRIPPER_INNER_PAD_OFFSET_M = 0.006
# After the measured-width gap is reached, allow a tightly bounded compliance
# search. Every 0.25 mm step is guarded by peak force, object displacement and
# tilt; this is geometry-based, not a named-object exception.
GRASP_SEATING_EXTRA_M = 0.003
# One hand drop after both pads touch and the seating budget is spent.
# The 8 mm support clearance already used for finger height stays in force.
GRASP_RESEAT_STEP_M = 0.012
GRASP_RESEAT_MIN_STEP_M = 0.008
# A side approach inside this angle of a measured protrusion is refused.
PROTRUSION_APPROACH_CLEARANCE_RAD = np.deg2rad(50.0)
GRASP_TRIAL_LIFT_FORCE_FRACTION = 0.65
# A side below this has not formed a real contact. It is the same noise
# floor already used when a pad force is treated as absent. It is not the
# higher load that used to be required before a micro-lift.
GRASP_CONTACT_PRESENT_N = 0.05
# One 0.3 mm in-grasp translation took the loaded side from 0.517 N to 3.246 N.
# Substeps stay at or below the 0.1 mm bound so force can be remeasured between them.
TRANSLATION_FORCE_SUBSTEP_M = 0.0001
# Diagnostic override. A positive value is clipped to the committed allowance
# through 6.0 mm and never bypasses the live force/motion interlocks.
EXPERIMENT_SEATING_EXTRA_M: float | None = None
EXPERIMENT_SEATING_EXTRA_MAX_M = 0.006
# Independent of the 5 mm translation budget. Declared here so the summary
# can point at one number: at most ±2°, one degree per step.
GRASP_YAW_ADJUST_STEP_RAD = float(np.deg2rad(1.0))
GRASP_YAW_ADJUST_TOTAL_RAD = float(np.deg2rad(2.0))

# D-1 adopts a support-mask centroid only when it disagrees by at least 5 mm.
# D-2 will not replace the locked support with a refresh farther than 20 mm.
# The 6 cm association gate stays exactly where it is.
SUPPORT_POINT_SNAP_MIN_M = 0.005
SUPPORT_REFRESH_ADOPT_MAX_M = 0.020
SUPPORT_ASSOCIATION_GATE_M = 0.06

# Batch9 O-1: reproduce the 2026-10-08 cross-base planning incident exactly.
# The timeout (1.6 s), dense joint step (0.035 rad), candidate count (the
# existing three IK branches), and 12/15 degree payload pose tolerance are all
# existing values.  Only the planner RNG seed is new, and it is diagnostic,
# not a motion or safety threshold.
TRANSPORT_OMPL_TIMEOUT_S = 1.6
TRANSPORT_OMPL_DENSE_STEP_RAD = 0.035
# Index bins for a saved route. They do not change clearance or collision checks.
TRANSPORT_ROUTE_SECTOR_M = 0.10
TRANSPORT_ROUTE_YAW_BUCKET_RAD = np.pi / 4.0
TRANSPORT_ROUTE_LIBRARY_CAP = 32
TRANSPORT_ROUTE_LIBRARY = (
    Path(__file__).resolve().parents[1] / "outputs" / "transport_routes.json"
)


def reference_point_within_color_gate(
    point_xy: np.ndarray,
    color_xy: np.ndarray | None,
) -> bool:
    """Keep an LLM support point only when it is inside the existing 6 cm gate.

    A missing colour detection cannot veto the point. The gate value is the
    same association gate used by support refresh; this does not add a limit.
    """
    if color_xy is None:
        return True
    point = np.asarray(point_xy, dtype=float)[:2]
    color = np.asarray(color_xy, dtype=float)[:2]
    return float(np.linalg.norm(point - color)) < SUPPORT_ASSOCIATION_GATE_M


def decide_support_snap(
    original_xy: np.ndarray,
    centroid_xy: np.ndarray | None,
) -> dict[str, object]:
    """Adopt the colour-mask centroid when it is at least 5 mm from the point."""
    original = np.asarray(original_xy, dtype=float)[:2]
    if centroid_xy is None:
        return {
            "adopted": False,
            "reason": "mask_unavailable",
            "original_xy_m": original.tolist(),
            "centroid_xy_m": None,
            "deviation_m": None,
            "xy": original.copy(),
        }
    centroid = np.asarray(centroid_xy, dtype=float)[:2]
    deviation = float(np.linalg.norm(centroid - original))
    adopted = deviation >= SUPPORT_POINT_SNAP_MIN_M
    return {
        "adopted": adopted,
        "reason": "centroid" if adopted else "within_5mm",
        "original_xy_m": original.tolist(),
        "centroid_xy_m": centroid.tolist(),
        "deviation_m": deviation,
        "xy": centroid.copy() if adopted else original.copy(),
    }


def apply_support_point_snap(
    detection: Detection,
    rgb: np.ndarray,
    depth: np.ndarray,
    color: str,
    model: mujoco.MjModel,
    data: mujoco.MjData | None = None,
    camera_pose: tuple | None = None,
) -> dict[str, object]:
    """Snap a support point to the colour-mask centroid on that same frame."""
    centroid = support_mask_centroid_xy(
        rgb, depth, color, detection.base_xyz_m[:2], model, data,
        CAMERA_NAME, camera_pose,
    )
    decision = decide_support_snap(detection.base_xyz_m[:2], centroid)
    detection.support_snap = {
        key: value for key, value in decision.items() if key != "xy"
    }
    if decision["adopted"]:
        detection.base_xyz_m = np.array(detection.base_xyz_m, dtype=float, copy=True)
        detection.base_xyz_m[:2] = np.asarray(decision["xy"], dtype=float)
        print(
            "  支撑点吸附到色块质心："
            f"{np.round(decision['original_xy_m'], 3)} → "
            f"{np.round(detection.base_xyz_m[:2], 3)}，"
            f"偏差 {float(decision['deviation_m']) * 1000:.0f} mm"
        )
    elif decision["reason"] == "mask_unavailable":
        print("  支撑色块不可用，保留原始放置点")
    else:
        print(
            f"  支撑色块质心距原始点 {float(decision['deviation_m']) * 1000:.1f} mm，"
            "小于 5 mm，保留原始点"
        )
    return decision


def arbitrate_support_refresh(
    locked_xy: np.ndarray,
    refresh_xy: np.ndarray | None,
) -> dict[str, object]:
    """Keep the locked support unless the refresh is inside 20 mm and the 6 cm gate."""
    locked = np.asarray(locked_xy, dtype=float)[:2]
    if refresh_xy is None:
        return {
            "adopted": False,
            "source": "locked",
            "reason": "not_found",
            "deviation_m": None,
            "xy": locked.copy(),
            "within_association_gate": False,
        }
    refresh = np.asarray(refresh_xy, dtype=float)[:2]
    deviation = float(np.linalg.norm(refresh - locked))
    if deviation >= SUPPORT_ASSOCIATION_GATE_M:
        return {
            "adopted": False,
            "source": "locked",
            "reason": "association_gate",
            "deviation_m": deviation,
            "xy": locked.copy(),
            "within_association_gate": False,
        }
    if deviation > SUPPORT_REFRESH_ADOPT_MAX_M:
        return {
            "adopted": False,
            "source": "locked",
            "reason": "deviation",
            "deviation_m": deviation,
            "xy": locked.copy(),
            "within_association_gate": True,
        }
    return {
        "adopted": True,
        "source": "refresh_adopted",
        "reason": "within_20mm",
        "deviation_m": deviation,
        "xy": refresh.copy(),
        "within_association_gate": True,
    }


def place_error_pair(
    object_xy: np.ndarray,
    used_xy: np.ndarray,
    locked_xy: np.ndarray,
) -> dict[str, float]:
    """Both reported errors. Placement still judges the reference actually used."""
    point = np.asarray(object_xy, dtype=float)[:2]
    return {
        "error_vs_used_support_m": float(np.linalg.norm(point - np.asarray(used_xy, dtype=float)[:2])),
        "error_vs_locked_reference_m": float(np.linalg.norm(point - np.asarray(locked_xy, dtype=float)[:2])),
    }


def inline_translation_should_brake(
    forces_before: dict[str, float],
    forces_after: dict[str, float],
    high_side: str,
    hand_delta_xy: np.ndarray,
    object_delta_xy: np.ndarray,
) -> str | None:
    """Stop every in-grasp correction once a slide starts loading one pad."""
    before = float(forces_before.get(high_side, 0.0))
    after = float(forces_after.get(high_side, 0.0))
    if after > before + 1e-6:
        return "contact_force_rose"
    hand = float(np.linalg.norm(np.asarray(hand_delta_xy, dtype=float)))
    followed = float(np.linalg.norm(np.asarray(object_delta_xy, dtype=float)))
    if hand > 1e-4 and followed > 0.5 * hand:
        return "object_followed"
    return None


def inline_yaw_adjust_rad(
    *,
    high_side: str,
    yaw_policy: str,
    spent_rad: float,
    servo_residual_m: float,
    translation_attempted: bool,
    force_rising: bool,
) -> float | None:
    """±1° step, ±2° total. ``avoid_handle`` is never overridden.

    Positive when the right pad touched first, negative when the left pad did.
    That matches ``solve_hand_ik``: increasing ``yaw_rad`` swings the closing
    axis so a left contact is driven harder, so the relief step has the
    opposite sign.  The step is skipped until the visual residual is already
    inside 1 mm, or one translation has been tried, and the contact force is
    still climbing.
    """
    if yaw_policy == "avoid_handle":
        return None
    if not force_rising:
        return None
    if float(servo_residual_m) > GRASP_SERVO_TARGET_M and not translation_attempted:
        return None
    remaining = GRASP_YAW_ADJUST_TOTAL_RAD - abs(float(spent_rad))
    if remaining <= 1e-9:
        return None
    sign = -1.0 if high_side == "left" else 1.0
    step = min(GRASP_YAW_ADJUST_STEP_RAD, remaining)
    if abs(float(spent_rad) + sign * step) > GRASP_YAW_ADJUST_TOTAL_RAD + 1e-9:
        return None
    return sign * step


_TRANSPORT_HEIGHT_EXTRAS = (0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.26, 0.32)
# Existing high tier: +0.20 m of travel height and +0.10 m of outer radius.
_CROSS_BASE_SEED_EXTRA = 0.20
_TRANSPORT_SEED_CACHE: dict[tuple[int, int], tuple[float, int]] = {}


def _bearing_sector(xy: np.ndarray, sectors: int = 8) -> int:
    angle = float(np.arctan2(float(xy[1]), float(xy[0])))
    return int(np.floor((angle + np.pi) / (2.0 * np.pi) * sectors)) % sectors


def order_transport_plans(
    crosses_base: bool,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
) -> list[tuple[float, int]]:
    """Attempt order only. Every entry still goes through the existing validator.

    A cache miss on a same-side move is the historical low-to-high list.
    Opposite sides of the base start at the existing high radius/height tier.
    """
    if crosses_base:
        historical = [(extra, slot) for extra in _TRANSPORT_HEIGHT_EXTRAS for slot in (0, 1)]
        front = [(_CROSS_BASE_SEED_EXTRA, 0), (_CROSS_BASE_SEED_EXTRA, 1)]
        ordered = front + [item for item in historical if item not in front]
    else:
        ordered = [(extra, 0) for extra in _TRANSPORT_HEIGHT_EXTRAS]
    cached = _TRANSPORT_SEED_CACHE.get((
        _bearing_sector(start_xy), _bearing_sector(goal_xy),
    ))
    if cached is None:
        return ordered
    return [cached] + [item for item in ordered if item != cached]


def remember_validated_transport_seed(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    extra: float,
    via_slot: int,
    *,
    validated: bool,
) -> None:
    """Remember a combination only after the segment collision check has passed."""
    if not validated:
        return
    _TRANSPORT_SEED_CACHE[(
        _bearing_sector(start_xy), _bearing_sector(goal_xy),
    )] = (float(extra), int(via_slot))


def _alignment_components(vector_xy: np.ndarray, closing_xy: np.ndarray) -> dict[str, float]:
    """Resolve an XY tracking vector along and across the gripper closing axis."""
    vector = np.asarray(vector_xy, dtype=float)
    closing = np.asarray(closing_xy, dtype=float)
    norm = float(np.linalg.norm(closing))
    if norm < 1e-9:
        return {"closing_axis_m": 0.0, "across_axis_m": 0.0}
    closing = closing / norm
    across = np.array((-closing[1], closing[0]), dtype=float)
    return {
        "closing_axis_m": float(vector @ closing),
        "across_axis_m": float(vector @ across),
    }


def _estimate_centering_correction_m(
    aperture_m: float,
    object_width_m: float,
    forces: dict[str, float],
    target_force_n: float,
) -> float:
    """Estimate lateral offset from aperture geometry and force imbalance.

    The model's literal inner-pad gap is approximately controller aperture
    minus 6 mm.  Half of the remaining geometric slack is the offset needed
    to make both pads arrive together.  Force imbalance scales the additional
    closure already spent loading one pad.  The result is always capped at the
    conservative 5 mm centering budget.
    """
    literal_pad_gap = max(
        0.0, float(aperture_m) - GRIPPER_INNER_PAD_OFFSET_M
    )
    geometric_slack = max(0.0, literal_pad_gap - float(object_width_m))
    left = max(0.0, float(forces.get("left", 0.0)))
    right = max(0.0, float(forces.get("right", 0.0)))
    peak = max(left, right)
    imbalance = abs(left - right) / max(left + right, 0.05)
    force_loading = min(0.002, 0.001 * peak / max(float(target_force_n), 0.05))
    estimate = 0.5 * geometric_slack + imbalance * force_loading
    # The 0.25 mm floor is the existing force-search resolution, not a new
    # unmeasured calibration constant.
    return float(np.clip(estimate, 0.00025, GRASP_CENTERING_MAX_STEP_M))


def contact_axis_offset(
    first_contact_aperture_m: dict[str, float | None],
    object_width_m: float,
) -> dict[str, object]:
    """Infer signed closing-axis offset from first-contact apertures.

    Positive means the object was biased toward the left finger; negative
    means toward the right finger.  With only one contact the magnitude is a
    lower bound, because the opposite pad has not yet constrained the object.
    """
    left = first_contact_aperture_m.get("left")
    right = first_contact_aperture_m.get("right")
    if left is not None and right is not None:
        difference = float(left) - float(right)
        if abs(difference) <= 1e-9:
            return {
                "delta_closing_axis_m": 0.0,
                "delta_magnitude_m": 0.0,
                "delta_direction": "simultaneous",
                "lower_bound": False,
            }
        first_side = "left" if difference > 0.0 else "right"
        magnitude = 0.5 * abs(difference)
        return {
            "delta_closing_axis_m": magnitude if first_side == "left" else -magnitude,
            "delta_magnitude_m": magnitude,
            "delta_direction": f"toward_{first_side}_finger",
            "lower_bound": False,
        }
    if left is not None or right is not None:
        first_side = "left" if left is not None else "right"
        aperture = float(left if left is not None else right)
        magnitude = max(0.0, 0.5 * (aperture - float(object_width_m)))
        return {
            "delta_closing_axis_m": magnitude if first_side == "left" else -magnitude,
            "delta_magnitude_m": magnitude,
            "delta_direction": f"toward_{first_side}_finger",
            "lower_bound": True,
        }
    return {
        "delta_closing_axis_m": None,
        "delta_magnitude_m": None,
        "delta_direction": "unknown_no_contact",
        "lower_bound": True,
    }


def recovery_centering_measurement(
    geometry: dict[str, object] | None,
    aperture_m: float,
    object_width_m: float,
    forces: dict[str, float],
    target_force_n: float,
) -> tuple[str, float, str]:
    """Select one A-3 correction source without double-applying estimates."""
    evidence = geometry or {}
    direction = str(evidence.get("delta_direction", ""))
    delta = evidence.get("delta_closing_axis_m")
    if delta is not None and direction in {
        "toward_left_finger", "toward_right_finger",
    }:
        magnitude = float(np.clip(
            abs(float(delta)), 0.00025, GRASP_CENTERING_MAX_STEP_M
        ))
        high_side = "left" if direction == "toward_left_finger" else "right"
        return high_side, magnitude, "contact_geometry_delta"
    high_side = max(forces, key=forces.get)
    estimate = _estimate_centering_correction_m(
        aperture_m, object_width_m, forces, target_force_n
    )
    return high_side, estimate, "force_geometry_fallback"


def _clip_centering_step(step_m: float) -> float:
    """Keep one recovery step inside the existing 0.25–5 mm clamp."""
    magnitude = abs(float(step_m))
    if magnitude < 1e-12:
        return 0.0
    clipped = float(np.clip(
        magnitude, 0.00025, GRASP_CENTERING_MAX_STEP_M
    ))
    return float(np.copysign(clipped, step_m))


def solve_adaptive_centering_step(
    samples: list[tuple[float, float]],
) -> dict[str, object]:
    """Next signed closing-axis correction from this grasp's own samples.

    Each sample is (correction already applied, δ measured after it), in
    metres, with the same sign as ``delta_closing_axis_m``. Two or more
    samples use a secant step aimed at δ = 0. One sample takes the damped
    step. An empty list leaves the force estimator in charge.
    """
    if not samples:
        return {
            "method": "force_geometry_fallback",
            "step_signed_m": None,
            "unclipped_step_signed_m": None,
            "gain": None,
            "expected_residual_m": None,
        }
    _applied, delta = samples[-1]
    if len(samples) == 1:
        raw = CENTERING_DAMPED_GAIN * float(delta)
        return {
            "method": "damped_delta",
            "step_signed_m": _clip_centering_step(raw),
            "unclipped_step_signed_m": float(raw),
            "gain": CENTERING_DAMPED_GAIN,
            "expected_residual_m": None,
        }
    x0, d0 = (float(samples[-2][0]), float(samples[-2][1]))
    x1, d1 = (float(samples[-1][0]), float(samples[-1][1]))
    span = x1 - x0
    slope_denom = d1 - d0
    if abs(span) < 1e-9 or abs(slope_denom) < 1e-9:
        raw = CENTERING_DAMPED_GAIN * d1
        return {
            "method": "damped_delta",
            "step_signed_m": _clip_centering_step(raw),
            "unclipped_step_signed_m": float(raw),
            "gain": CENTERING_DAMPED_GAIN,
            "expected_residual_m": None,
        }
    raw = -d1 * span / slope_denom
    step = _clip_centering_step(raw)
    slope = slope_denom / span
    return {
        "method": "secant_zero",
        "step_signed_m": step,
        "unclipped_step_signed_m": float(raw),
        "gain": None if abs(d1) < 1e-12 else float(step / d1),
        "expected_residual_m": float(d1 + slope * step),
    }


def _inline_centering_ready(
    state: str,
    forces: dict[str, float],
    target_force_n: float,
) -> bool:
    """True once asymmetric contact is material enough to pause closure."""
    if state not in {"left_only", "right_only", "bilateral_unbalanced"}:
        return False
    peak = max(float(forces.get("left", 0.0)), float(forces.get("right", 0.0)))
    return peak >= max(
        GRASP_INLINE_CENTERING_MIN_FORCE_N,
        GRASP_INLINE_CENTERING_FORCE_FRACTION * float(target_force_n),
    )


def classify_grasp_contact(
    contacts: set[str],
    forces: dict[str, float],
    target_force_n: float,
    maximum_force_n: float,
    *,
    shift_m: float = 0.0,
    drop_m: float = 0.0,
    tilt_deg: float = 0.0,
) -> str:
    """Pure form of the unchanged closure success and safety criteria."""
    if shift_m > 0.010 or drop_m > 0.008 or tilt_deg > 8.0:
        return "object_displaced"
    peak = max(float(forces.get("left", 0.0)), float(forces.get("right", 0.0)))
    if peak > float(maximum_force_n) * 1.05:
        return "force_limit"
    if contacts == {"left"}:
        return "left_only"
    if contacts == {"right"}:
        return "right_only"
    if contacts != {"left", "right"}:
        return "no_contact"
    minimum = min(float(forces.get("left", 0.0)), float(forces.get("right", 0.0)))
    ratio = peak / max(minimum, 0.05)
    if minimum >= float(target_force_n) and ratio <= 2.2:
        return "bilateral_balanced"
    if ratio > 2.2:
        return "bilateral_unbalanced"
    return "bilateral_loading"


def grasp_trial_release_decision(
    contacts: set[str],
    forces: Mapping[str, float],
    target_force_n: float,
    maximum_force_n: float,
    *,
    shift_m: float = 0.0,
    drop_m: float = 0.0,
    tilt_deg: float = 0.0,
    closure_disturbed: bool = False,
) -> str:
    """Decide whether closure may proceed to the standard micro-lift.

    Both pads must be in real contact and the strong side must be at or under
    the hard cap. The higher load fraction is not an entry requirement: rise,
    slip, and tilt during the micro-lift are the stability verdict. The
    left/right ratio is intentionally not an input. A closure that already
    dragged the object while contact force was still rising does not lift.
    """
    if (
        float(shift_m) > OBJECT_STABILITY_MAX_SHIFT_M
        or float(drop_m) > 0.008
        or float(tilt_deg) > OBJECT_STABILITY_MAX_TILT_DEG
    ):
        return "object_displaced"
    peak = max(float(forces.get("left", 0.0)), float(forces.get("right", 0.0)))
    if contacts != {"left", "right"}:
        return "bilateral_contact_required"
    minimum = min(float(forces.get("left", 0.0)), float(forces.get("right", 0.0)))
    if peak >= float(maximum_force_n):
        return "strong_force_limit"
    if closure_disturbed and float(shift_m) > GRASP_SERVO_TARGET_M:
        return "closure_disturbed"
    if minimum >= GRASP_CONTACT_PRESENT_N and peak <= float(maximum_force_n):
        return "ready_for_trial_lift"
    return "continue_closing"


def grasp_stability_decision(
    *,
    contacts: set[str],
    forces: Mapping[str, float],
    target_force_n: float,
    maximum_force_n: float,
    object_rise_m: float,
    relative_slip_m: float,
    lateral_shift_m: float,
    tilt_deg: float,
) -> str:
    """Result-based grasp verdict after the standard 5 mm micro-lift.

    Real bilateral contact is enough to attempt the lift. Once the payload
    demonstrably follows the hand, a small force drop is diagnostic rather
    than a reason to reject otherwise persistent, low-slip contact.
    """
    if contacts != {"left", "right"}:
        return "contact_lost"
    peak = max(float(forces.get("left", 0.0)), float(forces.get("right", 0.0)))
    if peak > float(maximum_force_n):
        return "force_limit"
    if float(relative_slip_m) > GRASP_VERIFY_MAX_SLIP_M:
        return "relative_slip"
    if float(lateral_shift_m) > OBJECT_STABILITY_MAX_SHIFT_M:
        return "large_displacement"
    if float(tilt_deg) > OBJECT_STABILITY_MAX_TILT_DEG:
        return "unstable_tilt"
    if float(object_rise_m) < GRASP_VERIFY_MIN_RISE_M:
        return "not_lifted"
    return "stable"


def placement_release_stability_decision(
    *,
    displacement_m: float,
    tilt_deg: float,
    support_contacts: int,
    object_bottom_z_m: float,
    support_z_m: float,
) -> str:
    """Validate the payload while finger force is gradually unloaded.

    The displacement is relative to the first supported pose, rather than the
    support centre, so a valid off-centre placement is not mistaken for motion.
    """
    if float(displacement_m) > OBJECT_STABILITY_MAX_SHIFT_M:
        return "large_displacement"
    if float(tilt_deg) > OBJECT_STABILITY_MAX_TILT_DEG:
        return "unstable_tilt"
    if not (
        float(support_z_m) - 0.008
        <= float(object_bottom_z_m)
        <= float(support_z_m) + 0.025
    ):
        return "unstable_height"
    if int(support_contacts) <= 0:
        return "support_contact_lost"
    return "stable"


def seating_extra_allowance_m() -> float:
    """Seating travel cap. The diagnostic override defaults off.

    ``EXPERIMENT_SEATING_EXTRA_M`` or the same-named environment variable may
    lengthen the cap up to 6.0 mm for comparison runs. Unset uses the committed
    bounded allowance.
    """
    extra = EXPERIMENT_SEATING_EXTRA_M
    raw = os.environ.get("EXPERIMENT_SEATING_EXTRA_M", "").strip()
    if raw:
        extra = float(raw)
    if extra is None:
        return GRASP_SEATING_EXTRA_M
    return float(np.clip(
        float(extra), GRASP_SEATING_EXTRA_M, EXPERIMENT_SEATING_EXTRA_MAX_M,
    ))


def translation_substep_lengths(magnitude_m: float) -> list[float]:
    """Split one translation command into steps of at most 0.1 mm."""
    remaining = float(magnitude_m)
    lengths: list[float] = []
    while remaining > 1e-9:
        step = min(TRANSLATION_FORCE_SUBSTEP_M, remaining)
        lengths.append(step)
        remaining -= step
    return lengths


def translation_force_stop(
    forces_before: dict[str, float],
    forces_now: dict[str, float],
    target_force_n: float,
) -> str | None:
    """Stop later translation substeps. This does not stop closure.

    Either side above the object target force, or the peak more than doubled
    from the pre-translation peak. The hard force cap stays in classify.
    """
    for side in ("left", "right"):
        if float(forces_now.get(side, 0.0)) > float(target_force_n) + 1e-9:
            return "force_above_target"
    peak_before = max(
        float(forces_before.get("left", 0.0)),
        float(forces_before.get("right", 0.0)),
    )
    peak_now = max(
        float(forces_now.get("left", 0.0)),
        float(forces_now.get("right", 0.0)),
    )
    if peak_before > 1e-9 and peak_now > (2.0 * peak_before) + 1e-9:
        return "peak_force_doubled"
    return None


def consume_translation_force_trace(
    force_trace: list[dict[str, float]],
    forces_before: dict[str, float],
    target_force_n: float,
) -> tuple[list[dict[str, float]], str | None, bool]:
    """Apply the translation gate to a force trace.

    The third value is always true: a gate stops translation and leaves the
    close running.
    """
    executed: list[dict[str, float]] = []
    for forces in force_trace:
        executed.append(forces)
        reason = translation_force_stop(forces_before, forces, target_force_n)
        if reason is not None:
            return executed, reason, True
    return executed, None, True


def next_grasp_squeeze_m(
    squeeze_m: float,
    desired_gap_m: float,
    state: str,
    has_contact: bool,
) -> tuple[float | None, float]:
    """Return the next bounded gap and total seating travel.

    ``None`` means closure must stop. No-contact behavior at the measured gap
    is unchanged. The pre-existing 14 mm absolute floor remains beneath the
    bounded seating budget.
    """
    squeeze = float(squeeze_m)
    desired = float(desired_gap_m)
    step = 0.00025 if has_contact else 0.001
    if squeeze > desired + 1e-7:
        return max(desired, squeeze - step), 0.0
    if state not in {
        "left_only", "right_only", "bilateral_unbalanced", "bilateral_loading",
    }:
        return None, max(0.0, desired - squeeze)
    seating_floor = max(0.014, desired - seating_extra_allowance_m())
    next_gap = max(seating_floor, squeeze - 0.00025)
    if next_gap >= squeeze - 1e-7:
        return None, max(0.0, desired - squeeze)
    return next_gap, max(0.0, desired - next_gap)


def grasp_close_step_budget(squeeze_m: float, desired_gap_m: float) -> int:
    """Iterations needed to preserve the full bounded seating allowance.

    Contact closure uses 0.25 mm steps.  Budget for that slowest step over
    the entire remaining travel so a late first contact cannot consume the
    iterations needed to reach the bounded seating floor.
    """
    seating_floor = max(0.014, float(desired_gap_m) - seating_extra_allowance_m())
    return int(np.ceil(
        max(0.0, float(squeeze_m) - seating_floor) / 0.00025
    )) + 2


# Log bound only. Closure control does not read this. A longer trace is
# equally spaced down to this count, always keeping the first contact step,
# the first peak-force step, and the final step.
FORCE_SEQUENCE_MAX_ENTRIES = 24


def thin_force_sequence(
    samples: list[dict[str, object]],
    limit: int = FORCE_SEQUENCE_MAX_ENTRIES,
) -> list[dict[str, object]]:
    """Return an ordered close trace of at most ``limit`` samples.

    Short traces are copied unchanged. Equal spacing fills any remaining
    room after the required anchors are reserved.
    """
    if limit < 1:
        raise ValueError("force sequence limit must be positive")
    if len(samples) <= limit:
        return [dict(sample) for sample in samples]

    def peak_force(sample: dict[str, object]) -> float:
        return max(
            float(sample["left_force_n"]), float(sample["right_force_n"]),
        )

    must = {0, len(samples) - 1}
    for index, sample in enumerate(samples):
        if str(sample["contact_state"]) != "no_contact":
            must.add(index)
            break
    peak_value = max(peak_force(sample) for sample in samples)
    for index, sample in enumerate(samples):
        if peak_force(sample) >= peak_value - 1e-12:
            must.add(index)
            break
    if len(must) > limit:
        raise ValueError("force sequence limit cannot drop the required anchors")
    chosen = set(must)
    slots = limit - len(chosen)
    candidates = [index for index in range(len(samples)) if index not in chosen]
    if slots > 0 and candidates:
        if len(candidates) <= slots:
            chosen.update(candidates)
        else:
            for position in np.linspace(0, len(candidates) - 1, slots):
                chosen.add(candidates[int(round(float(position)))])
            for index in candidates:
                if len(chosen) >= limit:
                    break
                chosen.add(index)
    return [dict(samples[index]) for index in sorted(chosen)[:limit]]


def _place_micro_adjust(
    object_xy: np.ndarray,
    support_xy: np.ndarray,
    bottom_z: float,
    support_z: float,
) -> np.ndarray | None:
    """Hand translation (<= 2 cm) if the pre-release pose would fail verify_place.

    Returns None when no adjustment is needed.  The decision thresholds are
    exactly the verify_place ones: xy error <= 9 cm and bottom within
    [support - 8 mm, support + 25 mm].
    """
    error = np.asarray(support_xy, dtype=float) - np.asarray(object_xy, dtype=float)
    distance = float(np.linalg.norm(error))
    height_ok = support_z - 0.008 <= bottom_z <= support_z + 0.025
    if distance <= 0.09 and height_ok:
        return None
    if distance < 1e-6:
        return np.zeros(2)
    return error / distance * min(distance, PLACE_MICRO_ADJUST_MAX_M)
ARM_JOINTS = tuple(f"joint{index}" for index in range(1, 8))
ROBOT_BODIES = frozenset({
    "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
    "hand", "left_finger", "right_finger",
})
BODY_OF_COLOR = {
    record.primary_color: record.body_name
    for record in OBJECTS
    if record.movable and record.body_name is not None
}
HALF_HEIGHT = {
    record.body_name: record.half_height_m
    for record in OBJECTS
    if record.movable and record.body_name is not None
}
BOX_BODIES = frozenset(
    record.body_name for record in OBJECTS
    if record.movable and record.body_name is not None and record.shape_prior == "box"
    and record.grip_width_key == "box"
)
CUP_BODIES = frozenset(
    record.body_name for record in OBJECTS
    if record.movable and record.body_name is not None
    and record.shape_prior in {"cylinder", "handled_cylinder"}
)
# The box offset keeps the palm mesh above the top face.  At the old 0.144 m
# hand height the palm penetrated the blue box's top edge by about 2 mm.
# The cup is 10 cm tall.  Its old 65 mm palm-over-top clearance put the hand
# collision mesh through the rim before the finger pads reached the sides.
# 105 - 20 = 85 mm keeps the palm clear while the long pads still cover the
# upper side wall.  This is geometry, not an LLM-selected low-level number.
HAND_OFFSET = {
    record.display_name: record.hand_offset_m
    for record in OBJECTS if record.movable
}
FALLBACK_WIDTH = {
    record.display_name: record.fallback_width_m
    for record in OBJECTS if record.movable
}
MAX_GRASP_WIDTH = 0.078  # keep 2 mm margin below Panda's 80 mm hard limit
# Move the long finger pads down around the object's side center.  The hand
# origin is ~75 mm above the middle of the useful pad surface; a shallow value
# only catches a tall box at its top edge and it slips out during lift.
# Bite near the vertical middle.  A cup grabbed on its upper half is cammed
# out of a round contact as soon as the fingers tighten.
GRASP_DEPTH = {
    record.display_name: record.grasp_depth_m
    for record in OBJECTS if record.movable
}
# Measured pad force in this model saturates near 0.8 N before the safe
# squeeze limit.  Asking for 2 N only made the fingers keep closing.
GRIP_FORCE_TARGET_N = {
    record.grip_width_key: record.target_force_n
    for record in OBJECTS if record.movable
}
MAX_GRIP_FORCE_N = {
    record.grip_width_key: record.max_force_n
    for record in OBJECTS if record.movable
}


@dataclass(frozen=True)
class GraspCandidate:
    """Metric grasp produced locally from a semantic region and fresh RGB-D."""

    region: str
    height_ratio: float
    yaw_rad: float
    width_m: float
    score: float
    source: str


@dataclass(frozen=True)
class ClosureResult:
    contacts: set[str]
    gap_m: float
    forces: dict[str, float]
    state: str
    ready_for_trial_lift: bool = False
    trial_release_decision: str = "not_evaluated"
    yaw_rad: float | None = None
    contact_geometry: dict[str, object] | None = None


@dataclass(frozen=True)
class TransportPlanCandidate:
    candidate_id: str
    joint_waypoints: tuple[tuple[float, ...], ...]
    execution_points: tuple[tuple[float, ...], ...]
    hand_waypoints: tuple[tuple[float, ...], ...]
    scores: dict[str, float]
    planning_time_s: float
    validated: bool = True
    details: dict[str, object] = field(default_factory=dict)

    def execution_arrays(self) -> list[np.ndarray]:
        return [np.asarray(point, dtype=float) for point in self.execution_points]


CONTACT_STATES = frozenset({
    "no_contact", "left_only", "right_only", "bilateral_balanced",
    "bilateral_loading", "bilateral_unbalanced", "force_limit", "object_displaced",
})


PRELOAD_SLIP_DEADBAND_M = 0.001


def stability_preload_request(
    *,
    slip_m: float,
    slip_reference_m: float,
    peak_force_n: float,
    force_ceiling_n: float,
    bilateral: bool,
) -> str:
    """Hold a stable grasp. Add force only when slip is rising and the ceiling is free.

    The force ceiling and the slip abort limit are not inputs that this
    function can change. A weak pad under the force target is not itself a
    reason to keep closing.
    """
    if not bilateral:
        return "hold"
    if float(peak_force_n) >= float(force_ceiling_n):
        return "hold_at_limit"
    if float(slip_m) > float(slip_reference_m) + PRELOAD_SLIP_DEADBAND_M:
        return "preload"
    return "hold"


def strong_limit_transport_decision(
    *,
    slip_m: float,
    slip_before_m: float,
    tilt_deg: float,
    tilt_limit_deg: float,
    bilateral: bool,
) -> str:
    """The ceiling stops the close. It is a failure only when the grasp got worse.

    ``hold_at_limit`` keeps the current aperture and lets transport continue.
    ``abort_unstable`` is the regrasp stop: slip rose during the close, the
    pose left the existing tilt limit, or both pads are no longer in contact.
    """
    pose_ok = float(tilt_deg) <= float(tilt_limit_deg) + 1e-9
    slip_rose = float(slip_m) > float(slip_before_m) + PRELOAD_SLIP_DEADBAND_M
    if bilateral and pose_ok and not slip_rose:
        return "hold_at_limit"
    return "abort_unstable"


def next_transport_stability_action(
    recovery_stage: str,
    signal_present: bool,
    later_segment: bool,
) -> str | None:
    """P-3 ordering: a weakening signal slows first, then permits preload."""
    if recovery_stage == "hold_at_limit":
        return None
    if not signal_present:
        return None
    if recovery_stage == "monitoring":
        return "slow_down"
    if recovery_stage == "slowed" and later_segment:
        return "preload"
    return None


@dataclass
class TransportMonitor:
    """Fast local grasp validator used while a payload trajectory executes.

    The relative pose is measured in the hand frame, so moving the arm in the
    world does not look like object motion.  The monitor runs every MuJoCo
    step and can stop the controller without waiting for an LLM call.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    body_name: str
    baseline_local_xyz: np.ndarray
    baseline_up_local: np.ndarray
    motion_profile: str = "payload"
    motion_scale: float = 1.0
    maximum_slip_m: float = 0.010
    maximum_tilt_deg: float = 12.0
    contact_grace_s: float = 0.080
    no_contact_s: float = 0.0
    max_contact_gap_seen_s: float = 0.0
    max_slip_seen_m: float = 0.0
    max_tilt_seen_deg: float = 0.0
    minimum_force_seen_n: float = float("inf")
    initial_gripper_ctrl: float = 0.0
    minimum_gripper_ctrl: float = 0.0
    adaptive_preload_seen_mm: float = 0.0
    runtime_collision_checks: int = 0
    samples: int = 0
    segment_index: int = -1
    recovery_stage: str = "monitoring"
    recovery_signal_segment: int = -1
    preload_requested: bool = False
    stability_events: list[dict[str, object]] = field(default_factory=list)
    clearance_trace: dict[str, object] = field(default_factory=lambda: {
        "live_min_robot_clearance_m": None,
        "live_min_robot_pair": None,
        "live_min_payload_clearance_m": None,
        "live_min_payload_pair": None,
        "max_joint_tracking_error_rad": 0.0,
    })
    preload_slip_reference_m: float = 0.0
    force_hold: bool = False

    @classmethod
    def capture(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_name: str,
        motion_profile: str = "payload",
        motion_scale: float = 1.0,
    ) -> "TransportMonitor":
        hand_id = model.body("hand").id
        rotation = data.xmat[hand_id].reshape(3, 3).copy()
        hand_xyz = data.xpos[hand_id].copy()
        object_xyz = data.xpos[model.body(body_name).id].copy()
        object_up = _body_up_axis(model, data, body_name)
        initial_ctrl = float(data.ctrl[-1])
        return cls(
            model=model,
            data=data,
            body_name=body_name,
            baseline_local_xyz=rotation.T @ (object_xyz - hand_xyz),
            baseline_up_local=rotation.T @ object_up,
            motion_profile=motion_profile,
            motion_scale=motion_scale,
            # A 5 cm cube can rotate slightly inside two flat pads without
            # losing support.  Permit recovery up to 15 degrees; placement
            # still requires the absolute upright check below 4 degrees.
            maximum_tilt_deg=15.0 if body_name == "cube" else 12.0,
            initial_gripper_ctrl=initial_ctrl,
            # At most 4 mm of additional aperture preload during transit.
            minimum_gripper_ctrl=max(
                _ctrl_for_gap(0.014), initial_ctrl - 4.0 * 255.0 / 80.0
            ),
        )

    def check(self) -> str | None:
        hand_id = self.model.body("hand").id
        rotation = self.data.xmat[hand_id].reshape(3, 3)
        hand_xyz = self.data.xpos[hand_id]
        object_xyz = self.data.xpos[self.model.body(self.body_name).id]
        local_xyz = rotation.T @ (object_xyz - hand_xyz)
        slip = float(np.linalg.norm(local_xyz - self.baseline_local_xyz))
        up_local = rotation.T @ _body_up_axis(self.model, self.data, self.body_name)
        cosine = float(np.clip(up_local @ self.baseline_up_local, -1.0, 1.0))
        tilt = float(np.rad2deg(np.arccos(cosine)))
        contacts, forces = _contact_metrics(self.model, self.data, self.body_name)
        minimum_force = min(forces.values()) if contacts == {"left", "right"} else 0.0
        if self.body_name in BOX_BODIES:
            force_kind = "box"
        elif self.body_name == "cube":
            force_kind = "cube"
        elif "bottle" in self.body_name:
            force_kind = "bottle"
        else:
            force_kind = "cup"
        force_ceiling = MAX_GRIP_FORCE_N[force_kind]
        peak_force = max(forces.values())
        if contacts == {"left", "right"}:
            self.no_contact_s = 0.0
            self.minimum_force_seen_n = min(self.minimum_force_seen_n, minimum_force)
        else:
            self.no_contact_s += float(self.model.opt.timestep)
            self.max_contact_gap_seen_s = max(
                self.max_contact_gap_seen_s, self.no_contact_s
            )
        slip_decision = stability_preload_request(
            slip_m=slip,
            slip_reference_m=self.preload_slip_reference_m,
            peak_force_n=peak_force,
            force_ceiling_n=force_ceiling,
            bilateral=contacts == {"left", "right"},
        )
        if (
            not self.force_hold
            and slip_decision == "hold"
            and slip + 1e-12 < self.preload_slip_reference_m
        ):
            self.preload_slip_reference_m = slip
        if slip_decision == "hold_at_limit" and not self.force_hold:
            self.force_hold = True
            self.preload_slip_reference_m = slip
            self.recovery_stage = "hold_at_limit"
            if self.motion_scale > 0.63 + 1e-9:
                self.motion_scale *= 0.63
            self.stability_events.append({
                "action": "hold_at_limit",
                "segment_index": self.segment_index,
                "peak_force_n": peak_force,
                "relative_slip_m": slip,
            })
        recovery_signal = bool(
            slip_decision == "preload"
            and not self.force_hold
            and peak_force < force_ceiling
            and slip < 0.012
            and tilt < self.maximum_tilt_deg
            and float(self.data.ctrl[-1]) > self.minimum_gripper_ctrl
        )
        action = next_transport_stability_action(
            self.recovery_stage,
            recovery_signal,
            self.segment_index > self.recovery_signal_segment,
        )
        if action == "slow_down":
            previous_scale = self.motion_scale
            self.motion_scale *= 0.63
            self.recovery_stage = "slowed"
            self.recovery_signal_segment = self.segment_index
            self.stability_events.append({
                "action": "slow_down",
                "segment_index": self.segment_index,
                "motion_scale_before": previous_scale,
                "motion_scale_after": self.motion_scale,
                "weak_force_n": minimum_force,
                "relative_slip_m": slip,
            })
        elif action == "preload":
            self.preload_requested = True
            self.recovery_stage = "preload_requested"
            self.stability_events.append({
                "action": "preload_requested",
                "segment_index": self.segment_index,
                "weak_force_n": minimum_force,
                "relative_slip_m": slip,
            })
        self.samples += 1
        self.max_slip_seen_m = max(self.max_slip_seen_m, slip)
        self.max_tilt_seen_deg = max(self.max_tilt_seen_deg, tilt)
        bottom = _body_bottom_z(self.model, self.data, self.body_name)
        if bottom < 0.010:
            return "object_dropped_to_floor"
        if slip > self.maximum_slip_m:
            return f"object_slip:{slip:.4f}m"
        if tilt > self.maximum_tilt_deg:
            return f"object_tilt:{tilt:.1f}deg"
        # MuJoCo's discrete contact set can report only one side when a pad is
        # touching an edge of a stable, position-controlled grasp.  Do not
        # confuse that contact-classification chatter with a drop: relative
        # displacement and tilt are the primary slip signals.  If support is
        # genuinely lost, gravity moves the payload past these pose limits in
        # a few simulation steps; once that happens the original 80 ms contact
        # timer becomes an additional fail-fast condition.
        contact_loss_confirmed = slip > 0.008 or tilt > 10.0
        if contact_loss_confirmed and self.no_contact_s > self.contact_grace_s:
            return f"grasp_contact_lost:{self.no_contact_s:.3f}s"
        return None

    def begin_segment(self) -> None:
        self.segment_index += 1

    def consume_preload_request(self) -> bool:
        requested = self.preload_requested
        self.preload_requested = False
        return requested

    def relative_slip_and_tilt(self) -> tuple[float, float]:
        """Current hand-frame slip and tilt. Reading them does not change the abort."""
        hand_id = self.model.body("hand").id
        rotation = self.data.xmat[hand_id].reshape(3, 3)
        hand_xyz = self.data.xpos[hand_id]
        object_xyz = self.data.xpos[self.model.body(self.body_name).id]
        local_xyz = rotation.T @ (object_xyz - hand_xyz)
        slip = float(np.linalg.norm(local_xyz - self.baseline_local_xyz))
        up_local = rotation.T @ _body_up_axis(self.model, self.data, self.body_name)
        cosine = float(np.clip(up_local @ self.baseline_up_local, -1.0, 1.0))
        tilt = float(np.rad2deg(np.arccos(cosine)))
        return slip, tilt

    def rebase_relative_pose(self) -> None:
        """Accept a deliberately validated new load pose after local correction."""
        hand_id = self.model.body("hand").id
        rotation = self.data.xmat[hand_id].reshape(3, 3).copy()
        hand_xyz = self.data.xpos[hand_id].copy()
        object_xyz = self.data.xpos[self.model.body(self.body_name).id].copy()
        self.baseline_local_xyz = rotation.T @ (object_xyz - hand_xyz)
        self.baseline_up_local = rotation.T @ _body_up_axis(
            self.model, self.data, self.body_name
        )
        self.no_contact_s = 0.0

    def metrics(self) -> dict[str, object]:
        force = None if not np.isfinite(self.minimum_force_seen_n) else self.minimum_force_seen_n
        return {
            "samples": self.samples,
            "max_relative_slip_m": self.max_slip_seen_m,
            "max_relative_tilt_deg": self.max_tilt_seen_deg,
            "minimum_bilateral_normal_force_n": force,
            "maximum_contact_gap_s": self.max_contact_gap_seen_s,
            "adaptive_preload_mm": self.adaptive_preload_seen_mm,
            "runtime_collision_checks": self.runtime_collision_checks,
            "motion_profile": self.motion_profile,
            "motion_scale": self.motion_scale,
            "stability_events": list(self.stability_events),
        }

    def note_execution_clearance(
        self,
        parts: dict[str, object],
        tracking_error_rad: float,
    ) -> None:
        """Record live gaps and joint tracking. This does not change an abort."""
        trace = self.clearance_trace
        robot_gap = float(parts["robot_scene_gap_m"])
        payload_gap = float(parts["payload_scene_gap_m"])
        robot_best = trace.get("live_min_robot_clearance_m")
        if robot_best is None or robot_gap < float(robot_best):
            trace["live_min_robot_clearance_m"] = robot_gap
            trace["live_min_robot_pair"] = parts.get("robot_scene_pair")
        payload_best = trace.get("live_min_payload_clearance_m")
        if payload_best is None or payload_gap < float(payload_best):
            trace["live_min_payload_clearance_m"] = payload_gap
            trace["live_min_payload_pair"] = parts.get("payload_scene_pair")
        trace["max_joint_tracking_error_rad"] = max(
            float(trace.get("max_joint_tracking_error_rad") or 0.0),
            float(tracking_error_rad),
        )


def _payload_motion_profile(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    forces: dict[str, float],
    requested_mode: str,
) -> tuple[str, dict[str, float]]:
    """Choose a local dynamic envelope from measured grasp stability.

    This is a conservative predictor, not a claim that Coulomb friction is
    known exactly.  It uses an effective friction coefficient below the scene
    value and includes gravity before allowing the normal payload profile.
    """
    body_id = model.body(body_name).id
    mass = float(model.body_subtreemass[body_id])
    bilateral = max(0.0, min(float(forces.get("left", 0.0)), float(forces.get("right", 0.0))))
    effective_mu = 0.45
    available_friction = 2.0 * bilateral * effective_mu
    required_static = mass * 9.81
    hand = data.xpos[model.body("hand").id]
    obj = data.xpos[body_id]
    grasp_offset = float(np.linalg.norm((obj - hand)[:2]))
    # Reserve capacity for lateral acceleration and an off-centre rotational
    # disturbance.  The numerical caps remain controller-owned.
    required_dynamic = mass * 0.55 + 4.0 * mass * grasp_offset
    margin_n = available_friction - required_static - required_dynamic
    margin_ratio = margin_n / max(available_friction, 1e-6)
    fragile = (
        requested_mode == "fragile"
        or margin_ratio < 0.35
        or grasp_offset > 0.018
        or bilateral <= 0.0
    )
    # 0.63 is the strictest ratio already established by the two existing
    # payload envelopes: 0.35 / 0.55 acceleration = 0.636. Applying 0.63 to
    # the normal payload envelope is slightly tighter than the historical
    # payload_fragile limit without inventing a second independent envelope.
    motion_scale = 0.63 if fragile else 1.0
    profile = "payload"
    return profile, {
        "mass_kg": mass,
        "bilateral_force_n": bilateral,
        "grasp_offset_m": grasp_offset,
        "available_friction_n": available_friction,
        "required_force_n": required_static + required_dynamic,
        "margin_ratio": margin_ratio,
        "motion_scale": motion_scale,
        "effective_profile": "payload_fragile" if fragile else "payload",
    }


def _arm_q(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return np.array([
        data.qpos[int(model.joint(name).qposadr[0])] for name in ARM_JOINTS
    ], dtype=float)


def _hand_xyz(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return data.xpos[model.body("hand").id].copy()


def _hand_yaw(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Yaw of the current finger-closing axis, matching ``solve_hand_ik``."""
    closing = data.xmat[model.body("hand").id].reshape(3, 3)[:, 1]
    return float(np.arctan2(-closing[0], -closing[1]))


def _same_grasp_yaw(desired: float, current: float) -> float:
    """Same finger axis as ``desired``, the representative nearest ``current``.

    Flipping the fingers by 180 degrees is the same pinch.  Picking that
    representative stops the wrist from unwinding almost a full turn when the
    arm arrives from another side of the table.
    """
    best_delta = float("inf")
    chosen = 0.0
    for turn in range(-2, 3):
        candidate = float(desired) + turn * np.pi
        delta = (candidate - current + np.pi) % (2.0 * np.pi) - np.pi
        if abs(delta) < best_delta:
            best_delta = abs(delta)
            chosen = float(delta)
    return float(current + chosen)


def _body_xyz(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    return data.xpos[model.body(body_name).id].copy()


def _body_bottom_z(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> float:
    """Lowest point of a body's collision geoms, including a tipped box."""
    body_id = model.body(body_name).id
    bottoms = [
        float(data.geom_xpos[geom_id, 2]) - _geom_vertical_radius(model, data, geom_id)
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id and model.geom_contype[geom_id]
    ]
    return min(bottoms) if bottoms else float(data.xpos[body_id, 2])


def _holding_gap(width_m: float) -> float:
    """Controller gap for a slightly firm pad squeeze.

    Panda's finger ``qpos`` measures joint travel, not the literal inner pad
    gap.  In this model the inner gap is approximately ``2*qpos - 6 mm``, so
    contact starts near controller gap ``w + 6 mm``.  Commanding ``w - 1 mm``
    preloads the finger spring by about 7 mm and keeps the object seated.
    """
    return float(np.clip(float(width_m) - 0.001, 0.0, 0.08))


def _ctrl_for_gap(gap_m: float) -> float:
    # ctrl 0..255 maps linearly onto finger joint 0..0.04 m, and the inner
    # pad gap is twice that joint position.
    return float(np.clip(gap_m / 0.08 * 255.0, 0.0, GRIPPER_OPEN))


def _snap_grasp_yaw(yaw: float) -> float:
    """Snap near-cardinal faces so IK does not chase a few degrees of PCA noise."""
    snap = np.deg2rad(10.0)
    if abs(yaw) < snap:
        return 0.0
    if abs(abs(yaw) - np.pi / 2.0) < snap:
        return float(np.sign(yaw) * np.pi / 2.0) if yaw != 0.0 else 0.0
    return float(yaw)


def _grasp_face_options(
    span: dict[str, float] | None,
    fallback: float,
    current_yaw: float | None = None,
) -> list[tuple[float, float]]:
    """Every gripper-fitting face, narrowest and least wrist turn first."""
    if span is None:
        if fallback <= MAX_GRASP_WIDTH:
            return [(0.0, float(fallback))]
        return []
    raw = [
        (float(span.get("minor_m", 0.0)), float(span.get("minor_yaw_rad", 0.0))),
        (float(span.get("major_m", 0.0)), float(span.get("major_yaw_rad", 0.0))),
        (float(span.get("extent_y", 0.0)), 0.0),
        (float(span.get("extent_x", 0.0)), float(np.pi / 2.0)),
    ]
    valid = [(width, yaw) for width, yaw in raw if 0.025 <= width <= MAX_GRASP_WIDTH]
    if not valid:
        return []

    def _turn(yaw: float) -> float:
        if current_yaw is None:
            return 0.0
        aligned = _same_grasp_yaw(yaw, current_yaw)
        return abs((aligned - current_yaw + np.pi) % (2.0 * np.pi) - np.pi)

    ordered = sorted(
        valid,
        key=lambda item: (_turn(item[1]) > 0.9, item[0], _turn(item[1])),
    )
    snapped: list[tuple[float, float]] = []
    seen: list[float] = []
    for width, yaw in ordered:
        yaw = _snap_grasp_yaw(yaw)
        if any(grasp_axis_distance_rad(yaw, previous) <= _AXIS_BLOCK_M for previous in seen):
            continue
        seen.append(yaw)
        snapped.append((yaw, width))
    return snapped


def _plan_grasp_direction(
    span: dict[str, float] | None,
    fallback: float,
    current_yaw: float | None = None,
) -> tuple[float, float] | None:
    """Choose an oriented face that physically fits the 80 mm gripper.

    PCA from the latest RGB-D mask handles a box that rotated or fell over.
    If neither measured face fits, return None instead of forcing a bad grasp.
    Among similarly reachable faces, prefer the narrower one: it leaves more
    finger travel for preload and reduces the torque arm during transport.
    A large wrist unwind is still rejected before comparing widths.
    """
    options = _grasp_face_options(span, fallback, current_yaw)
    if not options:
        return None
    yaw, width = options[0]
    return yaw, width


def _semantic_height_ratio(region: str, object_name: str) -> float:
    """Map an LLM semantic region to a bounded local geometry target."""
    if object_name in {"cup", "mug", "bottle"}:
        return {
            "lower_side": 0.36,
            "middle_side": 0.50,
            "upper_side": 0.68,
        }.get(region, 0.50)
    return {
        "lower_side": 0.42,
        "middle_side": 0.52,
        "upper_side": 0.62,
    }.get(region, 0.52)


def stabilize_handle_direction(
    history_rad: list[float],
    new_direction_rad: float,
) -> tuple[float, bool]:
    """Use the recent-three angular median after a >10° adjacent jump."""
    current = float(new_direction_rad)
    previous = None if not history_rad else float(history_rad[-1])
    history_rad.append(current)
    if previous is None:
        return current, False
    adjacent = (current - previous + np.pi) % (2.0 * np.pi) - np.pi
    if abs(adjacent) <= np.deg2rad(10.0):
        return current, False
    recent = np.unwrap(np.asarray(history_rad[-3:], dtype=float))
    median = float(np.median(recent))
    stabilized = float((median + np.pi) % (2.0 * np.pi) - np.pi)
    return stabilized, True


def handle_avoid_yaw_from_direction(handle_direction_rad: float) -> float:
    """Convert handle direction to the existing bidirectional closing yaw."""
    raw = -float(handle_direction_rad)
    return float((raw + np.pi / 2.0) % np.pi - np.pi / 2.0)


def grasp_reseat_drop_m(finger_z: float, support_z: float) -> float | None:
    """One downward step when both pads already touch and seating is spent.

    Returns ``None`` when the pads are already within the existing 8 mm
    support clearance. This does not add closure and does not lower the
    trial-lift force floor.
    """
    floor = float(support_z) + 0.008
    room = float(finger_z) - floor
    if room < GRASP_RESEAT_MIN_STEP_M:
        return None
    return min(GRASP_RESEAT_STEP_M, room)


def approach_side_clears_protrusion(
    candidate_xy: np.ndarray,
    object_xy: np.ndarray,
    protrusion_rad: float | None,
) -> bool:
    """True when a side waypoint is not standing on a measured protrusion.

    ``None`` means no protrusion was measured, so every side stays eligible.
    The angle is between the incoming waypoint and the protrusion ray.
    """
    if protrusion_rad is None:
        return True
    travel = np.asarray(candidate_xy, dtype=float)[:2] - np.asarray(object_xy, dtype=float)[:2]
    norm = float(np.linalg.norm(travel))
    if norm < 1e-6:
        return False
    protrusion = np.array((
        np.cos(float(protrusion_rad)),
        np.sin(float(protrusion_rad)),
    ))
    cosine = float(np.clip((travel / norm) @ protrusion, -1.0, 1.0))
    return float(np.arccos(cosine)) >= PROTRUSION_APPROACH_CLEARANCE_RAD


def grasp_axis_distance_rad(first: float, second: float) -> float:
    """Shortest distance between parallel-jaw axes (period pi)."""
    return abs(float((float(first) - float(second) + np.pi / 2.0) % np.pi - np.pi / 2.0))


# A fresh RGB-D pose this far from the attempt that failed is a new observation.
GRASP_OBSERVATION_CHANGE_M = 0.02
_AXIS_BLOCK_M = np.deg2rad(10.0)


def grasp_failure_dof(failure_type: str) -> str:
    """Which search degree of freedom a measured failure should change.

    The names are failure types, not object names. ``no_contact`` refreshes
    the measurement and does not retire every yaw.
    """
    if failure_type in {"left_only", "right_only"}:
        return "xy_center"
    if failure_type in {"unstable_tilt", "object_moved", "large_displacement"}:
        return "grasp_axis"
    if failure_type in {
        "relative_slip", "slipped", "slip_after_lift", "lift_failed",
    }:
        return "region_yaw_depth"
    if failure_type == "no_contact":
        return "center_size_depth"
    if failure_type == "weak_force_insufficient":
        return "region_yaw_depth"
    if failure_type in {"approach_blocked", "collision", "approach_collision"}:
        return "approach"
    return "observe"


def grasp_candidate_id(yaw_rad: float, region: str, approach: str) -> str:
    """Stable id for one jaw axis, region, and approach. A half-turn is the same id."""
    folded = (float(yaw_rad) + np.pi / 2.0) % np.pi - np.pi / 2.0
    bucket = round(float(np.rad2deg(folded)) / 5.0) * 5.0
    if abs(bucket) == 90.0:
        bucket = -90.0
    return f"yaw{bucket:.0f}|{region}|{approach}"


def failure_type_for_search(diagnosis: dict[str, object]) -> str:
    """Map a diagnosis to the search degree of freedom.

    A two-sided force ratio stays ``bilateral_unbalanced`` and does not retire
    the yaw.  A one-sided overload is the same search change as left/right only.
    """
    failure = diagnosis.get("failure")
    failure_type = str(failure.get("type", "")) if isinstance(failure, dict) else ""
    if failure_type != "bilateral_unbalanced":
        return failure_type
    observation = diagnosis.get("observation")
    if not isinstance(observation, dict):
        return failure_type
    left = float(observation.get("left_normal_force_n") or 0.0)
    right = float(observation.get("right_normal_force_n") or 0.0)
    if min(left, right) > 0.05 or max(left, right) <= 0.05:
        return failure_type
    return "left_only" if left >= right else "right_only"


def grasp_failure_status(failure_type: str) -> str | None:
    """Status written into this observation, or None when yaw stays eligible."""
    change = grasp_failure_dof(failure_type)
    if change == "xy_center":
        return "failed_contact"
    if change == "grasp_axis":
        return "failed_axis"
    if change == "region_yaw_depth":
        return "failed_lift"
    if change == "approach":
        return "failed_approach"
    return None


def remember_grasp_failure(
    entries: list[dict[str, object]],
    *,
    yaw_rad: float,
    region: str,
    approach: str,
    failure_type: str,
    attempt: int,
) -> list[dict[str, object]]:
    """Append one failure. ``no_contact`` does not retire the yaw."""
    status = grasp_failure_status(failure_type)
    if status is None:
        return list(entries)
    updated = list(entries)
    updated.append({
        "candidate_id": grasp_candidate_id(yaw_rad, region, approach),
        "yaw_rad": float(yaw_rad),
        "region": region,
        "approach": approach,
        "status": status,
        "failure_type": failure_type,
        "attempt": int(attempt),
    })
    return updated


def observation_pose_changed(
    previous_xy: np.ndarray | None,
    current_xy: np.ndarray | None,
    threshold_m: float = GRASP_OBSERVATION_CHANGE_M,
) -> bool:
    if previous_xy is None or current_xy is None:
        return False
    delta = np.asarray(current_xy, dtype=float)[:2] - np.asarray(previous_xy, dtype=float)[:2]
    return float(np.linalg.norm(delta)) > float(threshold_m)


def refresh_grasp_attempt_memory(
    entries: list[dict[str, object]],
    previous_xy: np.ndarray | None,
    current_xy: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Drop this round's failures after the object visibly moves."""
    current = np.asarray(current_xy, dtype=float)[:2].copy()
    if observation_pose_changed(previous_xy, current):
        return [], current
    anchor = current if previous_xy is None else np.asarray(previous_xy, dtype=float)[:2].copy()
    return list(entries), anchor


def _failure_matches_candidate(
    entry: dict[str, object],
    yaw_rad: float,
    region: str,
    approach: str,
) -> bool:
    status = str(entry.get("status") or "")
    same_axis = (
        grasp_axis_distance_rad(yaw_rad, float(entry["yaw_rad"])) <= _AXIS_BLOCK_M
    )
    if status == "failed_approach":
        return same_axis and str(entry.get("approach")) == approach
    if status == "failed_lift":
        return same_axis and str(entry.get("region") or region) == region
    if status in {"failed_axis", "failed_contact"}:
        return same_axis
    return False


def grasp_candidate_status(
    yaw_rad: float,
    region: str,
    approach: str,
    entries: list[dict[str, object]],
) -> str:
    """untried, or the strongest failure recorded for this candidate."""
    rank = {
        "untried": 0,
        "failed_contact": 1,
        "failed_approach": 1,
        "failed_lift": 2,
        "failed_axis": 2,
    }
    best = "untried"
    for entry in entries:
        if not _failure_matches_candidate(entry, yaw_rad, region, approach):
            continue
        status = str(entry.get("status") or "untried")
        if rank.get(status, 0) > rank.get(best, 0):
            best = status
    return best


def order_grasp_candidates(
    candidates: list[GraspCandidate],
    entries: list[dict[str, object]],
    approach: str,
) -> list[GraspCandidate]:
    """Prefer an untried axis. A failed axis stays available after the others."""
    rank = {
        "untried": 0,
        "failed_contact": 1,
        "failed_approach": 1,
        "failed_lift": 2,
        "failed_axis": 2,
    }

    def sort_key(candidate: GraspCandidate) -> tuple[int, int, float]:
        status = grasp_candidate_status(
            candidate.yaw_rad, candidate.region, approach, entries,
        )
        recent = max(
            (
                int(entry["attempt"])
                for entry in entries
                if _failure_matches_candidate(
                    entry, candidate.yaw_rad, candidate.region, approach,
                )
            ),
            default=-1,
        )
        return (rank[status], recent, -float(candidate.score))

    return sorted(candidates, key=sort_key)


def _generate_grasp_candidates(
    body_name: str,
    object_name: str,
    span: dict[str, float] | None,
    fallback_width: float,
    current_yaw: float,
    semantic_region: str,
    yaw_policy: str,
) -> list[GraspCandidate]:
    """Generate and rank grasp poses locally; the LLM never supplies metres.

    Axisymmetric bodies deliberately ignore PCA yaw.  When a protruding handle
    is visible, its direction is used only to keep fingertips away from it.
    """
    region = semantic_region if semantic_region != "auto" else (
        "middle_side" if body_name in CUP_BODIES else "middle_side"
    )
    ratio = _semantic_height_ratio(region, object_name)
    candidates: list[GraspCandidate] = []
    if body_name in CUP_BODIES:
        width = float(np.clip(fallback_width, 0.025, MAX_GRASP_WIDTH))
        handle_visible = bool(span and span.get("has_handle", 0.0) > 0.5)
        if yaw_policy == "avoid_handle" and handle_visible:
            raw_yaw = float(span.get("handle_avoid_yaw_rad", current_yaw))
            yaw = _same_grasp_yaw(raw_yaw, current_yaw)
            source = "RGB-D handle direction"
            score = 1.0 - 0.15 * abs(yaw - current_yaw) / np.pi
        else:
            # A cylinder has no observable body yaw.  Preserve the current
            # equivalent wrist branch instead of chasing PCA noise.
            yaw = float(current_yaw)
            source = "axisymmetric preserve-wrist"
            score = 0.95
        candidates.append(GraspCandidate(region, ratio, yaw, width, score, source))
        # A circular body admits a continuum of equivalent closing axes.  Four
        # deterministic samples cover the half-turn without attaching special
        # cases to a named cup.  The exact endpoint preflight below can then
        # choose a collision-free wrist/link pose near clutter while retaining
        # the handle-aware direction as the preferred candidate.
        for offset, penalty, label in (
            (np.pi / 2.0, 0.25, "orthogonal local fallback"),
            (np.pi / 4.0, 0.32, "+45deg local fallback"),
            (-np.pi / 4.0, 0.34, "-45deg local fallback"),
        ):
            alternate = _same_grasp_yaw(yaw + offset, current_yaw)
            candidates.append(GraspCandidate(
                region, ratio, alternate, width, score - penalty, label,
            ))
    else:
        for index, (yaw, width) in enumerate(
            _grasp_face_options(span, fallback_width, current_yaw)
        ):
            aligned = _same_grasp_yaw(yaw, current_yaw)
            candidates.append(GraspCandidate(
                region,
                ratio,
                aligned,
                float(width),
                (
                    1.0 - 0.10 * abs(aligned - current_yaw)
                    if index == 0
                    else 0.70 - 0.08 * index
                ),
                "RGB-D face geometry" if index == 0 else "alternate face",
            ))
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _is_free_movable_body(model: mujoco.MjModel, body_name: str) -> bool:
    """Whether a fingertip/body contact can be an intentional grasp contact."""
    if not body_name:
        return False
    try:
        body_id = int(model.body(body_name).id)
    except (KeyError, ValueError):
        return False
    if int(model.body_jntnum[body_id]) < 1:
        return False
    joint_id = int(model.body_jntadr[body_id])
    return joint_id >= 0 and int(model.jnt_type[joint_id]) == int(
        mujoco.mjtJoint.mjJNT_FREE
    )


def _scene_hit(
    model: mujoco.MjModel,
    scratch: mujoco.MjData,
    ignore_bodies: set[str],
    payload_body: str | None = None,
    contact_distance_threshold_m: float = -0.0002,
) -> str | None:
    """A robot link penetrating the scene, or None when this pose is clear."""
    for index in range(scratch.ncon):
        contact = scratch.contact[index]
        # Planning should reject a scene contact as soon as penetration starts.
        # The former 3 mm tolerance was large enough for a finger or carried
        # cube to visibly push the orange cube before the route was rejected.
        if contact.dist > contact_distance_threshold_m:
            continue
        body1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom1])
        ) or ""
        body2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom2])
        ) or ""
        pair = {body1, body2}
        if payload_body is not None and payload_body in pair and not (pair & ROBOT_BODIES):
            other = next(iter(pair - {payload_body}), "")
            if other and other != "world":
                return f"payload:{payload_body}->{other}"
            if other == "world":
                geom1 = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
                ) or ""
                geom2 = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
                ) or ""
                world_geom = geom1 if body1 == "world" else geom2
                if world_geom != "floor":
                    return f"payload:{payload_body}->{world_geom or 'world'}"
            continue
        if pair <= ROBOT_BODIES:
            # Adjacent links rest against each other.  A hit only counts when
            # the gripper or wrist has gone through the upper arm or the base.
            distal = {"hand", "left_finger", "right_finger", "link6", "link7"}
            proximal = {"link0", "link1", "link2", "link3", "link4"}
            if (pair & distal) and (pair & proximal) and contact.dist < -0.004:
                return f"self:{body1}->{body2}"
            continue
        if not (pair & ROBOT_BODIES):
            continue
        other = next(iter(pair - ROBOT_BODIES))
        robot = body1 if body1 in ROBOT_BODIES else body2
        # Contact with the object being grasped is only legal at the two
        # fingertips.  The old rule ignored every robot-target pair, which
        # allowed the palm/forearm to push through the box.
        if (
            other in ignore_bodies
            and robot in {"left_finger", "right_finger"}
            and _is_free_movable_body(model, other)
        ):
            continue
        if other == "world":
            geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
            geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
            label = geom1 if body1 == "world" else geom2
            if label == "floor":
                continue
            other = label or "world"
        return f"{robot}->{other}"
    return None


def _robot_scene_clearance_hit(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ignore_bodies: set[str],
    clearance_m: float,
) -> str | None:
    """Return the first robot/scene pair closer than ``clearance_m``.

    Contact-only validity lets OMPL skim an obstacle at zero clearance.  A
    position controller can then deviate by a millimetre and push a movable
    object.  MuJoCo's exact signed geom distance gives the planner a real
    safety margin without approximating the arm or obstacles as spheres.
    """
    robot_geoms: list[tuple[int, str]] = []
    scene_geoms: list[tuple[int, str]] = []
    for geom_id in range(model.ngeom):
        if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        ) or ""
        if body_name in ROBOT_BODIES:
            robot_geoms.append((geom_id, body_name))
        elif body_name != "world":
            scene_geoms.append((geom_id, body_name))
    for robot_geom, robot_body in robot_geoms:
        for scene_geom, scene_body in scene_geoms:
            if (
                scene_body in ignore_bodies
                and robot_body in {"left_finger", "right_finger"}
                and _is_free_movable_body(model, scene_body)
            ):
                continue
            distance = float(mujoco.mj_geomDistance(
                model, data, robot_geom, scene_geom, clearance_m, None
            ))
            if distance < clearance_m:
                return f"{robot_body}->{scene_body}({distance * 1000:.1f}mm)"
    return None


def _payload_scene_clearance_hit(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    payload_body: str,
    ignore_bodies: set[str],
    clearance_m: float,
    approach_body: str | None = None,
) -> str | None:
    """Require a swept safety envelope around the carried object itself.

    ``approach_body`` skips the clearance test against that one support during
    a placement descent. Penetration and every other body stay in force.
    """
    payload_id = model.body(payload_body).id
    payload_geoms = [
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == payload_id
        and int(model.geom_contype[geom_id]) != 0
    ]
    for payload_geom in payload_geoms:
        for scene_geom in range(model.ngeom):
            scene_body_id = int(model.geom_bodyid[scene_geom])
            if scene_body_id == payload_id or int(model.geom_contype[scene_geom]) == 0:
                continue
            scene_body = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, scene_body_id
            ) or "world"
            scene_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, scene_geom
            ) or scene_body
            if scene_body in ROBOT_BODIES or scene_body in ignore_bodies:
                continue
            if approach_body and scene_body == approach_body:
                continue
            if scene_body == "world" and scene_name == "floor":
                continue
            distance = mujoco.mj_geomDistance(
                model, data, payload_geom, scene_geom, clearance_m, None
            )
            # MuJoCo returns `distmax` (with tiny float error) when a pair is
            # farther away than the query horizon.  Do not turn that sentinel
            # into a false collision with a distant object.
            if distance < clearance_m * 0.95:
                return f"payload-clearance:{payload_body}->{scene_body or scene_name}"
    return None


def _minimum_exact_clearance_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ignore_bodies: set[str],
    payload_body: str,
    horizon_m: float = 0.25,
) -> float:
    """Smallest exact geom gap among the pairs the transport checks consider.

    Fingertip contact with the carried object is excluded, matching
    ``_robot_scene_clearance_hit``. The horizon is only the query limit.
    """
    horizon = float(horizon_m)
    minimum = horizon
    robot_geoms: list[tuple[int, str]] = []
    scene_geoms: list[tuple[int, str]] = []
    payload_id = model.body(payload_body).id
    payload_geoms = []
    for geom_id in range(model.ngeom):
        if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name in ROBOT_BODIES:
            robot_geoms.append((geom_id, body_name))
        elif body_name != "world":
            scene_geoms.append((geom_id, body_name))
        if body_id == payload_id and int(model.geom_contype[geom_id]) != 0:
            payload_geoms.append(geom_id)
    for robot_geom, robot_body in robot_geoms:
        for scene_geom, scene_body in scene_geoms:
            if scene_body in ignore_bodies and robot_body in {"left_finger", "right_finger"}:
                continue
            distance = float(mujoco.mj_geomDistance(
                model, data, robot_geom, scene_geom, horizon, None
            ))
            if distance < horizon * 0.999:
                minimum = min(minimum, distance)
    for payload_geom in payload_geoms:
        for scene_geom in range(model.ngeom):
            scene_body_id = int(model.geom_bodyid[scene_geom])
            if scene_body_id == payload_id or int(model.geom_contype[scene_geom]) == 0:
                continue
            scene_body = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, scene_body_id
            ) or "world"
            scene_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, scene_geom
            ) or scene_body
            if scene_body in ROBOT_BODIES or scene_body in ignore_bodies:
                continue
            if scene_body == "world" and scene_name == "floor":
                continue
            distance = float(mujoco.mj_geomDistance(
                model, data, payload_geom, scene_geom, horizon, None
            ))
            if distance < horizon * 0.999:
                minimum = min(minimum, distance)
    return float(minimum)


def _component_exact_clearances(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ignore_bodies: set[str],
    payload_body: str,
    horizon_m: float = 0.25,
) -> dict[str, object]:
    """Separate robot and payload gaps. The horizon is only a query limit."""
    horizon = float(horizon_m)
    robot_gap = horizon
    robot_pair = None
    payload_gap = horizon
    payload_pair = None
    payload_id = model.body(payload_body).id
    robot_geoms: list[tuple[int, str]] = []
    scene_geoms: list[tuple[int, str]] = []
    payload_geoms: list[int] = []
    for geom_id in range(model.ngeom):
        if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name in ROBOT_BODIES:
            robot_geoms.append((geom_id, body_name))
        elif body_name != "world":
            scene_geoms.append((geom_id, body_name))
        if body_id == payload_id and int(model.geom_contype[geom_id]) != 0:
            payload_geoms.append(geom_id)
    for robot_geom, robot_body in robot_geoms:
        for scene_geom, scene_body in scene_geoms:
            if scene_body in ignore_bodies and robot_body in {"left_finger", "right_finger"}:
                continue
            distance = float(mujoco.mj_geomDistance(
                model, data, robot_geom, scene_geom, horizon, None
            ))
            if distance < robot_gap:
                robot_gap = distance
                robot_pair = f"{robot_body}->{scene_body}"
    for payload_geom in payload_geoms:
        for scene_geom in range(model.ngeom):
            scene_body_id = int(model.geom_bodyid[scene_geom])
            if scene_body_id == payload_id or int(model.geom_contype[scene_geom]) == 0:
                continue
            scene_body = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, scene_body_id
            ) or "world"
            scene_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, scene_geom
            ) or scene_body
            if scene_body in ROBOT_BODIES or scene_body in ignore_bodies:
                continue
            if scene_body == "world" and scene_name == "floor":
                continue
            distance = float(mujoco.mj_geomDistance(
                model, data, payload_geom, scene_geom, horizon, None
            ))
            if distance < payload_gap:
                payload_gap = distance
                payload_pair = f"{payload_body}->{scene_body or scene_name}"
    return {
        "robot_scene_gap_m": float(robot_gap),
        "robot_scene_pair": robot_pair,
        "payload_scene_gap_m": float(payload_gap),
        "payload_scene_pair": payload_pair,
    }


def execution_sample_released(
    *,
    scene_hit: str | None,
    robot_clearance_hit: str | None,
    payload_clearance_hit: str | None,
    exact_minimum_clearance_m: float,
) -> bool:
    """Exact trajectory evidence is the only release gate.

    Approximate marker clearance is not an input. A negative marker value
    cannot reject or accept a candidate by itself.
    """
    if scene_hit or robot_clearance_hit or payload_clearance_hit:
        return False
    if not np.isfinite(exact_minimum_clearance_m):
        return False
    return float(exact_minimum_clearance_m) >= 0.0


def _executed_trajectory_limits(
    profile: str, motion_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Joint limits matching ``move_to_ruckig`` for this profile and scale."""
    limits = scaled_motion_limits(profile, motion_scale)
    velocity = np.full(7, limits.max_velocity, dtype=float)
    acceleration = np.full(7, limits.max_acceleration, dtype=float)
    jerk = np.full(7, limits.max_jerk, dtype=float)
    if str(profile).startswith("payload"):
        velocity[4:] *= 0.65
        acceleration[4:] *= 0.55
        jerk[4:] *= 0.50
    return velocity, acceleration, jerk, float(limits.minimum_duration)


def _geom_vertical_radius(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """Conservative world-Z half extent for common primitive geoms."""
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    z_row = np.abs(rotation[2])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(z_row @ size[:3])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return float(z_row @ size[:3])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(size[0])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        axis_z = float(z_row[2])
        return axis_z * float(size[1]) + np.sqrt(max(0.0, 1.0 - axis_z**2)) * float(size[0])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return float(size[0]) + float(z_row[2]) * float(size[1])
    return float(model.geom_rbound[geom_id])


def _geom_horizontal_radius(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> float:
    """Horizontal footprint radius.  A neighbor outside this radius is not underneath."""
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        extent = 0.0
        half = size[:3]
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                corner = rotation @ (half * np.array((sx, sy, 0.0)))
                extent = max(extent, float(np.hypot(corner[0], corner[1])))
        return extent
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        axis = np.abs(rotation[:, 2])
        return float(size[0] + axis[0] * size[1] + axis[1] * size[1])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(size[0])
    return float(model.geom_rbound[geom_id])


def _surface_under_xy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    xy: np.ndarray,
    excluded_bodies: set[str],
) -> float:
    """Highest support whose footprint contains this point.

    A tall box beside the cup does not raise the pinch.  Only a surface
    directly under the fingers, such as the pedestal under the cube, does.
    """
    point = np.asarray(xy, dtype=float)
    highest = 0.0
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if (
            body_name in ROBOT_BODIES
            or body_name in excluded_bodies
            or geom_name == "floor"
            or int(model.geom_contype[geom_id]) == 0
        ):
            continue
        center = data.geom_xpos[geom_id, :2]
        if float(np.linalg.norm(center - point)) > _geom_horizontal_radius(model, data, geom_id):
            continue
        top = float(data.geom_xpos[geom_id, 2]) + _geom_vertical_radius(model, data, geom_id)
        highest = max(highest, top)
    return highest


def _nearest_obstacle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    xy: np.ndarray,
    excluded_bodies: set[str],
) -> tuple[float, float]:
    """Return (top height, horizontal distance) of the nearest raised neighbor."""
    point = np.asarray(xy, dtype=float)
    best_distance = float("inf")
    best_top = 0.0
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if body_name in ROBOT_BODIES or body_name in excluded_bodies or geom_name == "floor":
            continue
        if int(model.geom_contype[geom_id]) == 0:
            continue
        center = data.geom_xpos[geom_id, :2]
        distance = float(np.linalg.norm(center - point))
        radius = _geom_horizontal_radius(model, data, geom_id)
        gap = distance - radius
        top = float(data.geom_xpos[geom_id, 2]) + _geom_vertical_radius(model, data, geom_id)
        if top < 0.03 or gap > 0.25 or gap >= best_distance:
            continue
        best_distance = gap
        best_top = top
    if not np.isfinite(best_distance):
        return 0.0, float("inf")
    return best_top, best_distance


def palm_from_height_ratio(
    bottom_z: float,
    top_z: float,
    ratio: float,
    support_top: float,
) -> tuple[float, float, float]:
    """Palm height whose finger pad stays on the object and above the support."""
    height = max(float(top_z) - float(bottom_z), 1e-3)
    used = float(np.clip(ratio, 0.28, 0.72))
    finger_z = float(bottom_z) + used * height
    min_finger = float(support_top) + 0.008
    max_finger = float(top_z) - 0.010
    if max_finger <= min_finger:
        finger_z = max(float(bottom_z) + 0.008, min(float(top_z) - 0.004, min_finger))
    else:
        finger_z = float(np.clip(finger_z, min_finger, max_finger))
    return finger_z + FINGER_PAD_BELOW_HAND, finger_z, used


def _local_route_ceiling(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    excluded_bodies: set[str],
) -> float:
    """Highest obstacle near this XY route, instead of the whole scene."""
    segment = np.asarray(end_xy, dtype=float) - np.asarray(start_xy, dtype=float)
    length2 = float(segment @ segment)
    highest = 0.0
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if body_name in ROBOT_BODIES or body_name in excluded_bodies or geom_name == "floor":
            continue
        center_xy = data.geom_xpos[geom_id, :2]
        if length2 < 1e-9:
            distance = float(np.linalg.norm(center_xy - start_xy))
        else:
            alpha = float(np.clip((center_xy - start_xy) @ segment / length2, 0.0, 1.0))
            distance = float(np.linalg.norm(center_xy - (start_xy + alpha * segment)))
        horizontal_radius = min(float(model.geom_rbound[geom_id]), 0.30)
        if distance <= horizontal_radius + 0.09:
            top = float(data.geom_xpos[geom_id, 2]) + _geom_vertical_radius(model, data, geom_id)
            highest = max(highest, top)
    return highest


def _marked_route_ceiling(
    obstacles: list[dict] | None,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
) -> float:
    """Highest tall marker recorded during the scan and lying near this route."""
    if not obstacles:
        return 0.0
    start_xy = np.asarray(start_xy, dtype=float)
    end_xy = np.asarray(end_xy, dtype=float)
    segment = end_xy - start_xy
    length2 = float(segment @ segment)
    highest = 0.0
    for marker in obstacles:
        center = np.asarray(marker["xy"], dtype=float)
        if length2 < 1e-9:
            distance = float(np.linalg.norm(center - start_xy))
        else:
            alpha = float(np.clip((center - start_xy) @ segment / length2, 0.0, 1.0))
            distance = float(np.linalg.norm(center - (start_xy + alpha * segment)))
        if distance <= float(marker.get("radius", 0.14)) + 0.08:
            highest = max(highest, float(marker["top_z"]))
    return highest


def _segment_clears_base(start_xy: np.ndarray, end_xy: np.ndarray, radius: float = 0.20) -> bool:
    """False when the straight XY line passes through the robot column."""
    start_xy = np.asarray(start_xy, dtype=float)
    end_xy = np.asarray(end_xy, dtype=float)
    segment = end_xy - start_xy
    length2 = float(segment @ segment)
    if length2 < 1e-8:
        return float(np.linalg.norm(start_xy)) >= radius
    alpha = float(np.clip(-start_xy @ segment / length2, 0.0, 1.0))
    closest = start_xy + alpha * segment
    return float(np.linalg.norm(closest)) >= radius


def _wide_route(
    start: np.ndarray,
    target: np.ndarray,
    travel_z: float,
    direction: float = 1.0,
    radius: float | None = None,
) -> list[np.ndarray]:
    """Swing outside both pedestals when the goal is on the other side of the base.

    A straight line here drives the gripper through the upper arm.  The arc
    stays near 0.72 m from the column, which is past the pedestal corners.
    """
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    travel_z = max(float(travel_z), float(start[2]), float(target[2]), 0.36)
    if radius is None:
        radius = float(np.clip(
            max(np.linalg.norm(start[:2]), np.linalg.norm(target[:2])) + 0.10,
            0.50,
            0.74,
        ))
    radius = float(radius)
    a0 = float(np.arctan2(start[1], start[0]))
    a1 = float(np.arctan2(target[1], target[0]))
    delta = (a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi
    if direction < 0.0:
        delta += -2.0 * np.pi if delta > 0.0 else 2.0 * np.pi
    outer0 = np.array([radius * np.cos(a0), radius * np.sin(a0), travel_z])
    points = [start.copy()]
    lifted = np.array([start[0], start[1], travel_z])
    lift_distance = float(travel_z - start[2])
    if lift_distance > 0.012:
        lift_steps = max(1, int(np.ceil(lift_distance / 0.04)))
        for index in range(1, lift_steps + 1):
            alpha = index / lift_steps
            points.append((1.0 - alpha) * start + alpha * lifted)
    # Collision checking below is dense in joint space, independently of the
    # Cartesian route spacing.  Twelve-centimetre geometry vertices avoid
    # making a compliant grasp stop and settle twenty-plus times on a long
    # outer arc, which otherwise accumulates contact creep.
    route_step = 0.12
    out_steps = max(1, int(np.ceil(
        float(np.linalg.norm(outer0[:2] - start[:2])) / route_step
    )))
    for index in range(1, out_steps + 1):
        alpha = index / out_steps
        points.append((1.0 - alpha) * lifted + alpha * outer0)
    sweep_steps = max(2, int(np.ceil(abs(delta) * radius / route_step)))
    for index in range(1, sweep_steps + 1):
        angle = a0 + delta * index / sweep_steps
        points.append(np.array([
            radius * np.cos(angle), radius * np.sin(angle), travel_z,
        ]))
    in_steps = max(1, int(np.ceil(
        float(np.linalg.norm(target[:2] - points[-1][:2])) / route_step
    )))
    begin = points[-1].copy()
    end = np.array([target[0], target[1], travel_z])
    for index in range(1, in_steps + 1):
        alpha = index / in_steps
        points.append((1.0 - alpha) * begin + alpha * end)
    if float(end[2] - target[2]) > 0.02:
        points.append(target.copy())
    elif np.linalg.norm(points[-1] - target) > 0.008:
        points.append(target.copy())
    return points


def _side_points(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    obstacles: list[dict] | None,
    offset: float = 0.32,
) -> list[np.ndarray]:
    """Points beside the straight line, the clearer side first.

    Going around a pedestal keeps a carried object low.  Climbing over it
    unloads the finger pads and the object drops.
    """
    start_xy = np.asarray(start_xy, dtype=float)
    end_xy = np.asarray(end_xy, dtype=float)
    delta = end_xy - start_xy
    length = float(np.linalg.norm(delta))
    if length < 1e-4:
        return []
    tangent = delta / length
    normal = np.array((-tangent[1], tangent[0]))
    mid = (start_xy + end_xy) / 2.0
    points = [mid + normal * offset, mid - normal * offset]

    def clearance(point: np.ndarray) -> float:
        if obstacles:
            return min(
                float(np.linalg.norm(point - np.asarray(marker["xy"], dtype=float)))
                for marker in obstacles
            )
        return float(np.linalg.norm(point))

    points.sort(key=clearance, reverse=True)
    return points


def _segment_hit(
    model: mujoco.MjModel,
    scene_qpos: np.ndarray,
    q_from: np.ndarray,
    q_to: np.ndarray,
    ignore_bodies: set[str],
    payload_clearance: tuple[float, float] | None = None,
    finger_q: float | None = None,
    payload_transform: tuple[np.ndarray, np.ndarray] | None = None,
    payload_approach_body: str | None = None,
) -> str | None:
    """Sample one joint-space segment on a scratch state so planning cannot move the robot."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = scene_qpos
    addresses = [int(model.joint(name).qposadr[0]) for name in ARM_JOINTS]
    fingers = [
        int(model.joint("finger_joint1").qposadr[0]),
        int(model.joint("finger_joint2").qposadr[0]),
    ]
    payload_body = None
    payload_qpos = None
    relative_position = None
    relative_rotation = None
    if payload_clearance is not None:
        movable = [
            name for name in ignore_bodies
            if name in BODY_OF_COLOR.values()
        ]
        if len(movable) == 1:
            payload_body = movable[0]
            body_id = model.body(payload_body).id
            joint_id = int(model.body_jntadr[body_id])
            if joint_id >= 0 and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                payload_qpos = int(model.jnt_qposadr[joint_id])
                if payload_transform is not None:
                    relative_position = np.asarray(payload_transform[0], dtype=float).copy()
                    relative_rotation = np.asarray(payload_transform[1], dtype=float).copy()
                else:
                    for address, value in zip(addresses, q_from, strict=True):
                        scratch.qpos[address] = value
                    mujoco.mj_forward(model, scratch)
                    hand_rotation = scratch.xmat[model.body("hand").id].reshape(3, 3).copy()
                    hand_position = scratch.xpos[model.body("hand").id].copy()
                    object_rotation = scratch.xmat[body_id].reshape(3, 3).copy()
                    object_position = scratch.xpos[body_id].copy()
                    relative_position = hand_rotation.T @ (object_position - hand_position)
                    relative_rotation = hand_rotation.T @ object_rotation
    # Validate the complete joint interpolation densely.  Endpoint-only or
    # eight-sample checks missed thin contacts between arc waypoints.
    samples = max(
        12,
        1 + int(np.ceil(float(np.max(np.abs(q_to - q_from))) / 0.015)),
    )
    for alpha in np.linspace(0.0, 1.0, samples):
        blend = alpha * alpha * (3.0 - 2.0 * alpha)
        q = q_from + blend * (q_to - q_from)
        for address, value in zip(addresses, q, strict=True):
            scratch.qpos[address] = value
        # Open fingers are the wider shape.  While carrying, use the real
        # aperture so a closed gripper is not planned through the forearm.
        opening = 0.04 if finger_q is None else float(finger_q)
        scratch.qpos[fingers[0]] = opening
        scratch.qpos[fingers[1]] = opening
        mujoco.mj_forward(model, scratch)
        if (
            payload_body is not None
            and payload_qpos is not None
            and relative_position is not None
            and relative_rotation is not None
        ):
            hand_id = model.body("hand").id
            hand_rotation = scratch.xmat[hand_id].reshape(3, 3).copy()
            hand_position = scratch.xpos[hand_id].copy()
            object_position = hand_position + hand_rotation @ relative_position
            object_rotation = hand_rotation @ relative_rotation
            quaternion = np.empty(4, dtype=float)
            mujoco.mju_mat2Quat(quaternion, object_rotation.reshape(-1))
            scratch.qpos[payload_qpos : payload_qpos + 3] = object_position
            scratch.qpos[payload_qpos + 3 : payload_qpos + 7] = quaternion
            mujoco.mj_forward(model, scratch)
        hit = _scene_hit(model, scratch, ignore_bodies, payload_body=payload_body)
        if hit is None and payload_body is not None:
            hit = _payload_scene_clearance_hit(
                model,
                scratch,
                payload_body,
                ignore_bodies,
                clearance_m=0.010,
                approach_body=payload_approach_body,
            )
        if hit is not None:
            return hit
        if payload_clearance is not None:
            drop, minimum_z = payload_clearance
            hand_z = float(scratch.xpos[model.body("hand").id][2])
            if hand_z - drop < minimum_z:
                return "carried-object"
    return None


def _route(
    start: np.ndarray,
    target: np.ndarray,
    travel_z: float,
    *,
    fast: bool = False,
    quick_descend: bool = False,
) -> list[np.ndarray]:
    """Lift, travel in a straight horizontal line, then descend.

    ``fast`` is only for an empty hand racing to a hover above the target.
    A carried object still uses short steps so the pads are not unloaded.
    """
    travel_z = max(float(travel_z), float(start[2]), float(target[2]))
    lifted = np.array([start[0], start[1], travel_z])
    hovered = np.array([target[0], target[1], travel_z])
    points = [start.copy()]
    lift_step = 0.10 if fast else 0.03
    # Carried moves stay short so each inverse solution is a small joint
    # change.  A 10 cm jump across the base often has no solution, and the
    # one tall path that did had the base swing hard enough to drop the block.
    horizontal_step = 0.22 if fast else 0.06
    descend_step = 0.05 if (fast or quick_descend) else 0.015
    # Lift in short vertical increments while holding.  A single fast
    # 20-30 cm joint move can exceed the frictional grip and drop the object.
    lift_distance = float(lifted[2] - start[2])
    if lift_distance > 0.012:
        lift_steps = max(1, int(np.ceil(lift_distance / lift_step)))
        for index in range(1, lift_steps + 1):
            alpha = index / lift_steps
            points.append((1.0 - alpha) * start + alpha * lifted)
    distance = float(np.linalg.norm(hovered[:2] - lifted[:2]))
    steps = max(1, int(np.ceil(distance / horizontal_step)))
    for index in range(1, steps + 1):
        alpha = index / steps
        points.append((1.0 - alpha) * lifted + alpha * hovered)
    cursor = travel_z
    while cursor - float(target[2]) > descend_step + 0.01:
        cursor -= descend_step
        points.append(np.array([target[0], target[1], cursor]))
    if np.linalg.norm(points[-1] - target) > 0.008:
        points.append(target.copy())
    return points


def ik_branch_seeds(
    q_now: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    bearing: float,
) -> list[np.ndarray]:
    """Joint seeds for one Cartesian target: elbow, wrist, and redundancy.

    Every seed is clipped to the joint limits. The live posture stays first.
    """
    q = np.asarray(q_now, dtype=float).reshape(7).copy()
    low = np.asarray(lower, dtype=float).reshape(7)
    high = np.asarray(upper, dtype=float).reshape(7)

    def clipped(seed: np.ndarray) -> np.ndarray:
        return np.clip(seed, low + 1e-3, high - 1e-3)

    seeds = [clipped(q)]
    bearing_seed = q.copy()
    bearing_seed[0] = float(bearing)
    seeds.append(clipped(bearing_seed))
    for elbow in (low[3] + 0.20, high[3] - 0.20):
        branch = q.copy()
        branch[3] = float(elbow)
        seeds.append(clipped(branch))
    wrist = q.copy()
    flipped = float(q[6] + np.pi)
    if flipped > float(high[6]) - 1e-3:
        flipped = float(q[6] - np.pi)
    wrist[6] = flipped
    seeds.append(clipped(wrist))
    redundant = q.copy()
    redundant[2] = float(q[2] + np.pi / 2.0)
    seeds.append(clipped(redundant))
    opposite = bearing_seed.copy()
    opposite[0] = float(bearing + np.pi)
    seeds.append(clipped(opposite))
    unique: list[np.ndarray] = []
    for seed in seeds:
        if any(float(np.max(np.abs(seed - old))) < 0.20 for old in unique):
            continue
        unique.append(seed)
    return unique


def _ik_goal_candidates(
    model: mujoco.MjModel,
    target_xyz: np.ndarray,
    q_now: np.ndarray,
    yaw_rad: float,
) -> list[np.ndarray]:
    """Solve the Cartesian goal from distinct base, elbow, and wrist branches."""
    bearing = float(np.arctan2(target_xyz[1], target_xyz[0]))
    lower = np.asarray([model.joint(name).range[0] for name in ARM_JOINTS], dtype=float)
    upper = np.asarray([model.joint(name).range[1] for name in ARM_JOINTS], dtype=float)
    seeds = ik_branch_seeds(q_now, lower, upper, bearing)
    canonical = LEFT_ABOVE.copy()
    canonical[0] = float(np.clip(bearing, lower[0] + 1e-3, upper[0] - 1e-3))
    seeds.append(np.clip(canonical, lower + 1e-3, upper - 1e-3))
    candidates: list[np.ndarray] = []
    for seed in seeds:
        try:
            candidate = solve_hand_ik(
                model, target_xyz, seed=seed, yaw_rad=yaw_rad
            )
        except RuntimeError:
            continue
        if not any(np.max(np.abs(candidate - old)) < 0.05 for old in candidates):
            candidates.append(candidate)
    candidates.sort(key=lambda q: float(np.linalg.norm(q - q_now)))
    return candidates


def transport_execution_evidence(
    plan: dict[str, object] | None,
    monitor: TransportMonitor,
) -> dict[str, object]:
    """Plan gaps plus the live minimums. Recording does not change the abort."""
    trace = dict(monitor.clearance_trace)
    planned = None if not isinstance(plan, dict) else plan.get("exact_payload_clearance_m")
    live = trace.get("live_min_payload_clearance_m")
    loss = None
    if planned is not None and live is not None:
        loss = float(planned) - float(live)
    return {
        "plan": plan,
        "execution": monitor.metrics(),
        "clearance_trace": trace,
        "live_gap_loss_m": loss,
    }


def stamp_grasp_attachment(robot: object, monitor: TransportMonitor) -> None:
    """Copy the monitor capture and the planning capture onto the transport record."""
    record = getattr(robot, "last_transport_plan_record", None)
    if not isinstance(record, dict):
        return
    record["monitor_local_xyz_m"] = [
        float(value) for value in monitor.baseline_local_xyz
    ]
    record["monitor_up_local"] = [
        float(value) for value in monitor.baseline_up_local
    ]
    context = getattr(robot, "last_transport_planning_context", None)
    if isinstance(context, dict):
        record["planning_capture"] = context


def classify_transport_goal(
    queries: list[dict[str, object]],
    *,
    ruckig_category: str | None = None,
    exact_minimum_clearance_m: float | None = None,
) -> str:
    """Name the first stage that kept this IK goal from becoming a candidate.

    A negative exact gap is a rejection even when the sampled checks did not
    already name a body. The live clearance standards are not inputs.
    """
    if not any(bool(query.get("goal_valid")) for query in queries):
        return "goal_invalid"
    if not any(bool(query.get("ompl_exact")) for query in queries):
        return "ompl_no_exact"
    if not any(bool(query.get("chord_dense")) for query in queries):
        return "chord_dense_rejected"
    if ruckig_category:
        return str(ruckig_category)
    if exact_minimum_clearance_m is not None and float(exact_minimum_clearance_m) < 0.0:
        return "negative_exact_gap"
    return "validated"


# Route entries from gap-v2 used the hand-body pivot. A version miss is no entry.
CLEARANCE_POLICY_VERSION = "gap-v3"


def transport_route_identity(
    payload_body: str,
    target_xyz: np.ndarray,
    yaw_rad: float,
    *,
    support_name: str | None = None,
    plan_stage: str = "",
    geometry_class: str = "",
    attachment_class: str = "",
    clearance_policy_version: str = CLEARANCE_POLICY_VERSION,
) -> dict[str, object]:
    """Key a carried route by body, support, sector, and grasp-yaw bucket.

    Yaw is folded by half a turn because a 180 degree finger flip is the same
    pinch. The sector and bucket only choose a file entry. ``plan_stage`` keeps
    the camera hover from replacing the place hover. A different clearance
    policy or attachment class is a different key.
    """
    xy = np.asarray(target_xyz, dtype=float)
    sector = float(TRANSPORT_ROUTE_SECTOR_M)
    sx = round(float(np.floor(float(xy[0]) / sector + 1e-9) * sector), 2)
    sy = round(float(np.floor(float(xy[1]) / sector + 1e-9) * sector), 2)
    folded = float(yaw_rad) % np.pi
    bucket = float(
        np.round(folded / TRANSPORT_ROUTE_YAW_BUCKET_RAD)
        * TRANSPORT_ROUTE_YAW_BUCKET_RAD
    ) % np.pi
    support = support_name or "unspecified"
    stage = plan_stage or "move"
    version = clearance_policy_version or CLEARANCE_POLICY_VERSION
    geometry = geometry_class or "unspecified"
    attachment = attachment_class or "unspecified"
    return {
        "key": (
            f"{payload_body}|{support}|{sx:.2f},{sy:.2f}|{bucket:.3f}|{stage}"
            f"|{version}|{geometry}|{attachment}"
        ),
        "payload_body": payload_body,
        "support_name": support,
        "support_sector_m": [sx, sy],
        "yaw_bucket_rad": bucket,
        "plan_stage": stage,
        "clearance_policy_version": version,
        "geometry_class": geometry,
        "attachment_class": attachment,
    }


def splice_reused_waypoints(
    q_start: np.ndarray,
    archived: Sequence[Sequence[float]],
) -> list[np.ndarray] | None:
    """Replace the archived first joint point with the live start."""
    points = [np.asarray(point, dtype=float) for point in archived]
    if len(points) < 2:
        return None
    return [np.asarray(q_start, dtype=float), *points[1:]]


def upsert_transport_routes(
    routes: list[dict[str, object]],
    entry: dict[str, object],
    cap: int = TRANSPORT_ROUTE_LIBRARY_CAP,
) -> list[dict[str, object]]:
    """Replace the same key. A new key past the cap drops the oldest entry."""
    key = entry.get("key")
    kept = [route for route in routes if route.get("key") != key]
    kept.append(entry)
    if len(kept) > int(cap):
        kept = kept[-int(cap):]
    return kept


def route_reuse_rejection(detail: str, key: str) -> dict[str, object]:
    """A stored path that misses any current check is only a rejection."""
    return {"reason": "route_reuse_rejected", "detail": detail, "key": key}


def dense_chord_death(
    points: Sequence[Sequence[float]],
    sample_death: Callable[[np.ndarray], str | None],
    maximum_step: float,
) -> str | None:
    """Name the first current-check failure on the same samples as a dense chord."""
    arrays = [np.asarray(point, dtype=float) for point in points]
    if not arrays:
        return "chord_dense_rejected"
    for point in arrays:
        death = sample_death(point)
        if death:
            return death
    for start, end in zip(arrays, arrays[1:]):
        samples = max(2, 1 + int(np.ceil(
            np.max(np.abs(end - start)) / max(float(maximum_step), 1e-4)
        )))
        for alpha in np.linspace(0.0, 1.0, samples):
            death = sample_death(start + float(alpha) * (end - start))
            if death:
                return death
    return None


def reuse_sample_death(report: dict[str, object]) -> str | None:
    """Name the existing check that rejects one reused sample, if any."""
    if float(report["pose_deviation_deg"]) > float(report["pose_limit_deg"]) + 1e-9:
        return "pose"
    if float(report["payload_bottom_z_m"]) + 1e-9 < float(report["payload_bottom_limit_z_m"]):
        return "height"
    if report.get("scene_hit"):
        return "scene"
    if report.get("robot_clearance_hit"):
        return "robot_clearance"
    if report.get("payload_clearance_hit"):
        return "payload_clearance"
    return None


def transport_pose_within_limit(
    maximum_deviation_deg: float,
    allowed_deviation_deg: float,
) -> bool:
    """A candidate may not exceed the existing payload pose tolerance."""
    return float(maximum_deviation_deg) <= float(allowed_deviation_deg) + 1e-9


def transport_candidate_score_key(
    candidate: TransportPlanCandidate | dict[str, object],
    preference: str,
) -> tuple[float, ...]:
    """Deterministic local ranking over validated candidate measurements."""
    scores = candidate.scores if isinstance(candidate, TransportPlanCandidate) else candidate
    duration = float(scores["estimated_duration_s"])
    pose = float(scores["maximum_pose_deviation_deg"])
    acceleration = float(scores["maximum_tangential_acceleration_m_s2"])
    clearance = float(scores["minimum_clearance_m"])
    base = float(scores["maximum_base_turn_rad"])
    wrist = float(scores["maximum_wrist_swing_rad"])
    slack = scores.get("safety_slack_m")
    slack_key = 0.0 if slack is None else -float(slack)
    if preference == "time_first":
        return (slack_key, duration, pose, acceleration, -clearance, base, wrist)
    # Stability-first prefers physical clearance before dynamic quality.  All
    # candidates passed the hard collision gate, but the larger measured
    # margin is the safer choice under live tracking error.
    return (slack_key, -clearance, pose, acceleration, base, wrist, duration)


def choose_transport_candidate(
    candidates: list[TransportPlanCandidate],
    preference: str,
    requested_id: str | None = None,
) -> TransportPlanCandidate | None:
    """Honor a manager ID only when it names a locally validated candidate."""
    validated = [candidate for candidate in candidates if candidate.validated]
    if requested_id is not None:
        for candidate in validated:
            if candidate.candidate_id == requested_id:
                return candidate
    if not validated:
        return None
    return min(
        validated,
        key=lambda candidate: (
            *transport_candidate_score_key(candidate, preference),
            candidate.candidate_id,
        ),
    )


def should_fallback_to_arc(candidates: list[TransportPlanCandidate]) -> bool:
    """Historical predicate. Carried transport must not call this."""
    return not any(candidate.validated for candidate in candidates)


def ompl_carried_search_plan(search_index: int, preference: str) -> dict[str, object] | None:
    """At most one re-search, with the 1.6 s budget doubled once.

    The second search flips stability_first/time_first and reverses IK goal
    order. A third search is refused.
    """
    current = "time_first" if preference == "time_first" else "stability_first"
    if search_index == 0:
        return {
            "preference": current,
            "timeout_s": TRANSPORT_OMPL_TIMEOUT_S,
            "reverse_ik_goals": False,
            "budget_doubled": False,
        }
    if search_index == 1:
        flipped = "stability_first" if current == "time_first" else "time_first"
        return {
            "preference": flipped,
            "timeout_s": TRANSPORT_OMPL_TIMEOUT_S * 2.0,
            "reverse_ik_goals": True,
            "budget_doubled": True,
        }
    return None


def carried_ompl_stop_record(
    rejected: list[dict[str, object]],
    *,
    preference: str,
    planning_time_s: float,
    planning_budget_s: float,
    budget_doubled: bool,
    ik_goal_order: str,
) -> dict[str, object]:
    """No validated OMPL candidate: stop without executing any trajectory."""
    return {
        "planner_used": "ompl",
        "arc_fallback_reason": None,
        "candidates": [],
        "rejected_candidates": list(rejected),
        "selected_candidate_id": None,
        "decision_maker": "local_ompl_stop",
        "route_preference": preference,
        "planner_seed": OMPL_RANDOM_SEED,
        "planning_time_s": float(planning_time_s),
        "planning_budget_s": float(planning_budget_s),
        "budget_doubled": bool(budget_doubled),
        "ik_goal_order": ik_goal_order,
        "executed": False,
    }


def _rotation_deviation_deg(reference: np.ndarray, current: np.ndarray) -> float:
    relative = np.asarray(reference, dtype=float).T @ np.asarray(current, dtype=float)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def payload_motion_envelope_m(
    slip_m: float,
    tilt_deg: float,
    r_pivot_m: float,
    tracking_margin_m: float = 0.0,
) -> float:
    """Slip plus the tilted pivot radius. tracking_margin stays 0 until measured."""
    return (
        float(slip_m)
        + float(r_pivot_m) * float(np.sin(np.deg2rad(tilt_deg)))
        + float(tracking_margin_m)
    )


def grasp_contact_origin_m(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Midpoint of the left and right finger-pad centroids at this pose.

    Each centroid is the volume-weighted center of that finger's box geoms,
    which are the five fingertip pad boxes. The hand body origin is not used.
    """
    def centroid(body_name: str) -> np.ndarray:
        body_id = model.body(body_name).id
        weighted = np.zeros(3)
        volume = 0.0
        box = int(mujoco.mjtGeom.mjGEOM_BOX)
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(model.geom_type[geom_id]) != box:
                continue
            size = np.asarray(model.geom_size[geom_id], dtype=float)
            geom_volume = 8.0 * float(size[0]) * float(size[1]) * float(size[2])
            weighted += geom_volume * np.asarray(data.geom_xpos[geom_id], dtype=float)
            volume += geom_volume
        if volume <= 0.0:
            return np.asarray(data.xpos[body_id], dtype=float).copy()
        return weighted / volume

    return 0.5 * (centroid("left_finger") + centroid("right_finger"))


def payload_pivot_radius_from_points(
    grasp_xyz: np.ndarray,
    surface_points: Sequence[Sequence[float]],
) -> float:
    """Largest distance from the grasp point to a payload surface point."""
    origin = np.asarray(grasp_xyz, dtype=float)
    points = [np.asarray(point, dtype=float) for point in surface_points]
    if not points:
        return 0.0
    return float(max(np.linalg.norm(point - origin) for point in points))


def _farthest_geom_surface_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    hand_xyz: np.ndarray,
) -> np.ndarray | None:
    """One farthest surface point. The geom type only locates that surface."""
    center = np.asarray(data.geom_xpos[geom_id], dtype=float)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    hand = np.asarray(hand_xyz, dtype=float)
    geom_type = int(model.geom_type[geom_id])
    box = int(mujoco.mjtGeom.mjGEOM_BOX)
    cylinder = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    capsule = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    sphere = int(mujoco.mjtGeom.mjGEOM_SPHERE)
    ellipsoid = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    if geom_type == box:
        corners = [
            center + rotation @ np.array([
                sx * float(size[0]), sy * float(size[1]), sz * float(size[2]),
            ])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
        return max(corners, key=lambda point: float(np.linalg.norm(point - hand)))
    if geom_type == cylinder:
        radius = float(size[0])
        half = float(size[1])
        local_hand = rotation.T @ (hand - center)
        planar = local_hand[:2]
        norm = float(np.linalg.norm(planar))
        direction = np.array([1.0, 0.0]) if norm < 1e-9 else -planar / norm
        rims = [
            center + rotation @ np.array([
                radius * float(direction[0]),
                radius * float(direction[1]),
                z,
            ])
            for z in (-half, half)
        ]
        return max(rims, key=lambda point: float(np.linalg.norm(point - hand)))
    if geom_type in {capsule, sphere}:
        radius = float(size[0])
        ends = [center]
        if geom_type == capsule:
            half = float(size[1])
            ends = [
                center + rotation @ np.array([0.0, 0.0, sign * half])
                for sign in (-1.0, 1.0)
            ]
        samples = []
        for end in ends:
            away = end - hand
            norm = float(np.linalg.norm(away))
            if norm < 1e-9:
                samples.append(end + rotation @ np.array([radius, 0.0, 0.0]))
            else:
                samples.append(end + radius * away / norm)
        return max(samples, key=lambda point: float(np.linalg.norm(point - hand)))
    if geom_type == ellipsoid:
        poles = []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                local = np.zeros(3)
                local[axis] = sign * float(size[axis])
                poles.append(center + rotation @ local)
        return max(poles, key=lambda point: float(np.linalg.norm(point - hand)))
    return None


def payload_pivot_radius_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    payload_body: str,
) -> float:
    """Pivot radius about the finger-pad midpoint, including handle geoms."""
    grasp_xyz = grasp_contact_origin_m(model, data)
    payload_id = model.body(payload_body).id
    points = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != payload_id:
            continue
        point = _farthest_geom_surface_point(model, data, geom_id, grasp_xyz)
        if point is not None:
            points.append(point)
    return payload_pivot_radius_from_points(grasp_xyz, points)


def residual_tracking_margin_m(
    live_gap_loss_m: float,
    slip_m: float,
    tilt_deg: float,
    r_pivot_m: float,
) -> float:
    """Gap loss left after the measured slip and tilt are removed."""
    physical = payload_motion_envelope_m(slip_m, tilt_deg, r_pivot_m, 0.0)
    return max(0.0, float(live_gap_loss_m) - physical)


def ceil_tracking_margin_m(margin_m: float) -> float:
    """Round a non-positive residual to 0, otherwise up to the next 0.5 mm."""
    if float(margin_m) <= 0.0:
        return 0.0
    step = 0.0005
    return float(np.ceil(float(margin_m) / step - 1e-9) * step)


def _blank_component_tracker() -> dict[str, object]:
    return {
        "robot": float("inf"),
        "robot_pair": None,
        "payload": float("inf"),
        "payload_pair": None,
        "payload_samples": [],
    }


def _track_components(tracker: dict[str, object], parts: dict[str, object]) -> None:
    robot_gap = float(parts["robot_scene_gap_m"])
    payload_gap = float(parts["payload_scene_gap_m"])
    samples = tracker["payload_samples"]
    assert isinstance(samples, list)
    samples.append(payload_gap)
    if robot_gap < float(tracker["robot"]):
        tracker["robot"] = robot_gap
        tracker["robot_pair"] = parts.get("robot_scene_pair")
    if payload_gap < float(tracker["payload"]):
        tracker["payload"] = payload_gap
        tracker["payload_pair"] = parts.get("payload_scene_pair")


def payload_gap_summary(samples: Sequence[float]) -> dict[str, float | None]:
    values = [float(sample) for sample in samples]
    if not values:
        return {"min": None, "median": None, "max": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        median = 0.5 * (ordered[mid - 1] + ordered[mid])
    else:
        median = ordered[mid]
    return {"min": ordered[0], "median": float(median), "max": ordered[-1]}


def _component_score_fields(tracker: dict[str, object]) -> dict[str, float]:
    robot = float(tracker["robot"])
    payload = float(tracker["payload"])
    fields: dict[str, float] = {}
    if np.isfinite(robot):
        fields["exact_robot_clearance_m"] = robot
    if np.isfinite(payload):
        fields["exact_payload_clearance_m"] = payload
    return fields


def _component_detail_fields(tracker: dict[str, object]) -> dict[str, object]:
    samples = tracker["payload_samples"]
    assert isinstance(samples, list)
    summary = payload_gap_summary(samples)
    return {
        "exact_robot_pair": tracker["robot_pair"],
        "exact_payload_pair": tracker["payload_pair"],
        "payload_gap_min_m": summary["min"],
        "payload_gap_median_m": summary["median"],
        "payload_gap_max_m": summary["max"],
    }


# Blue-box live gap loss 12.834 mm, measured slip 2.993 mm, tilt 6.534°,
# grasp-point R 74.36 mm. Slip plus tilt is 11.455 mm, leaving 1.379 mm
# (tasklog_20261009_025406). Rounded up to the next 0.5 mm.
TRACKING_MARGIN_M = 0.0015
RAISE_TRANSPORT_STEP_M = 0.015
RAISE_TRANSPORT_CAP_M = 0.15


def payload_sweep_clearance_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    payload_body: str,
    monitor: TransportMonitor,
) -> float:
    """One envelope for every payload: slip, tilted pivot radius, tracking margin."""
    r_pivot = payload_pivot_radius_m(model, data, payload_body)
    return payload_motion_envelope_m(
        float(monitor.maximum_slip_m),
        float(monitor.maximum_tilt_deg),
        r_pivot,
        TRACKING_MARGIN_M,
    )


def apply_safety_slack(scores: dict[str, float], payload_limit_m: float) -> None:
    """Slack is the smaller surplus over the 12 mm robot gate and the envelope."""
    robot_gap = scores.get("exact_robot_clearance_m")
    payload_gap = scores.get("exact_payload_clearance_m")
    if robot_gap is None or payload_gap is None:
        return
    scores["safety_slack_m"] = min(
        float(robot_gap) - 0.012,
        float(payload_gap) - float(payload_limit_m),
    )


def transport_release_reason(
    robot_gap_m: float,
    payload_gap_m: float,
    payload_limit_m: float,
    robot_limit_m: float = 0.012,
) -> str | None:
    """Full-value release. The 0.95 factor is not used here."""
    if float(robot_gap_m) + 1e-12 < float(robot_limit_m):
        return "robot_clearance"
    if float(payload_gap_m) + 1e-12 < float(payload_limit_m):
        return "payload_gap_below_sweep"
    return None


def stored_route_matches(route: Mapping[str, object], identity: Mapping[str, object]) -> bool:
    """A missing or different clearance version is no entry, not a rejection."""
    return (
        route.get("key") == identity.get("key")
        and route.get("clearance_policy_version") == identity.get("clearance_policy_version")
        and identity.get("clearance_policy_version") == CLEARANCE_POLICY_VERSION
    )


def payload_gap_only_wipeout(rejected: Sequence[Mapping[str, object]]) -> bool:
    """True only when every found candidate died of the payload envelope.

    ``ompl_no_exact`` and any other death keep the raise from starting. A
    route-reuse miss is not a candidate.
    """
    reasons = []
    for item in rejected:
        reason = str(item.get("reason") or "")
        if reason in {"route_reuse_rejected", "start_state_rejected"}:
            continue
        reasons.append(reason)
    return bool(reasons) and all(
        reason == "payload_gap_below_sweep" for reason in reasons
    )


def segmented_raise_recovery_allowed(
    rejected: Sequence[Mapping[str, object]],
) -> bool:
    """Whether a validated raised midpoint is a relevant next corridor.

    A start/goal validity, collision, pose, height, or robot-clearance failure
    is not repaired here.  A search that found no exact connection, or whose
    otherwise valid trajectory was eliminated only by the payload envelope,
    may be split once through the existing raised-midpoint ladder.
    """
    reasons = [
        str(item.get("reason") or "")
        for item in rejected
        if str(item.get("reason") or "")
        not in {"route_reuse_rejected", "start_state_rejected"}
    ]
    return bool(reasons) and all(
        reason in {"ompl_no_exact", "payload_gap_below_sweep"}
        for reason in reasons
    )


def raise_deltas_m(
    step_m: float = RAISE_TRANSPORT_STEP_M,
    cap_m: float = RAISE_TRANSPORT_CAP_M,
) -> list[float]:
    """15 mm steps from the first step through the 0.15 m cap, once."""
    count = int(round(float(cap_m) / float(step_m)))
    return [round(float(step_m) * (index + 1), 6) for index in range(count)]


def raise_midpoint_xyz(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    delta_m: float,
) -> np.ndarray:
    """One hand target above both ends. The extra height is local geometry."""
    start = np.asarray(start_xyz, dtype=float)
    end = np.asarray(end_xyz, dtype=float)
    midpoint = 0.5 * (start + end)
    midpoint[2] = max(float(start[2]), float(end[2])) + float(delta_m)
    return midpoint


def transport_corridor_waypoints(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
) -> list[np.ndarray]:
    """Return geometry-only left/right outer-corridor waypoints.

    These are OMPL sub-goals, never executable Cartesian waypoints.  The
    radius is the existing wide-route radius rule and each resulting OMPL
    segment must still pass the complete executed-trajectory validator.
    """
    start = np.asarray(start_xyz, dtype=float)
    end = np.asarray(end_xyz, dtype=float)
    radius = float(np.clip(
        max(np.linalg.norm(start[:2]), np.linalg.norm(end[:2])) + 0.10,
        0.50,
        0.74,
    ))
    a0 = float(np.arctan2(start[1], start[0]))
    a1 = float(np.arctan2(end[1], end[0]))
    short = (a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi
    long = short - np.sign(short if abs(short) > 1e-9 else 1.0) * 2.0 * np.pi
    z = max(float(start[2]), float(end[2]), 0.36)
    return [
        np.array((radius * np.cos(a0 + 0.5 * turn),
                  radius * np.sin(a0 + 0.5 * turn), z))
        for turn in (short, long)
    ]


def transport_escape_offsets_m(
    start_xyz: np.ndarray,
    goal_xyz: np.ndarray | None,
    step_m: float,
) -> list[tuple[str, np.ndarray]]:
    """Short transport-ready steps. Straight up stays first.

    Every step rises. The horizontal part retreats from the goal, then to
    either side. Length stays the caller's step; this does not raise a cap.
    """
    start = np.asarray(start_xyz, dtype=float)
    if goal_xyz is None:
        away = -start[:2]
    else:
        away = start[:2] - np.asarray(goal_xyz, dtype=float)[:2]
    if float(np.linalg.norm(away)) < 1e-6:
        away = np.array((1.0, 0.0))
    away = away / float(np.linalg.norm(away))
    side = np.array((-away[1], away[0]))
    directions = [
        ("up", np.array((0.0, 0.0, 1.0))),
        ("back_up", np.array((away[0], away[1], 1.0))),
        ("left_back_up", np.array((*(0.7 * away + 0.7 * side), 1.0))),
        ("right_back_up", np.array((*(0.7 * away - 0.7 * side), 1.0))),
    ]
    offsets: list[tuple[str, np.ndarray]] = []
    for name, direction in directions:
        unit = direction / float(np.linalg.norm(direction))
        offsets.append((name, unit * float(step_m)))
    return offsets


def experience_corridor_vias(
    polylines: Sequence[Sequence[np.ndarray]],
    start_xyz: np.ndarray,
    goal_xyz: np.ndarray,
    *,
    limit: int = 2,
) -> list[np.ndarray]:
    """Pick a few midpoints from earlier successful hand paths.

    The archived joints are not copied. A midpoint becomes an OMPL sub-goal
    only when it is clear of the current start and goal.
    """
    start = np.asarray(start_xyz, dtype=float)
    goal = np.asarray(goal_xyz, dtype=float)
    samples: list[np.ndarray] = []
    for polyline in polylines:
        points = [np.asarray(point, dtype=float) for point in polyline]
        if len(points) < 3:
            continue
        for fraction in (1.0 / 3.0, 2.0 / 3.0):
            index = int(round(fraction * (len(points) - 1)))
            index = min(max(index, 1), len(points) - 2)
            via = points[index]
            if float(np.linalg.norm(via - start)) < 0.08:
                continue
            if float(np.linalg.norm(via - goal)) < 0.08:
                continue
            if any(float(np.linalg.norm(via - old)) < 0.05 for old in samples):
                continue
            samples.append(via)
    samples.sort(key=lambda via: (
        float(np.linalg.norm(via - start) + np.linalg.norm(goal - via))
    ))
    return samples[: int(limit)]


def join_raised_transport(
    first: TransportPlanCandidate,
    second: TransportPlanCandidate,
    delta_m: float,
) -> TransportPlanCandidate:
    """Concatenate two released segments and keep the tighter clearance."""
    scores: dict[str, float] = {
        key: float(value)
        for key, value in second.scores.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for key in (
        "exact_payload_clearance_m",
        "exact_robot_clearance_m",
        "minimum_clearance_m",
        "safety_slack_m",
        "exact_minimum_clearance_m",
    ):
        left = first.scores.get(key)
        right = scores.get(key)
        if left is None or right is None:
            continue
        scores[key] = min(float(left), float(right))
    for key in (
        "maximum_pose_deviation_deg",
        "maximum_base_turn_rad",
        "maximum_wrist_swing_rad",
        "maximum_tangential_acceleration_m_s2",
    ):
        left = first.scores.get(key)
        if left is None or key not in scores:
            continue
        scores[key] = max(float(left), float(scores[key]))
    if "estimated_duration_s" in scores:
        scores["estimated_duration_s"] = (
            float(first.scores.get("estimated_duration_s") or 0.0)
            + float(scores["estimated_duration_s"])
        )
    details = dict(second.details)
    details["raise_triggered"] = True
    details["raise_delta_m"] = float(delta_m)
    return TransportPlanCandidate(
        candidate_id=(
            f"raise:{route_root_id(first.candidate_id)}"
            f"+{route_root_id(second.candidate_id)}"
        ),
        joint_waypoints=(*first.joint_waypoints, *second.joint_waypoints[1:]),
        execution_points=(*first.execution_points, *second.execution_points),
        hand_waypoints=(*first.hand_waypoints, *second.hand_waypoints[1:]),
        scores=scores,
        planning_time_s=float(first.planning_time_s) + float(second.planning_time_s),
        validated=True,
        details=details,
    )


def route_root_id(candidate_id: str) -> str:
    """Drop repeated route_reuse prefixes so the archive stores the root id."""
    root = str(candidate_id)
    prefix = "route_reuse:"
    while root.startswith(prefix):
        root = root[len(prefix):]
    return root


def payload_geometry_class(model: mujoco.MjModel, payload_body: str) -> str:
    payload_id = model.body(payload_body).id
    parts = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != payload_id:
            continue
        size = np.round(np.asarray(model.geom_size[geom_id], dtype=float), 4)
        parts.append(f"{int(model.geom_type[geom_id])}:{','.join(f'{v:.4f}' for v in size)}")
    return "|".join(parts)


def attachment_class_of(relative_xyz: np.ndarray) -> str:
    rounded = np.round(np.asarray(relative_xyz, dtype=float) / 0.005) * 0.005
    return ",".join(f"{float(value):.3f}" for value in rounded)


def pre_ompl_separation_mode(
    *,
    scene_hit: str | None,
    robot_clearance_hit: str | None,
    pose_deviation_deg: float,
    pose_limit_deg: float,
    payload_bottom_z_m: float,
    payload_bottom_limit_z_m: float,
    payload_clearance_hit: str | None,
) -> str:
    """Lift when only vertical separation or payload sweep is still missing."""
    motion_safe = (
        scene_hit is None
        and robot_clearance_hit is None
        and pose_deviation_deg <= pose_limit_deg + 1e-9
    )
    height_ready = payload_bottom_z_m + 1e-9 >= payload_bottom_limit_z_m
    if motion_safe and height_ready and payload_clearance_hit is None:
        return "ready"
    if motion_safe and (not height_ready or payload_clearance_hit is not None):
        return "lift"
    return "blocked"


def pre_ompl_step_allowed(
    *,
    scene_hit: str | None,
    robot_clearance_hit: str | None,
    payload_gap_m: float,
    previous_payload_gap_m: float,
    pose_deviation_deg: float,
    pose_limit_deg: float,
    payload_bottom_z_m: float,
    payload_bottom_limit_z_m: float,
    payload_limit_m: float | None = None,
) -> bool:
    """Validate one vertical separation step against the actual hard gates.

    A gap may decrease slightly as the wrist follows IK only when it remains
    at or above the full payload envelope.  Without an envelope value the
    legacy monotonic rule is retained for callers and regression tests.
    """
    if scene_hit is not None or robot_clearance_hit is not None:
        return False
    gap_decreased = float(payload_gap_m) + 1e-9 < float(previous_payload_gap_m)
    absolute_gap_safe = (
        payload_limit_m is not None
        and float(payload_gap_m) + 1e-9 >= float(payload_limit_m)
    )
    if gap_decreased and not absolute_gap_safe:
        return False
    if pose_deviation_deg > pose_limit_deg + 1e-9:
        return False
    if payload_bottom_z_m + 1e-9 < payload_bottom_limit_z_m:
        return False
    return True


def pre_ompl_separation_status(
    *,
    lifted_m: float,
    lift_cap_m: float,
    payload_clearance_hit: str | None,
    payload_bottom_z_m: float | None = None,
    payload_bottom_limit_z_m: float | None = None,
) -> str:
    """Both height and sweep must be met before OMPL can use the start."""
    height_ready = (
        payload_bottom_z_m is None
        or payload_bottom_limit_z_m is None
        or float(payload_bottom_z_m) + 1e-9 >= float(payload_bottom_limit_z_m)
    )
    if payload_clearance_hit is None and height_ready:
        return "sweep_met"
    if float(lifted_m) + 1e-9 >= float(lift_cap_m):
        return "ceiling"
    return "continue"


def pre_ompl_may_plan(status: str) -> bool:
    """A ceiling or a rejected step is a stop, not a validated start."""
    return status in {"ready", "sweep_met"}


def _plan_ompl_transport_candidates(
    robot: PandaController,
    model: mujoco.MjModel,
    target_xyz: np.ndarray,
    *,
    yaw_rad: float,
    ignore_bodies: set[str],
    payload_clearance: tuple[float, float],
    obstacles: list[dict] | None,
    transport_monitor: TransportMonitor,
    timeout_s: float = TRANSPORT_OMPL_TIMEOUT_S,
    reverse_goals: bool = False,
    q_start: np.ndarray | None = None,
) -> tuple[list[TransportPlanCandidate], list[dict[str, object]], float]:
    """Plan payload-aware OMPL candidates without moving the live robot."""
    begun = time.perf_counter()
    lock = robot.data_lock if robot.data_lock is not None else nullcontext()
    with lock:
        q_now = (
            _arm_q(model, robot.data) if q_start is None
            else np.asarray(q_start, dtype=float).reshape(7).copy()
        )
        scene_qpos = robot.data.qpos.copy()
        finger_q = float(robot.data.qpos[
            int(model.joint("finger_joint1").qposadr[0])
        ])
        hand_id = model.body("hand").id
        initial_hand_rotation = robot.data.xmat[hand_id].reshape(3, 3).copy()
        payload_body = transport_monitor.body_name
        payload_id = model.body(payload_body).id
        payload_joint = int(model.body_jntadr[payload_id])
        payload_qpos = int(model.jnt_qposadr[payload_joint])
        object_rotation = robot.data.xmat[payload_id].reshape(3, 3).copy()
        object_position = robot.data.xpos[payload_id].copy()
        hand_position = robot.data.xpos[hand_id].copy()
        relative_position = initial_hand_rotation.T @ (object_position - hand_position)
        relative_rotation = initial_hand_rotation.T @ object_rotation
        grasp_origin = grasp_contact_origin_m(model, robot.data)
        r_pivot_m = payload_pivot_radius_m(model, robot.data, payload_body)
    shadow_envelope_m = payload_motion_envelope_m(
        float(transport_monitor.maximum_slip_m),
        float(transport_monitor.maximum_tilt_deg),
        r_pivot_m,
        TRACKING_MARGIN_M,
    )
    goals = _ik_goal_candidates(model, np.asarray(target_xyz, dtype=float), q_now, yaw_rad)
    if reverse_goals:
        goals = list(reversed(goals))
    addresses = [int(model.joint(name).qposadr[0]) for name in ARM_JOINTS]
    finger_addresses = [
        int(model.joint("finger_joint1").qposadr[0]),
        int(model.joint("finger_joint2").qposadr[0]),
    ]
    lower = np.asarray([model.joint(name).range[0] for name in ARM_JOINTS], dtype=float)
    upper = np.asarray([model.joint(name).range[1] for name in ARM_JOINTS], dtype=float)
    drop_m, minimum_z = (float(value) for value in payload_clearance)
    scratch = mujoco.MjData(model)
    rejection_log: list[dict[str, object]] = []
    payload_sweep_clearance = float(shadow_envelope_m)

    def set_state(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scratch.qpos[:] = scene_qpos
        for address, value in zip(addresses, q, strict=True):
            scratch.qpos[address] = float(value)
        scratch.qpos[finger_addresses[0]] = finger_q
        scratch.qpos[finger_addresses[1]] = finger_q
        mujoco.mj_forward(model, scratch)
        hand_rotation = scratch.xmat[hand_id].reshape(3, 3).copy()
        hand_xyz = scratch.xpos[hand_id].copy()
        payload_xyz = hand_xyz + hand_rotation @ relative_position
        payload_rotation = hand_rotation @ relative_rotation
        quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quaternion, payload_rotation.reshape(-1))
        scratch.qpos[payload_qpos : payload_qpos + 3] = payload_xyz
        scratch.qpos[payload_qpos + 3 : payload_qpos + 7] = quaternion
        mujoco.mj_forward(model, scratch)
        return hand_xyz, hand_rotation

    def valid(q: np.ndarray) -> bool:
        hand_xyz, hand_rotation = set_state(np.asarray(q, dtype=float))
        if _rotation_deviation_deg(initial_hand_rotation, hand_rotation) > (
            transport_monitor.maximum_tilt_deg + 1e-9
        ):
            return False
        if float(hand_xyz[2]) - drop_m < minimum_z:
            return False
        if _scene_hit(
            model, scratch, ignore_bodies, payload_body=payload_body,
        ) is not None:
            return False
        if _robot_scene_clearance_hit(
            model, scratch, ignore_bodies, clearance_m=0.012,
        ) is not None:
            return False
        return _payload_scene_clearance_hit(
            model,
            scratch,
            payload_body,
            ignore_bodies,
            clearance_m=payload_sweep_clearance,
        ) is None

    start_hand, start_rotation = set_state(q_now)
    start_payload_position = scratch.xpos[payload_id].copy()
    start_payload_rotation = scratch.xmat[payload_id].reshape(3, 3).copy()
    reconstructed_start_payload = (
        start_hand + start_rotation @ relative_position
    )
    start_pose_deg = _rotation_deviation_deg(initial_hand_rotation, start_rotation)
    start_bottom = float(start_hand[2]) - drop_m
    start_scene = _scene_hit(
        model, scratch, ignore_bodies, payload_body=payload_body,
    )
    start_robot = _robot_scene_clearance_hit(
        model, scratch, ignore_bodies, clearance_m=0.012,
    )
    start_payload = _payload_scene_clearance_hit(
        model, scratch, payload_body, ignore_bodies,
        clearance_m=payload_sweep_clearance,
    )
    start_gaps = _component_exact_clearances(
        model, scratch, ignore_bodies, payload_body,
    )
    start_report = {
        "reason": "start_state_check",
        "pose_deviation_deg": float(start_pose_deg),
        "pose_limit_deg": float(transport_monitor.maximum_tilt_deg),
        "payload_bottom_z_m": float(start_bottom),
        "payload_bottom_limit_z_m": float(minimum_z),
        "scene_hit": start_scene,
        "robot_clearance_hit": start_robot,
        "robot_clearance_standard_m": 0.012,
        "payload_clearance_hit": start_payload,
        "payload_sweep_clearance_m": float(payload_sweep_clearance),
        **start_gaps,
        "payload_reconstruction_error_m": float(np.linalg.norm(
            start_payload_position - reconstructed_start_payload
        )),
        "hypothetical_start_from_live_payload_m": float(np.linalg.norm(
            start_payload_position - object_position
        )),
    }
    start_rejected = not valid(q_now)
    if start_rejected:
        rejection_log.append({**start_report, "reason": "start_state_rejected"})
    print(
        "  夹持起点检查："
        f"姿态 {start_pose_deg:.2f}/{transport_monitor.maximum_tilt_deg:.1f}°，"
        f"底高 {start_bottom:.4f}/{minimum_z:.4f} m，"
        f"碰撞={start_scene}，"
        f"机器人间隙={start_robot}（最近 {start_gaps['robot_scene_gap_m']*1000:.1f} mm），"
        f"负载间隙={start_payload}"
        f"（最近 {start_gaps['payload_scene_gap_m']*1000:.1f} mm，"
        f"标准 {payload_sweep_clearance*1000:.1f} mm），"
        f"重建误差 {start_report['payload_reconstruction_error_m']*1000:.2f} mm"
    )

    planning_context = {
        "q_start": [float(value) for value in q_now],
        "object_xyz_m": [float(value) for value in start_payload_position],
        "object_rotation_rowmajor": [
            float(value) for value in start_payload_rotation.reshape(-1)
        ],
        "live_object_xyz_m": [float(value) for value in object_position],
        "hypothetical_start": q_start is not None,
        "T_gripper_object_translation_m": [float(value) for value in relative_position],
        "T_gripper_object_rotation_rowmajor": [
            float(value) for value in relative_rotation.reshape(-1)
        ],
        "planner_seed": int(OMPL_RANDOM_SEED),
        "scene_qpos": [float(value) for value in scene_qpos],
        "drop_m": float(drop_m),
        "minimum_z_m": float(minimum_z),
        "payload_sweep_clearance_m": float(payload_sweep_clearance),
        "maximum_tilt_deg": float(transport_monitor.maximum_tilt_deg),
        "payload_body": payload_body,
        "grasp_origin_m": [float(value) for value in grasp_origin],
        "r_pivot_m": float(r_pivot_m),
        "payload_motion_envelope_m": float(shadow_envelope_m),
    }
    robot.last_transport_planning_context = planning_context
    candidates: list[TransportPlanCandidate] = []
    for index, goal in enumerate(goals):
        if index >= 3 and candidates:
            break
        goal_queries: list[dict[str, object]] = []
        plan = plan_rrt_connect(
            q_now, [goal], lower, upper, valid,
            timeout_s=float(timeout_s),
            interpolation_step_rad=TRANSPORT_OMPL_DENSE_STEP_RAD,
            diagnostics=goal_queries,
        )
        goal_record = {
            "goal_index": index,
            "goal_q": [float(value) for value in goal],
            "queries": goal_queries,
            **planning_context,
        }
        if plan is None:
            reason = classify_transport_goal(goal_queries)
            goal_hand, goal_rotation = set_state(np.asarray(goal, dtype=float))
            goal_parts = _component_exact_clearances(
                model, scratch, ignore_bodies, payload_body,
            )
            goal_gap = float(goal_parts["payload_scene_gap_m"])
            failed_checks: list[str] = []
            if _rotation_deviation_deg(initial_hand_rotation, goal_rotation) > (
                transport_monitor.maximum_tilt_deg + 1e-9
            ):
                failed_checks.append("pose")
            if float(goal_hand[2]) - drop_m < minimum_z:
                failed_checks.append("height")
            if _scene_hit(
                model, scratch, ignore_bodies, payload_body=payload_body,
            ) is not None:
                failed_checks.append("scene")
            if _robot_scene_clearance_hit(
                model, scratch, ignore_bodies, clearance_m=0.012,
            ) is not None:
                failed_checks.append("robot_clearance")
            if goal_gap + 1e-12 < payload_sweep_clearance:
                failed_checks.append("payload_envelope")
            rejection_log.append({
                **goal_record,
                "reason": reason,
                "goal_payload_gap_m": goal_gap,
                "goal_payload_pair": goal_parts.get("payload_scene_pair"),
                "goal_robot_gap_m": float(goal_parts["robot_scene_gap_m"]),
                "goal_shortfall_m": float(payload_sweep_clearance) - goal_gap,
                "goal_failed_checks": failed_checks,
            })
            continue
        execution: list[np.ndarray] = []
        points = plan.as_arrays()
        for start_q, end_q in zip(points, points[1:]):
            count = max(1, int(np.ceil(
                float(np.max(np.abs(end_q - start_q)))
                / TRANSPORT_OMPL_DENSE_STEP_RAD
            )))
            execution.extend(
                start_q + (step / count) * (end_q - start_q)
                for step in range(1, count + 1)
            )
        hand_points: list[np.ndarray] = []
        pose_deviation = 0.0
        minimum_clearance = float("inf")
        for q in [q_now, *execution]:
            hand_xyz, hand_rotation = set_state(q)
            hand_points.append(hand_xyz.copy())
            pose_deviation = max(
                pose_deviation,
                _rotation_deviation_deg(initial_hand_rotation, hand_rotation),
            )
            payload_bottom = float(hand_xyz[2]) - drop_m
            for marker in obstacles or []:
                horizontal = float(np.linalg.norm(
                    hand_xyz[:2] - np.asarray(marker["xy"], dtype=float)
                )) - 0.06
                vertical = payload_bottom - float(marker["top_z"])
                minimum_clearance = min(minimum_clearance, max(horizontal, vertical))
        if not transport_pose_within_limit(
            pose_deviation, transport_monitor.maximum_tilt_deg,
        ):
            rejection_log.append({
                **goal_record,
                "plan_id": plan.plan_id,
                "reason": "pose",
                "maximum_pose_deviation_deg": pose_deviation,
            })
            continue
        joint_points = [q_now, *execution]
        base_turn = max(
            abs(float(q[0] - q_now[0])) for q in joint_points
        )
        wrist_swing = max(
            float(np.max(np.abs(q[4:] - q_now[4:]))) for q in joint_points
        )
        path_length = float(sum(
            np.linalg.norm(b - a) for a, b in zip(joint_points, joint_points[1:])
        ))
        estimated_duration = path_length / 0.20
        tangential_accel = 0.0
        if len(hand_points) >= 3:
            tangential_accel = max(
                float(np.linalg.norm(c - 2.0 * b + a))
                / max((estimated_duration / (len(hand_points) - 1)) ** 2, 1e-6)
                for a, b, c in zip(hand_points, hand_points[1:], hand_points[2:])
            )
        scores = {
            "estimated_duration_s": estimated_duration,
            "maximum_base_turn_rad": base_turn,
            "maximum_wrist_swing_rad": wrist_swing,
            "minimum_clearance_m": (
                1.0 if not np.isfinite(minimum_clearance) else minimum_clearance
            ),
            "maximum_pose_deviation_deg": pose_deviation,
            "maximum_tangential_acceleration_m_s2": tangential_accel,
            "payload_sweep_clearance_m": payload_sweep_clearance,
        }
        velocity, acceleration, jerk, minimum_duration = _executed_trajectory_limits(
            transport_monitor.motion_profile, transport_monitor.motion_scale,
        )
        trajectory = sample_executed_joint_trajectory(
            joint_points,
            velocity_limits=velocity,
            acceleration_limits=acceleration,
            jerk_limits=jerk,
            minimum_duration=minimum_duration,
            sample_step_rad=TRANSPORT_OMPL_DENSE_STEP_RAD,
        )
        exact_minimum = float("inf")
        component_tracker = _blank_component_tracker()
        trajectory_hit: str | None = None
        ruckig_category: str | None = None
        if trajectory is None:
            trajectory_hit = "ruckig_trajectory_unavailable"
            ruckig_category = "ruckig_trajectory_unavailable"
        else:
            for sample_q in trajectory:
                hand_xyz, hand_rotation = set_state(sample_q)
                pose_now = _rotation_deviation_deg(initial_hand_rotation, hand_rotation)
                pose_deviation = max(pose_deviation, pose_now)
                if pose_now > transport_monitor.maximum_tilt_deg + 1e-9:
                    trajectory_hit = "payload_pose_deviation"
                    ruckig_category = "pose"
                elif float(hand_xyz[2]) - drop_m < minimum_z:
                    trajectory_hit = "payload_height"
                    ruckig_category = "height"
                else:
                    scene_hit = _scene_hit(
                        model, scratch, ignore_bodies, payload_body=payload_body,
                    )
                    sample_parts = _component_exact_clearances(
                        model, scratch, ignore_bodies, payload_body,
                    )
                    release = transport_release_reason(
                        float(sample_parts["robot_scene_gap_m"]),
                        float(sample_parts["payload_scene_gap_m"]),
                        payload_sweep_clearance,
                    )
                    if scene_hit is not None:
                        trajectory_hit = scene_hit
                        ruckig_category = "scene"
                    elif release is not None:
                        trajectory_hit = release
                        ruckig_category = release
                _track_components(
                    component_tracker,
                    _component_exact_clearances(
                        model, scratch, ignore_bodies, payload_body,
                    ),
                )
                exact_minimum = min(
                    exact_minimum,
                    _minimum_exact_clearance_m(
                        model, scratch, ignore_bodies, payload_body,
                    ),
                )
                if trajectory_hit is not None:
                    break
        scores.update(_component_score_fields(component_tracker))
        apply_safety_slack(scores, payload_sweep_clearance)
        component_details = _component_detail_fields(component_tracker)
        scores["exact_minimum_clearance_m"] = (
            None if not np.isfinite(exact_minimum) else float(exact_minimum)
        )
        scores["maximum_pose_deviation_deg"] = pose_deviation
        if ruckig_category is None and (
            not np.isfinite(exact_minimum) or float(exact_minimum) < 0.0
        ):
            ruckig_category = "negative_exact_gap"
        if trajectory_hit is not None or not execution_sample_released(
            scene_hit=trajectory_hit,
            robot_clearance_hit=None,
            payload_clearance_hit=None,
            exact_minimum_clearance_m=(
                -1.0 if not np.isfinite(exact_minimum) else float(exact_minimum)
            ),
        ):
            rejection_log.append({
                **goal_record,
                "plan_id": plan.plan_id,
                "reason": classify_transport_goal(
                    goal_queries,
                    ruckig_category=ruckig_category,
                    exact_minimum_clearance_m=scores["exact_minimum_clearance_m"],
                ),
                "detail": trajectory_hit or "negative_exact_clearance",
                "exact_minimum_clearance_m": scores["exact_minimum_clearance_m"],
                "exact_robot_clearance_m": scores.get("exact_robot_clearance_m"),
                "exact_payload_clearance_m": scores.get("exact_payload_clearance_m"),
                "approximate_minimum_clearance_m": scores["minimum_clearance_m"],
                **component_details,
            })
            continue
        candidates.append(TransportPlanCandidate(
            candidate_id=plan.plan_id,
            joint_waypoints=tuple(tuple(float(v) for v in q) for q in points),
            execution_points=tuple(tuple(float(v) for v in q) for q in execution),
            hand_waypoints=tuple(tuple(float(v) for v in point) for point in hand_points),
            scores=scores,
            planning_time_s=plan.planning_time_s,
            validated=True,
            details=component_details,
        ))
    total = time.perf_counter() - begun
    for entry in rejection_log:
        entry["planning_batch_time_s"] = total
    return candidates, rejection_log, total


def public_transport_candidate(candidate: TransportPlanCandidate) -> dict[str, object]:
    """Manager-facing row, including the two gaps and the slack in millimetres."""
    robot_gap = candidate.scores.get("exact_robot_clearance_m")
    payload_gap = candidate.scores.get("exact_payload_clearance_m")
    slack = candidate.scores.get("safety_slack_m")
    return {
        "candidate_id": candidate.candidate_id,
        "validated": bool(candidate.validated),
        **candidate.scores,
        "robot_gap_mm": None if robot_gap is None else float(robot_gap) * 1000.0,
        "payload_gap_mm": None if payload_gap is None else float(payload_gap) * 1000.0,
        "safety_slack_mm": None if slack is None else float(slack) * 1000.0,
        "exact_robot_pair": candidate.details.get("exact_robot_pair"),
        "exact_payload_pair": candidate.details.get("exact_payload_pair"),
        "planning_time_s": float(candidate.planning_time_s),
    }


def _raise_carried_transport_once(
    robot: PandaController,
    model: mujoco.MjModel,
    target_xyz: np.ndarray,
    *,
    yaw_rad: float,
    ignore_bodies: set[str],
    payload_clearance: tuple[float, float],
    obstacles: list[dict] | None,
    transport_monitor: TransportMonitor,
    preference: str,
) -> tuple[dict[str, object], TransportPlanCandidate | None, float]:
    """One ladder of raised midpoints. A miss does not start a second ladder."""
    lock = robot.data_lock if robot.data_lock is not None else nullcontext()
    with lock:
        start_hand = robot.data.xpos[model.body("hand").id].copy()
    note: dict[str, object] = {
        "raise_triggered": True,
        "raise_delta_m": None,
        "raise_result": "no_release",
    }
    elapsed = 0.0
    for delta in raise_deltas_m():
        midpoint = raise_midpoint_xyz(start_hand, target_xyz, delta)
        print(f"→ 直连候选全灭，分段改试中间抬高 {delta * 1000:.0f} mm")
        firsts, first_rejected, elapsed_a = _plan_ompl_transport_candidates(
            robot, model, midpoint,
            yaw_rad=yaw_rad,
            ignore_bodies=ignore_bodies,
            payload_clearance=payload_clearance,
            obstacles=obstacles,
            transport_monitor=transport_monitor,
            timeout_s=TRANSPORT_OMPL_TIMEOUT_S,
            reverse_goals=False,
        )
        elapsed += float(elapsed_a)
        released_first = [item for item in firsts if item.validated]
        if not released_first:
            note["raise_delta_m"] = delta
            note["raise_result"] = (
                str(first_rejected[-1]["reason"]) if first_rejected else "ompl_no_exact"
            )
            continue
        first = choose_transport_candidate(released_first, preference)
        assert first is not None
        seconds, second_rejected, elapsed_b = _plan_ompl_transport_candidates(
            robot, model, np.asarray(target_xyz, dtype=float),
            yaw_rad=yaw_rad,
            ignore_bodies=ignore_bodies,
            payload_clearance=payload_clearance,
            obstacles=obstacles,
            transport_monitor=transport_monitor,
            timeout_s=TRANSPORT_OMPL_TIMEOUT_S,
            reverse_goals=False,
            q_start=np.asarray(first.joint_waypoints[-1], dtype=float),
        )
        elapsed += float(elapsed_b)
        released_second = [item for item in seconds if item.validated]
        if not released_second:
            note["raise_delta_m"] = delta
            note["raise_result"] = (
                str(second_rejected[-1]["reason"]) if second_rejected else "ompl_no_exact"
            )
            continue
        second = choose_transport_candidate(released_second, preference)
        assert second is not None
        note["raise_delta_m"] = delta
        note["raise_result"] = "released"
        return note, join_raised_transport(first, second, delta), elapsed
    note["raise_delta_m"] = RAISE_TRANSPORT_CAP_M
    if note["raise_result"] == "no_release":
        note["raise_result"] = "payload_gap_below_sweep"
    return note, None, elapsed


def _archived_hand_polylines(model: mujoco.MjModel) -> list[list[np.ndarray]]:
    """Forward-kinematics of stored arm waypoints. The joints are not executed."""
    polylines: list[list[np.ndarray]] = []
    addresses = [int(model.joint(name).qposadr[0]) for name in ARM_JOINTS]
    hand_id = model.body("hand").id
    for route in _load_transport_route_entries():
        waypoints = route.get("joint_waypoints")
        if not isinstance(waypoints, list) or len(waypoints) < 3:
            continue
        scratch = mujoco.MjData(model)
        points: list[np.ndarray] = []
        for waypoint in waypoints:
            q = np.asarray(waypoint, dtype=float).reshape(-1)
            if q.shape != (7,):
                points = []
                break
            for address, value in zip(addresses, q, strict=True):
                scratch.qpos[address] = float(value)
            mujoco.mj_forward(model, scratch)
            points.append(scratch.xpos[hand_id].copy())
        if len(points) >= 3:
            polylines.append(points)
    return polylines


def _plan_segmented_ompl_corridors(
    robot: PandaController,
    model: mujoco.MjModel,
    target_xyz: np.ndarray,
    *,
    yaw_rad: float,
    ignore_bodies: set[str],
    payload_clearance: tuple[float, float],
    obstacles: list[dict] | None,
    transport_monitor: TransportMonitor,
    preference: str,
    via_points: list[np.ndarray] | None = None,
    source: str = "geometry",
) -> tuple[dict[str, object], TransportPlanCandidate | None, float]:
    """Try corridor sub-goals as independently validated OMPL segments."""
    lock = robot.data_lock if robot.data_lock is not None else nullcontext()
    with lock:
        start_hand = _hand_xyz(model, robot.data)
    vias = (
        list(via_points)
        if via_points is not None
        else transport_corridor_waypoints(start_hand, target_xyz)
    )
    note: dict[str, object] = {
        "attempted": True,
        "source": source,
        "selected_corridor": None,
        "result": "no_validated_corridor",
        "attempts": [],
    }
    elapsed = 0.0
    for corridor_index, via in enumerate(vias):
        print(
            f"→ OMPL 分段{source} {corridor_index + 1}/{len(vias)}："
            f"中继 {np.round(via, 3).tolist()}"
        )
        firsts, first_rejected, elapsed_a = _plan_ompl_transport_candidates(
            robot,
            model,
            via,
            yaw_rad=yaw_rad,
            ignore_bodies=ignore_bodies,
            payload_clearance=payload_clearance,
            obstacles=obstacles,
            transport_monitor=transport_monitor,
            timeout_s=TRANSPORT_OMPL_TIMEOUT_S,
            reverse_goals=bool(corridor_index),
        )
        elapsed += float(elapsed_a)
        first = choose_transport_candidate(firsts, preference)
        attempt: dict[str, object] = {
            "corridor_index": corridor_index,
            "via_xyz_m": [float(value) for value in via],
            "first_rejections": first_rejected,
            "first_validated": first is not None,
        }
        if first is None:
            note["attempts"].append(attempt)  # type: ignore[union-attr]
            continue
        seconds, second_rejected, elapsed_b = _plan_ompl_transport_candidates(
            robot,
            model,
            np.asarray(target_xyz, dtype=float),
            yaw_rad=yaw_rad,
            ignore_bodies=ignore_bodies,
            payload_clearance=payload_clearance,
            obstacles=obstacles,
            transport_monitor=transport_monitor,
            timeout_s=TRANSPORT_OMPL_TIMEOUT_S,
            reverse_goals=not bool(corridor_index),
            q_start=np.asarray(first.joint_waypoints[-1], dtype=float),
        )
        elapsed += float(elapsed_b)
        second = choose_transport_candidate(seconds, preference)
        attempt["second_rejections"] = second_rejected
        attempt["second_validated"] = second is not None
        note["attempts"].append(attempt)  # type: ignore[union-attr]
        if second is None:
            continue
        joined = join_raised_transport(first, second, 0.0)
        joined = replace(
            joined,
            candidate_id=(
                f"{source}{corridor_index}:"
                f"{route_root_id(first.candidate_id)}+"
                f"{route_root_id(second.candidate_id)}"
            ),
            details={
                **joined.details,
                "raise_triggered": False,
                "raise_delta_m": 0.0,
                "corridor_index": corridor_index,
                "corridor_via_xyz_m": [float(value) for value in via],
            },
        )
        note["selected_corridor"] = corridor_index
        note["result"] = "validated"
        return note, joined, elapsed
    return note, None, elapsed


def _execute_ompl_transport_candidate(
    candidate: TransportPlanCandidate,
    robot: PandaController,
    model: mujoco.MjModel,
    target_xyz: np.ndarray,
    *,
    ignore_bodies: set[str],
    transport_monitor: TransportMonitor,
    payload_body_name: str,
    plan_review: Callable[[dict[str, object]], bool] | None,
    plan_stage: str,
    route_policy: str,
) -> bool:
    """Execute one already validated payload-aware OMPL candidate."""
    if not candidate.validated:
        robot.last_abort_reason = "unvalidated_transport_candidate"
        return False
    hand_points = [np.asarray(point, dtype=float) for point in candidate.hand_waypoints]
    robot.last_plan_brief = {
        **_summarize_plan(
            hand_points,
            payload_drop_m=float(
                _hand_xyz(model, robot.data)[2]
                - _body_bottom_z(model, robot.data, payload_body_name)
            ),
            route_policy=route_policy,
            plan_id=candidate.candidate_id,
            stage=plan_stage,
        ),
        "planner_used": "ompl",
        "candidate_scores": dict(candidate.scores),
        "planning_time_s": float(candidate.planning_time_s),
        "validated": True,
    }
    if plan_review is not None and not plan_review(robot.last_plan_brief):
        if not robot.last_abort_reason:
            robot.last_abort_reason = "plan_brief_rejected"
        return False

    def execution_monitor() -> str | None:
        reason = transport_monitor.check()
        if reason is not None:
            return reason
        transport_monitor.runtime_collision_checks += 1
        hit = _scene_hit(
            model, robot.data, ignore_bodies,
            payload_body=payload_body_name,
            contact_distance_threshold_m=0.001,
        )
        if hit is None:
            hit = _payload_scene_clearance_hit(
                model, robot.data, payload_body_name, ignore_bodies,
                clearance_m=0.004,
            )
        q_actual = _arm_q(model, robot.data)
        q_command = np.asarray(robot.data.ctrl[:7], dtype=float)
        transport_monitor.note_execution_clearance(
            _component_exact_clearances(
                model, robot.data, ignore_bodies, payload_body_name,
            ),
            float(np.max(np.abs(q_actual - q_command))),
        )
        return None if hit is None else f"runtime_collision:{hit}"

    points = candidate.execution_arrays()
    previous = _arm_q(model, robot.data)
    for index, q_next in enumerate(points):
        final_segment = index == len(points) - 1
        transport_monitor.begin_segment()
        moved = robot.move_to_ruckig(
            q_next,
            profile=transport_monitor.motion_profile,
            monitor=execution_monitor,
            settle=final_segment or abs(float(q_next[0] - previous[0])) > 0.35,
            motion_scale=transport_monitor.motion_scale,
        )
        if not moved:
            return False
        if transport_monitor.consume_preload_request():
            if not _refresh_payload_preload(
                robot, model, transport_monitor.body_name, transport_monitor,
            ):
                return False
        previous = q_next
    final_error = float(np.linalg.norm(_hand_xyz(model, robot.data) - target_xyz))
    if final_error > 0.025:
        robot.last_abort_reason = "ompl_transport_final_error"
        return False
    return True


def _try_ompl_move(
    robot: PandaController,
    model: mujoco.MjModel,
    target_xyz: np.ndarray,
    *,
    yaw_rad: float,
    ignore_bodies: set[str],
) -> bool | None:
    """Use OMPL to switch IK branches; None means fall back to local routing."""
    lock = robot.data_lock if robot.data_lock is not None else nullcontext()
    with lock:
        q_now = _arm_q(model, robot.data)
        scene_qpos = robot.data.qpos.copy()
        finger_q = float(robot.data.qpos[
            int(model.joint("finger_joint1").qposadr[0])
        ])
    goals = _ik_goal_candidates(model, target_xyz, q_now, yaw_rad)
    if not goals:
        return None
    addresses = [int(model.joint(name).qposadr[0]) for name in ARM_JOINTS]
    finger_addresses = [
        int(model.joint("finger_joint1").qposadr[0]),
        int(model.joint("finger_joint2").qposadr[0]),
    ]
    lower = np.empty(7, dtype=float)
    upper = np.empty(7, dtype=float)
    for index, name in enumerate(ARM_JOINTS):
        joint = model.joint(name)
        lower[index], upper[index] = joint.range
    scratch = mujoco.MjData(model)

    def valid(q: np.ndarray) -> bool:
        scratch.qpos[:] = scene_qpos
        for address, value in zip(addresses, q, strict=True):
            scratch.qpos[address] = float(value)
        scratch.qpos[finger_addresses[0]] = finger_q
        scratch.qpos[finger_addresses[1]] = finger_q
        mujoco.mj_forward(model, scratch)
        if _scene_hit(model, scratch, ignore_bodies) is not None:
            return False
        return _robot_scene_clearance_hit(
            model, scratch, ignore_bodies, clearance_m=0.012
        ) is None

    plan = plan_rrt_connect(
        q_now, goals, lower, upper, valid,
        timeout_s=1.6, interpolation_step_rad=0.06,
    )
    if plan is None:
        print("  OMPL RRTConnect 未找到安全的 IK 分支连接，回退局部几何候选")
        return None
    print(
        f"→ {plan.planner} 生成 {plan.plan_id}：{len(plan.waypoints)} 个关键路点，"
        f"关节路径 {plan.path_length_rad:.2f} rad，规划 {plan.planning_time_s:.2f} s"
    )
    # Ruckig is time-optimal in each joint and does not promise that a long
    # multi-axis move follows OMPL's straight joint-space segment.  Executing
    # only the 3-4 shortcut vertices therefore bowed into the orange cube even
    # though the geometric path was valid.  Densify first so every dynamic
    # segment stays in a narrow tube around the checked path.
    execution_points: list[np.ndarray] = []
    plan_points = plan.as_arrays()
    for start, end in zip(plan_points, plan_points[1:]):
        count = max(1, int(np.ceil(float(np.max(np.abs(end - start))) / 0.035)))
        execution_points.extend(
            start + (index / count) * (end - start)
            for index in range(1, count + 1)
        )
    print(
        "  ✓ OMPL 路径已逐状态复核全身碰撞；"
        f"细分为 {len(execution_points)} 个执行点后交给 Ruckig"
    )

    def execution_monitor() -> str | None:
        hit = _scene_hit(
            model,
            robot.data,
            ignore_bodies,
            # Stop at first contact, before a movable obstacle is visibly
            # pushed.  Planning remains penetration based to avoid rejecting
            # harmless numerical near-contact in scratch states.
            contact_distance_threshold_m=0.001,
        )
        if hit is None:
            hit = _robot_scene_clearance_hit(
                model, robot.data, ignore_bodies, clearance_m=0.004
            )
        return None if hit is None else f"runtime_collision:{hit}"

    for q_next in execution_points:
        if not robot.move_to_ruckig(
            q_next,
            profile="normal",
            monitor=execution_monitor,
        ):
            print(f"  ✗ Ruckig 执行停止：{robot.last_abort_reason or 'controller_stopped'}")
            return False
    final_error = float(np.linalg.norm(_hand_xyz(model, robot.data) - target_xyz))
    if final_error > 0.025:
        print(f"  ✗ OMPL 执行后末端误差 {final_error * 1000:.0f} mm")
        return False
    return True


def _summarize_plan(
    waypoints: list[np.ndarray] | np.ndarray,
    *,
    durations: list[float] | None = None,
    obstacles: list[dict] | None = None,
    payload_drop_m: float | None = None,
    peak_force_n: float = 0.0,
    route_policy: str = "auto",
    crosses_base: bool = False,
    plan_id: str = "",
    stage: str = "",
) -> dict[str, object]:
    """Human/LLM-readable summary of a validated hand path (a PlanBrief).

    Pure geometry over the hand waypoints and the approximate depth-marked
    obstacles; it describes the plan and never changes it.  ``payload_drop_m`` is the
    distance from the hand origin to the carried object's bottom when a
    payload is held, otherwise the finger pads are the lowest point.
    """
    points = [np.asarray(point, dtype=float) for point in waypoints]
    if not points:
        return {"stage": stage, "plan_id": plan_id, "path_length_m": 0.0, "key_waypoints": []}
    path_length = float(sum(
        np.linalg.norm(points[index + 1] - points[index]) for index in range(len(points) - 1)
    ))
    lowest_offset = float(payload_drop_m) if payload_drop_m is not None else FINGER_PAD_BELOW_HAND
    bottoms = [float(point[2]) - lowest_offset for point in points]
    # The first waypoint is where the hand already is (e.g. just after the
    # trial lift); the travel clearance is what the plan itself adds.
    start_bottom_clearance = float(bottoms[0])
    min_table_clearance = float(min(bottoms[1:])) if len(bottoms) > 1 else start_bottom_clearance
    min_obstacle_clearance: float | None = None
    for point, bottom in zip(points, bottoms, strict=True):
        for marker in obstacles or []:
            marker_xy = np.asarray(marker["xy"], dtype=float)
            top = float(marker["top_z"])
            horizontal = float(np.linalg.norm(point[:2] - marker_xy)) - 0.06
            vertical = bottom - top
            clearance = max(horizontal, vertical)
            if min_obstacle_clearance is None or clearance < min_obstacle_clearance:
                min_obstacle_clearance = clearance
    highest = max(range(len(points)), key=lambda index: float(points[index][2]))
    picks = sorted({0, highest, len(points) - 1})
    key_waypoints = [np.round(points[index], 3).tolist() for index in picks][:8]
    risk_flags: list[str] = []
    if crosses_base:
        risk_flags.append("crosses_base")
    if max(float(point[2]) for point in points) >= 0.62:
        risk_flags.append("near_ceiling")
    if payload_drop_m is not None:
        risk_flags.append("payload_held")
    if route_policy == "high_clearance":
        risk_flags.append("high_clearance_route")
    if min_obstacle_clearance is not None and min_obstacle_clearance < 0.03:
        risk_flags.append("tight_obstacle_clearance")
    return {
        "stage": stage,
        "plan_id": plan_id,
        "path_length_m": round(path_length, 3),
        "min_obstacle_clearance_m": (
            None if min_obstacle_clearance is None else round(float(min_obstacle_clearance), 3)
        ),
        "min_obstacle_clearance_is_approximate": True,
        "min_obstacle_clearance_semantics": "approximate_depth_marker_clearance",
        "start_bottom_clearance_m": round(start_bottom_clearance, 3),
        "min_table_clearance_m": round(min_table_clearance, 3),
        "predicted_peak_force_n": round(float(peak_force_n), 2),
        "predicted_duration_s": round(float(sum(durations or [])), 2),
        "waypoint_count": len(points),
        "key_waypoints": key_waypoints,
        "route_policy": route_policy,
        "risk_flags": risk_flags,
    }


def _carried_clearance_report(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ignore_bodies: set[str],
    payload_body: str,
    payload_sweep_clearance: float,
    initial_hand_rotation: np.ndarray,
    drop_m: float,
    minimum_z: float,
    maximum_tilt_deg: float,
) -> dict[str, object]:
    """The same five start checks OMPL uses, read from an already-forwarded state."""
    hand_id = model.body("hand").id
    hand_xyz = data.xpos[hand_id].copy()
    hand_rotation = data.xmat[hand_id].reshape(3, 3).copy()
    pose_deg = _rotation_deviation_deg(initial_hand_rotation, hand_rotation)
    bottom = float(hand_xyz[2]) - float(drop_m)
    scene_hit = _scene_hit(
        model, data, ignore_bodies, payload_body=payload_body,
    )
    robot_hit = _robot_scene_clearance_hit(
        model, data, ignore_bodies, clearance_m=0.012,
    )
    payload_hit = _payload_scene_clearance_hit(
        model, data, payload_body, ignore_bodies,
        clearance_m=payload_sweep_clearance,
    )
    gaps = _component_exact_clearances(
        model, data, ignore_bodies, payload_body,
    )
    return {
        "hand_z_m": float(hand_xyz[2]),
        "pose_deviation_deg": float(pose_deg),
        "pose_limit_deg": float(maximum_tilt_deg),
        "payload_bottom_z_m": float(bottom),
        "payload_bottom_limit_z_m": float(minimum_z),
        "scene_hit": scene_hit,
        "robot_clearance_hit": robot_hit,
        "payload_clearance_hit": payload_hit,
        "payload_scene_gap_m": float(gaps["payload_scene_gap_m"]),
        "payload_scene_pair": gaps["payload_scene_pair"],
        "robot_scene_gap_m": float(gaps["robot_scene_gap_m"]),
    }


def _preview_carried_clearance(
    model: mujoco.MjModel,
    scratch: mujoco.MjData,
    q: np.ndarray,
    *,
    scene_qpos: np.ndarray,
    addresses: list[int],
    finger_addresses: list[int],
    finger_q: float,
    relative_position: np.ndarray,
    relative_rotation: np.ndarray,
    payload_qpos: int,
    ignore_bodies: set[str],
    payload_body: str,
    payload_sweep_clearance: float,
    initial_hand_rotation: np.ndarray,
    drop_m: float,
    minimum_z: float,
    maximum_tilt_deg: float,
) -> dict[str, object]:
    """Replay one joint pose with the hand-relative payload, then measure it."""
    hand_id = model.body("hand").id
    scratch.qpos[:] = scene_qpos
    for address, value in zip(addresses, q, strict=True):
        scratch.qpos[address] = float(value)
    scratch.qpos[finger_addresses[0]] = finger_q
    scratch.qpos[finger_addresses[1]] = finger_q
    mujoco.mj_forward(model, scratch)
    hand_rotation = scratch.xmat[hand_id].reshape(3, 3).copy()
    hand_xyz = scratch.xpos[hand_id].copy()
    payload_xyz = hand_xyz + hand_rotation @ relative_position
    payload_rotation = hand_rotation @ relative_rotation
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, payload_rotation.reshape(-1))
    scratch.qpos[payload_qpos : payload_qpos + 3] = payload_xyz
    scratch.qpos[payload_qpos + 3 : payload_qpos + 7] = quaternion
    mujoco.mj_forward(model, scratch)
    return _carried_clearance_report(
        model, scratch,
        ignore_bodies=ignore_bodies,
        payload_body=payload_body,
        payload_sweep_clearance=payload_sweep_clearance,
        initial_hand_rotation=initial_hand_rotation,
        drop_m=drop_m,
        minimum_z=minimum_z,
        maximum_tilt_deg=maximum_tilt_deg,
    )


def _separate_payload_vertically(
    robot: PandaController,
    model: mujoco.MjModel,
    *,
    yaw_rad: float,
    ignore_bodies: set[str],
    payload_body: str,
    payload_clearance: tuple[float, float],
    transport_monitor: TransportMonitor,
    goal_xyz: np.ndarray | None = None,
) -> dict[str, object]:
    """Separate until height and payload sweep are met, or the cap is hit.

    Straight up is tried first. A rejected step tries the other rising
    directions and the other IK branches of that same step. The step is
    15 mm and the vertical cap is 30 mm. Neither number changes a clearance
    standard.
    """
    lift_step_m = 0.015
    lift_cap_m = 0.030
    drop_m, minimum_z = (float(value) for value in payload_clearance)
    lock = robot.data_lock if robot.data_lock is not None else nullcontext()
    with lock:
        hand_id = model.body("hand").id
        sweep = payload_sweep_clearance_m(
            model, robot.data, payload_body, transport_monitor,
        )
        initial_hand_rotation = robot.data.xmat[hand_id].reshape(3, 3).copy()
        start_hand_z = float(robot.data.xpos[hand_id][2])
        report = _carried_clearance_report(
            model, robot.data,
            ignore_bodies=ignore_bodies,
            payload_body=payload_body,
            payload_sweep_clearance=sweep,
            initial_hand_rotation=initial_hand_rotation,
            drop_m=drop_m,
            minimum_z=minimum_z,
            maximum_tilt_deg=transport_monitor.maximum_tilt_deg,
        )
    mode = pre_ompl_separation_mode(
        scene_hit=report["scene_hit"],
        robot_clearance_hit=report["robot_clearance_hit"],
        pose_deviation_deg=float(report["pose_deviation_deg"]),
        pose_limit_deg=float(report["pose_limit_deg"]),
        payload_bottom_z_m=float(report["payload_bottom_z_m"]),
        payload_bottom_limit_z_m=float(report["payload_bottom_limit_z_m"]),
        payload_clearance_hit=report["payload_clearance_hit"],
    )
    samples = [{**report, "lift_m": 0.0}]
    record: dict[str, object] = {
        "attempted": mode == "lift",
        "mode": mode,
        "reached_sweep": mode == "ready",
        "stop_reason": "start_already_valid" if mode == "ready" else (
            "start_blocked" if mode == "blocked" else None
        ),
        "lift_m": 0.0,
        "lift_step_m": lift_step_m,
        "lift_cap_m": lift_cap_m,
        "escape_pose": None,
        "payload_sweep_clearance_m": float(sweep),
        "samples": samples,
    }
    if mode != "lift":
        return record
    print(
        "→ OMPL 前脱离：负载底高/扫掠尚未全部达标，"
        f"其余检查已过，最多上提 {lift_cap_m * 1000:.0f} mm；"
        "直上不行再试斜后上和其它关节分支"
    )
    previous_gap = float(report["payload_scene_gap_m"])
    payload_id = model.body(payload_body).id
    payload_joint = int(model.body_jntadr[payload_id])
    payload_qpos = int(model.jnt_qposadr[payload_joint])
    addresses = [int(model.joint(name).qposadr[0]) for name in ARM_JOINTS]
    finger_addresses = [
        int(model.joint("finger_joint1").qposadr[0]),
        int(model.joint("finger_joint2").qposadr[0]),
    ]
    scratch = mujoco.MjData(model)
    while True:
        lifted = float(record["lift_m"])
        status = pre_ompl_separation_status(
            lifted_m=lifted,
            lift_cap_m=lift_cap_m,
            payload_clearance_hit=report["payload_clearance_hit"],
            payload_bottom_z_m=float(report["payload_bottom_z_m"]),
            payload_bottom_limit_z_m=float(report["payload_bottom_limit_z_m"]),
        )
        if status == "sweep_met":
            record["reached_sweep"] = True
            record["stop_reason"] = "sweep_met"
            return record
        if status == "ceiling":
            record["reached_sweep"] = False
            record["stop_reason"] = "ceiling"
            return record
        step = min(lift_step_m, lift_cap_m - lifted)
        with lock:
            hand = _hand_xyz(model, robot.data)
            scene_qpos = robot.data.qpos.copy()
            finger_q = float(robot.data.qpos[finger_addresses[0]])
            hand_rotation = robot.data.xmat[hand_id].reshape(3, 3).copy()
            object_rotation = robot.data.xmat[payload_id].reshape(3, 3).copy()
            relative_position = hand_rotation.T @ (
                robot.data.xpos[payload_id] - robot.data.xpos[hand_id]
            )
            relative_rotation = hand_rotation.T @ object_rotation
            q_now = _arm_q(model, robot.data)
        lower = np.asarray(
            [model.joint(name).range[0] for name in ARM_JOINTS], dtype=float,
        )
        upper = np.asarray(
            [model.joint(name).range[1] for name in ARM_JOINTS], dtype=float,
        )
        bearing = float(np.arctan2(hand[1], hand[0]))
        last_preview: dict[str, object] | None = None
        ik_error = "no_branch"
        risen = False
        pose_name = "up"
        for pose_name, offset in transport_escape_offsets_m(hand, goal_xyz, step):
            with lock:
                hand_now = _hand_xyz(model, robot.data)
                q_now = _arm_q(model, robot.data)
                scene_qpos = robot.data.qpos.copy()
                finger_q = float(robot.data.qpos[finger_addresses[0]])
                hand_rotation = robot.data.xmat[hand_id].reshape(3, 3).copy()
                object_rotation = robot.data.xmat[payload_id].reshape(3, 3).copy()
                relative_position = hand_rotation.T @ (
                    robot.data.xpos[payload_id] - robot.data.xpos[hand_id]
                )
                relative_rotation = hand_rotation.T @ object_rotation
            target = hand_now + offset
            q_next: np.ndarray | None = None
            for seed in ik_branch_seeds(q_now, lower, upper, bearing):
                try:
                    solved = solve_hand_ik(
                        model, target, seed=seed, yaw_rad=yaw_rad,
                    )
                except RuntimeError as error:
                    ik_error = str(error)
                    continue
                preview = _preview_carried_clearance(
                    model, scratch, solved,
                    scene_qpos=scene_qpos,
                    addresses=addresses,
                    finger_addresses=finger_addresses,
                    finger_q=finger_q,
                    relative_position=relative_position,
                    relative_rotation=relative_rotation,
                    payload_qpos=payload_qpos,
                    ignore_bodies=ignore_bodies,
                    payload_body=payload_body,
                    payload_sweep_clearance=sweep,
                    initial_hand_rotation=initial_hand_rotation,
                    drop_m=drop_m,
                    minimum_z=minimum_z,
                    maximum_tilt_deg=transport_monitor.maximum_tilt_deg,
                )
                last_preview = preview
                if not pre_ompl_step_allowed(
                    scene_hit=preview["scene_hit"],
                    robot_clearance_hit=preview["robot_clearance_hit"],
                    payload_gap_m=float(preview["payload_scene_gap_m"]),
                    previous_payload_gap_m=previous_gap,
                    pose_deviation_deg=float(preview["pose_deviation_deg"]),
                    pose_limit_deg=float(preview["pose_limit_deg"]),
                    payload_bottom_z_m=float(preview["payload_bottom_z_m"]),
                    payload_bottom_limit_z_m=float(preview["payload_bottom_limit_z_m"]),
                    payload_limit_m=float(sweep),
                ):
                    continue
                if _segment_hit(
                    model, scene_qpos, q_now, solved, ignore_bodies,
                ) is not None:
                    continue
                q_next = solved
                break
            if q_next is None:
                continue
            span = float(np.max(np.abs(q_next - q_now)))
            if not robot.move_to(
                q_next,
                float(np.clip(span / 0.6, 0.25, 0.7)),
                profile="precision",
            ):
                continue
            if not robot.hold(0.08):
                record["stop_reason"] = "hold_failed"
                return record
            with lock:
                report = _carried_clearance_report(
                    model, robot.data,
                    ignore_bodies=ignore_bodies,
                    payload_body=payload_body,
                    payload_sweep_clearance=sweep,
                    initial_hand_rotation=initial_hand_rotation,
                    drop_m=drop_m,
                    minimum_z=minimum_z,
                    maximum_tilt_deg=transport_monitor.maximum_tilt_deg,
                )
            lifted_before = float(record["lift_m"])
            lifted = float(report["hand_z_m"]) - start_hand_z
            if lifted <= lifted_before + 1e-4:
                continue
            record["lift_m"] = lifted
            record["escape_pose"] = pose_name
            risen = True
            break
        if not risen:
            if last_preview is not None:
                samples.append({**last_preview, "lift_m": lifted, "accepted": False})
                print(
                    "  直上、斜后上和其它关节分支都未通过现有门："
                    f"碰撞={last_preview.get('scene_hit')}，"
                    f"机器人间隙={last_preview.get('robot_clearance_hit')}，"
                    f"负载间隙={last_preview.get('payload_clearance_hit')}，"
                    f"姿态={float(last_preview.get('pose_deviation_deg') or 0.0):.1f}°"
                )
            record["stop_reason"] = "step_rejected"
            record["reject_detail"] = (
                "no_safe_escape" if last_preview is not None else f"ik:{ik_error}"
            )
            return record
        samples.append({**report, "lift_m": lifted, "accepted": True})
        print(
            f"  脱离 {pose_name} {lifted * 1000:.1f} mm：负载间隙 "
            f"{float(report['payload_scene_gap_m']) * 1000:.1f} mm"
            f"（{report['payload_scene_pair']}），"
            f"扫掠命中={report['payload_clearance_hit']}"
        )
        height_still_short = (
            report["scene_hit"] is None
            and report["robot_clearance_hit"] is None
            and float(report["pose_deviation_deg"])
            <= float(report["pose_limit_deg"]) + 1e-9
            and float(report["payload_bottom_z_m"]) + 1e-9
            < float(report["payload_bottom_limit_z_m"])
            and float(record["lift_m"]) + 1e-9 < lift_cap_m
        )
        if height_still_short:
            previous_gap = float(report["payload_scene_gap_m"])
            continue
        if not pre_ompl_step_allowed(
            scene_hit=report["scene_hit"],
            robot_clearance_hit=report["robot_clearance_hit"],
            payload_gap_m=float(report["payload_scene_gap_m"]),
            previous_payload_gap_m=previous_gap,
            pose_deviation_deg=float(report["pose_deviation_deg"]),
            pose_limit_deg=float(report["pose_limit_deg"]),
            payload_bottom_z_m=float(report["payload_bottom_z_m"]),
            payload_bottom_limit_z_m=float(report["payload_bottom_limit_z_m"]),
            payload_limit_m=float(sweep),
        ):
            print(
                "  脱离后的实测没有通过现有门："
                f"碰撞={report['scene_hit']}，"
                f"机器人间隙={report['robot_clearance_hit']}，"
                f"底高={float(report['payload_bottom_z_m']):.4f}"
                f"/{float(report['payload_bottom_limit_z_m']):.4f}，"
                f"姿态={float(report['pose_deviation_deg']):.1f}°"
            )
            record["stop_reason"] = "step_rejected"
            record["reject_detail"] = "executed_step_rejected"
            return record
        previous_gap = float(report["payload_scene_gap_m"])


def _load_transport_route_entries() -> list[dict[str, object]]:
    path = TRANSPORT_ROUTE_LIBRARY
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    routes = payload.get("routes") if isinstance(payload, dict) else None
    if not isinstance(routes, list):
        return []
    return [route for route in routes if isinstance(route, dict)]


def _write_transport_route_entries(routes: list[dict[str, object]]) -> None:
    path = TRANSPORT_ROUTE_LIBRARY
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "cap": TRANSPORT_ROUTE_LIBRARY_CAP,
        "eviction": (
            "同一键只保留一条，后一次成功替换旧条；"
            "新键超过 32 条时丢掉最旧的一条。"
            "扇区与偏航桶只用于索引，不改变任何间隙、碰撞或力标准。"
        ),
        "routes": routes,
    }
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _remember_executed_transport_route(
    identity: dict[str, object],
    candidate: TransportPlanCandidate,
    candidate_source: str,
) -> None:
    """Store the executed joint waypoints only after the motion returns success."""
    exact = candidate.scores.get("exact_minimum_clearance_m")
    if exact is None:
        print("  已执行路径没有精确最小间隙，不写入存档")
        return
    entry = {
        **identity,
        "selected_candidate_id": route_root_id(candidate.candidate_id),
        "exact_minimum_clearance_m": float(exact),
        "candidate_source": candidate_source,
        "joint_waypoints": [
            [float(value) for value in point]
            for point in candidate.joint_waypoints
        ],
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        routes = upsert_transport_routes(_load_transport_route_entries(), entry)
        _write_transport_route_entries(routes)
    except OSError as error:
        print(f"  运输路径存档写入失败：{error}")
        return
    print(
        f"  已存夹持路径 {identity['key']}，"
        f"候选 {candidate.candidate_id}，"
        f"精确最小间隙 {float(exact) * 1000:.1f} mm，"
        f"库内 {len(routes)} 条"
    )


def _validate_reused_carried_path(
    robot: PandaController,
    model: mujoco.MjModel,
    archived_waypoints: Sequence[Sequence[float]],
    archived_id: str,
    target_xyz: np.ndarray,
    *,
    yaw_rad: float,
    ignore_bodies: set[str],
    payload_clearance: tuple[float, float],
    obstacles: list[dict] | None,
    transport_monitor: TransportMonitor,
) -> tuple[TransportPlanCandidate | None, str | None, float]:
    """Re-check a spliced archive with the current dense chord and Ruckig gates.

    The archive does not relax any check. A miss returns the death name and
    no candidate.
    """
    begun = time.perf_counter()
    lock = robot.data_lock if robot.data_lock is not None else nullcontext()
    with lock:
        q_now = _arm_q(model, robot.data)
        scene_qpos = robot.data.qpos.copy()
        finger_q = float(robot.data.qpos[
            int(model.joint("finger_joint1").qposadr[0])
        ])
        hand_id = model.body("hand").id
        initial_hand_rotation = robot.data.xmat[hand_id].reshape(3, 3).copy()
        payload_body = transport_monitor.body_name
        payload_id = model.body(payload_body).id
        payload_joint = int(model.body_jntadr[payload_id])
        payload_qpos = int(model.jnt_qposadr[payload_joint])
        object_rotation = robot.data.xmat[payload_id].reshape(3, 3).copy()
        object_position = robot.data.xpos[payload_id].copy()
        hand_position = robot.data.xpos[hand_id].copy()
        relative_position = initial_hand_rotation.T @ (object_position - hand_position)
        relative_rotation = initial_hand_rotation.T @ object_rotation
        grasp_origin = grasp_contact_origin_m(model, robot.data)
        r_pivot_m = payload_pivot_radius_m(model, robot.data, payload_body)
    shadow_envelope_m = payload_motion_envelope_m(
        float(transport_monitor.maximum_slip_m),
        float(transport_monitor.maximum_tilt_deg),
        r_pivot_m,
        TRACKING_MARGIN_M,
    )
    spliced = splice_reused_waypoints(q_now, archived_waypoints)
    if spliced is None:
        return None, "route_too_short", time.perf_counter() - begun
    addresses = [int(model.joint(name).qposadr[0]) for name in ARM_JOINTS]
    finger_addresses = [
        int(model.joint("finger_joint1").qposadr[0]),
        int(model.joint("finger_joint2").qposadr[0]),
    ]
    drop_m, minimum_z = (float(value) for value in payload_clearance)
    scratch = mujoco.MjData(model)
    payload_sweep_clearance = float(shadow_envelope_m)

    def set_state(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scratch.qpos[:] = scene_qpos
        for address, value in zip(addresses, q, strict=True):
            scratch.qpos[address] = float(value)
        scratch.qpos[finger_addresses[0]] = finger_q
        scratch.qpos[finger_addresses[1]] = finger_q
        mujoco.mj_forward(model, scratch)
        hand_rotation = scratch.xmat[hand_id].reshape(3, 3).copy()
        hand_xyz = scratch.xpos[hand_id].copy()
        payload_xyz = hand_xyz + hand_rotation @ relative_position
        payload_rotation = hand_rotation @ relative_rotation
        quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quaternion, payload_rotation.reshape(-1))
        scratch.qpos[payload_qpos : payload_qpos + 3] = payload_xyz
        scratch.qpos[payload_qpos + 3 : payload_qpos + 7] = quaternion
        mujoco.mj_forward(model, scratch)
        return hand_xyz, hand_rotation

    def valid(q: np.ndarray) -> bool:
        hand_xyz, hand_rotation = set_state(np.asarray(q, dtype=float))
        if _rotation_deviation_deg(initial_hand_rotation, hand_rotation) > (
            transport_monitor.maximum_tilt_deg + 1e-9
        ):
            return False
        if float(hand_xyz[2]) - drop_m < minimum_z:
            return False
        if _scene_hit(
            model, scratch, ignore_bodies, payload_body=payload_body,
        ) is not None:
            return False
        if _robot_scene_clearance_hit(
            model, scratch, ignore_bodies, clearance_m=0.012,
        ) is not None:
            return False
        return _payload_scene_clearance_hit(
            model, scratch, payload_body, ignore_bodies,
            clearance_m=payload_sweep_clearance,
        ) is None

    def sample_death(q: np.ndarray) -> str | None:
        hand_xyz, hand_rotation = set_state(np.asarray(q, dtype=float))
        return reuse_sample_death({
            "pose_deviation_deg": _rotation_deviation_deg(
                initial_hand_rotation, hand_rotation,
            ),
            "pose_limit_deg": float(transport_monitor.maximum_tilt_deg),
            "payload_bottom_z_m": float(hand_xyz[2]) - drop_m,
            "payload_bottom_limit_z_m": float(minimum_z),
            "scene_hit": _scene_hit(
                model, scratch, ignore_bodies, payload_body=payload_body,
            ),
            "robot_clearance_hit": _robot_scene_clearance_hit(
                model, scratch, ignore_bodies, clearance_m=0.012,
            ),
            "payload_clearance_hit": _payload_scene_clearance_hit(
                model, scratch, payload_body, ignore_bodies,
                clearance_m=payload_sweep_clearance,
            ),
        })

    if not _densely_valid(spliced, valid, TRANSPORT_OMPL_DENSE_STEP_RAD):
        death = dense_chord_death(
            spliced, sample_death, TRANSPORT_OMPL_DENSE_STEP_RAD,
        )
        return None, death or "chord_dense_rejected", time.perf_counter() - begun
    execution: list[np.ndarray] = []
    for start_q, end_q in zip(spliced, spliced[1:]):
        count = max(1, int(np.ceil(
            float(np.max(np.abs(end_q - start_q))) / TRANSPORT_OMPL_DENSE_STEP_RAD
        )))
        execution.extend(
            start_q + (step / count) * (end_q - start_q)
            for step in range(1, count + 1)
        )
    hand_points: list[np.ndarray] = []
    pose_deviation = 0.0
    minimum_clearance = float("inf")
    for q in [q_now, *execution]:
        hand_xyz, hand_rotation = set_state(q)
        hand_points.append(hand_xyz.copy())
        pose_deviation = max(
            pose_deviation,
            _rotation_deviation_deg(initial_hand_rotation, hand_rotation),
        )
        payload_bottom = float(hand_xyz[2]) - drop_m
        for marker in obstacles or []:
            horizontal = float(np.linalg.norm(
                hand_xyz[:2] - np.asarray(marker["xy"], dtype=float)
            )) - 0.06
            vertical = payload_bottom - float(marker["top_z"])
            minimum_clearance = min(minimum_clearance, max(horizontal, vertical))
    if not transport_pose_within_limit(
        pose_deviation, transport_monitor.maximum_tilt_deg,
    ):
        return None, "pose", time.perf_counter() - begun
    joint_points = [q_now, *execution]
    base_turn = max(abs(float(q[0] - q_now[0])) for q in joint_points)
    wrist_swing = max(
        float(np.max(np.abs(q[4:] - q_now[4:]))) for q in joint_points
    )
    path_length = float(sum(
        np.linalg.norm(b - a) for a, b in zip(joint_points, joint_points[1:])
    ))
    estimated_duration = path_length / 0.20
    tangential_accel = 0.0
    if len(hand_points) >= 3:
        tangential_accel = max(
            float(np.linalg.norm(c - 2.0 * b + a))
            / max((estimated_duration / (len(hand_points) - 1)) ** 2, 1e-6)
            for a, b, c in zip(hand_points, hand_points[1:], hand_points[2:])
        )
    scores = {
        "estimated_duration_s": estimated_duration,
        "maximum_base_turn_rad": base_turn,
        "maximum_wrist_swing_rad": wrist_swing,
        "minimum_clearance_m": (
            1.0 if not np.isfinite(minimum_clearance) else minimum_clearance
        ),
        "maximum_pose_deviation_deg": pose_deviation,
        "maximum_tangential_acceleration_m_s2": tangential_accel,
        "payload_sweep_clearance_m": payload_sweep_clearance,
    }
    velocity, acceleration, jerk, minimum_duration = _executed_trajectory_limits(
        transport_monitor.motion_profile, transport_monitor.motion_scale,
    )
    trajectory = sample_executed_joint_trajectory(
        joint_points,
        velocity_limits=velocity,
        acceleration_limits=acceleration,
        jerk_limits=jerk,
        minimum_duration=minimum_duration,
        sample_step_rad=TRANSPORT_OMPL_DENSE_STEP_RAD,
    )
    exact_minimum = float("inf")
    component_tracker = _blank_component_tracker()
    trajectory_hit: str | None = None
    ruckig_category: str | None = None
    if trajectory is None:
        trajectory_hit = "ruckig_trajectory_unavailable"
        ruckig_category = "ruckig_trajectory_unavailable"
    else:
        for sample_q in trajectory:
            hand_xyz, hand_rotation = set_state(sample_q)
            pose_now = _rotation_deviation_deg(initial_hand_rotation, hand_rotation)
            pose_deviation = max(pose_deviation, pose_now)
            if pose_now > transport_monitor.maximum_tilt_deg + 1e-9:
                trajectory_hit = "payload_pose_deviation"
                ruckig_category = "pose"
            elif float(hand_xyz[2]) - drop_m < minimum_z:
                trajectory_hit = "payload_height"
                ruckig_category = "height"
            else:
                scene_hit = _scene_hit(
                    model, scratch, ignore_bodies, payload_body=payload_body,
                )
                sample_parts = _component_exact_clearances(
                    model, scratch, ignore_bodies, payload_body,
                )
                release = transport_release_reason(
                    float(sample_parts["robot_scene_gap_m"]),
                    float(sample_parts["payload_scene_gap_m"]),
                    payload_sweep_clearance,
                )
                if scene_hit is not None:
                    trajectory_hit = scene_hit
                    ruckig_category = "scene"
                elif release is not None:
                    trajectory_hit = release
                    ruckig_category = release
            _track_components(
                component_tracker,
                _component_exact_clearances(
                    model, scratch, ignore_bodies, payload_body,
                ),
            )
            exact_minimum = min(
                exact_minimum,
                _minimum_exact_clearance_m(
                    model, scratch, ignore_bodies, payload_body,
                ),
            )
            if trajectory_hit is not None:
                break
    scores.update(_component_score_fields(component_tracker))
    apply_safety_slack(scores, payload_sweep_clearance)
    component_details = _component_detail_fields(component_tracker)
    scores["exact_minimum_clearance_m"] = (
        -1.0 if not np.isfinite(exact_minimum) else float(exact_minimum)
    )
    scores["maximum_pose_deviation_deg"] = pose_deviation
    if ruckig_category is None and float(scores["exact_minimum_clearance_m"]) < 0.0:
        ruckig_category = "negative_exact_gap"
    if trajectory_hit is not None or not execution_sample_released(
        scene_hit=trajectory_hit,
        robot_clearance_hit=None,
        payload_clearance_hit=None,
        exact_minimum_clearance_m=float(scores["exact_minimum_clearance_m"]),
    ):
        return (
            None,
            ruckig_category or "negative_exact_gap",
            time.perf_counter() - begun,
        )
    # Same 25 mm hand-error gate execute already uses. Apply it before motion
    # so a stale goal is discarded and the normal search still runs.
    end_hand, _ = set_state(spliced[-1])
    if float(np.linalg.norm(end_hand - np.asarray(target_xyz, dtype=float))) > 0.025:
        return None, "goal_mismatch", time.perf_counter() - begun
    robot.last_transport_planning_context = {
        "q_start": [float(value) for value in q_now],
        "object_xyz_m": [float(value) for value in object_position],
        "object_rotation_rowmajor": [
            float(value) for value in object_rotation.reshape(-1)
        ],
        "T_gripper_object_translation_m": [float(value) for value in relative_position],
        "T_gripper_object_rotation_rowmajor": [
            float(value) for value in relative_rotation.reshape(-1)
        ],
        "planner_seed": int(OMPL_RANDOM_SEED),
        "scene_qpos": [float(value) for value in scene_qpos],
        "drop_m": float(drop_m),
        "minimum_z_m": float(minimum_z),
        "payload_sweep_clearance_m": float(payload_sweep_clearance),
        "maximum_tilt_deg": float(transport_monitor.maximum_tilt_deg),
        "payload_body": payload_body,
        "grasp_origin_m": [float(value) for value in grasp_origin],
        "r_pivot_m": float(r_pivot_m),
        "payload_motion_envelope_m": float(shadow_envelope_m),
        "candidate_source": "route_reuse",
    }
    candidate = TransportPlanCandidate(
        candidate_id=f"route_reuse:{route_root_id(archived_id)}",
        joint_waypoints=tuple(tuple(float(v) for v in q) for q in spliced),
        execution_points=tuple(tuple(float(v) for v in q) for q in execution),
        hand_waypoints=tuple(tuple(float(v) for v in point) for point in hand_points),
        scores=scores,
        planning_time_s=time.perf_counter() - begun,
        validated=True,
        details=component_details,
    )
    return candidate, None, time.perf_counter() - begun


def _attempt_stored_route_reuse(
    robot: PandaController,
    model: mujoco.MjModel,
    identity: dict[str, object],
    target_xyz: np.ndarray,
    *,
    yaw_rad: float,
    ignore_bodies: set[str],
    payload_clearance: tuple[float, float],
    obstacles: list[dict] | None,
    transport_monitor: TransportMonitor,
) -> dict[str, object]:
    """Try one archived path. No entry means the normal search, not a rejection."""
    key = str(identity["key"])
    archived = next(
        (
            route for route in _load_transport_route_entries()
            if stored_route_matches(route, identity)
        ),
        None,
    )
    if archived is None:
        return {
            "attempted": False,
            "candidate": None,
            "detail": None,
            "planning_time_s": 0.0,
        }
    waypoints = archived.get("joint_waypoints")
    if not isinstance(waypoints, list):
        return {
            "attempted": True,
            "candidate": None,
            "detail": "route_too_short",
            "planning_time_s": 0.0,
        }
    print(f"→ 同键夹持先试复用 {key}")
    candidate, detail, elapsed = _validate_reused_carried_path(
        robot,
        model,
        waypoints,
        str(archived.get("selected_candidate_id") or "stored"),
        target_xyz,
        yaw_rad=yaw_rad,
        ignore_bodies=ignore_bodies,
        payload_clearance=payload_clearance,
        obstacles=obstacles,
        transport_monitor=transport_monitor,
    )
    return {
        "attempted": True,
        "candidate": candidate,
        "detail": detail,
        "planning_time_s": float(elapsed),
    }


def pre_place_pose_specs() -> list[dict[str, object]]:
    """Cartesian pre-place samples above a support. Elbow branches are IK seeds.

    Offsets stay small. Yaw stays inside the existing 12° hand-pose gate.
    Heights are added on top of the caller's minimum hover, never below it.
    """
    specs: list[dict[str, object]] = []
    offsets = (
        ("center", 0.0, 0.0),
        ("left", 0.0, 0.02),
        ("right", 0.0, -0.02),
    )
    for xy_name, dx, dy in offsets:
        for z_extra in (0.0, 0.04, 0.08):
            for yaw_delta in (0.0, 0.12, -0.12):
                specs.append({
                    "name": f"{xy_name}/z{z_extra:.2f}/yaw{yaw_delta:+.2f}",
                    "xy_offset_m": (float(dx), float(dy)),
                    "z_extra_m": float(z_extra),
                    "yaw_delta_rad": float(yaw_delta),
                })
    return specs


def pre_place_pose_accepted(report: Mapping[str, object]) -> bool:
    """The same pose gates a transport goal uses. No clearance number changes."""
    if report.get("scene_hit") is not None:
        return False
    if report.get("robot_clearance_hit") is not None:
        return False
    if report.get("payload_clearance_hit") is not None:
        return False
    if float(report["pose_deviation_deg"]) > float(report["pose_limit_deg"]) + 1e-9:
        return False
    if (
        float(report["payload_bottom_z_m"]) + 1e-9
        < float(report["payload_bottom_limit_z_m"])
    ):
        return False
    return True


def pre_place_rank_key(
    xy_offset_m: float,
    z_extra_m: float,
    yaw_delta_rad: float,
    robot_gap_m: float,
    payload_gap_m: float,
) -> tuple[float, float, float, float]:
    """Prefer the plate center, then the lower hover, then the larger gap."""
    return (
        abs(float(xy_offset_m)),
        abs(float(z_extra_m)),
        abs(float(yaw_delta_rad)),
        -min(float(robot_gap_m), float(payload_gap_m)),
    )


def _approach_pre_place(
    robot: PandaController,
    model: mujoco.MjModel,
    place_hand_xy: np.ndarray,
    *,
    base_z: float,
    yaw_rad: float,
    ignore_bodies: set[str],
    payload_body: str,
    payload_clearance: tuple[float, float],
    transport_monitor: TransportMonitor,
    obstacles: list[dict] | None,
    route_policy: str,
    support_name: str,
) -> float | None:
    """Pick a collision-free pre-place pose, then reach it without planning contact.

    A short remainder is a checked Cartesian step. A long remainder still uses
    OMPL, but only as far as the pre-place pose. The vertical descent is separate.
    """
    lock = robot.data_lock if robot.data_lock is not None else nullcontext()
    with lock:
        q_now = _arm_q(model, robot.data)
        scene_qpos = robot.data.qpos.copy()
        finger_q = float(robot.data.qpos[int(model.joint("finger_joint1").qposadr[0])])
        hand_id = model.body("hand").id
        initial_hand_rotation = robot.data.xmat[hand_id].reshape(3, 3).copy()
        hand_rotation = initial_hand_rotation
        object_rotation = robot.data.xmat[model.body(payload_body).id].reshape(3, 3).copy()
        relative_position = hand_rotation.T @ (
            robot.data.xpos[model.body(payload_body).id] - robot.data.xpos[hand_id]
        )
        relative_rotation = hand_rotation.T @ object_rotation
        payload_joint = int(model.body_jntadr[model.body(payload_body).id])
        payload_qpos = int(model.jnt_qposadr[payload_joint])
        sweep = payload_sweep_clearance_m(
            model, robot.data, payload_body, transport_monitor,
        )
    drop_m, minimum_z = (float(value) for value in payload_clearance)
    addresses = [int(model.joint(name).qposadr[0]) for name in ARM_JOINTS]
    finger_addresses = [
        int(model.joint("finger_joint1").qposadr[0]),
        int(model.joint("finger_joint2").qposadr[0]),
    ]
    lower = np.asarray([model.joint(name).range[0] for name in ARM_JOINTS], dtype=float)
    upper = np.asarray([model.joint(name).range[1] for name in ARM_JOINTS], dtype=float)
    bearing = float(np.arctan2(place_hand_xy[1], place_hand_xy[0]))
    seeds = ik_branch_seeds(q_now, lower, upper, bearing)
    scratch = mujoco.MjData(model)
    accepted: list[dict[str, object]] = []
    for spec in pre_place_pose_specs():
        if len(accepted) >= 4:
            break
        offset = np.asarray(spec["xy_offset_m"], dtype=float)
        target = np.array([
            float(place_hand_xy[0] + offset[0]),
            float(place_hand_xy[1] + offset[1]),
            float(base_z) + float(spec["z_extra_m"]),
        ])
        yaw = float(yaw_rad) + float(spec["yaw_delta_rad"])
        for seed in seeds[:4]:
            try:
                q_goal = solve_hand_ik(model, target, seed=seed, yaw_rad=yaw)
            except RuntimeError:
                continue
            report = _preview_carried_clearance(
                model, scratch, q_goal,
                scene_qpos=scene_qpos,
                addresses=addresses,
                finger_addresses=finger_addresses,
                finger_q=finger_q,
                relative_position=relative_position,
                relative_rotation=relative_rotation,
                payload_qpos=payload_qpos,
                ignore_bodies=ignore_bodies,
                payload_body=payload_body,
                payload_sweep_clearance=sweep,
                initial_hand_rotation=initial_hand_rotation,
                drop_m=drop_m,
                minimum_z=minimum_z,
                maximum_tilt_deg=transport_monitor.maximum_tilt_deg,
            )
            if not pre_place_pose_accepted(report):
                continue
            accepted.append({
                "name": spec["name"],
                "xyz": target,
                "yaw": yaw,
                "q": q_goal,
                "xy_offset_m": float(np.linalg.norm(offset)),
                "z_extra_m": float(spec["z_extra_m"]),
                "yaw_delta_rad": float(spec["yaw_delta_rad"]),
                "robot_gap_m": float(report["robot_scene_gap_m"]),
                "payload_gap_m": float(report["payload_scene_gap_m"]),
            })
            break
    if not accepted:
        robot.last_abort_reason = "pre_place_no_clear_pose"
        print("  盘上预备姿态都没有通过现有碰撞和间隙门，不下降")
        return None
    accepted.sort(key=lambda item: pre_place_rank_key(
        float(item["xy_offset_m"]),
        float(item["z_extra_m"]),
        float(item["yaw_delta_rad"]),
        float(item["robot_gap_m"]),
        float(item["payload_gap_m"]),
    ))
    ompl_attempts = 0
    for pose in accepted:
        with lock:
            hand_now = _hand_xyz(model, robot.data)
            q_seed = _arm_q(model, robot.data)
            scene_qpos = robot.data.qpos.copy()
            finger_q = float(robot.data.qpos[int(model.joint("finger_joint1").qposadr[0])])
        target = np.asarray(pose["xyz"], dtype=float)
        yaw = float(pose["yaw"])
        distance = float(np.linalg.norm(target - hand_now))
        print(
            f"→ 盘上预备姿态 {pose['name']}："
            f"水平 {distance * 1000:.0f} mm，"
            f"机器人间隙 {float(pose['robot_gap_m']) * 1000:.1f} mm，"
            f"负载间隙 {float(pose['payload_gap_m']) * 1000:.1f} mm"
        )
        if distance <= 0.12:
            q_goal = np.asarray(pose["q"], dtype=float)
            span = float(np.max(np.abs(q_goal - q_seed)))
            count = max(1, int(np.ceil(span / 0.15)))
            reached = True
            previous_q = q_seed
            for step in range(1, count + 1):
                q_next = q_seed + (step / count) * (q_goal - q_seed)
                report = _preview_carried_clearance(
                    model, scratch, q_next,
                    scene_qpos=scene_qpos,
                    addresses=addresses,
                    finger_addresses=finger_addresses,
                    finger_q=finger_q,
                    relative_position=relative_position,
                    relative_rotation=relative_rotation,
                    payload_qpos=payload_qpos,
                    ignore_bodies=ignore_bodies,
                    payload_body=payload_body,
                    payload_sweep_clearance=sweep,
                    initial_hand_rotation=initial_hand_rotation,
                    drop_m=drop_m,
                    minimum_z=minimum_z,
                    maximum_tilt_deg=transport_monitor.maximum_tilt_deg,
                )
                if not pre_place_pose_accepted(report):
                    reached = False
                    break
                hit = _segment_hit(
                    model, scene_qpos, previous_q, q_next, ignore_bodies,
                    payload_clearance,
                    finger_q=finger_q,
                    payload_transform=(relative_position, relative_rotation),
                )
                if hit is not None:
                    reached = False
                    break
                step_span = float(np.max(np.abs(q_next - previous_q)))
                if not robot.move_to(
                    q_next,
                    float(np.clip(step_span / 0.6, 0.25, 0.7)),
                    profile=transport_monitor.motion_profile,
                    monitor=transport_monitor.check,
                ):
                    robot.last_abort_reason = robot.last_abort_reason or "pre_place_step_rejected"
                    return None
                previous_q = q_next
            if reached:
                print("  预备姿态用短距离关节步到达，最后下降不再交给 OMPL")
                return yaw
            print("  这条近处预备姿态走不过，试下一个")
            continue
        if ompl_attempts >= 2:
            continue
        ompl_attempts += 1
        if move_hand_clear(
            robot, model, target, yaw_rad=yaw, obstacle_top=0.0,
            ignore_bodies=ignore_bodies,
            payload_clearance=payload_clearance,
            obstacles=obstacles,
            transport_monitor=transport_monitor,
            route_policy=route_policy,
            plan_stage="place",
            support_name=support_name,
        ):
            print("  OMPL 只到盘上预备姿态，接触段改受约束下降")
            return yaw
    robot.last_abort_reason = robot.last_abort_reason or "pre_place_no_clear_pose"
    print("  预备姿态都没到达，不开始下降")
    return None


def move_hand_clear(
    robot: PandaController,
    model: mujoco.MjModel,
    target_xyz: np.ndarray,
    *,
    yaw_rad: float,
    obstacle_top: float,
    ignore_bodies: set[str] | None = None,
    descend_ignore: set[str] | None = None,
    payload_clearance: tuple[float, float] | None = None,
    obstacles: list[dict] | None = None,
    fast: bool = False,
    quick_descend: bool = False,
    transport_monitor: TransportMonitor | None = None,
    route_policy: str = "auto",
    plan_review: Callable[[dict[str, object]], bool] | None = None,
    candidate_selector: Callable[
        [list[dict[str, object]], str], tuple[str | None, str]
    ] | None = None,
    plan_stage: str = "",
    support_name: str | None = None,
) -> bool:
    """Move the hand to ``target_xyz`` while staying above depth-measured obstacles.

    After a route is validated and before it is executed, a PlanBrief is
    stored on ``robot.last_plan_brief``.  ``plan_review`` may veto that plan
    (returns False); it can never alter the waypoints.

    Joint interpolation between two safe poses can still sweep the forearm
    through a tall block.  Each candidate is checked on a scratch physics
    state, and a collision raises the transit height and replans.
    """
    ignore_bodies = set() if ignore_bodies is None else set(ignore_bodies)
    descend_ignore = set() if descend_ignore is None else set(descend_ignore)
    robot.last_abort_reason = None
    target_xyz = np.asarray(target_xyz, dtype=float)
    lock = robot.data_lock if robot.data_lock is not None else nullcontext()
    with lock:
        start = _hand_xyz(model, robot.data)
        q_now = _arm_q(model, robot.data)
        scene_qpos = robot.data.qpos.copy()
        finger_q = float(robot.data.qpos[int(model.joint("finger_joint1").qposadr[0])])
        payload_transform = None
        payload_body_name = None
        if payload_clearance is not None:
            movable = [
                name for name in ignore_bodies
                if name in BODY_OF_COLOR.values()
            ]
            if len(movable) == 1:
                payload_body_name = movable[0]
                hand_rotation = robot.data.xmat[
                    model.body("hand").id
                ].reshape(3, 3).copy()
                object_rotation = robot.data.xmat[
                    model.body(movable[0]).id
                ].reshape(3, 3).copy()
                relative_position = hand_rotation.T @ (
                    robot.data.xpos[model.body(movable[0]).id]
                    - robot.data.xpos[model.body("hand").id]
                )
                relative_rotation = hand_rotation.T @ object_rotation
                payload_transform = (relative_position, relative_rotation)
        local_top = max(
            _local_route_ceiling(
                model, robot.data, start[:2], target_xyz[:2], ignore_bodies
            ),
            _marked_route_ceiling(obstacles, start[:2], target_xyz[:2]),
        )
    if (
        payload_clearance is not None
        and transport_monitor is not None
        and payload_body_name is not None
    ):
        preference = (
            "time_first" if route_policy == "time_first" else "stability_first"
        )
        separation = _separate_payload_vertically(
            robot, model,
            yaw_rad=yaw_rad,
            ignore_bodies=ignore_bodies,
            payload_body=payload_body_name,
            payload_clearance=payload_clearance,
            transport_monitor=transport_monitor,
            goal_xyz=target_xyz,
        )
        if not pre_ompl_may_plan(
            "sweep_met" if separation["reached_sweep"] else str(separation["stop_reason"]),
        ):
            robot.last_transport_plan_record = {
                "planner_used": "ompl",
                "arc_fallback_reason": None,
                "candidates": [],
                "rejected_candidates": [],
                "selected_candidate_id": None,
                "decision_maker": "local_ompl_stop",
                "route_preference": preference,
                "planner_seed": OMPL_RANDOM_SEED,
                "planning_time_s": 0.0,
                "planning_budget_s": TRANSPORT_OMPL_TIMEOUT_S,
                "budget_doubled": False,
                "ik_goal_order": "nominal",
                "executed": False,
                "pre_ompl_vertical_separation": separation,
            }
            stamp_grasp_attachment(robot, transport_monitor)
            robot.last_abort_reason = (
                "pre_ompl_separation_short_of_sweep"
                if separation["stop_reason"] == "ceiling"
                else "pre_ompl_separation_step_rejected"
            )
            print(
                "  垂直脱离未达到负载扫掠，安全停止，不规划："
                f"{separation['stop_reason']}，已抬 "
                f"{float(separation['lift_m']) * 1000:.1f} mm"
            )
            return False
        candidates: list[TransportPlanCandidate] = []
        rejected: list[dict[str, object]] = []
        ompl_planning_time = 0.0
        search_plan = ompl_carried_search_plan(0, preference)
        assert search_plan is not None
        identity = transport_route_identity(
            payload_body_name,
            target_xyz,
            yaw_rad,
            support_name=support_name,
            plan_stage=plan_stage,
            geometry_class=payload_geometry_class(model, payload_body_name),
            attachment_class=attachment_class_of(transport_monitor.baseline_local_xyz),
        )
        reuse = _attempt_stored_route_reuse(
            robot,
            model,
            identity,
            target_xyz,
            yaw_rad=yaw_rad,
            ignore_bodies=ignore_bodies,
            payload_clearance=payload_clearance,
            obstacles=obstacles,
            transport_monitor=transport_monitor,
        )
        chosen: TransportPlanCandidate | None = None
        candidate_source = "ompl_search"
        decision_maker = "local_stability_first"
        if reuse["candidate"] is not None:
            chosen = reuse["candidate"]
            candidates = [chosen]
            candidate_source = "route_reuse"
            decision_maker = "local_route_reuse"
            ompl_planning_time = float(reuse["planning_time_s"])
        else:
            if reuse["attempted"]:
                rejected.append(route_reuse_rejection(
                    str(reuse["detail"] or "chord_dense_rejected"),
                    str(identity["key"]),
                ))
                print(
                    "  存档路径未重新通过校验，丢弃："
                    f"route_reuse_rejected/{reuse['detail']}，回到 OMPL"
                )
            for search_index in range(2):
                search_plan = ompl_carried_search_plan(search_index, preference)
                if search_plan is None:
                    break
                found, new_rejected, elapsed = _plan_ompl_transport_candidates(
                    robot,
                    model,
                    target_xyz,
                    yaw_rad=yaw_rad,
                    ignore_bodies=ignore_bodies,
                    payload_clearance=payload_clearance,
                    obstacles=obstacles,
                    transport_monitor=transport_monitor,
                    timeout_s=float(search_plan["timeout_s"]),
                    reverse_goals=bool(search_plan["reverse_ik_goals"]),
                )
                rejected.extend(new_rejected)
                ompl_planning_time += float(elapsed)
                candidates = found
                preference = str(search_plan["preference"])
                if any(candidate.validated for candidate in found):
                    break
        public_candidates = [
            public_transport_candidate(candidate) for candidate in candidates
        ]
        raise_note: dict[str, object] = {
            "raise_triggered": False,
            "raise_delta_m": None,
            "raise_result": None,
        }
        corridor_note: dict[str, object] = {
            "attempted": False,
            "selected_corridor": None,
            "result": None,
            "attempts": [],
        }
        if chosen is None and not any(candidate.validated for candidate in candidates):
            if segmented_raise_recovery_allowed(rejected):
                with lock:
                    experience_start = _hand_xyz(model, robot.data)
                experience_vias = experience_corridor_vias(
                    _archived_hand_polylines(model),
                    experience_start,
                    target_xyz,
                )
                corridor_passes: list[tuple[str, list[np.ndarray] | None]] = []
                if experience_vias:
                    corridor_passes.append(("experience", experience_vias))
                corridor_passes.append(("geometry", None))
                for corridor_source, via_points in corridor_passes:
                    if chosen is not None:
                        break
                    corridor_note, corridor, corridor_elapsed = (
                        _plan_segmented_ompl_corridors(
                            robot,
                            model,
                            target_xyz,
                            yaw_rad=yaw_rad,
                            ignore_bodies=ignore_bodies,
                            payload_clearance=payload_clearance,
                            obstacles=obstacles,
                            transport_monitor=transport_monitor,
                            preference=preference,
                            via_points=via_points,
                            source=corridor_source,
                        )
                    )
                    ompl_planning_time += float(corridor_elapsed)
                    if corridor is None:
                        continue
                    chosen = corridor
                    candidates = [corridor]
                    candidate_source = (
                        "experience_corridor"
                        if corridor_source == "experience"
                        else "segmented_corridor"
                    )
                    decision_maker = (
                        "local_experience_corridor"
                        if corridor_source == "experience"
                        else "local_geometry_corridor"
                    )
                    public_candidates = [public_transport_candidate(corridor)]
            if chosen is None and payload_gap_only_wipeout(rejected):
                raise_note, raised, raise_elapsed = _raise_carried_transport_once(
                    robot, model, target_xyz,
                    yaw_rad=yaw_rad,
                    ignore_bodies=ignore_bodies,
                    payload_clearance=payload_clearance,
                    obstacles=obstacles,
                    transport_monitor=transport_monitor,
                    preference=preference,
                )
                ompl_planning_time += float(raise_elapsed)
                if raised is not None:
                    chosen = raised
                    candidates = [raised]
                    candidate_source = "raise_midpoint"
                    decision_maker = "local_raise_midpoint"
                    public_candidates = [public_transport_candidate(raised)]
            if chosen is None:
                robot.last_transport_plan_record = carried_ompl_stop_record(
                    rejected,
                    preference=preference,
                    planning_time_s=ompl_planning_time,
                    planning_budget_s=float(search_plan["timeout_s"]),
                    budget_doubled=bool(search_plan["budget_doubled"]),
                    ik_goal_order=(
                        "reversed" if search_plan["reverse_ik_goals"] else "nominal"
                    ),
                )
                robot.last_transport_plan_record["pre_ompl_vertical_separation"] = separation
                robot.last_transport_plan_record["candidate_source"] = "ompl_search"
                robot.last_transport_plan_record["raise_triggered"] = raise_note["raise_triggered"]
                robot.last_transport_plan_record["raise_delta_m"] = raise_note["raise_delta_m"]
                robot.last_transport_plan_record["raise_result"] = raise_note["raise_result"]
                robot.last_transport_plan_record["segmented_corridor"] = corridor_note
                stamp_grasp_attachment(robot, transport_monitor)
                robot.last_abort_reason = "ompl_no_validated_candidate"
                print("  OMPL 没有已验证夹持候选，安全停止，不执行弧线")
                return False
        if chosen is None:
            requested_id = None
            decision_maker = "local_stability_first"
            if candidate_selector is not None:
                requested_id, decision_maker = candidate_selector(
                    public_candidates, preference,
                )
            chosen = choose_transport_candidate(
                candidates, preference, requested_id=requested_id,
            )
            assert chosen is not None
            if requested_id != chosen.candidate_id and requested_id is not None:
                decision_maker = f"{decision_maker}:local_validated_fallback"
        robot.last_transport_plan_record = {
            "planner_used": "ompl",
            "arc_fallback_reason": None,
            "candidates": public_candidates,
            "rejected_candidates": rejected,
            "selected_candidate_id": chosen.candidate_id,
            "decision_maker": decision_maker,
            "candidate_source": candidate_source,
            "route_preference": preference,
            "planner_seed": OMPL_RANDOM_SEED,
            "planning_time_s": ompl_planning_time,
            "planning_budget_s": float(search_plan["timeout_s"]),
            "budget_doubled": bool(search_plan["budget_doubled"]),
            "ik_goal_order": (
                "reversed" if search_plan["reverse_ik_goals"] else "nominal"
            ),
            "executed": True,
            "exact_minimum_clearance_m": chosen.scores.get("exact_minimum_clearance_m"),
            "exact_robot_clearance_m": chosen.scores.get("exact_robot_clearance_m"),
            "exact_payload_clearance_m": chosen.scores.get("exact_payload_clearance_m"),
            "exact_robot_pair": chosen.details.get("exact_robot_pair"),
            "exact_payload_pair": chosen.details.get("exact_payload_pair"),
            "payload_gap_min_m": chosen.details.get("payload_gap_min_m"),
            "payload_gap_median_m": chosen.details.get("payload_gap_median_m"),
            "payload_gap_max_m": chosen.details.get("payload_gap_max_m"),
            "pre_ompl_vertical_separation": separation,
            "raise_triggered": raise_note["raise_triggered"],
            "raise_delta_m": raise_note["raise_delta_m"],
            "raise_result": raise_note["raise_result"],
            "segmented_corridor": corridor_note,
        }
        stamp_grasp_attachment(robot, transport_monitor)
        if candidate_source == "route_reuse":
            print(
                f"→ 夹持运输复用已重新校验的存档：{chosen.candidate_id}，"
                f"来源 route_reuse，"
                f"精确最小间隙 {chosen.scores.get('exact_minimum_clearance_m')} m"
            )
        else:
            print(
                f"→ 夹持运输只用 OMPL：{len(candidates)} 条已验证候选，"
                f"执行 {chosen.candidate_id}，来源 {candidate_source}，"
                f"精确最小间隙 {chosen.scores.get('exact_minimum_clearance_m')} m"
            )
        executed_ok = _execute_ompl_transport_candidate(
            chosen,
            robot,
            model,
            target_xyz,
            ignore_bodies=ignore_bodies,
            transport_monitor=transport_monitor,
            payload_body_name=payload_body_name,
            plan_review=plan_review,
            plan_stage=plan_stage,
            route_policy=route_policy,
        )
        if executed_ok:
            _remember_executed_transport_route(
                identity, chosen, candidate_source,
            )
        return executed_ok
    if payload_clearance is not None:
        robot.last_abort_reason = robot.last_abort_reason or "ompl_no_validated_candidate"
        print("  夹持移动不使用弧线，安全停止")
        return False
    arc_planning_started = time.perf_counter()
    bearing = float(np.arctan2(target_xyz[1], target_xyz[0]))
    base_delta = abs((bearing - float(q_now[0]) + np.pi) % (2.0 * np.pi) - np.pi)
    if fast and payload_clearance is None and base_delta > 1.45:
        print(
            f"  当前扫描分支与目标方位相差 {np.rad2deg(base_delta):.0f}°，"
            "先用 OMPL 在关节空间切换 IK 分支"
        )
        ompl_result = _try_ompl_move(
            robot, model, target_xyz,
            yaw_rad=yaw_rad, ignore_bodies=ignore_bodies,
        )
        if ompl_result is not None:
            return ompl_result
    # Do not climb to the tallest nearby marker up front.  A lower path is
    # kept when it is actually clear; only a real hit raises the arm.
    clearance = local_top + FINGER_BELOW_HAND + TRANSIT_MARGIN
    base = max(float(target_xyz[2]), min(float(start[2]), clearance))
    if payload_clearance is not None and route_policy == "high_clearance":
        base = min(0.68, base + 0.08)
    # via_slot 0 is the straight line.  1 and 2 are the two side detours.
    # A hit tries the clearer side before any extra climb, because lifting a
    # friction grasp over a pedestal is what shook the block loose.
    sides = _side_points(start[:2], target_xyz[:2], obstacles)
    crosses_base_route = (
        payload_clearance is not None
        and not _segment_clears_base(start[:2], target_xyz[:2])
    )
    plans = order_transport_plans(crosses_base_route, start[:2], target_xyz[:2])
    plan_index = 0
    while plan_index < len(plans):
        extra, via_slot = plans[plan_index]
        plan_index += 1
        travel_z = min(0.68, base + extra)
        via = None if via_slot == 0 or via_slot > len(sides) else sides[via_slot - 1]
        crosses_base = (
            payload_clearance is not None
            and not _segment_clears_base(start[:2], target_xyz[:2])
        )
        # Opposite sides of the base: leave the straight line and the short
        # side-step.  Those both put the gripper through the arm.
        if crosses_base and via_slot <= 1:
            which = "短边" if via_slot == 0 else "另一侧"
            wide_radius = float(np.clip(
                max(np.linalg.norm(start[:2]), np.linalg.norm(target_xyz[:2]))
                + 0.08 + 0.50 * extra,
                0.50,
                0.74,
            ))
            print(
                f"  起点和终点在底座两侧，直线会穿过机械臂。"
                f"尝试半径 {wide_radius:.2f} m 的自适应外圈走{which}"
            )
            first_direction = -1.0 if route_policy == "clockwise" else 1.0
            if route_policy == "counterclockwise":
                first_direction = 1.0
            waypoints = _wide_route(
                start, target_xyz, max(travel_z, 0.36),
                direction=first_direction if via_slot == 0 else -first_direction,
                radius=wide_radius,
            )
        elif via is None:
            waypoints = _route(
                start, target_xyz, travel_z,
                fast=fast and payload_clearance is None,
                quick_descend=quick_descend and payload_clearance is None,
            )
        else:
            via_point = np.array([via[0], via[1], max(travel_z, float(start[2]), float(target_xyz[2]))])
            first = _route(start, via_point, travel_z, fast=fast and payload_clearance is None)
            second = _route(
                via_point, target_xyz, travel_z,
                fast=fast and payload_clearance is None,
                quick_descend=quick_descend and payload_clearance is None,
            )
            waypoints = first + second[1:]
        solved_points: list[np.ndarray] = []
        qs: list[np.ndarray] = []
        seed = q_now
        final = waypoints[-1]
        unreachable = False
        for point in waypoints[1:]:
            solved = None
            ik_error = None
            try:
                solved = solve_hand_ik(model, point, seed=seed, yaw_rad=yaw_rad)
            except RuntimeError as error:
                solved = None
                ik_error = str(error)
                # The straight solve stays on the current wrist branch and can
                # miss a target on the other side of the base.  Nudge joint 1
                # a short step toward that target and solve again, instead of
                # letting one waypoint swing the base the long way around.
                bearing = float(np.arctan2(point[1], point[0]))
                turn = (bearing - float(seed[0]) + np.pi) % (2.0 * np.pi) - np.pi
                if abs(turn) > 0.35:
                    nudged = seed.copy()
                    nudged[0] = float(seed[0]) + float(np.clip(turn, -0.55, 0.55))
                    try:
                        solved = solve_hand_ik(
                            model, point, seed=nudged, yaw_rad=yaw_rad
                        )
                        ik_error = None
                    except RuntimeError as retry_error:
                        solved = None
                        ik_error = str(retry_error)
            if solved is not None:
                base_jump = float(abs(solved[0] - seed[0]))
                wrist_jump = float(np.max(np.abs(solved[5:] - seed[5:])))
                # A base turn has to move the wrist with it.  A spin is a large
                # wrist change while the base stays put, which is what made the
                # gripper twirl just above the block.
                if payload_clearance is not None and base_jump > 1.2:
                    print(
                        f"  这一步底座要转 {np.rad2deg(base_jump):.0f}°，"
                        "跨度太大，物体容易被甩掉，拆开重规划"
                    )
                    solved = None
                    ik_error = "base-jump"
                elif wrist_jump > 1.4 and base_jump < 0.35:
                    print(
                        f"  底座几乎没动（{np.rad2deg(base_jump):.0f}°），"
                        f"腕部却要转 {np.rad2deg(wrist_jump):.0f}°，这是空转，丢弃该点"
                    )
                    solved = None
                    ik_error = "wrist-spin"
            if solved is None:
                if np.linalg.norm(point - final) < 1e-5:
                    unreachable = True
                    break
                continue
            seed = solved
            solved_points.append(point)
            qs.append(solved.copy())
        if unreachable or not qs:
            if crosses_base and via_slot == 0:
                print(f"  高度 {travel_z:.2f} m 这一侧外圈到不了，换另一侧绕")
                plans.insert(plan_index, (extra, 1))
                continue
            if via_slot < len(sides) and not crosses_base:
                detour = sides[via_slot]
                print(
                    f"  高度 {travel_z:.2f} m 直线到不了终点。"
                    f"先改经 ({detour[0]:+.2f}, {detour[1]:+.2f})，不先抬高"
                )
                plans.insert(plan_index, (extra, via_slot + 1))
                continue
            print(
                f"  高度 {travel_z:.2f} m 绕开后仍到不了目标 "
                f"{np.round(final, 3).tolist()}"
            )
            continue
        route_top = max(float(point[2]) for point in waypoints)
        blocked = None
        blocked_at = None
        previous = q_now
        for point, q_next in zip(solved_points, qs, strict=True):
            descending = (
                abs(float(point[0] - target_xyz[0])) < 1e-4
                and abs(float(point[1] - target_xyz[1])) < 1e-4
                and float(point[2]) < route_top - 0.02
            )
            ignore = ignore_bodies | descend_ignore if descending else ignore_bodies
            blocked = _segment_hit(
                model, scene_qpos, previous, q_next, ignore,
                # The carried object's collision volume matters during the
                # entire route, not only during final descent.  The old
                # conditional let the green block skim the 0.250 m pedestal
                # on the outer arc even though every robot link was clear.
                payload_clearance,
                finger_q if payload_clearance is not None else None,
                payload_transform,
            )
            if blocked is None and payload_clearance is not None and obstacles:
                drop = float(payload_clearance[0])
                object_bottom = float(point[2]) - drop
                for marker in obstacles:
                    if float(marker["top_z"]) < object_bottom - 0.03:
                        continue
                    gap = float(np.linalg.norm(
                        np.asarray(point[:2]) - np.asarray(marker["xy"], dtype=float)
                    ))
                    # Markers are fat on purpose for the empty arm.  The carried
                    # box only clips a pedestal when the hand itself is over it.
                    if gap < 0.12:
                        blocked = (
                            f"夹着的物体会蹭到高台"
                            f"({float(marker['xy'][0]):+.2f},{float(marker['xy'][1]):+.2f})"
                        )
                        break
            if blocked is not None:
                blocked_at = np.asarray(point, dtype=float).copy()
                break
            previous = q_next
        if blocked is not None and crosses_base and via_slot == 0:
            where = "" if blocked_at is None else f"（手目标 {np.round(blocked_at, 3).tolist()}）"
            print(f"  这一侧外圈会碰到 {blocked}{where}，换另一侧绕，仍然不从胳膊中间穿")
            plans.insert(plan_index, (extra, 1))
            continue
        if blocked is not None and via_slot < len(sides) and not crosses_base:
            detour = sides[via_slot]
            if blocked.startswith("self:"):
                reason = f"这条路会让夹爪插进机械臂（{blocked}）"
            else:
                reason = f"这条路会碰到 {blocked}"
            carrying = payload_clearance is not None
            print(
                f"  {reason}。{'夹着物体，先绕开而不抬高，避免甩掉' if carrying else '先绕开'}"
                f"，改经 ({detour[0]:+.2f}, {detour[1]:+.2f})"
            )
            plans.insert(plan_index, (extra, via_slot + 1))
            continue
        if blocked is not None:
            where = "" if blocked_at is None else f"（手目标 {np.round(blocked_at, 3).tolist()}）"
            print(
                f"  高度 {travel_z:.3f} m 绕开后仍会碰到 {blocked}{where}，"
                "再抬高一点重规划"
            )
            continue
        if crosses_base_route:
            remember_validated_transport_seed(
                start[:2], target_xyz[:2], extra, via_slot, validated=True,
            )
        plan_bytes = np.asarray(qs, dtype=np.float64).tobytes()
        plan_id = "plan_" + hashlib.sha256(plan_bytes).hexdigest()[:12]
        if payload_clearance is not None and hasattr(robot, "last_transport_plan_record"):
            record = dict(robot.last_transport_plan_record)
            record.update({
                "planner_used": "arc",
                "selected_candidate_id": plan_id,
                "arc_planning_time_s": float(time.perf_counter() - arc_planning_started),
                "planning_time_s": (
                    float(record.get("ompl_planning_time_s", 0.0))
                    + float(time.perf_counter() - arc_planning_started)
                ),
            })
            robot.last_transport_plan_record = record
        print(
            f"  ✓ 路径 {plan_id} 已在 MuJoCo 副本逐段验证："
            f"{len(qs)} 个关节路点；先验证、后执行"
        )
        base_turn = float(qs[-1][0] - q_now[0])
        if abs(base_turn) > 0.6:
            print(
                f"  底座从 {np.rad2deg(q_now[0]):+.0f}° 转到 {np.rad2deg(qs[-1][0]):+.0f}°，"
                "这是把夹爪转到物体所在方位，腕部保持原朝向"
            )
        # Plan brief: the same duration rule as the execution loop below, so
        # the predicted time is what the robot will actually request.
        predicted_durations: list[float] = []
        brief_previous = q_now
        for q_next in qs:
            span = float(np.max(np.abs(q_next - brief_previous)))
            if (fast or quick_descend) and payload_clearance is None:
                predicted_durations.append(float(np.clip(span / 2.2, 0.08, 0.40)))
            else:
                base_step = float(abs(q_next[0] - brief_previous[0]))
                if base_step > 0.35:
                    predicted_durations.append(float(np.clip(base_step / 0.25, 0.6, 2.2)))
                else:
                    predicted_durations.append(float(np.clip(span / 0.55, 0.28, 1.3)))
            brief_previous = q_next
        robot.last_plan_brief = _summarize_plan(
            [start] + list(solved_points),
            durations=predicted_durations,
            obstacles=obstacles,
            payload_drop_m=None if payload_clearance is None else float(payload_clearance[0]),
            peak_force_n=float(getattr(robot, "plan_peak_force_hint_n", 0.0) or 0.0),
            route_policy=route_policy,
            crosses_base=bool(crosses_base),
            plan_id=plan_id,
            stage=plan_stage,
        )
        if plan_review is not None and not plan_review(robot.last_plan_brief):
            if not robot.last_abort_reason:
                robot.last_abort_reason = "plan_brief_rejected"
            print(f"  ✗ 路径述职未放行（{robot.last_abort_reason}），不执行该路径")
            return False
        climb = float(travel_z - start[2])
        if climb > 0.03 and local_top > 0.08:
            print(
                f"→ 先抬到 {travel_z:.3f} m 再移动：路径旁最高 {local_top:.3f} m，"
                f"指尖在手心下方 {FINGER_BELOW_HAND:.2f} m，贴着走会撞上"
            )
        elif fast and payload_clearance is None:
            print(f"→ 快速移到目标上方，高度 {travel_z:.3f} m（路径旁最高 {local_top:.3f} m）")
        elif quick_descend and payload_clearance is None:
            print(f"→ 已在上方，按 5 cm 一步下降到夹取高度（不再 1.5 cm 细磨）")
        else:
            print(f"→ 沿高度 {travel_z:.3f} m 移动（路径旁最高 {local_top:.3f} m）")
        previous = q_now
        for q_index, q_next in enumerate(qs):
            span = float(np.max(np.abs(q_next - previous)))
            if (fast or quick_descend) and payload_clearance is None:
                duration = float(np.clip(span / 2.2, 0.08, 0.40))
            else:
                # A carried block slips if the base yaws quickly.
                base_step = float(abs(q_next[0] - previous[0]))
                if base_step > 0.35:
                    duration = float(np.clip(base_step / 0.25, 0.6, 2.2))
                else:
                    duration = float(np.clip(span / 0.55, 0.28, 1.3))
            profile = (
                transport_monitor.motion_profile
                if payload_clearance is not None and transport_monitor is not None
                else "payload" if payload_clearance is not None else "normal"
            )
            monitor = None
            if transport_monitor is not None:
                def monitor() -> str | None:
                    reason = transport_monitor.check()
                    if reason is not None:
                        return reason
                    # Planning validates a rigidly attached payload on a copy
                    # of the MuJoCo state.  Dynamics and tracking error can
                    # still deviate slightly from that path, so execution has
                    # an independent full-body/payload collision interlock.
                    transport_monitor.runtime_collision_checks += 1
                    hit = _scene_hit(
                        model,
                        robot.data,
                        ignore_bodies,
                        payload_body=payload_body_name,
                        contact_distance_threshold_m=0.001,
                    )
                    if hit is None and payload_body_name is not None:
                        hit = _payload_scene_clearance_hit(
                            model,
                            robot.data,
                            payload_body_name,
                            ignore_bodies,
                            clearance_m=0.004,
                        )
                    if hit is not None:
                        return f"runtime_collision:{hit}"
                    return None
            # A carried object must not see the acceleration discontinuities
            # produced by independently time-scaled quintic segments.  Ruckig
            # enforces velocity, acceleration *and jerk* limits for every
            # validated segment.  Keep the proven quintic controller for the
            # empty arm so this change cannot regress the fast scan/pick path.
            if payload_clearance is not None:
                # Preserve a small velocity through smooth route vertices.
                # Stopping at every one of 20+ arc samples unloads and reloads
                # the pads; Ruckig can cross monotonic joints continuously
                # while still enforcing the payload velocity/accel/jerk caps.
                via_velocity = np.zeros(7, dtype=float)
                final_segment = q_index == len(qs) - 1
                if not final_segment:
                    outgoing = qs[q_index + 1] - q_next
                    incoming = q_next - previous
                    same_direction = incoming * outgoing > 0.0
                    via_velocity[same_direction] = np.sign(
                        outgoing[same_direction]
                    ) * np.minimum(
                        0.08,
                        np.abs(outgoing[same_direction]) / 0.8,
                    )
                transport_monitor.begin_segment()
                moved = robot.move_to_ruckig(
                    q_next,
                    duration,
                    profile=profile,
                    monitor=monitor,
                    target_velocity=via_velocity,
                    settle=final_segment or base_step > 0.35,
                    motion_scale=transport_monitor.motion_scale,
                )
            else:
                moved = robot.move_to(
                    q_next, duration, profile=profile, monitor=monitor
                )
            if not moved:
                if robot.last_abort_reason:
                    print(
                        f"  ✗ 本地负载监控触发安全停止：{robot.last_abort_reason}；"
                        "不等待 LLM 继续驱动"
                    )
                return False
            if (
                payload_clearance is not None
                and transport_monitor is not None
                and transport_monitor.consume_preload_request()
            ):
                if not _refresh_payload_preload(
                    robot,
                    model,
                    transport_monitor.body_name,
                    transport_monitor,
                ):
                    return False
            previous = q_next
        return True
    if payload_clearance is None:
        print("→ 空手局部候选全灭，改用 OMPL 搜索其他 IK/通道")
        ompl_result = _try_ompl_move(
            robot,
            model,
            target_xyz,
            yaw_rad=yaw_rad,
            ignore_bodies=ignore_bodies | descend_ignore,
        )
        if ompl_result is not None:
            return ompl_result
    print("✗ 抬高并绕开后仍然躲不开场景或机械臂自身")
    robot.last_abort_reason = "motion_plan_failed"
    return False


def nudge_hand(
    robot: PandaController,
    model: mujoco.MjModel,
    target_xyz: np.ndarray,
    *,
    yaw_rad: float,
    ignore_bodies: set[str] | None = None,
    payload_clearance: tuple[float, float] | None = None,
    transport_monitor: TransportMonitor | None = None,
    target_rotation: np.ndarray | None = None,
    safety_monitor: Callable[[], str | None] | None = None,
    payload_approach_body: str | None = None,
) -> bool:
    """Short joint move from the current pose.  Used for the last centimetres of a place."""
    ignore_bodies = set() if ignore_bodies is None else set(ignore_bodies)
    lock = robot.data_lock if robot.data_lock is not None else nullcontext()
    with lock:
        q_now = _arm_q(model, robot.data)
        scene_qpos = robot.data.qpos.copy()
    try:
        q_next = solve_hand_ik(
            model, np.asarray(target_xyz, dtype=float), seed=q_now, yaw_rad=yaw_rad,
            target_rotation=target_rotation,
        )
    except RuntimeError as error:
        print(f"  下降逆解失败：{error}")
        return False
    hit = _segment_hit(
        model, scene_qpos, q_now, q_next, ignore_bodies, payload_clearance,
        payload_approach_body=payload_approach_body,
    )
    if hit is not None:
        print(f"  这一小步会碰到 {hit}，停在当前位置松手")
        return False
    span = float(np.max(np.abs(q_next - q_now)))
    profile = (
        transport_monitor.motion_profile
        if payload_clearance is not None and transport_monitor is not None
        else "payload" if payload_clearance is not None else "precision"
    )
    monitor = None
    if transport_monitor is not None or safety_monitor is not None:
        def monitor() -> str | None:
            if transport_monitor is not None:
                reason = transport_monitor.check()
                if reason is not None:
                    return reason
            return None if safety_monitor is None else safety_monitor()
    return robot.move_to(
        q_next,
        float(np.clip(span / 0.6, 0.25, 0.7)),
        profile=profile,
        monitor=monitor,
    )


def _align_wrist(
    robot: PandaController,
    model: mujoco.MjModel,
    yaw_rad: float,
) -> bool:
    """Turn to the grasp yaw while staying above the object.

    Doing this together with the descent makes one IK target change position
    and wrist at once, and that solve often fails or spins the gripper on the
    way down.
    """
    current = _hand_yaw(model, robot.data)
    delta = (yaw_rad - current + np.pi) % (2.0 * np.pi) - np.pi
    if abs(delta) < 0.20:
        return True
    steps = max(1, int(np.ceil(abs(delta) / 0.30)))
    print(
        f"→ 先在物体上方把腕部转到夹持面"
        f"（{np.rad2deg(current):+.0f}° → {np.rad2deg(current + delta):+.0f}°，"
        f"分 {steps} 步）。下降时夹爪不再边降边转"
    )
    for index in range(1, steps + 1):
        yaw = current + delta * index / steps
        hand = _hand_xyz(model, robot.data)
        if not nudge_hand(robot, model, hand, yaw_rad=yaw):
            print("  腕部这一步转不过去")
            return False
    return True


def body_region_center_xy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mask: np.ndarray,
    depth: np.ndarray,
    span: dict[str, float],
) -> np.ndarray | None:
    """Median of mask points that sit inside the existing handle-core cut.

    The 65th-percentile radius is the one ``object_span`` already uses to
    decide that a handle is present. Points farther along that handle are
    outside the selected body region and do not enter the median.
    """
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0.02)
    rows, cols = np.nonzero(valid)
    if rows.size < 40:
        return None
    stride = max(1, rows.size // 1500)
    points = _backproject_pixels(
        model, data, CAMERA_NAME, rows[::stride], cols[::stride], depth,
    )
    finite = points[np.all(np.isfinite(points), axis=1)]
    if finite.shape[0] < 30:
        return None
    xy = finite[:, :2]
    median = np.median(xy, axis=0)
    centered = xy - median
    core_radius = float(np.percentile(np.linalg.norm(centered, axis=1), 65))
    axis = np.array((
        np.cos(float(span["handle_direction_rad"])),
        np.sin(float(span["handle_direction_rad"])),
    ))
    keep = centered @ axis <= core_radius
    if int(np.count_nonzero(keep)) < 15:
        return median
    return np.median(xy[keep], axis=0)


def _observe_down(
    robot: PandaController,
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    color: str,
    object_name: str,
    xy: np.ndarray,
    hover_z: float,
    obstacle_top: float,
    yaw_rad: float,
    ignore_bodies: set[str] | None = None,
    payload_clearance: tuple[float, float] | None = None,
    obstacles: list[dict] | None = None,
    fast: bool = False,
    transport_monitor: TransportMonitor | None = None,
    route_policy: str = "auto",
    plan_review: Callable[[dict[str, object]], bool] | None = None,
    candidate_selector: Callable[
        [list[dict[str, object]], str], tuple[str | None, str]
    ] | None = None,
    plan_stage: str = "",
) -> tuple[Detection | None, dict[str, float] | None, float]:
    """Hover over one XY, then read a fresh RGB-D measurement."""
    hover = np.array([float(xy[0]), float(xy[1]), float(hover_z)])
    if not move_hand_clear(
        robot, model, hover, yaw_rad=yaw_rad, obstacle_top=obstacle_top,
        ignore_bodies=ignore_bodies, payload_clearance=payload_clearance,
        obstacles=obstacles, fast=fast, transport_monitor=transport_monitor,
        route_policy=route_policy, plan_review=plan_review,
        candidate_selector=candidate_selector, plan_stage=plan_stage,
        support_name=object_name,
    ):
        return None, None, obstacle_top
    lock = robot.data_lock if robot.data_lock is not None else nullcontext()
    with lock:
        rgb, depth = render_rgbd(renderer, robot.data)
        detection = detect_colored_object(
            rgb, depth, color, object_name, model, robot.data, CAMERA_NAME,
            near_xy=xy, near_radius=0.12,
        )
        if detection is None:
            detection = detect_occluded_support(
                rgb, depth, color, object_name, model, robot.data,
                CAMERA_NAME, near_xy=xy,
            )
            if detection is not None:
                print(
                    "  俯视复核：负载遮挡使支撑面颜色轮廓分裂，"
                    "已用碎片化盘沿重建中心（仍执行 6 cm 关联门）"
                )
        if detection is None:
            print("  俯视时已确认坐标附近没有同色物体，改用该点的深度")
            detection = detection_from_image_point(
                rgb, depth, 0.5, 0.5, f"{color} {object_name}",
                model, robot.data, CAMERA_NAME,
            )
        ceiling = workspace_ceiling_m(model, robot.data, CAMERA_NAME, depth)
        _note_obstacles(obstacles, model, robot.data, depth)
        span = None
        if detection is not None:
            span = object_span(model, robot.data, CAMERA_NAME, detection.mask, depth)
            if span is not None and float(span.get("has_handle", 0.0)) > 0.5:
                region_xy = body_region_center_xy(
                    model, robot.data, detection.mask, depth, span,
                )
                if region_xy is not None:
                    span = dict(span)
                    span["body_region_center_x"] = float(region_xy[0])
                    span["body_region_center_y"] = float(region_xy[1])
    return detection, span, max(obstacle_top, ceiling)


def _contact_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> tuple[set[str], dict[str, float]]:
    """Return valid side contacts and summed normal force for each finger."""
    target_id = model.body(body_name).id
    finger_ids = {
        model.body("left_finger").id: "left",
        model.body("right_finger").id: "right",
    }
    found: set[str] = set()
    normal_force = {"left": 0.0, "right": 0.0}
    for index in range(data.ncon):
        contact = data.contact[index]
        # Touching the top with a fingertip is not an opposing friction grasp.
        # Keep only contacts whose normal is mainly horizontal (a side face).
        if abs(float(contact.frame[2])) > 0.65:
            continue
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        side = None
        if body1 == target_id and body2 in finger_ids:
            side = finger_ids[body2]
        elif body2 == target_id and body1 in finger_ids:
            side = finger_ids[body1]
        if side is not None:
            found.add(side)
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            normal_force[side] += abs(float(wrench[0]))
    return found, normal_force


def _support_contact_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    payload_body: str,
    support_body: str,
) -> tuple[int, float]:
    """Number and total normal force of payload/support contacts."""
    payload_id = model.body(payload_body).id
    support_id = model.body(support_body).id
    count = 0
    normal_force = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        bodies = {
            int(model.geom_bodyid[contact.geom1]),
            int(model.geom_bodyid[contact.geom2]),
        }
        if bodies != {payload_id, support_id}:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        count += 1
        normal_force += abs(float(wrench[0]))
    return count, normal_force


def _palm_pressing_top(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> bool:
    """True when the palm, not the finger pads, is sitting on the object's top."""
    hand_id = model.body("hand").id
    target_id = model.body(body_name).id
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.dist > -0.0005:
            continue
        bodies = {
            int(model.geom_bodyid[contact.geom1]),
            int(model.geom_bodyid[contact.geom2]),
        }
        if bodies != {hand_id, target_id}:
            continue
        normal = contact.frame[:3]
        if abs(float(normal[2])) > 0.7:
            return True
    return False


def _body_up_axis(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    return data.xmat[model.body(body_name).id].reshape(3, 3)[:, 2].copy()


def _upright_hand_rotation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> np.ndarray:
    """Hand orientation that makes the currently held object's Z axis vertical."""
    hand_rotation = data.xmat[model.body("hand").id].reshape(3, 3).copy()
    object_rotation = data.xmat[model.body(body_name).id].reshape(3, 3).copy()
    object_in_hand = hand_rotation.T @ object_rotation
    object_x = object_rotation[:, 0].copy()
    object_x[2] = 0.0
    norm = float(np.linalg.norm(object_x))
    if norm < 1e-6:
        object_x = np.array((1.0, 0.0, 0.0))
    else:
        object_x /= norm
    world_up = np.array((0.0, 0.0, 1.0))
    object_y = np.cross(world_up, object_x)
    upright_object = np.column_stack((object_x, object_y, world_up))
    return upright_object @ object_in_hand.T


def _motion_since(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    rest_xyz: np.ndarray,
    rest_up: np.ndarray,
) -> tuple[float, float, float]:
    current = _body_xyz(model, data, body_name)
    shift = float(np.linalg.norm(current[:2] - rest_xyz[:2]))
    drop = float(rest_xyz[2] - current[2])
    cosine = float(np.clip(rest_up @ _body_up_axis(model, data, body_name), -1.0, 1.0))
    tilt_deg = float(np.rad2deg(np.arccos(cosine)))
    return shift, drop, tilt_deg


def _close_once(
    robot: PandaController,
    gap_m: float,
    body_name: str,
    model: mujoco.MjModel,
    rest_xyz: np.ndarray,
    *,
    object_width_m: float,
    yaw_rad: float,
    task_log: "TaskLog | None" = None,
    attempt: int = 0,
    servo_residual_m: float = 1.0,
    yaw_policy: str = "geometry",
    perceived_center_xy_m: np.ndarray | None = None,
    closing_axis_xy: np.ndarray | None = None,
    handle_direction_rad: float | None = None,
    handle_direction_stabilized: bool = False,
) -> ClosureResult | None:
    """Fast free-space close, then sub-millimetre contact/force regulation."""
    desired_gap = float(np.clip(gap_m, 0.014, 0.08))
    if body_name in BOX_BODIES:
        target_force, maximum_force = GRIP_FORCE_TARGET_N["box"], MAX_GRIP_FORCE_N["box"]
    elif body_name == "cube":
        target_force, maximum_force = GRIP_FORCE_TARGET_N["cube"], MAX_GRIP_FORCE_N["cube"]
    elif "bottle" in body_name:
        target_force, maximum_force = GRIP_FORCE_TARGET_N["bottle"], MAX_GRIP_FORCE_N["bottle"]
    else:
        target_force, maximum_force = GRIP_FORCE_TARGET_N["cup"], MAX_GRIP_FORCE_N["cup"]
    rest_up = _body_up_axis(model, robot.data, body_name)
    # In this Panda model the inner pad gap is about controller_gap - 6 mm.
    # Arrive quickly 4 mm before predicted contact, then seek at 1 mm and
    # regulate force at 0.25 mm.  The former fixed 2 mm steps could exceed the
    # stated force ceiling before the first measurement was taken.
    predicted_contact_gap = float(np.clip(desired_gap + 0.007, 0.014, 0.08))
    squeeze = float(np.clip(predicted_contact_gap + 0.004, 0.014, 0.08))
    print(
        f"→ 两阶段闭爪：先快速到预测接触前 {squeeze:.3f} m，"
        f"再以 1.0/0.25 mm 搜索并调力；目标 {target_force:.2f} N，"
        f"硬上限 {maximum_force:.2f} N"
    )
    contacts: set[str] = set()
    forces = {"left": 0.0, "right": 0.0}
    centering_spent = 0.0
    centering_events = 0
    yaw_spent = 0.0
    translation_attempted = False
    previous_peak: float | None = None
    yaw_skip_logged = False
    inline_adjustments_disabled = False
    first_contact_aperture: dict[str, float | None] = {
        "left": None,
        "right": None,
    }
    first_contact_controller_aperture: dict[str, float | None] = {
        "left": None,
        "right": None,
    }
    peak_forces = {"left": 0.0, "right": 0.0}
    seating_travel_m = 0.0
    reseated = False
    closure_disturbed = False
    force_sequence: list[dict[str, object]] = []
    logged_step = 0

    def record_force_sample(step_index: int, aperture_m: float, state: str) -> None:
        sample = {
            "step": int(step_index),
            "aperture_m": float(aperture_m),
            "left_force_n": float(forces.get("left", 0.0)),
            "right_force_n": float(forces.get("right", 0.0)),
            "contact_state": state,
            "seating_depth_m": max(0.0, float(desired_gap) - float(aperture_m)),
        }
        if force_sequence and force_sequence[-1] == sample:
            return
        force_sequence.append(sample)

    def observe_contact_sample(aperture_m: float) -> None:
        literal_aperture = max(
            0.0, float(aperture_m) - GRIPPER_INNER_PAD_OFFSET_M
        )
        for side in ("left", "right"):
            force = float(forces.get(side, 0.0))
            peak_forces[side] = max(peak_forces[side], force)
            if side in contacts and first_contact_aperture[side] is None:
                first_contact_aperture[side] = literal_aperture
                first_contact_controller_aperture[side] = float(aperture_m)

    def contact_geometry(final_state: str) -> dict[str, object]:
        shift_diag, drop_diag, tilt_diag = _motion_since(
            model, robot.data, body_name, rest_xyz, rest_up,
        )
        inferred = contact_axis_offset(first_contact_aperture, object_width_m)
        perceived = (
            None if perceived_center_xy_m is None
            else np.asarray(perceived_center_xy_m, dtype=float)[:2]
        )
        truth = np.asarray(rest_xyz, dtype=float)[:2]
        truth_error = None if perceived is None else truth - perceived
        closing_axis = (
            None if closing_axis_xy is None
            else np.asarray(closing_axis_xy, dtype=float)[:2]
        )
        handle_angle_deg = None
        if closing_axis is not None and handle_direction_rad is not None:
            closing_norm = float(np.linalg.norm(closing_axis))
            handle_axis = np.array((
                np.cos(float(handle_direction_rad)),
                np.sin(float(handle_direction_rad)),
            ))
            if closing_norm > 1e-9:
                cosine = abs(float((closing_axis / closing_norm) @ handle_axis))
                handle_angle_deg = float(np.rad2deg(np.arccos(np.clip(cosine, 0.0, 1.0))))
        return {
            "attempt": attempt,
            "left_first_contact_aperture_m": first_contact_aperture["left"],
            "right_first_contact_aperture_m": first_contact_aperture["right"],
            "left_first_contact_controller_aperture_m": (
                first_contact_controller_aperture["left"]
            ),
            "right_first_contact_controller_aperture_m": (
                first_contact_controller_aperture["right"]
            ),
            "controller_to_inner_pad_offset_m": GRIPPER_INNER_PAD_OFFSET_M,
            **inferred,
            "measured_object_width_m": float(object_width_m),
            "final_contact_state": final_state,
            "peak_forces_n": {
                "left": float(peak_forces["left"]),
                "right": float(peak_forces["right"]),
            },
            "seating_travel_m": float(seating_travel_m),
            "perceived_center_xy_m": None if perceived is None else perceived.tolist(),
            "mujoco_body_center_xy_m": truth.tolist(),
            "perception_truth_error_xy_m": (
                None if truth_error is None else truth_error.tolist()
            ),
            "perception_truth_error_m": (
                None if truth_error is None else float(np.linalg.norm(truth_error))
            ),
            "closing_axis_world_xy": (
                None if closing_axis is None else closing_axis.tolist()
            ),
            "handle_direction_rad": handle_direction_rad,
            "handle_direction_stabilized": bool(handle_direction_stabilized),
            "closing_axis_handle_angle_deg": handle_angle_deg,
            "truth_used_for_control": False,
            "object_shift_m": float(shift_diag),
            "object_drop_m": float(drop_diag),
            "object_tilt_deg": float(tilt_diag),
            "force_sequence": thin_force_sequence(force_sequence),
            "force_sequence_total_steps": len(force_sequence),
            "force_sequence_limit": FORCE_SEQUENCE_MAX_ENTRIES,
        }

    def log_contact_geometry(final_state: str) -> dict[str, object]:
        evidence = contact_geometry(final_state)
        if task_log is not None:
            task_log.add_checkpoint(
                "grasp_contact_geometry", evidence, final_state,
            )
        return evidence

    def classify() -> str:
        shift, drop, tilt = _motion_since(model, robot.data, body_name, rest_xyz, rest_up)
        return classify_grasp_contact(
            contacts, forces, target_force, maximum_force,
            shift_m=shift, drop_m=drop, tilt_deg=tilt,
        )

    def trial_release_decision() -> str:
        shift, drop, tilt = _motion_since(
            model, robot.data, body_name, rest_xyz, rest_up,
        )
        return grasp_trial_release_decision(
            contacts,
            forces,
            target_force,
            maximum_force,
            shift_m=shift,
            drop_m=drop,
            tilt_deg=tilt,
            closure_disturbed=closure_disturbed,
        )

    if not robot.grasp_to(_ctrl_for_gap(squeeze), 0.18):
        log_contact_geometry("controller_abort")
        return None
    if not robot.hold(0.03):
        log_contact_geometry("controller_abort")
        return None

    phase = "seek"
    max_close_steps = grasp_close_step_budget(squeeze, desired_gap)
    for step in range(max_close_steps):
        contacts, forces = _contact_metrics(model, robot.data, body_name)
        observe_contact_sample(squeeze)
        state = classify()
        logged_step = step + 1
        record_force_sample(logged_step, squeeze, state)
        shift, _, tilt = _motion_since(model, robot.data, body_name, rest_xyz, rest_up)
        print(
            f"  {phase} {step + 1}: 开口={squeeze:.4f} m，状态={state}，"
            f"左/右={forces['left']:.2f}/{forces['right']:.2f} N，"
            f"移动={shift * 1000:.1f} mm，倾斜={tilt:.1f}°"
        )
        release_decision = trial_release_decision()
        # The low contact gate allows a later micro-lift. It does not end the
        # existing seating search. Squeezing still stops once the established
        # load is reached, the hard cap is hit, or the object is displaced.
        established_load = (
            release_decision == "ready_for_trial_lift"
            and min(float(forces.get("left", 0.0)), float(forces.get("right", 0.0)))
            >= GRASP_TRIAL_LIFT_FORCE_FRACTION * float(target_force)
        )
        if (
            release_decision in {
                "strong_force_limit", "object_displaced", "closure_disturbed",
            }
            or established_load
            or state in {"force_limit", "object_displaced"}
        ):
            break
        # A one-sided or unbalanced contact is a centering observation, not a
        # reason to squeeze hard into the object.  A few hundredths of a
        # newton is merely first touch, though: continue at 0.25 mm until the
        # opposite pad arrives.  Abort only when asymmetry becomes material.
        if (
            state in {"left_only", "right_only", "bilateral_unbalanced"}
            and not inline_adjustments_disabled
        ):
            peak_now = max(forces.values())
            centering_ready = _inline_centering_ready(state, forces, target_force)
            material_contact = peak_now >= GRASP_INLINE_CENTERING_MIN_FORCE_N
            if material_contact or centering_ready:
                high_side = max(forces, key=forces.get)
                force_rising = (
                    previous_peak is not None and peak_now > previous_peak + 1e-6
                )
                yaw_step = (
                    inline_yaw_adjust_rad(
                        high_side=high_side,
                        yaw_policy=yaw_policy,
                        spent_rad=yaw_spent,
                        servo_residual_m=servo_residual_m,
                        translation_attempted=translation_attempted,
                        force_rising=force_rising,
                    )
                    if centering_ready else None
                )
                if (
                    yaw_step is None
                    and yaw_policy == "avoid_handle"
                    and force_rising
                    and (
                        float(servo_residual_m) <= GRASP_SERVO_TARGET_M
                        or translation_attempted
                    )
                    and not yaw_skip_logged
                ):
                    yaw_skip_logged = True
                    print("  偏航微调跳过：yaw_policy=avoid_handle，不覆盖安全策略")
                    if task_log is not None:
                        task_log.add_checkpoint(
                            "grasp_inline_yaw_adjust",
                            {
                                "attempt": attempt,
                                "skipped": True,
                                "reason": "yaw_policy_avoid_handle",
                                "state_before": state,
                                "forces_before_n": {
                                    "left": float(forces.get("left", 0.0)),
                                    "right": float(forces.get("right", 0.0)),
                                },
                            },
                            "skipped",
                        )
                if yaw_step is not None:
                    new_yaw = float(yaw_rad) + float(yaw_step)
                    hand_before_yaw = _hand_xyz(model, robot.data)
                    left_finger = robot.data.xpos[model.body("left_finger").id]
                    right_finger = robot.data.xpos[model.body("right_finger").id]
                    midpoint = (left_finger + right_finger) / 2.0
                    offset = hand_before_yaw - midpoint
                    cosine, sine = float(np.cos(yaw_step)), float(np.sin(yaw_step))
                    yaw_target = midpoint + np.array([
                        cosine * offset[0] - sine * offset[1],
                        sine * offset[0] + cosine * offset[1],
                        offset[2],
                    ])
                    yaw_target[2] = hand_before_yaw[2]
                    forces_before_yaw = forces.copy()
                    print(
                        f"  ↺ 闭爪在环偏航 {np.rad2deg(yaw_step):+.1f}°"
                        f"（累计 {np.rad2deg(yaw_spent + yaw_step):+.1f}/"
                        f"±{np.rad2deg(GRASP_YAW_ADJUST_TOTAL_RAD):.0f}°），"
                        f"先接触侧={high_side}"
                    )
                    moved = nudge_hand(
                        robot, model, yaw_target, yaw_rad=new_yaw,
                        ignore_bodies={body_name},
                    )
                    if moved:
                        moved = robot.hold(0.04)
                    contacts, forces = _contact_metrics(model, robot.data, body_name)
                    observe_contact_sample(squeeze)
                    next_state = classify()
                    yaw_spent += float(yaw_step)
                    yaw_rad = new_yaw
                    if task_log is not None:
                        task_log.add_checkpoint(
                            "grasp_inline_yaw_adjust",
                            {
                                "attempt": attempt,
                                "delta_rad": float(yaw_step),
                                "yaw_rad": float(yaw_rad),
                                "spent_rad": float(yaw_spent),
                                "state_before": state,
                                "state_after": next_state,
                                "forces_before_n": {
                                    "left": float(forces_before_yaw.get("left", 0.0)),
                                    "right": float(forces_before_yaw.get("right", 0.0)),
                                },
                                "forces_after_n": {
                                    "left": float(forces.get("left", 0.0)),
                                    "right": float(forces.get("right", 0.0)),
                                },
                                "motion_completed": bool(moved),
                            },
                            "continue_closing" if moved else "abort",
                        )
                    yaw_made_it_worse = (
                        float(forces.get(high_side, 0.0))
                        > float(forces_before_yaw.get(high_side, 0.0)) + 1e-6
                    )
                    previous_peak = max(forces.values())
                    if yaw_made_it_worse:
                        inline_adjustments_disabled = True
                        print(
                            "    ✗ 偏航后接触侧力上升，停止后续在环修正，"
                            "继续有界闭爪"
                        )
                        if task_log is not None:
                            task_log.add_checkpoint(
                                "grasp_inline_yaw_adjust",
                                {
                                    "attempt": attempt,
                                    "brake": "contact_force_rose",
                                    "delta_rad": float(yaw_step),
                                    "forces_before_n": {
                                        "left": float(forces_before_yaw.get("left", 0.0)),
                                        "right": float(forces_before_yaw.get("right", 0.0)),
                                    },
                                    "forces_after_n": {
                                        "left": float(forces.get("left", 0.0)),
                                        "right": float(forces.get("right", 0.0)),
                                    },
                                },
                                "contact_force_rose",
                            )
                    controller_aborted = (
                        not moved
                        and (
                            robot.last_abort_reason is not None
                            or not robot._can_continue()
                        )
                    )
                    if controller_aborted or next_state in {
                        "force_limit", "object_displaced", "bilateral_balanced",
                    }:
                        break
                    if not moved:
                        # A rejected IK/collision candidate means only that
                        # the proposed D/E adjustment is unavailable. It does
                        # not invalidate the unchanged closure path.
                        inline_adjustments_disabled = True
                        previous_peak = max(forces.values())
                        phase = "force"
                        continue
                    if yaw_made_it_worse:
                        # A D/E brake stops more corrective motion, not I-2's
                        # closure. Unchanged safety states above still abort.
                        phase = "force"
                        continue
                    phase = "force"
                    continue
                if not centering_ready:
                    previous_peak = peak_now
                    phase = "force"
                else:
                    # Keep the remaining yaw budget for the next force climb and
                    # do not spend the translation budget on that sample.
                    yaw_still_available = (
                        yaw_policy != "avoid_handle"
                        and (GRASP_YAW_ADJUST_TOTAL_RAD - abs(yaw_spent)) > 1e-9
                        and (
                            float(servo_residual_m) <= GRASP_SERVO_TARGET_M
                            or translation_attempted
                        )
                    )
                    if yaw_still_available and not force_rising:
                        previous_peak = peak_now
                        phase = "force"
                    elif (
                        GRASP_CENTERING_TOTAL_BUDGET_M - centering_spent <= 1e-7
                        or centering_events >= GRASP_CENTERING_MAX_EVENTS
                    ):
                        # Translation budget exhaustion is not a success or a
                        # safety stop. Continue closure so bounded seating can
                        # still reach the unchanged bilateral criterion.
                        previous_peak = peak_now
                        phase = "force"
                    else:
                        remaining = GRASP_CENTERING_TOTAL_BUDGET_M - centering_spent
                        estimate = _estimate_centering_correction_m(
                            squeeze, object_width_m, forces, target_force
                        )
                        magnitude = min(estimate, remaining, GRASP_CENTERING_MAX_STEP_M)
                        local_correction, world_correction = (
                            _centering_correction_in_gripper_frame(
                                model, robot.data, high_side, magnitude
                            )
                        )
                        forces_before = forces.copy()
                        event_before = _body_xyz(model, robot.data, body_name)
                        hand_before = _hand_xyz(model, robot.data)

                        translation_gate_reasons = {
                            "force_above_target", "peak_force_doubled",
                        }
                        substeps: list[dict[str, object]] = []
                        translation_stop_reason: str | None = None
                        commanded_translation = 0.0
                        any_motion = False
                        print(
                            f"  ↺ 闭爪在环对中 {centering_events + 1}/"
                            f"{GRASP_CENTERING_MAX_EVENTS}: 状态={state}，"
                            f"开口={squeeze * 1000:.1f} mm，测宽={object_width_m * 1000:.1f} mm，"
                            f"估计/执行={estimate * 1000:.1f}/{magnitude * 1000:.1f} mm，"
                            f"夹爪系={np.round(local_correction * 1000, 1)} mm，"
                            f"分 {len(translation_substep_lengths(magnitude))} 步"
                            f"≤{TRANSLATION_FORCE_SUBSTEP_M * 1000:.1f} mm"
                        )
                        for substep_index, step_len in enumerate(
                            translation_substep_lengths(magnitude), start=1,
                        ):
                            step_world = world_correction * (step_len / magnitude)

                            def centering_monitor() -> str | None:
                                shift_now, drop_now, tilt_now = _motion_since(
                                    model, robot.data, body_name, rest_xyz, rest_up
                                )
                                if shift_now > 0.010 or drop_now > 0.008 or tilt_now > 8.0:
                                    return "object_displaced"
                                _, live_forces = _contact_metrics(
                                    model, robot.data, body_name,
                                )
                                if max(live_forces.values()) > maximum_force * 1.05:
                                    return "grasp_force_limit"
                                return translation_force_stop(
                                    forces_before, live_forces, target_force,
                                )

                            hand_now = _hand_xyz(model, robot.data)
                            centering_target = hand_now.copy()
                            centering_target[:2] += step_world
                            step_moved = nudge_hand(
                                robot,
                                model,
                                centering_target,
                                yaw_rad=yaw_rad,
                                ignore_bodies={body_name},
                                safety_monitor=centering_monitor,
                            )
                            abort_reason = robot.last_abort_reason
                            if abort_reason in translation_gate_reasons:
                                robot.last_abort_reason = None
                                translation_stop_reason = str(abort_reason)
                            if step_moved:
                                step_moved = robot.hold(0.04)
                            contacts, forces = _contact_metrics(
                                model, robot.data, body_name,
                            )
                            settled_gate = translation_force_stop(
                                forces_before, forces, target_force,
                            )
                            if translation_stop_reason is None and settled_gate is not None:
                                translation_stop_reason = settled_gate
                            commanded_translation += float(step_len)
                            if step_moved or translation_stop_reason is not None:
                                any_motion = True
                            substeps.append({
                                "substep": substep_index,
                                "commanded_m": float(step_len),
                                "left_force_n": float(forces.get("left", 0.0)),
                                "right_force_n": float(forces.get("right", 0.0)),
                                "motion_completed": bool(step_moved),
                                "stop_reason": translation_stop_reason,
                            })
                            print(
                                f"    平移子步 {substep_index}: "
                                f"{step_len * 1000:.2f} mm，"
                                f"左/右={forces['left']:.2f}/{forces['right']:.2f} N"
                                + (
                                    f"，停止={translation_stop_reason}"
                                    if translation_stop_reason else ""
                                )
                            )
                            if translation_stop_reason is not None or not step_moved:
                                break
                        if translation_stop_reason is None and any_motion:
                            translation_stop_reason = "completed"
                        elif translation_stop_reason is None:
                            translation_stop_reason = "substep_rejected"
                        moved = any_motion
                        event_after = _body_xyz(model, robot.data, body_name)
                        achieved = _hand_xyz(model, robot.data)[:2] - hand_before[:2]
                        contacts, forces = _contact_metrics(model, robot.data, body_name)
                        observe_contact_sample(squeeze)
                        next_state = classify()
                        object_delta = event_after[:2] - event_before[:2]
                        centering_events += 1
                        centering_spent += commanded_translation
                        event = {
                            "attempt": attempt,
                            "event": centering_events,
                            "state_before": state,
                            "state_after": next_state,
                            "high_force_side": high_side,
                            "aperture_m": squeeze,
                            "object_width_m": float(object_width_m),
                            "estimated_offset_m": estimate,
                            "commanded_local_xy_m": local_correction.tolist(),
                            "commanded_world_xy_m": world_correction.tolist(),
                            "achieved_world_xy_m": achieved.tolist(),
                            "object_delta_xy_m": object_delta.tolist(),
                            "total_commanded_m": centering_spent,
                            "forces_before_n": {
                                "left": float(forces_before.get("left", 0.0)),
                                "right": float(forces_before.get("right", 0.0)),
                            },
                            "motion_completed": bool(moved),
                            "result": next_state,
                            "substeps": substeps,
                            "translation_stop_reason": translation_stop_reason,
                        }
                        if task_log is not None:
                            task_log.add_checkpoint(
                                "grasp_inline_centering", event,
                                "continue_closing" if moved else "abort",
                            )
                        print(
                            f"    对中后状态={next_state}，手实际移动="
                            f"{np.round(achieved * 1000, 1)} mm，物体移动="
                            f"{np.linalg.norm(object_delta) * 1000:.1f} mm，"
                            f"累计预算={centering_spent * 1000:.1f}/"
                            f"{GRASP_CENTERING_TOTAL_BUDGET_M * 1000:.1f} mm"
                        )
                        translation_attempted = True
                        brake = inline_translation_should_brake(
                            forces_before, forces, high_side, achieved, object_delta,
                        )
                        if (
                            translation_stop_reason in translation_gate_reasons
                            and brake is None
                        ):
                            brake = translation_stop_reason
                        if brake is not None:
                            event["brake"] = brake
                            closure_disturbed = True
                            print(
                                f"    ✗ 平移后{brake}，停止后续在环修正，"
                                "继续有界闭爪"
                            )
                            if task_log is not None:
                                task_log.add_checkpoint(
                                    "grasp_inline_translation_brake", event, brake,
                                )
                            inline_adjustments_disabled = True
                        controller_aborted = (
                            not moved
                            and (
                                robot.last_abort_reason is not None
                                or not robot._can_continue()
                            )
                        )
                        if controller_aborted or next_state in {
                            "force_limit", "object_displaced", "bilateral_balanced"
                        }:
                            break
                        if not moved:
                            # The lateral candidate was rejected before it
                            # moved. Disable further corrections but preserve
                            # I-2's remaining bounded squeeze.
                            inline_adjustments_disabled = True
                            previous_peak = max(forces.values())
                            phase = "force"
                            continue
                        if brake is not None:
                            previous_peak = max(forces.values())
                            phase = "force"
                            continue
                        # Re-measure before any further 0.25 mm closure.  This keeps
                        # the self-correction inside the force-search loop.
                        previous_peak = max(forces.values())
                        phase = "force"
                        continue
            previous_peak = max(forces.values())
            phase = "force"
        if contacts:
            phase = "force"
        previous_peak = max(forces.values())
        next_squeeze, next_seating_travel = next_grasp_squeeze_m(
            squeeze, desired_gap, state, bool(contacts),
        )
        if next_squeeze is None:
            state = classify()
            # Both pads are on the object and it has not moved, but the
            # seating budget is spent before the existing trial-lift floor.
            # Lower the hand once along the wall. Do not close further and
            # do not change the grasp yaw.
            if (
                not reseated
                and trial_release_decision() == "continue_closing"
                and contacts == {"left", "right"}
            ):
                reseated = True
                shift_now, drop_now, tilt_now = _motion_since(
                    model, robot.data, body_name, rest_xyz, rest_up,
                )
                finger_now = (
                    float(_hand_xyz(model, robot.data)[2]) - FINGER_PAD_BELOW_HAND
                )
                support_now = _body_bottom_z(model, robot.data, body_name)
                drop_m = None
                if (
                    shift_now <= OBJECT_STABILITY_MAX_SHIFT_M
                    and drop_now <= 0.008
                    and tilt_now <= OBJECT_STABILITY_MAX_TILT_DEG
                ):
                    drop_m = grasp_reseat_drop_m(finger_now, support_now)
                if drop_m is not None:
                    lower = _hand_xyz(model, robot.data).copy()
                    lower[2] -= float(drop_m)
                    print(
                        f"→ 双侧已接触且物体未动，夹爪再下放 {drop_m * 1000:.0f} mm；"
                        "不追加闭合力，不更换夹持面"
                    )

                    def reseat_monitor() -> str | None:
                        shift_live, drop_live, tilt_live = _motion_since(
                            model, robot.data, body_name, rest_xyz, rest_up,
                        )
                        if (
                            shift_live > OBJECT_STABILITY_MAX_SHIFT_M
                            or drop_live > 0.008
                            or tilt_live > OBJECT_STABILITY_MAX_TILT_DEG
                        ):
                            return "object_displaced"
                        return None

                    if nudge_hand(
                        robot,
                        model,
                        lower,
                        yaw_rad=yaw_rad,
                        ignore_bodies={body_name},
                        safety_monitor=reseat_monitor,
                    ) and robot.hold(0.04):
                        continue
            break
        if next_seating_travel > seating_travel_m + 1e-7:
            if seating_travel_m <= 1e-7:
                print(
                    f"→ 到达测宽夹距仍未满足双侧目标力，进入有界坐实；"
                    f"最多 {seating_extra_allowance_m() * 1000:.1f} mm"
                )
            phase = "seat"
        seating_travel_m = next_seating_travel
        squeeze = next_squeeze
        duration = 0.10 if phase == "force" else 0.14
        if not robot.grasp_to(_ctrl_for_gap(squeeze), duration):
            log_contact_geometry("controller_abort")
            return None
        if not robot.hold(0.025):
            log_contact_geometry("controller_abort")
            return None
    contacts, forces = _contact_metrics(model, robot.data, body_name)
    observe_contact_sample(squeeze)
    state = classify()
    release_decision = trial_release_decision()
    record_force_sample(logged_step, squeeze, state)
    if release_decision == "ready_for_trial_lift":
        ratio = max(forces.values()) / max(min(forces.values()), 0.05)
        weak = min(float(forces.get("left", 0.0)), float(forces.get("right", 0.0)))
        if weak >= GRASP_TRIAL_LIFT_FORCE_FRACTION * float(target_force):
            print(
                "  ✓ 双侧绝对承重达到试抬下限，停止继续闭爪；"
                f"力比 {ratio:.2f}:1 仅记账"
            )
        else:
            print(
                "  ✓ 双侧已形成接触且物体未被推动，允许试抬；"
                f"力比 {ratio:.2f}:1 仅记账"
            )
    elif release_decision == "closure_disturbed":
        print("  ✗ 闭合时物体被带动且接触力仍在上升，不试抬")
    elif release_decision == "strong_force_limit" or state == "force_limit":
        print("  ✗ 实测力触及硬上限；立即停止，不再追加闭合")
    geometry = log_contact_geometry(state)
    return ClosureResult(
        contacts, squeeze, forces, state,
        ready_for_trial_lift=release_decision == "ready_for_trial_lift",
        trial_release_decision=release_decision,
        yaw_rad=float(yaw_rad), contact_geometry=geometry,
    )


def preload_force_decision(
    contacts: set[str],
    forces: dict[str, float],
    safe_minimum_force: float,
    maximum_force: float,
) -> str:
    """Pass on bilateral force levels. The force ratio is recorded, not a gate.

    Both pads must be in contact. The weak side must reach the existing safe
    floor and the strong side must stay at or under the object hard limit.
    A weak side that is still short keeps closing unless the strong side has
    already reached that limit.
    """
    minimum = min(forces.values()) if contacts == {"left", "right"} else 0.0
    peak = max(forces.values())
    if (
        contacts == {"left", "right"}
        and minimum >= safe_minimum_force
        and peak <= maximum_force
    ):
        return "passed"
    if peak >= maximum_force:
        return "strong_limit"
    return "continue"


def _stabilize_payload(
    robot: PandaController,
    model: mujoco.MjModel,
    body_name: str,
    object_name: str,
    current_gap_m: float,
) -> tuple[bool, float, dict[str, float]]:
    """Build bounded preload from measured bilateral contact before transit.

    Position control is not treated as a force command: after every small
    aperture change we read MuJoCo contact forces and reject excessive object
    motion.  This remains a normal friction grasp, never a weld/attachment.
    """
    target_force = GRIP_FORCE_TARGET_N.get(object_name, 1.1)
    safe_fraction = 0.90 if object_name in {"box", "cube"} else 0.70
    safe_minimum_force = safe_fraction * target_force
    maximum_force = MAX_GRIP_FORCE_N.get(object_name, 2.4)
    start_xyz = _body_xyz(model, robot.data, body_name)
    start_up = _body_up_axis(model, robot.data, body_name)
    gap = float(current_gap_m)
    contacts, forces = _contact_metrics(model, robot.data, body_name)
    samples: list[dict[str, object]] = []

    def record_sample(step: int, decision: str) -> None:
        sample_minimum = (
            min(forces.values()) if contacts == {"left", "right"} else 0.0
        )
        sample_peak = max(forces.values())
        samples.append({
            "step": step,
            "gap_m": gap,
            "left_normal_force_n": float(forces["left"]),
            "right_normal_force_n": float(forces["right"]),
            "force_ratio": float(sample_peak / max(sample_minimum, 0.05)),
            "decision": decision,
        })

    def finish(passed: bool, decision: str) -> tuple[bool, float, dict[str, float]]:
        if (
            not samples
            or samples[-1]["left_normal_force_n"] != float(forces["left"])
            or samples[-1]["right_normal_force_n"] != float(forces["right"])
            or samples[-1]["decision"] != decision
        ):
            record_sample(len(samples), decision)
        robot.last_preload_record = {
            "passed": passed,
            "safe_minimum_force_n": safe_minimum_force,
            "maximum_force_n": maximum_force,
            "maximum_balance_ratio": 2.2,
            "samples": samples,
            "final_forces_n": dict(forces),
            "final_decision": decision,
        }
        return passed, gap, forces
    print(
        f"→ 运输前建立受限预紧：目标双侧至少 {target_force:.2f} N，"
        f"安全下限 {safe_minimum_force:.2f} N，任一侧不超过 {maximum_force:.2f} N"
    )
    for step in range(28):
        minimum = min(forces.values()) if contacts == {"left", "right"} else 0.0
        peak = max(forces.values())
        force_ratio = peak / max(minimum, 0.05)
        decision = preload_force_decision(
            contacts, forces, safe_minimum_force, maximum_force,
        )
        record_sample(step, decision)
        if decision == "passed":
            print(
                f"  ✓ 预紧通过：左/右={forces['left']:.2f}/{forces['right']:.2f} N，"
                f"开口命令 {gap:.3f} m"
            )
            return finish(True, decision)
        if decision in {"strong_limit", "unbalanced"}:
            print(
                f"  ✗ 预紧停止（{decision}）：左/右="
                f"{forces['left']:.2f}/{forces['right']:.2f} N，强弱比={force_ratio:.2f}"
            )
            return finish(False, decision)
        # Contact is already established here; use 0.25 mm force-regulation
        # increments instead of the old 2 mm jumps.
        next_gap = max(0.014, gap - 0.00025)
        if next_gap >= gap - 1e-6:
            break
        def preload_monitor() -> str | None:
            live_contacts, live_forces = _contact_metrics(model, robot.data, body_name)
            live_decision = preload_force_decision(
                live_contacts, live_forces, safe_minimum_force, maximum_force,
            )
            if live_decision in {"strong_limit", "unbalanced"}:
                return f"preload_{live_decision}"
            return None

        moved = robot.grasp_to(
            _ctrl_for_gap(next_gap), 0.30, monitor=preload_monitor,
        )
        if not moved:
            contacts, forces = _contact_metrics(model, robot.data, body_name)
            decision = preload_force_decision(
                contacts, forces, safe_minimum_force, maximum_force,
            )
            return finish(decision == "passed", decision)
        if not robot.hold(0.08):
            return finish(False, "controller_abort")
        gap = next_gap
        contacts, forces = _contact_metrics(model, robot.data, body_name)
        shift, drop, tilt = _motion_since(
            model, robot.data, body_name, start_xyz, start_up
        )
        print(
            f"  预紧 {step + 1}: 左/右={forces['left']:.2f}/{forces['right']:.2f} N，"
            f"位移={shift * 1000:.1f} mm，下降={drop * 1000:.1f} mm，"
            f"倾斜={tilt:.1f}°"
        )
        if shift > 0.008 or abs(drop) > 0.008 or tilt > 6.0:
            print("  ✗ 增加预紧时物体发生明显位移，停止运输")
            return finish(False, "object_displaced")
        if max(forces.values()) > maximum_force:
            print("  ✗ 实测夹持力超过上限，停止运输")
            return finish(False, "strong_limit")
    minimum = min(forces.values()) if contacts == {"left", "right"} else 0.0
    peak = max(forces.values())
    if (
        contacts != {"left", "right"}
        or minimum < safe_minimum_force
        or peak > maximum_force
    ):
        print(
            f"  ✗ 预紧后仍不稳定：接触={sorted(contacts)}，"
            f"最小侧力={minimum:.2f} N"
        )
        return finish(False, "insufficient_or_unbalanced")
    return finish(True, "passed")


def _refresh_payload_preload(
    robot: PandaController,
    model: mujoco.MjModel,
    body_name: str,
    monitor: TransportMonitor,
) -> bool:
    """Restore weakening bilateral preload between trajectory segments.

    This is deliberately force gated: it never closes on a one-sided grasp,
    never tightens when either pad is already at the object-specific force
    ceiling, and verifies relative pose immediately after the 1 mm command.
    """
    contacts, before_forces = _contact_metrics(model, robot.data, body_name)
    if contacts != {"left", "right"}:
        robot.last_abort_reason = "adaptive_preload_rejected:unilateral_contact"
        monitor.stability_events.append({
            "action": "preload_failed",
            "reason": "unilateral_contact",
            "segment_index": monitor.segment_index,
        })
        return False
    minimum = min(before_forces.values())
    maximum = max(before_forces.values())
    target_force = GRIP_FORCE_TARGET_N.get(body_name.removesuffix("_block"), 1.1)
    if body_name in BOX_BODIES:
        target_force = GRIP_FORCE_TARGET_N["box"]
    elif body_name in CUP_BODIES:
        target_force = GRIP_FORCE_TARGET_N["cup" if "bottle" not in body_name else "bottle"]
    force_ceiling = (
        MAX_GRIP_FORCE_N["box"] if body_name in BOX_BODIES
        else MAX_GRIP_FORCE_N.get("bottle" if "bottle" in body_name else "cup", 2.0)
    )
    slip_before, _tilt_before = monitor.relative_slip_and_tilt()
    if minimum >= 0.65 * target_force or maximum >= force_ceiling:
        if maximum >= force_ceiling:
            monitor.force_hold = True
            monitor.preload_slip_reference_m = slip_before
            monitor.recovery_stage = "hold_at_limit"
        else:
            monitor.preload_slip_reference_m = slip_before
            monitor.recovery_stage = "slowed"
        monitor.recovery_signal_segment = monitor.segment_index
        monitor.stability_events.append({
            "action": "preload_skipped",
            "reason": "force_recovered_or_at_ceiling",
            "segment_index": monitor.segment_index,
            "weak_force_n": minimum,
            "peak_force_n": maximum,
        })
        return True

    old_ctrl = float(robot.data.ctrl[-1])
    # 255 actuator units span 80 mm, so 3.19 units are one millimetre.
    new_ctrl = max(_ctrl_for_gap(0.014), old_ctrl - 255.0 / 80.0)
    if new_ctrl >= old_ctrl - 1e-6:
        return True
    print(
        f"  ↻ 双指预紧衰减到 {before_forces['left']:.2f}/"
        f"{before_forces['right']:.2f} N；按力反馈收紧 1 mm"
    )
    def adaptive_preload_monitor() -> str | None:
        live_contacts, live_forces = _contact_metrics(model, robot.data, body_name)
        decision = preload_force_decision(
            live_contacts,
            live_forces,
            0.65 * target_force,
            force_ceiling,
        )
        if decision in {"strong_limit", "unbalanced"}:
            return f"adaptive_preload_{decision}"
        return None

    if (
        not robot.grasp_to(new_ctrl, 0.14, monitor=adaptive_preload_monitor)
        or not robot.hold(0.04)
    ):
        reason = robot.last_abort_reason or "controller_abort"
        slip_after, tilt_after = monitor.relative_slip_and_tilt()
        contacts, after_forces = _contact_metrics(model, robot.data, body_name)
        if "strong_limit" in reason:
            decision = strong_limit_transport_decision(
                slip_m=slip_after,
                slip_before_m=slip_before,
                tilt_deg=tilt_after,
                tilt_limit_deg=monitor.maximum_tilt_deg,
                bilateral=contacts == {"left", "right"},
            )
            if decision == "hold_at_limit":
                monitor.force_hold = True
                monitor.preload_slip_reference_m = slip_after
                monitor.recovery_stage = "hold_at_limit"
                monitor.stability_events.append({
                    "action": "hold_at_limit",
                    "reason": "strong_limit_stable",
                    "segment_index": monitor.segment_index,
                    "relative_slip_m": slip_after,
                    "peak_force_n": max(after_forces.values()) if after_forces else 0.0,
                })
                robot.last_abort_reason = None
                print(
                    "  强侧到达上限且物体仍稳定，停止补力，按当前开口继续运输"
                )
                return True
            reason = "adaptive_preload_unstable"
            robot.last_abort_reason = reason
        monitor.stability_events.append({
            "action": "preload_failed",
            "reason": reason,
            "segment_index": monitor.segment_index,
        })
        return False
    contacts, after_forces = _contact_metrics(model, robot.data, body_name)
    reason = monitor.check()
    peak = max(after_forces.values())
    slip_after, tilt_after = monitor.relative_slip_and_tilt()
    if reason is None and contacts == {"left", "right"} and peak > force_ceiling:
        decision = strong_limit_transport_decision(
            slip_m=slip_after,
            slip_before_m=slip_before,
            tilt_deg=tilt_after,
            tilt_limit_deg=monitor.maximum_tilt_deg,
            bilateral=True,
        )
        if decision == "hold_at_limit":
            monitor.force_hold = True
            monitor.preload_slip_reference_m = slip_after
            monitor.recovery_stage = "hold_at_limit"
            robot.last_abort_reason = None
            print("  强侧到达上限且物体仍稳定，停止补力，按当前开口继续运输")
            return True
    if reason is not None or contacts != {"left", "right"} or peak > force_ceiling:
        # Return to the last known bounded aperture before reporting failure.
        robot.grasp_to(old_ctrl, 0.10)
        robot.last_abort_reason = reason or "adaptive_preload_rejected"
        print(
            f"  ✗ 自适应预紧未通过：接触={sorted(contacts)}，"
            f"力={after_forces['left']:.2f}/{after_forces['right']:.2f} N"
        )
        return False
    monitor.adaptive_preload_seen_mm = max(
        monitor.adaptive_preload_seen_mm,
        (monitor.initial_gripper_ctrl - float(robot.data.ctrl[-1])) * 80.0 / 255.0,
    )
    monitor.preload_slip_reference_m = slip_after
    monitor.recovery_stage = "slowed"
    monitor.recovery_signal_segment = monitor.segment_index
    monitor.stability_events.append({
        "action": "preload_completed",
        "segment_index": monitor.segment_index,
        "left_normal_force_n": float(after_forces["left"]),
        "right_normal_force_n": float(after_forces["right"]),
    })
    print(
        f"  ✓ 自适应预紧后左/右={after_forces['left']:.2f}/"
        f"{after_forces['right']:.2f} N"
    )
    return True


def _holding_check(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> tuple[bool, dict[str, object]]:
    """Local proof that the object is still in the hand before a transport resume.

    Requires both finger pads in contact, the object centred under the hand,
    and the object clear of the table.  These are the same limits the
    transport monitor and the post-transit check already use.
    """
    contacts, forces = _contact_metrics(model, data, body_name)
    hand = _hand_xyz(model, data)
    held = _body_xyz(model, data, body_name)
    bottom = _body_bottom_z(model, data, body_name)
    xy_offset = float(np.linalg.norm(held[:2] - hand[:2]))
    report = {
        "bilateral_contacts": contacts == {"left", "right"},
        "minimum_normal_force_n": round(min(forces.values()) if forces else 0.0, 3),
        "object_to_hand_xy_m": round(xy_offset, 4),
        "object_bottom_z_m": round(bottom, 4),
    }
    held_ok = contacts == {"left", "right"} and xy_offset <= 0.07 and bottom >= 0.012
    return held_ok, report


def _grasp_symptom(
    before: np.ndarray,
    after_close: np.ndarray,
    after_lift: np.ndarray | None,
    contacts: set[str],
    hand_xy: np.ndarray,
    hand_rise: float,
) -> str:
    """Decide whether the fingers merely touched the object or actually took it."""
    if contacts != {"left", "right"}:
        return "no_contact"
    if after_close[2] < before[2] - 0.008 or np.linalg.norm(after_close[:2] - before[:2]) > 0.018:
        return "tipped"
    if after_lift is None:
        return "not_lifted"
    rise = float(after_lift[2] - before[2])
    stayed = float(np.linalg.norm(after_lift[:2] - hand_xy)) <= 0.06
    followed = rise >= 0.03 and rise >= 0.55 * max(hand_rise, 0.01)
    if followed and stayed:
        return "lifted"
    return "not_lifted"


def _grasp_endpoint_hit(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_xyz: np.ndarray,
    yaw_rad: float,
    body_name: str,
) -> str | None:
    """Preflight one grasp candidate at its final IK pose on a clone."""
    q_now = _arm_q(model, data)
    try:
        q_goal = solve_hand_ik(
            model, np.asarray(target_xyz, dtype=float), seed=q_now,
            yaw_rad=float(yaw_rad),
        )
    except RuntimeError:
        return "ik_unreachable"
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    for name, value in zip(ARM_JOINTS, q_goal, strict=True):
        scratch.qpos[int(model.joint(name).qposadr[0])] = float(value)
    # The endpoint is approached open; validate the wider robot shape.
    scratch.qpos[int(model.joint("finger_joint1").qposadr[0])] = 0.04
    scratch.qpos[int(model.joint("finger_joint2").qposadr[0])] = 0.04
    mujoco.mj_forward(model, scratch)
    return _scene_hit(model, scratch, {body_name})


def _move_to_grasp(
    robot: PandaController,
    model: mujoco.MjModel,
    grasp: np.ndarray,
    yaw: float,
    approach: str,
    pregrasp_distance: float,
    finger_z: float,
    body_name: str,
    obstacle_top: float,
    obstacles: list[dict] | None,
    protrusion_rad: float | None = None,
) -> bool:
    """Descend onto the object, or slide in from an open side at the grasp height."""
    if approach != "horizontal":
        print("→ 已在物体正上方，下降改成几步完成，不再一点点磨")
        return move_hand_clear(
            robot, model, grasp, yaw_rad=yaw, obstacle_top=obstacle_top,
            descend_ignore={body_name}, obstacles=obstacles, quick_descend=True,
        )
    approach_dir = np.array((np.cos(yaw), np.sin(yaw)))
    chosen = None
    for sign in (1.0, -1.0):
        candidate = grasp[:2] - approach_dir * sign * pregrasp_distance
        if not approach_side_clears_protrusion(candidate, grasp[:2], protrusion_rad):
            continue
        occupied = _surface_under_xy(model, robot.data, candidate, {body_name})
        if occupied < finger_z:
            chosen = candidate
            break
    if chosen is None and protrusion_rad is not None:
        print("→ 水平接近会经过侧向凸起或侧面被挡，改为从上方下降")
        return move_hand_clear(
            robot, model, grasp, yaw_rad=yaw, obstacle_top=obstacle_top,
            descend_ignore={body_name}, obstacles=obstacles, quick_descend=True,
        )
    if chosen is None:
        chosen = grasp[:2] - approach_dir * pregrasp_distance
    pregrasp = np.array((float(chosen[0]), float(chosen[1]), float(grasp[2])))
    print(
        f"→ 水平接近：先降到侧面 {np.round(pregrasp, 3)} m，"
        f"再平移 {pregrasp_distance:.2f} m"
    )
    if not move_hand_clear(
        robot, model, pregrasp, yaw_rad=yaw, obstacle_top=obstacle_top,
        descend_ignore={body_name}, obstacles=obstacles, quick_descend=True,
    ):
        return False
    steps = max(1, int(np.ceil(pregrasp_distance / 0.04)))
    for index in range(1, steps + 1):
        point = pregrasp.copy()
        point[:2] = chosen + (grasp[:2] - chosen) * (index / steps)
        point[2] = grasp[2]
        if not nudge_hand(robot, model, point, yaw_rad=yaw, ignore_bodies={body_name}):
            print("  ✗ 水平进入被挡住")
            robot.last_abort_reason = "grasp_approach_blocked"
            return False
    return True


def _grasp_failure(
    symptom: str,
    contacts: set[str],
    finger_z: float,
    top_z: float,
) -> tuple[str, str]:
    clearance = float(finger_z) - float(top_z)
    if symptom == "no_contact" and clearance > 0.015:
        return (
            "no_contact",
            "fingers remained above the object top and never contacted it",
        )
    if symptom == "no_contact":
        return "no_contact", "fingers closed without contacting the object"
    if symptom in {"left_only", "right_only"}:
        return symptom, f"{symptom.removesuffix('_only')} finger alone contacted the object"
    if symptom == "bilateral_loading":
        return (
            "weak_force_insufficient",
            "both fingers contacted, but the weak-side absolute load stayed below the trial-lift floor",
        )
    if symptom in {"bilateral_unbalanced", "force_limit"}:
        return (
            "bilateral_unbalanced",
            "both fingers contacted, but the weak side stayed below its load floor or the strong side reached its limit",
        )
    if symptom == "object_displaced":
        return "object_moved", "the object moved while contact force was being established"
    if symptom == "tipped":
        return "object_moved", "the object was pushed or tilted before it was held"
    if symptom == "not_lifted" and contacts == {"left", "right"}:
        return "lift_failed", "contact was present, but the object did not rise"
    if symptom == "not_lifted":
        return "slipped", "contact was lost while lifting"
    if symptom == "approach_blocked":
        return "approach_blocked", "the horizontal approach was blocked"
    return symptom, symptom


def _centering_correction_in_gripper_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    high_side: str,
    magnitude_m: float = 0.0010,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn signed pad imbalance into a correction in the hand frame.

    The correction is intentionally computed and clipped in gripper
    coordinates, then transformed to base/world XY.  A wrist-yaw change can
    therefore never reinterpret `left > right` as a fixed world-X motion.
    """
    hand_id = model.body("hand").id
    rotation = data.xmat[hand_id].reshape(3, 3).copy()
    hand = data.xpos[hand_id].copy()
    finger = data.xpos[model.body(f"{high_side}_finger").id].copy()
    toward_high_world = finger - hand
    toward_high_local = rotation.T @ toward_high_world
    toward_high_local[2] = 0.0
    norm = float(np.linalg.norm(toward_high_local[:2]))
    if norm < 1e-8:
        return np.zeros(2), np.zeros(2)
    correction_local = toward_high_local / norm * float(magnitude_m)
    correction_world = rotation @ correction_local
    return correction_local[:2].copy(), correction_world[:2].copy()


def execute_pick_and_place(
    task: TaskSpec,
    model: mujoco.MjModel,
    robot: PandaController,
    target: Detection,
    reference: Detection | None,
    renderer: mujoco.Renderer | None = None,
    obstacle_top_m: float | None = None,
    obstacles: list[dict] | None = None,
    checkpoint_reviewer: Callable[
        [str, dict[str, object], np.ndarray | None], "CheckpointReview | bool"
    ] | None = None,
    route_policy: str = "auto",
    grasp_reasoner: Callable[
        [dict[str, object], np.ndarray | None], GraspStrategyUpdate
    ] | None = None,
    initial_grasp: GraspStrategyUpdate | None = None,
    start_from: str = "pick",
    task_log: "TaskLog | None" = None,
    pick_strategy: "PickStrategy | None" = None,
    plan_brief_reviewer: Callable[[dict[str, object]], "CheckpointReview"] | None = None,
    transport_candidate_selector: Callable[
        [list[dict[str, object]], str], tuple[str | None, str]
    ] | None = None,
) -> bool:
    """Pick, confirm the object actually rose with the hand, then place clear of the support.

    ``initial_grasp`` seeds the first attempt with a strategy the recovery
    layer already clipped.  ``start_from="transport"`` resumes after a
    transport failure while the object is still verifiably in the hand.
    ``task_log`` receives checkpoint evidence, grasp diagnoses and the final
    placement verdict; it never changes what the robot does.  ``pick_strategy``
    carries the manager's five enumerated pre-execution choices; only the
    local mapping below turns them into approach, pre-opening and lift height.
    """

    def log_checkpoint(stage: str, evidence: dict[str, object], verdict: object) -> None:
        if task_log is not None:
            task_log.add_checkpoint(stage, evidence, verdict)

    def review_transport_brief(brief: dict[str, object]) -> bool:
        """Three-state verdict on the validated transport plan, before it runs."""
        brief = {**brief, "transport_mode": task.constraints.transport_mode}
        if plan_brief_reviewer is None:
            if task_log is not None:
                task_log.add_plan_brief(brief, "local_only")
            return True
        try:
            verdict = plan_brief_reviewer(brief)
        except Exception as error:
            if task_log is not None:
                task_log.add_plan_brief(brief, f"error:{type(error).__name__}")
            print(f"  ✗ 路径述职审核异常：{error}；安全停止")
            robot.last_abort_reason = "llm_checkpoint_error:plan_brief"
            return False
        if task_log is not None:
            task_log.add_plan_brief(brief, verdict)
        print(
            f"  路径述职审核 ({verdict.backend}): {verdict.decision}"
            + (f"/{verdict.adjustment}" if verdict.adjustment else "")
            + f" — {verdict.reason}"
        )
        if verdict.may_continue:
            return True
        if verdict.wants_adjust:
            already_applied = (
                (verdict.adjustment == "raise_transport_height" and route_policy == "high_clearance")
                or (verdict.adjustment == "slow_down"
                    and task.constraints.transport_mode == "fragile")
            )
            if already_applied:
                # The requested tightening is already in force and the path
                # is locally validated; repeating the same adjust would only
                # burn recovery rounds without changing anything.
                print(
                    f"  调整 {verdict.adjustment} 已经生效（route={route_policy}, "
                    f"mode={task.constraints.transport_mode}），按本地验证继续"
                )
                if task_log is not None:
                    task_log.add_checkpoint("plan_brief_adjust_already_applied", {
                        "adjustment": verdict.adjustment, "route_policy": route_policy,
                        "transport_mode": task.constraints.transport_mode,
                    }, "continue")
                return True
            robot.last_abort_reason = f"checkpoint_adjust:plan_brief:{verdict.adjustment}"
            robot.last_checkpoint_adjustment = verdict.adjustment
            return False
        robot.last_abort_reason = "plan_brief_rejected"
        return False

    def review_checkpoint(stage: str, evidence: dict[str, object]) -> bool:
        if checkpoint_reviewer is None:
            log_checkpoint(stage, evidence, "local_only")
            return True
        frame = None
        if renderer is not None:
            lock = robot.data_lock if robot.data_lock is not None else nullcontext()
            with lock:
                frame, _ = render_rgbd(renderer, robot.data)
        try:
            verdict = checkpoint_reviewer(stage, evidence, frame)
        except Exception as error:
            print(f"  ✗ LLM checkpoint {stage} 异常：{error}；安全停止")
            robot.last_abort_reason = f"llm_checkpoint_error:{stage}"
            log_checkpoint(stage, evidence, f"error:{type(error).__name__}")
            return False
        log_checkpoint(stage, evidence, verdict)
        if isinstance(verdict, bool):
            approved = verdict
        else:
            approved = bool(verdict.may_continue)
            if verdict.wants_adjust:
                # The LLM asked for one whitelisted adjustment.  Stop this
                # attempt cleanly; the recovery layer maps the word to a
                # conservative local action and spends one round on it.
                print(
                    f"  ↺ LLM 要求调整 {stage}: {verdict.adjustment}；"
                    "交由恢复层处理"
                )
                robot.last_abort_reason = f"checkpoint_adjust:{stage}:{verdict.adjustment}"
                robot.last_checkpoint_adjustment = verdict.adjustment
                return False
        if not approved:
            print(f"  ✗ LLM 审核未放行 {stage}；不绕过审核继续执行")
            robot.last_abort_reason = f"llm_checkpoint_rejected:{stage}"
        return approved

    if route_policy not in {
        "auto", "clockwise", "counterclockwise", "high_clearance",
        "stability_first", "time_first",
    }:
        raise ValueError(f"未授权的 route_policy: {route_policy}")
    robot.plan_peak_force_hint_n = 0.0
    robot.last_plan_brief = None
    object_record = record_for_color(task.object_color)
    if object_record is None:
        object_record = generic_record(task.object_color, task.object_name)
    known_bodies = {record.body_name for record in OBJECTS}
    body_name = object_record.body_name
    if task.body_name:
        try:
            model.body(task.body_name)
        except KeyError:
            pass
        else:
            body_name = task.body_name
            if body_name not in known_bodies:
                dims = tuple(task.measured_size_m or object_record.dimensions_m)
                object_record = replace(
                    object_record, body_name=body_name, dimensions_m=dims,
                )
    if body_name is None or (body_name not in known_bodies and body_name != task.body_name):
        print(
            f"✗ {task.object_color} {task.object_name} 已通过通用圆柱低置信度规划，"
            "但当前 MuJoCo 场景没有可绑定的碰撞 body；本地执行层安全停止"
        )
        robot.last_abort_reason = "unregistered_object_body_missing"
        return False
    half_height = object_record.half_height_m
    hand_offset = object_record.hand_offset_m
    fallback_width = object_record.fallback_width_m
    obstacle_top = 0.0 if obstacle_top_m is None else float(obstacle_top_m)
    if renderer is not None:
        lock = robot.data_lock if robot.data_lock is not None else nullcontext()
        with lock:
            _, depth = render_rgbd(renderer, robot.data)
        obstacle_top = max(obstacle_top, workspace_ceiling_m(model, robot.data, CAMERA_NAME, depth))
    print(f"深度测得场景最高 {obstacle_top:.3f} m，夹爪转运将高于它")

    # The initial scan records the support, but we deliberately do not visit
    # it yet.  The short order is: scan -> object -> grasp -> support -> place.
    support_xy = None if reference is None else reference.base_xyz_m[:2].copy()
    support_z = None if reference is None else float(reference.base_xyz_m[2])
    locked_support_xy = (
        None if support_xy is None else np.array(support_xy, dtype=float, copy=True)
    )
    place_reference_source = "locked"
    robot.last_place_disclosure = None
    if task_log is not None and locked_support_xy is not None:
        task_log.add_checkpoint("locked_support_reference", {
            "locked_support_xy_m": locked_support_xy.tolist(),
        }, "locked")
    long_cross_base_transport = bool(
        support_xy is not None
        and task.object_name in {"box", "cube"}
        and not _segment_clears_base(target.base_xyz_m[:2], support_xy)
    )

    if start_from not in {"pick", "transport"}:
        raise ValueError(f"未授权的 start_from: {start_from}")
    look_xy = target.base_xyz_m[:2].copy()
    remembered_top = float(target.base_xyz_m[2])
    lifted = False
    held_yaw = 0.0
    gap = _holding_gap(fallback_width)
    grasp_xy_bias = np.zeros(2, dtype=float)
    centering_samples: list[tuple[float, float]] = []
    signed_centering_applied = 0.0
    pending_centering_recovery: dict[str, object] | None = None
    handle_direction_history: list[float] = []
    grasp_memory = getattr(robot, "grasp_attempt_memory", None)
    if not isinstance(grasp_memory, dict):
        grasp_memory = {}
        robot.grasp_attempt_memory = grasp_memory
    body_memory = grasp_memory.setdefault(
        body_name, {"entries": [], "anchor": None},
    )
    grasp_failure_entries = list(body_memory["entries"])
    grasp_pose_anchor = body_memory["anchor"]
    executed_region = "middle_side"
    executed_approach = "vertical"
    if start_from == "transport":
        held, holding_report = _holding_check(model, robot.data, body_name)
        if held:
            lifted = True
            held_yaw = _hand_yaw(model, robot.data)
            gap = float(robot.data.ctrl[-1]) / 255.0 * 0.08
            print(
                "→ 运输重规划：本地握持核查通过，跳过抓取段，"
                f"{json.dumps(holding_report, ensure_ascii=False)}"
            )
        else:
            print(
                "  运输重规划：握持核查未通过，退回完整抓取，"
                f"{json.dumps(holding_report, ensure_ascii=False)}"
            )
            start_from = "pick"
    if start_from == "pick" and float(robot.data.ctrl[-1]) < 200.0 and not robot.release(0.35):
        return False
    strategy_approach = (
        pick_strategy.approach if pick_strategy is not None else task.approach_direction
    )
    grip_width_choice = (
        pick_strategy.grip_width
        if pick_strategy is not None else object_record.default_grip_width
    )
    lift_offset = _lift_offset_for(
        pick_strategy.lift_height if pick_strategy is not None else "medium"
    )
    if pick_strategy is not None:
        print(
            f"→ 经理事前策略 ({pick_strategy.backend}): approach={pick_strategy.approach}, "
            f"grip_width={pick_strategy.grip_width}, lift_height={pick_strategy.lift_height}, "
            f"transport={pick_strategy.transport}, speed={pick_strategy.speed_profile}"
            + (f"；已裁剪 {list(pick_strategy.clipped_fields)}" if pick_strategy.clipped_fields else "")
        )
    active_grasp = {
        "grasp_strategy": task.grasp_strategy,
        "approach_direction": (
            strategy_approach
            if strategy_approach in {"vertical", "horizontal"} else "vertical"
        ),
        "height_ratio": None,
        "lateral_offset_m": 0.0,
        "yaw_delta_rad": 0.0,
        "pregrasp_distance_m": 0.08,
        "grasp_region": (
            "middle_side" if body_name in CUP_BODIES else "auto"
        ),
        "yaw_policy": (
            "avoid_handle" if body_name in {"red_cup", "cyan_mug"}
            else "preserve_wrist" if body_name == "purple_bottle"
            else "geometry"
        ),
    }
    grasp_plan_state: dict[str, object] = {
        "plan": None, "blocked": set(), "measured": {}, "anchor": None,
    }

    def choose_grasp_plan(span_now: dict | None) -> None:
        from task_planner import select_grasp_plan

        if grasp_plan_state["plan"] is not None:
            return
        support = task.reference or "none"
        measured = {
            "shape": object_record.shape_prior,
            "color": task.object_color,
            "object_id": task.object_id,
            "width_m": float(object_record.fallback_width_m),
            "has_handle": bool(span_now and float(span_now.get("has_handle", 0.0)) > 0.5),
            "handle_direction_rad": None if span_now is None else span_now.get("handle_direction_rad"),
            "support": support,
        }
        leader = getattr(grasp_reasoner, "__self__", None)
        proposer = None
        if leader is not None and getattr(leader, "online", False):
            proposer = leader.propose_grasp_plan
        plan, info = select_grasp_plan(
            task, measured, grasp_plan_state["blocked"], proposer=proposer,  # type: ignore[arg-type]
        )
        grasp_plan_state["plan"] = plan
        grasp_plan_state["measured"] = measured
        yaw_mode = plan.plan_pick_argument("yaw_mode")
        if yaw_mode == "avoid_handle":
            active_grasp["yaw_policy"] = "avoid_handle"
        elif yaw_mode == "preserve_wrist":
            active_grasp["yaw_policy"] = "preserve_wrist"
        elif yaw_mode in {"geometry", "face"}:
            active_grasp["yaw_policy"] = "geometry"
        if task_log is not None:
            for rejection in info.get("rejections") or []:
                task_log.add_checkpoint("case_rejected", rejection, "rejected")
            if info.get("verdict") == "case_matched":
                task_log.add_checkpoint("case_matched", {
                    "case_id": info.get("case_id"), "criteria": info.get("criteria"),
                }, "matched")
            task_log.add_checkpoint("grasp_plan_decision", {
                **plan.to_dict(), "verdict": info.get("verdict"),
            }, info.get("verdict"))

    def remember_success_case() -> None:
        plan = grasp_plan_state["plan"]
        measured = grasp_plan_state["measured"]
        if plan is None or not isinstance(measured, dict) or task_log is None:
            return
        from task_log import append_grasp_case
        from task_planner import success_case

        append_grasp_case(success_case(task, measured, plan))

    def apply_update(update: GraspStrategyUpdate) -> None:
        active_grasp["grasp_strategy"] = update.grasp_strategy
        active_grasp["approach_direction"] = update.approach_direction
        active_grasp["height_ratio"] = float(update.grasp_height_ratio)
        active_grasp["lateral_offset_m"] = float(update.lateral_offset_m)
        active_grasp["yaw_delta_rad"] = float(update.yaw_delta_rad)
        active_grasp["pregrasp_distance_m"] = float(update.pregrasp_distance_m)
        active_grasp["grasp_region"] = update.grasp_region
        active_grasp["yaw_policy"] = update.yaw_policy

    if initial_grasp is not None and not lifted:
        seeded = initial_grasp
        if getattr(robot, "last_grasp_diagnosis", None):
            seeded = clip_grasp_strategy(initial_grasp, robot.last_grasp_diagnosis)
        apply_update(seeded)
        print(
            f"→ 采用恢复层给出的初始抓取方案 ({seeded.backend}): "
            f"{seeded.grasp_strategy} / {seeded.approach_direction}，"
            f"语义区域 {seeded.grasp_region}，偏航策略 {seeded.yaw_policy}"
        )
        print("  理由:", seeded.reason)

    def adopt_replan(diagnosis: dict[str, object]) -> bool:
        """Ask for a new grasp.  False means this attempt was the last one."""
        nonlocal grasp_failure_entries
        robot.last_grasp_diagnosis = diagnosis
        failure_type = str(diagnosis["failure"]["type"])
        search_type = failure_type_for_search(diagnosis)
        failed_orientation = diagnosis.get("object", {}).get("orientation_deg")
        if failed_orientation is not None:
            failed_yaw = float(np.deg2rad(float(failed_orientation)))
            before = len(grasp_failure_entries)
            grasp_failure_entries = remember_grasp_failure(
                grasp_failure_entries,
                yaw_rad=failed_yaw,
                region=executed_region,
                approach=executed_approach,
                failure_type=search_type,
                attempt=attempt,
            )
            body_memory["entries"] = grasp_failure_entries
            if len(grasp_failure_entries) != before:
                recorded = grasp_failure_entries[-1]
                log_checkpoint(
                    "grasp_axis_rejected",
                    {
                        "attempt": attempt,
                        "candidate_id": recorded["candidate_id"],
                        "failure_type": failure_type,
                        "search_type": search_type,
                        "search_dof": grasp_failure_dof(search_type),
                        "status": recorded["status"],
                        "yaw_rad": failed_yaw,
                        "memory": [
                            {
                                "candidate_id": item["candidate_id"],
                                "status": item["status"],
                                "failure_type": item["failure_type"],
                            }
                            for item in grasp_failure_entries
                        ],
                    },
                    str(recorded["status"]),
                )
                print(
                    f"  候选 {recorded['candidate_id']} 记为 {recorded['status']}"
                    f"（{grasp_failure_dof(search_type)}），下一帧优先未试过的自由度"
                )
        robot.last_abort_reason = f"grasp_{failure_type}"
        print("  失败诊断:", json.dumps(diagnosis, ensure_ascii=False))
        if task_log is not None:
            task_log.add_failure(
                f"grasp_{failure_type}", diagnosis, stage="grasp", attempt=attempt,
            )
        # max_retries means retries *after* the initial attempt.  The old
        # loop treated it as total attempts and silently provided only two
        # retries when the plan explicitly requested three.
        if attempt > task.constraints.max_retries:
            print("  抓取次数已用完，不再生成下一次方案")
            return False
        failure_frame = None
        if renderer is not None:
            lock = robot.data_lock if robot.data_lock is not None else nullcontext()
            with lock:
                failure_frame, _ = render_rgbd(renderer, robot.data)
        # Contact centering and a freshly moved object are local metric
        # failures.  Reobserve and correct them twice without spending an LLM
        # round trip; escalate only after repetition or a strategic blockage.
        local_recoverable = failure_type in {
            "left_only", "right_only", "bilateral_unbalanced",
            "object_moved", "no_contact", "lift_failed", "slipped",
        }
        use_local_recovery = grasp_reasoner is None or (
            local_recoverable and attempt < 3
        )
        update = (
            offline_grasp_replan(diagnosis)
            if use_local_recovery
            else grasp_reasoner(diagnosis, failure_frame)
        )
        # Re-apply the local envelope even for custom/online reasoners.
        update = clip_grasp_strategy(update, diagnosis)
        if task_log is not None:
            task_log.add_recovery_round({
                "scope": "grasp_attempt", "attempt": attempt,
                "trigger": f"grasp_{failure_type}", "action": update.action,
                "backend": update.backend, "strategy_update": update.to_dict(),
            })
        if update.action == "safe_stop":
            print(f"  抓取重规划 ({update.backend}): 安全停止 — {update.reason}")
            return False
        apply_update(update)
        current_plan = grasp_plan_state["plan"]
        if current_plan is not None:
            from task_planner import revise_grasp_plan

            revised = revise_grasp_plan(current_plan, failure_type)
            case_id = getattr(current_plan, "case_id", None)
            if case_id:
                grasp_plan_state["blocked"].add(case_id)  # type: ignore[union-attr]
            grasp_plan_state["plan"] = revised
            if task_log is not None:
                task_log.add_checkpoint(
                    "grasp_plan_decision", revised.to_dict(), "revised",
                )
        print(
            f"  抓取重规划 ({update.backend}): {update.action}，"
            f"{update.grasp_strategy} / {update.approach_direction}，"
            f"语义区域 {update.grasp_region}，偏航策略 {update.yaw_policy}；"
            "毫米级位姿由下一帧 RGB-D 本地重算"
        )
        print("  理由:", update.reason)
        return True

    attempts = [] if lifted else range(1, task.constraints.max_retries + 2)
    for attempt in attempts:
        if not robot._can_continue():
            return False
        print(f"→ 第 {attempt} 次抓取：先快速到物体正上方，再微调")
        detection: Detection | None = target if renderer is None else None
        span = None
        if renderer is not None:
            hover_z = float(np.clip(remembered_top + 0.22, 0.34, 0.55))
            # Keep the wrist that the scan already settled.  Forcing yaw 0
            # here twisted joint 7 by more than 90 degrees and the gripper
            # spun a turn while it was only flying over to the object.
            travel_yaw = _hand_yaw(model, robot.data)
            if long_cross_base_transport:
                # Across-base transport has sustained lateral acceleration.
                # Approach directly on the orthogonal box face so IK selects
                # the matching branch before descent, instead of trying to
                # rotate to it at the hover singularity.
                travel_yaw = float(np.pi / 2.0)
            detection_now, span, obstacle_top = _observe_down(
                robot, model, renderer, task.object_color, task.object_name,
                look_xy, hover_z, obstacle_top, travel_yaw,
                ignore_bodies={body_name}, obstacles=obstacles, fast=True,
            )
            if detection_now is not None:
                detection = detection_now
                print("  俯视复核:", np.round(detection.base_xyz_m, 3), "m")
        if detection is None:
            if robot.last_abort_reason == "motion_plan_failed":
                print("  ✗ 到目标观察位的运动规划失败；不是视觉漏检，不重复同一抓取计划")
                return False
            print("  本次新画面没有找到目标，不使用旧抓取点")
            continue
        grasp_failure_entries, grasp_pose_anchor = refresh_grasp_attempt_memory(
            grasp_failure_entries,
            grasp_pose_anchor,
            detection.base_xyz_m[:2],
        )
        body_memory["entries"] = grasp_failure_entries
        body_memory["anchor"] = grasp_pose_anchor
        top_z = float(span["top_z"]) if span else float(detection.base_xyz_m[2])
        handle_direction_for_attempt: float | None = None
        handle_direction_was_stabilized = False
        if span is not None and span.get("has_handle", 0.0) > 0.5:
            raw_handle_direction = float(span["handle_direction_rad"])
            handle_direction_for_attempt, handle_direction_was_stabilized = (
                stabilize_handle_direction(
                    handle_direction_history, raw_handle_direction
                )
            )
            span = dict(span)
            span["handle_direction_rad"] = handle_direction_for_attempt
            span["handle_avoid_yaw_rad"] = handle_avoid_yaw_from_direction(
                handle_direction_for_attempt
            )
            if handle_direction_was_stabilized:
                print(
                    "  把手方向相邻读数跳变超过 10°；"
                    f"采用近三次中值 {np.rad2deg(handle_direction_for_attempt):+.1f}°"
                )
        current_yaw = _hand_yaw(model, robot.data)
        choose_grasp_plan(span if isinstance(span, dict) else None)
        plan_now = grasp_plan_state["plan"]
        if (
            plan_now is not None
            and plan_now.plan_pick_argument("region") == "body"
            and isinstance(span, dict)
            and span.get("body_region_center_x") is not None
        ):
            detection.base_xyz_m = np.array(detection.base_xyz_m, dtype=float, copy=True)
            detection.base_xyz_m[0] = float(span["body_region_center_x"])
            detection.base_xyz_m[1] = float(span["body_region_center_y"])
            grasp_plan_state["anchor"] = detection.base_xyz_m[:2].copy()
        candidates = _generate_grasp_candidates(
            body_name,
            task.object_name,
            span,
            fallback_width,
            current_yaw,
            str(active_grasp["grasp_region"]),
            str(active_grasp["yaw_policy"]),
        )
        candidates = order_grasp_candidates(
            candidates,
            grasp_failure_entries,
            str(active_grasp["approach_direction"]),
        )
        plan = None if not candidates else (
            candidates[0].yaw_rad, candidates[0].width_m
        )
        if long_cross_base_transport and span is not None:
            # A nearly square cube has no stable PCA major axis: a few depth
            # pixels rotated the reported orange face from 90 to 37 degrees,
            # leaving both fingers outside the cube.  The camera-frame extent
            # is deterministic and matches the +90 degree IK branch selected
            # before descent.  Use PCA only when that cardinal face does not
            # fit the gripper.
            cardinal_width = float(span.get("extent_x", 0.0))
            if 0.025 <= cardinal_width <= MAX_GRASP_WIDTH:
                plan = (float(np.pi / 2.0), cardinal_width)
            else:
                major_width = float(span.get("major_m", 0.0))
                major_yaw = float(span.get("major_yaw_rad", 0.0))
                if 0.025 <= major_width <= MAX_GRASP_WIDTH:
                    plan = (major_yaw, major_width)
        if plan is None:
            measured = "unknown" if span is None else (
                f"minor={span['minor_m']:.3f} m, major={span['major_m']:.3f} m"
            )
            print(
                f"  ✗ 当前物体两个可见夹持宽度都超过 {MAX_GRASP_WIDTH:.3f} m"
                f"（{measured}），拒绝硬夹"
            )
            look_xy = detection.base_xyz_m[:2].copy()
            remembered_top = top_z
            continue
        chosen_candidate = candidates[0]
        planned_yaw, width = plan
        planned_yaw = float(planned_yaw) + float(active_grasp["yaw_delta_rad"])
        yaw = _same_grasp_yaw(planned_yaw, current_yaw)
        yaw_turn = (yaw - current_yaw + np.pi) % (2.0 * np.pi) - np.pi
        gap = _holding_gap(width)
        grasp = detection.base_xyz_m.copy()
        grasp[:2] += grasp_xy_bias
        bottom_z = top_z - 2.0 * half_height
        support_top = _surface_under_xy(model, robot.data, grasp[:2], {body_name})
        if body_name in CUP_BODIES or active_grasp["grasp_region"] != "auto":
            height_ratio = chosen_candidate.height_ratio
        elif active_grasp["height_ratio"] is None:
            legacy_palm = top_z + hand_offset - object_record.grasp_depth_m
            legacy_finger = legacy_palm - FINGER_PAD_BELOW_HAND
            height_ratio = (legacy_finger - bottom_z) / max(top_z - bottom_z, 1e-3)
        else:
            height_ratio = float(active_grasp["height_ratio"])
        palm_z, finger_z, height_ratio = palm_from_height_ratio(
            bottom_z, top_z, height_ratio, support_top
        )
        if len(candidates) > 1:
            candidate_checks: list[dict[str, object]] = []
            reachable_choice: tuple[GraspCandidate, float, float] | None = None
            for candidate in candidates:
                candidate_planned_yaw = (
                    float(candidate.yaw_rad)
                    + float(active_grasp["yaw_delta_rad"])
                )
                candidate_yaw = _same_grasp_yaw(
                    candidate_planned_yaw, current_yaw
                )
                candidate_target = grasp.copy()
                candidate_target[:2] += np.array((
                    -np.sin(candidate_yaw), -np.cos(candidate_yaw),
                )) * float(active_grasp["lateral_offset_m"])
                candidate_target[2] = palm_z
                endpoint_hit = _grasp_endpoint_hit(
                    model, robot.data, candidate_target,
                    candidate_yaw, body_name,
                )
                candidate_checks.append({
                    "candidate_id": grasp_candidate_id(
                        candidate_yaw,
                        candidate.region,
                        str(active_grasp["approach_direction"]),
                    ),
                    "status": grasp_candidate_status(
                        candidate_yaw,
                        candidate.region,
                        str(active_grasp["approach_direction"]),
                        grasp_failure_entries,
                    ),
                    "source": candidate.source,
                    "yaw_rad": float(candidate_yaw),
                    "width_m": float(candidate.width_m),
                    "endpoint_hit": endpoint_hit,
                })
                if endpoint_hit is None and reachable_choice is None:
                    reachable_choice = (
                        candidate, candidate_planned_yaw, candidate_yaw,
                    )
            if reachable_choice is not None:
                chosen_candidate, planned_yaw, yaw = reachable_choice
                width = float(chosen_candidate.width_m)
                gap = _holding_gap(width)
                yaw_turn = (
                    yaw - current_yaw + np.pi
                ) % (2.0 * np.pi) - np.pi
            log_checkpoint(
                "grasp_candidate_preflight",
                {
                    "attempt": attempt,
                    "checks": candidate_checks,
                    "selected_source": chosen_candidate.source,
                    "selected_yaw_rad": float(yaw),
                },
                "selected" if reachable_choice is not None else "no_clear_endpoint",
            )
            if candidate_checks and candidate_checks[0]["endpoint_hit"] is not None:
                if reachable_choice is not None:
                    print(
                        "  首选夹持面终点被挡（"
                        f"{candidate_checks[0]['endpoint_hit']}）；"
                        f"改用 {chosen_candidate.source}"
                    )
                else:
                    print("  所有夹持面终点均被挡，不尝试闭爪")
                    robot.last_abort_reason = "grasp_no_clear_candidate"
                    continue
        closing = np.array((-np.sin(yaw), -np.cos(yaw)))
        grasp[:2] = grasp[:2] + closing * float(active_grasp["lateral_offset_m"])
        grasp[2] = palm_z
        neighbor_top, neighbor_gap = _nearest_obstacle(
            model, robot.data, detection.base_xyz_m[:2], {body_name}
        )
        executed_region = chosen_candidate.region
        executed_approach = str(active_grasp["approach_direction"])
        selected_status = grasp_candidate_status(
            yaw, executed_region, executed_approach, grasp_failure_entries,
        )
        print(
            f"  抓取方案 {active_grasp['grasp_strategy']} / "
            f"{executed_approach}："
            f"候选={grasp_candidate_id(yaw, executed_region, executed_approach)}"
            f"/{selected_status}，来源={chosen_candidate.source}，"
            f"指尖 {finger_z:.3f} m，上沿 {top_z:.3f} m，"
            f"手心 {palm_z:.3f} m，正下方支撑 {support_top:.3f} m"
        )
        if neighbor_gap < 0.20 and neighbor_top > finger_z:
            print(
                f"  旁边 {neighbor_gap:.2f} m 处有 {neighbor_top:.3f} m 的物体，"
                "不因此把指尖抬过上沿"
            )
        print(
            f"  抓取点 {np.round(grasp, 3)} m，物体宽 {width:.3f} m。"
            f"夹持面偏航 {np.rad2deg(planned_yaw):+.0f}°，"
            f"腕部从 {np.rad2deg(current_yaw):+.0f}° 收到 {np.rad2deg(yaw):+.0f}°"
            f"（转 {np.rad2deg(yaw_turn):+.0f}°，同一夹持面，不空转一圈）。"
            "手心留在上沿上方，指尖落在物体侧面"
        )
        before_approach = _body_xyz(model, robot.data, body_name)
        if not _align_wrist(robot, model, yaw):
            if robot.last_abort_reason is None:
                robot.last_abort_reason = "grasp_wrist_alignment_failed"
            grasp_failure_entries = remember_grasp_failure(
                grasp_failure_entries,
                yaw_rad=float(yaw),
                region=executed_region,
                approach=executed_approach,
                failure_type="approach_blocked",
                attempt=attempt,
            )
            body_memory["entries"] = grasp_failure_entries
            print(
                f"  候选 {grasp_candidate_id(yaw, executed_region, executed_approach)}"
                " 记为 failed_approach（approach），下一帧换接近方向或未试过的轴"
            )
            return False
        pre_open = _pre_open_gap(grip_width_choice, float(width))
        if pre_open < 0.078 - 1e-6:
            print(
                f"→ 预张开 {pre_open * 1000:.0f} mm（{grip_width_choice}，"
                f"测宽 {float(width) * 1000:.0f} mm）；夹持开口仍由接触反馈决定"
            )
            if not robot.grasp_to(_ctrl_for_gap(pre_open), 0.20):
                return False
        if not _move_to_grasp(
            robot, model, grasp, yaw, str(active_grasp["approach_direction"]),
            float(active_grasp["pregrasp_distance_m"]), finger_z, body_name,
            obstacle_top, obstacles,
            protrusion_rad=handle_direction_for_attempt,
        ):
            if robot.last_abort_reason == "motion_plan_failed":
                robot.last_abort_reason = "grasp_approach_plan_failed"
                return False
            if robot.last_abort_reason != "grasp_approach_blocked":
                return False
            blocked = _grasp_failure("approach_blocked", set(), finger_z, top_z)
            diagnosis = {
                "task": f"pick {task.object_color} {task.object_name}",
                "attempt": attempt,
                "object": {
                    "type": task.object_name,
                    "position": [float(value) for value in detection.base_xyz_m],
                    "width": float(width),
                    "height": float(top_z - bottom_z),
                    "bottom_z": float(bottom_z),
                    "top_z": float(top_z),
                    "orientation_deg": float(np.rad2deg(yaw)),
                },
                "current_grasp": {
                    "strategy": active_grasp["grasp_strategy"],
                    "approach_direction": active_grasp["approach_direction"],
                    "grasp_region": active_grasp["grasp_region"],
                    "palm_height": float(palm_z),
                    "finger_target_height": float(finger_z),
                    "grasp_height_ratio": float(height_ratio),
                },
                "observation": {
                    "cup_rim_height": float(top_z),
                    "nearest_obstacle_height": float(neighbor_top),
                    "finger_clearance_to_rim": float(finger_z - top_z),
                    "contact_force": 0.0,
                    "object_displacement": 0.0,
                    "collision": True,
                },
                "failure": {"type": blocked[0], "description": blocked[1]},
            }
            if not adopt_replan(diagnosis):
                return False
            continue
        approach_brief = getattr(robot, "last_plan_brief", None)
        if task_log is not None and isinstance(approach_brief, dict):
            task_log.add_plan_brief({**approach_brief, "stage": "grasp_approach"}, "local_only")
        before = _body_xyz(model, robot.data, body_name)
        approach_slide = float(np.linalg.norm(before[:2] - before_approach[:2]))
        if approach_slide > 0.012 or before[2] < before_approach[2] - 0.008:
            print(
                f"  ✗ 接近阶段已推动物体 {approach_slide * 1000:.0f} mm，"
                "不在错误位置闭合"
            )
            if not robot.release(0.35):
                return False
            retreat = _hand_xyz(model, robot.data)
            retreat[2] += 0.12
            if not move_hand_clear(
                robot, model, retreat, yaw_rad=yaw, obstacle_top=obstacle_top,
                ignore_bodies={body_name}, obstacles=obstacles, fast=True,
            ):
                return False
            look_xy = before[:2].copy()
            remembered_top = max(remembered_top, float(before[2]))
            pushed = _grasp_failure("tipped", set(), finger_z, top_z)
            if not adopt_replan({
                "task": f"pick {task.object_color} {task.object_name}",
                "attempt": attempt,
                "object": {
                    "type": task.object_name,
                    "position": [float(value) for value in before],
                    "width": float(width),
                    "height": float(top_z - bottom_z),
                    "bottom_z": float(bottom_z),
                    "top_z": float(top_z),
                    "orientation_deg": float(np.rad2deg(yaw)),
                },
                "current_grasp": {
                    "strategy": active_grasp["grasp_strategy"],
                    "approach_direction": active_grasp["approach_direction"],
                    "grasp_region": active_grasp["grasp_region"],
                    "palm_height": float(palm_z),
                    "finger_target_height": float(finger_z),
                    "grasp_height_ratio": float(height_ratio),
                },
                "observation": {
                    "cup_rim_height": float(top_z),
                    "nearest_obstacle_height": float(neighbor_top),
                    "finger_clearance_to_rim": float(finger_z - top_z),
                    "contact_force": 0.0,
                    "object_displacement": approach_slide,
                    "collision": False,
                },
                "failure": {"type": pushed[0], "description": pushed[1]},
            }):
                return False
            continue

        # The wide/overhead camera chooses the object and gets us close.  The
        # final millimetres belong to a local visual servo, not to an LLM
        # image guess.  Re-anchor to the object's current measured position
        # immediately before closing so small IK tracking errors cannot turn
        # into a one-pad grasp.
        # Default keeps in-loop yaw off unless this attempt's servo actually
        # settled inside 1 mm.  Objects without a servo still close as before.
        last_servo_residual_m = 1.0
        temporary_servo = (
            task.object_confidence == "low"
            and object_record.shape_prior in {"cylinder", "handled_cylinder", "box"}
        )
        if (
            body_name in BOX_BODIES or body_name == "cube" or body_name in CUP_BODIES
            or temporary_servo
        ):
            object_before_servo = _body_xyz(model, robot.data, body_name)
            residual = float("inf")
            servo_push = 0.0
            # Measured within this attempt only.  Position control can settle a
            # few millimetres short even when IK solved the exact target.  Feed
            # the observed under-travel into the next target instead of using a
            # robot- or object-specific magic offset.
            servo_feedforward = np.zeros(2, dtype=float)
            desired_hand_xy = object_before_servo[:2].copy()
            for servo_iteration in range(1, GRASP_SERVO_MAX_ITERS + 1):
                object_now = _body_xyz(model, robot.data, body_name)
                anchor_xy = grasp_plan_state["anchor"]
                servo_xy = (
                    np.asarray(anchor_xy, dtype=float)
                    if anchor_xy is not None else object_now[:2]
                )
                desired_hand_xy = (
                    servo_xy
                    + grasp_xy_bias
                    + closing * float(active_grasp["lateral_offset_m"])
                )
                hand_before_servo = _hand_xyz(model, robot.data)
                servo_error = desired_hand_xy - hand_before_servo[:2]
                servo_norm = float(np.linalg.norm(servo_error))
                error_axes = _alignment_components(servo_error, closing)
                print(
                    f"  闭爪前精对准 {servo_iteration}/{GRASP_SERVO_MAX_ITERS}: "
                    f"误差={servo_norm * 1000:.1f} mm "
                    f"(闭合轴={error_axes['closing_axis_m'] * 1000:+.1f}, "
                    f"横轴={error_axes['across_axis_m'] * 1000:+.1f} mm)"
                )
                if servo_norm <= GRASP_SERVO_TARGET_M:
                    residual = servo_norm
                    break
                correction = servo_error + servo_feedforward
                correction_norm = float(np.linalg.norm(correction))
                if correction_norm > GRASP_SERVO_MAX_STEP_M:
                    correction = (
                        correction / correction_norm * GRASP_SERVO_MAX_STEP_M
                    )
                servo_target = hand_before_servo.copy()
                servo_target[:2] += correction
                if not nudge_hand(
                    robot,
                    model,
                    servo_target,
                    yaw_rad=yaw,
                    ignore_bodies={body_name},
                ):
                    robot.last_abort_reason = "grasp_visual_servo_failed"
                    return False
                if not robot.hold(0.05):
                    return False
                hand_after_servo = _hand_xyz(model, robot.data)
                object_after_servo = _body_xyz(model, robot.data, body_name)
                actual_hand_delta = hand_after_servo[:2] - hand_before_servo[:2]
                measured_undertravel = correction - actual_hand_delta
                object_delta = object_after_servo[:2] - object_now[:2]
                desired_after = (
                    servo_xy
                    + grasp_xy_bias
                    + closing * float(active_grasp["lateral_offset_m"])
                )
                residual_vector = desired_after - hand_after_servo[:2]
                residual = float(np.linalg.norm(residual_vector))
                servo_push = float(np.linalg.norm(
                    object_after_servo[:2] - object_before_servo[:2]
                ))
                residual_axes = _alignment_components(residual_vector, closing)
                requested_axes = _alignment_components(correction, closing)
                achieved_axes = _alignment_components(actual_hand_delta, closing)
                evidence = {
                    "attempt": attempt,
                    "iteration": servo_iteration,
                    "requested_xy_m": correction.tolist(),
                    "feedforward_xy_m": servo_feedforward.tolist(),
                    "measured_undertravel_xy_m": measured_undertravel.tolist(),
                    "achieved_hand_xy_m": actual_hand_delta.tolist(),
                    "object_delta_xy_m": object_delta.tolist(),
                    "residual_xy_m": residual_vector.tolist(),
                    "residual_m": residual,
                    "requested_components": requested_axes,
                    "achieved_components": achieved_axes,
                    "residual_components": residual_axes,
                    "object_motion_from_start_m": servo_push,
                    "closing_axis_world_xy": closing.tolist(),
                    "target_m": GRASP_SERVO_TARGET_M,
                }
                log_checkpoint("grasp_visual_servo", evidence, (
                    "converged" if residual <= GRASP_SERVO_TARGET_M else "iterate"
                ))
                print(
                    f"  精对准后残差={residual * 1000:.1f} mm "
                    f"(闭合轴={residual_axes['closing_axis_m'] * 1000:+.1f}, "
                    f"横轴={residual_axes['across_axis_m'] * 1000:+.1f} mm)，"
                    f"本步手实际移动={np.round(actual_hand_delta * 1000, 1)} mm，"
                    f"物体累计移动={servo_push * 1000:.1f} mm"
                )
                # Apply only the under-travel measured by the immediately
                # preceding motion.  The next command is still clipped by the
                # same 6 mm per-step cap above.
                servo_feedforward = measured_undertravel
                if (
                    residual <= GRASP_SERVO_TARGET_M
                    or servo_push > GRASP_SERVO_OBJECT_MOTION_LIMIT_M
                ):
                    break
            object_after_servo = _body_xyz(model, robot.data, body_name)
            if (
                servo_push > GRASP_SERVO_OBJECT_MOTION_LIMIT_M
                or residual > GRASP_SERVO_TARGET_M
            ):
                print("  ✗ 精对准未通过，不在偏心位置闭爪")
                if task_log is not None:
                    task_log.add_failure(
                        "grasp_visual_servo_residual",
                        {
                            "attempt": attempt,
                            "residual_m": residual,
                            "object_motion_m": servo_push,
                            "target_m": GRASP_SERVO_TARGET_M,
                            "iterations": GRASP_SERVO_MAX_ITERS,
                        },
                        stage="grasp_visual_servo",
                    )
                if not robot.release(0.35):
                    return False
                look_xy = object_after_servo[:2].copy()
                remembered_top = max(remembered_top, float(object_after_servo[2]))
                continue
            # Closure displacement and lift verification start from this
            # last, freshly aligned state.
            before = object_after_servo
            grasp[:2] = desired_hand_xy
            last_servo_residual_m = float(residual)
        hand_at_grasp = float(_hand_xyz(model, robot.data)[2])
        if _palm_pressing_top(model, robot.data, body_name):
            print(
                "  ✗ 物体上沿已经顶到掌心。手再抬 2 cm，让摩擦垫去夹侧面"
            )
            raised = _hand_xyz(model, robot.data)
            raised[2] += 0.020
            if not nudge_hand(
                robot, model, raised, yaw_rad=yaw, ignore_bodies={body_name},
            ):
                return False
            hand_at_grasp = float(_hand_xyz(model, robot.data)[2])
        print("→ 阶段一完成：视觉对齐，手指已位于物体两侧")
        grip_result = _close_once(
            robot,
            gap,
            body_name,
            model,
            before,
            object_width_m=float(width),
            yaw_rad=yaw,
            task_log=task_log,
            attempt=attempt,
            servo_residual_m=last_servo_residual_m,
            yaw_policy=str(active_grasp["yaw_policy"]),
            perceived_center_xy_m=detection.base_xyz_m[:2],
            closing_axis_xy=closing,
            handle_direction_rad=handle_direction_for_attempt,
            handle_direction_stabilized=handle_direction_was_stabilized,
        )
        if grip_result is None:
            return False
        if grip_result.yaw_rad is not None:
            yaw = float(grip_result.yaw_rad)
        contacts = grip_result.contacts
        grip_gap = grip_result.gap_m
        contact_forces = grip_result.forces
        contact_state = grip_result.state
        if pending_centering_recovery is not None:
            recovery_effect = {
                **pending_centering_recovery,
                "effect_attempt": attempt,
                "resulting_contact_state": contact_state,
                "resulting_contacts": sorted(contacts),
                "resulting_forces_n": {
                    "left": float(contact_forces.get("left", 0.0)),
                    "right": float(contact_forces.get("right", 0.0)),
                },
            }
            log_checkpoint(
                "grasp_recovery_centering_effect", recovery_effect,
                "balanced" if contact_state == "bilateral_balanced" else "measured",
            )
            pending_centering_recovery = None
        after_close = _body_xyz(model, robot.data, body_name)
        preliminary = _grasp_symptom(before, after_close, None, contacts, grasp[:2], 0.0)
        force_values = list(contact_forces.values())
        force_min = min(force_values)
        force_max = max(force_values)
        if body_name in BOX_BODIES:
            maximum_grip_force = MAX_GRIP_FORCE_N["box"]
        elif body_name == "cube":
            maximum_grip_force = MAX_GRIP_FORCE_N["cube"]
        elif "bottle" in body_name:
            maximum_grip_force = MAX_GRIP_FORCE_N["bottle"]
        else:
            maximum_grip_force = MAX_GRIP_FORCE_N["cup"]
        if body_name in BOX_BODIES:
            target_grip_force = GRIP_FORCE_TARGET_N["box"]
        elif body_name == "cube":
            target_grip_force = GRIP_FORCE_TARGET_N["cube"]
        elif "bottle" in body_name:
            target_grip_force = GRIP_FORCE_TARGET_N["bottle"]
        else:
            target_grip_force = GRIP_FORCE_TARGET_N["cup"]
        force_ratio = (
            force_max / force_min if force_min > 0.05 else float("inf")
        )
        contact_correction = np.zeros(2, dtype=float)
        if (
            not grip_result.ready_for_trial_lift
            and contact_state in {
                "left_only", "right_only", "bilateral_unbalanced",
                "bilateral_loading", "force_limit",
            }
        ):
            geometry = grip_result.contact_geometry or {}
            signed_delta = geometry.get("delta_closing_axis_m")
            delta_direction = str(geometry.get("delta_direction", ""))
            solved: dict[str, object]
            if signed_delta is not None and delta_direction in {
                "toward_left_finger", "toward_right_finger",
            }:
                centering_samples.append((
                    float(signed_centering_applied), float(signed_delta),
                ))
                solved = solve_adaptive_centering_step(centering_samples)
                step = solved["step_signed_m"]
                if step is None:
                    high_side, estimated_correction, correction_source = (
                        recovery_centering_measurement(
                            grip_result.contact_geometry,
                            grip_gap,
                            float(width),
                            contact_forces,
                            target_grip_force,
                        )
                    )
                else:
                    step_m = float(step)
                    high_side = "left" if step_m > 0.0 else "right"
                    estimated_correction = abs(step_m)
                    correction_source = str(solved["method"])
                    signed_centering_applied += step_m
            else:
                high_side, estimated_correction, correction_source = (
                    recovery_centering_measurement(
                        grip_result.contact_geometry,
                        grip_gap,
                        float(width),
                        contact_forces,
                        target_grip_force,
                    )
                )
                solved = {
                    "method": correction_source,
                    "gain": None,
                    "expected_residual_m": None,
                    "step_signed_m": None,
                }
            local_correction, correction = _centering_correction_in_gripper_frame(
                model, robot.data, high_side, estimated_correction
            )
            grasp_xy_bias += correction
            contact_correction = correction.copy()
            pending_centering_recovery = {
                "source_attempt": attempt,
                "source_contact_state": contact_state,
                "source_aperture_m": float(grip_gap),
                "measured_object_width_m": float(width),
                "source_forces_n": {
                    "left": float(contact_forces.get("left", 0.0)),
                    "right": float(contact_forces.get("right", 0.0)),
                },
                "estimated_offset_m": estimated_correction,
                "correction_source": correction_source,
                "contact_delta_closing_axis_m": (
                    None if grip_result.contact_geometry is None
                    else grip_result.contact_geometry.get("delta_closing_axis_m")
                ),
                "contact_delta_lower_bound": (
                    None if grip_result.contact_geometry is None
                    else grip_result.contact_geometry.get("lower_bound")
                ),
                "commanded_local_xy_m": local_correction.tolist(),
                "commanded_world_xy_m": correction.tolist(),
                "accumulated_grasp_bias_xy_m": grasp_xy_bias.tolist(),
                "single_step_cap_m": GRASP_CENTERING_MAX_STEP_M,
                "centering_samples": [
                    {"applied_m": applied, "delta_m": measured}
                    for applied, measured in centering_samples
                ],
                "gain": solved.get("gain"),
                "expected_residual_m": solved.get("expected_residual_m"),
                "centering_method": solved.get("method"),
            }
            log_checkpoint(
                "grasp_recovery_centering", pending_centering_recovery,
                "apply_next_attempt",
            )
            print(
                f"✗ 接触状态={contact_state}，"
                f"试抬门={grip_result.trial_release_decision}，"
                f"受力比={force_ratio:.1f}:1（仅诊断）；"
                f"实测估计 {estimated_correction * 1000:.1f} mm，"
                f"夹爪系修正 {np.round(local_correction * 1000, 1)} mm → "
                f"基座系 {np.round(correction * 1000, 1)} mm"
            )
            if not robot.release(0.4):
                return False
            after_lift = None
            symptom = contact_state
        elif (
            not grip_result.ready_for_trial_lift
            and (
                contact_state == "object_displaced"
                or preliminary in {"no_contact", "tipped"}
            )
        ):
            if preliminary == "no_contact":
                print("✗ 闭合时指尖没有夹住物体，不抬升")
            else:
                print("✗ 闭合时物体被推倒，不抬升")
            if not robot.release(0.4):
                return False
            after_lift = None
            symptom = "object_displaced" if contact_state == "object_displaced" else preliminary
        else:
            print(
                f"→ 夹持进入统一验证：抬 {GRASP_VERIFY_LIFT_M * 1000:.0f} mm，"
                "保持 0.25 s，检查相对滑动/倾斜/大偏移；开口保持不变"
            )
            trial_hand_before = _hand_xyz(model, robot.data)
            trial_object_before = _body_xyz(model, robot.data, body_name)
            relative_before = trial_object_before - trial_hand_before
            object_up_before = _body_up_axis(model, robot.data, body_name)
            commanded_lift = 0.0
            moved = True
            stability = "not_lifted"
            after_lift = trial_object_before.copy()
            hand_now = trial_hand_before.copy()
            contacts = set(grip_result.contacts)
            contact_forces = dict(grip_result.forces)
            rise = relative_slip = lateral_shift = tilt = 0.0

            def measure_micro_lift() -> str:
                nonlocal after_lift, hand_now, contacts, contact_forces
                nonlocal rise, relative_slip, lateral_shift, tilt
                after_lift = _body_xyz(model, robot.data, body_name)
                hand_now = _hand_xyz(model, robot.data)
                contacts, contact_forces = _contact_metrics(
                    model, robot.data, body_name
                )
                rise = float(after_lift[2] - trial_object_before[2])
                relative_slip = float(np.linalg.norm(
                    (after_lift - hand_now) - relative_before
                ))
                lateral_shift = float(np.linalg.norm(
                    after_lift[:2] - trial_object_before[:2]
                ))
                tilt = float(np.rad2deg(np.arccos(np.clip(
                    object_up_before @ _body_up_axis(
                        model, robot.data, body_name
                    ),
                    -1.0,
                    1.0,
                ))))
                return grasp_stability_decision(
                    contacts=contacts,
                    forces=contact_forces,
                    target_force_n=target_grip_force,
                    maximum_force_n=maximum_grip_force,
                    object_rise_m=rise,
                    relative_slip_m=relative_slip,
                    lateral_shift_m=lateral_shift,
                    tilt_deg=tilt,
                )

            # MuJoCo's finger compliance and joint tracking can absorb much of
            # a 5 mm Cartesian request. Use measured payload rise as feedback,
            # with a bounded 30 mm command window. Every sample still applies
            # the motion/contact interlocks, so this never becomes a blind
            # high lift.
            maximum_feedback_rounds = int(
                np.ceil(GRASP_VERIFY_MAX_COMMAND_M / GRASP_VERIFY_LIFT_M)
            )
            for feedback_round in range(1, maximum_feedback_rounds + 1):
                trial_target = _hand_xyz(model, robot.data)
                trial_target[2] = min(
                    0.60, trial_target[2] + GRASP_VERIFY_LIFT_M
                )
                moved = nudge_hand(
                    robot, model, trial_target, yaw_rad=yaw,
                    ignore_bodies={body_name},
                )
                if not moved:
                    stability = "lift_motion_failed"
                    break
                commanded_lift += GRASP_VERIFY_LIFT_M
                if not robot.hold(0.08):
                    return False
                stability = measure_micro_lift()
                print(
                    f"  试抬反馈 {feedback_round}: "
                    f"命令累计={commanded_lift * 1000:.0f} mm，"
                    f"物体上升={rise * 1000:.1f} mm，"
                    f"相对滑动={relative_slip * 1000:.1f} mm，"
                    f"水平偏移={lateral_shift * 1000:.1f} mm，"
                    f"倾斜={tilt:.1f}°，判定={stability}"
                )
                if stability == "stable":
                    # Total observation time for the successful sample is
                    # 0.25 s (0.08 above + 0.17 here).
                    if not robot.hold(0.17):
                        return False
                    stability = measure_micro_lift()
                    break
                if stability != "not_lifted":
                    break
            minimum_force = (
                min(contact_forces.values())
                if contacts == {"left", "right"} else 0.0
            )
            trial_evidence = {
                "local_valid": stability == "stable",
                "decision": stability,
                "commanded_lift_m": commanded_lift,
                "command_step_m": GRASP_VERIFY_LIFT_M,
                "maximum_commanded_lift_m": GRASP_VERIFY_MAX_COMMAND_M,
                "actual_hand_rise_m": float(hand_now[2] - trial_hand_before[2]),
                "object_rise_m": rise,
                "relative_slip_m": relative_slip,
                "lateral_shift_m": lateral_shift,
                "tilt_deg": tilt,
                "bilateral_contacts": contacts == {"left", "right"},
                "minimum_normal_force_n": minimum_force,
                "maximum_normal_force_n": max(contact_forces.values()),
                "force_ratio_diagnostic": (
                    max(contact_forces.values()) / max(minimum_force, 0.05)
                ),
                "object_to_hand_xy_m": float(np.linalg.norm(
                    after_lift[:2] - hand_now[:2]
                )),
                "object_xyz_m": [float(value) for value in after_lift],
                "object_up_axis": [
                    float(value) for value in _body_up_axis(
                        model, robot.data, body_name
                    )
                ],
                "validator": "micro_lift_stability",
            }
            log_checkpoint(
                "grasp_stability", trial_evidence,
                "pass" if stability == "stable" else stability,
            )
            print(
                f"  试抬结果={stability}：物体上升={rise * 1000:.1f} mm，"
                f"相对滑动={relative_slip * 1000:.1f} mm，"
                f"水平偏移={lateral_shift * 1000:.1f} mm，倾斜={tilt:.1f}°，"
                f"力比={trial_evidence['force_ratio_diagnostic']:.2f}:1（仅诊断）"
            )
            if stability == "stable":
                print("✓ 微抬稳定性通过，这一次闭爪即算首次抓取成功")
                if not review_checkpoint("verify_grasp", trial_evidence):
                    return False
                lifted = True
                held_yaw = yaw
                gap = grip_gap
                break
            symptom = (
                "tipped"
                if stability in {"large_displacement", "unstable_tilt"}
                else "not_lifted"
            )
            print("✗ 微抬稳定性未通过，不盲目继续抬高")
            if not robot.release(0.4):
                return False

        failure_type, failure_description = _grasp_failure(
            symptom, contacts, finger_z, top_z
        )
        displacement = float(np.linalg.norm(after_close[:2] - before[:2]))
        diagnosis = {
            "task": f"pick {task.object_color} {task.object_name}",
            "attempt": attempt,
            "object": {
                "type": task.object_name,
                "position": [float(value) for value in before],
                "width": float(width),
                "height": float(top_z - bottom_z),
                "bottom_z": float(bottom_z),
                "top_z": float(top_z),
                "orientation_deg": float(np.rad2deg(yaw)),
            },
            "current_grasp": {
                "strategy": active_grasp["grasp_strategy"],
                "approach_direction": active_grasp["approach_direction"],
                "grasp_region": active_grasp["grasp_region"],
                "palm_height": float(palm_z),
                "finger_target_height": float(finger_z),
                "grasp_height_ratio": float(height_ratio),
                "lateral_offset_m": float(active_grasp["lateral_offset_m"]),
                "pregrasp_distance_m": float(active_grasp["pregrasp_distance_m"]),
            },
            "observation": {
                "cup_rim_height": float(top_z),
                "nearest_obstacle_height": float(neighbor_top),
                "finger_clearance_to_rim": float(finger_z - top_z),
                "contact_force": float(max(contact_forces.values()) if contact_forces else 0.0),
                "left_normal_force_n": float(contact_forces.get("left", 0.0)),
                "right_normal_force_n": float(contact_forces.get("right", 0.0)),
                "contact_state": contact_state,
                "recommended_centering_xy_m": contact_correction.tolist(),
                "object_displacement": displacement,
                "collision": False,
            },
            "failure": {"type": failure_type, "description": failure_description},
        }
        print(f"  失败症状={failure_type}；丢弃旧抓取点和旧 yaw")
        if not adopt_replan(diagnosis):
            look_xy = detection.base_xyz_m[:2].copy()
            remembered_top = top_z
            retreat = _hand_xyz(model, robot.data)
            retreat[2] = min(0.68, retreat[2] + 0.10)
            # The retreat move resets last_abort_reason; the recovery layer
            # needs the grasp failure, not the outcome of the retreat.
            grasp_reason = robot.last_abort_reason
            move_hand_clear(
                robot, model, retreat, yaw_rad=yaw, obstacle_top=obstacle_top,
                ignore_bodies={body_name}, obstacles=obstacles, fast=True,
            )
            robot.last_abort_reason = grasp_reason
            return False
        print("  下一次按新的抓取参数执行，并重新观察")
        look_xy = detection.base_xyz_m[:2].copy()
        remembered_top = top_z
        retreat = _hand_xyz(model, robot.data)
        retreat[2] = min(0.68, retreat[2] + 0.10)
        if not move_hand_clear(
            robot, model, retreat, yaw_rad=yaw, obstacle_top=obstacle_top,
            ignore_bodies={body_name}, obstacles=obstacles, fast=True,
        ):
            print("  ✗ 失败后无法安全撤离")
            return False
        print("  已垂直撤离目标，准备重新观察")

    if not lifted:
        robot.last_abort_reason = robot.last_abort_reason or "grasp_validation_failed"
        print("✗ 多次纠正后仍没有拿起物体")
        return False
    if task.action != "place" or reference is None:
        remember_success_case()
        return True

    if support_xy is None or support_z is None:
        print("✗ 没有放置面坐标")
        return False
    # Color/depth finds the plate in XY, but a centre pixel can land on the
    # cylinder side wall and underestimate its top (0.024 m observed versus
    # 0.036 m geometry).  Fuse perception with the collision scene so place
    # height always uses the highest physical support under that XY.
    geometric_support_z = _surface_under_xy(
        model, robot.data, support_xy, {body_name}
    )
    if geometric_support_z > support_z + 0.002:
        print(
            f"  放置面深度 {support_z:.3f} m 低于几何顶面 "
            f"{geometric_support_z:.3f} m；使用碰撞场景顶面"
        )
        support_z = geometric_support_z
    stable, gap, preload_forces = _stabilize_payload(
        robot, model, body_name, task.object_name, gap
    )
    if task_log is not None:
        task_log.add_checkpoint(
            "transport_preload",
            getattr(robot, "last_preload_record", {}),
            "passed" if stable else "failed",
        )
    if not stable:
        robot.last_abort_reason = "grasp_preload_unbalanced"
        print("✗ 抓取能试抬但不满足运输稳定性门槛，交给上层重新抓取")
        return False
    # Only after the object is securely held do we travel to the support and
    # take a fresh RGB-D measurement there.  This removes the old wasted trip.
    hand = _hand_xyz(model, robot.data)
    held_now = _body_xyz(model, robot.data, body_name)
    held_bottom = _body_bottom_z(model, robot.data, body_name)
    carry_drop = float(hand[2] - held_bottom)
    # The trial lift intentionally raises the object only ~35 mm.  Requiring
    # it to already clear the destination plate by 12 mm can make the current
    # (valid) start state invalid.  Transit still keeps it above the floor;
    # explicit payload geometry handles pedestals and other obstacles.
    transit_minimum_z = max(
        0.012,
        min(float(held_bottom) - 0.003, float(support_z) + 0.012),
    )
    transport_obstacles = obstacles
    if obstacles:
        # Depth markers at the pickup location contain the target's own top
        # surface.  Once that object is in the gripper the marker is stale and
        # must not block every vertical lift.  MuJoCo's exact geometry still
        # checks the pedestal that remains underneath.
        pickup_xy = np.asarray(target.base_xyz_m[:2], dtype=float)
        transport_obstacles = [
            marker for marker in obstacles
            if float(np.linalg.norm(
                np.asarray(marker["xy"], dtype=float) - pickup_xy
            )) > 0.13
        ]
        removed = len(obstacles) - len(transport_obstacles)
        if removed:
            print(f"  已移除 {removed} 个随目标一起离开的抓取区旧深度标记；底台仍由精确几何检查")
    motion_profile, stability_prediction = _payload_motion_profile(
        model,
        robot.data,
        body_name,
        preload_forces,
        task.constraints.transport_mode,
    )
    print(
        "  负载稳定性预测: "
        f"质量={stability_prediction['mass_kg']:.3f} kg，"
        f"双侧最小力={stability_prediction['bilateral_force_n']:.2f} N，"
        f"偏心={stability_prediction['grasp_offset_m'] * 1000:.1f} mm，"
        f"摩擦余量={stability_prediction['margin_ratio']:.2f} → {motion_profile}"
        f"×{stability_prediction['motion_scale']:.2f}"
    )
    transport_monitor = TransportMonitor.capture(
        model,
        robot.data,
        body_name,
        motion_profile=motion_profile,
        motion_scale=float(stability_prediction["motion_scale"]),
    )
    print(
        f"→ 启用负载感知运输（{task.constraints.transport_mode}）："
        "预测摩擦裕量 + 连续 Ruckig 路点速度/加速度继承 + 腕部角运动限幅 + "
        "夹爪坐标系相对位姿/双指接触逐步监测"
    )
    robot.plan_peak_force_hint_n = float(stability_prediction["bilateral_force_n"])
    if renderer is not None and task.reference:
        print("→ 抓取成功，现在前往放置物并复核")
        # Stay low: the wrist camera only needs a local plate check.  Raising
        # a friction-held cup to 40-60 cm wastes time and accumulates slip.
        hover_z = float(np.clip(support_z + 0.22, 0.25, 0.34))
        # A tall carried bottle can hang below the former camera hover and
        # geometrically intersect the plate at the planning goal. Reuse the
        # existing 12 mm transit bottom clearance; this changes the goal pose,
        # not any collision threshold.
        hover_z = max(hover_z, float(support_z) + carry_drop + 0.012)
        found, support_span, obstacle_top = _observe_down(
            robot, model, renderer, task.reference_color or "yellow", task.reference,
            support_xy, hover_z, obstacle_top, held_yaw,
            ignore_bodies={body_name},
            payload_clearance=(carry_drop, transit_minimum_z),
            obstacles=transport_obstacles,
            transport_monitor=transport_monitor,
            route_policy=route_policy,
            plan_review=review_transport_brief,
            candidate_selector=transport_candidate_selector,
            plan_stage="transport",
        )
        transport_plan_record = getattr(robot, "last_transport_plan_record", None)
        if task_log is not None and isinstance(transport_plan_record, dict):
            task_log.add_checkpoint(
                "transport_plan_selection", transport_plan_record,
                str(transport_plan_record.get("planner_used", "unknown")),
            )
        if robot.last_abort_reason:
            if task_log is not None:
                task_log.add_checkpoint(
                    "transport_execution_stability",
                    transport_execution_evidence(
                        transport_plan_record, transport_monitor,
                    ),
                    f"aborted:{robot.last_abort_reason}",
                )
            if robot.last_abort_reason == "motion_plan_failed":
                robot.last_abort_reason = "transport_motion_plan_failed"
            print(
                "✗ 运输控制器已安全停止：",
                robot.last_abort_reason,
                "；反馈=",
                json.dumps(transport_monitor.metrics(), ensure_ascii=False),
            )
            return False
        if task_log is not None:
            task_log.add_checkpoint(
                "transport_execution_stability",
                transport_execution_evidence(
                    transport_plan_record, transport_monitor,
                ),
                "completed",
            )
        # One extra look is allowed when the fresh plate sighting disagrees
        # with the scan; the 6 cm association gate itself never loosens.
        for recheck in range(2):
            refresh_xy = None if found is None else found.base_xyz_m[:2]
            decision = arbitrate_support_refresh(locked_support_xy, refresh_xy)
            if decision["adopted"]:
                support_xy = np.array(decision["xy"], dtype=float, copy=True)
                place_reference_source = "refresh_adopted"
                support_z = (
                    float(support_span["top_z"])
                    if support_span else float(found.base_xyz_m[2])
                )
                support_z = max(
                    support_z,
                    _surface_under_xy(model, robot.data, support_xy, {body_name}),
                )
                print("  放置面新鲜坐标:", np.round(np.append(support_xy, support_z), 3), "m")
                break
            if decision["reason"] == "deviation":
                place_reference_source = "refresh_arbitrated"
                print(
                    f"  放置复核偏离锁定参考 {float(decision['deviation_m']) * 1000:.0f} mm，"
                    "超过 20 mm，不采用"
                )
                if task_log is not None:
                    task_log.add_checkpoint("support_refresh_deviation", {
                        "locked_xy_m": locked_support_xy.tolist(),
                        "refresh_xy_m": np.asarray(refresh_xy, dtype=float).tolist(),
                        "deviation_m": float(decision["deviation_m"]),
                        "adopted": False,
                    }, "refresh_arbitrated")
            elif decision["reason"] == "not_found":
                print("  放置物复核失败：support_refresh_rejected:not_found")
            else:
                print(
                    "  放置物复核失败："
                    f"support_refresh_rejected:{float(decision['deviation_m']):.3f}"
                )
            if recheck == 1:
                print("  两次复核都不一致，保留初次扫描坐标")
                break
            print("  → 再做一次俯视复核")
            found, support_span, obstacle_top = _observe_down(
                robot, model, renderer, task.reference_color or "yellow", task.reference,
                support_xy, hover_z, obstacle_top, held_yaw,
                ignore_bodies={body_name},
                payload_clearance=(carry_drop, transit_minimum_z),
                obstacles=transport_obstacles,
                transport_monitor=transport_monitor,
                route_policy=route_policy,
                candidate_selector=transport_candidate_selector,
            )
            if robot.last_abort_reason:
                if robot.last_abort_reason == "motion_plan_failed":
                    robot.last_abort_reason = "transport_motion_plan_failed"
                print("✗ 复核移动被本地控制器停止：", robot.last_abort_reason)
                return False
        hand_after_transit = _hand_xyz(model, robot.data)
        held_after_transit = _body_xyz(model, robot.data, body_name)
        if (
            _body_bottom_z(model, robot.data, body_name) < 0.012
            or np.linalg.norm(held_after_transit[:2] - hand_after_transit[:2]) > 0.07
        ):
            print("✗ 转运中物体已滑落，停止放置，不假报任务成功")
            return False
        if not review_checkpoint("verify_transport", {
            "local_valid": True,
            "object_bottom_z_m": _body_bottom_z(model, robot.data, body_name),
            "object_to_hand_xy_m": float(np.linalg.norm(
                held_after_transit[:2] - hand_after_transit[:2]
            )),
            "transport_metrics": transport_monitor.metrics(),
            "validator": "relative_pose_contact_and_collision_monitor",
        }):
            return False
        # P-3 applies to execute_transport. Placement alignment keeps all
        # original slip/tilt/force abort gates but does not begin a second
        # autonomous recovery sequence after transport was already verified.
        transport_monitor.recovery_stage = "transport_completed"
        # If force feedback applied bounded preload during transit, use the
        # real actuator aperture as the starting point for gradual release.
        gap = float(robot.data.ctrl[-1]) / 255.0 * 0.08
    if task.relation == "next_to":
        support_xy = support_xy.copy()
        support_xy[0] += 0.16
    # Travel above the plate first, then step down while watching the real
    # object bottom.  A single joint move lets the block slide in the fingers
    # and get pushed into the solid plate.
    hand = _hand_xyz(model, robot.data)
    held_now = _body_xyz(model, robot.data, body_name)
    held_bottom = _body_bottom_z(model, robot.data, body_name)
    carry_drop = float(hand[2] - held_bottom)
    # Align the carried object's center, not the hand origin, with the plate.
    carried_xy_offset = held_now[:2] - hand[:2]
    place_hand_xy = support_xy - carried_xy_offset
    minimum_place_hand_z = support_z + carry_drop + 0.030
    hover = np.array([
        place_hand_xy[0], place_hand_xy[1],
        max(float(hand[2]), minimum_place_hand_z),
    ])
    print(
        f"→ 放到支撑面上方 {RELEASE_CLEARANCE * 1000:.0f} mm"
        f"（支撑面 {support_z:.3f} m，不压入盘子）"
    )
    log_checkpoint("transport_phase", {
        "phase": "PRE_PLACE",
        "payload_body": body_name,
        "support_body": task.reference or "plate",
        "payload_support_contact_allowed": False,
        "robot_support_contact_allowed": False,
    }, "active")
    place_yaw = _approach_pre_place(
        robot, model, place_hand_xy,
        base_z=float(hover[2]),
        yaw_rad=held_yaw,
        ignore_bodies={body_name},
        payload_body=body_name,
        payload_clearance=(carry_drop, support_z + 0.012),
        transport_monitor=transport_monitor,
        obstacles=transport_obstacles,
        route_policy=route_policy,
        support_name=task.reference or "plate",
    )
    if place_yaw is None:
        return False
    held_yaw = place_yaw
    print("→ 放置前校正负载姿态：以物体竖直为目标求解手腕，而不是只校正手掌")
    # This is an intentional, bounded pose correction.  The ordinary monitor
    # compares against the pre-correction hand frame and would interpret the
    # corrective rotation itself as slip.  Permit a narrow transient window,
    # then require an absolute upright check and rebase the relative monitor.
    upright_error = float("inf")
    upright_rotation = _upright_hand_rotation(model, robot.data, body_name)
    previous_error = float(np.rad2deg(np.arccos(np.clip(
        _body_up_axis(model, robot.data, body_name)[2], -1.0, 1.0
    ))))
    for correction_round in range(1, 4):
        upright_rotation = _upright_hand_rotation(model, robot.data, body_name)
        upright_hand = _hand_xyz(model, robot.data)
        old_tilt_limit = transport_monitor.maximum_tilt_deg
        transport_monitor.maximum_tilt_deg = max(old_tilt_limit, 22.0)
        upright_moved = nudge_hand(
            robot,
            model,
            upright_hand,
            yaw_rad=held_yaw,
            ignore_bodies={body_name},
            payload_clearance=(carry_drop, support_z + 0.012),
            transport_monitor=transport_monitor,
            target_rotation=upright_rotation,
        )
        transport_monitor.maximum_tilt_deg = old_tilt_limit
        if not upright_moved:
            print(
                "✗ 放置前负载姿态校正被本地控制器拒绝："
                f"{robot.last_abort_reason or 'ik_or_collision_validation_failed'}"
            )
            return False
        upright_error = float(np.rad2deg(np.arccos(np.clip(
            _body_up_axis(model, robot.data, body_name)[2], -1.0, 1.0
        ))))
        print(f"  姿态校正 {correction_round}: 负载竖直误差={upright_error:.2f}°")
        if upright_error > previous_error + 1.0:
            print("✗ 校正后姿态反而变差，拒绝继续旋转")
            return False
        transport_monitor.rebase_relative_pose()
        if not robot.hold(0.06, monitor=transport_monitor.check):
            return False
        if upright_error <= 4.0:
            break
        previous_error = upright_error
    if upright_error > 4.0:
        print("✗ 负载姿态无法校正到安全放置范围，拒绝继续下降")
        return False
    def descend_to_clearance() -> bool:
        """Guarded 3 mm steps down to RELEASE_CLEARANCE above the support."""
        for _ in range(40):
            if not robot._can_continue():
                return False
            bottom = _body_bottom_z(model, robot.data, body_name)
            if bottom <= support_z + RELEASE_CLEARANCE:
                break
            hand = _hand_xyz(model, robot.data)
            held_now = _body_xyz(model, robot.data, body_name)
            xy_error = support_xy - held_now[:2]
            correction = np.clip(xy_error, -0.006, 0.006)
            drop = min(0.003, bottom - (support_z + RELEASE_CLEARANCE))
            nxt = np.array([
                float(hand[0] + correction[0]),
                float(hand[1] + correction[1]),
                float(hand[2]) - drop,
            ])
            if not nudge_hand(
                robot, model, nxt, yaw_rad=held_yaw, ignore_bodies={body_name},
                payload_clearance=(float(hand[2]) - bottom, support_z + 0.002),
                transport_monitor=transport_monitor,
                target_rotation=upright_rotation,
                payload_approach_body=task.reference or "plate",
            ):
                break
        return True

    def probe_support() -> tuple[bool, int, float]:
        """1 mm guarded descents until MuJoCo reports a real support contact."""
        # Height equality is not proof that the plate carries the payload.  Make
        # a few 1 mm guarded descents until MuJoCo reports a real normal contact;
        # never open the fingers while the object is merely hovering inside the
        # solver's contact tolerance.
        support_contacts, support_force = _support_contact_force(
            model, robot.data, body_name, support_body
        )
        if not robot.hold(0.10, monitor=transport_monitor.check):
            return False, support_contacts, support_force
        print(
            f"  承重探测起点：物体底部="
            f"{_body_bottom_z(model, robot.data, body_name):.3f} m，"
            f"支撑面={support_z:.3f} m"
        )
        for seat_step in range(12):
            support_contacts, support_force = _support_contact_force(
                model, robot.data, body_name, support_body
            )
            # Contact existence is the guarded-stop signal.  Requiring a large
            # force would make the still-closed gripper press the object several
            # millimetres into the support before release.
            if support_contacts > 0:
                break
            hand = _hand_xyz(model, robot.data)
            seated = hand.copy()
            seated[2] -= 0.001
            if not nudge_hand(
                robot,
                model,
                seated,
                yaw_rad=held_yaw,
                ignore_bodies={body_name, support_body},
                target_rotation=upright_rotation,
                payload_clearance=(
                    float(hand[2]) - _body_bottom_z(model, robot.data, body_name),
                    support_z - 0.004,
                ),
                transport_monitor=transport_monitor,
                payload_approach_body=support_body,
            ):
                break
            if not robot.hold(0.05, monitor=transport_monitor.check):
                return False, support_contacts, support_force
            seated_bottom = _body_bottom_z(model, robot.data, body_name)
            print(
                f"  承重探测 {seat_step + 1}: 底部={seated_bottom:.3f} m，"
                f"盘面接触={support_contacts}"
            )
            if seated_bottom < support_z - 0.004:
                print("  ✗ 已超过支撑面几何容差，立即停止下降")
                break
        if support_contacts == 0:
            print("✗ 未检测到物体由盘面承重，拒绝松爪")
            return False, support_contacts, support_force
        if support_force > 3.0:
            print(
                f"  盘面瞬时约束冲量较高（等效 {support_force:.2f} N）；"
                "按几何不穿透和渐卸后的稳定状态复核，不把 no-slip 冲量当作夹持力"
            )
        print(
            f"  ✓ 盘面承重接触={support_contacts}，法向力={support_force:.2f} N，"
            "允许开始卸载夹爪"
        )
        return True, support_contacts, support_force

    log_checkpoint("transport_phase", {
        "phase": "DESCENT",
        "payload_body": body_name,
        "support_body": task.reference or "plate",
        "payload_support_contact_allowed": False,
        "robot_support_contact_allowed": False,
    }, "active")
    if not descend_to_clearance():
        return False
    bottom = _body_bottom_z(model, robot.data, body_name)
    held_now = _body_xyz(model, robot.data, body_name)
    hand = _hand_xyz(model, robot.data)
    support_body = task.reference or "plate"
    early_support_contacts, early_support_force = _support_contact_force(
        model, robot.data, body_name, support_body
    )
    if bottom < support_z + RELEASE_CLEARANCE - 0.0005:
        if (
            bottom < 0.012
            or float(np.linalg.norm(held_now[:2] - hand[:2])) > 0.06
        ):
            print(
                f"✗ 物体已经不在夹爪里（底部 {bottom:.3f} m，"
                f"距手心 {np.linalg.norm(held_now[:2] - hand[:2]):.3f} m），"
                "不再抬高重规划，避免越抬越掉"
            )
            return False
        if early_support_contacts > 0:
            print(
                f"  物体底部 {bottom:.3f} m 已接触支撑面；"
                "不再抬手，避免夹爪与已承重物体相互顶住"
            )
        else:
            deficit = support_z + RELEASE_CLEARANCE - bottom
            print(
                f"  物体还夹着，但底部 {bottom:.3f} m 低于支撑面。"
                f"只把手抬高 {deficit * 1000:.0f} mm，不重新拉高整条轨迹"
            )
            raised = hand.copy()
            raised[2] += deficit
            if not nudge_hand(
                robot, model, raised, yaw_rad=held_yaw, ignore_bodies={body_name},
                target_rotation=upright_rotation,
            ):
                return False
            bottom = _body_bottom_z(model, robot.data, body_name)
    log_checkpoint("transport_phase", {
        "phase": "CONTACT",
        "payload_body": body_name,
        "support_body": support_body,
        "payload_support_contact_allowed": True,
        "robot_support_contact_allowed": False,
    }, "active")
    seated_ok, support_contacts, support_force = probe_support()
    if not seated_ok:
        return False
    # One bounded correction before opening: if the measured pre-release pose
    # would already fail verify_place, lift 1 cm, shift at most 2 cm toward
    # the support centre, re-seat with the same guarded descent and re-probe.
    pre_xy = _body_xyz(model, robot.data, body_name)[:2]
    pre_bottom = _body_bottom_z(model, robot.data, body_name)
    adjust = _place_micro_adjust(pre_xy, support_xy, pre_bottom, support_z)
    if adjust is not None:
        print(
            f"→ 松爪前微调一次：预判水平偏差 {np.linalg.norm(support_xy - pre_xy) * 1000:.0f} mm，"
            f"底部 {pre_bottom:.3f} m，平移 {np.round(adjust * 1000, 1).tolist()} mm"
        )
        hand = _hand_xyz(model, robot.data)
        lifted_hand = hand.copy()
        lifted_hand[2] += 0.010
        shifted_hand = lifted_hand.copy()
        shifted_hand[:2] += adjust
        for waypoint in (lifted_hand, shifted_hand):
            if not nudge_hand(
                robot, model, waypoint, yaw_rad=held_yaw, ignore_bodies={body_name},
                payload_clearance=(float(hand[2]) - pre_bottom, support_z + 0.002),
                transport_monitor=transport_monitor,
                target_rotation=upright_rotation,
                payload_approach_body=support_body,
            ):
                print("✗ 微调移动被本地控制器拒绝，停止放置")
                robot.last_abort_reason = robot.last_abort_reason or "place_geometry_failed"
                return False
        if not descend_to_clearance():
            return False
        seated_ok, support_contacts, support_force = probe_support()
        if not seated_ok:
            return False
    # Let residual motion die out before opening; otherwise the box can have
    # enough lateral velocity to skate off the plate.
    if not robot.hold(0.25):
        return False
    before_release = _body_xyz(model, robot.data, body_name)
    bottom = _body_bottom_z(model, robot.data, body_name)
    absolute_tilt = float(np.rad2deg(np.arccos(np.clip(
        _body_up_axis(model, robot.data, body_name)[2], -1.0, 1.0
    ))))
    print(
        f"松爪前物体 xyz={np.round(before_release, 3)} m，"
        f"底部 {bottom:.3f} m（支撑面 {support_z:.3f} m），"
        f"绝对倾斜={absolute_tilt:.1f}°"
    )
    print("→ 盘面已承重；每次张开 2 mm 渐卸预紧，再完全松爪")
    # ``gap`` is the actuator's nominal aperture; the physical inner pad gap
    # in this model is about 6 mm smaller.  Open to 78 mm in small increments
    # so contact is genuinely gone before the final 2 mm actuator travel.
    unload_gap = 0.078
    release_gaps = np.arange(float(gap) + 0.002, unload_gap + 0.001, 0.002)
    release_up = _body_up_axis(model, robot.data, body_name)
    unload_samples: list[dict[str, object]] = []
    for release_gap in release_gaps:
        if not robot.grasp_to(_ctrl_for_gap(float(release_gap)), 0.18):
            return False
        if not robot.hold(0.08):
            return False
        current = _body_xyz(model, robot.data, body_name)
        current_bottom = _body_bottom_z(model, robot.data, body_name)
        seated_contacts, seated_force = _support_contact_force(
            model, robot.data, body_name, support_body
        )
        cosine = float(np.clip(
            release_up @ _body_up_axis(model, robot.data, body_name), -1.0, 1.0
        ))
        release_tilt = float(np.rad2deg(np.arccos(cosine)))
        release_displacement = float(np.linalg.norm(
            current[:2] - before_release[:2]
        ))
        release_decision = placement_release_stability_decision(
            displacement_m=release_displacement,
            tilt_deg=release_tilt,
            support_contacts=seated_contacts,
            object_bottom_z_m=current_bottom,
            support_z_m=support_z,
        )
        unload_samples.append({
            "gripper_gap_m": float(release_gap),
            "object_xy_m": [float(value) for value in current[:2]],
            "displacement_from_supported_pose_m": release_displacement,
            "tilt_delta_deg": release_tilt,
            "support_contacts": int(seated_contacts),
            "support_force_n": float(seated_force),
            "object_bottom_z_m": float(current_bottom),
            "decision": release_decision,
        })
        print(
            f"  卸载到 {float(release_gap):.3f} m：底部={current_bottom:.3f} m，"
            f"偏移={release_displacement * 1000:.1f} mm，"
            f"倾斜增量={release_tilt:.1f}°，盘面接触={seated_contacts}/"
            f"{seated_force:.2f} N，稳定性={release_decision}"
        )
        if release_decision != "stable":
            print(
                f"✗ 渐卸预紧时稳定性失败（{release_decision}），"
                "立即停止继续张开，保持当前支撑"
            )
            robot.last_abort_reason = f"place_unload_unstable:{release_decision}"
            log_checkpoint("place_support_stability", {
                "local_valid": False,
                "decision": release_decision,
                "reference_supported_xyz_m": [
                    float(value) for value in before_release
                ],
                "maximum_displacement_m": max(
                    float(sample["displacement_from_supported_pose_m"])
                    for sample in unload_samples
                ),
                "maximum_tilt_deg": max(
                    float(sample["tilt_delta_deg"]) for sample in unload_samples
                ),
                "samples": unload_samples,
                "validator": "support_contact_and_gradual_unload",
            }, release_decision)
            return False
    if not robot.release(0.25):
        return False
    if not robot.hold(0.45):
        return False
    rested = _body_xyz(model, robot.data, body_name)
    bottom = _body_bottom_z(model, robot.data, body_name)
    final_support_contacts, final_support_force = _support_contact_force(
        model, robot.data, body_name, support_body
    )
    final_tilt = float(np.rad2deg(np.arccos(np.clip(
        release_up @ _body_up_axis(model, robot.data, body_name), -1.0, 1.0
    ))))
    final_displacement = float(np.linalg.norm(
        rested[:2] - before_release[:2]
    ))
    final_release_decision = placement_release_stability_decision(
        displacement_m=final_displacement,
        tilt_deg=final_tilt,
        support_contacts=final_support_contacts,
        object_bottom_z_m=bottom,
        support_z_m=support_z,
    )
    unload_samples.append({
        "gripper_gap_m": 0.080,
        "object_xy_m": [float(value) for value in rested[:2]],
        "displacement_from_supported_pose_m": final_displacement,
        "tilt_delta_deg": final_tilt,
        "support_contacts": int(final_support_contacts),
        "support_force_n": float(final_support_force),
        "object_bottom_z_m": float(bottom),
        "decision": final_release_decision,
        "fully_released": True,
    })
    release_evidence = {
        "local_valid": final_release_decision == "stable",
        "decision": final_release_decision,
        "reference_supported_xyz_m": [float(value) for value in before_release],
        "maximum_displacement_m": max(
            float(sample["displacement_from_supported_pose_m"])
            for sample in unload_samples
        ),
        "maximum_tilt_deg": max(
            float(sample["tilt_delta_deg"]) for sample in unload_samples
        ),
        "final_support_contacts": int(final_support_contacts),
        "final_support_force_n": float(final_support_force),
        "samples": unload_samples,
        "validator": "support_contact_and_gradual_unload",
    }
    log_checkpoint(
        "place_support_stability",
        release_evidence,
        "pass" if final_release_decision == "stable" else final_release_decision,
    )
    if final_release_decision != "stable":
        print(
            f"✗ 完全松爪后稳定性失败（{final_release_decision}），"
            f"相对支撑起点偏移 {final_displacement * 1000:.1f} mm"
        )
        robot.last_abort_reason = f"place_release_unstable:{final_release_decision}"
        return False
    used_support_xy = np.asarray(support_xy, dtype=float)
    locked_xy = locked_support_xy if locked_support_xy is not None else used_support_xy
    errors = place_error_pair(rested[:2], used_support_xy, locked_xy)
    xy_error = float(errors["error_vs_used_support_m"])
    on_height = support_z - 0.008 <= bottom <= support_z + 0.025
    on_target = xy_error <= 0.09
    disclosure = {
        "locked_xy_m": [float(v) for v in locked_xy],
        "used_xy_m": [float(v) for v in used_support_xy],
        "source": place_reference_source,
        "error_vs_used_support_m": float(errors["error_vs_used_support_m"]),
        "error_vs_locked_reference_m": float(errors["error_vs_locked_reference_m"]),
    }
    robot.last_place_disclosure = disclosure
    print(
        f"放置后物体 xyz={np.round(rested, 3)} m，底部 {bottom:.3f} m，"
        f"支撑面 {support_z:.3f} m，"
        f"相对所用参考 {xy_error * 1000:.0f} mm，"
        f"相对锁定参考 {disclosure['error_vs_locked_reference_m'] * 1000:.0f} mm"
        f"（{place_reference_source}）"
    )
    place_evidence = {
        "object_xyz_m": rested, "object_bottom_z_m": bottom,
        "support_xy_m": used_support_xy, "support_top_z_m": float(support_z),
        "locked_support_xy_m": np.asarray(locked_xy, dtype=float),
        "horizontal_error_m": xy_error,
        "error_vs_used_support_m": disclosure["error_vs_used_support_m"],
        "error_vs_locked_reference_m": disclosure["error_vs_locked_reference_m"],
        "place_reference_source": place_reference_source,
        "on_height": on_height, "on_target": on_target,
    }
    if task_log is not None:
        task_log.add_checkpoint(
            "place_geometry", place_evidence,
            "pass" if (on_height and on_target) else "place_geometry_failed",
        )
    if not on_height or not on_target:
        print(
            "✗ 没有放到终点：需要底部贴在支撑面上，并且水平偏差不超过 9 cm。"
            "这次不算完成"
        )
        robot.last_abort_reason = "place_geometry_failed"
        if task_log is not None:
            task_log.add_failure("place_geometry_failed", stage="place",
                                 horizontal_error_m=xy_error, object_bottom_z_m=bottom)
        return False
    print("✓ 放置已核对：物体在支撑面上，且落在目标范围内。夹爪保持张开")
    if not review_checkpoint("verify_place", {
        "local_valid": True,
        "object_bottom_z_m": bottom,
        "support_top_z_m": float(support_z),
        "horizontal_error_m": xy_error,
        "error_vs_used_support_m": disclosure["error_vs_used_support_m"],
        "error_vs_locked_reference_m": disclosure["error_vs_locked_reference_m"],
        "locked_support_xy_m": disclosure["locked_xy_m"],
        "used_support_xy_m": disclosure["used_xy_m"],
        "place_reference_source": place_reference_source,
        "on_height": on_height,
        "on_target": on_target,
        "validator": "support_contact_height_and_region",
    }):
        return False
    print("  运输监控指标:", json.dumps(transport_monitor.metrics(), ensure_ascii=False))
    remember_success_case()
    return True


def main() -> None:
    from embodied_agent import LLMRobotLeader
    from task_runner import run_task

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default="拿起橙色方块")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "detection.png")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--final-output",
        type=Path,
        help="Optional extra copy of the delivery image (outputs/delivery_*.png).",
    )
    args = parser.parse_args()

    leader = LLMRobotLeader()
    print("用户指令:", args.instruction)
    report = {
        "instruction": args.instruction,
        "planner_backend": "",
        "llm_input": {"instruction": args.instruction},
        "llm_output": None,
        "detector_input": {
            "camera": CAMERA_NAME,
            "rgb_shape": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
            "depth_shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
            "scan_angles_deg": list(SCAN_ANGLES_DEG),
            "target_query": "",
        },
        "target_detection": None,
        "reference_detection": None,
        "execution": {"attempted": False, "success": False},
    }

    model = create_model()
    data = mujoco.MjData(model)
    reset_scene(model, data)
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
    viewer_context = nullcontext(None)
    if not args.no_viewer:
        from mujoco import viewer as mj_viewer

        viewer_context = mj_viewer.launch_passive(model, data)

    try:
        with viewer_context as viewer:
            if viewer is not None:
                viewer.cam.lookat[:] = (0.0, 0.0, 0.25)
                viewer.cam.distance = 1.8
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -28

            robot = PandaController(model, data, viewer=viewer, speed=args.speed)
            inventory = None
            if leader.online:
                from scene_perception import collect_open_inventory

                print("→ 开放清点：每条指令重新扫描，不沿用上一条的物体清单")
                inventory = collect_open_inventory(
                    robot, model, data, renderer, leader,
                )
                print(f"  清点 {len(inventory)} 条")
                for item in inventory:
                    if item.confidence == "low":
                        print(
                            f"  临时条目 {item.color} {item.description} "
                            f"body={item.body_name} size={item.dimensions_m}"
                        )
            try:
                task, planner_backend = leader.interpret_instruction(
                    args.instruction, inventory=inventory,
                )
            except NotFoundReport as report:
                from task_log import TaskLog

                log = TaskLog.start(args.instruction, None, "not_found")
                log.add_checkpoint("not_found", {
                    "message": report.message, **report.details,
                }, "not_found")
                print(report.message)
                print("任务日志:", log.write())
                return
            print(f"任务规划 ({planner_backend}):")
            print(json.dumps(task.to_dict(), ensure_ascii=False, indent=2))
            report["planner_backend"] = planner_backend
            report["llm_output"] = task.to_dict()
            report["detector_input"]["target_query"] = (
                f"{task.object_color} {task.object_name}"
            )
            saved_first_frame = {"done": False}

            def scan(
                current_task: TaskSpec,
                ceiling: list[float],
                obstacles: list[dict],
                captures: dict[str, np.ndarray],
            ) -> tuple[Detection | None, Detection | None, float | None]:
                rgb, depth, found, angle = search_for_target(
                    current_task, model, data, renderer, robot,
                    ceiling=ceiling, obstacles=obstacles,
                )
                report["target_detection"] = detection_to_dict(found)
                report["target_observation_angle_deg"] = angle
                captures["target"] = rgb
                if not saved_first_frame["done"]:
                    saved_first_frame["done"] = True
                    print("RGB:", rgb.shape, rgb.dtype)
                    print("Depth:", depth.shape, depth.dtype)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    result = annotate(rgb, found) if found else rgb
                    Image.fromarray(result, mode="RGB").save(args.output)
                    np.save(args.output.with_suffix(".depth.npy"), depth)
                    print("Saved:", args.output.resolve())
                    if not args.no_show:
                        open_image(args.output)
                reference_found: Detection | None = None
                if found is not None and current_task.reference:
                    reference_task = replace(
                        current_task,
                        object_name=current_task.reference,
                        object_color=current_task.reference_color or "yellow",
                        reference=None,
                        relation=None,
                    )
                    print(
                        f"\n→ 继续搜索参照物: {reference_task.object_color} "
                        f"{reference_task.object_name}"
                    )
                    reference_rgb, _, reference_found, _ = search_for_target(
                        reference_task, model, data, renderer, robot,
                        ceiling=ceiling, obstacles=obstacles,
                    )
                    captures["destination"] = reference_rgb
                    report["reference_detection"] = detection_to_dict(reference_found)
                return found, reference_found, angle

            print("\n坐标将由扫描记录，随后进入 IK 抓放执行。")
            report["execution"]["attempted"] = True
            outcome = run_task(
                args.instruction, leader, model, data, renderer, robot,
                scan=scan, task=task, planner_backend=planner_backend,
            )
            detection = outcome.target
            angle = outcome.observation_angle_deg or 0.0
            task = outcome.task or task
            if outcome.review is not None:
                review = outcome.review
                report["llm_scene_review"] = {
                    "approved": review.approved,
                    "needs_reobserve": review.needs_reobserve,
                    "transport_mode": review.transport_mode,
                    "route_policy": review.route_policy,
                    "reason": review.reason,
                    "backend": review.backend,
                    "skill_calls": review.skill_plan.to_list(),
                }
            report["execution"]["success"] = bool(outcome.success)
            report["execution"]["status"] = "succeeded" if outcome.success else "failed"
            report["execution"]["stage"] = "verify_place" if outcome.success else "manipulation"
            report["execution"]["reason"] = outcome.reason
            report["execution"]["recovery_rounds"] = outcome.recovery_rounds
            report["execution"]["recovery_log"] = outcome.recovery_log
            report["execution"]["final_object_world_xyz_m"] = outcome.final_object_xyz_m
            report["execution"]["final_object_bottom_z_m"] = outcome.final_object_bottom_m
            report["execution"]["xy_error_m"] = outcome.xy_error_m
            body_name = BODY_OF_COLOR.get(task.object_color)
            if body_name is not None:
                report["execution"]["final_object_tilt_deg"] = float(np.rad2deg(np.arccos(np.clip(
                    _body_up_axis(model, data, body_name)[2], -1.0, 1.0
                ))))
            report["execution"]["delivery_image"] = outcome.delivery_path
            report["execution"]["task_log"] = outcome.task_log_path
            if outcome.delivery_path and args.final_output:
                from delivery import copy_delivery

                copied = copy_delivery(Path(outcome.delivery_path), args.final_output)
                report["execution"]["final_screenshot"] = str(copied.resolve())
                print("交差图副本:", copied.resolve())
            print("任务结果:", json.dumps(outcome.summary(), ensure_ascii=False))

            if args.report:
                report["llm_usage"] = leader.usage_summary()
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print("Report:", args.report.resolve())

            print("LLM token 用量:", json.dumps(leader.usage_summary()))

            if viewer is not None and viewer.is_running():
                if detection:
                    print(
                        f"目标观察角度 {angle:+.0f}°；计划抓取方式: "
                        f"{task.grasp_strategy} / {task.approach_direction}"
                    )
                print("关闭 Viewer 结束。")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
    finally:
        renderer.close()
    if not bool(report["execution"].get("success")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
