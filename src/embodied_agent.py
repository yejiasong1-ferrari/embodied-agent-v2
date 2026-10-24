"""LLM-led, safety-gated orchestration for the Panda manipulation stack.

The task planner turns language into an object and a destination.  The grasp
reasoner runs after a failed attempt: it reads the measured failure and
returns a new high-level grasp.  Joint angles, torque, speed, and collision
limits stay in the local controller, which clips any proposal before execution.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from robot_skills import DEFAULT_SKILL_LIBRARY, SkillLibrary, SkillPlan, SkillPlanError
from task_planner import (
    CategoryMismatch, NotFoundReport, TaskSpec, apply_manager_rereview, plan_task,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _json_default(value: object) -> object:
    """Convert sensor-library scalars to ordinary JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_local_env() -> None:
    """Load the project-local, git-ignored .env without overriding the shell."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value)


class LeaderUnavailableError(RuntimeError):
    """Raised when LLM-required mode cannot reach a configured model."""


@dataclass(frozen=True)
class SceneObject:
    role: str
    label: str
    color: str
    name: str
    base_xyz_m: tuple[float, float, float]
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    coordinate_source: str = "rgbd_geometry"
    registry_confidence: str = "high"


@dataclass(frozen=True)
class SceneSnapshot:
    target: SceneObject | None
    destination: SceneObject | None
    obstacles: tuple[dict[str, object], ...]
    maximum_obstacle_top_m: float
    unknown_space_policy: str = "unobserved_is_unknown"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SceneReview:
    approved: bool
    needs_reobserve: bool
    transport_mode: str
    route_policy: str
    reason: str
    skill_plan: SkillPlan
    backend: str


CHECKPOINT_DECISIONS = frozenset({"continue", "adjust", "stop"})
# The only adjustments an LLM checkpoint may request.  Each one maps to a
# conservative local action in task_runner; nothing here is a pose or a limit.
CHECKPOINT_ADJUSTMENTS = frozenset({
    "switch_grasp_side", "raise_transport_height", "slow_down", "reobserve",
})
_CHECKPOINT_ALLOWED_KEYS = frozenset({"decision", "adjustment", "reason"})


@dataclass(frozen=True)
class CheckpointReview:
    decision: str
    reason: str
    backend: str
    adjustment: str | None = None

    @property
    def may_continue(self) -> bool:
        return self.decision == "continue"

    @property
    def wants_adjust(self) -> bool:
        return self.decision == "adjust" and self.adjustment in CHECKPOINT_ADJUSTMENTS


def parse_checkpoint_review(
    raw: object, local_valid: bool, backend: str
) -> CheckpointReview:
    """Three-state checkpoint verdict.  Anything outside the whitelist stops."""
    if not isinstance(raw, dict):
        return CheckpointReview("stop", "checkpoint 回复不是 JSON 对象", backend)
    reason = str(raw.get("reason", "LLM checkpoint audit"))[:500]
    extra = sorted(set(raw.keys()) - _CHECKPOINT_ALLOWED_KEYS)
    if extra:
        return CheckpointReview(
            "stop", f"checkpoint 回复带有非白名单字段 {extra}；{reason}", backend
        )
    decision = str(raw.get("decision", "stop"))
    adjustment_raw = raw.get("adjustment")
    adjustment = None if adjustment_raw is None else str(adjustment_raw)
    # Older prompts answered reobserve/replan; both are now an adjust request.
    if decision == "reobserve":
        decision, adjustment = "adjust", adjustment or "reobserve"
    elif decision == "replan":
        decision, adjustment = "adjust", adjustment or "reobserve"
    if decision not in CHECKPOINT_DECISIONS:
        return CheckpointReview("stop", f"未知 checkpoint 决定 {decision!r}；{reason}", backend)
    if decision == "adjust":
        if adjustment not in CHECKPOINT_ADJUSTMENTS:
            return CheckpointReview(
                "stop", f"adjustment {adjustment!r} 不在白名单内；{reason}", backend
            )
    else:
        adjustment = None
    if not local_valid:
        return CheckpointReview("stop", f"本地验证未通过，LLM 不能放行；{reason}", backend)
    return CheckpointReview(decision, reason, backend, adjustment)


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    reason: str
    backend: str
    route_preference: str | None = None


@dataclass(frozen=True)
class GraspStrategyUpdate:
    """High-level grasp intent.  Metric poses stay with the local planner."""

    action: str
    grasp_strategy: str
    approach_direction: str
    grasp_height_ratio: float
    lateral_offset_m: float
    yaw_delta_rad: float
    pregrasp_distance_m: float
    need_reobserve: bool
    reason: str
    backend: str
    grasp_region: str = "middle_side"
    yaw_policy: str = "geometry"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_GRASP_STRATEGIES = {"side_grasp", "top_grasp", "edge_grasp", "handle_grasp"}
_APPROACHES = {"vertical", "horizontal"}
_REPLAN_ACTIONS = {"retry_grasp", "reobserve_scene", "safe_stop"}
_SIDE_OBJECTS = {"cup", "mug", "bottle"}
_BOX_OBJECTS = {"box", "cube", "block"}
_GRASP_REGIONS = {"auto", "lower_side", "middle_side", "upper_side"}
_YAW_POLICIES = {"geometry", "preserve_wrist", "avoid_handle"}

# --- Pre-execution pick strategy (manager decides words, local maps them) ---
PICK_STRATEGY_ENUMS: dict[str, frozenset[str]] = {
    "approach": frozenset({"vertical", "horizontal"}),
    "grip_width": frozenset({"narrow", "medium", "wide"}),
    "lift_height": frozenset({"low", "medium", "high"}),
    "transport": frozenset({"direct", "via_safe_point"}),
    "speed_profile": frozenset({"slow", "normal", "fast"}),
}
_PICK_STRATEGY_ALLOWED_KEYS = frozenset(PICK_STRATEGY_ENUMS) | {"reason"}
_FORBIDDEN_KEY_FRAGMENTS = ("joint", "qpos", "torque", "ctrl", "trajectory", "waypoint")


@dataclass(frozen=True)
class PickStrategy:
    """Five enumerated choices made before the first grasp attempt."""

    approach: str
    grip_width: str
    lift_height: str
    transport: str
    speed_profile: str
    reason: str
    backend: str
    clipped_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_pick_strategy(task: Any, backend: str = "fallback-default", reason: str = "") -> PickStrategy:
    approach = getattr(task, "approach_direction", "vertical")
    if approach not in PICK_STRATEGY_ENUMS["approach"]:
        approach = "vertical"
    fragile = getattr(getattr(task, "constraints", None), "transport_mode", "normal") == "fragile"
    return PickStrategy(
        approach=approach,
        grip_width="wide",
        lift_height="medium",
        transport="direct",
        speed_profile="slow" if fragile else "normal",
        reason=reason or "默认保守策略（与改动前执行层行为一致）",
        backend=backend,
    )


def parse_pick_strategy(raw: object) -> PickStrategy:
    """Strict enum parse.  Unknown fields or low-level keys are rejected."""
    if not isinstance(raw, dict):
        raise ValueError("抓取策略不是 JSON 对象")
    for key in raw:
        lowered = str(key).lower()
        if any(fragment in lowered for fragment in _FORBIDDEN_KEY_FRAGMENTS):
            raise ValueError(f"抓取策略含底层字段 {key!r}")
    extra = sorted(set(raw) - _PICK_STRATEGY_ALLOWED_KEYS)
    if extra:
        raise ValueError(f"抓取策略含非白名单字段 {extra}")
    values: dict[str, str] = {}
    for name, allowed in PICK_STRATEGY_ENUMS.items():
        value = str(raw.get(name, "")).strip().lower()
        if value not in allowed:
            raise ValueError(f"{name}={value!r} 不在枚举 {sorted(allowed)} 内")
        values[name] = value
    return PickStrategy(
        reason=str(raw.get("reason", ""))[:500], backend="llm", **values
    )


def clip_pick_strategy(strategy: PickStrategy, task: Any) -> PickStrategy:
    """Resolve conflicts the same way clip_grasp_strategy does; record them."""
    clipped: list[str] = []
    approach = strategy.approach
    speed = strategy.speed_profile
    object_name = str(getattr(task, "object_name", "")).lower()
    fragile = getattr(getattr(task, "constraints", None), "transport_mode", "normal") == "fragile"
    planned_approach = str(getattr(task, "approach_direction", "vertical"))
    if object_name in _BOX_OBJECTS and approach == "horizontal":
        approach = "vertical"
        clipped.append("approach")
    elif approach == "horizontal" and planned_approach != "horizontal":
        # A horizontal slide-in can push a free-standing object before the
        # pads straddle it (measured: 48-53 mm on the red cup in the online
        # run).  The first attempt keeps the planner's approach; horizontal
        # entry stays available to replan_grasp after a measured failure.
        approach = planned_approach if planned_approach in PICK_STRATEGY_ENUMS["approach"] else "vertical"
        clipped.append("approach")
    if speed == "fast" and (fragile or object_name in _SIDE_OBJECTS):
        speed = "normal"
        clipped.append("speed_profile")
    if fragile and speed != "slow":
        # A fragile task may only get slower, never faster.
        speed = "slow"
        if "speed_profile" not in clipped:
            clipped.append("speed_profile")
    reason = strategy.reason
    if clipped:
        reason = f"{reason}（本地安全层裁剪: {', '.join(clipped)}）"[:500]
    return replace(
        strategy, approach=approach, speed_profile=speed, reason=reason,
        clipped_fields=tuple(clipped),
    )


def _region_ratio(region: str, fallback: float) -> float:
    """Compatibility value; the local candidate generator owns final height."""
    return {
        "lower_side": 0.36,
        "middle_side": 0.50,
        "upper_side": 0.68,
    }.get(region, fallback)


def _diagnosis_ratio(diagnosis: dict[str, object]) -> float:
    current = diagnosis.get("current_grasp")
    if isinstance(current, dict) and current.get("grasp_height_ratio") is not None:
        return float(current["grasp_height_ratio"])
    return 0.45


def parse_grasp_replan(raw: dict[str, Any], diagnosis: dict[str, object]) -> GraspStrategyUpdate:
    """Read a parameterized grasp update.  Joint and torque fields are rejected."""
    if not isinstance(raw, dict):
        raise ValueError("抓取重规划不是 JSON 对象")
    forbidden = {"joint_angles", "qpos", "torque", "ctrl", "velocity", "actuator"}
    present = forbidden.intersection(raw)
    update = raw.get("strategy_update") if isinstance(raw.get("strategy_update"), dict) else {}
    present |= forbidden.intersection(update)
    if present:
        raise ValueError("抓取重规划不能包含关节、力矩或速度")
    action = str(raw.get("action") or raw.get("decision") or "retry_grasp")
    if action == "replan_grasp":
        action = "retry_grasp"
    if action not in _REPLAN_ACTIONS:
        raise ValueError(f"未授权的恢复动作 {action}")
    current = diagnosis.get("current_grasp") if isinstance(diagnosis.get("current_grasp"), dict) else {}
    strategy = str(update.get("grasp_strategy") or current.get("strategy") or "side_grasp")
    approach = str(update.get("approach_direction") or current.get("approach_direction") or "vertical")
    if strategy not in _GRASP_STRATEGIES or approach not in _APPROACHES:
        raise ValueError("抓取策略或接近方向不在允许范围内")
    ratio = update.get("grasp_height_ratio")
    if ratio is None:
        ratio = _ratio_from_absolute_height(update, diagnosis)
    if ratio is None:
        ratio = _diagnosis_ratio(diagnosis)
    region = str(update.get("grasp_region") or "auto")
    if region == "auto":
        numeric = float(ratio)
        region = (
            "upper_side" if numeric >= 0.60
            else "lower_side" if numeric <= 0.40
            else "middle_side"
        )
    yaw_policy = str(update.get("yaw_policy") or "geometry")
    if region not in _GRASP_REGIONS or yaw_policy not in _YAW_POLICIES:
        raise ValueError("抓取区域或偏航策略不在允许范围内")
    return GraspStrategyUpdate(
        action=action,
        grasp_strategy=strategy,
        approach_direction=approach,
        grasp_height_ratio=float(ratio),
        lateral_offset_m=float(update.get("lateral_offset", update.get("lateral_offset_m", 0.0))),
        yaw_delta_rad=float(update.get("yaw_delta_rad", np.deg2rad(float(update.get("yaw_delta_deg", 0.0))))),
        pregrasp_distance_m=float(update.get("pregrasp_distance", update.get("pregrasp_distance_m", 0.08))),
        need_reobserve=bool(raw.get("need_reobserve", True)),
        reason=str(raw.get("reason") or "根据失败诊断调整抓取参数")[:500],
        backend="parsed",
        grasp_region=region,
        yaw_policy=yaw_policy,
    )


def _ratio_from_absolute_height(update: dict[str, Any], diagnosis: dict[str, object]) -> float | None:
    """Accept a finger or palm height and convert it to a fraction of the object."""
    obj = diagnosis.get("object") if isinstance(diagnosis.get("object"), dict) else {}
    bottom = obj.get("bottom_z")
    top = obj.get("top_z")
    if bottom is None or top is None:
        return None
    height = float(top) - float(bottom)
    if height < 1e-3:
        return None
    if update.get("finger_target_height") is not None:
        finger = float(update["finger_target_height"])
    elif update.get("target_grasp_height") is not None:
        absolute = float(update["target_grasp_height"])
        finger = absolute - 0.147 if absolute > float(top) else absolute
    else:
        return None
    return (finger - float(bottom)) / height


def clip_grasp_strategy(
    update: GraspStrategyUpdate,
    diagnosis: dict[str, object],
) -> GraspStrategyUpdate:
    """Keep the proposal inside the grasp band.  A plan above the rim is lowered."""
    notes: list[str] = []
    strategy = update.grasp_strategy
    obj = diagnosis.get("object") if isinstance(diagnosis.get("object"), dict) else {}
    current = diagnosis.get("current_grasp") if isinstance(diagnosis.get("current_grasp"), dict) else {}
    observation = diagnosis.get("observation") if isinstance(diagnosis.get("observation"), dict) else {}
    failure = diagnosis.get("failure") if isinstance(diagnosis.get("failure"), dict) else {}
    object_type = str(obj.get("type", ""))
    failure_type = str(failure.get("type", ""))
    region = update.grasp_region
    if region not in _GRASP_REGIONS:
        region = "middle_side"
        notes.append("未知抓取区域改为杯身中部")
    yaw_policy = update.yaw_policy
    if yaw_policy not in _YAW_POLICIES:
        yaw_policy = "geometry"
    approach = update.approach_direction
    if object_type in _SIDE_OBJECTS and strategy != "side_grasp":
        strategy = "side_grasp"
        notes.append("杯子和瓶子保持侧握")
    if object_type in {"box", "cube", "block"}:
        # A horizontal entry can push a freestanding block several centimetres
        # before the pads straddle it.  Blocks stay on the proven vertical
        # approach; the LLM may choose a height band or yaw, not a shove.
        strategy = "top_grasp"
        approach = "vertical"
        if update.approach_direction != "vertical":
            notes.append("方块保持垂直接近，禁止水平侧推")
    ratio = float(np.clip(_region_ratio(region, update.grasp_height_ratio), 0.28, 0.72))
    clearance = float(observation.get("finger_clearance_to_rim", 0.0))
    if failure_type == "no_contact" and clearance > 0.015:
        lowered = min(ratio, 0.50)
        if lowered < ratio - 1e-6:
            notes.append("指尖不得再高于物体上沿")
        ratio = lowered
        region = "middle_side"
    # An image is not metric calibration.  Fresh RGB-D and signed left/right
    # force correction own all metric alignment; semantic model output has no
    # direct lateral/yaw/pregrasp authority.
    lateral = 0.0
    if abs(update.lateral_offset_m) > 1e-9:
        notes.append("忽略模型给出的米制侧移，由本地视觉/触觉闭环计算")
    if failure_type in {
        "partial_contact", "left_only", "right_only",
        "bilateral_unbalanced", "object_moved",
    }:
        lateral = 0.0
        notes.append("横向位置由新视觉和左右接触力重新校准")
    if failure_type in {"partial_contact", "left_only", "right_only", "bilateral_unbalanced"}:
        ratio = _diagnosis_ratio(diagnosis)
        region = str(current.get("grasp_region") or region)
        approach = str(current.get("approach_direction") or approach)
        notes.append("偏心接触只修正中心，不无依据改变抓取高度")
    yaw_delta = 0.0
    if abs(update.yaw_delta_rad) > 1e-9:
        notes.append("忽略模型给出的数值偏航，由局部候选规划器计算")
    pregrasp = 0.10 if approach == "horizontal" else 0.08
    reason = update.reason
    if notes:
        reason = f"{reason}（本地安全层：{'，'.join(notes)}）"
    return replace(
        update,
        grasp_strategy=strategy,
        approach_direction=approach,
        grasp_height_ratio=ratio,
        lateral_offset_m=lateral,
        yaw_delta_rad=yaw_delta,
        pregrasp_distance_m=pregrasp,
        reason=reason[:500],
        grasp_region=region,
        yaw_policy=yaw_policy,
        # Every retry is grounded in a new RGB-D frame.  An LLM cannot opt
        # into reusing the pose of an object that may have been pushed.
        need_reobserve=(
            True if update.action in {"retry_grasp", "reobserve_scene"}
            else update.need_reobserve
        ),
    )


def offline_grasp_replan(diagnosis: dict[str, object]) -> GraspStrategyUpdate:
    """Deterministic grasp update used when the vision-language model is offline."""
    failure = diagnosis.get("failure") if isinstance(diagnosis.get("failure"), dict) else {}
    failure_type = str(failure.get("type") or "unknown")
    observation = diagnosis.get("observation") if isinstance(diagnosis.get("observation"), dict) else {}
    current = diagnosis.get("current_grasp") if isinstance(diagnosis.get("current_grasp"), dict) else {}
    obj = diagnosis.get("object") if isinstance(diagnosis.get("object"), dict) else {}
    clearance = float(observation.get("finger_clearance_to_rim", 0.0))
    ratio = _diagnosis_ratio(diagnosis)
    approach = str(current.get("approach_direction") or "vertical")
    lateral = 0.0
    yaw_delta = 0.0
    pregrasp = 0.08
    action = "retry_grasp"
    need_reobserve = True
    region = str(current.get("grasp_region") or "middle_side")
    yaw_policy = (
        "avoid_handle" if str(obj.get("type", "")) in {"cup", "mug"}
        else "preserve_wrist" if str(obj.get("type", "")) == "bottle"
        else "geometry"
    )
    if failure_type == "no_contact" and clearance > 0.015:
        ratio = 0.45
        region = "middle_side"
        approach = "horizontal"
        pregrasp = 0.10
        reason = "没有接触，而且指尖在物体上沿之上，改为杯身中部并水平接近"
    elif failure_type == "no_contact":
        ratio = max(0.32, ratio - 0.10)
        region = "middle_side"
        approach = "horizontal"
        pregrasp = 0.10
        reason = "闭合时没有接触，降低夹持高度并从侧面进入"
    elif failure_type in {"partial_contact", "left_only", "right_only", "bilateral_unbalanced"}:
        reason = "接触不对称；保持语义抓取区域，由夹爪坐标系局部闭环重新居中"
    elif failure_type == "object_moved":
        ratio = min(0.68, ratio + 0.08)
        region = "upper_side"
        pregrasp = 0.05
        reason = "接近时物体被推动，提高夹持点并缩短水平进入距离"
    elif failure_type in {"lift_failed", "slipped"}:
        ratio = 0.50
        region = "middle_side"
        reason = "物体没有稳定跟随；回到质心附近并重新生成抓取候选"
    elif failure_type in {"collision", "approach_blocked"}:
        action = "safe_stop" if failure_type == "collision" else "reobserve_scene"
        yaw_delta = np.pi / 2.0
        reason = "接近被挡住，停止或换一个夹持方向后再观察"
    else:
        action = "reobserve_scene"
        reason = "失败原因不足以安全改写抓取高度，先重新观察"
    proposal = GraspStrategyUpdate(
        action=action,
        grasp_strategy=str(current.get("strategy") or "side_grasp"),
        approach_direction=approach if approach in _APPROACHES else "vertical",
        grasp_height_ratio=ratio,
        lateral_offset_m=lateral,
        yaw_delta_rad=yaw_delta,
        pregrasp_distance_m=pregrasp,
        need_reobserve=need_reobserve,
        reason=reason,
        backend="offline-grasp-reasoner",
        grasp_region=region,
        yaw_policy=yaw_policy,
    )
    return clip_grasp_strategy(proposal, diagnosis)


def _scene_object(
    role: str,
    color: str,
    name: str,
    detection: Any,
    registry_confidence: str = "high",
) -> SceneObject | None:
    if detection is None:
        return None
    xyz = tuple(float(value) for value in np.asarray(detection.base_xyz_m, dtype=float))
    return SceneObject(
        role=role,
        label=str(detection.label),
        color=color,
        name=name,
        base_xyz_m=xyz,
        confidence=float(detection.score),
        bbox_xyxy=tuple(int(value) for value in detection.bbox_xyxy),
        coordinate_source=str(getattr(detection, "source", "rgbd_geometry")),
        registry_confidence=registry_confidence,
    )


def build_scene_snapshot(
    task: TaskSpec,
    target: Any,
    destination: Any,
    obstacles: list[dict] | None,
    maximum_obstacle_top_m: float,
) -> SceneSnapshot:
    compact_obstacles: list[dict[str, object]] = []
    for marker in obstacles or []:
        compact_obstacles.append({
            "xy_m": [float(value) for value in np.asarray(marker["xy"], dtype=float)],
            "top_z_m": float(marker["top_z"]),
            "source": "rgbd_depth_cluster",
        })
    return SceneSnapshot(
        target=_scene_object(
            "target", task.object_color, task.object_name, target,
            task.object_confidence,
        ),
        destination=_scene_object(
            "destination",
            task.reference_color or "yellow",
            task.reference or "",
            destination,
        ),
        obstacles=tuple(compact_obstacles),
        maximum_obstacle_top_m=float(maximum_obstacle_top_m),
    )


def _compact_contact_sheet(frames: dict[str, np.ndarray] | None) -> str | None:
    """Encode at most two 320px views as one low-token JPEG data URL."""
    if not frames:
        return None
    selected: list[tuple[str, np.ndarray]] = []
    for key in ("target", "destination"):
        frame = frames.get(key)
        if frame is not None:
            selected.append((key, frame))
    if not selected and frames.get("latest") is not None:
        selected.append(("latest", frames["latest"]))
    selected = selected[:2]
    tiles: list[Image.Image] = []
    for label, frame in selected:
        tile = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
        tile.thumbnail((320, 240), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (320, 260), (24, 24, 26))
        canvas.paste(tile, ((320 - tile.width) // 2, 20))
        ImageDraw.Draw(canvas).text((8, 3), label, fill=(240, 240, 240))
        tiles.append(canvas)
    sheet = Image.new("RGB", (320 * len(tiles), 260), (24, 24, 26))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * 320, 0))
    buffer = BytesIO()
    sheet.save(buffer, format="JPEG", quality=68, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@dataclass
class ImagePointing:
    """Normalized image point of the object to pick, and of the destination."""

    target_view: int
    target_u: float
    target_v: float
    destination_view: int | None
    destination_u: float | None
    destination_v: float | None
    reason: str


def _unit_fraction(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 不是画面坐标") from error
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} 必须在 0 到 1 之间")
    return number


def _view_index(value: object, view_count: int) -> int:
    try:
        index = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("画面编号无效") from error
    if not 0 <= index < view_count:
        raise ValueError("画面编号超出本次拍摄")
    return index


def parse_image_pointing(
    raw: dict[str, Any],
    view_count: int,
    need_destination: bool,
) -> ImagePointing:
    """Accept only an in-frame point.  A missing object is not a coordinate."""
    if view_count < 1:
        raise ValueError("没有广角画面")
    if not isinstance(raw, dict):
        raise ValueError("定位结果不是 JSON 对象")
    if not bool(raw.get("target_seen")):
        raise ValueError(str(raw.get("reason") or "广角画面里没有确认要拿的物体"))
    destination_view = destination_u = destination_v = None
    if need_destination:
        if not bool(raw.get("destination_seen")):
            raise ValueError(str(raw.get("reason") or "广角画面里没有确认放置位置"))
        destination_view = _view_index(raw.get("destination_view", 0), view_count)
        destination_u = _unit_fraction(raw.get("destination_u"), "destination_u")
        destination_v = _unit_fraction(raw.get("destination_v"), "destination_v")
    return ImagePointing(
        target_view=_view_index(raw.get("target_view", 0), view_count),
        target_u=_unit_fraction(raw.get("target_u"), "target_u"),
        target_v=_unit_fraction(raw.get("target_v"), "target_v"),
        destination_view=destination_view,
        destination_u=destination_u,
        destination_v=destination_v,
        reason=str(raw.get("reason") or "已在广角画面中确认物体")[:500],
    )


def _wide_jpeg(frame: np.ndarray) -> str:
    """Keep the wrist frame at its captured size so shape stays readable."""
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class LLMRobotLeader:
    """High-level leader with explicit online, auto, and offline modes."""

    VALID_MODES = {"required", "auto", "offline"}

    def __init__(
        self,
        *,
        mode: str | None = None,
        skill_library: SkillLibrary = DEFAULT_SKILL_LIBRARY,
    ) -> None:
        _load_local_env()
        self.mode = (mode or os.getenv("EMBODIED_AGENT_MODE", "auto")).strip().lower()
        if self.mode not in self.VALID_MODES:
            raise ValueError("EMBODIED_AGENT_MODE 必须是 required、auto 或 offline")
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = os.getenv(
            "VLM_BASE_URL",
            "https://llm-us3whhqdyc0mhmso.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )
        self.model = os.getenv("VLM_MODEL", "qwen3-vl-plus")
        self.skill_library = skill_library
        self._client = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        if self.mode == "required" and not self.api_key:
            raise LeaderUnavailableError(
                "已启用 LLM required 模式，但没有 DASHSCOPE_API_KEY；拒绝伪装成 LLM 运行。"
            )

    @property
    def online(self) -> bool:
        return self.mode != "offline" and bool(self.api_key)

    @property
    def backend_label(self) -> str:
        return f"{self.model} online" if self.online else "deterministic offline leader"

    def _get_client(self):
        if not self.online:
            raise LeaderUnavailableError("LLM 未配置")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _chat_json(
        self,
        *,
        system: str,
        payload: dict[str, object],
        image_data_url: str | None = None,
        image_data_urls: list[str] | None = None,
        max_pixels: int = 640 * 260,
        max_tokens: int = 1400,
    ) -> dict[str, Any]:
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": json.dumps(
                    payload, ensure_ascii=False, default=_json_default
                ),
            }
        ]
        urls = list(image_data_urls or [])
        if image_data_url is not None:
            urls.append(image_data_url)
        for url in urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": url},
                "min_pixels": 256 * 256,
                "max_pixels": max_pixels,
            })
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.total_prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.total_completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        text = response.choices[0].message.content or ""
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"LLM 没有返回 JSON：{text[:200]}")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("LLM 返回值不是 JSON 对象")
        return parsed

    def _fragile_ratchet(
        self, instruction: str, task: TaskSpec, backend: str
    ) -> tuple[TaskSpec, str]:
        # A language model may occasionally miss an explicit caution word.
        # The deterministic parser acts as a one-way safety ratchet: it
        # can escalate normal -> fragile, but can never relax the LLM or
        # user's requested constraints. A not-found offline parse must not
        # erase a plan the manager already accepted.
        try:
            fallback, _ = plan_task(instruction, use_llm=False)
        except NotFoundReport:
            return task, backend
        if (
            fallback.constraints.transport_mode == "fragile"
            and task.constraints.transport_mode != "fragile"
        ):
            task = replace(
                task,
                constraints=replace(task.constraints, transport_mode="fragile"),
            )
            backend += " + local-fragile-safety-ratchet"
        return task, backend

    def _rereview_mismatch(
        self, instruction: str, mismatch: CategoryMismatch
    ) -> tuple[TaskSpec, str]:
        """One manager pass. A second inconsistency becomes a human not-found."""
        system = (
            "You reconcile one object request with the real scene catalog. "
            "Reply with one JSON object: decision (select or not_found) and "
            "object_id (a catalog id, or null). Select only when the user's "
            "words agree with that object's shape. A cylinder may be a mug or "
            "bottle. A cube, box, or block must not be selected as a mug. "
            "A different color must not be selected. Do not output coordinates, "
            "joint angles, or forces."
        )
        payload = {
            "instruction": instruction,
            "mismatch": mismatch.receipt,
            "candidates": [mismatch.receipt],
        }
        try:
            reply = self._chat_json(system=system, payload=payload)
        except Exception as error:
            if self.mode == "required":
                raise LeaderUnavailableError(f"LLM 任务理解失败：{error}") from error
            raise
        checked = apply_manager_rereview(mismatch, reply)
        return checked, f"{self.model}-manager-rereview"

    def ask_inventory(self, views: list[dict[str, object]]) -> dict:
        """Open inventory of one fresh set of wide frames. Not a coordinate plan."""
        if not self.online:
            raise LeaderUnavailableError("LLM 未配置，不能开放清点场景")
        if not views:
            raise ValueError("没有可清点的广角画面")
        system = (
            "List every distinct object on the table in these wide-angle photos. "
            "Do not limit the list to a known catalog. A bottle is a tall narrow "
            "cylinder that may have a neck. A mug may have a handle. A plate is a "
            "low wide disk. Reply with one JSON object: "
            "{\"objects\": [{\"description\": short noun, \"color\": english color, "
            "\"shape\": \"cylinder\" or \"box\" or \"plate\", \"view_index\": int, "
            "\"u\": 0-1 from the left, \"v\": 0-1 from the top}]}. "
            "Point at the center of each object body. A pink or magenta object is "
            "pink, not purple and not red. Do not output joint angles, "
            "metre coordinates, or forces."
        )
        payload = {
            "views": [
                {"view_index": index, "base_angle_deg": view.get("angle_deg")}
                for index, view in enumerate(views)
            ],
        }
        return self._chat_json(
            system=system,
            payload=payload,
            image_data_urls=[_wide_jpeg(view["rgb"]) for view in views],  # type: ignore[arg-type]
            max_pixels=480 * 360,
            max_tokens=2200,
        )

    def interpret_instruction(
        self,
        instruction: str,
        inventory: list | None = None,
    ) -> tuple[TaskSpec, str]:
        if not self.online:
            return plan_task(instruction, use_llm=False)
        try:
            task, backend = plan_task(instruction, use_llm=True, inventory=inventory)
        except CategoryMismatch as mismatch:
            try:
                task, backend = self._rereview_mismatch(instruction, mismatch)
            except (NotFoundReport, LeaderUnavailableError):
                raise
            except Exception as error:
                if self.mode == "required":
                    raise LeaderUnavailableError(f"LLM 任务理解失败：{error}") from error
                task, backend = plan_task(instruction, use_llm=False)
                return task, f"{backend}; online planner failed: {type(error).__name__}"
        except NotFoundReport:
            raise
        except Exception as error:
            if self.mode == "required":
                raise LeaderUnavailableError(f"LLM 任务理解失败：{error}") from error
            task, backend = plan_task(instruction, use_llm=False)
            return task, f"{backend}; online planner failed: {type(error).__name__}"
        return self._fragile_ratchet(instruction, task, backend)

    def locate_in_views(
        self,
        instruction: str,
        task: TaskSpec,
        views: list[dict[str, object]],
    ) -> ImagePointing:
        """Point at the instructed object in the wide-angle photos.

        The reply is an image location only.  Depth at that pixel becomes the
        robot coordinate afterwards.
        """
        if not self.online:
            raise LeaderUnavailableError("LLM 未配置，不能从画面确认物体")
        if not views:
            raise ValueError("没有可分析的广角画面")
        destination = None
        if task.reference is not None:
            destination = f"{task.reference_color or ''} {task.reference}".strip()
        system = (
            "You identify objects in wide-angle wrist-camera photos of a cluttered table. "
            "Images are in view_index order, origin at the top-left. u is the fraction of "
            "image width from the left, v is the fraction of image height from the top. "
            "Decide what to pick from shape and appearance. Do not choose the largest "
            "region of the requested color. A cup or mug is an upright cylinder and may "
            "have a handle. A flat stack of books, a cereal box, a tray, a bowl, or a "
            "can is not that cup even when its color is similar. A bottle is a tall "
            "narrow cylinder with a neck. A plate is a low wide disk, not a box or mug "
            "standing nearby. Point at the center of the object body. Reply with one "
            "JSON object: target_seen (bool), target_view (int), target_u (0-1), "
            "target_v (0-1), destination_seen (bool), destination_view (int), "
            "destination_u (0-1), destination_v (0-1), reason (string). If the "
            "destination is null, set destination_seen false."
        )
        payload = {
            "instruction": instruction,
            "object_to_pick": f"{task.object_color} {task.object_name}",
            "destination": destination,
            "views": [
                {
                    "view_index": index,
                    "base_angle_deg": view.get("angle_deg"),
                }
                for index, view in enumerate(views)
            ],
        }
        try:
            raw = self._chat_json(
                system=system,
                payload=payload,
                image_data_urls=[_wide_jpeg(view["rgb"]) for view in views],  # type: ignore[arg-type]
                max_pixels=640 * 480,
            )
            pointing = parse_image_pointing(
                raw, len(views), need_destination=task.reference is not None
            )
        except Exception as error:
            if self.mode == "required":
                raise LeaderUnavailableError(f"广角画面定位失败：{error}") from error
            raise
        return pointing

    def review_scene(
        self,
        instruction: str,
        task: TaskSpec,
        scene: SceneSnapshot,
        frames: dict[str, np.ndarray] | None = None,
    ) -> SceneReview:
        canonical = self.skill_library.canonical_plan(task)

        def locally_consistent() -> bool:
            if scene.target is None:
                return False
            if task.reference is not None and scene.destination is None:
                return False
            if (
                scene.destination is not None
                and task.reference == "plate"
                and not (0.015 <= scene.destination.base_xyz_m[2] <= 0.10)
            ):
                return False
            if scene.destination is not None:
                target_xy = np.asarray(scene.target.base_xyz_m[:2])
                destination_xy = np.asarray(scene.destination.base_xyz_m[:2])
                if float(np.linalg.norm(target_xy - destination_xy)) < 0.12:
                    return False
            return True

        if not self.online:
            approved = locally_consistent()
            return SceneReview(
                approved=approved,
                needs_reobserve=not approved,
                transport_mode=task.constraints.transport_mode,
                route_policy="auto",
                reason=(
                    "本地 RGB-D 已确认目标、放置物与障碍物；使用安全技能标准计划。"
                    if approved else "缺少目标或放置物，需要重新观察。"
                ),
                skill_plan=canonical,
                backend="offline-scene-agent",
            )

        system = (
            "You are the high-level leader of a Franka Panda robot. The structured "
            "coordinates were back-projected from an image point already chosen on the "
            "wide-angle photo. Do not invent a different object or new coordinates. "
            "Choose only skills from the supplied manifest. Never output Python, joint "
            "angles, actuator values, force commands, or trajectories. Collision checking, "
            "IK, OMPL validation and local grasp/place validators are mandatory and cannot "
            "be disabled. Reply with one JSON object containing: approved (bool), "
            "needs_reobserve (bool), transport_mode (normal|fragile), route_policy "
            "(auto|clockwise|counterclockwise|high_clearance), reason (string), and "
            "skill_calls (array of {name, arguments})."
        )
        payload = {
            "instruction": instruction,
            "task": task.to_dict(),
            "scene": scene.to_dict(),
            "available_skills": self.skill_library.manifest(),
            "canonical_safe_plan": canonical.to_list(),
        }
        try:
            raw = self._chat_json(
                system=system,
                payload=payload,
                image_data_url=_compact_contact_sheet(frames),
            )
            skill_plan = self.skill_library.validate_plan(raw.get("skill_calls"), task)
            mode = str(raw.get("transport_mode", task.constraints.transport_mode))
            if mode not in {"normal", "fragile"}:
                raise ValueError("transport_mode 非法")
            route = str(raw.get("route_policy", "auto"))
            if route not in {"auto", "clockwise", "counterclockwise", "high_clearance"}:
                raise ValueError("route_policy 非法")
            local_complete = locally_consistent()
            approved = bool(raw.get("approved", False)) and local_complete
            return SceneReview(
                approved=approved,
                needs_reobserve=bool(raw.get("needs_reobserve", False)) or not local_complete,
                transport_mode=mode,
                route_policy=route,
                reason=str(raw.get("reason", "LLM 已审核场景"))[:500],
                skill_plan=skill_plan,
                backend=f"{self.model}-scene-agent",
            )
        except Exception as error:
            if self.mode == "required":
                raise LeaderUnavailableError(f"LLM 场景审核失败：{error}") from error
            approved = locally_consistent()
            return SceneReview(
                approved=approved,
                needs_reobserve=not approved,
                transport_mode=task.constraints.transport_mode,
                route_policy="auto",
                reason=f"在线场景审核失败，采用本地安全计划：{type(error).__name__}",
                skill_plan=canonical,
                backend="offline-scene-agent-fallback",
            )

    def apply_review(self, task: TaskSpec, review: SceneReview) -> TaskSpec:
        # The visual leader may make transport more conservative, never relax
        # a user-requested fragile constraint.
        transport_mode = (
            "fragile"
            if "fragile" in {task.constraints.transport_mode, review.transport_mode}
            else "normal"
        )
        constraints = replace(task.constraints, transport_mode=transport_mode)
        return replace(task, constraints=constraints)

    def review_checkpoint(
        self,
        stage: str,
        evidence: dict[str, object],
        frame: np.ndarray | None = None,
    ) -> CheckpointReview:
        local_valid = bool(evidence.get("local_valid", False))
        if not self.online:
            return CheckpointReview(
                decision="continue" if local_valid else "stop",
                reason="本地传感器验证通过。" if local_valid else "本地验证未通过。",
                backend="offline-checkpoint-validator",
            )
        system = (
            "You audit a robot manipulation checkpoint. Sensor evidence is authoritative. "
            "You may be more conservative but may never override local_valid=false. The "
            "image is a compact semantic cross-check, not a source of precise coordinates. "
            "Reply with one JSON object with exactly these keys: decision "
            "(continue|adjust|stop), adjustment (only when decision is adjust; one of "
            "switch_grasp_side|raise_transport_height|slow_down|reobserve), reason. "
            "Never output coordinates, waypoints, joint angles or forces."
        )
        frames = None if frame is None else {stage: frame, "latest": frame}
        try:
            raw = self._chat_json(
                system=system,
                payload={"stage": stage, "evidence": evidence},
                image_data_url=_compact_contact_sheet(frames),
            )
            return parse_checkpoint_review(
                raw, local_valid, f"{self.model}-checkpoint-agent"
            )
        except Exception as error:
            if self.mode == "required":
                return CheckpointReview(
                    decision="stop",
                    reason=f"required 模式下 LLM checkpoint 失败：{error}",
                    backend="required-llm-error",
                )
            return CheckpointReview(
                decision="continue" if local_valid else "stop",
                reason=f"LLM checkpoint 不可用；本地验证结果生效：{type(error).__name__}",
                backend="offline-checkpoint-fallback",
            )

    def review_plan_brief(self, brief: dict[str, object]) -> CheckpointReview:
        """Three-state verdict on a validated transport plan summary.

        Same vocabulary as review_checkpoint.  The brief is descriptive; the
        LLM can only say continue / adjust(whitelist) / stop about it.
        """
        if not self.online:
            return CheckpointReview(
                "continue", "离线：路径已由本地几何验证。", "offline-plan-brief"
            )
        system = (
            "You audit a validated robot transport plan summary before it executes. "
            "The plan was already checked for collisions by the local planner; your "
            "role is a conservative sanity check on clearance, duration, force and "
            "risk flags. start_bottom_clearance_m is where the carried object already "
            "is (right after the trial lift) and cannot be changed by this plan; judge "
            "min_table_clearance_m and min_obstacle_clearance_m, which describe the "
            "travel. min_obstacle_clearance_m is explicitly an approximate depth-marker "
            "clearance, not an exact geometric distance. If route_policy is already "
            "high_clearance do not ask to raise it "
            "again; if transport_mode is already fragile do not ask to slow down again. "
            "Reply with one JSON object with exactly these keys: decision "
            "(continue|adjust|stop), adjustment (only when decision is adjust; one of "
            "switch_grasp_side|raise_transport_height|slow_down|reobserve), reason. "
            "Never output waypoints, coordinates, joint angles or forces."
        )
        try:
            raw = self._chat_json(system=system, payload={"plan_brief": brief})
            return parse_checkpoint_review(raw, True, f"{self.model}-plan-brief-agent")
        except Exception as error:
            if self.mode == "required":
                return CheckpointReview(
                    "stop", f"required 模式下 LLM 路径述职失败：{error}", "required-llm-error"
                )
            return CheckpointReview(
                "continue",
                f"LLM 路径述职不可用；按本地验证继续：{type(error).__name__}",
                "offline-plan-brief-fallback",
            )

    def select_transport_candidate(
        self,
        candidates: list[dict[str, object]],
        preference: str,
    ) -> tuple[str | None, str]:
        """Select only an ID from locally validated, measured candidates."""
        validated = [item for item in candidates if bool(item.get("validated", False))]
        if not validated:
            return None, "local_no_validated_candidate"

        def score(item: dict[str, object]) -> tuple[float, ...]:
            duration = float(item["estimated_duration_s"])
            pose = float(item["maximum_pose_deviation_deg"])
            acceleration = float(item["maximum_tangential_acceleration_m_s2"])
            clearance = float(item["minimum_clearance_m"])
            base = float(item["maximum_base_turn_rad"])
            wrist = float(item["maximum_wrist_swing_rad"])
            if preference == "time_first":
                return duration, pose, acceleration, -clearance, base, wrist
            return -clearance, pose, acceleration, base, wrist, duration

        local = min(
            validated,
            key=lambda item: (*score(item), str(item["candidate_id"])),
        )
        local_id = str(local["candidate_id"])
        if not self.online:
            return local_id, f"local_{preference}"
        system = (
            "Choose one transport candidate from the supplied locally validated IDs. "
            "Use only the measured scores and requested preference. Reply with exactly "
            "one JSON key: selected_candidate_id. Never output or request coordinates, "
            "waypoints, joint angles, forces, or a new candidate."
        )
        try:
            raw = self._chat_json(
                system=system,
                payload={"preference": preference, "validated_candidates": validated},
            )
            selected = str(raw.get("selected_candidate_id", ""))
            valid_ids = {str(item["candidate_id"]) for item in validated}
            if selected in valid_ids:
                return selected, f"{self.model}-transport-candidate-agent"
            return local_id, f"{self.model}-invalid-id-local-fallback"
        except Exception as error:
            return local_id, f"transport-selector-error:{type(error).__name__}:local-fallback"

    def plan_pick_strategy(
        self,
        task: TaskSpec,
        scene: SceneSnapshot,
        frames: dict[str, np.ndarray] | None = None,
    ) -> PickStrategy:
        """Choose five words before the first grasp.  Never blocks the task."""
        if not self.online:
            return default_pick_strategy(task)
        system = (
            "You are the manager of a Franka Panda pick-and-place robot. Before the "
            "first grasp attempt, choose a high-level pick strategy from fixed enums "
            "only. approach: vertical|horizontal. grip_width: narrow|medium|wide "
            "(pre-opening before descent; the controller still measures the object). "
            "lift_height: low|medium|high (trial lift). transport: direct|via_safe_point "
            "(via_safe_point lifts the route above all seen obstacles). speed_profile: "
            "slow|normal|fast (slow uses the fragile motion profile). Prefer "
            "horizontal approach for cups, mugs and bottles; vertical for boxes. "
            "Reply with one JSON object with exactly the keys approach, grip_width, "
            "lift_height, transport, speed_profile, reason. Never output coordinates, "
            "joint angles, qpos, torques, actuator commands or trajectories."
        )
        try:
            raw = self._chat_json(
                system=system,
                payload={"task": task.to_dict(), "scene": scene.to_dict()},
                image_data_url=_compact_contact_sheet(frames),
            )
            parsed = parse_pick_strategy(raw)
            return clip_pick_strategy(
                replace(parsed, backend=f"{self.model}-pick-strategist"), task
            )
        except Exception as error:
            return default_pick_strategy(
                task,
                backend="fallback-default",
                reason=f"LLM 抓取策略不可用，使用默认值：{type(error).__name__}: {error}"[:500],
            )

    def propose_grasp_plan(self, task: TaskSpec, measured: dict) -> object:
        """Ask for one composition. An illegal reply is fed back once, then the local plan."""
        from task_planner import accept_grasp_plan_reply, apply_grasp_plan_feedback, offline_grasp_plan

        if not self.online:
            return offline_grasp_plan(task, measured)
        system = (
            "You compose a grasp from registered skills only: analyze_grasp, "
            "plan_pick, execute_pick, verify_pick, reobserve_scene. Arguments are "
            "semantic words. analyze_grasp.focus: handle_direction|face|body. "
            "plan_pick.region: auto|body|face|middle_side|lower_side|upper_side. "
            "plan_pick.approach: vertical|horizontal. plan_pick.yaw_mode: "
            "geometry|preserve_wrist|avoid_handle|face. execute_pick.mode: validated. "
            "For a mug, look at the handle, then grasp the cup body with avoid_handle. "
            "Never output coordinates, joint angles, forces, or metres. Reply with one "
            "JSON object: steps (array of {name, arguments}) and reason."
        )
        payload = {"task": task.to_dict(), "measured": {
            key: measured.get(key) for key in ("shape", "color", "has_handle", "support")
        }}
        try:
            reply = self._chat_json(system=system, payload=payload)
            return accept_grasp_plan_reply(reply)
        except Exception as error:
            receipt = getattr(error, "receipt", None) or {
                "error": type(error).__name__, "message": str(error)[:300],
            }
        try:
            reply = self._chat_json(
                system=system,
                payload={**payload, "rejection": receipt,
                         "instruction": "The previous composition was rejected. Reply with a valid one."},
            )
            return apply_grasp_plan_feedback(receipt, reply)
        except Exception:
            return offline_grasp_plan(task, measured)

    def replan_grasp(
        self,
        diagnosis: dict[str, object],
        frame: np.ndarray | None = None,
    ) -> GraspStrategyUpdate:
        """Propose the next grasp from a measured failure, then let safety clip it."""
        if not self.online:
            return offline_grasp_replan(diagnosis)
        system = (
            "You are the grasp reasoner for a Franka Panda. A grasp attempt just failed. "
            "The diagnosis is measured geometry and contact, and it is authoritative. "
            "A failure image may be attached as a semantic cross-check, but RGB-D geometry "
            "and signed contact forces own metric alignment. Propose the next high-level "
            "grasp. You may change grasp_strategy "
            "(side_grasp, top_grasp, edge_grasp, handle_grasp), approach_direction "
            "(vertical or horizontal), grasp_region (lower_side, middle_side, or upper_side), "
            "and yaw_policy (geometry, preserve_wrist, or avoid_handle). Select semantic "
            "regions only: the local RGB-D grasp planner owns the exact height, width, yaw, "
            "and position. Do not output grasp_height_ratio, metric lateral offsets, yaw "
            "deltas, or pregrasp distances. "
            "For a box or cube, keep a vertical approach because horizontal entry can push "
            "the object. If fingers closed above the object with zero contact, choose "
            "middle_side and use a horizontal approach. For a cup choose avoid_handle; for "
            "a handleless cylinder preserve the current wrist yaw. Do not repeat a grasp "
            "whose finger height is above the rim. Never output joint angles, "
            "qpos, torque, actuator commands, or velocities. Reply with one JSON object: "
            "action (retry_grasp, reobserve_scene, or safe_stop), need_reobserve (bool), "
            "reason (string), strategy_update (object with the fields above)."
        )
        try:
            raw = self._chat_json(
                system=system,
                payload={"diagnosis": diagnosis},
                image_data_url=(
                    None if frame is None else _compact_contact_sheet({"latest": frame})
                ),
            )
            proposal = parse_grasp_replan(raw, diagnosis)
            clipped = clip_grasp_strategy(proposal, diagnosis)
            return replace(clipped, backend=f"{self.model}-grasp-reasoner")
        except Exception as error:
            if self.mode == "required":
                raise LeaderUnavailableError(f"抓取重规划失败：{error}") from error
            print(f"  在线抓取重规划失败，改用本地失败规则：{error}")
            return offline_grasp_replan(diagnosis)

    def decide_recovery(self, feedback: dict[str, object]) -> RecoveryDecision:
        allowed = {"reobserve_scene", "retry_grasp", "replan_transport", "safe_stop"}
        reason = str(feedback.get("reason", "unknown")).lower()
        safe_to_retry = bool(feedback.get("safe_to_retry", False))
        if not safe_to_retry:
            return RecoveryDecision(
                "safe_stop", "本地安全层判定当前状态不可自动重试。", "local-safety-veto"
            )
        if bool(feedback.get("retries_exhausted", False)):
            return RecoveryDecision(
                "safe_stop",
                "恢复轮次已经用完。失败诊断已用于前面的参数重规划，不再重复同一计划。",
                "local-retry-budget",
            )
        has_diagnosis = isinstance(feedback.get("diagnosis"), dict)
        if not self.online:
            if reason.startswith("grasp_"):
                # A measured diagnosis can drive a different grasp; without
                # one the only safe move is a fresh observation.
                action = "retry_grasp" if has_diagnosis else "reobserve_scene"
            elif reason.startswith("place_") or "not_found" in reason:
                action = "reobserve_scene"
            elif reason.startswith("checkpoint_adjust"):
                action = "retry_grasp"
            elif reason.startswith("transport_"):
                action = "replan_transport"
            elif "slip" in reason or "contact" in reason:
                action = "retry_grasp"
            elif "transport" in reason or "motion_plan" in reason or "collision" in reason:
                action = "reobserve_scene"
            else:
                action = "safe_stop"
            return RecoveryDecision(
                action,
                "根据结构化失败原因选择本地恢复策略。",
                "offline-recovery",
                "time_first" if action == "replan_transport" else None,
            )
        system = (
            "You choose one high-level recovery action for a robot. Reply with JSON fields "
            "action, reason, and route_preference. action must be reobserve_scene, "
            "retry_grasp, replan_transport, or safe_stop. route_preference must be "
            "stability_first or time_first only when replanning transport, otherwise null. "
            "Never output motion commands."
        )
        try:
            raw = self._chat_json(system=system, payload={"feedback": feedback})
            action = str(raw.get("action", "safe_stop"))
            if action not in allowed:
                action = "safe_stop"
            # Failure-stage evidence is authoritative.  A failure before
            # contact cannot be repaired by replanning a transport that has
            # not started yet, even if the LLM suggests it.
            if reason.startswith("grasp_") and action == "replan_transport":
                action = "reobserve_scene"
            if reason.startswith("transport_") and action == "retry_grasp":
                action = "replan_transport"
            # After a release the object is no longer in the hand, so neither a
            # transport replan nor a blind regrasp applies; look again first.
            if reason.startswith("place_") and action in {"replan_transport", "retry_grasp"}:
                action = "reobserve_scene"
            preference = str(raw.get("route_preference", ""))
            if action != "replan_transport" or preference not in {
                "stability_first", "time_first",
            }:
                preference = None
            return RecoveryDecision(
                action, str(raw.get("reason", "LLM recovery decision"))[:500],
                f"{self.model}-recovery-agent",
                preference,
            )
        except Exception as error:
            return RecoveryDecision(
                "safe_stop", f"恢复决策失败，保持安全停止：{error}", "recovery-error"
            )

    def usage_summary(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
        }
