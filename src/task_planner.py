"""Turn a natural-language robot command into a checked task specification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import os
import re

from object_registry import (
    OBJECTS, BY_ID, color_tokens, generic_record, name_matches,
    object_name_from_text, record_for_body, record_for_color, registry_catalog,
    shape_query, shape_query_in_text, shapes_compatible,
)

DEFAULT_BASE_URL = (
    "https://llm-us3whhqdyc0mhmso.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)


DEFAULT_STEPS = (
    "locate_objects",
    "analyze_grasp",
    "plan_pick",
    "execute_pick",
    "verify_pick",
    "plan_transport",
    "execute_transport",
    "verify_transport",
    "place",
    "verify_place",
)


@dataclass(frozen=True)
class TaskConstraints:
    avoid_collisions: bool = True
    transport_mode: str = "normal"
    verify_grasp: bool = True
    max_retries: int = 3


@dataclass(frozen=True)
class TaskSpec:
    action: str
    object_name: str
    object_color: str
    reference: str | None
    relation: str | None
    grasp_strategy: str
    approach_direction: str
    reason: str
    reference_color: str | None = None
    object_id: str | None = None
    object_confidence: str = "high"
    constraints: TaskConstraints = field(default_factory=TaskConstraints)
    steps: tuple[str, ...] = DEFAULT_STEPS
    # How the noun was resolved: alias, shape_match, manager_rereview,
    # inventory_temporary.
    parse_path: str = ""
    # Set only for a task-local temporary entry. Registered plans leave these
    # empty so execution keeps using the registry body.
    body_name: str | None = None
    measured_size_m: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TaskValidationError(ValueError):
    """The language plan breaks a safety check or names a missing reference."""


class CategoryMismatch(Exception):
    """Color hit a registry object, but the requested category does not.

    This is a receipt for one manager re-review, not a terminal failure.
    """

    def __init__(self, task: TaskSpec, record: "ObjectRecord") -> None:
        self.task = task
        self.record = record
        self.receipt = {
            "requested_color": task.object_color,
            "requested_name": task.object_name,
            "object_id": record.object_id,
            "display_name": record.display_name,
            "shape": record.shape_prior,
            "aliases": list(record.aliases),
        }
        super().__init__(
            f"{task.object_color} {task.object_name} != {record.object_id}"
        )


class NotFoundReport(Exception):
    """The manager looked and could not justify a grasp. No stack-trace death."""

    def __init__(self, message: str, details: dict[str, object]) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


def _color_zh(color: str) -> str:
    record = record_for_color(color)
    if record is not None:
        for alias in record.color_aliases:
            if any("\u4e00" <= char <= "\u9fff" for char in alias):
                return alias
    return {
        "pink": "粉色", "black": "黑色", "gray": "灰色", "brown": "棕色",
    }.get(color, color)


def _name_zh(name: str, record_name: str | None = None) -> str:
    if any("\u4e00" <= char <= "\u9fff" for char in str(name)):
        return str(name)
    return {
        "cube": "方块", "box": "方块", "block": "方块", "cup": "马克杯",
        "mug": "马克杯", "cylinder": "圆柱", "bottle": "瓶子", "plate": "盘子",
    }.get(record_name or name, str(name))


def not_found_for(color: str, name: str, *, reason: str) -> NotFoundReport:
    """Human report. Same color / same shape are hints, never a substitute grasp."""
    same_color = record_for_color(color)
    nearest: list[dict[str, object]] = []
    if same_color is not None and same_color.movable:
        nearest.append({
            "relation": "same_color",
            "object_id": same_color.object_id,
            "category": same_color.display_name,
            "shape": same_color.shape_prior,
        })
    for record in OBJECTS:
        if not record.movable or record.primary_color == color:
            continue
        if shapes_compatible(name, record) or (
            shape_query(name) == "cylinder" and record.shape_prior in {"cylinder", "handled_cylinder"}
        ):
            nearest.append({
                "relation": "same_shape",
                "object_id": record.object_id,
                "color": record.primary_color,
                "category": record.display_name,
                "shape": record.shape_prior,
            })
            if len([item for item in nearest if item["relation"] == "same_shape"]) >= 2:
                break
    color_text = _color_zh(color)
    asked = _name_zh(name)
    if same_color is not None and same_color.movable and not shapes_compatible(name, same_color):
        actual = _name_zh(same_color.display_name, same_color.display_name)
        message = f"没找到{color_text}{asked}；场景里{color_text}的是{actual}，是不是它？"
    elif same_color is None:
        message = f"没找到{color_text}{asked}；场景里没有这个颜色的物体。"
    else:
        message = f"没找到{color_text}{asked}。"
    return NotFoundReport(message, {
        "requested_color": color,
        "requested_name": name,
        "searched": "registry",
        "nearest": nearest,
        "suggestion": message,
        "reason": reason,
    })


def apply_manager_rereview(mismatch: CategoryMismatch, reply: dict[str, object]) -> TaskSpec:
    """One re-review. A contradictory shape or color cannot select a substitute."""
    decision = str(reply.get("decision", "not_found"))
    requested = mismatch.task
    if decision != "select":
        raise not_found_for(
            requested.object_color, requested.object_name, reason="manager_not_found",
        )
    record = BY_ID.get(str(reply.get("object_id", "")))
    if record is None or record.primary_color != requested.object_color:
        raise not_found_for(
            requested.object_color, requested.object_name, reason="color_contradiction",
        )
    if not shapes_compatible(requested.object_name, record):
        raise not_found_for(
            requested.object_color, requested.object_name, reason="shape_contradiction",
        )
    revised = replace(
        requested,
        object_name=record.display_name,
        object_color=record.primary_color,
        parse_path="manager_rereview",
    )
    try:
        checked = validate_task(revised)
    except CategoryMismatch:
        raise not_found_for(
            requested.object_color, requested.object_name, reason="rereview_still_inconsistent",
        ) from None
    return replace(checked, parse_path="manager_rereview")


SCENE_OBJECTS = {
    record.primary_color: {
        "id": record.object_id,
        "name": record.display_name,
        "movable": record.movable,
    }
    for record in OBJECTS
}


def _inventory_match(inventory: list, color: str):
    same = [item for item in inventory if getattr(item, "color", None) == color]
    high = [
        item for item in same
        if getattr(item, "confidence", "") == "high" and getattr(item, "body_name", None)
    ]
    if high:
        return high[0]
    bound = [item for item in same if getattr(item, "body_name", None)]
    fresh = [
        item for item in bound
        if record_for_body(getattr(item, "body_name", None)) is None
    ]
    if fresh:
        return fresh[0]
    if bound:
        return bound[0]
    return same[0] if same else None


def validate_task(task: TaskSpec, inventory: list | None = None) -> TaskSpec:
    """Audit an LLM/rule plan against the actual MuJoCo scene.

    Color alone is not enough: asking for a red cube must not silently select
    the red cup.  "Blue/green block" is accepted as the box-shaped scene
    object, while incompatible shape/color pairs are rejected visibly.
    """
    if not task.constraints.avoid_collisions or not task.constraints.verify_grasp:
        raise TaskValidationError("碰撞检查和抓取验证是不可绕过的硬约束。")
    if task.constraints.transport_mode not in {"normal", "fragile"}:
        raise TaskValidationError("transport_mode 必须是 normal 或 fragile。")
    if not 1 <= task.constraints.max_retries <= 5:
        raise TaskValidationError("max_retries 必须位于 1..5。")
    unknown_steps = set(task.steps) - set(DEFAULT_STEPS)
    if unknown_steps:
        raise TaskValidationError(f"任务包含未授权技能步骤：{sorted(unknown_steps)}")
    forced_low = None
    if inventory is not None:
        match = _inventory_match(inventory, task.object_color)
        if match is None:
            raise not_found_for(
                task.object_color, task.object_name, reason="inventory_miss",
            )
        if getattr(match, "confidence", "") == "low":
            if not getattr(match, "body_name", None):
                raise not_found_for(
                    task.object_color, task.object_name, reason="temporary_unbound",
                )
            forced_low = match
    if forced_low is not None:
        target_record = replace(
            generic_record(task.object_color, task.object_name),
            object_id=str(forced_low.object_id),
            body_name=str(forced_low.body_name),
            dimensions_m=tuple(forced_low.dimensions_m),
        )
        actual_name = task.object_name
        confidence = "low"
    else:
        target_record = record_for_color(task.object_color)
        if target_record is None:
            target_record = generic_record(task.object_color, task.object_name)
            actual_name = target_record.display_name
            confidence = "low"
        else:
            actual_name = target_record.display_name
            confidence = "high"
            if not name_matches(target_record, task.object_name):
                raise CategoryMismatch(task, target_record)
    matched_by_shape = (
        target_record is not None
        and confidence == "high"
        and shape_query(task.object_name) is not None
        and name_matches(target_record, task.object_name)
    )
    parse_path = task.parse_path or (
        "inventory_temporary" if confidence == "low" and forced_low is not None
        else "shape_match" if matched_by_shape else "alias"
    )
    if not target_record.movable:
        raise TaskValidationError(f"{task.object_color} {actual_name} 是放置参照物，不能作为拿取目标。")

    checked = replace(
        task,
        object_name=actual_name,
        object_id=target_record.object_id,
        object_confidence=confidence,
        grasp_strategy=target_record.grasp_strategy,
        approach_direction=target_record.approach_direction,
        parse_path=parse_path,
        body_name=target_record.body_name if confidence == "low" and forced_low is not None else None,
        measured_size_m=(
            tuple(target_record.dimensions_m)
            if confidence == "low" and forced_low is not None else None
        ),
    )
    # The current Panda skill library implements a vertical parallel-jaw
    # pinch on the cup body; it does not contain a validated handle grasp.
    # Do not advertise an LLM-selected skill that the executor cannot perform.
    if actual_name in {"cup", "bottle"} or (confidence == "low" and forced_low is not None):
        checked = replace(
            checked,
            grasp_strategy="side_grasp",
            approach_direction="vertical",
        )
    if checked.reference:
        reference_color = checked.reference_color or "yellow"
        reference_record = record_for_color(reference_color)
        if reference_record is None:
            raise TaskValidationError(f"场景中没有 {reference_color} 色参照物。")
        actual_reference = reference_record.display_name
        if not name_matches(reference_record, checked.reference):
            raise TaskValidationError(
                f"指令要求参照物 {reference_color} {checked.reference}，"
                f"但场景中实际是 {actual_reference}。"
            )
        checked = replace(checked, reference=actual_reference, reference_color=reference_color)
    # Target semantics, not an unreliable single LLM field, determine whether
    # this is pick-only or pick-and-place.  A named destination means the
    # object must not be reported complete immediately after the trial lift.
    placing = checked.reference is not None
    checked = replace(
        checked,
        action="place" if placing else "pick",
        relation=(checked.relation or "on") if placing else None,
        steps=DEFAULT_STEPS if placing else DEFAULT_STEPS[:5],
    )
    return checked


def _extract_json(text: str) -> dict:
    """Accept plain JSON or a JSON object wrapped in a Markdown code fence."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"VLM did not return a JSON object: {text[:200]}")
    return json.loads(match.group(0))


def _fallback_plan(instruction: str) -> TaskSpec:
    """Deterministic offline fallback for the objects in our MuJoCo scene."""
    text = instruction.lower()
    colors = color_tokens()
    colors.update({
        "pink": "pink", "粉色": "pink", "粉": "pink",
        "black": "black", "黑色": "black", "黑": "black",
        "gray": "gray", "grey": "gray", "灰色": "gray", "灰": "gray",
        "brown": "brown", "棕色": "brown", "棕": "brown",
    })
    # Separate the thing being moved from its destination.  Looking for nouns
    # in the whole sentence made "把橙色方块拿到黄色盘子上" choose the plate as
    # the target merely because the word 盘 appeared later in the sentence.
    target_text = text
    reference_text = ""
    for delimiter in ("移动到", "搬到", "拿到", "放到", "放在", "移到", "put ", "move "):
        if delimiter in text:
            target_text, reference_text = text.split(delimiter, 1)
            break

    def find_color(fragment: str, default: str | None) -> str | None:
        return next(
            (
                value
                for key, value in sorted(colors.items(), key=lambda item: -len(item[0]))
                if key in fragment
            ),
            default,
        )

    def find_object(fragment: str, default: str | None = None) -> str | None:
        named = object_name_from_text(fragment, None)
        if named:
            return named
        shaped = shape_query_in_text(fragment)
        if shaped:
            return shaped
        return default

    color = find_color(target_text, "orange") or "orange"
    object_name = find_object(target_text)
    if object_name is None:
        raise not_found_for(color, "物体", reason="unknown_noun")
    if record_for_color(color) is None:
        raise not_found_for(color, object_name, reason="color_absent")
    target_record = record_for_color(color) or generic_record(color, object_name)
    grasp, approach = target_record.grasp_strategy, target_record.approach_direction

    reference = find_object(reference_text)
    reference_color = find_color(reference_text, "yellow" if reference == "plate" else None)
    if any(x in reference_text for x in ("旁", "next to", "beside")):
        relation = "next_to"
    elif any(x in reference_text for x in ("上", "on ", "onto")):
        relation = "on"
    elif any(x in reference_text for x in ("里", "内", "inside", "into")):
        relation = "inside"
    else:
        relation = None
    fragile = any(word in text for word in (
        "不要滑", "滑落", "防滑", "平稳", "稳定", "小心", "轻拿",
        "fragile", "stable", "gently", "careful",
    ))
    steps = DEFAULT_STEPS if reference else DEFAULT_STEPS[:5]
    return TaskSpec(
        action="place" if reference else "pick",
        object_name=object_name,
        object_color=color,
        reference=reference,
        relation=relation,
        grasp_strategy=grasp,
        approach_direction=approach,
        reason="Offline rule-based plan; configure DASHSCOPE_API_KEY to use the VLM.",
        reference_color=reference_color,
        constraints=TaskConstraints(transport_mode="fragile" if fragile else "normal"),
        steps=steps,
    )


def plan_task(
    instruction: str,
    *,
    use_llm: bool | None = None,
    inventory: list | None = None,
) -> tuple[TaskSpec, str]:
    """Use Qwen-VL when configured; otherwise return a runnable offline plan."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if use_llm is False or not api_key:
        try:
            return validate_task(
                _fallback_plan(instruction), inventory=inventory,
            ), "offline-fallback + scene-audit"
        except CategoryMismatch as mismatch:
            raise not_found_for(
                mismatch.task.object_color,
                mismatch.task.object_name,
                reason="offline_category_mismatch",
            ) from None

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("VLM_BASE_URL", DEFAULT_BASE_URL),
    )
    schema = {
        "action": "pick or place",
        "object_name": (
            "the user's own short noun, which may be a shape word such as "
            "cylinder; do not replace an off-catalog word with a catalog category"
        ),
        "object_color": "English basic color",
        "reference": "reference object or null",
        "relation": "next_to, on, inside, or null",
        "grasp_strategy": "top_grasp, side_grasp, edge_grasp, or handle_grasp",
        "approach_direction": "vertical or horizontal",
        "reason": "short safety-oriented explanation",
        "reference_color": "English basic color of the reference, or null",
        "constraints": {
            "avoid_collisions": "must be true",
            "transport_mode": "normal or fragile",
            "verify_grasp": "must be true",
            "max_retries": "integer 1..5",
        },
        "steps": list(DEFAULT_STEPS),
    }
    response = client.chat.completions.create(
        model=os.getenv("VLM_MODEL", "qwen3-vl-plus"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a robot task planner. Convert the user's instruction into "
                    "one JSON object only. Do not output joint angles, coordinates, or "
                    "forces. Choose grasp style from object geometry. Prefer a real "
                    "entry from the scene catalog. If the user used a description that "
                    "is not a catalog category, return that description as object_name "
                    "(a short noun or shape word). Do not force it into cube/box/cup/"
                    "bottle/plate. Scene catalog: "
                    + json.dumps(list(registry_catalog()), ensure_ascii=False)
                    + ". Required schema: " + json.dumps(schema)
                ),
            },
            {"role": "user", "content": instruction},
        ],
        temperature=0,
    )
    payload = _extract_json(response.choices[0].message.content or "")
    payload.setdefault("reference_color", None)
    raw_constraints = payload.pop("constraints", {})
    payload["constraints"] = TaskConstraints(
        avoid_collisions=bool(raw_constraints.get("avoid_collisions", True)),
        transport_mode=str(raw_constraints.get("transport_mode", "normal")),
        verify_grasp=bool(raw_constraints.get("verify_grasp", True)),
        max_retries=int(raw_constraints.get("max_retries", 3)),
    )
    payload["steps"] = tuple(payload.get("steps", DEFAULT_STEPS))
    return validate_task(TaskSpec(**payload), inventory=inventory), "qwen-vl + scene-audit"


def _clamp_regrasp(raw: dict, context: dict) -> dict[str, float | str]:
    """Keep a language suggestion inside one small, safe correction."""
    def number(name: str, default: float, low: float, high: float) -> float:
        try:
            value = float(raw.get(name, default))
        except (TypeError, ValueError):
            value = default
        return float(min(high, max(low, value)))

    yaw = number("yaw_deg", float(context.get("yaw_deg", 0.0)), -180.0, 180.0)
    return {
        "dx_m": number("dx_m", 0.0, -0.03, 0.03),
        "dy_m": number("dy_m", 0.0, -0.03, 0.03),
        "dz_m": number("dz_m", 0.0, -0.03, 0.02),
        "yaw_deg": yaw,
        "gap_m": number("gap_m", float(context.get("gap_m", 0.04)), 0.0, 0.075),
        "reason": str(raw.get("reason", "调整抓取点后重试"))[:300],
    }


def _fallback_regrasp(context: dict) -> dict[str, float | str]:
    """Geometric correction used when the VLM is not configured or fails.

    The numbers are deliberately small: the next attempt also re-observes the
    object from above, so this only biases that new measurement.
    """
    yaw = float(context.get("yaw_deg", 0.0))
    gap = float(context.get("gap_m", 0.04))
    symptom = str(context.get("symptom", "not_lifted"))
    if symptom == "tipped":
        raw = {
            "dx_m": 0.0, "dy_m": 0.0, "dz_m": -0.012, "yaw_deg": yaw,
            "gap_m": gap + 0.006,
            "reason": (
                "物体在闭合时被推倒，说明夹持点偏心或闭合太猛。"
                "重新俯视定位后，把夹持高度降低 1.2 cm，并少夹 6 mm，减小推倒力矩。"
            ),
        }
    elif symptom == "no_contact":
        raw = {
            "dx_m": 0.0, "dy_m": 0.0, "dz_m": -0.015, "yaw_deg": yaw,
            "gap_m": min(0.074, gap + 0.01),
            "reason": "两侧指尖没有同时碰到物体。降低抓取点 1.5 cm，并先张大开口再闭合。",
        }
    else:
        # Contact happened, but the object stayed on the table.  Tighten once;
        # a later attempt also turns 90 degrees onto the other horizontal face.
        turn = 90.0 if int(context.get("attempt", 1)) >= 2 else 0.0
        raw = {
            "dx_m": 0.0, "dy_m": 0.0, "dz_m": -0.008, "yaw_deg": yaw + turn,
            "gap_m": max(0.0, gap - 0.008),
            "reason": (
                "指尖碰到了物体，但抬升后物体没有跟着上来。"
                "略微加紧，并把夹持点降低 8 mm。"
            ),
        }
    return _clamp_regrasp(raw, context)


def advise_regrasp(context: dict) -> tuple[dict[str, float | str], str]:
    """Ask the VLM how to fix a grasp that did not actually lift the object.

    The model may only shift the Cartesian grasp, the wrist yaw, and the
    opening.  Joint angles and weld constraints are not part of the schema.
    Without an API key this returns the same kind of correction from geometry.
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return _fallback_regrasp(context), "offline-regrasp"

    from openai import OpenAI

    schema = {
        "dx_m": "lateral correction in the base frame, -0.03 to 0.03",
        "dy_m": "lateral correction in the base frame, -0.03 to 0.03",
        "dz_m": "vertical correction of the grasp point, -0.03 to 0.02",
        "yaw_deg": "absolute finger-closing yaw about the vertical, degrees",
        "gap_m": "commanded inner finger gap in metres, 0 to 0.075",
        "reason": "one sentence on why the previous grasp missed",
    }
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("VLM_BASE_URL", DEFAULT_BASE_URL),
    )
    try:
        response = client.chat.completions.create(
            model=os.getenv("VLM_MODEL", "qwen3-vl-plus"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You correct a failed robot grasp. Reply with one JSON object "
                        "only. Do not output joint angles. A tall block tips when the "
                        "fingers close off-center or too high; a slip means the opening "
                        "was too wide. Required schema: " + json.dumps(schema)
                    ),
                },
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            temperature=0,
        )
        payload = _extract_json(response.choices[0].message.content or "")
    except Exception as error:
        advice = _fallback_regrasp(context)
        advice["reason"] = f"{advice['reason']}（VLM 不可用：{error}）"
        return advice, "offline-regrasp"
    return _clamp_regrasp(payload, context), "qwen-vl-regrasp"


# Half the 6 mm gap between the repeated white-mug grasp width (0.064 m)
# and the red cup (0.070 m). Same-object repeats matched exactly; the other
# handled cup must not.
CASE_SIZE_TOLERANCE_M = 0.003


@dataclass(frozen=True)
class GraspStep:
    name: str
    arguments: tuple[tuple[str, str], ...] = ()

    def arg(self, key: str, default: str | None = None) -> str | None:
        for name, value in self.arguments:
            if name == key:
                return value
        return default

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class GraspPlan:
    steps: tuple[GraspStep, ...]
    reason: str
    source: str
    case_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "reason": self.reason,
            "source": self.source,
            "case_id": self.case_id,
        }

    def plan_pick_argument(self, key: str, default: str | None = None) -> str | None:
        for step in self.steps:
            if step.name == "plan_pick":
                return step.arg(key, default)
        return default


def _steps_from_pairs(pairs: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]) -> tuple[GraspStep, ...]:
    return tuple(GraspStep(name, arguments) for name, arguments in pairs)


def accept_grasp_plan_reply(raw: object) -> GraspPlan:
    """Validate one manager reply. Raises GraspPlanRejected with a receipt."""
    from robot_skills import GraspPlanRejected, validate_grasp_steps

    if not isinstance(raw, dict):
        raise GraspPlanRejected("抓取方案不是 JSON 对象", {"error": "not_object"})
    reason = str(raw.get("reason", ""))[:500]
    steps = _steps_from_pairs(validate_grasp_steps(raw.get("steps")))
    return GraspPlan(steps, reason or "经理给出的抓取组合", "custom")


def apply_grasp_plan_feedback(receipt: object, reply: object) -> GraspPlan:
    """Feed a rejection receipt back and accept only a new valid reply."""
    from robot_skills import GraspPlanRejected

    if not isinstance(receipt, dict) or "error" not in receipt:
        raise GraspPlanRejected("回喂缺少拒绝回执", {"error": "missing_receipt"})
    return accept_grasp_plan_reply(reply)


def default_grasp_plan(task: TaskSpec) -> GraspPlan:
    """The monolithic flow already in the executor. No yaw override."""
    approach = task.approach_direction if task.approach_direction in {"vertical", "horizontal"} else "vertical"
    return GraspPlan(
        steps=(
            GraspStep("plan_pick", (("region", "auto"), ("approach", approach))),
            GraspStep("execute_pick", (("mode", "validated"),)),
        ),
        reason="默认单体抓取，与现行流程一致",
        source="default",
    )


def offline_grasp_plan(task: TaskSpec, measured: dict | None = None) -> GraspPlan:
    """Local composition when no case matches. Semantic parameters only."""
    measured = measured or {}
    shape = str(measured.get("shape") or "")
    name = str(task.object_name)
    if name in {"cup", "mug"} or shape == "handled_cylinder":
        steps = (
            GraspStep("analyze_grasp", (("focus", "handle_direction"),)),
            GraspStep("plan_pick", (
                ("region", "body"), ("approach", "vertical"), ("yaw_mode", "avoid_handle"),
            )),
            GraspStep("execute_pick", (("mode", "validated"),)),
        )
        reason = "带把手的杯子先看把手朝向，再只在杯身上定抓取中心，偏航不压把手"
    elif name in {"box", "cube", "block"} or shape == "box":
        steps = (
            GraspStep("analyze_grasp", (("focus", "face"),)),
            GraspStep("plan_pick", (
                ("region", "face"), ("approach", "vertical"), ("yaw_mode", "face"),
            )),
            GraspStep("execute_pick", (("mode", "validated"),)),
        )
        reason = "方块先认可见的面，再沿垂直接近夹那一面"
    else:
        steps = (
            GraspStep("analyze_grasp", (("focus", "body"),)),
            GraspStep("plan_pick", (
                ("region", "body"), ("approach", "vertical"), ("yaw_mode", "preserve_wrist"),
            )),
            GraspStep("execute_pick", (("mode", "validated"),)),
        )
        reason = "未套用杯或方块模板，先看杯身再保持当前腕部"
    return GraspPlan(steps, reason, "custom")


def grasp_center_for_plan(
    plan: GraspPlan,
    perceived_xy: tuple[float, float],
    body_region_xy: tuple[float, float] | None,
) -> tuple[float, float]:
    """Body plans use the in-region RGB-D center. Every other plan keeps the current point."""
    if plan.plan_pick_argument("region") == "body" and body_region_xy is not None:
        return (float(body_region_xy[0]), float(body_region_xy[1]))
    return (float(perceived_xy[0]), float(perceived_xy[1]))


def match_case(case: dict, measured: dict) -> dict[str, object]:
    """Every criterion must pass. A miss is not a partial match."""
    shape_ok = str(case.get("shape") or "") == str(measured.get("shape") or "") and bool(case.get("shape"))
    try:
        difference = abs(float(case.get("width_m")) - float(measured.get("width_m")))
    except (TypeError, ValueError):
        difference = float("inf")
    size_ok = difference <= CASE_SIZE_TOLERANCE_M
    handle_ok = bool(case.get("has_handle")) == bool(measured.get("has_handle"))
    support_ok = str(case.get("support") or "") == str(measured.get("support") or "") and bool(case.get("support"))
    criteria = {
        "shape": {"ok": shape_ok, "case": case.get("shape"), "measured": measured.get("shape")},
        "size": {
            "ok": size_ok, "difference_m": None if difference == float("inf") else difference,
            "tolerance_m": CASE_SIZE_TOLERANCE_M,
        },
        "handle": {"ok": handle_ok, "case": bool(case.get("has_handle")), "measured": bool(measured.get("has_handle"))},
        "support": {"ok": support_ok, "case": case.get("support"), "measured": measured.get("support")},
    }
    return {"matched": shape_ok and size_ok and handle_ok and support_ok, "criteria": criteria}


def _plan_from_case(case: dict) -> GraspPlan:
    raw = case.get("grasp_plan") if isinstance(case.get("grasp_plan"), dict) else {}
    built = accept_grasp_plan_reply(raw)
    return GraspPlan(built.steps, str(raw.get("reason") or built.reason), "case", str(case.get("id") or ""))


def select_grasp_plan(
    task: TaskSpec,
    measured: dict,
    blocked_ids: set[str] | None = None,
    proposer: object | None = None,
    cases: list[dict] | None = None,
) -> tuple[GraspPlan, dict[str, object]]:
    """Local match first. Otherwise the proposer, or the offline composition."""
    from task_log import load_grasp_cases

    blocked = blocked_ids or set()
    library = load_grasp_cases() if cases is None else cases
    rejections: list[dict[str, object]] = []
    for case in library:
        case_id = str(case.get("id") or "")
        if case_id and case_id in blocked:
            rejections.append({"case_id": case_id, "criteria": {"reuse": {"ok": False, "reason": "failed_earlier_in_task"}}})
            continue
        report = match_case(case, measured)
        if report["matched"]:
            plan = _plan_from_case(case)
            return plan, {
                "verdict": "case_matched", "case_id": case_id,
                "criteria": report["criteria"], "rejections": rejections,
            }
        rejections.append({"case_id": case_id, "criteria": report["criteria"]})
    if proposer is not None:
        plan = proposer(task, measured)  # type: ignore[operator]
    else:
        plan = offline_grasp_plan(task, measured)
    return plan, {
        "verdict": "case_rejected" if rejections else "cold_start",
        "rejections": rejections,
    }


def revise_grasp_plan(plan: GraspPlan, failure_type: str) -> GraspPlan:
    """Change the function list or a semantic argument. Never return the same steps."""
    names = [step.name for step in plan.steps]
    if "reobserve_scene" not in names:
        steps = (
            GraspStep("reobserve_scene", (("reason", "contact_evidence"),)),
            *plan.steps,
        )
    else:
        updated: list[GraspStep] = []
        for step in plan.steps:
            if step.name != "plan_pick":
                updated.append(step)
                continue
            approach = "horizontal" if step.arg("approach") != "horizontal" else "vertical"
            arguments = tuple(
                (key, approach if key == "approach" else value) for key, value in step.arguments
            )
            if step.arg("approach") is None:
                arguments = arguments + (("approach", approach),)
            updated.append(GraspStep("plan_pick", arguments))
        steps = tuple(updated)
    reason = f"接触证据 {failure_type}，更换组合后再试"
    return GraspPlan(steps, reason, "revised", plan.case_id)


def success_case(task: TaskSpec, measured: dict, plan: GraspPlan) -> dict:
    import time
    return {
        "id": f"{task.object_id or task.object_color}-{int(time.time() * 1000)}",
        "object_id": task.object_id,
        "shape": measured.get("shape"),
        "color": measured.get("color"),
        "width_m": measured.get("width_m"),
        "has_handle": bool(measured.get("has_handle")),
        "handle_direction_rad": measured.get("handle_direction_rad"),
        "support": measured.get("support"),
        "grasp_plan": plan.to_dict(),
        "evidence": {
            "width_m": measured.get("width_m"),
            "has_handle": bool(measured.get("has_handle")),
        },
        "result": "success",
    }
