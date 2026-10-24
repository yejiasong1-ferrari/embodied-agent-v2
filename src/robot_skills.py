"""Allow-listed high-level skills exposed to the robot's LLM leader.

The language model writes this small JSON DSL, never Python, joint angles,
actuator commands, or trajectories.  The local robot stack resolves every
skill into perception, IK, OMPL, collision checking, Ruckig, and feedback
control before anything can move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from task_planner import DEFAULT_STEPS, TaskSpec


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    phase: str
    mutates_robot: bool = False
    requires_validated_plan: bool = False

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "phase": self.phase,
            "mutates_robot": self.mutates_robot,
            "requires_validated_plan": self.requires_validated_plan,
        }


@dataclass(frozen=True)
class SkillCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class SkillPlan:
    calls: tuple[SkillCall, ...]

    def to_list(self) -> list[dict[str, object]]:
        return [call.to_dict() for call in self.calls]


class SkillPlanError(ValueError):
    """The proposed LLM program is outside the robot's safe skill DSL."""


_DEFINITIONS = (
    SkillDefinition("locate_objects", "主动观察并定位拿取物、放置物和障碍物", "perception"),
    SkillDefinition("analyze_grasp", "根据尺寸、姿态和夹爪范围生成抓取候选", "planning"),
    SkillDefinition("plan_pick", "用 IK/OMPL 规划并验证到抓取位的路径", "planning"),
    SkillDefinition("execute_pick", "执行已经验证的抓取路径", "execution", True, True),
    SkillDefinition("verify_pick", "用双指接触、试抬和相对位姿确认抓取", "validation"),
    SkillDefinition("plan_transport", "为携带物体规划全身及负载无碰撞路径", "planning"),
    SkillDefinition("execute_transport", "用 Ruckig 和负载监控执行已验证路径", "execution", True, True),
    SkillDefinition("verify_transport", "检查滑移、倾斜、接触和掉落", "validation"),
    SkillDefinition("place", "在支撑面上渐进卸载并松爪", "execution", True, True),
    SkillDefinition("verify_place", "检查支撑接触、高度、倾斜和水平误差", "validation"),
    SkillDefinition("reobserve_scene", "失败后从新视角更新目标和障碍物", "recovery"),
    SkillDefinition("safe_stop", "停止控制并保持当前安全状态", "safety", True),
)


class SkillLibrary:
    """Registry and validator for high-level programs proposed by an LLM."""

    def __init__(self, definitions: Iterable[SkillDefinition] = _DEFINITIONS) -> None:
        self._definitions = {definition.name: definition for definition in definitions}

    def manifest(self) -> list[dict[str, object]]:
        return [definition.to_prompt_dict() for definition in self._definitions.values()]

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._definitions)

    def canonical_plan(self, task: TaskSpec) -> SkillPlan:
        calls: list[SkillCall] = []
        for name in task.steps:
            arguments: dict[str, Any] = {}
            if name == "locate_objects":
                arguments = {
                    "target": f"{task.object_color} {task.object_name}",
                    "destination": (
                        None if not task.reference else
                        f"{task.reference_color or 'yellow'} {task.reference}"
                    ),
                    "observe_obstacles": True,
                }
            elif name == "plan_transport":
                arguments = {
                    "transport_mode": task.constraints.transport_mode,
                    "avoid_collisions": True,
                    "include_payload_geometry": True,
                }
            elif name in {"execute_pick", "execute_transport", "place"}:
                arguments = {"validated_plan_only": True}
            elif name.startswith("verify_"):
                arguments = {"sensor_evidence_required": True}
            calls.append(SkillCall(name, arguments))
        return SkillPlan(tuple(calls))

    def validate_plan(self, raw_calls: object, task: TaskSpec) -> SkillPlan:
        if not isinstance(raw_calls, list) or not raw_calls:
            raise SkillPlanError("skill_calls 必须是非空数组")
        if len(raw_calls) > 20:
            raise SkillPlanError("技能调用过多")
        calls: list[SkillCall] = []
        forbidden_fragments = {
            "joint", "qpos", "qvel", "actuator", "ctrl", "torque",
            "trajectory", "python", "shell", "force_n", "关节", "力矩",
        }

        def audit_value(value: object, path: str) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    lowered = str(key).lower()
                    if any(fragment in lowered for fragment in forbidden_fragments):
                        raise SkillPlanError(f"禁止的底层参数：{path}.{key}")
                    audit_value(child, f"{path}.{key}")
            elif isinstance(value, list):
                if len(value) > 32:
                    raise SkillPlanError(f"参数数组过长：{path}")
                for index, child in enumerate(value):
                    audit_value(child, f"{path}[{index}]")
            elif not isinstance(value, (str, int, float, bool, type(None))):
                raise SkillPlanError(f"不支持的参数类型：{path}")

        for index, raw in enumerate(raw_calls):
            if not isinstance(raw, dict):
                raise SkillPlanError(f"skill_calls[{index}] 不是对象")
            name = str(raw.get("name", ""))
            if name not in self._definitions:
                raise SkillPlanError(f"未授权技能：{name or '<empty>'}")
            arguments = raw.get("arguments", {})
            if not isinstance(arguments, dict):
                raise SkillPlanError(f"{name}.arguments 必须是对象")
            audit_value(arguments, name)
            definition = self._definitions[name]
            if definition.mutates_robot and definition.requires_validated_plan:
                if arguments.get("validated_plan_only") is not True:
                    raise SkillPlanError(f"{name} 必须声明 validated_plan_only=true")
            calls.append(SkillCall(name, dict(arguments)))

        names = [call.name for call in calls]
        cursor = -1
        for required in task.steps:
            try:
                cursor = names.index(required, cursor + 1)
            except ValueError as error:
                raise SkillPlanError(f"缺少或乱序的必要技能：{required}") from error
        for execute_name, plan_name in (
            ("execute_pick", "plan_pick"),
            ("execute_transport", "plan_transport"),
        ):
            if execute_name in names and names.index(plan_name) > names.index(execute_name):
                raise SkillPlanError(f"{execute_name} 不能早于 {plan_name}")
        return SkillPlan(tuple(calls))


DEFAULT_SKILL_LIBRARY = SkillLibrary()

# Semantic arguments a grasp composition may use. Values are words, never metres.
GRASP_PLAN_ARGUMENTS: dict[str, dict[str, frozenset[str]]] = {
    "analyze_grasp": {"focus": frozenset({"handle_direction", "face", "body"})},
    "plan_pick": {
        "region": frozenset({"auto", "body", "face", "middle_side", "lower_side", "upper_side"}),
        "approach": frozenset({"vertical", "horizontal"}),
        "yaw_mode": frozenset({"geometry", "preserve_wrist", "avoid_handle", "face"}),
    },
    "execute_pick": {"mode": frozenset({"validated"})},
    "verify_pick": {"mode": frozenset({"sensor"})},
    "reobserve_scene": {"reason": frozenset({"contact_evidence", "region_shift"})},
}
_COORDINATE_FRAGMENTS = (
    "coord", "xyz", "position", "metre", "meter", "offset", "yaw_rad",
    "force", "joint", "qpos", "torque", "trajectory",
)


class GraspPlanRejected(SkillPlanError):
    """A proposed grasp composition failed local validation."""

    def __init__(self, message: str, receipt: dict[str, object]) -> None:
        super().__init__(message)
        self.receipt = receipt


def validate_grasp_steps(raw_steps: object) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Accept an ordered skill list of semantic arguments only."""
    if not isinstance(raw_steps, list) or not raw_steps:
        raise GraspPlanRejected("grasp_plan.steps 必须是非空数组", {"error": "empty_steps"})
    if len(raw_steps) > 8:
        raise GraspPlanRejected("抓取方案步数过多", {"error": "too_many_steps"})
    checked: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise GraspPlanRejected(
                f"steps[{index}] 不是对象", {"error": "step_type", "index": index},
            )
        name = str(raw.get("name", ""))
        if name not in DEFAULT_SKILL_LIBRARY.names:
            raise GraspPlanRejected(
                f"未注册功能：{name or '<empty>'}",
                {"error": "unregistered", "rejected_name": name or "<empty>", "index": index},
            )
        if name not in GRASP_PLAN_ARGUMENTS:
            raise GraspPlanRejected(
                f"功能 {name} 不能编进抓取方案",
                {"error": "not_a_grasp_step", "rejected_name": name, "index": index},
            )
        arguments = raw.get("arguments", {})
        if not isinstance(arguments, dict):
            raise GraspPlanRejected(
                f"{name}.arguments 必须是对象",
                {"error": "arguments_type", "rejected_name": name},
            )
        allowed = GRASP_PLAN_ARGUMENTS[name]
        pairs: list[tuple[str, str]] = []
        for key, value in arguments.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in _COORDINATE_FRAGMENTS):
                raise GraspPlanRejected(
                    f"{name}.{key} 是数值坐标或底层字段",
                    {"error": "numeric_coordinate", "rejected_name": name, "key": str(key)},
                )
            if isinstance(value, (int, float, list, tuple)):
                raise GraspPlanRejected(
                    f"{name}.{key} 带了数值，方案只许语义参数",
                    {"error": "numeric_coordinate", "rejected_name": name, "key": str(key)},
                )
            if lowered not in allowed or not isinstance(value, str) or value not in allowed[lowered]:
                raise GraspPlanRejected(
                    f"{name}.{key}={value!r} 不是该功能的合法语义参数",
                    {"error": "illegal_argument", "rejected_name": name, "key": str(key)},
                )
            pairs.append((lowered, value))
        checked.append((name, tuple(pairs)))
    return tuple(checked)
