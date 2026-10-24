"""Shared task loop: perceive, review, execute, diagnose, recover.

Both entry points (Qt viewer and CLI) and the headless bench call
``run_task``.  The LLM leader chooses the recovery action; this module turns
that enumerated choice into a concrete next round (rescan, regrasp with a
clipped strategy, or resume transport on another route) and enforces the
recovery budget.  Joint control, IK, collision checking and all validators
stay in ``camera_demo``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import time
from typing import Any, Callable

import mujoco
import numpy as np

from camera_demo import (
    _body_bottom_z, _body_xyz, _holding_check, _surface_under_xy,
    execute_pick_and_place,
)
from embodied_agent import (
    CHECKPOINT_ADJUSTMENTS, CheckpointReview, GraspStrategyUpdate, LLMRobotLeader,
    PickStrategy, RecoveryDecision, SceneReview, build_scene_snapshot,
    clip_grasp_strategy,
)
from joint_control import PandaController
from object_registry import record_for_color
from task_log import TaskLog
from task_planner import NotFoundReport, TaskSpec
from vision_detector import Detection


ScanFn = Callable[
    [TaskSpec, list[float], list[dict], dict[str, np.ndarray]],
    tuple[Detection | None, Detection | None, float | None],
]

MAX_RECOVERY_ROUNDS = 2
UNSAFE_TO_RETRY = ("object_dropped_to_floor", "runtime_collision")
_ROUTE_ESCALATION = ("time_first", "stability_first")


@dataclass
class TaskOutcome:
    instruction: str
    success: bool = False
    stopped: bool = False
    task: TaskSpec | None = None
    planner_backend: str = ""
    reason: str | None = None
    recovery_rounds: int = 0
    recovery_log: list[dict[str, object]] = field(default_factory=list)
    target: Detection | None = None
    reference: Detection | None = None
    observation_angle_deg: float | None = None
    review: SceneReview | None = None
    route_policy: str = "auto"
    elapsed_s: float = 0.0
    final_object_xyz_m: list[float] | None = None
    final_object_bottom_m: float | None = None
    support_xy_m: list[float] | None = None
    support_top_m: float | None = None
    xy_error_m: float | None = None
    error_vs_used_support_m: float | None = None
    error_vs_locked_reference_m: float | None = None
    locked_support_xy_m: list[float] | None = None
    place_reference_source: str | None = None
    obstacles_marked: int = 0
    scene_ceiling_m: float = 0.0
    task_log_path: str | None = None
    delivery_path: str | None = None
    pick_strategy: PickStrategy | None = None

    def summary(self) -> dict[str, object]:
        return {
            "instruction": self.instruction,
            "success": self.success,
            "stopped": self.stopped,
            "reason": self.reason,
            "recovery_rounds": self.recovery_rounds,
            "route_policy": self.route_policy,
            "elapsed_s": round(self.elapsed_s, 2),
            "final_object_xyz_m": self.final_object_xyz_m,
            "final_object_bottom_m": self.final_object_bottom_m,
            "support_xy_m": self.support_xy_m,
            "support_top_m": self.support_top_m,
            "xy_error_m": self.xy_error_m,
            "error_vs_used_support_m": self.error_vs_used_support_m,
            "error_vs_locked_reference_m": self.error_vs_locked_reference_m,
            "locked_support_xy_m": self.locked_support_xy_m,
            "place_reference_source": self.place_reference_source,
            "task_log_path": self.task_log_path,
            "delivery_path": self.delivery_path,
        }


def _next_route_policy(current: str) -> str:
    for candidate in _ROUTE_ESCALATION:
        if candidate != current:
            return candidate
    return "high_clearance"


def _apply_pick_strategy(
    task: TaskSpec, strategy: PickStrategy, route_policy: str
) -> tuple[TaskSpec, str]:
    """Map transport/speed words onto route policy and motion profile.

    Only tightening is possible: ``via_safe_point`` raises the route, ``slow``
    selects the fragile profile.  ``normal``/``fast`` never relax a fragile
    task, and ``direct`` keeps whatever the scene review already chose.
    """
    if strategy.transport == "via_safe_point":
        route_policy = "high_clearance"
    if strategy.speed_profile == "slow" and task.constraints.transport_mode != "fragile":
        task = replace(task, constraints=replace(task.constraints, transport_mode="fragile"))
    return task, route_policy


def _is_unsafe(reason: str | None) -> bool:
    text = str(reason or "")
    return any(text.startswith(flag) for flag in UNSAFE_TO_RETRY)


def _map_checkpoint_adjustment(
    adjustment: str,
    task: TaskSpec,
    diagnosis: dict[str, object] | None,
    route_policy: str,
    holding: bool,
) -> tuple[RecoveryDecision, GraspStrategyUpdate | None, str, TaskSpec]:
    """Translate one whitelisted checkpoint adjustment into a local action.

    Only conservative moves exist here: a re-observed grasp retry, a higher
    route, the fragile motion profile, or a fresh scan.  Nothing the LLM says
    can loosen a limit, and nothing can start a retry that would drop a held
    object.
    """
    backend = "local-checkpoint-adjust"
    if adjustment == "raise_transport_height":
        return (
            RecoveryDecision("replan_transport", "检查点要求抬高运输路线", backend),
            None, "high_clearance", task,
        )
    if adjustment == "slow_down":
        constraints = replace(task.constraints, transport_mode="fragile")
        slower = replace(task, constraints=constraints)
        return (
            RecoveryDecision("replan_transport", "检查点要求放慢：改用 fragile 运动档", backend),
            None, route_policy, slower,
        )
    if holding:
        # Re-grasping or re-scanning with the object still in hand has no
        # safe put-down skill in this round; stop instead of dropping it.
        return (
            RecoveryDecision(
                "safe_stop", f"物体仍在手中，不能执行 {adjustment}；安全停止", backend
            ),
            None, route_policy, task,
        )
    if adjustment == "reobserve":
        return (
            RecoveryDecision("reobserve_scene", "检查点要求重新观察", backend),
            None, route_policy, task,
        )
    # switch_grasp_side: a semantic request.  The metric yaw is owned by the
    # local candidate planner, so clip_grasp_strategy will zero the delta and
    # force a fresh observation; the retry itself is what changes the side.
    proposal = GraspStrategyUpdate(
        action="retry_grasp",
        grasp_strategy=task.grasp_strategy,
        approach_direction=task.approach_direction,
        grasp_height_ratio=0.5,
        lateral_offset_m=0.0,
        yaw_delta_rad=float(np.pi / 2),
        pregrasp_distance_m=0.10,
        need_reobserve=True,
        reason="检查点要求换一侧抓取",
        backend=backend,
        grasp_region="middle_side",
        yaw_policy="geometry",
    )
    clipped = clip_grasp_strategy(proposal, diagnosis if isinstance(diagnosis, dict) else {})
    return (
        RecoveryDecision("retry_grasp", clipped.reason, backend),
        clipped, route_policy, task,
    )


def _final_geometry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    task: TaskSpec,
    reference: Detection | None,
    outcome: TaskOutcome,
) -> None:
    record = record_for_color(task.object_color)
    body_name = task.body_name or (None if record is None else record.body_name)
    if body_name is None:
        return
    xyz = _body_xyz(model, data, body_name)
    outcome.final_object_xyz_m = [round(float(v), 4) for v in xyz]
    outcome.final_object_bottom_m = round(_body_bottom_z(model, data, body_name), 4)
    if reference is not None:
        support_xy = np.asarray(reference.base_xyz_m[:2], dtype=float)
        if task.relation == "next_to":
            support_xy = support_xy.copy()
            support_xy[0] += 0.16
        outcome.support_xy_m = [round(float(v), 4) for v in support_xy]
        outcome.support_top_m = round(
            max(float(reference.base_xyz_m[2]),
                _surface_under_xy(model, data, support_xy, {body_name})),
            4,
        )
        outcome.xy_error_m = round(float(np.linalg.norm(xyz[:2] - support_xy)), 4)


def run_task(
    instruction: str,
    leader: LLMRobotLeader,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer | None,
    robot: PandaController,
    *,
    scan: ScanFn,
    max_recovery_rounds: int = MAX_RECOVERY_ROUNDS,
    should_stop: Callable[[], bool] | None = None,
    status: Callable[[str], None] | None = None,
    task: TaskSpec | None = None,
    planner_backend: str = "",
    task_log: TaskLog | None = None,
) -> TaskOutcome:
    """Run one instruction to completion, a safe stop, or the recovery budget."""
    started = time.perf_counter()
    outcome = TaskOutcome(instruction=instruction)

    def say(text: str) -> None:
        if status is not None:
            status(text)

    def stop_requested() -> bool:
        return bool(should_stop is not None and should_stop())

    def checkpoint_reviewer(
        stage: str, evidence: dict[str, object], frame: np.ndarray | None
    ) -> CheckpointReview:
        checkpoint = leader.review_checkpoint(stage, evidence, frame)
        print(
            f"LLM checkpoint {stage} ({checkpoint.backend}): "
            f"{checkpoint.decision}"
            + (f"/{checkpoint.adjustment}" if checkpoint.adjustment else "")
            + f" — {checkpoint.reason}"
        )
        return checkpoint

    def object_in_hand() -> bool:
        record = record_for_color(task.object_color) if task is not None else None
        body_name = None
        if task is not None:
            body_name = task.body_name or (None if record is None else record.body_name)
        if model is None or data is None or body_name is None:
            return False
        held, _ = _holding_check(model, data, body_name)
        return bool(held)

    ceiling = [0.0]
    obstacles: list[dict] = []
    log = task_log
    try:
        if task is None:
            try:
                task, planner_backend = leader.interpret_instruction(instruction)
            except NotFoundReport as report:
                outcome.reason = "not_found"
                outcome.planner_backend = "not_found"
                if log is None:
                    log = TaskLog.start(instruction, None, "not_found")
                log.add_checkpoint("not_found", {
                    "message": report.message, **report.details,
                }, "not_found")
                print(report.message)
                return outcome
        outcome.task = task
        outcome.planner_backend = planner_backend
        if log is None:
            log = TaskLog.start(instruction, task, planner_backend)
        print(f"LLM 领导层: {leader.backend_label}（mode={leader.mode}）")
        print("任务:", json.dumps(task.to_dict(), ensure_ascii=False))
        if task.object_confidence == "low" and task.body_name:
            log.add_checkpoint("temporary_entry", {
                "description": f"{task.object_color} {task.object_name}",
                "object_id": task.object_id,
                "body_name": task.body_name,
                "measured_size_m": list(task.measured_size_m or ()),
                "confidence": "low",
            }, "temporary")

        captures: dict[str, np.ndarray] = {}
        target: Detection | None = None
        reference: Detection | None = None
        angle: float | None = None
        review: SceneReview | None = None
        route_policy = "auto"
        pick_strategy: PickStrategy | None = None
        initial_grasp: GraspStrategyUpdate | None = None
        start_from = "pick"
        need_scan = True
        rounds = 0

        def record_round(trigger: str, action: str, reason: str, backend: str,
                         strategy: GraspStrategyUpdate | None) -> None:
            entry = {
                "round": rounds,
                "trigger": trigger,
                "action": action,
                "reason": reason[:500],
                "backend": backend,
                "strategy_update": None if strategy is None else strategy.to_dict(),
                "route_policy": route_policy,
                "elapsed_s": round(time.perf_counter() - started, 2),
            }
            outcome.recovery_log.append(entry)
            if log is not None:
                log.add_recovery_round({"scope": "task", **entry})
            print("恢复轮次:", json.dumps(entry, ensure_ascii=False))

        while True:
            if stop_requested():
                outcome.stopped = True
                outcome.reason = "stopped_by_user"
                break
            if need_scan:
                say("正在扫描，寻找拿取物和放置物…")
                target, reference, angle = scan(task, ceiling, obstacles, captures)
                outcome.target, outcome.reference = target, reference
                outcome.observation_angle_deg = angle
                snap = getattr(reference, "support_snap", None) if reference is not None else None
                if log is not None and isinstance(snap, dict):
                    log.add_checkpoint(
                        "support_point_snap", snap,
                        "adopted" if snap.get("adopted") else "kept",
                    )
                print(f"扫描期间深度最高点 {ceiling[0]:.3f} m，高障碍 {len(obstacles)} 个")
                if stop_requested():
                    outcome.stopped = True
                    outcome.reason = "stopped_by_user"
                    break
                missing = None
                if target is None:
                    missing = "perception_target_not_found"
                elif task.reference and reference is None:
                    missing = "perception_reference_not_found"
                if missing is not None:
                    recovery = leader.decide_recovery({
                        "status": "failed", "stage": "locate_objects",
                        "reason": missing, "object": f"{task.object_color} {task.object_name}",
                        "recovery_round": rounds,
                        "retries_exhausted": rounds >= max_recovery_rounds,
                        "safe_to_retry": True,
                    })
                    record_round(missing, recovery.action, recovery.reason,
                                 recovery.backend, None)
                    if recovery.action != "reobserve_scene" or rounds >= max_recovery_rounds:
                        outcome.reason = (
                            f"recovery_rounds_exhausted:{missing}"
                            if rounds >= max_recovery_rounds else f"{missing}:{recovery.action}"
                        )
                        break
                    rounds += 1
                    continue
                print("拿取物基座坐标:", np.round(target.base_xyz_m, 3), "m")
                if reference is not None:
                    print("放置物基座坐标:", np.round(reference.base_xyz_m, 3), "m")
                scene = build_scene_snapshot(task, target, reference, obstacles, ceiling[0])
                review = leader.review_scene(instruction, task, scene, frames=captures)
                outcome.review = review
                if log is not None:
                    log.add_checkpoint("scene_review", {
                        "target": None if scene.target is None else scene.target,
                        "destination": None if scene.destination is None else scene.destination,
                        "obstacles": len(scene.obstacles),
                        "maximum_obstacle_top_m": scene.maximum_obstacle_top_m,
                    }, review)
                print(
                    f"LLM 场景审核 ({review.backend}): approved={review.approved}, "
                    f"reobserve={review.needs_reobserve}, transport={review.transport_mode}, "
                    f"route={review.route_policy}"
                )
                print("  审核理由:", review.reason)
                print("  受约束技能程序:", json.dumps(review.skill_plan.to_list(), ensure_ascii=False))
                if not review.approved or review.needs_reobserve:
                    record_round("scene_review_reobserve", "reobserve_scene",
                                 review.reason, review.backend, None)
                    if rounds >= max_recovery_rounds:
                        outcome.reason = "recovery_rounds_exhausted:scene_review_reobserve"
                        break
                    rounds += 1
                    continue
                task = leader.apply_review(task, review)
                outcome.task = task
                route_policy = review.route_policy
                # Manager picks five words before the first attempt; the local
                # layer maps them and may only make the task more conservative.
                pick_strategy = leader.plan_pick_strategy(task, scene, frames=captures)
                task, route_policy = _apply_pick_strategy(task, pick_strategy, route_policy)
                outcome.task = task
                outcome.pick_strategy = pick_strategy
                print(
                    f"经理事前策略 ({pick_strategy.backend}): "
                    f"{json.dumps(pick_strategy.to_dict(), ensure_ascii=False)}"
                )
                if log is not None:
                    log.set_strategy(pick_strategy, route_policy=route_policy,
                                     transport_mode=task.constraints.transport_mode)
                need_scan = False

            outcome.route_policy = route_policy
            say("物体已锁定，开始执行…")
            success = execute_pick_and_place(
                task, model, robot, target, reference,
                renderer=renderer, obstacle_top_m=ceiling[0], obstacles=obstacles,
                checkpoint_reviewer=checkpoint_reviewer, route_policy=route_policy,
                grasp_reasoner=leader.replan_grasp,
                initial_grasp=initial_grasp, start_from=start_from,
                task_log=log, pick_strategy=pick_strategy,
                plan_brief_reviewer=leader.review_plan_brief,
                transport_candidate_selector=leader.select_transport_candidate,
            )
            if stop_requested():
                outcome.stopped = True
                outcome.reason = "stopped_by_user"
                break
            if success:
                outcome.success = True
                outcome.reason = "verify_place_passed"
                break
            reason = robot.last_abort_reason or "skill_validation_failed"
            diagnosis = getattr(robot, "last_grasp_diagnosis", None)
            adjustment = getattr(robot, "last_checkpoint_adjustment", None)
            if log is not None:
                log.add_failure(reason, None, stage="manipulation", round=rounds,
                                checkpoint_adjustment=adjustment)
            strategy: GraspStrategyUpdate | None = None
            next_route = route_policy
            next_task = task
            if reason.startswith("checkpoint_adjust:") and adjustment in CHECKPOINT_ADJUSTMENTS:
                # A whitelisted word from the LLM becomes one conservative local
                # action.  It costs a recovery round like any other retry.
                robot.last_checkpoint_adjustment = None
                recovery, strategy, next_route, next_task = _map_checkpoint_adjustment(
                    adjustment, task, diagnosis, route_policy, object_in_hand()
                )
            else:
                recovery = leader.decide_recovery({
                    "status": "failed",
                    "stage": "manipulation",
                    "reason": reason,
                    "object": f"{task.object_color} {task.object_name}",
                    "diagnosis": diagnosis,
                    "recovery_round": rounds,
                    "retries_exhausted": rounds >= max_recovery_rounds,
                    "safe_to_retry": not _is_unsafe(reason),
                })
                if recovery.action == "retry_grasp" and isinstance(diagnosis, dict):
                    strategy = leader.replan_grasp(diagnosis)
                    if strategy.action == "safe_stop":
                        recovery = RecoveryDecision(
                            "safe_stop", f"抓取重规划要求停止：{strategy.reason}",
                            strategy.backend,
                        )
            record_round(reason, recovery.action, recovery.reason, recovery.backend, strategy)
            if recovery.action == "safe_stop":
                outcome.reason = (
                    f"recovery_rounds_exhausted:{reason}"
                    if rounds >= max_recovery_rounds else f"safe_stop:{reason}"
                )
                break
            if rounds >= max_recovery_rounds:
                outcome.reason = f"recovery_rounds_exhausted:{reason}"
                break
            rounds += 1
            initial_grasp = None
            start_from = "pick"
            if next_task is not task:
                task = next_task
                outcome.task = task
            if recovery.action == "reobserve_scene":
                need_scan = True
                captures.clear()
            elif recovery.action == "retry_grasp":
                initial_grasp = strategy
                if strategy is not None and strategy.need_reobserve:
                    need_scan = True
                    captures.clear()
            elif recovery.action == "replan_transport":
                start_from = "transport"
                route_policy = (
                    recovery.route_preference
                    if recovery.route_preference is not None
                    else next_route if next_route != route_policy
                    else _next_route_policy(route_policy)
                )
                print(f"  运输重规划：候选评分偏好切换为 {route_policy}")
            else:
                outcome.reason = f"unknown_recovery_action:{recovery.action}"
                break
        outcome.recovery_rounds = rounds
    finally:
        outcome.elapsed_s = time.perf_counter() - started
        if outcome.task is not None and model is not None and data is not None:
            _final_geometry(model, data, outcome.task, outcome.reference, outcome)
        disclosure = getattr(robot, "last_place_disclosure", None)
        if isinstance(disclosure, dict) and disclosure.get("error_vs_locked_reference_m") is not None:
            outcome.xy_error_m = round(float(disclosure["error_vs_locked_reference_m"]), 4)
            outcome.error_vs_used_support_m = round(float(disclosure["error_vs_used_support_m"]), 4)
            outcome.error_vs_locked_reference_m = outcome.xy_error_m
            outcome.place_reference_source = str(disclosure.get("source") or "locked")
            locked_xy = disclosure.get("locked_xy_m")
            if locked_xy is not None:
                outcome.locked_support_xy_m = [round(float(v), 4) for v in locked_xy]
            used_xy = disclosure.get("used_xy_m")
            if used_xy is not None:
                outcome.support_xy_m = [round(float(v), 4) for v in used_xy]
        outcome.obstacles_marked = len(obstacles)
        outcome.scene_ceiling_m = float(ceiling[0])
        # Delivery photo first (success or not), then the log that names it.
        if model is not None and data is not None and renderer is not None:
            try:
                from delivery import render_delivery, save_delivery

                image = render_delivery(model, renderer, data, outcome.task, outcome)
                outcome.delivery_path = str(save_delivery(image, outcome.task))
                print("交差图:", outcome.delivery_path)
            except Exception as error:  # rendering must never mask the task result
                print(f"交差图生成失败: {type(error).__name__}: {error}")
        if log is not None:
            log.set_delivery(outcome=outcome.summary(),
                             route_policy=outcome.route_policy,
                             obstacles_marked=outcome.obstacles_marked,
                             delivery_image=outcome.delivery_path)
            try:
                outcome.task_log_path = str(log.write())
                print("任务日志:", outcome.task_log_path)
            except OSError as error:
                print(f"任务日志写入失败: {error}")
        if model is not None and data is not None and renderer is not None:
            from delivery import deliver

            deliver(outcome.delivery_path, outcome.task_log_path, outcome.summary())
    return outcome
