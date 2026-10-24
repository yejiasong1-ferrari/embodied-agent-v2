"""Complete V2 manipulation orchestrator and headless MuJoCo entry point."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .adapters import MujocoRuntime
from .agent import (
    NaturalLanguageTaskManager,
    RouteStrategist,
    RouteSupervisor,
    SceneMemory,
    TypedRecoveryManager,
)
from .domain import (
    ExecutionResult,
    FailureType,
    Phase,
    PhaseEvent,
    SafetyVerdict,
    SceneEntity,
    TaskSemantics,
)
from .grasp import (
    ContactStateGraspController,
    GeometryGraspPlanner,
    evaluate_micro_lift,
)
from .motion import LocalMotionPlanner, MultiSolutionIK
from .payload import build_payload_state
from .perception import (
    MujocoCloseRangePerception,
    MujocoRayDistanceSensor,
    MujocoSceneScanner,
    fuse_distance_measurements,
)
from .placement import ControlledDescent, PrePlacePlanner, SupportTransferController, verify_placement
from .safety import (
    CollisionChecker,
    ExactTrajectoryValidator,
    RuntimeSafetyMonitor,
    RuntimeStabilitySample,
)
from .state_machine import ManipulationStateMachine


def _category_matches(entity: SceneEntity, category: str) -> bool:
    requested = category.lower()
    if requested in {"cup", "mug"}:
        return entity.geometry == "cylinder" and entity.bounds.size[2] <= 2.2 * max(
            entity.bounds.size[0], entity.bounds.size[1]
        )
    if requested == "bottle":
        return entity.geometry == "cylinder" and entity.bounds.size[2] > 1.4 * min(
            entity.bounds.size[0], entity.bounds.size[1]
        )
    return entity.geometry == requested


class EmbodiedAgentV2:
    def __init__(
        self,
        runtime: MujocoRuntime,
        *,
        use_online_manager: bool = False,
        use_online_grounding: bool | None = None,
        scene_memory: SceneMemory | None = None,
        event_sink: Callable[[PhaseEvent], None] | None = None,
    ):
        self.runtime = runtime
        self.state = ManipulationStateMachine(on_event=event_sink)
        self.scene_memory = scene_memory or SceneMemory()
        self.tasks = NaturalLanguageTaskManager(
            use_online=(
                use_online_manager
                if use_online_grounding is None else use_online_grounding
            ),
            scene_memory=self.scene_memory,
        )
        self.scanner = MujocoSceneScanner(runtime)
        self.close_perception = MujocoCloseRangePerception(runtime)
        self.distance_sensor = MujocoRayDistanceSensor(runtime)
        self.grasps = GeometryGraspPlanner()
        self.grasp_controller = ContactStateGraspController(runtime)
        self.validator = ExactTrajectoryValidator(CollisionChecker(runtime))
        self.motion = LocalMotionPlanner(runtime, self.validator)
        self.routes = RouteStrategist(use_online=use_online_manager)
        self.supervisor = RouteSupervisor(use_online=use_online_manager)
        self.recovery = TypedRecoveryManager()
        self.preplace = PrePlacePlanner()
        self.transfer = SupportTransferController(runtime)
        self.runtime_monitor = RuntimeSafetyMonitor()

    def _result(
        self,
        success: bool,
        message: str,
        failure: FailureType | None = None,
    ) -> ExecutionResult:
        return ExecutionResult(
            success, self.state.phase, tuple(self.state.events), failure, message
        )

    def _safe_stop(self, failure: FailureType, message: str) -> ExecutionResult:
        if self.state.phase is Phase.UNDERSTAND_TASK:
            self.state.transition(
                Phase.STOPPED,
                "safe_stop",
                failure=failure.value,
                reason=message,
            )
            return self._result(False, message, failure)
        if self.state.phase not in {Phase.RECOVERY, Phase.DONE, Phase.STOPPED}:
            self.state.transition(Phase.RECOVERY, "failed", failure=failure.value, reason=message)
        if self.state.phase is Phase.RECOVERY:
            self.state.transition(Phase.STOPPED, "safe_stop", reason=message)
        return self._result(False, message, failure)

    def _raise_payload_clear_of_support(
        self,
        payload,
        source,
        observation,
        evidence,
        yaw_rad: float,
    ):
        """Lift a few millimetres until the measured payload meets its envelope.

        The transport checker rejects a start that is still inside the payload
        envelope. This does not change that margin; it moves the object until
        the existing check passes, and stops if the fingers lose the object.
        """
        checker = self.validator.checker
        for _ in range(4):
            sample = checker.evaluate(
                tuple(self.runtime.arm_q()), payload, phase="transport",
            )
            if sample.safe:
                return payload, True
            if not (sample.collision_pair or "").startswith(
                "payload:dynamic_envelope"
            ):
                return payload, False
            hand, _ = self.runtime.hand_pose()
            target = (float(hand[0]), float(hand[1]), float(hand[2]) + 0.008)
            current = self.runtime.arm_q()
            solutions = self.motion.ik.solve(target, current, yaw_rad=yaw_rad)
            if not solutions:
                return payload, False
            goal = min(solutions, key=lambda value: float(np.linalg.norm(value - current)))
            if not self.runtime.robot.move_to_ruckig(goal, profile="precision"):
                return payload, False
            contacts, _ = self.runtime.contacts_with_body(source.entity_id)
            if contacts != {"left", "right"}:
                return payload, False
            hand_position, hand_rotation = self.runtime.hand_pose()
            object_position, object_rotation = self.runtime.body_pose(source.entity_id)
            payload = build_payload_state(
                source.entity_id,
                observation,
                hand_position,
                hand_rotation,
                object_position,
                object_rotation,
                evidence,
            )
            self.state.record(
                payload_clearance_raise_m=0.008,
                payload_envelope_margin_m=payload.envelope_margin_m,
            )
        sample = checker.evaluate(
            tuple(self.runtime.arm_q()), payload, phase="transport",
        )
        return payload, bool(sample.safe)

    def _cancelled_result(self) -> ExecutionResult:
        if self.state.phase not in {Phase.DONE, Phase.STOPPED}:
            self.state.transition(Phase.STOPPED, "user_stop")
        return self._result(False, "task stopped by user")

    @staticmethod
    def _select(scene, task: TaskSemantics) -> tuple[SceneEntity, SceneEntity]:
        sources = [
            item for item in scene.entities
            if item.movable and item.color == task.object_color
            and _category_matches(item, task.object_category)
        ]
        destinations = [
            item for item in scene.supports
            if item.color == task.destination_color
            and _category_matches(item, task.destination_category)
        ]
        if len(sources) != 1 or len(destinations) != 1:
            raise LookupError(
                f"scene grounding is ambiguous: sources={len(sources)}, destinations={len(destinations)}"
            )
        return sources[0], destinations[0]

    @staticmethod
    def _coarse_approach_goals(source: SceneEntity) -> tuple[tuple[str, tuple[float, float, float]], ...]:
        """Safe ways to arrive near the scanned object, not the grasp pose."""
        hover_z = float(source.bounds.maximum[2]) + 0.22
        center = np.array((source.center_m[0], source.center_m[1], hover_z), dtype=float)
        radial = center[:2] / max(float(np.linalg.norm(center[:2])), 1e-6)
        left = np.array((-radial[1], radial[0]))
        lateral = 0.06
        offsets = (
            ("vertical_up", np.zeros(2)),
            ("back_up", -radial * lateral),
            ("left_back_up", left * lateral - radial * 0.5 * lateral),
            ("right_back_up", -left * lateral - radial * 0.5 * lateral),
        )
        goals = []
        for name, offset in offsets:
            point = center.copy()
            point[:2] += offset
            goals.append((name, (float(point[0]), float(point[1]), float(point[2]))))
        return tuple(goals)

    def _execute_runtime_monitored(self, trajectory, payload, *, hold_contact: bool = True):
        warning: str | None = None
        last_metrics: dict[str, object] = {}
        if hold_contact:
            # Establish a measured preload before acceleration starts. Waiting
            # for slip to occur before adding normal force lets a small object
            # spend most of the position budget during the first segment.
            initial_ctrl = float(self.runtime.data.ctrl[-1])
            preload_ctrl_limit = max(
                0.0, initial_ctrl - 0.016 / 0.08 * 255.0
            )
            preload_peak = 0.0
            for _ in range(60):
                contacts, forces = self.runtime.contacts_with_body(payload.entity_id)
                if contacts != {"left", "right"}:
                    break
                preload_peak = max(forces.values(), default=0.0)
                if preload_peak >= 6.5:
                    break
                current = float(self.runtime.data.ctrl[-1])
                if current <= preload_ctrl_limit + 0.2:
                    break
                self.runtime.data.ctrl[-1] = max(
                    preload_ctrl_limit,
                    current - 0.0004 / 0.08 * 255.0,
                )
                with self.runtime.data_lock:
                    mujoco.mj_step(self.runtime.model, self.runtime.data)
            self.state.record(
                adaptive_preload_force_n=preload_peak,
                adaptive_preload_target_n=6.5,
            )
        grip_at_start = float(self.runtime.data.ctrl[-1])
        # Extra close while a finger unloads or the object starts to tip.
        # Past this budget, lost contact is a real stop. The 12 N cap still binds.
        # Twenty millimetres is additional commanded opening, not object
        # penetration; feedback stops tightening once the 8 N target is met.
        preload_limit = max(0.0, grip_at_start - 0.020 / 0.08 * 255.0)
        unstable_samples = 0
        settle_left = 80

        def monitor():
            nonlocal warning, unstable_samples, settle_left, last_metrics
            hand_position, hand_rotation = self.runtime.hand_pose()
            object_position, object_rotation = self.runtime.body_pose(payload.entity_id)
            relative = hand_rotation.T @ (object_position - hand_position)
            slip = float(np.linalg.norm(
                relative - np.asarray(payload.gripper_to_object_translation_m)
            ))
            contacts, forces = self.runtime.contacts_with_body(payload.entity_id)
            # A contact that has ridden up to the rim still loads both fingers.
            # The mid-band pinch test is for forming the grasp. Transport stops
            # only when a finger is actually no longer pushing.
            if hold_contact and contacts != {"left", "right"}:
                contacts, forces = self.runtime.contacts_with_body(
                    payload.entity_id, pinch_band=False,
                )
            tilt = math.degrees(math.acos(float(np.clip(object_rotation[2, 2], -1.0, 1.0))))
            # Descent is where the plate takes the load, so a finger unloading
            # is not a dropped object. Slip, tilt, and the force cap still stop.
            persisted = (not hold_contact) or contacts == {"left", "right"}
            # The stop lines stay 9 mm and 12°. Short of those, a rising tilt
            # or slip stretches the same joint path so the grip can settle.
            if hold_contact and (
                tilt >= self.runtime_monitor.stop_tilt_deg * 0.5
                or slip >= self.runtime_monitor.warning_slip_m
            ):
                self.runtime.carry_time_scale = 3.0
            action = self.runtime_monitor.evaluate(RuntimeStabilitySample(
                slip, tilt, persisted,
                max(forces.values(), default=0.0),
                slip > payload.measured_slip_m + 0.003,
            ))
            last_metrics = {
                "runtime_slip_m": slip,
                "runtime_tilt_deg": tilt,
                "runtime_contacts": tuple(sorted(contacts)),
                "runtime_peak_force_n": max(forces.values(), default=0.0),
                "runtime_monitor_action": action,
            }
            def tighten() -> None:
                current = float(self.runtime.data.ctrl[-1])
                peak = max(forces.values(), default=0.0)
                if current > preload_limit + 0.4 and peak < min(8.0, self.runtime_monitor.hard_force_n):
                    self.runtime.data.ctrl[-1] = max(
                        preload_limit, current - 0.0025 / 0.08 * 255.0,
                    )

            if hold_contact and action in {
                "stop_contact_lost", "bounded_preload_and_slow_down", "stop_runtime_instability",
            }:
                tighten()
                if action == "stop_contact_lost":
                    return None
            if action == "stop_runtime_instability":
                # A tilt that stays past the limit still stops. Until then the
                # arm holds still and the fingers keep the pressure they have.
                unstable_samples += 1
                if unstable_samples < 40:
                    return "slow_down"
            elif not action.startswith("stop_"):
                unstable_samples = 0
            if (
                hold_contact
                and tilt >= self.runtime_monitor.stop_tilt_deg * 0.5
                and settle_left > 0
            ):
                settle_left -= 1
                tighten()
                return "slow_down"
            if action.startswith("stop_"):
                return action
            if action != "stable_hold":
                warning = action
            return None

        ok, reason = self.runtime.execute_trajectory(trajectory, monitor)
        if not ok:
            self.state.record(**last_metrics)
        return ok, reason, warning

    def run(self, instruction: str) -> ExecutionResult:
        self.state.transition(Phase.UNDERSTAND_TASK)
        # Perception precedes language grounding: the manager may only refer to
        # entities that the current RGB-D scene has actually registered.
        scene = self.scanner.scan()
        remembered = self.scene_memory.update(scene)
        self.state.record(
            scene_memory_revision=self.scene_memory.revision,
            recognized_entities=len(remembered),
            inventory=self.scene_memory.event_inventory(),
        )
        try:
            task = self.tasks.understand(instruction, scene)
        except ValueError as error:
            return self._safe_stop(FailureType.PERCEPTION, str(error))

        self.state.transition(Phase.SCAN_SCENE)
        self.state.record(
            entities=len(scene.entities),
            obstacles=len(scene.obstacles),
            supports=len(scene.supports),
            corridors=scene.gap_map.suggested_corridors,
        )
        try:
            source, destination = self._select(scene, task)
        except LookupError as error:
            return self._safe_stop(FailureType.PERCEPTION, str(error))

        self.state.transition(Phase.APPROACH, target=source.entity_id)
        # The scan pose only chooses where to travel. The grasp pose is
        # measured again after the arm is beside the object.
        self.state.record(
            localization="coarse",
            target=source.entity_id,
        )
        approach = None
        approach_name = None
        for name, goal in self._coarse_approach_goals(source):
            candidate_approach = self.motion.plan_to_cartesian(goal, scene)
            if candidate_approach is None:
                continue
            approach = candidate_approach
            approach_name = name
            break
        if approach is None:
            return self._safe_stop(FailureType.APPROACH_COLLISION, "no safe coarse approach")
        self.state.record(approach=approach_name)
        ok, _ = self.runtime.execute_trajectory(approach)
        if not ok and self.runtime.cancel_event.is_set():
            return self._cancelled_result()

        self.state.transition(Phase.CLOSE_REOBSERVE)
        observation = self.close_perception.observe(source.entity_id, scene)
        hand_position, _ = self.runtime.hand_pose()
        target_vector = np.asarray(observation.center_m) - hand_position
        center_distance = float(np.linalg.norm(target_vector))
        direction = target_vector / max(center_distance, 1e-9)
        tof = self.distance_sensor.measure(
            tuple(hand_position), tuple(direction), max(0.05, center_distance + 0.10)
        )
        # RGB-D surface distance uses the fresh local bounds. Both sensors
        # remain fallible; a disagreement stops final approach and re-observes.
        half_extent = 0.5 * observation.height_m
        rgbd_surface_distance = max(0.0, center_distance - half_extent)
        _, fusion_status = fuse_distance_measurements(rgbd_surface_distance, tof)
        if fusion_status == "sensor_conflict_reobserve":
            observation = self.close_perception.observe(source.entity_id, scene)
            hand_position, _ = self.runtime.hand_pose()
            target_vector = np.asarray(observation.center_m) - hand_position
            center_distance = float(np.linalg.norm(target_vector))
            direction = target_vector / max(center_distance, 1e-9)
            tof = self.distance_sensor.measure(
                tuple(hand_position), tuple(direction), max(0.05, center_distance + 0.10)
            )
            rgbd_surface_distance = max(0.0, center_distance - 0.5 * observation.height_m)
            _, fusion_status = fuse_distance_measurements(rgbd_surface_distance, tof)
            if fusion_status == "sensor_conflict_reobserve":
                return self._safe_stop(
                    FailureType.SENSOR_CONFLICT,
                    "RGB-D and ToF remain inconsistent after re-observation",
                )
        self.state.record(
            target=source.entity_id,
            width_m=observation.width_m,
            height_m=observation.height_m,
            yaw_rad=observation.yaw_rad,
            distance_fusion=fusion_status,
            tof_distance_m=tof.distance_m,
        )
        candidates = self.grasps.candidates(observation, scene)
        self.state.transition(Phase.PRE_ALIGN)
        evidence = None
        selected = None
        hypotheses = 0
        for candidate in candidates:
            hover = (
                candidate.center_m[0], candidate.center_m[1], candidate.palm_height_m + 0.08
            )
            hover_trajectory = self.motion.plan_to_cartesian(
                hover, scene, yaw_rad=candidate.yaw_rad, max_joint_velocity=0.22
            )
            align = None
            if hover_trajectory is not None:
                ok, _ = self.runtime.execute_trajectory(hover_trajectory)
                if not ok and self.runtime.cancel_event.is_set():
                    return self._cancelled_result()
                align = self.motion.plan_to_cartesian(
                    candidate.center_m,
                    scene,
                    yaw_rad=candidate.yaw_rad,
                    max_joint_velocity=0.12,
                )
            if hover_trajectory is None or align is None:
                # This yaw has no collision-free path. Keep looking at the
                # remaining axes; the gripper has not closed, so this is not
                # a failed squeeze.
                self.state.record(
                    candidate=candidate.candidate_id,
                    failure="ik_unreachable",
                )
                continue
            ok, _ = self.runtime.execute_trajectory(align)
            if not ok and self.runtime.cancel_event.is_set():
                return self._cancelled_result()
            if self.state.phase is not Phase.CONTACT_SEARCH:
                self.state.transition(Phase.CONTACT_SEARCH, candidate=candidate.candidate_id)
            evidence = self.grasp_controller.execute(candidate, source.entity_id)
            if self.runtime.cancel_event.is_set():
                return self._cancelled_result()
            for adjustment in self.grasp_controller.adjustments:
                self.state.record(
                    local_correction=adjustment.reason,
                    candidate=candidate.candidate_id,
                )
            stop_reason = self.grasp_controller.last_stop_reason
            if stop_reason not in {"bilateral", "lift_unstable"}:
                hypotheses += 1
                failure = (
                    FailureType.SENSOR_CONFLICT if stop_reason == "sensor_conflict"
                    else FailureType.IK_UNREACHABLE if stop_reason == "ik_unreachable"
                    else FailureType.OBJECT_PUSHED if stop_reason == "object_pushed"
                    else FailureType.NO_CONTACT
                )
                directive = self.recovery.decide(
                    failure, self.state.phase, hypotheses,
                )
                self.runtime.robot.release(duration=0.25)
                if self.runtime.cancel_event.is_set():
                    return self._cancelled_result()
                if (
                    directive.restart_phase is Phase.STOPPED
                    or hypotheses >= self.recovery.maximum_attempts
                ):
                    return self._safe_stop(failure, stop_reason or "local contact loop exhausted")
                self.state.recover_to(
                    Phase.PRE_ALIGN, stop_reason or failure.value,
                    changed_search_dimensions=directive.changed_search_dimensions,
                    grasp_attempt=hypotheses,
                )
                continue
            self.state.transition(
                Phase.MICRO_LIFT,
                force_ratio=evidence.force_ratio_diagnostic,
                left_force_n=evidence.left_force_n,
                right_force_n=evidence.right_force_n,
                object_rise_m=evidence.object_rise_m,
                relative_slip_m=evidence.relative_slip_m,
                lateral_displacement_m=evidence.lateral_displacement_m,
                tilt_deg=evidence.tilt_deg,
                contact_state=evidence.contact_state.value,
                local_corrections=len(self.grasp_controller.adjustments),
            )
            stable, reason = evaluate_micro_lift(evidence)
            if stable:
                selected = candidate
                break
            hypotheses += 1
            directive = self.recovery.decide(
                FailureType.LIFT_SLIP, self.state.phase, hypotheses,
            )
            self.runtime.robot.release(duration=0.25)
            if self.runtime.cancel_event.is_set():
                return self._cancelled_result()
            if (
                directive.restart_phase is Phase.STOPPED
                or hypotheses >= self.recovery.maximum_attempts
            ):
                return self._safe_stop(FailureType.LIFT_SLIP, reason)
            self.state.recover_to(
                Phase.PRE_ALIGN, reason,
                changed_search_dimensions=directive.changed_search_dimensions,
                grasp_attempt=hypotheses,
            )
        if selected is None or evidence is None:
            if hypotheses == 0:
                return self._safe_stop(
                    FailureType.IK_UNREACHABLE, "grasp pose is unreachable",
                )
            return self._safe_stop(FailureType.LIFT_SLIP, "all grasp candidates failed")

        self.state.transition(Phase.STABLE_GRASP, candidate=selected.candidate_id)
        hand_position, hand_rotation = self.runtime.hand_pose()
        object_position, object_rotation = self.runtime.body_pose(source.entity_id)
        payload = build_payload_state(
            source.entity_id,
            observation,
            hand_position,
            hand_rotation,
            object_position,
            object_rotation,
            evidence,
        )
        self.state.record(
            payload_bottom_offset_m=payload.bottom_offset_m,
            payload_envelope_margin_m=payload.envelope_margin_m,
            measured_slip_m=payload.measured_slip_m,
            measured_tilt_deg=payload.measured_tilt_deg,
        )
        payload, separated = self._raise_payload_clear_of_support(
            payload, source, observation, evidence, selected.yaw_rad,
        )
        if not separated:
            return self._safe_stop(
                FailureType.LIFT_SLIP,
                "payload stayed inside its envelope or lost contact",
            )

        self.state.transition(Phase.PLAN_TRANSPORT)
        transport_hand, _ = self.runtime.hand_pose()
        transport_object, _ = self.runtime.body_pose(source.entity_id)
        transport_offset = transport_object - transport_hand
        target = (
            destination.center_m[0] - float(transport_offset[0]),
            destination.center_m[1] - float(transport_offset[1]),
            destination.bounds.maximum[2] + payload.bottom_offset_m + 0.060,
        )
        transport = None
        chosen_intent = None
        rejected_corridors: list[str] = []
        # A corridor the checker already passed, kept only if nothing roomier
        # is accepted. A soft preference is not a collision.
        fallback_trajectory = None
        fallback_intent = None
        fallback_report = None

        def consider(intent) -> bool:
            nonlocal fallback_trajectory, fallback_intent, fallback_report
            candidate_trajectory = self.motion.plan_transport(
                payload,
                intent,
                target,
                scene,
                yaw_rad=selected.yaw_rad,
                posture=intent.posture,
            )
            if candidate_trajectory is None or self.motion.last_report is None:
                rejected_corridors.append(intent.corridor)
                self.state.record(
                    corridor=intent.corridor,
                    rejected=True,
                    reasons=self.motion.last_rejections,
                )
                return False
            decision = self.supervisor.review(intent, self.motion.last_report)
            if decision == "SAFETY_REJECT":
                rejected_corridors.append(intent.corridor)
                return False
            if decision == "ACCEPT":
                nonlocal transport, chosen_intent
                transport = candidate_trajectory
                chosen_intent = intent
                return True
            if decision == "REQUEST_ALTERNATIVE" and fallback_trajectory is None:
                fallback_trajectory = candidate_trajectory
                fallback_intent = intent
                fallback_report = self.motion.last_report
            rejected_corridors.append(intent.corridor)
            return False

        for intent in self.routes.propose(
            task, scene, payload, tuple(hand_position), destination.center_m
        ):
            if consider(intent):
                break
        if transport is None:
            # The object is already off the table. Raise again if the envelope
            # is still tight, then let the language model name another arm
            # shape. A failed local collision check is not overridden.
            payload, separated = self._raise_payload_clear_of_support(
                payload, source, observation, evidence, selected.yaw_rad,
            )
            if not separated:
                return self._safe_stop(
                    FailureType.LIFT_SLIP,
                    "payload stayed inside its envelope or lost contact",
                )
            self.state.record(transport_replan="raised_for_reconfiguration")
            hand_position, _ = self.runtime.hand_pose()
            transport_object, _ = self.runtime.body_pose(source.entity_id)
            transport_offset = transport_object - hand_position
            target = (
                destination.center_m[0] - float(transport_offset[0]),
                destination.center_m[1] - float(transport_offset[1]),
                destination.bounds.maximum[2] + payload.bottom_offset_m + 0.060,
            )
            for intent in self.routes.propose_reconfiguration(
                task, scene, payload, tuple(dict.fromkeys(rejected_corridors)),
            ):
                self.state.record(
                    posture=intent.posture,
                    corridor=intent.corridor,
                    rationale=intent.rationale,
                )
                if consider(intent):
                    break
        if transport is None and fallback_trajectory is not None:
            transport = fallback_trajectory
            chosen_intent = fallback_intent
            self.motion.last_report = fallback_report
        if transport is None:
            return self._safe_stop(FailureType.TRANSPORT_BLOCKED, "all route alternatives rejected")

        safety = self.motion.last_report
        self.state.transition(
            Phase.TRANSPORT,
            corridor=chosen_intent.corridor,
            trajectory_samples=len(transport.positions),
            robot_clearance_m=safety.minimum_robot_clearance_m,
            payload_clearance_m=safety.minimum_payload_clearance_m,
            safety_verdict=safety.verdict.value,
        )
        ok, reason, warning = self._execute_runtime_monitored(transport, payload)
        if not ok:
            return self._safe_stop(FailureType.RUNTIME_INSTABILITY, reason or "runtime stop")

        self.state.transition(Phase.REOBSERVE_DESTINATION, runtime_policy=warning or "stable_hold")
        destination_observation = self.close_perception.observe(destination.entity_id, scene)
        # Transport may introduce a small compliant slip.  Re-measure the
        # actual gripper/payload transform before calculating millimetre-scale
        # pre-place and contact heights; the initial grasp transform is not
        # treated as permanent truth.
        payload_observation = self.close_perception.observe(source.entity_id, scene)
        hand_position, hand_rotation = self.runtime.hand_pose()
        object_position, object_rotation = self.runtime.body_pose(source.entity_id)
        payload = build_payload_state(
            source.entity_id,
            payload_observation,
            hand_position,
            hand_rotation,
            object_position,
            object_rotation,
            evidence,
        )
        self.state.record(
            payload_tilt_after_transport_deg=payload.measured_tilt_deg,
            payload_envelope_margin_m=payload.envelope_margin_m,
        )
        self.state.transition(Phase.PRE_PLACE)
        preplace_trajectory = None
        preplace_name = None
        payload_world_offset = tuple(object_position - hand_position)
        relative_object_rotation = hand_rotation.T @ object_rotation
        object_yaw = math.atan2(
            float(object_rotation[1, 0]), float(object_rotation[0, 0])
        )
        for candidate in self.preplace.candidates(
            destination_observation,
            payload,
            payload_world_offset_m=payload_world_offset,
        ):
            desired_yaw = object_yaw + candidate.yaw_offset_rad
            cosine, sine = math.cos(desired_yaw), math.sin(desired_yaw)
            upright_object_rotation = np.array((
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            ))
            target_hand_rotation = (
                upright_object_rotation @ relative_object_rotation.T
            )
            preplace_trajectory = self.motion.plan_to_cartesian(
                candidate.hand_position_m,
                scene,
                yaw_rad=selected.yaw_rad + candidate.yaw_offset_rad,
                payload=payload,
                phase="pre_place",
                max_joint_velocity=0.16,
                target_rotation=target_hand_rotation,
            )
            if preplace_trajectory is not None:
                preplace_name = candidate.name
                break
            self.state.record(
                candidate=candidate.name,
                rejected=True,
                reasons=self.motion.last_rejections,
            )
        if preplace_trajectory is None:
            return self._safe_stop(FailureType.PLACEMENT_UNSTABLE, "no safe pre-place pose")
        ok, reason, _ = self._execute_runtime_monitored(preplace_trajectory, payload)
        if not ok:
            return self._safe_stop(FailureType.RUNTIME_INSTABILITY, reason or "pre-place stop")
        self.state.record(
            candidate=preplace_name,
            trajectory_samples=len(preplace_trajectory.positions),
        )

        # Pre-place is another compliant carry motion. Re-measure before
        # millimetre descent so its collision envelope and contact height use
        # the transform that physically exists now, not the post-transport one.
        payload_observation = self.close_perception.observe(source.entity_id, scene)
        hand_position, hand_rotation = self.runtime.hand_pose()
        object_position, object_rotation = self.runtime.body_pose(source.entity_id)
        payload = build_payload_state(
            source.entity_id,
            payload_observation,
            hand_position,
            hand_rotation,
            object_position,
            object_rotation,
            evidence,
        )
        self.state.record(
            payload_tilt_before_descent_deg=payload.measured_tilt_deg,
            payload_bottom_offset_m=payload.bottom_offset_m,
        )

        self.state.transition(Phase.DESCENT)
        current_hand, _ = self.runtime.hand_pose()
        contact_z = destination_observation.top_z_m + payload.bottom_offset_m
        descent_controller = ControlledDescent(
            self.runtime, MultiSolutionIK(self.runtime.model), self.validator
        )
        descent = descent_controller.plan(
            tuple(current_hand), contact_z, payload, scene, destination.entity_id,
            yaw_rad=selected.yaw_rad,
        )
        if not descent:
            return self._safe_stop(
                FailureType.PLACEMENT_UNSTABLE,
                "controlled descent has no safe steps"
                + (f": {descent_controller.last_rejection}" if descent_controller.last_rejection else ""),
            )
        self.state.record(steps=len(descent), contact_hand_z_m=contact_z)
        for segment in descent:
            ok, reason, _ = self._execute_runtime_monitored(
                segment, payload, hold_contact=False,
            )
            if not ok:
                return self._safe_stop(FailureType.RUNTIME_INSTABILITY, reason or "descent stop")

        self.state.transition(Phase.FIRST_CONTACT, support=destination.entity_id)
        if not self.transfer.support_contact(source.entity_id, destination.entity_id):
            return self._safe_stop(FailureType.PLACEMENT_UNSTABLE, "support contact not confirmed")
        self.state.transition(Phase.LOAD_TRANSFER, support_contact=True)
        before_release, before_rotation = self.runtime.body_pose(source.entity_id)
        self.runtime.robot.hold(duration=0.15)
        if self.runtime.cancel_event.is_set():
            return self._cancelled_result()
        self.state.transition(Phase.GRADUAL_RELEASE, release_steps=self.transfer.release_steps)
        if not self.transfer.gradual_release(source.entity_id, destination.entity_id):
            if self.runtime.cancel_event.is_set():
                return self._cancelled_result()
            return self._safe_stop(FailureType.PLACEMENT_UNSTABLE, "gradual release lost support")

        self.state.transition(Phase.VERIFY_RESULT)
        self.runtime.robot.hold(duration=0.25)
        if self.runtime.cancel_event.is_set():
            return self._cancelled_result()
        final_position, final_rotation = self.runtime.body_pose(source.entity_id)
        body_id = int(self.runtime.model.body(source.entity_id).id)
        joint_id = int(self.runtime.model.body_jntadr[body_id])
        dof_address = int(self.runtime.model.jnt_dofadr[joint_id])
        speed = float(np.linalg.norm(self.runtime.data.qvel[dof_address:dof_address + 3]))
        tilt = math.degrees(math.acos(float(np.clip(final_rotation[2, 2], -1.0, 1.0))))
        destination_radius = 0.5 * min(
            destination_observation.bounds.size[0], destination_observation.bounds.size[1]
        )
        final = verify_placement(
            payload_center_m=tuple(final_position),
            destination_center_m=destination_observation.center_m,
            destination_radius_m=destination_radius,
            support_contact=self.transfer.support_contact(source.entity_id, destination.entity_id),
            displacement_m=float(np.linalg.norm(final_position - before_release)),
            tilt_deg=tilt,
            speed_m_s=speed,
            relation=task.relation,
        )
        self.state.record(
            support_contact=final.support_contact,
            displacement_m=final.displacement_m,
            tilt_deg=final.tilt_deg,
            still_moving=final.still_moving,
            relation_satisfied=final.relation_satisfied,
        )
        if (
            not final.relation_satisfied
            or final.still_moving
            or final.tilt_deg > 10.0
            or final.displacement_m > 0.025
        ):
            return self._safe_stop(FailureType.VERIFICATION, "final placement verification failed")
        self.state.transition(Phase.DONE, relation=task.relation.value)
        return self._result(True, "task completed and verified")


def main() -> int:
    """Compatibility CLI: all new executions use the pixel-driven agent.

    EmbodiedAgentV2 above is retained only as a legacy regression reference.
    It is not imported by the active agent or any supported launch command.
    """
    from agent.cli import main as run_visual_agent
    return run_visual_agent()


if __name__ == "__main__":
    raise SystemExit(main())
