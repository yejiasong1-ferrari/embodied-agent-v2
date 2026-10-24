"""Fast regression tests for the embodied manipulation safety layers."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import ruckig

from camera_demo import (
    GRASP_CENTERING_MAX_STEP_M,
    GRASP_CENTERING_TOTAL_BUDGET_M,
    GRIPPER_INNER_PAD_OFFSET_M,
    GRASP_SERVO_MAX_STEP_M,
    GRASP_SERVO_OBJECT_MOTION_LIMIT_M,
    GRASP_SERVO_TARGET_M,
    GRASP_SEATING_EXTRA_M,
    GRASP_VERIFY_LIFT_M,
    GRASP_VERIFY_MAX_COMMAND_M,
    GRASP_YAW_ADJUST_STEP_RAD,
    GRASP_YAW_ADJUST_TOTAL_RAD,
    SUPPORT_ASSOCIATION_GATE_M,
    reference_point_within_color_gate,
    _CROSS_BASE_SEED_EXTRA,
    _TRANSPORT_HEIGHT_EXTRAS,
    _TRANSPORT_SEED_CACHE,
    _alignment_components,
    _estimate_centering_correction_m,
    _inline_centering_ready,
    _segment_clears_base,
    contact_axis_offset,
    classify_grasp_contact,
    grasp_stability_decision,
    grasp_axis_distance_rad,
    grasp_trial_release_decision,
    placement_release_stability_decision,
    classify_transport_goal,
    dense_chord_death,
    execution_sample_released,
    route_reuse_rejection,
    segmented_raise_recovery_allowed,
    splice_reused_waypoints,
    transport_route_identity,
    transport_corridor_waypoints,
    upsert_transport_routes,
    reuse_sample_death,
    grasp_close_step_budget,
    arbitrate_support_refresh,
    decide_support_snap,
    inline_translation_should_brake,
    inline_yaw_adjust_rad,
    next_transport_stability_action,
    next_grasp_squeeze_m,
    recovery_centering_measurement,
    solve_adaptive_centering_step,
    stabilize_handle_direction,
    TransportPlanCandidate,
    choose_transport_candidate,
    transport_candidate_score_key,
    transport_pose_within_limit,
    order_transport_plans,
    place_error_pair,
    preload_force_decision,
    pre_ompl_may_plan,
    pre_ompl_separation_mode,
    pre_ompl_separation_status,
    pre_ompl_step_allowed,
    remember_validated_transport_seed,
)
from joint_control import MOTION_LIMITS, retime_duration, scaled_motion_limits, smootherstep
from embodied_agent import (
    LLMRobotLeader, LeaderUnavailableError, SceneObject, SceneSnapshot,
    _json_default, clip_grasp_strategy, offline_grasp_replan, parse_grasp_replan,
    parse_image_pointing,
)
from motion_planning import plan_rrt_connect
from object_registry import OBJECTS, record_for_color
from vision_detector import (
    _fragmented_support_center, geometric_section_center_xy,
    support_centroid_from_points,
)
from robot_skills import DEFAULT_SKILL_LIBRARY, SkillPlanError
from task_log import SECTIONS, TaskLog, task_short_code
from task_planner import (
    DEFAULT_STEPS,
    CategoryMismatch,
    NotFoundReport,
    TaskConstraints,
    TaskSpec,
    TaskValidationError,
    apply_manager_rereview,
    plan_task,
    validate_task,
)


class GraspAlignmentTests(unittest.TestCase):
    def test_handle_direction_jump_uses_recent_three_median(self) -> None:
        history = [float(np.deg2rad(5.0)), float(np.deg2rad(7.0))]
        stabilized, applied = stabilize_handle_direction(
            history, float(np.deg2rad(35.0))
        )
        self.assertTrue(applied)
        self.assertAlmostEqual(float(np.rad2deg(stabilized)), 7.0)
        self.assertEqual(len(history), 3)

    def test_recovery_prefers_recent_contact_delta_over_force_estimator(self) -> None:
        side, magnitude, source = recovery_centering_measurement(
            {
                "delta_closing_axis_m": -0.0015,
                "delta_direction": "toward_right_finger",
                "lower_bound": False,
            },
            0.063,
            0.064,
            {"left": 9.0, "right": 0.0},
            0.9,
        )
        self.assertEqual((side, source), ("right", "contact_geometry_delta"))
        self.assertAlmostEqual(magnitude, 0.0015)

    def test_single_side_at_target_force_is_never_balanced(self) -> None:
        state = classify_grasp_contact(
            {"left"}, {"left": 2.0, "right": 0.0}, 2.0, 2.8
        )
        self.assertEqual(state, "left_only")

    def test_real_bilateral_contact_may_attempt_the_micro_lift(self) -> None:
        self.assertEqual(
            grasp_trial_release_decision(
                {"left", "right"},
                {"left": 0.57, "right": 0.51},
                0.90,
                2.00,
                shift_m=0.0002,
                tilt_deg=0.2,
            ),
            "ready_for_trial_lift",
        )
        self.assertEqual(
            grasp_trial_release_decision(
                {"left", "right"},
                {"left": 0.50, "right": 0.47},
                0.90,
                2.00,
            ),
            "ready_for_trial_lift",
        )
        self.assertEqual(
            grasp_trial_release_decision(
                {"left", "right"},
                {"left": 0.02, "right": 0.03},
                0.90,
                2.00,
            ),
            "continue_closing",
        )
        self.assertEqual(
            grasp_trial_release_decision(
                {"left", "right"},
                {"left": 0.28, "right": 0.95},
                0.90,
                2.00,
                shift_m=0.0017,
                closure_disturbed=True,
            ),
            "closure_disturbed",
        )
        self.assertEqual(
            grasp_trial_release_decision(
                {"right"},
                {"left": 0.0, "right": 0.80},
                0.90,
                2.00,
            ),
            "bilateral_contact_required",
        )

    def test_force_ratio_is_diagnostic_at_the_trial_lift_gate(self) -> None:
        # Ratio 1.80 / 0.75 > 2.2, but the weak side is above the absolute
        # trial-load floor and the strong side is below its hard cap.
        self.assertEqual(
            grasp_trial_release_decision(
                {"left", "right"},
                {"left": 0.75, "right": 1.80},
                0.90,
                2.00,
            ),
            "ready_for_trial_lift",
        )
        self.assertEqual(
            grasp_trial_release_decision(
                {"left", "right"},
                {"left": 0.40, "right": 2.00},
                0.90,
                2.00,
            ),
            "strong_force_limit",
        )

    def test_micro_lift_is_the_result_based_grasp_gate(self) -> None:
        self.assertAlmostEqual(GRASP_VERIFY_LIFT_M, 0.005)
        self.assertAlmostEqual(GRASP_VERIFY_MAX_COMMAND_M, 0.030)
        self.assertEqual(
            grasp_stability_decision(
                contacts={"left", "right"},
                forces={"left": 0.70, "right": 1.80},
                target_force_n=0.90,
                maximum_force_n=2.00,
                object_rise_m=0.004,
                relative_slip_m=0.002,
                lateral_shift_m=0.003,
                tilt_deg=2.0,
            ),
            "stable",
        )
        self.assertEqual(
            grasp_stability_decision(
                contacts={"left", "right"},
                forces={"left": 0.70, "right": 1.80},
                target_force_n=0.90,
                maximum_force_n=2.00,
                object_rise_m=0.004,
                relative_slip_m=0.002,
                lateral_shift_m=0.011,
                tilt_deg=2.0,
            ),
            "large_displacement",
        )

    def test_post_lift_force_drop_is_diagnostic_when_object_follows(self) -> None:
        self.assertEqual(
            grasp_stability_decision(
                contacts={"left", "right"},
                forces={"left": 0.42, "right": 0.80},
                target_force_n=0.90,
                maximum_force_n=2.00,
                object_rise_m=0.004,
                relative_slip_m=0.002,
                lateral_shift_m=0.003,
                tilt_deg=2.0,
            ),
            "stable",
        )

    def test_gradual_unload_uses_displacement_from_supported_pose(self) -> None:
        self.assertEqual(
            placement_release_stability_decision(
                displacement_m=0.009,
                tilt_deg=3.0,
                support_contacts=2,
                object_bottom_z_m=0.037,
                support_z_m=0.036,
            ),
            "stable",
        )
        self.assertEqual(
            placement_release_stability_decision(
                displacement_m=0.011,
                tilt_deg=3.0,
                support_contacts=2,
                object_bottom_z_m=0.037,
                support_z_m=0.036,
            ),
            "large_displacement",
        )
        self.assertEqual(
            placement_release_stability_decision(
                displacement_m=0.001,
                tilt_deg=3.0,
                support_contacts=0,
                object_bottom_z_m=0.037,
                support_z_m=0.036,
            ),
            "support_contact_lost",
        )

    def test_seating_budget_exhaustion_keeps_unbalanced_failure(self) -> None:
        next_gap, used = next_grasp_squeeze_m(
            0.060, 0.063, "left_only", True
        )
        self.assertIsNone(next_gap)
        self.assertAlmostEqual(used, GRASP_SEATING_EXTRA_M)
        self.assertEqual(GRASP_SEATING_EXTRA_M, 0.003)

    def test_close_step_budget_can_reach_full_seating_floor(self) -> None:
        steps = grasp_close_step_budget(0.074, 0.063)
        required_quarter_mm_steps = int(np.ceil((0.074 - 0.060) / 0.00025))
        self.assertGreaterEqual(steps, required_quarter_mm_steps)
        self.assertEqual(GRASP_SEATING_EXTRA_M, 0.003)

    def test_contact_axis_offset_from_two_first_contact_apertures(self) -> None:
        left_first = contact_axis_offset(
            {"left": 0.066, "right": 0.063}, 0.064
        )
        self.assertAlmostEqual(left_first["delta_closing_axis_m"], 0.0015)
        self.assertEqual(left_first["delta_direction"], "toward_left_finger")
        self.assertFalse(left_first["lower_bound"])
        right_first = contact_axis_offset(
            {"left": 0.062, "right": 0.064}, 0.060
        )
        self.assertAlmostEqual(right_first["delta_closing_axis_m"], -0.001)
        self.assertEqual(right_first["delta_direction"], "toward_right_finger")

    def test_single_contact_axis_offset_is_a_lower_bound(self) -> None:
        inferred = contact_axis_offset(
            {"left": 0.068, "right": None}, 0.064
        )
        self.assertAlmostEqual(inferred["delta_closing_axis_m"], 0.002)
        self.assertEqual(inferred["delta_direction"], "toward_left_finger")
        self.assertTrue(inferred["lower_bound"])

    def test_contact_geometry_uses_existing_inner_pad_calibration(self) -> None:
        self.assertEqual(GRIPPER_INNER_PAD_OFFSET_M, 0.006)

    def test_servo_gate_is_stricter_without_relaxing_object_motion_limit(self) -> None:
        self.assertEqual(GRASP_SERVO_TARGET_M, 0.001)
        self.assertLess(GRASP_SERVO_TARGET_M, 0.003)
        self.assertEqual(GRASP_SERVO_OBJECT_MOTION_LIMIT_M, 0.003)
        self.assertLessEqual(GRASP_SERVO_MAX_STEP_M, 0.006)

    def test_alignment_diagnostics_resolve_closing_and_cross_axis(self) -> None:
        components = _alignment_components(
            np.array((0.002, -0.003)), np.array((0.0, -1.0))
        )
        self.assertAlmostEqual(components["closing_axis_m"], 0.003)
        self.assertAlmostEqual(components["across_axis_m"], 0.002)

    def test_contact_centering_estimate_uses_geometry_force_and_stays_bounded(self) -> None:
        estimate = _estimate_centering_correction_m(
            0.071, 0.060, {"left": 1.2, "right": 0.0}, 2.0
        )
        self.assertGreater(estimate, 0.001)
        self.assertLessEqual(estimate, GRASP_CENTERING_MAX_STEP_M)
        extreme = _estimate_centering_correction_m(
            0.080, 0.025, {"left": 99.0, "right": 0.0}, 0.9
        )
        self.assertEqual(extreme, GRASP_CENTERING_MAX_STEP_M)
        self.assertEqual(GRASP_CENTERING_TOTAL_BUDGET_M, 0.005)

    def test_material_single_side_contact_pauses_for_inline_centering(self) -> None:
        self.assertFalse(_inline_centering_ready(
            "left_only", {"left": 0.71, "right": 0.0}, 2.0
        ))
        self.assertTrue(_inline_centering_ready(
            "left_only", {"left": 1.11, "right": 0.0}, 2.0
        ))
        self.assertFalse(_inline_centering_ready(
            "bilateral_loading", {"left": 0.90, "right": 0.80}, 2.0
        ))

    def test_secant_step_targets_zero_delta(self) -> None:
        solved = solve_adaptive_centering_step([
            (0.0, 0.002875),
            (0.002875, -0.001750),
        ])
        self.assertEqual(solved["method"], "secant_zero")
        step = float(solved["step_signed_m"])
        self.assertLess(step, 0.0)
        self.assertAlmostEqual(step, -0.0010878378, places=6)
        self.assertGreaterEqual(abs(step), 0.00025)
        self.assertLessEqual(abs(step), GRASP_CENTERING_MAX_STEP_M)
        self.assertAlmostEqual(float(solved["expected_residual_m"]), 0.0, places=6)

    def test_single_sample_uses_half_delta_inside_the_step_clamp(self) -> None:
        damped = solve_adaptive_centering_step([(0.0, 0.002875)])
        self.assertEqual(damped["method"], "damped_delta")
        self.assertEqual(damped["gain"], 0.5)
        self.assertAlmostEqual(float(damped["step_signed_m"]), 0.0014375)
        huge = solve_adaptive_centering_step([(0.0, 0.020)])
        self.assertAlmostEqual(float(huge["step_signed_m"]), GRASP_CENTERING_MAX_STEP_M)
        tiny = solve_adaptive_centering_step([(0.0, 0.0001)])
        self.assertAlmostEqual(float(tiny["step_signed_m"]), 0.00025)

    def test_no_centering_sample_keeps_the_force_estimator(self) -> None:
        solved = solve_adaptive_centering_step([])
        self.assertEqual(solved["method"], "force_geometry_fallback")
        self.assertIsNone(solved["step_signed_m"])
        forces = {"left": 1.2, "right": 0.0}
        side, magnitude, source = recovery_centering_measurement(
            {
                "delta_closing_axis_m": None,
                "delta_direction": "unknown_no_contact",
            },
            0.071,
            0.060,
            forces,
            2.0,
        )
        self.assertEqual(source, "force_geometry_fallback")
        self.assertEqual(side, "left")
        self.assertAlmostEqual(
            magnitude,
            _estimate_centering_correction_m(0.071, 0.060, forces, 2.0),
        )

    def test_trimmed_section_center_drops_a_handle_and_keeps_a_disk(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
        disk = np.column_stack((0.04 * np.cos(angles), 0.04 * np.sin(angles)))
        handle = np.column_stack((
            np.full(24, 0.062), np.linspace(-0.012, 0.012, 24),
        ))
        biased = geometric_section_center_xy(np.vstack((disk, handle)))
        self.assertLess(abs(float(biased[0])), 0.002)
        self.assertLess(abs(float(biased[1])), 0.002)
        plain = geometric_section_center_xy(disk + np.array((0.10, -0.20)))
        self.assertLess(abs(float(plain[0]) - 0.10), 0.002)
        self.assertLess(abs(float(plain[1]) + 0.20), 0.002)


class ObjectRegistryTests(unittest.TestCase):
    def test_reference_accepts_its_registered_object_id_as_an_alias(self) -> None:
        task = validate_task(TaskSpec(
            action="place", object_name="cup", object_color="white",
            reference="yellow_plate", relation="on", grasp_strategy="side_grasp",
            approach_direction="vertical", reason="online payload",
            reference_color="yellow",
        ))
        self.assertEqual(task.reference, "plate")

    def test_original_seven_objects_keep_exact_execution_numbers(self) -> None:
        expected = {
            "orange": ("cube", 0.025, 0.082, 0.050, 0.015, 1.80, 2.20),
            "red": ("red_cup", 0.050, 0.105, 0.070, 0.020, 0.90, 2.00),
            "blue": ("blue_box", 0.040, 0.103, 0.060, 0.018, 2.00, 2.80),
            "green": ("green_block", 0.040, 0.103, 0.060, 0.018, 2.00, 2.80),
            "cyan": ("cyan_mug", 0.040, 0.105, 0.070, 0.020, 0.90, 2.00),
            "purple": ("purple_bottle", 0.060, 0.110, 0.052, 0.022, 0.90, 2.00),
            "yellow": ("plate", 0.018, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
        for color, numbers in expected.items():
            record = record_for_color(color)
            self.assertIsNotNone(record)
            assert record is not None
            actual = (
                record.body_name, record.half_height_m, record.hand_offset_m,
                record.fallback_width_m, record.grasp_depth_m,
                record.target_force_n, record.max_force_n,
            )
            self.assertEqual(actual, numbers)

    def test_unknown_target_uses_low_confidence_generic_cylinder(self) -> None:
        checked = validate_task(TaskSpec(
            action="pick", object_name="canister", object_color="pink",
            reference=None, relation=None, grasp_strategy="top_grasp",
            approach_direction="horizontal", reason="table-external object",
        ))
        self.assertEqual(checked.object_confidence, "low")
        self.assertTrue(str(checked.object_id).startswith("generic_pink_canister"))
        self.assertEqual(checked.grasp_strategy, "side_grasp")
        self.assertEqual(checked.approach_direction, "vertical")
        self.assertEqual(checked.to_dict()["object_confidence"], "low")

    def test_white_mug_is_added_entirely_through_registry_semantics(self) -> None:
        record = record_for_color("white")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.body_name, "white_mug")
        self.assertEqual(record.shape_prior, "handled_cylinder")
        task, _ = plan_task("拿起白色马克杯", use_llm=False)
        self.assertEqual((task.object_id, task.object_name), ("white_mug", "cup"))
        self.assertEqual(task.object_confidence, "high")


class SupportRefreshTests(unittest.TestCase):
    def test_occluded_plate_fragments_recover_center_without_loosening_gate(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
        rim = np.column_stack((
            0.34 + 0.12 * np.cos(angles),
            0.30 + 0.12 * np.sin(angles),
            np.full_like(angles, 0.036),
        ))
        fragments = [rim[5:20], rim[25:40], rim[45:60], rim[65:80]]
        center = _fragmented_support_center(
            fragments, np.array((0.338, 0.294)), 0.24
        )
        self.assertIsNotNone(center)
        assert center is not None
        self.assertLess(float(np.linalg.norm(center[:2] - (0.34, 0.30))), 0.01)
        self.assertIsNone(_fragmented_support_center(
            fragments, np.array((0.20, 0.30)), 0.24
        ))


class TaskPlannerTests(unittest.TestCase):
    def test_fragile_language_becomes_a_constraint_not_joint_commands(self) -> None:
        task, backend = plan_task(
            "把绿色方块放到黄色盘子里，绕开高台，搬运时不要让方块滑落",
            use_llm=False,
        )
        self.assertIn("offline", backend)
        self.assertEqual(task.constraints.transport_mode, "fragile")
        self.assertTrue(task.constraints.avoid_collisions)
        self.assertTrue(task.constraints.verify_grasp)
        self.assertEqual(task.steps, DEFAULT_STEPS)
        serialized = task.to_dict()
        self.assertNotIn("joint_angles", serialized)
        self.assertNotIn("trajectory", serialized)

    def test_llm_cannot_disable_hard_safety_checks(self) -> None:
        unsafe = TaskSpec(
            action="place",
            object_name="box",
            object_color="green",
            reference="plate",
            relation="inside",
            grasp_strategy="top_grasp",
            approach_direction="vertical",
            reason="test",
            reference_color="yellow",
            constraints=TaskConstraints(avoid_collisions=False),
        )
        with self.assertRaises(TaskValidationError):
            validate_task(unsafe)

    def test_stable_transport_language_enables_fragile_profile(self) -> None:
        task, _ = plan_task(
            "把绿色方块拿到黄色盘子上，运输时保持稳定",
            use_llm=False,
        )
        self.assertEqual(task.constraints.transport_mode, "fragile")

    def test_cup_uses_the_implemented_side_grasp(self) -> None:
        task, _ = plan_task("把红色杯子拿到黄色盘子上", use_llm=False)
        self.assertEqual(task.object_name, "cup")
        self.assertEqual(task.grasp_strategy, "side_grasp")
        self.assertEqual(task.approach_direction, "vertical")

    def test_cyan_mug_and_purple_bottle_are_in_the_scene(self) -> None:
        mug, _ = plan_task("把青色杯子拿到黄色盘子上", use_llm=False)
        self.assertEqual(mug.object_color, "cyan")
        self.assertEqual(mug.object_name, "cup")
        bottle, _ = plan_task("把紫色瓶子拿到黄色盘子上", use_llm=False)
        self.assertEqual(bottle.object_color, "purple")
        self.assertEqual(bottle.object_name, "bottle")
        self.assertEqual(bottle.grasp_strategy, "side_grasp")

    def test_destination_forces_complete_pick_and_place_semantics(self) -> None:
        inconsistent = TaskSpec(
            action="pick",
            object_name="cube",
            object_color="orange",
            reference="plate",
            relation="on",
            grasp_strategy="top_grasp",
            approach_direction="vertical",
            reason="LLM returned inconsistent action",
            reference_color="yellow",
            steps=DEFAULT_STEPS[:5],
        )
        checked = validate_task(inconsistent)
        self.assertEqual(checked.action, "place")
        self.assertEqual(checked.steps, DEFAULT_STEPS)


class TrajectoryRetimingTests(unittest.TestCase):
    def test_friction_margin_scale_only_tightens_payload_limits(self) -> None:
        fragile = scaled_motion_limits("payload", 0.63)
        normal = scaled_motion_limits("payload", 1.0)
        self.assertAlmostEqual(fragile.max_velocity, 0.1764)
        self.assertAlmostEqual(fragile.max_acceleration, 0.3465)
        self.assertLess(fragile.max_velocity, normal.max_velocity)
        self.assertLess(fragile.max_acceleration, normal.max_acceleration)
        with self.assertRaises(ValueError):
            scaled_motion_limits("payload", 1.01)

    def test_payload_retiming_obeys_quintic_limits(self) -> None:
        limits = MOTION_LIMITS["payload"]
        delta = np.array((0.0, 0.4, -0.1, 0.2, 0.0, 0.0, 0.0))
        duration = retime_duration(delta, 0.01, limits)
        span = float(np.max(np.abs(delta)))
        self.assertLessEqual(1.875 * span / duration, limits.max_velocity + 1e-9)
        self.assertLessEqual(5.774 * span / duration**2, limits.max_acceleration + 1e-9)

    def test_quintic_profile_has_fixed_endpoints(self) -> None:
        self.assertEqual(smootherstep(0.0), 0.0)
        self.assertEqual(smootherstep(1.0), 1.0)
        self.assertGreater(smootherstep(0.5), 0.49)
        self.assertLess(smootherstep(0.5), 0.51)

    def test_ruckig_payload_trajectory_obeys_jerk_limits(self) -> None:
        limits = MOTION_LIMITS["payload"]
        inputs = ruckig.InputParameter(7)
        trajectory = ruckig.Trajectory(7)
        inputs.current_position = [0.0] * 7
        inputs.current_velocity = [0.0] * 7
        inputs.current_acceleration = [0.0] * 7
        inputs.target_position = [0.0, 0.25, -0.10, 0.0, 0.0, 0.0, 0.0]
        inputs.target_velocity = [0.0] * 7
        inputs.target_acceleration = [0.0] * 7
        inputs.max_velocity = [limits.max_velocity] * 7
        inputs.max_acceleration = [limits.max_acceleration] * 7
        inputs.max_jerk = [limits.max_jerk] * 7
        result = ruckig.Ruckig(7).calculate(inputs, trajectory)
        self.assertIn(result, {ruckig.Result.Working, ruckig.Result.Finished})
        self.assertGreater(trajectory.duration, 0.0)


class MotionPlannerTests(unittest.TestCase):
    @staticmethod
    def _transport_candidate(
        candidate_id: str,
        *,
        duration: float,
        pose: float,
        validated: bool = True,
    ) -> TransportPlanCandidate:
        return TransportPlanCandidate(
            candidate_id=candidate_id,
            joint_waypoints=(), execution_points=(), hand_waypoints=(),
            scores={
                "estimated_duration_s": duration,
                "maximum_pose_deviation_deg": pose,
                "maximum_tangential_acceleration_m_s2": 0.1,
                "minimum_clearance_m": 0.02,
                "maximum_base_turn_rad": 0.5,
                "maximum_wrist_swing_rad": 0.4,
            },
            planning_time_s=0.1,
            validated=validated,
        )

    def test_transport_candidate_preference_changes_selection(self) -> None:
        stable = self._transport_candidate("stable", duration=8.0, pose=1.0)
        fast = self._transport_candidate("fast", duration=4.0, pose=2.0)
        self.assertEqual(
            choose_transport_candidate([stable, fast], "stability_first").candidate_id,
            "stable",
        )
        self.assertEqual(
            choose_transport_candidate([stable, fast], "time_first").candidate_id,
            "fast",
        )
        self.assertLess(
            transport_candidate_score_key(stable, "stability_first"),
            transport_candidate_score_key(fast, "stability_first"),
        )

    def test_unvalidated_manager_candidate_is_never_executed(self) -> None:
        safe = self._transport_candidate("safe", duration=8.0, pose=1.0)
        unsafe = self._transport_candidate(
            "unsafe", duration=2.0, pose=0.1, validated=False,
        )
        chosen = choose_transport_candidate(
            [safe, unsafe], "time_first", requested_id="unsafe",
        )
        self.assertEqual(chosen.candidate_id, "safe")

    def test_no_validated_candidate_stops_without_executing_a_trajectory(self) -> None:
        from camera_demo import carried_ompl_stop_record, ompl_carried_search_plan

        rejected = [{"goal_index": 0, "reason": "execution_trajectory_rejected"}]
        record = carried_ompl_stop_record(
            rejected,
            preference="stability_first",
            planning_time_s=1.2,
            planning_budget_s=3.2,
            budget_doubled=True,
            ik_goal_order="reversed",
        )
        self.assertEqual(record["planner_used"], "ompl")
        self.assertIsNone(record["selected_candidate_id"])
        self.assertFalse(record["executed"])
        self.assertIsNone(record["arc_fallback_reason"])
        self.assertEqual(record["rejected_candidates"], rejected)
        self.assertNotIn("arc", str(record["planner_used"]))

        first = ompl_carried_search_plan(0, "stability_first")
        second = ompl_carried_search_plan(1, "stability_first")
        third = ompl_carried_search_plan(2, "stability_first")
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first["timeout_s"], 1.6)
        self.assertEqual(first["preference"], "stability_first")
        self.assertFalse(first["budget_doubled"])
        self.assertEqual(second["timeout_s"], 3.2)
        self.assertEqual(second["preference"], "time_first")
        self.assertTrue(second["reverse_ik_goals"])
        self.assertTrue(second["budget_doubled"])
        self.assertIsNone(third)
        flipped_back = ompl_carried_search_plan(1, "time_first")
        self.assertEqual(flipped_back["preference"], "stability_first")
        self.assertEqual(flipped_back["timeout_s"], 3.2)


class PreOmplSeparationTests(unittest.TestCase):
    def _allowed(self, **overrides: object) -> bool:
        sample = {
            "scene_hit": None,
            "robot_clearance_hit": None,
            "payload_gap_m": 0.020,
            "previous_payload_gap_m": 0.018,
            "pose_deviation_deg": 0.1,
            "pose_limit_deg": 12.0,
            "payload_bottom_z_m": 0.040,
            "payload_bottom_limit_z_m": 0.036,
        }
        sample.update(overrides)
        return pre_ompl_step_allowed(**sample)

    def test_decreasing_payload_gap_rejects_the_step(self) -> None:
        self.assertFalse(self._allowed(
            payload_gap_m=0.017, previous_payload_gap_m=0.018,
        ))
        self.assertTrue(self._allowed(
            payload_gap_m=0.140,
            previous_payload_gap_m=0.142,
            payload_limit_m=0.027,
        ))
        self.assertFalse(self._allowed(
            payload_gap_m=0.026,
            previous_payload_gap_m=0.142,
            payload_limit_m=0.027,
        ))
        self.assertTrue(self._allowed())

    def test_ceiling_short_of_sweep_does_not_release_a_plan(self) -> None:
        status = pre_ompl_separation_status(
            lifted_m=0.030,
            lift_cap_m=0.030,
            payload_clearance_hit="payload-clearance:blue_box->clutter_tray",
        )
        self.assertEqual(status, "ceiling")
        self.assertFalse(pre_ompl_may_plan(status))
        self.assertTrue(pre_ompl_may_plan("sweep_met"))

    def test_height_only_start_deficit_requests_vertical_separation(self) -> None:
        self.assertEqual(
            pre_ompl_separation_mode(
                scene_hit=None,
                robot_clearance_hit=None,
                pose_deviation_deg=0.0,
                pose_limit_deg=12.0,
                payload_bottom_z_m=0.003,
                payload_bottom_limit_z_m=0.012,
                payload_clearance_hit=None,
            ),
            "lift",
        )
        self.assertEqual(
            pre_ompl_separation_status(
                lifted_m=0.0,
                lift_cap_m=0.030,
                payload_clearance_hit=None,
                payload_bottom_z_m=0.003,
                payload_bottom_limit_z_m=0.012,
            ),
            "continue",
        )

    def test_no_exact_direct_search_can_try_one_segmented_raise_ladder(self) -> None:
        self.assertTrue(segmented_raise_recovery_allowed([
            {"reason": "route_reuse_rejected"},
            {"reason": "ompl_no_exact"},
        ]))
        self.assertTrue(segmented_raise_recovery_allowed([
            {"reason": "payload_gap_below_sweep"},
        ]))
        self.assertFalse(segmented_raise_recovery_allowed([
            {"reason": "robot_clearance"},
        ]))

    def test_outer_corridors_route_around_the_base_without_naming_an_object(self) -> None:
        start = np.array((-0.31, -0.25, 0.18))
        goal = np.array((0.34, 0.30, 0.31))
        corridors = transport_corridor_waypoints(start, goal)
        self.assertEqual(len(corridors), 2)
        for via in corridors:
            self.assertGreaterEqual(float(np.linalg.norm(via[:2])), 0.50 - 1e-9)
            self.assertGreaterEqual(float(via[2]), 0.36)
            self.assertTrue(_segment_clears_base(start[:2], via[:2]))
            self.assertTrue(_segment_clears_base(via[:2], goal[:2]))

    def test_escape_offsets_rise_and_keep_the_step_length(self) -> None:
        from camera_demo import transport_escape_offsets_m

        offsets = transport_escape_offsets_m(
            np.array((-0.24, 0.42, 0.19)),
            np.array((0.34, 0.30, 0.28)),
            0.015,
        )
        self.assertEqual(offsets[0][0], "up")
        self.assertGreaterEqual(len(offsets), 4)
        names = [name for name, _offset in offsets]
        self.assertEqual(len(names), len(set(names)))
        for _name, offset in offsets:
            self.assertGreater(float(offset[2]), 0.0)
            self.assertAlmostEqual(float(np.linalg.norm(offset)), 0.015)

    def test_ik_branches_stay_inside_limits_and_include_the_live_pose(self) -> None:
        from camera_demo import ik_branch_seeds

        lower = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8])
        upper = np.array([2.8, 1.7, 2.8, -0.1, 2.8, 3.7, 2.8])
        q_now = np.array([-1.2, -0.4, 0.2, -2.1, 0.1, 1.8, -0.7])
        seeds = ik_branch_seeds(q_now, lower, upper, 0.4)
        self.assertGreaterEqual(len(seeds), 4)
        self.assertTrue(np.allclose(seeds[0], q_now))
        for seed in seeds:
            self.assertTrue(np.all(seed >= lower - 1e-9))
            self.assertTrue(np.all(seed <= upper + 1e-9))

    def test_experience_vias_use_midpoints_not_the_archived_endpoints(self) -> None:
        from camera_demo import experience_corridor_vias

        polyline = [
            np.array((-0.20, 0.40, 0.20)),
            np.array((-0.05, 0.55, 0.32)),
            np.array((0.10, 0.48, 0.34)),
            np.array((0.34, 0.30, 0.30)),
        ]
        vias = experience_corridor_vias(
            [polyline],
            np.array((-0.24, 0.42, 0.19)),
            np.array((0.34, 0.30, 0.28)),
        )
        self.assertTrue(vias)
        for via in vias:
            self.assertGreater(float(np.linalg.norm(via - polyline[0])), 0.08)
            self.assertGreater(float(np.linalg.norm(via - polyline[-1])), 0.08)
        self.assertFalse(any(np.allclose(via, polyline[0]) for via in vias))


class TransportGoalRejectionTests(unittest.TestCase):
    def test_stage_records_map_to_named_reasons(self) -> None:
        invalid = [{"goal_valid": False, "ompl_exact": False, "chord_dense": False}]
        self.assertEqual(classify_transport_goal(invalid), "goal_invalid")
        no_exact = [{"goal_valid": True, "ompl_exact": False, "chord_dense": False}]
        self.assertEqual(classify_transport_goal(no_exact), "ompl_no_exact")
        dense_miss = [
            {"goal_valid": True, "ompl_exact": True, "chord_dense": False},
            {"goal_valid": True, "ompl_exact": True, "chord_dense": False},
        ]
        self.assertEqual(classify_transport_goal(dense_miss), "chord_dense_rejected")
        self.assertEqual(
            classify_transport_goal(
                [{"goal_valid": True, "ompl_exact": True, "chord_dense": True}],
                ruckig_category="payload_clearance",
            ),
            "payload_clearance",
        )
        self.assertEqual(
            classify_transport_goal(
                [{"goal_valid": True, "ompl_exact": True, "chord_dense": True}],
                exact_minimum_clearance_m=-0.002,
            ),
            "negative_exact_gap",
        )
        self.assertEqual(
            classify_transport_goal(
                [{"goal_valid": True, "ompl_exact": True, "chord_dense": True}],
                exact_minimum_clearance_m=0.014,
            ),
            "validated",
        )


class RouteReuseTests(unittest.TestCase):
    def test_key_folds_yaw_and_keeps_stages_apart(self) -> None:
        hover = transport_route_identity(
            "green_block", np.array([0.34, 0.30, 0.28]), 0.20,
            support_name="plate", plan_stage="transport",
        )
        flipped = transport_route_identity(
            "green_block", np.array([0.39, 0.31, 0.28]), 0.20 + np.pi,
            support_name="plate", plan_stage="transport",
        )
        other_stage = transport_route_identity(
            "green_block", np.array([0.34, 0.30, 0.28]), 0.20,
            support_name="plate", plan_stage="place",
        )
        self.assertEqual(hover["key"], flipped["key"])
        self.assertEqual(hover["support_sector_m"], [0.30, 0.30])
        self.assertNotEqual(hover["key"], other_stage["key"])

    def test_splice_replaces_only_the_first_waypoint(self) -> None:
        archived = [np.zeros(7), np.ones(7), np.full(7, 2.0)]
        spliced = splice_reused_waypoints(np.full(7, 0.5), archived)
        self.assertIsNotNone(spliced)
        assert spliced is not None
        self.assertTrue(np.allclose(spliced[0], 0.5))
        self.assertTrue(np.allclose(spliced[1], 1.0))
        self.assertTrue(np.allclose(spliced[2], 2.0))
        self.assertIsNone(splice_reused_waypoints(np.zeros(7), [np.zeros(7)]))

    def test_same_key_replaces_and_cap_drops_the_oldest(self) -> None:
        first = {"key": "a", "selected_candidate_id": "old"}
        replacement = {"key": "a", "selected_candidate_id": "new"}
        replaced = upsert_transport_routes([first], replacement, cap=2)
        self.assertEqual(len(replaced), 1)
        self.assertEqual(replaced[0]["selected_candidate_id"], "new")
        grown = upsert_transport_routes(
            replaced,
            {"key": "b", "selected_candidate_id": "b"},
            cap=2,
        )
        grown = upsert_transport_routes(
            grown,
            {"key": "c", "selected_candidate_id": "c"},
            cap=2,
        )
        self.assertEqual([route["key"] for route in grown], ["b", "c"])

    def test_failed_check_is_named_and_not_released(self) -> None:
        points = [np.zeros(7), np.ones(7)]

        def sample_death(q: np.ndarray) -> str | None:
            if float(q[0]) > 0.5:
                return "payload_clearance"
            return None

        death = dense_chord_death(points, sample_death, 0.035)
        record = route_reuse_rejection(str(death), "green_block|plate|0.30,0.30|0.000|transport")
        self.assertEqual(record["reason"], "route_reuse_rejected")
        self.assertEqual(record["detail"], "payload_clearance")
        self.assertFalse(execution_sample_released(
            scene_hit=None,
            robot_clearance_hit=None,
            payload_clearance_hit="payload-clearance:green_block->clutter_tray",
            exact_minimum_clearance_m=0.020,
        ))
        clean = {
            "pose_deviation_deg": 1.0,
            "pose_limit_deg": 12.0,
            "payload_bottom_z_m": 0.05,
            "payload_bottom_limit_z_m": 0.03,
            "scene_hit": None,
            "robot_clearance_hit": None,
            "payload_clearance_hit": None,
        }
        self.assertIsNone(reuse_sample_death(clean))
        self.assertEqual(
            reuse_sample_death({**clean, "pose_deviation_deg": 20.0}),
            "pose",
        )


class EnvelopeMeasurementTests(unittest.TestCase):
    def test_asymmetric_handle_sets_the_pivot_radius(self) -> None:
        from camera_demo import (
            payload_motion_envelope_m, payload_pivot_radius_from_points,
        )

        hand_origin = np.zeros(3)
        # Pad midpoint sits below the hand body. The radius is from this point.
        grasp = np.array([0.0, 0.02, -0.10])
        body = [np.array([0.04, 0.0, 0.0]), np.array([-0.04, 0.0, 0.0])]
        handle = np.array([0.09, 0.0, 0.01])
        radius = payload_pivot_radius_from_points(grasp, [*body, handle])
        handle_radius = float(np.linalg.norm(handle - grasp))
        body_radius = float(np.linalg.norm(body[0] - grasp))
        self.assertAlmostEqual(radius, handle_radius)
        self.assertGreater(handle_radius, body_radius)
        self.assertNotAlmostEqual(
            radius,
            payload_pivot_radius_from_points(hand_origin, [*body, handle]),
        )
        envelope = payload_motion_envelope_m(0.010, 12.0, radius)
        self.assertAlmostEqual(
            envelope, 0.010 + radius * float(np.sin(np.deg2rad(12.0))),
        )
        self.assertGreater(envelope, payload_motion_envelope_m(0.010, 12.0, body_radius))
        with_margin = payload_motion_envelope_m(0.010, 12.0, radius, 0.011)
        self.assertAlmostEqual(with_margin, envelope + 0.011)

    def test_grasp_origin_is_the_pad_midpoint(self) -> None:
        import mujoco
        from camera_demo import grasp_contact_origin_m
        from joint_control import create_model

        model = create_model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        origin = grasp_contact_origin_m(model, data)
        hand = np.asarray(data.xpos[model.body("hand").id], dtype=float)
        self.assertGreater(float(hand[2] - origin[2]), 0.05)

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
            self.assertGreater(volume, 0.0)
            return weighted / volume

        expected = 0.5 * (centroid("left_finger") + centroid("right_finger"))
        self.assertTrue(np.allclose(origin, expected))
        self.assertFalse(np.allclose(origin, hand))

    def test_tracking_margin_removes_measured_slip_and_tilt(self) -> None:
        from camera_demo import ceil_tracking_margin_m, residual_tracking_margin_m

        slip = 0.003
        tilt = 2.0
        radius = 0.05
        physical = slip + radius * float(np.sin(np.deg2rad(tilt)))
        loss = physical + 0.0006
        residual = residual_tracking_margin_m(loss, slip, tilt, radius)
        self.assertAlmostEqual(residual, 0.0006)
        self.assertEqual(residual_tracking_margin_m(physical - 0.001, slip, tilt, radius), 0.0)
        self.assertAlmostEqual(ceil_tracking_margin_m(0.0), 0.0)
        self.assertAlmostEqual(ceil_tracking_margin_m(0.0001), 0.0005)
        self.assertAlmostEqual(ceil_tracking_margin_m(0.0005), 0.0005)
        self.assertAlmostEqual(ceil_tracking_margin_m(0.0006), 0.001)

    def test_strict_release_names_a_short_payload_gap(self) -> None:
        from camera_demo import (
            classify_transport_goal, execution_sample_released, transport_release_reason,
        )

        reason = transport_release_reason(0.020, 0.040, 0.056)
        self.assertEqual(reason, "payload_gap_below_sweep")
        self.assertEqual(
            classify_transport_goal(
                [{"goal_valid": True, "ompl_exact": True, "chord_dense": True}],
                ruckig_category=reason,
                exact_minimum_clearance_m=0.040,
            ),
            "payload_gap_below_sweep",
        )
        self.assertFalse(execution_sample_released(
            scene_hit=None,
            robot_clearance_hit=None,
            payload_clearance_hit=reason,
            exact_minimum_clearance_m=0.040,
        ))
        self.assertIsNone(transport_release_reason(0.020, 0.056, 0.056))
        self.assertEqual(transport_release_reason(0.011, 0.080, 0.056), "robot_clearance")

    def test_larger_slack_ranks_first(self) -> None:
        from camera_demo import TransportPlanCandidate, choose_transport_candidate

        shared = {
            "estimated_duration_s": 1.0,
            "maximum_pose_deviation_deg": 1.0,
            "maximum_tangential_acceleration_m_s2": 1.0,
            "minimum_clearance_m": 0.08,
            "maximum_base_turn_rad": 0.1,
            "maximum_wrist_swing_rad": 0.1,
        }
        tighter = TransportPlanCandidate(
            candidate_id="tight",
            validated=True,
            scores={**shared, "safety_slack_m": 0.001},
            joint_waypoints=(),
            execution_points=(),
            hand_waypoints=(),
            planning_time_s=0.1,
        )
        looser = TransportPlanCandidate(
            candidate_id="loose",
            validated=True,
            scores={**shared, "safety_slack_m": 0.020, "estimated_duration_s": 3.0},
            joint_waypoints=(),
            execution_points=(),
            hand_waypoints=(),
            planning_time_s=0.1,
        )
        chosen = choose_transport_candidate([tighter, looser], "time_first")
        self.assertEqual(chosen.candidate_id, "loose")

    def test_version_mismatch_is_not_an_entry(self) -> None:
        from camera_demo import stored_route_matches, transport_route_identity

        identity = transport_route_identity("green_block", np.array([0.34, 0.30, 0.2]), 0.0)
        self.assertIn("gap-v3", str(identity["key"]))
        previous = {**identity, "clearance_policy_version": "gap-v2"}
        self.assertFalse(stored_route_matches(previous, identity))
        stale = {**identity, "clearance_policy_version": "gap-v1"}
        self.assertFalse(stored_route_matches(stale, identity))
        missing = {key: value for key, value in identity.items() if key != "clearance_policy_version"}
        self.assertFalse(stored_route_matches(missing, identity))
        self.assertTrue(stored_route_matches(identity, identity))

    def test_saved_id_does_not_nest_reuse_prefixes(self) -> None:
        from camera_demo import route_root_id

        self.assertEqual(route_root_id("route_reuse:route_reuse:ompl_abc"), "ompl_abc")
        self.assertEqual(route_root_id("ompl_abc"), "ompl_abc")
        self.assertEqual(
            f"route_reuse:{route_root_id('route_reuse:route_reuse:ompl_abc')}",
            "route_reuse:ompl_abc",
        )

    def test_raise_starts_only_after_a_payload_gap_wipeout(self) -> None:
        from camera_demo import payload_gap_only_wipeout, raise_deltas_m, raise_midpoint_xyz

        self.assertTrue(payload_gap_only_wipeout([
            {"reason": "route_reuse_rejected"},
            {"reason": "payload_gap_below_sweep"},
            {"reason": "payload_gap_below_sweep"},
        ]))
        self.assertFalse(payload_gap_only_wipeout([
            {"reason": "payload_gap_below_sweep"},
            {"reason": "ompl_no_exact"},
        ]))
        self.assertFalse(payload_gap_only_wipeout([{"reason": "ompl_no_exact"}]))
        self.assertFalse(payload_gap_only_wipeout([]))
        deltas = raise_deltas_m()
        self.assertEqual(deltas[0], 0.015)
        self.assertEqual(deltas[-1], 0.15)
        self.assertLessEqual(max(deltas), 0.15)
        midpoint = raise_midpoint_xyz(
            np.array([0.0, 0.0, 0.20]),
            np.array([0.40, 0.20, 0.25]),
            0.015,
        )
        self.assertAlmostEqual(float(midpoint[0]), 0.20)
        self.assertAlmostEqual(float(midpoint[2]), 0.265)


class TransportPreloadTests(unittest.TestCase):
    def test_preload_stops_as_soon_as_weak_side_is_safe(self) -> None:
        self.assertEqual(
            preload_force_decision(
                {"left", "right"}, {"left": 0.64, "right": 0.70}, 0.63, 2.0,
            ),
            "passed",
        )

    def test_preload_stops_before_strong_side_can_be_driven_past_limit(self) -> None:
        self.assertEqual(
            preload_force_decision(
                {"left", "right"}, {"left": 0.60, "right": 2.00}, 0.63, 2.0,
            ),
            "strong_limit",
        )
        self.assertEqual(
            preload_force_decision(
                {"left", "right"}, {"left": 0.64, "right": 1.50}, 0.63, 2.0,
            ),
            "passed",
        )

    def test_force_ratio_above_balance_does_not_fail_a_safe_weak_side(self) -> None:
        self.assertEqual(
            preload_force_decision(
                {"left", "right"}, {"left": 0.70, "right": 1.80}, 0.63, 2.0,
            ),
            "passed",
        )
        self.assertEqual(
            preload_force_decision(
                {"left", "right"}, {"left": 0.40, "right": 1.20}, 0.63, 2.0,
            ),
            "continue",
        )
        self.assertEqual(
            preload_force_decision(
                {"left", "right"}, {"left": 0.50, "right": 2.00}, 0.63, 2.0,
            ),
            "strong_limit",
        )

    def test_stable_grasp_does_not_chase_a_force_target(self) -> None:
        from camera_demo import stability_preload_request

        self.assertEqual(
            stability_preload_request(
                slip_m=0.0004,
                slip_reference_m=0.0,
                peak_force_n=1.2,
                force_ceiling_n=2.0,
                bilateral=True,
            ),
            "hold",
        )
        self.assertEqual(
            stability_preload_request(
                slip_m=0.0020,
                slip_reference_m=0.0,
                peak_force_n=1.2,
                force_ceiling_n=2.0,
                bilateral=True,
            ),
            "preload",
        )
        self.assertEqual(
            stability_preload_request(
                slip_m=0.0040,
                slip_reference_m=0.0,
                peak_force_n=2.0,
                force_ceiling_n=2.0,
                bilateral=True,
            ),
            "hold_at_limit",
        )
        self.assertEqual(
            stability_preload_request(
                slip_m=0.0040,
                slip_reference_m=0.0,
                peak_force_n=0.4,
                force_ceiling_n=2.0,
                bilateral=False,
            ),
            "hold",
        )

    def test_strong_limit_continues_only_while_the_grasp_stays_stable(self) -> None:
        from camera_demo import strong_limit_transport_decision

        self.assertEqual(
            strong_limit_transport_decision(
                slip_m=0.0037,
                slip_before_m=0.0034,
                tilt_deg=5.8,
                tilt_limit_deg=12.0,
                bilateral=True,
            ),
            "hold_at_limit",
        )
        self.assertEqual(
            strong_limit_transport_decision(
                slip_m=0.0060,
                slip_before_m=0.0034,
                tilt_deg=5.8,
                tilt_limit_deg=12.0,
                bilateral=True,
            ),
            "abort_unstable",
        )
        self.assertEqual(
            strong_limit_transport_decision(
                slip_m=0.0037,
                slip_before_m=0.0034,
                tilt_deg=12.4,
                tilt_limit_deg=12.0,
                bilateral=True,
            ),
            "abort_unstable",
        )
        self.assertEqual(
            strong_limit_transport_decision(
                slip_m=0.0037,
                slip_before_m=0.0034,
                tilt_deg=1.0,
                tilt_limit_deg=12.0,
                bilateral=False,
            ),
            "abort_unstable",
        )
        self.assertIsNone(
            next_transport_stability_action("hold_at_limit", True, True),
        )
        self.assertTrue(transport_pose_within_limit(4.6, 12.0))
        self.assertFalse(transport_pose_within_limit(12.1, 12.0))

    def test_pre_place_samples_center_offset_height_and_yaw(self) -> None:
        from camera_demo import (
            pre_place_pose_accepted,
            pre_place_pose_specs,
            pre_place_rank_key,
        )

        specs = pre_place_pose_specs()
        self.assertEqual(specs[0]["name"], "center/z0.00/yaw+0.00")
        names = {spec["name"] for spec in specs}
        self.assertIn("left/z0.00/yaw+0.00", names)
        self.assertIn("right/z0.00/yaw+0.00", names)
        self.assertIn("center/z0.04/yaw+0.00", names)
        self.assertIn("center/z0.08/yaw-0.12", names)
        self.assertTrue(all(
            abs(float(spec["yaw_delta_rad"])) < np.deg2rad(12.0) for spec in specs
        ))
        self.assertTrue(all(
            abs(float(spec["xy_offset_m"][0])) <= 0.02
            and abs(float(spec["xy_offset_m"][1])) <= 0.02
            for spec in specs
        ))
        clear = {
            "scene_hit": None,
            "robot_clearance_hit": None,
            "payload_clearance_hit": None,
            "pose_deviation_deg": 4.0,
            "pose_limit_deg": 12.0,
            "payload_bottom_z_m": 0.080,
            "payload_bottom_limit_z_m": 0.048,
        }
        self.assertTrue(pre_place_pose_accepted(clear))
        blocked = {**clear, "robot_clearance_hit": "link6->plate"}
        self.assertFalse(pre_place_pose_accepted(blocked))
        self.assertLess(
            pre_place_rank_key(0.0, 0.0, 0.0, 0.020, 0.040),
            pre_place_rank_key(0.02, 0.0, 0.0, 0.030, 0.050),
        )

    def test_transport_recovery_slows_before_requesting_preload(self) -> None:
        self.assertEqual(
            next_transport_stability_action("monitoring", True, False),
            "slow_down",
        )
        self.assertIsNone(
            next_transport_stability_action("slowed", True, False),
        )
        self.assertEqual(
            next_transport_stability_action("slowed", True, True),
            "preload",
        )

    def test_payload_pose_over_existing_monitor_limit_is_rejected(self) -> None:
        self.assertTrue(transport_pose_within_limit(11.9, 12.0))
        self.assertFalse(transport_pose_within_limit(12.1, 12.0))

    def test_rrt_connect_returns_a_densely_valid_plan(self) -> None:
        start = np.zeros(7)
        goal = np.array((0.35, -0.20, 0.12, -0.08, 0.0, 0.15, -0.10))
        plan = plan_rrt_connect(
            start,
            [goal],
            np.full(7, -1.0),
            np.full(7, 1.0),
            lambda state: bool(np.all(np.abs(state) <= 1.0)),
            timeout_s=0.3,
            interpolation_step_rad=0.04,
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan.validated)
        self.assertTrue(np.allclose(plan.as_arrays()[0], start))
        self.assertTrue(np.allclose(plan.as_arrays()[-1], goal, atol=1e-4))


class LLMLeaderSafetyTests(unittest.TestCase):
    def _task(self) -> TaskSpec:
        task, _ = plan_task("把绿色方块拿到黄色盘子上", use_llm=False)
        return task

    def _scene(self) -> SceneSnapshot:
        return SceneSnapshot(
            target=SceneObject(
                "target", "green box", "green", "box",
                (-0.32, -0.25, 0.08), 0.95, (10, 10, 80, 80),
            ),
            destination=SceneObject(
                "destination", "yellow plate", "yellow", "plate",
                (0.34, 0.31, 0.036), 0.95, (100, 100, 180, 180),
            ),
            obstacles=({"xy_m": [0.0, -0.48], "top_z_m": 0.30},),
            maximum_obstacle_top_m=0.30,
        )

    def test_required_mode_never_silently_falls_back(self) -> None:
        with patch("embodied_agent._load_local_env", return_value=None), patch.dict(
            "os.environ", {}, clear=True
        ):
            with self.assertRaises(LeaderUnavailableError):
                LLMRobotLeader(mode="required")

    def test_offline_leader_emits_a_complete_allowlisted_plan(self) -> None:
        leader = LLMRobotLeader(mode="offline")
        task = self._task()
        review = leader.review_scene("test", task, self._scene())
        self.assertTrue(review.approved)
        self.assertEqual(
            [call.name for call in review.skill_plan.calls], list(task.steps)
        )

    def test_skill_dsl_rejects_joint_commands(self) -> None:
        task = self._task()
        raw = DEFAULT_SKILL_LIBRARY.canonical_plan(task).to_list()
        raw[2]["arguments"]["joint_angles"] = [0.0] * 7
        with self.assertRaises(SkillPlanError):
            DEFAULT_SKILL_LIBRARY.validate_plan(raw, task)

    def test_local_checkpoint_veto_cannot_be_overridden(self) -> None:
        leader = LLMRobotLeader(mode="offline")
        review = leader.review_checkpoint(
            "verify_grasp", {"local_valid": False, "reason": "no_contact"}
        )
        self.assertFalse(review.may_continue)

    def test_checkpoint_review_is_three_state_with_whitelisted_adjustments(self) -> None:
        from embodied_agent import parse_checkpoint_review

        ok = parse_checkpoint_review(
            {"decision": "adjust", "adjustment": "raise_transport_height", "reason": "r"},
            True, "mock",
        )
        self.assertEqual(ok.decision, "adjust")
        self.assertTrue(ok.wants_adjust)
        self.assertEqual(ok.adjustment, "raise_transport_height")

        legacy = parse_checkpoint_review({"decision": "reobserve", "reason": "r"}, True, "mock")
        self.assertEqual((legacy.decision, legacy.adjustment), ("adjust", "reobserve"))

        unknown = parse_checkpoint_review(
            {"decision": "adjust", "adjustment": "move_left_3cm", "reason": "r"}, True, "mock"
        )
        self.assertEqual(unknown.decision, "stop")
        self.assertIn("白名单", unknown.reason)

        metric = parse_checkpoint_review(
            {"decision": "adjust", "adjustment": "raise_transport_height",
             "target_xyz": [0.3, 0.2, 0.1], "reason": "r"},
            True, "mock",
        )
        self.assertEqual(metric.decision, "stop")
        self.assertIn("target_xyz", metric.reason)

        vetoed = parse_checkpoint_review({"decision": "continue", "reason": "r"}, False, "mock")
        self.assertEqual(vetoed.decision, "stop")

    def test_online_checkpoint_goes_through_the_same_parser(self) -> None:
        leader = LLMRobotLeader(mode="offline")
        with patch.object(
            type(leader), "online", property(lambda self: True)
        ), patch.object(
            leader, "_chat_json",
            return_value={"decision": "adjust", "adjustment": "slow_down",
                          "waypoints": [[0.1, 0.2, 0.3]], "reason": "r"},
        ):
            review = leader.review_checkpoint("verify_transport", {"local_valid": True})
        self.assertEqual(review.decision, "stop")
        self.assertFalse(review.wants_adjust)

    def test_numpy_sensor_scalars_are_json_serializable(self) -> None:
        import json

        encoded = json.dumps(
            {"local_valid": np.bool_(True), "height": np.float64(0.036)},
            default=_json_default,
        )
        self.assertIn('"local_valid": true', encoded)

    def test_fragile_profile_is_stricter_than_normal_payload(self) -> None:
        normal = MOTION_LIMITS["payload"]
        fragile = MOTION_LIMITS["payload_fragile"]
        self.assertLess(fragile.max_velocity, normal.max_velocity)
        self.assertLess(fragile.max_acceleration, normal.max_acceleration)
        self.assertLess(fragile.max_jerk, normal.max_jerk)

    def test_elevated_cube_cannot_pass_as_destination_plate(self) -> None:
        scene = self._scene()
        false_plate = SceneObject(
            "destination", "yellow plate", "yellow", "plate",
            (0.0, -0.48, 0.30), 0.95, (100, 100, 180, 180),
        )
        invalid_scene = SceneSnapshot(
            target=scene.target,
            destination=false_plate,
            obstacles=scene.obstacles,
            maximum_obstacle_top_m=scene.maximum_obstacle_top_m,
        )
        review = LLMRobotLeader(mode="offline").review_scene(
            "test", self._task(), invalid_scene
        )
        self.assertFalse(review.approved)
        self.assertTrue(review.needs_reobserve)

    def test_recovery_does_not_replan_transport_before_grasp(self) -> None:
        leader = LLMRobotLeader(mode="offline")
        recovery = leader.decide_recovery({
            "reason": "grasp_approach_plan_failed",
            "safe_to_retry": True,
        })
        self.assertEqual(recovery.action, "reobserve_scene")

    def test_recovery_obeys_local_unsafe_veto(self) -> None:
        leader = LLMRobotLeader(mode="offline")
        recovery = leader.decide_recovery({
            "reason": "runtime_collision",
            "safe_to_retry": False,
        })
        self.assertEqual(recovery.action, "safe_stop")


class GraspReplanTests(unittest.TestCase):
    def _cup_miss(self) -> dict:
        return {
            "task": "pick red cup",
            "attempt": 1,
            "object": {
                "type": "cup",
                "position": [0.377, -0.189, 0.10],
                "width": 0.054,
                "height": 0.10,
                "bottom_z": 0.0,
                "top_z": 0.10,
                "orientation_deg": 0.0,
            },
            "current_grasp": {
                "strategy": "side_grasp",
                "approach_direction": "vertical",
                "palm_height": 0.307,
                "finger_target_height": 0.160,
                "grasp_height_ratio": 0.90,
            },
            "observation": {
                "cup_rim_height": 0.10,
                "nearest_obstacle_height": 0.160,
                "finger_clearance_to_rim": 0.060,
                "contact_force": 0.0,
                "object_displacement": 0.0,
                "collision": False,
            },
            "failure": {
                "type": "no_contact",
                "description": "fingers remained above the object top and never contacted it",
            },
        }

    def test_no_contact_above_the_rim_lowers_the_next_grasp(self) -> None:
        update = offline_grasp_replan(self._cup_miss())
        self.assertEqual(update.action, "retry_grasp")
        self.assertEqual(update.approach_direction, "horizontal")
        self.assertLessEqual(update.grasp_height_ratio, 0.50)
        self.assertGreaterEqual(update.grasp_height_ratio, 0.28)
        self.assertIn("上沿", update.reason)

    def test_safety_layer_rejects_joint_commands_and_clips_a_high_retry(self) -> None:
        diagnosis = self._cup_miss()
        with self.assertRaises(ValueError):
            parse_grasp_replan(
                {
                    "action": "retry_grasp",
                    "joint_angles": [0.1, 0.2],
                    "strategy_update": {"grasp_height_ratio": 0.4},
                },
                diagnosis,
            )
        proposal = parse_grasp_replan(
            {
                "decision": "replan_grasp",
                "reason": "try again just as high",
                "strategy_update": {
                    "grasp_strategy": "top_grasp",
                    "approach_direction": "vertical",
                    "grasp_height_ratio": 0.95,
                    "lateral_offset": 0.20,
                },
            },
            diagnosis,
        )
        clipped = clip_grasp_strategy(proposal, diagnosis)
        self.assertEqual(clipped.action, "retry_grasp")
        self.assertEqual(clipped.grasp_strategy, "side_grasp")
        self.assertLessEqual(clipped.grasp_height_ratio, 0.50)
        self.assertLessEqual(abs(clipped.lateral_offset_m), 0.003)

    def test_llm_semantic_region_is_mapped_by_the_local_planner(self) -> None:
        diagnosis = self._cup_miss()
        diagnosis["failure"] = {
            "type": "lift_failed",
            "description": "bilateral contact but the object did not rise",
        }
        diagnosis["observation"]["finger_clearance_to_rim"] = -0.05
        proposal = parse_grasp_replan(
            {
                "action": "retry_grasp",
                "reason": "use the upper body and avoid the handle",
                "strategy_update": {
                    "grasp_strategy": "side_grasp",
                    "approach_direction": "vertical",
                    "grasp_region": "upper_side",
                    "yaw_policy": "avoid_handle",
                },
            },
            diagnosis,
        )
        clipped = clip_grasp_strategy(proposal, diagnosis)
        self.assertEqual(clipped.grasp_region, "upper_side")
        self.assertEqual(clipped.yaw_policy, "avoid_handle")
        self.assertAlmostEqual(clipped.grasp_height_ratio, 0.68)

    def test_axisymmetric_grasp_does_not_chase_pca_yaw(self) -> None:
        from camera_demo import _generate_grasp_candidates

        current = 1.23
        span = {
            "minor_m": 0.068,
            "minor_yaw_rad": -0.75,
            "major_m": 0.072,
            "major_yaw_rad": 0.81,
            "has_handle": 0.0,
        }
        candidate = _generate_grasp_candidates(
            "purple_bottle", "bottle", span, 0.052, current,
            "middle_side", "preserve_wrist",
        )[0]
        self.assertAlmostEqual(candidate.yaw_rad, current)
        self.assertEqual(candidate.source, "axisymmetric preserve-wrist")

    def test_axisymmetric_grasp_samples_four_local_closing_axes(self) -> None:
        from camera_demo import _generate_grasp_candidates

        candidates = _generate_grasp_candidates(
            "white_mug", "cup",
            {
                "has_handle": 1.0,
                "handle_avoid_yaw_rad": 0.0,
            },
            0.064, 0.0, "middle_side", "avoid_handle",
        )
        self.assertEqual(len(candidates), 4)
        axes = {
            round(float(candidate.yaw_rad % np.pi), 6)
            for candidate in candidates
        }
        self.assertEqual(len(axes), 4)

    def test_grasp_axis_distance_treats_half_turn_as_equivalent(self) -> None:
        self.assertAlmostEqual(grasp_axis_distance_rad(0.0, np.pi), 0.0)
        self.assertAlmostEqual(
            grasp_axis_distance_rad(np.deg2rad(170.0), np.deg2rad(-10.0)),
            0.0,
        )
        self.assertAlmostEqual(
            grasp_axis_distance_rad(0.0, np.pi / 2.0), np.pi / 2.0,
        )

    def test_box_partial_contact_cannot_become_a_horizontal_shove(self) -> None:
        diagnosis = {
            "object": {
                "type": "box", "bottom_z": 0.0, "top_z": 0.08,
            },
            "current_grasp": {
                "strategy": "top_grasp",
                "approach_direction": "vertical",
                "grasp_height_ratio": 0.52,
            },
            "observation": {
                "left_normal_force_n": 5.61,
                "right_normal_force_n": 0.21,
                "recommended_centering_xy_m": [-0.0015, 0.0],
            },
            "failure": {"type": "partial_contact"},
        }
        proposal = parse_grasp_replan(
            {
                "action": "retry_grasp",
                "reason": "move sideways from the image",
                "strategy_update": {
                    "grasp_strategy": "side_grasp",
                    "approach_direction": "horizontal",
                    "grasp_height_ratio": 0.70,
                    "lateral_offset_m": 0.020,
                },
            },
            diagnosis,
        )
        clipped = clip_grasp_strategy(proposal, diagnosis)
        self.assertEqual(clipped.grasp_strategy, "top_grasp")
        self.assertEqual(clipped.approach_direction, "vertical")
        self.assertEqual(clipped.grasp_height_ratio, 0.52)
        self.assertEqual(clipped.lateral_offset_m, 0.0)

    def test_exhausted_retries_do_not_ask_for_the_same_grasp_again(self) -> None:
        recovery = LLMRobotLeader(mode="offline").decide_recovery({
            "reason": "grasp_no_contact",
            "safe_to_retry": True,
            "retries_exhausted": True,
            "diagnosis": self._cup_miss(),
        })
        self.assertEqual(recovery.action, "safe_stop")

    def test_cup_pinch_stays_on_the_cup_when_a_neighbor_is_taller(self) -> None:
        import mujoco

        from camera_demo import (
            _nearest_obstacle, _surface_under_xy, palm_from_height_ratio, reset_scene,
        )
        from joint_control import create_model

        model = create_model()
        data = mujoco.MjData(model)
        reset_scene(model, data)
        cup = data.xpos[model.body("red_cup").id]
        support = _surface_under_xy(model, data, cup[:2], {"red_cup"})
        neighbor_top, neighbor_gap = _nearest_obstacle(model, data, cup[:2], {"red_cup"})
        palm, finger, _ratio = palm_from_height_ratio(0.0, 0.10, 0.45, support)
        self.assertLess(support, 0.05)
        self.assertGreater(neighbor_top, 0.14)
        self.assertLess(neighbor_gap, 0.20)
        self.assertLess(finger, 0.10)
        self.assertGreater(finger, 0.02)
        self.assertLess(palm - 0.147, 0.10)


def _fake_detection(label: str, xyz: tuple[float, float, float]):
    from vision_detector import Detection

    return Detection(
        label=label, score=1.0, bbox_xyxy=(200, 160, 280, 280),
        contour=np.zeros((4, 1, 2), np.int32), mask=np.zeros((480, 640), np.uint8),
        center_uv=(320, 240), depth_m=0.4, camera_xyz_m=np.zeros(3),
        world_xyz_m=np.array(xyz, dtype=float), base_xyz_m=np.array(xyz, dtype=float),
    )


class _FakeRobot:
    def __init__(self) -> None:
        self.last_abort_reason: str | None = None
        self.last_grasp_diagnosis = None


class RecoveryLoopTests(unittest.TestCase):
    """The leader's enumerated recovery action must drive the next round."""

    CUP = (0.377, -0.189, 0.10)
    PLATE = (0.351, 0.264, 0.036)

    def _diagnosis(self) -> dict:
        return {
            "task": "pick red cup", "attempt": 2,
            "object": {"type": "cup", "position": list(self.CUP), "width": 0.054,
                       "height": 0.10, "bottom_z": 0.0, "top_z": 0.10, "orientation_deg": 0.0},
            "current_grasp": {"strategy": "side_grasp", "approach_direction": "vertical",
                              "palm_height": 0.185, "finger_target_height": 0.038,
                              "grasp_height_ratio": 0.38},
            "observation": {"cup_rim_height": 0.10, "nearest_obstacle_height": 0.16,
                            "finger_clearance_to_rim": -0.062, "contact_force": 0.0,
                            "object_displacement": 0.0, "collision": False},
            "failure": {"type": "no_contact", "description": "fingers closed without contact"},
        }

    def _run(self, execute_results, scan_results=None, reasons=None, diagnosis=None):
        from task_runner import run_task

        leader = LLMRobotLeader(mode="offline")
        task, backend = plan_task("把红色杯子拿到黄色盘子上", use_llm=False)
        robot = _FakeRobot()
        calls: list[dict] = []
        scans: list[int] = []
        results = list(execute_results)
        reason_list = list(reasons or [])
        scan_list = list(scan_results or [])

        def fake_execute(*args, **kwargs):
            calls.append(kwargs)
            ok = results.pop(0)
            if not ok:
                robot.last_abort_reason = reason_list.pop(0) if reason_list else "grasp_no_contact"
                robot.last_grasp_diagnosis = diagnosis
            return ok

        def scan(current_task, ceiling, obstacles, captures):
            scans.append(1)
            if scan_list:
                return scan_list.pop(0)
            return (_fake_detection("red cup", self.CUP),
                    _fake_detection("yellow plate", self.PLATE), -25.0)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "task_log.OUTPUT_DIR", Path(tmp)
        ), patch("task_runner.execute_pick_and_place", side_effect=fake_execute):
            outcome = run_task(
                "把红色杯子拿到黄色盘子上", leader, None, None, None, robot,
                scan=scan, task=task, planner_backend=backend,
            )
            self.assertIsNotNone(outcome.task_log_path)
            written = json.loads(Path(outcome.task_log_path).read_text(encoding="utf-8"))
        self.assertEqual(set(written), set(SECTIONS))
        self.assertEqual(len(written["recovery"]), len(outcome.recovery_log))
        return outcome, calls, scans

    def test_retry_grasp_reruns_execution_with_a_clipped_initial_strategy(self) -> None:
        outcome, calls, _ = self._run(
            [False, True], reasons=["grasp_no_contact"], diagnosis=self._diagnosis()
        )
        self.assertTrue(outcome.success)
        self.assertEqual(len(calls), 2)
        self.assertEqual(outcome.recovery_rounds, 1)
        self.assertEqual(outcome.recovery_log[0]["action"], "retry_grasp")
        self.assertIsNone(calls[0]["initial_grasp"])
        seeded = calls[1]["initial_grasp"]
        self.assertIsNotNone(seeded)
        self.assertEqual(seeded.grasp_strategy, "side_grasp")
        self.assertLessEqual(seeded.grasp_height_ratio, 0.72)
        self.assertGreaterEqual(seeded.grasp_height_ratio, 0.28)

    def test_recovery_budget_stops_after_two_rounds(self) -> None:
        outcome, calls, _ = self._run(
            [False, False, False, False],
            reasons=["grasp_no_contact"] * 4, diagnosis=self._diagnosis(),
        )
        self.assertFalse(outcome.success)
        self.assertEqual(len(calls), 3)
        self.assertEqual(outcome.recovery_rounds, 2)
        self.assertTrue(outcome.reason.startswith("recovery_rounds_exhausted"))
        self.assertEqual(outcome.recovery_log[-1]["action"], "safe_stop")

    def test_unsafe_failure_stops_without_a_second_round(self) -> None:
        outcome, calls, _ = self._run(
            [False, True], reasons=["object_dropped_to_floor"], diagnosis=None
        )
        self.assertFalse(outcome.success)
        self.assertEqual(len(calls), 1)
        self.assertEqual(outcome.recovery_rounds, 0)
        self.assertTrue(outcome.reason.startswith("safe_stop"))

    def test_transport_failure_resumes_from_transport_on_another_route(self) -> None:
        outcome, calls, _ = self._run(
            [False, True], reasons=["transport_motion_plan_failed"], diagnosis=None
        )
        self.assertTrue(outcome.success)
        self.assertEqual(calls[1]["start_from"], "transport")
        self.assertNotEqual(calls[1]["route_policy"], calls[0]["route_policy"])

    def test_checkpoint_adjust_raise_transport_maps_to_high_clearance(self) -> None:
        from task_runner import run_task

        leader = LLMRobotLeader(mode="offline")
        task, backend = plan_task("把红色杯子拿到黄色盘子上", use_llm=False)
        robot = _FakeRobot()
        robot.last_checkpoint_adjustment = None
        calls: list[dict] = []

        def fake_execute(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                verdict = kwargs["checkpoint_reviewer"]("verify_transport", {"local_valid": True}, None)
                self.assertEqual(verdict.adjustment, "raise_transport_height")
                robot.last_abort_reason = "checkpoint_adjust:verify_transport:raise_transport_height"
                robot.last_checkpoint_adjustment = "raise_transport_height"
                return False
            return True

        def scan(current_task, ceiling, obstacles, captures):
            return (_fake_detection("red cup", self.CUP),
                    _fake_detection("yellow plate", self.PLATE), -25.0)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "task_log.OUTPUT_DIR", Path(tmp)
        ), patch.object(
            type(leader), "online", property(lambda self: True)
        ), patch.object(
            leader, "_chat_json",
            return_value={"decision": "adjust", "adjustment": "raise_transport_height",
                          "reason": "box close to path"},
        ), patch("task_runner.execute_pick_and_place", side_effect=fake_execute):
            outcome = run_task(
                "把红色杯子拿到黄色盘子上", leader, None, None, None, robot,
                scan=scan, task=task, planner_backend=backend,
            )
        self.assertTrue(outcome.success)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["route_policy"], "high_clearance")
        self.assertEqual(calls[1]["start_from"], "transport")
        self.assertEqual(outcome.recovery_log[0]["backend"], "local-checkpoint-adjust")
        self.assertEqual(outcome.recovery_rounds, 1)

    def test_checkpoint_adjust_slow_down_only_tightens_transport_mode(self) -> None:
        from task_runner import _map_checkpoint_adjustment

        task, _ = plan_task("把红色杯子拿到黄色盘子上", use_llm=False)
        recovery, strategy, route, slower = _map_checkpoint_adjustment(
            "slow_down", task, None, "auto", holding=True
        )
        self.assertEqual(recovery.action, "replan_transport")
        self.assertIsNone(strategy)
        self.assertEqual(route, "auto")
        self.assertEqual(slower.constraints.transport_mode, "fragile")

    def test_checkpoint_adjust_cannot_regrasp_while_holding(self) -> None:
        from task_runner import _map_checkpoint_adjustment

        task, _ = plan_task("把红色杯子拿到黄色盘子上", use_llm=False)
        for adjustment in ("switch_grasp_side", "reobserve"):
            recovery, _strategy, _route, _task = _map_checkpoint_adjustment(
                adjustment, task, None, "auto", holding=True
            )
            self.assertEqual(recovery.action, "safe_stop")

    def test_checkpoint_switch_side_is_clipped_and_reobserves(self) -> None:
        from task_runner import _map_checkpoint_adjustment

        task, _ = plan_task("把红色杯子拿到黄色盘子上", use_llm=False)
        recovery, strategy, _route, _task = _map_checkpoint_adjustment(
            "switch_grasp_side", task, self._diagnosis(), "auto", holding=False
        )
        self.assertEqual(recovery.action, "retry_grasp")
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.yaw_delta_rad, 0.0)
        self.assertTrue(strategy.need_reobserve)
        self.assertEqual(strategy.grasp_strategy, "side_grasp")

    def test_missing_reference_triggers_a_rescan_instead_of_failure(self) -> None:
        first = (_fake_detection("red cup", self.CUP), None, -25.0)
        outcome, calls, scans = self._run([True], scan_results=[first])
        self.assertTrue(outcome.success)
        self.assertEqual(len(scans), 2)
        self.assertEqual(outcome.recovery_rounds, 1)
        self.assertEqual(outcome.recovery_log[0]["trigger"], "perception_reference_not_found")


class PickStrategyTests(unittest.TestCase):
    """The manager picks words; the local layer maps and clips them."""

    def _cup_task(self) -> TaskSpec:
        task, _ = plan_task("把红色杯子拿到黄色盘子上", use_llm=False)
        return task

    def _scene(self) -> SceneSnapshot:
        return SceneSnapshot(
            target=SceneObject("target", "red cup", "red", "cup",
                               (0.38, -0.18, 0.05), 0.95, (10, 10, 80, 80)),
            destination=SceneObject("destination", "yellow plate", "yellow", "plate",
                                    (0.34, 0.30, 0.036), 0.95, (100, 100, 180, 180)),
            obstacles=(), maximum_obstacle_top_m=0.16,
        )

    def test_fast_horizontal_on_a_cup_is_clipped_and_logged(self) -> None:
        leader = LLMRobotLeader(mode="offline")
        task = self._cup_task()
        self.assertEqual(task.approach_direction, "vertical")
        with patch.object(
            type(leader), "online", property(lambda self: True)
        ), patch.object(leader, "_chat_json", return_value={
            "approach": "horizontal", "grip_width": "medium", "lift_height": "high",
            "transport": "direct", "speed_profile": "fast", "reason": "cup is light",
        }):
            strategy = leader.plan_pick_strategy(task, self._scene())
        # First attempt keeps the planner's vertical approach; horizontal entry
        # is reserved for replan_grasp after a measured failure.
        self.assertEqual(strategy.approach, "vertical")
        self.assertEqual(strategy.speed_profile, "normal")
        self.assertEqual(set(strategy.clipped_fields), {"approach", "speed_profile"})
        self.assertIn("pick-strategist", strategy.backend)

    def test_horizontal_is_kept_when_the_planner_already_chose_it(self) -> None:
        from dataclasses import replace as dc_replace

        from embodied_agent import clip_pick_strategy, default_pick_strategy

        task = dc_replace(self._cup_task(), approach_direction="horizontal")
        clipped = clip_pick_strategy(
            dc_replace(default_pick_strategy(task), approach="horizontal"), task
        )
        self.assertEqual(clipped.approach, "horizontal")
        self.assertNotIn("approach", clipped.clipped_fields)

    def test_llm_failure_falls_back_to_defaults_without_blocking(self) -> None:
        leader = LLMRobotLeader(mode="offline")
        with patch.object(
            type(leader), "online", property(lambda self: True)
        ), patch.object(leader, "_chat_json", side_effect=RuntimeError("timeout")):
            strategy = leader.plan_pick_strategy(self._cup_task(), self._scene())
        self.assertEqual(strategy.backend, "fallback-default")
        self.assertEqual((strategy.grip_width, strategy.lift_height, strategy.transport),
                         ("wide", "medium", "direct"))

    def test_low_level_fields_are_rejected(self) -> None:
        from embodied_agent import parse_pick_strategy

        with self.assertRaises(ValueError):
            parse_pick_strategy({
                "approach": "vertical", "grip_width": "wide", "lift_height": "medium",
                "transport": "direct", "speed_profile": "normal",
                "joint_angles": [0.0] * 7,
            })
        with self.assertRaises(ValueError):
            parse_pick_strategy({
                "approach": "diagonal", "grip_width": "wide", "lift_height": "medium",
                "transport": "direct", "speed_profile": "normal",
            })

    def test_box_horizontal_is_clipped_to_vertical(self) -> None:
        from embodied_agent import clip_pick_strategy, default_pick_strategy
        from dataclasses import replace as dc_replace

        task, _ = plan_task("把绿色方块拿到黄色盘子上", use_llm=False)
        proposal = dc_replace(default_pick_strategy(task), approach="horizontal")
        clipped = clip_pick_strategy(proposal, task)
        self.assertEqual(clipped.approach, "vertical")
        self.assertIn("approach", clipped.clipped_fields)

    def test_local_mapping_only_tightens(self) -> None:
        from camera_demo import _lift_offset_for, _pre_open_gap
        from embodied_agent import default_pick_strategy
        from dataclasses import replace as dc_replace
        from task_runner import _apply_pick_strategy

        task = self._cup_task()
        slow_via = dc_replace(default_pick_strategy(task), transport="via_safe_point",
                              speed_profile="slow")
        mapped_task, route = _apply_pick_strategy(task, slow_via, "auto")
        self.assertEqual(route, "high_clearance")
        self.assertEqual(mapped_task.constraints.transport_mode, "fragile")
        fragile_task = mapped_task
        back, route2 = _apply_pick_strategy(
            fragile_task, dc_replace(default_pick_strategy(task), speed_profile="normal"),
            "high_clearance",
        )
        self.assertEqual(back.constraints.transport_mode, "fragile")
        self.assertEqual(route2, "high_clearance")
        self.assertAlmostEqual(_lift_offset_for("medium"), 0.07)
        self.assertAlmostEqual(_lift_offset_for("low"), 0.05)
        self.assertAlmostEqual(_lift_offset_for("high"), 0.09)
        self.assertAlmostEqual(_pre_open_gap("wide", 0.054), 0.078)
        self.assertAlmostEqual(_pre_open_gap("medium", 0.054), 0.074)
        self.assertAlmostEqual(_pre_open_gap("narrow", 0.054), 0.064)
        self.assertAlmostEqual(_pre_open_gap("narrow", 0.075), 0.078)


class PlanBriefTests(unittest.TestCase):
    def test_depth_marker_clearance_is_explicitly_approximate(self) -> None:
        from camera_demo import _summarize_plan

        brief = _summarize_plan(
            [np.array([0.30, 0.20, 0.30]), np.array([0.40, 0.20, 0.40])],
            obstacles=[{"xy": np.array([0.35, 0.20]), "top_z": 0.20}],
        )
        self.assertTrue(brief["min_obstacle_clearance_is_approximate"])
        self.assertEqual(
            brief["min_obstacle_clearance_semantics"],
            "approximate_depth_marker_clearance",
        )

    def test_summary_of_a_fixed_three_point_path(self) -> None:
        from camera_demo import FINGER_PAD_BELOW_HAND, _summarize_plan

        waypoints = [
            np.array([0.40, -0.20, 0.30]),
            np.array([0.40, -0.20, 0.45]),
            np.array([0.35, 0.30, 0.45]),
        ]
        brief = _summarize_plan(
            waypoints, durations=[0.4, 0.9],
            obstacles=[{"xy": np.array([0.0, -0.48]), "top_z": 0.30}],
            payload_drop_m=None, route_policy="auto", plan_id="plan_test", stage="test",
        )
        expected_length = 0.15 + float(np.hypot(0.05, 0.50))
        self.assertAlmostEqual(brief["path_length_m"], round(expected_length, 3))
        # The start pose is reported separately; travel clearance excludes it.
        self.assertAlmostEqual(brief["start_bottom_clearance_m"], round(0.30 - FINGER_PAD_BELOW_HAND, 3))
        self.assertAlmostEqual(brief["min_table_clearance_m"], round(0.45 - FINGER_PAD_BELOW_HAND, 3))
        self.assertAlmostEqual(brief["predicted_duration_s"], 1.3)
        self.assertEqual(brief["waypoint_count"], 3)
        self.assertEqual(len(brief["key_waypoints"]), 3)
        self.assertIsNotNone(brief["min_obstacle_clearance_m"])
        self.assertNotIn("payload_held", brief["risk_flags"])
        held = _summarize_plan(waypoints, payload_drop_m=0.12, route_policy="high_clearance")
        self.assertIn("payload_held", held["risk_flags"])
        self.assertIn("high_clearance_route", held["risk_flags"])
        self.assertAlmostEqual(held["start_bottom_clearance_m"], 0.18)
        self.assertAlmostEqual(held["min_table_clearance_m"], 0.33)

    def test_plan_brief_adjust_with_waypoints_is_rejected(self) -> None:
        leader = LLMRobotLeader(mode="offline")
        with patch.object(
            type(leader), "online", property(lambda self: True)
        ), patch.object(leader, "_chat_json", return_value={
            "decision": "adjust", "adjustment": "raise_transport_height",
            "waypoints": [[0.1, 0.2, 0.3]], "reason": "r",
        }):
            verdict = leader.review_plan_brief({"path_length_m": 0.5})
        self.assertEqual(verdict.decision, "stop")
        self.assertIn("waypoints", verdict.reason)

    def test_plan_brief_normal_adjust_and_api_error_paths(self) -> None:
        leader = LLMRobotLeader(mode="offline")
        online = patch.object(type(leader), "online", property(lambda self: True))
        with online, patch.object(leader, "_chat_json", return_value={
            "decision": "adjust", "adjustment": "slow_down", "reason": "long path",
        }):
            verdict = leader.review_plan_brief({"path_length_m": 1.2})
        self.assertTrue(verdict.wants_adjust)
        self.assertEqual(verdict.adjustment, "slow_down")
        with online, patch.object(leader, "_chat_json", side_effect=RuntimeError("timeout")):
            fallback = leader.review_plan_brief({"path_length_m": 1.2})
        self.assertTrue(fallback.may_continue)
        self.assertEqual(fallback.backend, "offline-plan-brief-fallback")
        with online, patch.object(leader, "mode", "required"), patch.object(
            leader, "_chat_json", side_effect=RuntimeError("timeout")
        ):
            strict = leader.review_plan_brief({"path_length_m": 1.2})
        self.assertEqual(strict.decision, "stop")
        with online, patch.object(leader, "_chat_json", side_effect=RuntimeError("timeout")):
            checkpoint = leader.review_checkpoint("verify_grasp", {"local_valid": True})
        self.assertTrue(checkpoint.may_continue)
        self.assertEqual(checkpoint.backend, "offline-checkpoint-fallback")

    def test_offline_plan_brief_continues(self) -> None:
        verdict = LLMRobotLeader(mode="offline").review_plan_brief({"path_length_m": 0.5})
        self.assertTrue(verdict.may_continue)
        self.assertEqual(verdict.backend, "offline-plan-brief")


class TaskLogTests(unittest.TestCase):
    def test_seven_sections_round_trip_through_json(self) -> None:
        from embodied_agent import CheckpointReview

        task, backend = plan_task("把红色杯子拿到黄色盘子上", use_llm=False)
        log = TaskLog.start("把红色杯子拿到黄色盘子上", task, backend)
        log.set_strategy({"grasp_family": "side_grasp"}, backend="offline")
        log.add_plan_brief({"path_length_m": np.float64(0.42)}, "continue")
        log.add_checkpoint("verify_grasp", {"local_valid": np.bool_(True)},
                           CheckpointReview("continue", "ok", "mock"))
        log.add_failure("grasp_no_contact", {"failure": {"type": "no_contact"}}, attempt=1)
        log.add_recovery_round({"round": 0, "action": "retry_grasp"})
        log.set_delivery(success=True, xyz=np.array([0.3, 0.2, 0.1]))
        with tempfile.TemporaryDirectory() as tmp:
            path = log.write(Path(tmp))
            self.assertTrue(path.name.startswith("tasklog_"))
            self.assertIn("red_cup_to_yellow_plate", path.name)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(data), set(SECTIONS))
        self.assertEqual(len(SECTIONS), 7)
        self.assertEqual(data["checkpoints"][0]["verdict"]["decision"], "continue")
        self.assertEqual(data["delivery"]["xyz"], [0.3, 0.2, 0.1])
        self.assertEqual(task_short_code(task), "red_cup_to_yellow_plate")


class PlaceMicroAdjustTests(unittest.TestCase):
    """Pre-release correction is bounded to 2 cm and uses verify_place bands."""

    def test_in_band_pose_needs_no_adjustment(self) -> None:
        from camera_demo import _place_micro_adjust

        self.assertIsNone(_place_micro_adjust(
            np.array([0.30, 0.30]), np.array([0.34, 0.30]), 0.040, 0.036,
        ))

    def test_large_xy_error_is_clipped_to_two_centimetres(self) -> None:
        from camera_demo import PLACE_MICRO_ADJUST_MAX_M, _place_micro_adjust

        shift = _place_micro_adjust(
            np.array([0.20, 0.30]), np.array([0.34, 0.30]), 0.040, 0.036,
        )
        self.assertIsNotNone(shift)
        self.assertAlmostEqual(float(np.linalg.norm(shift)), PLACE_MICRO_ADJUST_MAX_M)
        self.assertGreater(shift[0], 0.0)
        self.assertAlmostEqual(shift[1], 0.0)

    def test_height_out_of_band_reseats_without_lateral_drift(self) -> None:
        from camera_demo import _place_micro_adjust

        shift = _place_micro_adjust(
            np.array([0.34, 0.30]), np.array([0.34, 0.30]), 0.070, 0.036,
        )
        self.assertIsNotNone(shift)
        self.assertAlmostEqual(float(np.linalg.norm(shift)), 0.0)


class LocateFallbackTests(unittest.TestCase):
    """VLM pointing failures must fall back to RGB-D color, with evidence."""

    def _scan_with(self, locate_side_effect):
        import contextlib
        import io
        import tempfile

        import mujoco

        from camera_demo import IMAGE_HEIGHT, IMAGE_WIDTH, reset_scene
        from joint_control import PandaController, create_model
        import viewer_controller

        model = create_model()
        data = mujoco.MjData(model)
        reset_scene(model, data)
        robot = PandaController(model, data, speed=800.0)
        renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        task, _ = plan_task("把红色杯子拿到黄色盘子上", use_llm=False)
        leader = LLMRobotLeader(mode="offline")
        log = io.StringIO()
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.object(
                type(leader), "online", property(lambda self: True)
            ), patch.object(
                leader, "locate_in_views", side_effect=locate_side_effect
            ), patch.object(
                viewer_controller, "OUTPUT_DIR", __import__("pathlib").Path(tmp)
            ), contextlib.redirect_stdout(log):
                target, reference, _angle = viewer_controller.scan_scene_once(
                    task, model, data, renderer, robot, [0.0], [], {},
                    leader=leader, instruction="把红色杯子拿到黄色盘子上",
                )
                evidence = sorted(p.name for p in __import__("pathlib").Path(tmp).iterdir())
        finally:
            renderer.close()
        cup_xy = data.xpos[model.body("red_cup").id][:2]
        base = data.xpos[model.body("link0").id]
        rotation = data.xmat[model.body("link0").id].reshape(3, 3)
        cup_base_xy = (rotation.T @ (data.xpos[model.body("red_cup").id] - base))[:2]
        return target, reference, log.getvalue(), evidence, cup_base_xy

    def test_locate_error_falls_back_to_color_and_keeps_the_frames(self) -> None:
        target, reference, log, evidence, cup_xy = self._scan_with(ValueError("bad json"))
        self.assertIsNotNone(target)
        self.assertEqual(target.source, "color_fallback")
        self.assertLess(float(np.linalg.norm(target.base_xyz_m[:2] - cup_xy)), 0.05)
        self.assertIsNotNone(reference)
        self.assertEqual(reference.source, "color_fallback")
        self.assertIn("定位兜底=color_rgbd", log)
        self.assertIn("locate_in_views_error:ValueError", log)
        self.assertTrue(any(name.endswith(".json") for name in evidence))
        self.assertTrue(any(name.startswith("locate_fail_") and name.endswith(".png")
                            for name in evidence))

    def test_point_without_depth_falls_back_to_color(self) -> None:
        from embodied_agent import ImagePointing

        def point_at_nothing(instruction, task, views):
            # A pixel on the far horizon of the first view: no usable depth.
            return ImagePointing(
                target_view=0, target_u=0.5, target_v=0.001,
                destination_view=None, destination_u=None, destination_v=None,
                reason="test",
            )

        target, _reference, log, _evidence, cup_xy = self._scan_with(point_at_nothing)
        self.assertIsNotNone(target)
        self.assertEqual(target.source, "color_fallback")
        self.assertLess(float(np.linalg.norm(target.base_xyz_m[:2] - cup_xy)), 0.05)
        self.assertIn("定位兜底=color_rgbd", log)

    def test_missing_key_in_required_mode_still_stops(self) -> None:
        with patch("embodied_agent._load_local_env", return_value=None), patch.dict(
            "os.environ", {}, clear=True
        ):
            with self.assertRaises(LeaderUnavailableError):
                LLMRobotLeader(mode="required")


class ImagePointingTests(unittest.TestCase):
    def test_pointing_must_stay_inside_the_photo(self) -> None:
        pointing = parse_image_pointing(
            {
                "target_seen": True,
                "target_view": 0,
                "target_u": 0.42,
                "target_v": 0.61,
                "destination_seen": True,
                "destination_view": 1,
                "destination_u": 0.70,
                "destination_v": 0.40,
                "reason": "upright red cup, not the flat books",
            },
            view_count=2,
            need_destination=True,
        )
        self.assertEqual(pointing.target_view, 0)
        self.assertEqual(pointing.destination_view, 1)

    def test_missing_or_out_of_frame_point_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_image_pointing(
                {"target_seen": False, "reason": "only books"},
                view_count=1,
                need_destination=False,
            )
        with self.assertRaises(ValueError):
            parse_image_pointing(
                {"target_seen": True, "target_view": 0, "target_u": 1.4, "target_v": 0.2},
                view_count=1,
                need_destination=False,
            )

    def test_image_point_selects_the_cup_instead_of_the_larger_red_books(self) -> None:
        import mujoco

        from camera_demo import (
            CAMERA_NAME, IMAGE_HEIGHT, IMAGE_WIDTH, render_rgbd, reset_scene,
        )
        from joint_control import LEFT_ABOVE, PandaController, create_model
        from scene_layout import OBJECT_BEARING_DEG
        from vision_detector import detect_colored_object, detection_from_image_point

        model = create_model()
        data = mujoco.MjData(model)
        reset_scene(model, data)
        pose = LEFT_ABOVE.copy()
        pose[0] = np.deg2rad(OBJECT_BEARING_DEG["red"])
        robot = PandaController(model, data, speed=800.0)
        self.assertTrue(robot.move_to(pose, duration=0.22))
        renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        try:
            rgb, depth = render_rgbd(renderer, data)
            base_id = model.body("link0").id
            base_rotation = data.xmat[base_id].reshape(3, 3)

            def base_xy(body_name: str) -> np.ndarray:
                world = data.xpos[model.body(body_name).id]
                return (base_rotation.T @ (world - data.xpos[base_id]))[:2]

            cup_xy = base_xy("red_cup")
            book_xy = base_xy("clutter_books")

            def project(world: np.ndarray) -> tuple[float, float]:
                camera_id = model.camera(CAMERA_NAME).id
                point = data.cam_xmat[camera_id].reshape(3, 3).T @ (
                    world - data.cam_xpos[camera_id]
                )
                self.assertLess(point[2], -0.05)
                height, width = depth.shape
                fovy = np.deg2rad(float(model.cam_fovy[camera_id]))
                focal = height / (2.0 * np.tan(fovy / 2.0))
                cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
                u = cx + point[0] * focal / (-point[2])
                v = cy - point[1] * focal / (-point[2])
                self.assertGreaterEqual(u, 0.0)
                self.assertLess(u, width)
                self.assertGreaterEqual(v, 0.0)
                self.assertLess(v, height)
                return float(u / (width - 1)), float(v / (height - 1))

            cup_world = data.xpos[model.body("red_cup").id].copy()
            book_world = data.xpos[model.body("clutter_books").id].copy()
            book_world[2] = 0.075
            cup_uv = project(cup_world)
            book_uv = project(book_world)
            cup_detection = detection_from_image_point(
                rgb, depth, cup_uv[0], cup_uv[1], "red cup", model, data, CAMERA_NAME,
            )
            book_detection = detection_from_image_point(
                rgb, depth, book_uv[0], book_uv[1], "red books", model, data, CAMERA_NAME,
            )
            color_only = detect_colored_object(
                rgb, depth, "red", "cup", model, data, CAMERA_NAME,
            )
            self.assertIsNotNone(cup_detection)
            self.assertIsNotNone(book_detection)
            self.assertIsNotNone(color_only)
            assert cup_detection is not None and book_detection is not None
            assert color_only is not None
            self.assertLess(np.linalg.norm(cup_detection.base_xyz_m[:2] - cup_xy), 0.06)
            self.assertGreater(
                np.linalg.norm(cup_detection.base_xyz_m[:2] - book_xy), 0.12
            )
            self.assertLess(np.linalg.norm(book_detection.base_xyz_m[:2] - book_xy), 0.08)
            # Colour proposes components; the cup-size geometry gate rejects
            # the larger red book stack before scoring candidates.
            self.assertLess(np.linalg.norm(color_only.base_xyz_m[:2] - cup_xy), 0.06)
            self.assertGreater(np.linalg.norm(color_only.base_xyz_m[:2] - book_xy), 0.12)
            anchored = detect_colored_object(
                rgb, depth, "red", "cup", model, data, CAMERA_NAME,
                near_xy=cup_xy, near_radius=0.12,
            )
            self.assertIsNotNone(anchored)
            assert anchored is not None
            self.assertLess(np.linalg.norm(anchored.base_xyz_m[:2] - cup_xy), 0.12)
        finally:
            renderer.close()


class SupportReferenceTests(unittest.TestCase):
    def test_centroid_of_a_usable_mask_is_the_planar_mean(self) -> None:
        xy = np.array([0.34, 0.30]) + np.linspace(-0.02, 0.02, 40).reshape(40, 1) * np.array([1.0, 0.0])
        points = np.column_stack((xy, np.full(40, 0.04)))
        centroid = support_centroid_from_points(points, np.array([0.351, 0.264]))
        self.assertIsNotNone(centroid)
        assert centroid is not None
        self.assertLess(float(np.linalg.norm(centroid - np.array([0.34, 0.30]))), 0.002)

    def test_snap_adopts_8mm_keeps_3mm_and_keeps_a_missing_mask(self) -> None:
        original = np.array([0.351, 0.264])
        adopted = decide_support_snap(original, original + np.array([0.008, 0.0]))
        self.assertTrue(adopted["adopted"])
        self.assertGreaterEqual(float(adopted["deviation_m"]), 0.005)
        kept = decide_support_snap(original, original + np.array([0.003, 0.0]))
        self.assertFalse(kept["adopted"])
        self.assertEqual(kept["reason"], "within_5mm")
        missing = decide_support_snap(original, None)
        self.assertFalse(missing["adopted"])
        self.assertEqual(missing["reason"], "mask_unavailable")
        self.assertIsNone(support_centroid_from_points(np.zeros((5, 3)), original))

    def test_refresh_adopts_15mm_and_rejects_25mm_without_loosening_the_gate(self) -> None:
        locked = np.array([0.340, 0.300])
        near = arbitrate_support_refresh(locked, locked + np.array([0.015, 0.0]))
        self.assertTrue(near["adopted"])
        self.assertEqual(near["source"], "refresh_adopted")
        far = arbitrate_support_refresh(locked, locked + np.array([0.025, 0.0]))
        self.assertFalse(far["adopted"])
        self.assertEqual(far["reason"], "deviation")
        self.assertEqual(far["source"], "locked")
        np.testing.assert_allclose(far["xy"], locked)
        missing = arbitrate_support_refresh(locked, None)
        self.assertEqual(missing["reason"], "not_found")
        self.assertFalse(missing["adopted"])
        gated = arbitrate_support_refresh(locked, locked + np.array([0.060, 0.0]))
        self.assertEqual(gated["reason"], "association_gate")
        self.assertFalse(gated["within_association_gate"])
        self.assertEqual(SUPPORT_ASSOCIATION_GATE_M, 0.06)

    def test_destination_point_outside_the_6cm_gate_is_rejected(self) -> None:
        plate = np.array([0.34, 0.30])
        self.assertTrue(reference_point_within_color_gate(plate + np.array([0.037, 0.0]), plate))
        self.assertFalse(reference_point_within_color_gate(np.array([-0.50, 0.35]), plate))
        self.assertTrue(reference_point_within_color_gate(np.array([-0.50, 0.35]), None))
        self.assertEqual(SUPPORT_ASSOCIATION_GATE_M, 0.06)

    def test_headline_error_is_versus_locked_and_the_gate_still_uses_the_used_reference(self) -> None:
        used = np.array([0.340, 0.300])
        locked = np.array([0.300, 0.300])
        object_xy = used + np.array([0.010, 0.0])
        errors = place_error_pair(object_xy, used, locked)
        self.assertAlmostEqual(errors["error_vs_used_support_m"], 0.010)
        self.assertAlmostEqual(errors["error_vs_locked_reference_m"], 0.050)
        headline = errors["error_vs_locked_reference_m"]
        self.assertEqual(headline, errors["error_vs_locked_reference_m"])
        self.assertNotEqual(headline, errors["error_vs_used_support_m"])
        on_target = errors["error_vs_used_support_m"] <= 0.09
        self.assertTrue(on_target)
        self.assertFalse(errors["error_vs_locked_reference_m"] <= 0.0)


class GraspYawTests(unittest.TestCase):
    def _step(self, **overrides: object) -> float | None:
        arguments = {
            "high_side": "left",
            "yaw_policy": "geometry",
            "spent_rad": 0.0,
            "servo_residual_m": 0.0001,
            "translation_attempted": False,
            "force_rising": True,
        }
        arguments.update(overrides)
        return inline_yaw_adjust_rad(**arguments)  # type: ignore[arg-type]

    def test_left_is_negative_and_right_is_positive(self) -> None:
        left = self._step(high_side="left")
        right = self._step(high_side="right")
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        assert left is not None and right is not None
        self.assertAlmostEqual(left, -GRASP_YAW_ADJUST_STEP_RAD)
        self.assertAlmostEqual(right, GRASP_YAW_ADJUST_STEP_RAD)
        self.assertAlmostEqual(GRASP_YAW_ADJUST_TOTAL_RAD, np.deg2rad(2.0))

    def test_avoid_handle_and_large_residual_skip_yaw(self) -> None:
        self.assertIsNone(self._step(yaw_policy="avoid_handle"))
        self.assertIsNone(self._step(servo_residual_m=0.002, translation_attempted=False))
        self.assertIsNotNone(self._step(servo_residual_m=0.002, translation_attempted=True))
        self.assertIsNone(self._step(force_rising=False))

    def test_yaw_stops_at_plus_or_minus_two_degrees(self) -> None:
        spent = 0.0
        applied = 0
        for _ in range(4):
            step = self._step(spent_rad=spent)
            if step is None:
                break
            spent += step
            applied += 1
        self.assertEqual(applied, 2)
        self.assertAlmostEqual(spent, -GRASP_YAW_ADJUST_TOTAL_RAD)
        self.assertIsNone(self._step(spent_rad=spent))

    def test_translation_brake_on_rising_force_or_object_follow(self) -> None:
        before = {"left": 1.15, "right": 0.1}
        after = {"left": 2.34, "right": 0.1}
        self.assertEqual(
            inline_translation_should_brake(
                before, after, "left", np.array([0.001, 0.0]), np.array([0.0001, 0.0]),
            ),
            "contact_force_rose",
        )
        self.assertEqual(
            inline_translation_should_brake(
                before, before, "left", np.array([0.0014, 0.0]), np.array([0.0010, 0.0]),
            ),
            "object_followed",
        )
        self.assertIsNone(
            inline_translation_should_brake(
                before, before, "left", np.array([0.0014, 0.0]), np.array([0.0002, 0.0]),
            )
        )


class TransportSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        _TRANSPORT_SEED_CACHE.clear()

    def test_cache_miss_matches_the_historical_order(self) -> None:
        ordered = order_transport_plans(False, np.array([0.40, 0.20]), np.array([0.50, 0.10]))
        self.assertEqual(ordered, [(extra, 0) for extra in _TRANSPORT_HEIGHT_EXTRAS])
        self.assertEqual(ordered[0][0], 0.0)

    def test_opposite_sides_start_high_and_still_keep_every_candidate(self) -> None:
        ordered = order_transport_plans(True, np.array([0.50, 0.20]), np.array([-0.20, 0.40]))
        self.assertEqual(ordered[0], (_CROSS_BASE_SEED_EXTRA, 0))
        self.assertEqual(ordered[1], (_CROSS_BASE_SEED_EXTRA, 1))
        historical = [(extra, slot) for extra in _TRANSPORT_HEIGHT_EXTRAS for slot in (0, 1)]
        self.assertCountEqual(ordered, historical)
        self.assertIn((0.0, 0), ordered)

    def test_unvalidated_seed_is_not_remembered_and_a_validated_one_is_only_reordered(self) -> None:
        start = np.array([0.50, -0.30])
        goal = np.array([-0.40, 0.20])
        remember_validated_transport_seed(start, goal, 0.20, 1, validated=False)
        self.assertEqual(order_transport_plans(True, start, goal)[0], (_CROSS_BASE_SEED_EXTRA, 0))
        remember_validated_transport_seed(start, goal, 0.26, 0, validated=True)
        ordered = order_transport_plans(True, start, goal)
        self.assertEqual(ordered[0], (0.26, 0))
        historical = [(extra, slot) for extra in _TRANSPORT_HEIGHT_EXTRAS for slot in (0, 1)]
        self.assertCountEqual(ordered, historical)
        self.assertEqual(len(ordered), len(set(ordered)))


class ManagerWordingTests(unittest.TestCase):
    def test_white_cylinder_resolves_to_the_mug_by_shape(self) -> None:
        task, backend = plan_task("把白色圆柱拿到黄色盘子上", use_llm=False)
        self.assertIn("offline", backend)
        self.assertEqual(task.object_id, "white_mug")
        self.assertEqual(task.object_name, "cup")
        self.assertEqual(task.parse_path, "shape_match")
        self.assertEqual(task.object_confidence, "high")

    def test_unknown_noun_is_not_silently_a_cube(self) -> None:
        with self.assertRaises(NotFoundReport) as caught:
            plan_task("把白色小玩意拿到黄色盘子上", use_llm=False)
        self.assertNotIn("cube", caught.exception.message)
        self.assertIn("没找到", caught.exception.message)

    def test_pink_cylinder_is_not_found_and_names_no_grasp(self) -> None:
        with self.assertRaises(NotFoundReport) as caught:
            plan_task("把粉色圆柱拿到黄色盘子上", use_llm=False)
        self.assertIn("没找到", caught.exception.message)
        self.assertIn("粉色", caught.exception.message)
        self.assertNotIn("object_id", caught.exception.message)

    def test_white_cube_reports_the_mug_and_does_not_select_it(self) -> None:
        with self.assertRaises(NotFoundReport) as caught:
            plan_task("把白色方块拿到黄色盘子上", use_llm=False)
        self.assertIn("马克杯", caught.exception.message)
        self.assertNotEqual(getattr(caught.exception, "object_id", None), "white_mug")

    def test_rereview_accepts_a_cylinder_mug_and_rejects_a_cube(self) -> None:
        mug = record_for_color("white")
        assert mug is not None
        cylinder = TaskSpec(
            action="place", object_name="圆柱", object_color="white",
            reference="plate", relation="on", grasp_strategy="top_grasp",
            approach_direction="vertical", reason="shape word",
            reference_color="yellow",
        )
        accepted = apply_manager_rereview(
            CategoryMismatch(cylinder, mug),
            {"decision": "select", "object_id": "white_mug"},
        )
        self.assertEqual(accepted.object_id, "white_mug")
        self.assertEqual(accepted.parse_path, "manager_rereview")
        cube = TaskSpec(
            action="place", object_name="方块", object_color="white",
            reference="plate", relation="on", grasp_strategy="top_grasp",
            approach_direction="vertical", reason="shape word",
            reference_color="yellow",
        )
        with self.assertRaises(NotFoundReport) as caught:
            apply_manager_rereview(
                CategoryMismatch(cube, mug),
                {"decision": "select", "object_id": "white_mug"},
            )
        self.assertIn("马克杯", caught.exception.message)

    def test_rereview_uses_only_the_first_reply(self) -> None:
        mug = record_for_color("white")
        assert mug is not None
        requested = TaskSpec(
            action="place", object_name="圆柱", object_color="white",
            reference="plate", relation="on", grasp_strategy="side_grasp",
            approach_direction="vertical", reason="once",
            reference_color="yellow",
        )
        mismatch = CategoryMismatch(requested, mug)
        replies = (
            {"decision": "not_found", "object_id": None},
            {"decision": "select", "object_id": "white_mug"},
        )
        with self.assertRaises(NotFoundReport):
            apply_manager_rereview(mismatch, replies[0])


class OpenInventoryTests(unittest.TestCase):
    def test_reconcile_registry_hit_and_temporary_entry(self) -> None:
        from scene_perception import PerceivedObject, perceive_scene, reconcile_inventory

        self.assertEqual(perceive_scene([]), [])
        items = reconcile_inventory([
            PerceivedObject(
                "cup", "red", "cylinder", 0, 0.5, 0.5, xy_m=(0.38, -0.18),
            ),
            PerceivedObject(
                "bottle", "pink", "cylinder", 1, 0.5, 0.5,
                xy_m=(0.22, -0.30), dimensions_m=(0.048, 0.048, 0.11),
            ),
            PerceivedObject(
                "speck", "pink", "cylinder", 1, 0.2, 0.2, xy_m=(0.0, 0.90),
            ),
        ])
        red = items[0]
        self.assertEqual((red.object_id, red.confidence, red.body_name), ("red_cup", "high", "red_cup"))
        pink = items[1]
        self.assertEqual(pink.confidence, "low")
        self.assertEqual(pink.body_name, "pink_bottle")
        self.assertTrue(str(pink.object_id).startswith("generic_pink"))
        self.assertIsNone(items[2].body_name)
        magenta = reconcile_inventory([
            PerceivedObject(
                "bottle", "magenta", "cylinder", 0, 0.5, 0.5, xy_m=(0.22, -0.30),
            ),
        ])[0]
        self.assertEqual(magenta.color, "pink")
        self.assertEqual(magenta.body_name, "pink_bottle")

        checked = validate_task(TaskSpec(
            action="place", object_name="bottle", object_color="pink",
            reference="plate", relation="on", grasp_strategy="top_grasp",
            approach_direction="horizontal", reason="fixture",
            reference_color="yellow",
        ), inventory=[pink])
        self.assertEqual(checked.object_confidence, "low")
        self.assertEqual(checked.body_name, "pink_bottle")
        self.assertEqual(checked.parse_path, "inventory_temporary")
        self.assertEqual(checked.grasp_strategy, "side_grasp")
        log = TaskLog.start("把粉色瓶子拿到黄色盘子上", checked, "test")
        log.add_checkpoint("temporary_entry", {
            "description": "pink bottle",
            "measured_size_m": list(checked.measured_size_m or ()),
            "confidence": "low",
        }, "temporary")
        with tempfile.TemporaryDirectory() as tmp:
            data = json.loads(log.write(Path(tmp)).read_text(encoding="utf-8"))
        evidence = data["checkpoints"][0]["evidence"]
        self.assertEqual(evidence["confidence"], "low")
        self.assertEqual(evidence["measured_size_m"], list(checked.measured_size_m or ()))

    def test_offline_fixture_is_not_found_and_a_registered_plan_still_works(self) -> None:
        from scene_perception import perceive_scene

        self.assertEqual(perceive_scene([], ask_vlm=None), [])
        with self.assertRaises(NotFoundReport) as caught:
            plan_task("把粉色瓶子拿到黄色盘子上", use_llm=False)
        self.assertIn("没找到", str(caught.exception))
        red, backend = plan_task("把红色杯子拿到黄色盘子上", use_llm=False)
        self.assertEqual(red.object_id, "red_cup")
        self.assertIn("offline", backend)
        with self.assertRaises(NotFoundReport):
            validate_task(TaskSpec(
                action="place", object_name="bottle", object_color="pink",
                reference="plate", relation="on", grasp_strategy="side_grasp",
                approach_direction="vertical", reason="missing",
                reference_color="yellow",
            ), inventory=[])


class GraspCompositionTests(unittest.TestCase):
    def _cup(self) -> TaskSpec:
        return TaskSpec(
            action="pick", object_name="cup", object_color="white",
            reference=None, relation="on", grasp_strategy="side_grasp",
            approach_direction="vertical", reason="compose",
            object_id="white_mug",
        )

    def test_unregistered_step_is_rejected_and_fed_back(self) -> None:
        from task_planner import accept_grasp_plan_reply, apply_grasp_plan_feedback
        from robot_skills import GraspPlanRejected

        with self.assertRaises(GraspPlanRejected) as caught:
            accept_grasp_plan_reply({
                "steps": [{"name": "weld_object", "arguments": {}}],
                "reason": "attach it",
            })
        self.assertEqual(caught.exception.receipt["rejected_name"], "weld_object")
        repaired = apply_grasp_plan_feedback(caught.exception.receipt, {
            "steps": [
                {"name": "analyze_grasp", "arguments": {"focus": "handle_direction"}},
                {"name": "plan_pick", "arguments": {
                    "region": "body", "approach": "vertical", "yaw_mode": "avoid_handle",
                }},
            ],
            "reason": "grasp the cup body",
        })
        self.assertEqual(repaired.steps[0].name, "analyze_grasp")
        self.assertEqual(repaired.plan_pick_argument("region"), "body")

    def test_numeric_coordinate_is_rejected(self) -> None:
        from task_planner import accept_grasp_plan_reply
        from robot_skills import GraspPlanRejected

        with self.assertRaises(GraspPlanRejected) as caught:
            accept_grasp_plan_reply({
                "steps": [{"name": "plan_pick", "arguments": {"region": "body", "xy": [0.1, 0.2]}}],
                "reason": "point",
            })
        self.assertEqual(caught.exception.receipt["error"], "numeric_coordinate")

    def test_size_and_handle_gates_reject_a_partial_match(self) -> None:
        from task_planner import CASE_SIZE_TOLERANCE_M, match_case

        case = {
            "shape": "handled_cylinder", "width_m": 0.064,
            "has_handle": True, "support": "plate",
        }
        too_wide = match_case(case, {
            "shape": "handled_cylinder", "width_m": 0.070,
            "has_handle": True, "support": "plate",
        })
        self.assertFalse(too_wide["matched"])
        self.assertFalse(too_wide["criteria"]["size"]["ok"])
        self.assertEqual(CASE_SIZE_TOLERANCE_M, 0.003)
        other_handle = match_case(case, {
            "shape": "handled_cylinder", "width_m": 0.064,
            "has_handle": False, "support": "plate",
        })
        self.assertFalse(other_handle["matched"])
        self.assertFalse(other_handle["criteria"]["handle"]["ok"])

    def test_default_plan_keeps_the_current_center(self) -> None:
        from task_planner import default_grasp_plan, grasp_center_for_plan, offline_grasp_plan

        task = self._cup()
        default = default_grasp_plan(task)
        perceived = (0.10, 0.20)
        body = (0.095, 0.20)
        self.assertEqual(default.source, "default")
        self.assertEqual(default.plan_pick_argument("region"), "auto")
        self.assertIsNone(default.plan_pick_argument("yaw_mode"))
        self.assertEqual(grasp_center_for_plan(default, perceived, body), perceived)
        custom = offline_grasp_plan(task, {"shape": "handled_cylinder", "has_handle": True})
        self.assertEqual(grasp_center_for_plan(custom, perceived, body), body)
        grasp_steps = [step.name for step in custom.steps if step.name in {
            "analyze_grasp", "plan_pick", "execute_pick",
        }]
        self.assertGreaterEqual(len(grasp_steps), 2)
        self.assertNotEqual(
            [step.to_dict() for step in default.steps],
            [step.to_dict() for step in custom.steps],
        )

    def test_revision_is_recorded_as_a_different_plan(self) -> None:
        from task_planner import offline_grasp_plan, revise_grasp_plan

        plan = offline_grasp_plan(self._cup(), {"shape": "handled_cylinder"})
        revised = revise_grasp_plan(plan, "force_limit")
        self.assertNotEqual(
            [step.to_dict() for step in plan.steps],
            [step.to_dict() for step in revised.steps],
        )
        self.assertEqual(revised.source, "revised")
        self.assertIn("force_limit", revised.reason)
        self.assertEqual(revised.steps[0].name, "reobserve_scene")


class ForceSequenceTests(unittest.TestCase):
    def _sample(self, step: int, left: float, right: float, state: str) -> dict:
        return {
            "step": step,
            "aperture_m": 0.08 - step * 0.00025,
            "left_force_n": left,
            "right_force_n": right,
            "contact_state": state,
        }

    def test_short_sequence_keeps_every_sample_and_field(self) -> None:
        from camera_demo import thin_force_sequence

        samples = [
            self._sample(1, 0.0, 0.0, "no_contact"),
            self._sample(2, 0.4, 0.1, "left_only"),
            self._sample(3, 1.2, 0.9, "bilateral_loading"),
        ]
        kept = thin_force_sequence(samples)
        self.assertEqual(kept, samples)
        for sample in kept:
            self.assertEqual(
                set(sample),
                {
                    "step", "aperture_m", "left_force_n",
                    "right_force_n", "contact_state",
                },
            )

    def test_thinning_keeps_first_contact_peak_and_last_within_the_cap(self) -> None:
        from camera_demo import FORCE_SEQUENCE_MAX_ENTRIES, thin_force_sequence

        samples = [
            self._sample(step, 0.0, 0.0, "no_contact") for step in range(1, 41)
        ]
        samples.append(self._sample(41, 0.2, 0.05, "left_only"))
        samples.append(self._sample(42, 2.4, 0.3, "force_limit"))
        samples.extend(
            self._sample(step, 0.4, 0.4, "bilateral_loading")
            for step in range(43, 120)
        )
        kept = thin_force_sequence(samples)
        self.assertEqual(len(kept), FORCE_SEQUENCE_MAX_ENTRIES)
        self.assertLessEqual(len(kept), FORCE_SEQUENCE_MAX_ENTRIES)
        steps = [sample["step"] for sample in kept]
        self.assertEqual(steps, sorted(steps))
        self.assertIn(41, steps)
        self.assertIn(42, steps)
        self.assertEqual(steps[-1], 119)


class TranslationAndSeatingExperimentTests(unittest.TestCase):
    def test_translation_substep_stops_when_force_jumps_and_closure_continues(self) -> None:
        from camera_demo import (
            TRANSLATION_FORCE_SUBSTEP_M,
            consume_translation_force_trace,
            translation_substep_lengths,
        )

        lengths = translation_substep_lengths(0.0003)
        self.assertEqual(len(lengths), 3)
        self.assertTrue(all(abs(step - 0.0001) < 1e-9 for step in lengths))
        self.assertAlmostEqual(sum(lengths), 0.0003)
        self.assertTrue(all(step <= TRANSLATION_FORCE_SUBSTEP_M + 1e-12 for step in lengths))
        executed, reason, closure_continues = consume_translation_force_trace(
            [
                {"left": 0.70, "right": 0.20},
                {"left": 1.10, "right": 0.15},
                {"left": 3.246, "right": 0.0},
            ],
            {"left": 0.517, "right": 0.176},
            0.90,
        )
        self.assertEqual(reason, "force_above_target")
        self.assertTrue(closure_continues)
        self.assertEqual(len(executed), 2)
        self.assertNotIn(3.246, [sample["left"] for sample in executed])
        doubled, doubled_reason, doubled_continues = consume_translation_force_trace(
            [{"left": 0.85, "right": 0.10}],
            {"left": 0.40, "right": 0.10},
            0.90,
        )
        self.assertEqual(doubled_reason, "peak_force_doubled")
        self.assertTrue(doubled_continues)
        self.assertEqual(len(doubled), 1)

    def test_translation_substep_runs_to_completion_below_the_gate(self) -> None:
        from camera_demo import consume_translation_force_trace

        executed, reason, closure_continues = consume_translation_force_trace(
            [
                {"left": 0.48, "right": 0.30},
                {"left": 0.40, "right": 0.36},
            ],
            {"left": 0.517, "right": 0.176},
            0.90,
        )
        self.assertIsNone(reason)
        self.assertTrue(closure_continues)
        self.assertEqual(len(executed), 2)

    def test_experiment_switch_off_keeps_the_bounded_seating_cap(self) -> None:
        import os

        from camera_demo import (
            EXPERIMENT_SEATING_EXTRA_M,
            GRASP_SEATING_EXTRA_M,
            next_grasp_squeeze_m,
        )

        self.assertIsNone(EXPERIMENT_SEATING_EXTRA_M)
        self.assertEqual(GRASP_SEATING_EXTRA_M, 0.003)
        self.assertNotIn("EXPERIMENT_SEATING_EXTRA_M", os.environ)
        next_gap, used = next_grasp_squeeze_m(
            0.060, 0.063, "bilateral_unbalanced", True,
        )
        self.assertIsNone(next_gap)
        self.assertAlmostEqual(used, 0.003)

        import camera_demo

        previous = camera_demo.EXPERIMENT_SEATING_EXTRA_M
        camera_demo.EXPERIMENT_SEATING_EXTRA_M = 0.006
        try:
            extended, _used = next_grasp_squeeze_m(
                0.060, 0.063, "bilateral_loading", True,
            )
            self.assertAlmostEqual(extended, 0.05975)
        finally:
            camera_demo.EXPERIMENT_SEATING_EXTRA_M = previous


class ExecutionTrajectoryReleaseTests(unittest.TestCase):
    def test_exact_trajectory_check_rejects_a_negative_gap(self) -> None:
        from camera_demo import execution_sample_released

        self.assertFalse(execution_sample_released(
            scene_hit=None,
            robot_clearance_hit=None,
            payload_clearance_hit="payload-clearance:purple_bottle->right_pedestal",
            exact_minimum_clearance_m=-0.0022,
        ))
        self.assertFalse(execution_sample_released(
            scene_hit=None,
            robot_clearance_hit=None,
            payload_clearance_hit=None,
            exact_minimum_clearance_m=-0.001,
        ))
        self.assertTrue(execution_sample_released(
            scene_hit=None,
            robot_clearance_hit=None,
            payload_clearance_hit=None,
            exact_minimum_clearance_m=0.012,
        ))

    def test_approximate_marker_clearance_does_not_release_or_reject(self) -> None:
        from camera_demo import execution_sample_released

        # The -2.2 mm marker gap from ompl_439de2cd4b85 is not a release input.
        released = execution_sample_released(
            scene_hit=None,
            robot_clearance_hit=None,
            payload_clearance_hit=None,
            exact_minimum_clearance_m=0.011,
        )
        self.assertTrue(released)
        rejected = execution_sample_released(
            scene_hit="payload:purple_bottle->right_pedestal",
            robot_clearance_hit=None,
            payload_clearance_hit=None,
            exact_minimum_clearance_m=0.020,
        )
        self.assertFalse(rejected)


class GraspCandidateMemoryTests(unittest.TestCase):
    def _faces(self):
        from camera_demo import GraspCandidate

        return [
            GraspCandidate("middle_side", 0.50, 0.0, 0.040, 1.0, "RGB-D face geometry"),
            GraspCandidate("middle_side", 0.50, np.pi / 2.0, 0.055, 0.70, "alternate face"),
        ]

    def test_failed_axis_is_not_reselected_in_the_same_observation(self) -> None:
        from camera_demo import (
            grasp_candidate_status,
            order_grasp_candidates,
            remember_grasp_failure,
        )

        memory = remember_grasp_failure(
            [],
            yaw_rad=0.0,
            region="middle_side",
            approach="vertical",
            failure_type="large_displacement",
            attempt=1,
        )
        ordered = order_grasp_candidates(self._faces(), memory, "vertical")
        self.assertAlmostEqual(ordered[0].yaw_rad, np.pi / 2.0)
        self.assertEqual(
            grasp_candidate_status(np.pi, "middle_side", "vertical", memory),
            "failed_axis",
        )
        demoted = remember_grasp_failure(
            [],
            yaw_rad=0.0,
            region="middle_side",
            approach="vertical",
            failure_type="left_only",
            attempt=1,
        )
        recentered = order_grasp_candidates(self._faces(), demoted, "vertical")
        self.assertAlmostEqual(recentered[0].yaw_rad, np.pi / 2.0)
        self.assertEqual(
            grasp_candidate_status(0.0, "middle_side", "vertical", demoted),
            "failed_contact",
        )
        weak = remember_grasp_failure(
            [],
            yaw_rad=0.0,
            region="middle_side",
            approach="vertical",
            failure_type="weak_force_insufficient",
            attempt=1,
        )
        retried = order_grasp_candidates(self._faces(), weak, "vertical")
        self.assertAlmostEqual(retried[0].yaw_rad, np.pi / 2.0)

    def test_pose_change_reopens_the_failed_axis(self) -> None:
        from camera_demo import (
            order_grasp_candidates,
            refresh_grasp_attempt_memory,
            remember_grasp_failure,
        )

        memory = remember_grasp_failure(
            [],
            yaw_rad=0.0,
            region="middle_side",
            approach="vertical",
            failure_type="object_moved",
            attempt=1,
        )
        kept, _anchor = refresh_grasp_attempt_memory(
            memory, np.array([0.30, 0.20]), np.array([0.31, 0.20]),
        )
        self.assertEqual(len(kept), 1)
        cleared, _anchor = refresh_grasp_attempt_memory(
            memory, np.array([0.30, 0.20]), np.array([0.36, 0.20]),
        )
        self.assertEqual(cleared, [])
        ordered = order_grasp_candidates(self._faces(), cleared, "vertical")
        self.assertAlmostEqual(ordered[0].yaw_rad, 0.0)

    def test_no_contact_does_not_block_any_yaw(self) -> None:
        from camera_demo import (
            grasp_candidate_status,
            grasp_failure_dof,
            order_grasp_candidates,
            remember_grasp_failure,
        )

        self.assertEqual(grasp_failure_dof("no_contact"), "center_size_depth")
        memory = remember_grasp_failure(
            [],
            yaw_rad=0.0,
            region="middle_side",
            approach="vertical",
            failure_type="no_contact",
            attempt=1,
        )
        self.assertEqual(memory, [])
        blocked = remember_grasp_failure(
            [],
            yaw_rad=0.0,
            region="middle_side",
            approach="vertical",
            failure_type="approach_blocked",
            attempt=1,
        )
        shifted = order_grasp_candidates(self._faces(), blocked, "vertical")
        self.assertAlmostEqual(shifted[0].yaw_rad, np.pi / 2.0)
        self.assertEqual(
            grasp_candidate_status(np.pi / 2.0, "middle_side", "vertical", blocked),
            "untried",
        )
        ordered = order_grasp_candidates(self._faces(), memory, "vertical")
        self.assertAlmostEqual(ordered[0].yaw_rad, 0.0)
        self.assertAlmostEqual(ordered[1].yaw_rad, np.pi / 2.0)

    def test_bilateral_unbalanced_is_not_a_force_ratio_failure(self) -> None:
        from camera_demo import (
            failure_type_for_search,
            grasp_failure_status,
            remember_grasp_failure,
        )

        self.assertEqual(
            grasp_trial_release_decision(
                {"left", "right"},
                {"left": 0.75, "right": 1.80},
                0.90,
                2.00,
            ),
            "ready_for_trial_lift",
        )
        self.assertIsNone(grasp_failure_status("bilateral_unbalanced"))
        self.assertEqual(
            failure_type_for_search({
                "failure": {"type": "bilateral_unbalanced"},
                "observation": {
                    "left_normal_force_n": 0.75,
                    "right_normal_force_n": 1.80,
                },
            }),
            "bilateral_unbalanced",
        )
        self.assertEqual(
            failure_type_for_search({
                "failure": {"type": "bilateral_unbalanced"},
                "observation": {
                    "left_normal_force_n": 3.76,
                    "right_normal_force_n": 0.0,
                },
            }),
            "left_only",
        )
        self.assertEqual(
            remember_grasp_failure(
                [],
                yaw_rad=0.4,
                region="middle_side",
                approach="vertical",
                failure_type="bilateral_unbalanced",
                attempt=1,
            ),
            [],
        )

    def test_stable_contact_reseat_drops_once_and_stops_at_the_support(self) -> None:
        from camera_demo import grasp_reseat_drop_m

        self.assertAlmostEqual(grasp_reseat_drop_m(0.050, 0.0), 0.012)
        self.assertIsNone(grasp_reseat_drop_m(0.012, 0.0))
        self.assertAlmostEqual(grasp_reseat_drop_m(0.018, 0.0), 0.010)

    def test_side_approach_refuses_a_measured_protrusion(self) -> None:
        from camera_demo import approach_side_clears_protrusion

        origin = np.zeros(2)
        self.assertFalse(approach_side_clears_protrusion(
            np.array((0.10, 0.0)), origin, 0.0,
        ))
        self.assertTrue(approach_side_clears_protrusion(
            np.array((-0.10, 0.0)), origin, 0.0,
        ))
        self.assertTrue(approach_side_clears_protrusion(
            np.array((0.0, 0.10)), origin, 0.0,
        ))
        self.assertTrue(approach_side_clears_protrusion(
            np.array((0.10, 0.0)), origin, None,
        ))

    def test_transport_and_place_safety_contract_is_unchanged(self) -> None:
        from camera_demo import (
            execution_sample_released,
            transport_release_reason,
        )

        self.assertEqual(
            transport_release_reason(0.020, 0.040, 0.056),
            "payload_gap_below_sweep",
        )
        self.assertEqual(
            transport_release_reason(0.011, 0.080, 0.056),
            "robot_clearance",
        )
        self.assertFalse(execution_sample_released(
            scene_hit=None,
            robot_clearance_hit="link3->right_pedestal",
            payload_clearance_hit=None,
            exact_minimum_clearance_m=0.020,
        ))
        self.assertEqual(
            placement_release_stability_decision(
                displacement_m=0.020,
                tilt_deg=0.0,
                support_contacts=2,
                object_bottom_z_m=0.02,
                support_z_m=0.02,
            ),
            "large_displacement",
        )


if __name__ == "__main__":
    unittest.main()
