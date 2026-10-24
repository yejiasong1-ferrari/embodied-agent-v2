"""Closed-loop contact search inside one grasp axis, then a measured micro-lift."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from joint_control import GRIPPER_OPEN

from ..domain import ContactState, GraspCandidate, GraspEvidence
from ..motion.ik_solver import MultiSolutionIK
from ..perception.distance_sensor import fuse_distance_measurements


LOCAL_CORRECTION_STEP_M = 0.008
LOCAL_CORRECTION_LIMIT = 4
# Any finger load at or above this is pressure feedback. The close keeps
# going until that appears, then adds force in small steps.
CONTACT_PRESSURE_N = 0.05
SLOW_FORCE_STEP_M = 0.0015
SLOW_FORCE_STEPS = 8
SLOW_FORCE_HOLD_N = 3.0
# A pinch on the upper part of the body is still the rim. Lower before lifting.
RIM_CONTACT_FRACTION = 0.65
_HYPOTHESIS_FAILURES = frozenset({
    "local_exhausted",
    "lift_unstable",
    "ik_unreachable",
    "sensor_conflict",
    "object_pushed",
    "hard_force",
})


@dataclass(frozen=True)
class ContactAdjustment:
    """One millimetre correction inside the current grasp. Not a new attempt."""

    reason: str
    delta_m: tuple[float, float, float]


def _ray_aabb_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> float | None:
    """Distance from a ray to the first face of an axis-aligned box."""
    near = -math.inf
    far = math.inf
    for axis in range(3):
        if abs(float(direction[axis])) < 1e-9:
            if float(origin[axis]) < float(minimum[axis]) or float(origin[axis]) > float(maximum[axis]):
                return None
            continue
        first = (float(minimum[axis]) - float(origin[axis])) / float(direction[axis])
        second = (float(maximum[axis]) - float(origin[axis])) / float(direction[axis])
        near = max(near, min(first, second))
        far = min(far, max(first, second))
    if far < max(near, 0.0):
        return None
    return float(max(near, 0.0))


def is_grasp_hypothesis_failure(reason: str) -> bool:
    """True only when this axis can no longer be repaired in place."""
    return reason in _HYPOTHESIS_FAILURES


def contact_gap_correction(
    *,
    object_center_m: np.ndarray,
    top_z_m: float,
    bottom_z_m: float,
    finger_midpoint_m: np.ndarray,
    closing_axis_xy: np.ndarray,
    contact_state: ContactState,
) -> ContactAdjustment | None:
    """Choose one small hand move from a missed or one-sided contact.

    The step stays inside ``LOCAL_CORRECTION_STEP_M``. ``None`` means the
    fingers are already on the measured body and further motion would not
    explain the miss.
    """
    center = np.asarray(object_center_m, dtype=float)
    midpoint = np.asarray(finger_midpoint_m, dtype=float)
    closing = np.asarray(closing_axis_xy, dtype=float)[:2]
    norm = float(np.linalg.norm(closing))
    if norm < 1e-9:
        return None
    closing = closing / norm
    offset_xy = center[:2] - midpoint[:2]
    along = float(offset_xy @ closing)
    across_vec = offset_xy - closing * along
    across = float(np.linalg.norm(across_vec))
    finger_z = float(midpoint[2])
    step = LOCAL_CORRECTION_STEP_M

    if contact_state in {ContactState.LEFT_ONLY, ContactState.RIGHT_ONLY}:
        if abs(along) < 0.002:
            return None
        move = float(np.clip(along, -step, step))
        delta = np.array((closing[0] * move, closing[1] * move, 0.0))
        return ContactAdjustment("right" if along > 0.0 else "left", tuple(delta))

    if finger_z > float(top_z_m) - 0.004:
        return ContactAdjustment("height", (0.0, 0.0, -step))
    if finger_z < float(bottom_z_m) + 0.012:
        return ContactAdjustment("height", (0.0, 0.0, step))
    if abs(along) > 0.004:
        move = float(np.clip(along, -step, step))
        delta = np.array((closing[0] * move, closing[1] * move, 0.0))
        return ContactAdjustment("right" if along > 0.0 else "left", tuple(delta))
    if across > 0.004:
        direction = across_vec / across
        move = min(step, across)
        return ContactAdjustment(
            "too_far",
            (float(direction[0] * move), float(direction[1] * move), 0.0),
        )
    # Fingers are around the body but have not reached a useful depth.
    if finger_z > float(bottom_z_m) + 0.55 * max(float(top_z_m) - float(bottom_z_m), 1e-3):
        return ContactAdjustment("insertion", (0.0, 0.0, -min(step, 0.006)))
    return None


class ContactStateGraspController:
    def __init__(self, runtime, hard_force_n: float = 12.0, push_limit_m: float = 0.012):
        self.runtime = runtime
        self.hard_force_n = float(hard_force_n)
        self.push_limit_m = float(push_limit_m)
        self.ik = MultiSolutionIK(runtime.model)
        self.last_stop_reason = ""
        self.adjustments: tuple[ContactAdjustment, ...] = ()

    @staticmethod
    def _contact_state(contacts: set[str]) -> ContactState:
        if contacts == {"left", "right"}:
            return ContactState.BILATERAL
        if contacts == {"left"}:
            return ContactState.LEFT_ONLY
        if contacts == {"right"}:
            return ContactState.RIGHT_ONLY
        return ContactState.NO_CONTACT

    def _open_for_width(self, width_m: float) -> float:
        gap = min(0.08, max(0.03, float(width_m) + 0.018))
        return float(np.clip(gap / 0.08 * GRIPPER_OPEN, 0.0, GRIPPER_OPEN))

    def _close_floor(self, width_m: float) -> float:
        # Kept so a caller can still name the measured width. The close itself
        # does not stop at this gap: no pressure means keep closing.
        del width_m
        return float(np.clip(0.004 / 0.08 * GRIPPER_OPEN, 0.0, GRIPPER_OPEN))

    def _both_pressing(self, entity_id: str) -> bool:
        contacts, forces, _ = self._sensed_pressure(entity_id)
        return (
            contacts == {"left", "right"}
            and min(forces.get("left", 0.0), forces.get("right", 0.0)) >= CONTACT_PRESSURE_N
        )

    def _settled_bilateral(self, entity_id: str) -> bool:
        """Both fingers still pressing after a short pause.

        A single simulation step can show two sides and then lose one. That
        flicker is not a grip, so the close keeps going.
        """
        if not self._both_pressing(entity_id):
            return False
        self.runtime.robot.hold(duration=0.08)
        return self._both_pressing(entity_id)

    def _sensed_pressure(self, entity_id: str) -> tuple[set[str], dict[str, float], bool]:
        """Finger load, including contact that has ridden up to the rim.

        Returns the touching sides, their normal forces, and whether any side
        is at or above the pressure threshold.
        """
        contacts, forces = self.runtime.contacts_with_body(entity_id, pinch_band=False)
        pressed = max(forces.values(), default=0.0) >= CONTACT_PRESSURE_N
        return contacts, forces, pressed

    def _finger_frame(self) -> tuple[np.ndarray, np.ndarray]:
        left = self.runtime.data.xpos[int(self.runtime.model.body("left_finger").id)]
        right = self.runtime.data.xpos[int(self.runtime.model.body("right_finger").id)]
        midpoint = 0.5 * (np.asarray(left, dtype=float) + np.asarray(right, dtype=float))
        closing = np.asarray(right[:2] - left[:2], dtype=float)
        return midpoint, closing

    def _sensors_conflict(self, entity_id: str) -> bool:
        from ..domain import DistanceReading

        def once() -> str:
            hand, _ = self.runtime.hand_pose()
            center, _ = self.runtime.body_pose(entity_id)
            bounds = self.runtime.body_bounds(entity_id)
            origin = np.asarray(hand, dtype=float)
            vector = np.asarray(center, dtype=float) - origin
            distance = float(np.linalg.norm(vector))
            if distance < 1e-6:
                return "fused"
            direction = vector / distance
            rgbd = _ray_aabb_distance(
                origin, direction,
                np.asarray(bounds.minimum, dtype=float),
                np.asarray(bounds.maximum, dtype=float),
            )
            hand_id = int(self.runtime.model.body("hand").id)
            tof_distance, geom_id = self.runtime.raycast(
                tuple(origin), tuple(direction),
                max_distance_m=distance + 0.10,
                body_exclude=hand_id,
            )
            hit = None
            if geom_id is not None:
                body_id = int(self.runtime.model.geom_bodyid[geom_id])
                hit = self.runtime.model.body(body_id).name
            if hit is not None and (
                hit in {"hand", "left_finger", "right_finger"} or hit.startswith("link")
            ):
                tof_distance = None
                hit = None
            reading = DistanceReading(
                tof_distance, tuple(origin), tuple(direction),
                "mujoco_raycast_tof", tof_distance is not None, hit,
            )
            _, status = fuse_distance_measurements(rgbd, reading)
            return status

        if once() != "sensor_conflict_reobserve":
            return False
        return once() == "sensor_conflict_reobserve"

    def _close_until_contact(
        self,
        entity_id: str,
        object_start: np.ndarray,
        width_m: float,
    ) -> tuple[ContactState, dict[str, float], bool, bool]:
        """Keep closing until both fingers hold a settled pressure.

        A one-sided push, or a contact that disappears on the next step, is
        not a finished grasp. The fingers take another small step. The hard
        force limit still stops further compression.
        """
        del width_m
        pushed = False
        hard_force = False
        floor = self._close_floor(0.0)
        step = 18.0

        def guard() -> str | None:
            nonlocal pushed, hard_force
            position, _ = self.runtime.body_pose(entity_id)
            _, forces, _ = self._sensed_pressure(entity_id)
            pushed = float(np.linalg.norm(position[:2] - object_start[:2])) > self.push_limit_m
            hard_force = max(forces.values(), default=0.0) > self.hard_force_n
            if pushed:
                return "object_pushed"
            if hard_force:
                return "hard_force"
            return None

        while not pushed and not hard_force and self.runtime.robot._can_continue():
            if self._settled_bilateral(entity_id):
                break
            current = float(self.runtime.data.ctrl[-1])
            if current <= floor + 1.0:
                break
            target = max(floor, current - step)
            self.runtime.robot.grasp_to(target, duration=0.22, monitor=guard)
            if self.runtime.robot.last_abort_reason in {"object_pushed", "hard_force"}:
                self.runtime.robot.last_abort_reason = None
                break
        if not pushed and not hard_force and self._settled_bilateral(entity_id):
            pushed, hard_force = self._add_force_slowly(entity_id, object_start)
        pinch_contacts, pinch_forces = self.runtime.contacts_with_body(entity_id)
        wide_contacts, wide_forces, _ = self._sensed_pressure(entity_id)
        if pinch_contacts == {"left", "right"}:
            contacts, forces = pinch_contacts, pinch_forces
        else:
            contacts, forces = wide_contacts, wide_forces
        return self._contact_state(contacts), forces, pushed, hard_force

    def _add_force_slowly(self, entity_id: str, object_start: np.ndarray) -> tuple[bool, bool]:
        """A few millimetres tighter after both fingers are pressing."""
        contacts, forces, _ = self._sensed_pressure(entity_id)
        if contacts != {"left", "right"}:
            return False, False
        if min(forces.get("left", 0.0), forces.get("right", 0.0)) < CONTACT_PRESSURE_N:
            return False, False
        pushed = False
        hard_force = False
        step = SLOW_FORCE_STEP_M / 0.08 * GRIPPER_OPEN

        def guard() -> str | None:
            nonlocal pushed, hard_force
            position, _ = self.runtime.body_pose(entity_id)
            _, finger_forces, _ = self._sensed_pressure(entity_id)
            pushed = float(np.linalg.norm(position[:2] - object_start[:2])) > self.push_limit_m
            hard_force = max(finger_forces.values(), default=0.0) > self.hard_force_n
            if pushed:
                return "object_pushed"
            if hard_force:
                return "hard_force"
            return None

        for _ in range(SLOW_FORCE_STEPS):
            before = float(self.runtime.data.ctrl[-1])
            target = max(0.0, before - step)
            if target >= before - 0.4:
                break
            self.runtime.robot.grasp_to(target, duration=0.28, monitor=guard)
            if self.runtime.robot.last_abort_reason in {"object_pushed", "hard_force"}:
                self.runtime.robot.last_abort_reason = None
                break
            still_contacts, still_forces, _ = self._sensed_pressure(entity_id)
            if still_contacts != {"left", "right"}:
                self.runtime.robot.grasp_to(before, duration=0.12)
                break
            if min(
                still_forces.get("left", 0.0), still_forces.get("right", 0.0),
            ) >= SLOW_FORCE_HOLD_N:
                break
        return pushed, hard_force

    def _nudge(self, target_xyz: np.ndarray, yaw_rad: float) -> bool:
        current = self.runtime.arm_q()
        solutions = self.ik.solve(tuple(target_xyz), current, yaw_rad=yaw_rad)
        if not solutions:
            return False
        goal = min(solutions, key=lambda value: float(np.linalg.norm(value - current)))
        return bool(self.runtime.robot.move_to_ruckig(goal, profile="precision"))

    def _miss(
        self,
        state: ContactState,
        forces: dict[str, float],
        entity_id: str,
        object_start: np.ndarray,
        *,
        hard_force: bool,
    ) -> GraspEvidence:
        position, _ = self.runtime.body_pose(entity_id)
        return GraspEvidence(
            state,
            float(forces.get("left", 0.0)),
            float(forces.get("right", 0.0)),
            0.0,
            0.0,
            float(np.linalg.norm(position[:2] - object_start[:2])),
            0.0,
            False,
            hard_force,
        )

    def _micro_lift(
        self,
        candidate: GraspCandidate,
        entity_id: str,
        object_start: np.ndarray,
        hand_rotation_start: np.ndarray,
        relative_start: np.ndarray,
    ) -> GraspEvidence:
        object_before_lift, _ = self.runtime.body_pose(entity_id)
        hand_before_lift, hand_rotation_before = self.runtime.hand_pose()
        # Slip and rise belong to the lift itself. Motion while the fingers
        # were still closing is not evidence that the object failed to follow.
        relative_before_lift = hand_rotation_before.T @ (
            object_before_lift - hand_before_lift
        )
        lift_target = tuple(hand_before_lift + np.array((0.0, 0.0, 0.010)))
        solutions = self.ik.solve(lift_target, self.runtime.arm_q(), yaw_rad=candidate.yaw_rad)
        if solutions:
            current = self.runtime.arm_q()
            self.runtime.robot.move_to_ruckig(
                min(solutions, key=lambda value: float(np.linalg.norm(value - current))),
                profile="precision",
            )
        object_end, object_rotation_end = self.runtime.body_pose(entity_id)
        hand_end, hand_rotation_end = self.runtime.hand_pose()
        relative_end = hand_rotation_end.T @ (object_end - hand_end)
        pinch_contacts, pinch_forces = self.runtime.contacts_with_body(entity_id)
        wide_contacts, wide_forces, _ = self._sensed_pressure(entity_id)
        if pinch_contacts == {"left", "right"}:
            contacts_end, forces_end = pinch_contacts, pinch_forces
        else:
            contacts_end, forces_end = wide_contacts, wide_forces
        z_axis = object_rotation_end[:, 2]
        tilt = math.degrees(math.acos(float(np.clip(z_axis[2], -1.0, 1.0))))
        persisted = contacts_end == {"left", "right"}
        if not persisted:
            self.last_stop_reason = "lift_unstable"
        return GraspEvidence(
            self._contact_state(contacts_end),
            forces_end["left"], forces_end["right"],
            float(object_end[2] - object_before_lift[2]),
            float(np.linalg.norm(relative_end - relative_before_lift)),
            float(np.linalg.norm(object_end[:2] - object_before_lift[:2])),
            tilt,
            persisted,
            max(forces_end.values(), default=0.0) > self.hard_force_n,
        )

    def execute(self, candidate: GraspCandidate, entity_id: str) -> GraspEvidence:
        """Search for bilateral contact on this axis, then lift once."""
        self.adjustments = ()
        self.last_stop_reason = ""
        object_start, _ = self.runtime.body_pose(entity_id)
        _, hand_rotation_start = self.runtime.hand_pose()
        relative_start = hand_rotation_start.T @ (
            object_start - self.runtime.hand_pose()[0]
        )
        if self._sensors_conflict(entity_id):
            self.last_stop_reason = "sensor_conflict"
            return self._miss(
                ContactState.NO_CONTACT, {"left": 0.0, "right": 0.0},
                entity_id, object_start, hard_force=False,
            )

        open_control = self._open_for_width(candidate.expected_width_m)
        self.runtime.robot.grasp_to(open_control, duration=0.25)
        adjustments: list[ContactAdjustment] = []
        state = ContactState.NO_CONTACT
        forces = {"left": 0.0, "right": 0.0}
        hard_force = False
        for _ in range(LOCAL_CORRECTION_LIMIT + 1):
            state, forces, pushed, hard_force = self._close_until_contact(
                entity_id, object_start, candidate.expected_width_m,
            )
            if pushed:
                self.last_stop_reason = "object_pushed"
                self.adjustments = tuple(adjustments)
                return self._miss(state, forces, entity_id, object_start, hard_force=hard_force)
            if state is ContactState.BILATERAL and not hard_force:
                fraction = self.runtime.max_lateral_contact_fraction(entity_id)
                pinch_contacts, _ = self.runtime.contacts_with_body(entity_id)
                on_rim = pinch_contacts != {"left", "right"} or (
                    fraction is not None and fraction > RIM_CONTACT_FRACTION
                )
                if not on_rim or len(adjustments) >= LOCAL_CORRECTION_LIMIT:
                    self.adjustments = tuple(adjustments)
                    self.last_stop_reason = "bilateral"
                    return self._micro_lift(
                        candidate, entity_id, object_start,
                        hand_rotation_start, relative_start,
                    )
                # Pressure on the top edge is not a finished grip. Open, drop
                # the hand a few millimetres, and close again on this attempt.
                self.runtime.robot.grasp_to(open_control, duration=0.2)
                hand, _ = self.runtime.hand_pose()
                target = np.asarray(hand, dtype=float) + np.array(
                    (0.0, 0.0, -LOCAL_CORRECTION_STEP_M)
                )
                if not self._nudge(target, candidate.yaw_rad):
                    self.adjustments = tuple(adjustments)
                    self.last_stop_reason = "bilateral"
                    return self._micro_lift(
                        candidate, entity_id, object_start,
                        hand_rotation_start, relative_start,
                    )
                adjustments.append(ContactAdjustment(
                    "height", (0.0, 0.0, -LOCAL_CORRECTION_STEP_M),
                ))
                continue
            if len(adjustments) >= LOCAL_CORRECTION_LIMIT:
                break
            if hard_force:
                # The force cap stops further squeezing. Open a little so the
                # next close can start again, then shift toward the loaded finger.
                current = float(self.runtime.data.ctrl[-1])
                self.runtime.robot.grasp_to(min(GRIPPER_OPEN, current + 25.0), duration=0.15)
                hard_force = False
            bounds = self.runtime.body_bounds(entity_id)
            center, _ = self.runtime.body_pose(entity_id)
            midpoint, closing = self._finger_frame()
            hand_now, _ = self.runtime.hand_pose()
            pad = midpoint.copy()
            pad[2] = float(hand_now[2]) - 0.147
            adjustment = contact_gap_correction(
                object_center_m=center,
                top_z_m=float(bounds.maximum[2]),
                bottom_z_m=float(bounds.minimum[2]),
                finger_midpoint_m=pad,
                closing_axis_xy=closing,
                contact_state=state,
            )
            if adjustment is None and state in {ContactState.LEFT_ONLY, ContactState.RIGHT_ONLY}:
                axis = np.asarray(closing[:2], dtype=float)
                norm = float(np.linalg.norm(axis))
                if norm > 1e-9:
                    axis /= norm
                    if state is ContactState.LEFT_ONLY:
                        axis = -axis
                    adjustment = ContactAdjustment(
                        "right" if state is ContactState.RIGHT_ONLY else "left",
                        (float(axis[0] * LOCAL_CORRECTION_STEP_M), float(axis[1] * LOCAL_CORRECTION_STEP_M), 0.0),
                    )
            if adjustment is None:
                break
            self.runtime.robot.grasp_to(open_control, duration=0.2)
            hand, _ = self.runtime.hand_pose()
            target = np.asarray(hand, dtype=float) + np.asarray(adjustment.delta_m, dtype=float)
            if not self._nudge(target, candidate.yaw_rad):
                self.last_stop_reason = "ik_unreachable"
                self.adjustments = tuple(adjustments)
                return self._miss(state, forces, entity_id, object_start, hard_force=False)
            adjustments.append(adjustment)
        self.adjustments = tuple(adjustments)
        self.last_stop_reason = "local_exhausted"
        return self._miss(state, forces, entity_id, object_start, hard_force=hard_force)
