"""Joint-space pick-and-place demo for the MuJoCo Franka Panda.

Run on macOS with:
    .venv/bin/mjpython src/joint_control.py

The cube is moved only by MuJoCo contact forces.  Nothing in this script
directly overwrites the cube position, so failed grasps remain visible too.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import os
import time
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
import ruckig

from scene_layout import add_scene_bodies, place_movable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = (
    PROJECT_ROOT / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
)

# Seven arm-joint targets.  The left and right poses differ by pi radians at
# joint 1, so the base turns approximately 180 degrees during the transfer.
HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
LEFT_ABOVE = np.array([-1.5708, -0.2575, 0.0, -2.0464, 0.0, 1.7889, -0.7853])
LEFT_GRASP = np.array([-1.5708, -0.1386, 0.0, -2.3694, 0.0, 2.2309, -0.7853])
RIGHT_ABOVE = np.array([1.5708, -0.2575, 0.0, -2.0464, 0.0, 1.7889, -0.7853])
RIGHT_PLACE = np.array([1.5708, -0.1386, 0.0, -2.3694, 0.0, 2.2309, -0.7853])

GRIPPER_OPEN = 255.0

# Body ids for the one-sided friction rule, keyed by the live MjModel.
_FINGER_CONTACT: dict[int, tuple[int, int, set[int]]] = {}


def _finger_contact_ids(model: mujoco.MjModel) -> tuple[int, int, set[int]]:
    key = id(model)
    cached = _FINGER_CONTACT.get(key)
    if cached is not None:
        return cached
    robot: set[int] = set()
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name.startswith("link") or name in {"hand", "left_finger", "right_finger"}:
            robot.add(body_id)
    cached = (
        int(model.body("left_finger").id),
        int(model.body("right_finger").id),
        robot,
    )
    _FINGER_CONTACT[key] = cached
    return cached


def _one_sided_friction(model: mujoco.MjModel, data: mujoco.MjData) -> dict[int, np.ndarray]:
    """Return saved friction for geoms to clear, or nothing when the jaw is not one-sided."""
    left_id, right_id, robot = _finger_contact_ids(model)
    left_geoms: set[int] = set()
    right_geoms: set[int] = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        if float(contact.dist) > 0.001:
            continue
        geom_a = int(contact.geom1)
        geom_b = int(contact.geom2)
        body_a = int(model.geom_bodyid[geom_a])
        body_b = int(model.geom_bodyid[geom_b])
        for finger, other, finger_geom, object_geom in (
            (body_a, body_b, geom_a, geom_b),
            (body_b, body_a, geom_b, geom_a),
        ):
            if other in robot:
                continue
            if finger == left_id:
                left_geoms.add(finger_geom)
                left_geoms.add(object_geom)
            elif finger == right_id:
                right_geoms.add(finger_geom)
                right_geoms.add(object_geom)
    if bool(left_geoms) == bool(right_geoms):
        return {}
    saved: dict[int, np.ndarray] = {}
    for geom_id in left_geoms or right_geoms:
        saved[geom_id] = model.geom_friction[geom_id].copy()
        model.geom_friction[geom_id] = 0.0
    return saved


def step_with_contact_friction(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Advance one physics step.

    Exactly one finger touching a non-robot body clears friction for that
    step. Both fingers, or neither, leave the stored friction unchanged.
    """
    saved = _one_sided_friction(model, data)
    try:
        mujoco.mj_step(model, data)
    finally:
        for geom_id, friction in saved.items():
            model.geom_friction[geom_id] = friction


GRIPPER_CLOSED = 0.0


def smoothstep(value: float) -> float:
    """Cubic easing so joints start and stop without velocity jumps."""
    value = np.clip(value, 0.0, 1.0)
    return float(value * value * (3.0 - 2.0 * value))


def smootherstep(value: float) -> float:
    """Quintic time scaling with zero velocity and acceleration at both ends."""
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


@dataclass(frozen=True)
class MotionLimits:
    """Joint-space limits used to retime a geometric move.

    These are deliberately controller-side limits.  A task planner may ask
    for ``payload`` mode, but it cannot bypass the velocity/acceleration cap.
    The constants 1.875 and 5.774 are the peak first and second derivatives
    of the quintic scaling polynomial.
    """

    max_velocity: float
    max_acceleration: float
    max_jerk: float
    minimum_duration: float
    settle_duration: float


MOTION_LIMITS = {
    "normal": MotionLimits(1.60, 4.00, 12.0, 0.10, 0.08),
    "precision": MotionLimits(0.70, 1.50, 5.0, 0.20, 0.08),
    # A light friction-held payload needs much gentler starts and turns than
    # the empty arm.  This is the executable meaning of the LLM's "fragile"
    # constraint; the model never chooses raw joint velocities itself.
    # Extremely slow stop-and-go motion is also harmful: a compliant friction
    # grasp creeps while dozens of route vertices settle.  These limits remain
    # well below the empty-arm profile, while Ruckig's jerk cap removes the
    # acceleration shocks that caused the original green-block drop.
    "payload": MotionLimits(0.28, 0.55, 1.8, 0.35, 0.04),
    # Selected by the LLM only as a symbolic "fragile" policy.  The numeric
    # limits remain local and immutable from model output.
    "payload_fragile": MotionLimits(0.20, 0.35, 1.2, 0.45, 0.05),
}


def scaled_motion_limits(profile: str, scale: float) -> MotionLimits:
    """Apply a local tightening factor; callers can never raise a profile cap."""
    if profile not in MOTION_LIMITS:
        raise ValueError(f"未知运动模式: {profile}")
    if not 0.0 < float(scale) <= 1.0:
        raise ValueError("motion_scale 必须位于 (0, 1]")
    base = MOTION_LIMITS[profile]
    factor = float(scale)
    return MotionLimits(
        max_velocity=base.max_velocity * factor,
        max_acceleration=base.max_acceleration * factor,
        max_jerk=base.max_jerk * factor,
        minimum_duration=base.minimum_duration,
        settle_duration=base.settle_duration,
    )


def retime_duration(
    delta: np.ndarray,
    requested: float,
    limits: MotionLimits,
) -> float:
    """Return a duration satisfying quintic velocity and acceleration caps."""
    span = float(np.max(np.abs(np.asarray(delta, dtype=float))))
    velocity_time = 1.875 * span / max(limits.max_velocity, 1e-6)
    acceleration_time = np.sqrt(5.774 * span / max(limits.max_acceleration, 1e-6))
    return float(max(requested, limits.minimum_duration, velocity_time, acceleration_time))


class PandaController:
    """Franka Panda 的统一控制接口。

    上层只需要调用 move_to() / grasp() / release() / pick() /
    place()，不需要知道 data.ctrl 和 mj_step() 的底层细节。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        viewer=None,
        speed: float = 1.0,
        playback: dict[str, bool] | None = None,
        data_lock=None,
        one_sided_friction: bool = False,
    ) -> None:
        self.model = model
        self.data = data
        self.viewer = viewer
        self.speed = speed
        # A GUI may render ``data`` on its main thread while this controller
        # advances physics on a worker thread.  The normal command-line demos
        # leave this as None and therefore pay no locking overhead.
        self.data_lock = data_lock
        self.one_sided_friction = one_sided_friction
        self.playback = playback if playback is not None else {
            "paused": False,
            "restart": False,
            "loop": False,
        }
        self.last_abort_reason: str | None = None
        # Set by the execution layer when an LLM checkpoint asks for one
        # whitelisted adjustment; consumed by task_runner.
        self.last_checkpoint_adjustment: str | None = None
        # Last validated hand path summary (PlanBrief) and the predicted
        # bilateral grasp force used in it.  Descriptive only.
        self.last_plan_brief: dict | None = None
        self.plan_peak_force_hint_n: float = 0.0

    def _step_physics(self) -> None:
        if self.one_sided_friction:
            step_with_contact_friction(self.model, self.data)
        else:
            mujoco.mj_step(self.model, self.data)

    def _can_continue(self) -> bool:
        """检查 Viewer 和重播指令，决定当前动作是否继续。"""
        viewer_open = self.viewer is None or self.viewer.is_running()
        return viewer_open and not self.playback["restart"]

    def _move_control(
        self,
        target_ctrl: np.ndarray,
        duration: float,
        *,
        profile: str = "normal",
        monitor: Callable[[], str | None] | None = None,
    ) -> bool:
        """底层发动机：把当前控制值平滑移动到 target_ctrl。"""
        start_ctrl = self.data.ctrl.copy()
        try:
            limits = MOTION_LIMITS[profile]
        except KeyError as error:
            raise ValueError(f"未知运动模式: {profile}") from error
        # Gripper commands use a 0..255 actuator scale and are already given
        # an explicit duration.  Arm retiming must only look at the 7 joints.
        arm_delta = target_ctrl[:7] - start_ctrl[:7]
        duration = retime_duration(arm_delta, duration, limits)
        elapsed = 0.0
        self.last_abort_reason = None

        # duration 表示仿真中的动作时长。speed 只改变观看速度，
        # 不改变机器人的物理加速度和抓取结果。
        while elapsed < duration:
            if not self._can_continue():
                return False

            if self.playback["paused"]:
                if self.viewer is not None:
                    self.viewer.sync()
                time.sleep(0.02)
                continue

            step_started = time.perf_counter()
            alpha = smootherstep(elapsed / duration)
            lock_context = self.data_lock if self.data_lock is not None else nullcontext()
            with lock_context:
                self.data.ctrl[:] = start_ctrl + alpha * (target_ctrl - start_ctrl)
                self._step_physics()
                reason = None if monitor is None else monitor()
            if reason is not None:
                self.last_abort_reason = str(reason)
                # Freeze the position targets at the measured joint state.
                # This is the local controller's immediate safe stop; the
                # higher-level recovery code decides whether to reobserve.
                addresses = [
                    int(self.model.joint(f"joint{index}").qposadr[0])
                    for index in range(1, 8)
                ]
                self.data.ctrl[:7] = self.data.qpos[addresses]
                return False
            elapsed += self.model.opt.timestep

            if self.viewer is not None:
                self.viewer.sync()
                remaining = self.model.opt.timestep / self.speed - (
                    time.perf_counter() - step_started
                )
                if remaining > 0:
                    time.sleep(remaining)

        # The interpolation loop ends just before alpha reaches exactly 1.0.
        # Merely assigning the last control value does not advance physics, so
        # the real joints used to lag behind the planned pose.  That Cartesian
        # error was large enough for an open finger to push a 60 mm box.  Give
        # the position servos a short settling window at the exact target.
        settle_elapsed = 0.0
        while settle_elapsed < limits.settle_duration:
            if not self._can_continue():
                return False
            step_started = time.perf_counter()
            lock_context = self.data_lock if self.data_lock is not None else nullcontext()
            with lock_context:
                self.data.ctrl[:] = target_ctrl
                self._step_physics()
                reason = None if monitor is None else monitor()
            if reason is not None:
                self.last_abort_reason = str(reason)
                addresses = [
                    int(self.model.joint(f"joint{index}").qposadr[0])
                    for index in range(1, 8)
                ]
                self.data.ctrl[:7] = self.data.qpos[addresses]
                return False
            settle_elapsed += self.model.opt.timestep
            if self.viewer is not None:
                self.viewer.sync()
                remaining = self.model.opt.timestep / self.speed - (
                    time.perf_counter() - step_started
                )
                if remaining > 0:
                    time.sleep(remaining)
        return True

    def move_to(
        self,
        arm_target: np.ndarray,
        duration: float = 2.0,
        *,
        profile: str = "normal",
        monitor: Callable[[], str | None] | None = None,
    ) -> bool:
        """移动七个手臂关节，同时保持当前夹爪状态。"""
        if arm_target.shape != (7,):
            raise ValueError("arm_target 必须包含 7 个关节目标")
        current_gripper = self.data.ctrl[-1]
        target_ctrl = np.concatenate((arm_target, [current_gripper]))
        return self._move_control(
            target_ctrl, duration, profile=profile, monitor=monitor
        )

    def move_to_ruckig(
        self,
        arm_target: np.ndarray,
        duration: float = 0.0,
        *,
        profile: str = "normal",
        monitor: Callable[[], str | None] | None = None,
        target_velocity: np.ndarray | None = None,
        settle: bool = True,
        motion_scale: float = 1.0,
    ) -> bool:
        """Execute one joint-space segment with jerk-limited Ruckig timing."""
        arm_target = np.asarray(arm_target, dtype=float)
        if arm_target.shape != (7,):
            raise ValueError("arm_target 必须包含 7 个关节目标")
        try:
            limits = scaled_motion_limits(profile, motion_scale)
        except KeyError as error:
            raise ValueError(f"未知运动模式: {profile}") from error
        qpos_addresses = [
            int(self.model.joint(f"joint{index}").qposadr[0])
            for index in range(1, 8)
        ]
        dof_addresses = [
            int(self.model.joint(f"joint{index}").dofadr[0])
            for index in range(1, 8)
        ]
        current_position = self.data.qpos[qpos_addresses].copy()
        current_velocity = self.data.qvel[dof_addresses].copy()
        current_acceleration = self.data.qacc[dof_addresses].copy()
        inp = ruckig.InputParameter(7)
        trajectory = ruckig.Trajectory(7)
        inp.current_position = current_position.tolist()
        inp.current_velocity = current_velocity.tolist()
        # Preserve measured qdot/qddot across route vertices.  Resetting
        # acceleration to zero for every waypoint creates an artificial jerk
        # impulse even when a non-zero via velocity is supplied.
        inp.target_position = arm_target.tolist()
        if target_velocity is None:
            target_velocity = np.zeros(7, dtype=float)
        else:
            target_velocity = np.asarray(target_velocity, dtype=float)
            if target_velocity.shape != (7,):
                raise ValueError("target_velocity 必须包含 7 个关节目标")
        inp.target_acceleration = [0.0] * 7
        velocity_limits = np.full(7, limits.max_velocity, dtype=float)
        acceleration_limits = np.full(7, limits.max_acceleration, dtype=float)
        jerk_limits = np.full(7, limits.max_jerk, dtype=float)
        if profile.startswith("payload"):
            # Joint 5-7 dominate hand angular motion.  Their lower limits keep
            # a payload stable even when end-effector translation is small.
            velocity_limits[4:] *= 0.65
            acceleration_limits[4:] *= 0.55
            jerk_limits[4:] *= 0.50
        # A P-3 slowdown can tighten limits between route segments.  Clamp a
        # previously computed pass-through velocity to the newly effective
        # cap so Ruckig never sees an internally inconsistent request.
        target_velocity = np.clip(
            target_velocity,
            -0.95 * velocity_limits,
            0.95 * velocity_limits,
        )
        inp.target_velocity = target_velocity.tolist()
        current_acceleration = np.clip(
            current_acceleration,
            -0.95 * acceleration_limits,
            0.95 * acceleration_limits,
        )
        inp.current_acceleration = current_acceleration.tolist()
        inp.max_velocity = velocity_limits.tolist()
        inp.max_acceleration = acceleration_limits.tolist()
        inp.max_jerk = jerk_limits.tolist()
        inp.minimum_duration = max(float(duration), limits.minimum_duration)
        result = ruckig.Ruckig(7).calculate(inp, trajectory)
        if result not in {ruckig.Result.Working, ruckig.Result.Finished}:
            self.last_abort_reason = f"ruckig:{result}"
            return False

        self.last_abort_reason = None
        elapsed = 0.0
        while elapsed < trajectory.duration:
            if not self._can_continue():
                return False
            if self.playback["paused"]:
                if self.viewer is not None:
                    self.viewer.sync()
                time.sleep(0.02)
                continue
            started = time.perf_counter()
            position, _, _ = trajectory.at_time(elapsed)
            lock_context = self.data_lock if self.data_lock is not None else nullcontext()
            with lock_context:
                self.data.ctrl[:7] = position
                # Keep the live gripper command.  A force-feedback monitor may
                # apply a small bounded preload correction while the arm is
                # moving; restoring the segment's initial aperture here would
                # silently undo that correction on the next simulation step.
                self._step_physics()
                reason = None if monitor is None else monitor()
            if reason is not None:
                self.last_abort_reason = str(reason)
                self.data.ctrl[:7] = self.data.qpos[qpos_addresses]
                return False
            elapsed += float(self.model.opt.timestep)
            if self.viewer is not None:
                self.viewer.sync()
                remaining = self.model.opt.timestep / self.speed - (
                    time.perf_counter() - started
                )
                if remaining > 0:
                    time.sleep(remaining)

        if not settle:
            # Feed the endpoint to the servo once.  The next segment starts
            # from measured q/qdot, preserving the non-zero via velocity.
            lock_context = self.data_lock if self.data_lock is not None else nullcontext()
            with lock_context:
                self.data.ctrl[:7] = arm_target
                self._step_physics()
                reason = None if monitor is None else monitor()
            if reason is not None:
                self.last_abort_reason = str(reason)
                self.data.ctrl[:7] = self.data.qpos[qpos_addresses]
                return False
            return True

        # Ruckig already reaches the final target with zero velocity and zero
        # acceleration.  Hold the exact endpoint only for the configured
        # servo settling window.  Calling _move_control here used to retime a
        # nominal 60 ms hold to the profile's 350 ms minimum for *every*
        # waypoint, adding more than eight seconds to a 24-point carry route.
        settle_elapsed = 0.0
        while settle_elapsed < limits.settle_duration:
            if not self._can_continue():
                return False
            started = time.perf_counter()
            lock_context = self.data_lock if self.data_lock is not None else nullcontext()
            with lock_context:
                self.data.ctrl[:7] = arm_target
                self._step_physics()
                reason = None if monitor is None else monitor()
            if reason is not None:
                self.last_abort_reason = str(reason)
                self.data.ctrl[:7] = self.data.qpos[qpos_addresses]
                return False
            settle_elapsed += float(self.model.opt.timestep)
            if self.viewer is not None:
                self.viewer.sync()
                remaining = self.model.opt.timestep / self.speed - (
                    time.perf_counter() - started
                )
                if remaining > 0:
                    time.sleep(remaining)
        return True

    def grasp(self, duration: float = 1.0) -> bool:
        """关闭夹爪；方块只由真实接触力和摩擦力抓住。"""
        return self.grasp_to(GRIPPER_CLOSED, duration)

    def grasp_to(
        self,
        gripper_control: float,
        duration: float = 1.0,
        *,
        monitor: Callable[[], str | None] | None = None,
    ) -> bool:
        """Close to a calibrated aperture instead of always crushing to zero.

        actuator8 uses 0..255 for closed..open.  Object-specific callers can
        stop near the measured object width, which reduces penetration while
        still generating enough normal force for friction.
        """
        if not 0.0 <= gripper_control <= GRIPPER_OPEN:
            raise ValueError("gripper_control 必须位于 0..255")
        target_ctrl = self.data.ctrl.copy()
        target_ctrl[-1] = gripper_control
        return self._move_control(target_ctrl, duration, monitor=monitor)

    def release(self, duration: float = 1.0) -> bool:
        """打开夹爪，让方块在重力作用下落到平台。"""
        target_ctrl = self.data.ctrl.copy()
        target_ctrl[-1] = GRIPPER_OPEN
        return self._move_control(target_ctrl, duration)

    def hold(
        self,
        duration: float = 1.0,
        *,
        monitor: Callable[[], str | None] | None = None,
    ) -> bool:
        """保持当前控制目标，让物理系统继续演化。"""
        return self._move_control(
            self.data.ctrl.copy(), duration, monitor=monitor
        )

    def pick(self, above_pose: np.ndarray, grasp_pose: np.ndarray) -> bool:
        """执行“到上方 → 下降 → 抓取 → 抬起”。"""
        print("  → 移动到方块上方")
        if not self.move_to(above_pose, duration=2.0):
            return False
        print("  → 下降")
        if not self.move_to(grasp_pose, duration=1.5):
            return False
        print("  → 闭合夹爪（纯物理接触）")
        if not self.grasp(duration=1.0):
            return False
        print("  → 抬起")
        return self.move_to(above_pose, duration=1.5)

    def place(self, above_pose: np.ndarray, place_pose: np.ndarray) -> bool:
        """执行“搬运到上方 → 下降 → 释放 → 离开”。"""
        print("  → 搬运到右侧上方")
        if not self.move_to(above_pose, duration=3.0):
            return False
        print("  → 下降到放置位置")
        if not self.move_to(place_pose, duration=1.5):
            return False
        print("  → 打开夹爪并释放")
        if not self.release(duration=1.0):
            return False
        print("  → 离开方块")
        return self.move_to(above_pose, duration=1.5)


def create_model(spec_hook: Callable | None = None) -> mujoco.MjModel:
    """Load the scene, then add pedestals, a cube, and a wrist RGB camera."""
    spec = mujoco.MjSpec.from_file(str(SCENE_PATH))

    # Use a neutral floor so a color detector can distinguish the blue box.
    # The stock Menagerie blue checkerboard merged into the blue HSV mask and
    # was previously mistaken for the requested object.
    ground_texture = spec.texture("groundplane")
    ground_texture.rgb1 = (0.18, 0.19, 0.21)
    ground_texture.rgb2 = (0.30, 0.31, 0.33)
    ground_texture.markrgb = (0.55, 0.55, 0.58)

    add_scene_bodies(spec)

    # Wrist camera: it is a child of the hand, so it follows every arm
    # motion.  Panda hand local +Z points toward the grasp area in our poses;
    # a camera looks along local -Z, hence its +Z axis is set to hand -Z.
    hand = spec.body("hand")
    hand.add_camera(
        name="gripper_camera",
        pos=(0.0, 0.0, 0.02),
        zaxis=(0.0, 0.0, -1.0),
        mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
        # A wider wrist view finds target, destination and obstacles in fewer
        # views.  Local RGB-D keeps the full 640x480 frame; only a compressed
        # two-view contact sheet is sent to the VLM.
        fovy=float(os.getenv("WRIST_CAMERA_FOVY_DEG", "95")),
    )
    # The stock Panda model approximates each fingertip with several small
    # collision boxes covering only ~17 mm vertically.  Tall round objects can
    # therefore slide out of contact after a few centimetres even though the
    # visible black pad is much longer.  Add one continuous rubber pad per
    # finger.  Contact stays physical: no weld and no attach.
    for side in ("left_finger", "right_finger"):
        spec.body(side).add_geom(
            name=f"{side}_grip_pad",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            # 58 mm usable vertical face, matching the visible Panda pad.
            # The former 48 mm proxy lost one side of an 80 mm box after only
            # about 10 mm of settling on the long cross-base transport.
            size=(0.0085, 0.0060, 0.0290),
            # Further toward the fingertip, and thicker toward the object, so
            # the rubber can compress around a cup instead of only touching a line.
            pos=(0.0, 0.0080, 0.0500),
            friction=(5.0, 0.01, 0.001),
            contype=1,
            conaffinity=1,
            group=3,
        )
    # Sensor/benchmark setup is supplied by the simulation boundary. Legacy
    # callers retain their original scene and physics without a hook.
    if spec_hook is not None:
        spec_hook(spec)
    model = spec.compile()
    # Parallel-jaw grasps use high sliding friction.  The default pyramidal
    # approximation can exhibit visible contact creep even when the Coulomb
    # bound is comfortably above the payload weight.  The elliptic cone plus
    # MuJoCo's bounded no-slip post-processing keeps this a physical contact
    # grasp while avoiding that numerical drift (no weld/equality shortcut).
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.noslip_iterations = 8
    # Rubber pads plus a stiffer finger spring.  The stock tendon stiffness is
    # only 100 N/m, so a few millimetres of squeeze makes about 0.2 N and the
    # object slips as soon as the wrist accelerates.  A 650 N/m spring keeps
    # the same 0..255 aperture mapping and holds a light box without slamming
    # the fingers shut.  The pad solref is the compliant rubber, not a weld.
    actuator_id = model.actuator("actuator8").id
    finger_stiffness = 650.0
    model.actuator_gainprm[actuator_id, 0] = 0.04 * finger_stiffness / 255.0
    model.actuator_biasprm[actuator_id, 0] = 0.0
    model.actuator_biasprm[actuator_id, 1] = -finger_stiffness
    model.actuator_biasprm[actuator_id, 2] = -35.0
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        ) or ""
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if body_name in {"left_finger", "right_finger"} and model.geom_contype[geom_id]:
            model.geom_friction[geom_id, 0] = 5.0
        if geom_name.endswith("_grip_pad"):
            # condim 6 adds a little rolling resistance so a round cup does
            # not twist out of a flat pad while the wrist starts moving.
            model.geom_condim[geom_id] = 6
            model.geom_friction[geom_id] = (5.0, 0.20, 0.05)
            model.geom_solref[geom_id] = (0.02, 1.0)
            model.geom_solimp[geom_id] = (0.90, 0.99, 0.001, 0.5, 2.0)
    return model


def run_demo(
    headless: bool = False, speed: float = 1.0, auto_loop: bool = True
) -> tuple[np.ndarray, float]:
    model = create_model()
    data = mujoco.MjData(model)
    cube_qpos = int(model.joint("cube_joint").qposadr[0])
    controls = {"paused": False, "restart": False, "loop": auto_loop}

    def reset_scene() -> None:
        """Restore the Panda, then scatter pickable objects."""
        mujoco.mj_resetDataKeyframe(model, data, 0)
        place_movable(model, data)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    def key_callback(keycode: int) -> None:
        """Keyboard controls received by the MuJoCo Viewer."""
        if keycode == 32:  # Space
            controls["paused"] = not controls["paused"]
            print("⏸ 暂停" if controls["paused"] else "▶ 继续")
        elif keycode in (ord("R"), ord("r")):
            controls["restart"] = True
            controls["paused"] = False
            print("↻ 从头重播")
        elif keycode in (ord("L"), ord("l")):
            controls["loop"] = not controls["loop"]
            print("循环播放:", "开" if controls["loop"] else "关")

    reset_scene()
    viewer_context: Callable[..., object]

    if headless:
        from contextlib import nullcontext

        viewer_context = lambda: nullcontext(None)
    else:
        from mujoco import viewer as mj_viewer

        viewer_context = lambda: mj_viewer.launch_passive(
            model, data, key_callback=key_callback
        )

    with viewer_context() as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.38)
            viewer.cam.distance = 1.65
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

        if viewer is not None:
            print("控制：Space 暂停/继续 | R 重播 | L 开关循环 | 关闭窗口退出")

        robot = PandaController(
            model=model,
            data=data,
            viewer=viewer,
            speed=speed,
            playback=controls,
        )
        final_cube_position = data.qpos[cube_qpos : cube_qpos + 3].copy()
        while viewer is None or viewer.is_running():
            controls["restart"] = False
            reset_scene()

            print("→ 回到初始姿态")
            completed = robot.move_to(HOME, duration=1.0)

            if completed:
                print("→ 抓取左侧方块")
                completed = robot.pick(LEFT_ABOVE, LEFT_GRASP)

            if completed:
                print("→ 放置到右侧对称位置")
                completed = robot.place(RIGHT_ABOVE, RIGHT_PLACE)

            if completed:
                print("→ 返回初始姿态")
                completed = robot.move_to(HOME, duration=2.0)

            if not completed:
                if viewer is not None and not viewer.is_running():
                    return data.qpos[cube_qpos : cube_qpos + 3].copy(), data.time
                continue

            # 主程序不再直接调用 mj_step()，统一通过 Controller。
            if not robot.hold(duration=1.0):
                if viewer is not None and not viewer.is_running():
                    return data.qpos[cube_qpos : cube_qpos + 3].copy(), data.time
                continue

            final_cube_position = data.qpos[cube_qpos : cube_qpos + 3].copy()
            print("演示完成。方块最终位置:", np.round(final_cube_position, 3))

            if viewer is None:
                return final_cube_position, data.time
            if controls["loop"]:
                time.sleep(0.8)
                continue

            # Once-mode does not close the Viewer. Wait for R or L.
            print("画面已保留：按 R 重播，或按 L 打开自动循环。")
            while viewer.is_running() and not controls["restart"] and not controls["loop"]:
                viewer.sync()
                time.sleep(0.02)

        return final_cube_position, data.time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="不打开 Viewer，用于自动测试")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率")
    parser.add_argument("--once", action="store_true", help="播放一次后保留画面，不自动循环")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed 必须大于 0")
    run_demo(headless=args.headless, speed=args.speed, auto_loop=not args.once)


if __name__ == "__main__":
    main()
