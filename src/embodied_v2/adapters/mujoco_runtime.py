"""Thin V2 adapter over the reusable Panda/MuJoCo low-level infrastructure."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import math
import threading
import time
from typing import Callable, Iterable

import mujoco
import numpy as np

from joint_control import GRIPPER_OPEN, PandaController, create_model
from scene_layout import place_movable

from ..domain import AABB, JointTrajectory, Vec3, vec3


ARM_JOINTS = tuple(f"joint{index}" for index in range(1, 8))
FINGER_BODIES = frozenset({"left_finger", "right_finger"})


class _FrameEmitter:
    """Viewer-compatible bridge used by the low-level controller in web mode."""

    def __init__(self, runtime: "MujocoRuntime") -> None:
        self.runtime = runtime

    def is_running(self) -> bool:
        return not self.runtime.cancel_event.is_set()

    def sync(self) -> None:
        self.runtime.emit_frame()


@dataclass(frozen=True)
class SimulationSnapshot:
    """Thread-safe numerical state handed to a renderer on another thread."""

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray


@dataclass
class MujocoRuntime(AbstractContextManager["MujocoRuntime"]):
    """Owns simulation state without importing V1 orchestration code."""

    width: int = 640
    height: int = 480
    camera_name: str = "gripper_camera"
    speed: float = 1.0

    def __post_init__(self) -> None:
        self.model = create_model()
        self.data = mujoco.MjData(self.model)
        self._configure_v2_physics()
        self.data_lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.renderer = mujoco.Renderer(
            self.model, height=self.height, width=self.width
        )
        self.robot = PandaController(
            self.model,
            self.data,
            speed=self.speed,
            data_lock=self.data_lock,
        )
        self._frame_sink = None
        self._state_sink: Callable[[SimulationSnapshot], None] | None = None
        self._pace_execution = False
        self._last_frame_wall_s = 0.0
        self._frame_interval_s = 0.10
        self._scene_camera = mujoco.MjvCamera()
        self._scene_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._scene_camera.lookat[:] = (0.0, 0.0, 0.20)
        self._scene_camera.distance = 1.55
        self._scene_camera.azimuth = 135.0
        self._scene_camera.elevation = -30.0
        self.reset()

    @staticmethod
    def _geom_volume(geom_type: int, size: np.ndarray) -> float:
        """Analytic collision volume used for conservative payload mass floors."""
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            return float(8.0 * size[0] * size[1] * size[2])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return float(4.0 * math.pi * size[0] ** 3 / 3.0)
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            return float(math.pi * size[0] ** 2 * 2.0 * size[1])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            return float(
                math.pi * size[0] ** 2 * 2.0 * size[1]
                + 4.0 * math.pi * size[0] ** 3 / 3.0
            )
        if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            return float(4.0 * math.pi * size[0] * size[1] * size[2] / 3.0)
        return 0.0

    def _configure_v2_physics(self) -> None:
        """Apply V2-only physical calibration without changing the V1 model."""
        # The long tactile proxies are part of collision reality and must also
        # be visible. Their old coefficient (5.0) made a 21 g prop behave as if
        # glued to a visually empty gripper. Use a plausible rubber contact and
        # render the exact same volume as a dark finger pad.
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            ) or ""
            if not name.endswith("_grip_pad"):
                continue
            self.model.geom_group[geom_id] = 0
            self.model.geom_rgba[geom_id] = (0.08, 0.10, 0.10, 1.0)
            self.model.geom_condim[geom_id] = 4
            # 25 mm torsional length approximates the finite rubber-pad
            # contact patch. It resists rotation about the pinch normal while
            # preserving ordinary Coulomb slip and fully dynamic payloads.
            self.model.geom_friction[geom_id] = (1.2, 0.025, 0.001)

        # Free bodies get a geometry-derived lower mass bound. This keeps tiny
        # test props possible while preventing large, solid-looking objects
        # from behaving like nearly massless shells.
        density_floor_kg_m3 = 350.0
        for body_id in range(1, self.model.nbody):
            if int(self.model.body_jntnum[body_id]) != 1:
                continue
            joint_id = int(self.model.body_jntadr[body_id])
            if joint_id < 0 or int(self.model.jnt_type[joint_id]) != int(
                mujoco.mjtJoint.mjJNT_FREE
            ):
                continue
            volume = sum(
                self._geom_volume(
                    int(self.model.geom_type[geom_id]),
                    self.model.geom_size[geom_id],
                )
                for geom_id in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) == body_id
                and int(self.model.geom_contype[geom_id]) != 0
            )
            current_mass = float(self.model.body_mass[body_id])
            target_mass = max(current_mass, min(1.0, volume * density_floor_kg_m3))
            if target_mass <= current_mass + 1e-9:
                continue
            scale = target_mass / max(current_mass, 1e-9)
            self.model.body_mass[body_id] = target_mass
            self.model.body_inertia[body_id] *= scale
        mujoco.mj_setConst(self.model, self.data)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        renderer = getattr(self, "renderer", None)
        if renderer is not None:
            renderer.close()
            self.renderer = None  # type: ignore[assignment]

    def reset(self) -> None:
        self.cancel_event.clear()
        with self.data_lock:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            place_movable(self.model, self.data)
            self.data.qvel[:] = 0.0
            self.data.ctrl[-1] = GRIPPER_OPEN
            mujoco.mj_forward(self.model, self.data)

    def request_stop(self) -> None:
        self.cancel_event.set()

    def render_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        with self.data_lock:
            self.renderer.update_scene(self.data, camera=self.camera_name)
            rgb = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
        return rgb, depth

    def render_third_person(self) -> np.ndarray:
        """Render the complete multi-object tabletop from an operator view."""
        with self.data_lock:
            self.renderer.update_scene(self.data, camera=self._scene_camera)
            return self.renderer.render().copy()

    def attach_frame_sink(
        self,
        sink,
        *,
        playback_speed: float = 3.0,
        frame_interval_s: float = 0.10,
    ) -> None:
        """Enable paced third-person streaming for UI execution."""
        self._frame_sink = sink
        self._pace_execution = True
        self._frame_interval_s = max(0.04, float(frame_interval_s))
        self.robot.speed = max(0.25, float(playback_speed))
        self.robot.viewer = _FrameEmitter(self)
        self.emit_frame(force=True)

    def attach_state_sink(
        self,
        sink: Callable[[SimulationSnapshot], None],
        *,
        playback_speed: float = 2.5,
        frame_interval_s: float = 0.04,
    ) -> None:
        """Stream copied MuJoCo state without crossing OpenGL thread boundaries."""
        self._state_sink = sink
        self._pace_execution = True
        self._frame_interval_s = max(0.02, float(frame_interval_s))
        self.robot.speed = max(0.25, float(playback_speed))
        self.robot.viewer = _FrameEmitter(self)
        self.emit_frame(force=True)

    def enable_realtime_playback(self, playback_speed: float = 2.5) -> None:
        """Pace physics for a native viewer without duplicating rendering."""
        self._pace_execution = True
        self.robot.speed = max(0.25, float(playback_speed))
        self.robot.viewer = _FrameEmitter(self)

    def state_snapshot(self) -> SimulationSnapshot:
        """Copy all state required to reproduce the visible simulation scene."""
        with self.data_lock:
            return SimulationSnapshot(
                qpos=self.data.qpos.copy(),
                qvel=self.data.qvel.copy(),
                ctrl=self.data.ctrl.copy(),
                mocap_pos=self.data.mocap_pos.copy(),
                mocap_quat=self.data.mocap_quat.copy(),
            )

    def emit_frame(self, *, force: bool = False) -> None:
        if self._frame_sink is None and self._state_sink is None:
            return
        now = time.monotonic()
        if not force and now - self._last_frame_wall_s < self._frame_interval_s:
            return
        self._last_frame_wall_s = now
        if self._state_sink is not None:
            self._state_sink(self.state_snapshot())
        if self._frame_sink is not None:
            self._frame_sink(self.render_third_person())

    def arm_q(self) -> np.ndarray:
        return np.array([
            self.data.qpos[int(self.model.joint(name).qposadr[0])]
            for name in ARM_JOINTS
        ], dtype=float)

    def hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        body = self.model.body("hand").id
        return (
            self.data.xpos[body].copy(),
            self.data.xmat[body].reshape(3, 3).copy(),
        )

    def body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        body = self.model.body(body_name).id
        return (
            self.data.xpos[body].copy(),
            self.data.xmat[body].reshape(3, 3).copy(),
        )

    def movable_body_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for body_id in range(1, self.model.nbody):
            if int(self.model.body_jntnum[body_id]) != 1:
                continue
            joint_id = int(self.model.body_jntadr[body_id])
            if joint_id < 0 or int(self.model.jnt_type[joint_id]) != int(
                mujoco.mjtJoint.mjJNT_FREE
            ):
                continue
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if name:
                names.append(name)
        return tuple(names)

    def body_geom_ids(self, body_name: str) -> tuple[int, ...]:
        body_id = int(self.model.body(body_name).id)
        return tuple(
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) == body_id
        )

    def body_bounds(self, body_name: str) -> AABB:
        geom_ids = self.body_geom_ids(body_name)
        if not geom_ids:
            center = vec3(self.data.xpos[self.model.body(body_name).id])
            return AABB(center, center)
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for geom_id in geom_ids:
            center = self.data.geom_xpos[geom_id]
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            geom_type = int(self.model.geom_type[geom_id])
            size = self.model.geom_size[geom_id]
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                local_half = np.asarray(size[:3], dtype=float)
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                local_half = np.array((size[0], size[0], size[1]), dtype=float)
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                local_half = np.array(
                    (size[0], size[0], size[1] + size[0]), dtype=float
                )
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                local_half = np.full(3, size[0], dtype=float)
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue
            else:
                radius = float(self.model.geom_rbound[geom_id])
                local_half = np.full(3, radius, dtype=float)
            world_half = np.abs(rotation) @ local_half
            lows.append(center - world_half)
            highs.append(center + world_half)
        if not lows:
            center = self.data.xpos[self.model.body(body_name).id].copy()
            return AABB(vec3(center), vec3(center))
        return AABB(vec3(np.min(lows, axis=0)), vec3(np.max(highs, axis=0)))

    def body_rgba(self, body_name: str) -> tuple[float, float, float, float]:
        geoms = self.body_geom_ids(body_name)
        if not geoms:
            return (0.5, 0.5, 0.5, 1.0)
        values = self.model.geom_rgba[list(geoms)]
        rgba = np.mean(values, axis=0)
        return tuple(float(value) for value in rgba)  # type: ignore[return-value]

    def primary_geometry(self, body_name: str) -> str:
        geoms = self.body_geom_ids(body_name)
        if not geoms:
            return "unknown"
        geom_id = max(
            geoms,
            key=lambda item: float(self.model.geom_rbound[item]),
        )
        geom_type = int(self.model.geom_type[geom_id])
        return {
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
            int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
        }.get(geom_type, "mesh")

    def contacts_with_body(
        self,
        body_name: str,
        *,
        pinch_band: bool = True,
    ) -> tuple[set[str], dict[str, float]]:
        target_id = int(self.model.body(body_name).id)
        target_bounds = self.body_bounds(body_name)
        target_height = max(float(target_bounds.size[2]), 1e-6)
        left_position = self.data.xpos[int(self.model.body("left_finger").id)]
        right_position = self.data.xpos[int(self.model.body("right_finger").id)]
        closing_axis = np.asarray(right_position - left_position, dtype=float)
        closing_axis /= max(float(np.linalg.norm(closing_axis)), 1e-9)
        valid_points: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        valid_normals: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        forces = {"left": 0.0, "right": 0.0}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if target_id not in {body1, body2}:
                continue
            other = body2 if body1 == target_id else body1
            other_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, other
            ) or ""
            side = "left" if other_name == "left_finger" else (
                "right" if other_name == "right_finger" else None
            )
            if side is None:
                continue
            # MuJoCo's contact normal points from geom1 toward geom2. Orient it
            # from the payload toward the finger, then reject rim/top contacts
            # and same-direction contacts that cannot form an opposing pinch.
            normal = np.asarray(contact.frame[:3], dtype=float).copy()
            if body2 == target_id:
                normal *= -1.0
            point = np.asarray(contact.pos, dtype=float).copy()
            height_fraction = float(
                (point[2] - target_bounds.minimum[2]) / target_height
            )
            signed_lateral = float(normal @ closing_axis)
            lateral_normal = abs(signed_lateral) >= 0.75
            faces_finger = (
                signed_lateral <= -0.65 if side == "left"
                else signed_lateral >= 0.65
            )
            valid_height = (not pinch_band) or (0.10 <= height_fraction <= 0.94)
            if not (lateral_normal and faces_finger and valid_height):
                continue
            wrench = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            valid_points[side].append(point)
            valid_normals[side].append(normal)
            forces[side] += abs(float(wrench[0]))

        contacts = {side for side, points in valid_points.items() if points}
        if contacts == {"left", "right"}:
            left_normal = np.mean(valid_normals["left"], axis=0)
            right_normal = np.mean(valid_normals["right"], axis=0)
            left_normal /= max(float(np.linalg.norm(left_normal)), 1e-9)
            right_normal /= max(float(np.linalg.norm(right_normal)), 1e-9)
            left_point = np.mean(valid_points["left"], axis=0)
            right_point = np.mean(valid_points["right"], axis=0)
            separation = abs(float((right_point - left_point) @ closing_axis))
            minimum_span = 0.35 * min(
                float(target_bounds.size[0]), float(target_bounds.size[1])
            )
            if float(left_normal @ right_normal) > -0.50 or separation < minimum_span:
                # Two touches on the same face are not force closure.
                weaker = "left" if forces["left"] <= forces["right"] else "right"
                contacts.discard(weaker)
                forces[weaker] = 0.0
        return contacts, forces

    def max_lateral_contact_fraction(self, body_name: str) -> float | None:
        """Highest sideways finger contact, as a fraction of the body height."""
        target_id = int(self.model.body(body_name).id)
        bounds = self.body_bounds(body_name)
        height = max(float(bounds.size[2]), 1e-6)
        left_position = self.data.xpos[int(self.model.body("left_finger").id)]
        right_position = self.data.xpos[int(self.model.body("right_finger").id)]
        closing_axis = np.asarray(right_position - left_position, dtype=float)
        closing_axis /= max(float(np.linalg.norm(closing_axis)), 1e-9)
        highest: float | None = None
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if target_id not in {body1, body2}:
                continue
            other = body2 if body1 == target_id else body1
            other_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, other
            ) or ""
            side = "left" if other_name == "left_finger" else (
                "right" if other_name == "right_finger" else None
            )
            if side is None:
                continue
            normal = np.asarray(contact.frame[:3], dtype=float).copy()
            if body2 == target_id:
                normal *= -1.0
            signed_lateral = float(normal @ closing_axis)
            faces_finger = (
                signed_lateral <= -0.65 if side == "left"
                else signed_lateral >= 0.65
            )
            if abs(signed_lateral) < 0.75 or not faces_finger:
                continue
            point = np.asarray(contact.pos, dtype=float)
            fraction = float((point[2] - bounds.minimum[2]) / height)
            highest = fraction if highest is None else max(highest, fraction)
        return highest

    def raycast(
        self,
        origin_m: Vec3,
        direction: Vec3,
        *,
        max_distance_m: float,
        body_exclude: int = -1,
    ) -> tuple[float | None, int | None]:
        vector = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            raise ValueError("ray direction must be non-zero")
        vector /= norm
        geom_id = np.array([-1], dtype=np.int32)
        distance = float(mujoco.mj_ray(
            self.model,
            self.data,
            np.asarray(origin_m, dtype=float),
            vector,
            None,
            True,
            int(body_exclude),
            geom_id,
        ))
        if distance < 0.0 or distance > float(max_distance_m):
            return None, None
        return distance, int(geom_id[0])

    def iter_static_body_names(self) -> Iterable[str]:
        movable = set(self.movable_body_names())
        robot_prefixes = {"world", "link", "hand", "left_finger", "right_finger"}
        for body_id in range(1, self.model.nbody):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            ) or ""
            if not name or name in movable:
                continue
            if any(name == prefix or name.startswith(prefix) for prefix in robot_prefixes):
                continue
            if self.body_geom_ids(name):
                yield name

    def execute_trajectory(
        self,
        trajectory: JointTrajectory,
        monitor=None,
    ) -> tuple[bool, str | None]:
        """Execute exactly the samples accepted by the V2 validator."""
        self.robot.last_abort_reason = None
        self.carry_time_scale = 1.0
        previous_time = trajectory.times_s[0]
        for index, (position, sample_time) in enumerate(
            zip(trajectory.positions, trajectory.times_s, strict=True)
        ):
            duration = (
                self.model.opt.timestep
                if index == 0
                else max(float(sample_time - previous_time), self.model.opt.timestep)
            )
            scale = min(4.0, max(1.0, float(self.carry_time_scale)))
            steps = max(1, int(round(duration / self.model.opt.timestep * scale)))
            target = np.asarray(position, dtype=float)
            origin = self.arm_q().copy()
            use_interp = scale > 1.0
            taken = 0
            holds = 0
            frozen = False
            while taken < steps:
                if self.cancel_event.is_set():
                    self.data.ctrl[:7] = self.arm_q()
                    return False, "user_stop"
                if not frozen:
                    if use_interp:
                        alpha = (taken + 1) / steps
                        self.data.ctrl[:7] = origin + alpha * (target - origin)
                    else:
                        self.data.ctrl[:7] = target
                started = time.perf_counter()
                with self.data_lock:
                    mujoco.mj_step(self.model, self.data)
                    reason = None if monitor is None else monitor()
                new_scale = min(4.0, max(1.0, float(self.carry_time_scale)))
                if new_scale > scale:
                    steps = max(steps, int(round(
                        duration / self.model.opt.timestep * new_scale
                    )))
                    scale = new_scale
                if reason == "slow_down":
                    self.data.ctrl[:7] = self.arm_q()
                    holds += 1
                    if holds <= 50:
                        frozen = True
                        if self._pace_execution:
                            self.emit_frame()
                            remaining = self.model.opt.timestep / self.robot.speed - (
                                time.perf_counter() - started
                            )
                            if remaining > 0.0:
                                time.sleep(remaining)
                        continue
                    origin = self.arm_q().copy()
                    use_interp = True
                    frozen = False
                    holds = 0
                    reason = None
                if reason is not None:
                    self.data.ctrl[:7] = self.arm_q()
                    self.robot.last_abort_reason = str(reason)
                    return False, str(reason)
                frozen = False
                holds = 0
                taken += 1
                if self._pace_execution:
                    self.emit_frame()
                    remaining = self.model.opt.timestep / self.robot.speed - (
                        time.perf_counter() - started
                    )
                    if remaining > 0.0:
                        time.sleep(remaining)
            previous_time = sample_time
        self.emit_frame(force=True)
        return True, None
