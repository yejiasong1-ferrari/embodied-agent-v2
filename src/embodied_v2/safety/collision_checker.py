"""MuJoCo-clone state checker for robot, gripper, and attached payload."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from ..adapters.mujoco_runtime import ARM_JOINTS, MujocoRuntime
from ..domain import PayloadState


@dataclass(frozen=True)
class CollisionSample:
    safe: bool
    robot_clearance_m: float
    payload_clearance_m: float
    collision_pair: str | None


class CollisionChecker:
    """Evaluates hypothetical states in private ``MjData`` only."""

    def __init__(self, runtime: MujocoRuntime, clearance_cap_m: float = 0.20):
        self.runtime = runtime
        self.model = runtime.model
        self.clearance_cap_m = float(clearance_cap_m)
        self._qpos_addresses = tuple(
            int(self.model.joint(name).qposadr[0]) for name in ARM_JOINTS
        )
        self._robot_bodies = frozenset(
            body_id for body_id in range(1, self.model.nbody)
            if self._is_robot_body(self._body_name(body_id))
        )
        self._robot_geoms = tuple(
            geom_id for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in self._robot_bodies
        )

    @staticmethod
    def _is_robot_body(name: str) -> bool:
        return name.startswith("link") or name in {"hand", "left_finger", "right_finger"}

    def _body_name(self, body_id: int) -> str:
        return mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)
        ) or f"body:{body_id}"

    def _attach_payload(self, data: mujoco.MjData, payload: PayloadState) -> set[int]:
        try:
            body_id = int(self.model.body(payload.entity_id).id)
        except KeyError:
            return set()
        joint_id = int(self.model.body_jntadr[body_id])
        if joint_id < 0 or int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            return set()
        hand_id = int(self.model.body("hand").id)
        hand_position = data.xpos[hand_id]
        hand_rotation = data.xmat[hand_id].reshape(3, 3)
        object_position = hand_position + hand_rotation @ np.asarray(
            payload.gripper_to_object_translation_m, dtype=float
        )
        # Payload orientation is measured, not assumed from its catalog type.
        xyzw = np.asarray(payload.gripper_to_object_quaternion, dtype=float)
        object_quat = np.array((xyzw[3], xyzw[0], xyzw[1], xyzw[2]), dtype=float)
        object_rotation = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(object_rotation, object_quat)
        world_rotation = hand_rotation @ object_rotation.reshape(3, 3)
        world_quat = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(world_quat, world_rotation.reshape(-1))
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        data.qpos[qpos_address:qpos_address + 3] = object_position
        data.qpos[qpos_address + 3:qpos_address + 7] = world_quat
        mujoco.mj_forward(self.model, data)
        return set(self.runtime.body_geom_ids(payload.entity_id))

    def evaluate(
        self,
        q: tuple[float, ...],
        payload: PayloadState | None,
        *,
        phase: str,
        allowed_support_id: str | None = None,
    ) -> CollisionSample:
        data = mujoco.MjData(self.model)
        mujoco.mj_copyData(data, self.model, self.runtime.data)
        data.qpos[list(self._qpos_addresses)] = np.asarray(q, dtype=float)
        mujoco.mj_forward(self.model, data)
        payload_geoms = self._attach_payload(data, payload) if payload else set()
        payload_body = int(self.model.body(payload.entity_id).id) if payload and payload_geoms else -1
        allowed_support_body = -1
        if allowed_support_id is not None:
            try:
                allowed_support_body = int(self.model.body(allowed_support_id).id)
            except KeyError:
                pass

        robot_min = self.clearance_cap_m
        payload_min = self.clearance_cap_m
        payload_min_pair: str | None = None
        collision_pair: str | None = None

        if payload is not None and payload_body >= 0 and phase in {"transport", "pre_place"}:
            object_rotation = data.xmat[payload_body].reshape(3, 3)
            tilt_deg = math.degrees(math.acos(float(np.clip(
                object_rotation[2, 2], -1.0, 1.0
            ))))
            # Match the verified-placement attitude contract while retaining
            # two degrees of runtime stop headroom (runtime hard stop: 12°).
            tilt_limit_deg = max(10.25, float(payload.measured_tilt_deg) + 4.0)
            if tilt_deg > tilt_limit_deg:
                return CollisionSample(
                    False,
                    robot_min,
                    payload_min,
                    f"payload:tilt_constraint:{tilt_deg:.2f}>{tilt_limit_deg:.2f}",
                )
        environment_geoms = tuple(
            geom for geom in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom]) not in self._robot_bodies
            and int(self.model.geom_bodyid[geom]) != 0
            and geom not in payload_geoms
        )

        # mj_forward populates MuJoCo's narrow-phase contacts for this exact
        # clone state.  Use them as the authoritative penetration test; this
        # is substantially faster than asking geomDistance for every pair.
        for index in range(data.ncon):
            contact = data.contact[index]
            if float(contact.dist) >= -1e-5:
                continue
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            bodies = {body1, body2}
            geom_set = {geom1, geom2}
            robot_bodies = bodies & self._robot_bodies
            payload_hit = bool(geom_set & payload_geoms)
            if robot_bodies and payload_hit:
                forbidden = any(
                    self._body_name(body) not in {"hand", "left_finger", "right_finger"}
                    for body in robot_bodies
                )
                if forbidden:
                    pair = ":".join(self._body_name(body) for body in bodies)
                    return CollisionSample(False, float(contact.dist), payload_min, pair)
                continue
            if robot_bodies and not payload_hit:
                environment_body = next((body for body in bodies if body not in self._robot_bodies), -1)
                if environment_body > 0:
                    robot_name = self._body_name(next(iter(robot_bodies)))
                    pair = f"{robot_name}:{self._body_name(environment_body)}"
                    return CollisionSample(False, float(contact.dist), payload_min, pair)
            if payload_hit:
                other_body = next((
                    int(self.model.geom_bodyid[geom])
                    for geom in geom_set if geom not in payload_geoms
                ), -1)
                allowed = (
                    phase in {"descent", "first_contact", "load_transfer", "gradual_release"}
                    and other_body == allowed_support_body
                )
                if other_body > 0 and not allowed:
                    pair = f"payload:{self._body_name(other_body)}"
                    return CollisionSample(False, robot_min, float(contact.dist), pair)

        def exact_if_near(geom1: int, geom2: int, current: float) -> float:
            center_distance = float(np.linalg.norm(
                data.geom_xpos[geom1] - data.geom_xpos[geom2]
            ))
            lower_bound = center_distance - float(
                self.model.geom_rbound[geom1] + self.model.geom_rbound[geom2]
            )
            if lower_bound >= current:
                return current
            distance = float(mujoco.mj_geomDistance(
                self.model, data, geom1, geom2, self.clearance_cap_m, None
            ))
            return min(current, distance)

        for robot_geom in self._robot_geoms:
            for other_geom in environment_geoms:
                robot_min = exact_if_near(robot_geom, other_geom, robot_min)
                if robot_min < -1e-5:
                    pair = (
                        f"{self._body_name(int(self.model.geom_bodyid[robot_geom]))}:"
                        f"{self._body_name(int(self.model.geom_bodyid[other_geom]))}"
                    )
                    return CollisionSample(False, robot_min, payload_min, pair)

        for payload_geom in payload_geoms:
            for other_geom in environment_geoms:
                other_body = int(self.model.geom_bodyid[other_geom])
                previous = payload_min
                payload_min = exact_if_near(payload_geom, other_geom, payload_min)
                if payload_min < previous:
                    payload_min_pair = self._body_name(other_body)
                allowed = (
                    phase in {"descent", "first_contact", "load_transfer", "gradual_release"}
                    and other_body == allowed_support_body
                )
                if payload_min < -1e-5 and not allowed:
                    pair = f"payload:{self._body_name(other_body)}"
                    return CollisionSample(False, robot_min, payload_min, pair)
        if (
            payload is not None
            and phase == "transport"
            and payload_min < payload.envelope_margin_m
        ):
            return CollisionSample(
                False,
                robot_min,
                payload_min,
                (
                    "payload:dynamic_envelope"
                    if payload_min_pair is None
                    else f"payload:dynamic_envelope:{payload_min_pair}"
                ),
            )
        return CollisionSample(True, robot_min, payload_min, None)
