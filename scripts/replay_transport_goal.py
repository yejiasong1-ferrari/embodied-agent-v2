"""Replay one saved transport goal with a diagnostic budget.

The live planner budget is not changed. This script only answers whether the
saved start and goal can produce a densely valid OMPL path when given longer
than the carried 3.2 s cap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from camera_demo import (
    ARM_JOINTS,
    _payload_scene_clearance_hit,
    _robot_scene_clearance_hit,
    _rotation_deviation_deg,
    _scene_hit,
)
from joint_control import create_model
from motion_planning import plan_rrt_connect


DIAGNOSTIC_TIMEOUT_S = 8.0


def _first_failed_goal(log: dict) -> dict:
    for checkpoint in log.get("checkpoints", []):
        if checkpoint.get("stage") != "transport_plan_selection":
            continue
        evidence = checkpoint.get("evidence") or {}
        if evidence.get("executed"):
            continue
        for item in evidence.get("rejected_candidates") or []:
            if item.get("goal_q") and item.get("scene_qpos"):
                return item
    raise SystemExit("no failed goal with scene_qpos in this log")


def main() -> None:
    path = Path(sys.argv[1])
    item = _first_failed_goal(json.loads(path.read_text()))
    model = create_model()
    scratch = mujoco.MjData(model)
    scene_qpos = np.asarray(item["scene_qpos"], dtype=float)
    scratch.qpos[:] = scene_qpos
    mujoco.mj_forward(model, scratch)
    payload_body = str(item["payload_body"])
    payload_id = model.body(payload_body).id
    hand_id = model.body("hand").id
    initial_hand_rotation = scratch.xmat[hand_id].reshape(3, 3).copy()
    object_rotation = scratch.xmat[payload_id].reshape(3, 3).copy()
    relative_position = initial_hand_rotation.T @ (
        scratch.xpos[payload_id] - scratch.xpos[hand_id]
    )
    relative_rotation = initial_hand_rotation.T @ object_rotation
    addresses = [int(model.joint(name).qposadr[0]) for name in ARM_JOINTS]
    finger_addresses = [
        int(model.joint("finger_joint1").qposadr[0]),
        int(model.joint("finger_joint2").qposadr[0]),
    ]
    finger_q = float(scratch.qpos[finger_addresses[0]])
    payload_joint = int(model.body_jntadr[payload_id])
    payload_qpos = int(model.jnt_qposadr[payload_joint])
    drop_m = float(item["drop_m"])
    minimum_z = float(item["minimum_z_m"])
    sweep = float(item["payload_sweep_clearance_m"])
    tilt_limit = float(item["maximum_tilt_deg"])
    ignore = {payload_body}
    lower = np.asarray([model.joint(name).range[0] for name in ARM_JOINTS])
    upper = np.asarray([model.joint(name).range[1] for name in ARM_JOINTS])

    def set_state(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scratch.qpos[:] = scene_qpos
        for address, value in zip(addresses, q, strict=True):
            scratch.qpos[address] = float(value)
        scratch.qpos[finger_addresses[0]] = finger_q
        scratch.qpos[finger_addresses[1]] = finger_q
        mujoco.mj_forward(model, scratch)
        hand_rotation = scratch.xmat[hand_id].reshape(3, 3).copy()
        hand_xyz = scratch.xpos[hand_id].copy()
        payload_xyz = hand_xyz + hand_rotation @ relative_position
        payload_rotation = hand_rotation @ relative_rotation
        quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quaternion, payload_rotation.reshape(-1))
        scratch.qpos[payload_qpos : payload_qpos + 3] = payload_xyz
        scratch.qpos[payload_qpos + 3 : payload_qpos + 7] = quaternion
        mujoco.mj_forward(model, scratch)
        return hand_xyz, hand_rotation

    def valid(q: np.ndarray) -> bool:
        hand_xyz, hand_rotation = set_state(np.asarray(q, dtype=float))
        if _rotation_deviation_deg(initial_hand_rotation, hand_rotation) > tilt_limit + 1e-9:
            return False
        if float(hand_xyz[2]) - drop_m < minimum_z:
            return False
        if _scene_hit(model, scratch, ignore, payload_body=payload_body) is not None:
            return False
        if _robot_scene_clearance_hit(model, scratch, ignore, clearance_m=0.012) is not None:
            return False
        return _payload_scene_clearance_hit(
            model, scratch, payload_body, ignore, clearance_m=sweep,
        ) is None

    diagnostics: list[dict[str, object]] = []
    plan = plan_rrt_connect(
        np.asarray(item["q_start"], dtype=float),
        [np.asarray(item["goal_q"], dtype=float)],
        lower,
        upper,
        valid,
        timeout_s=DIAGNOSTIC_TIMEOUT_S,
        interpolation_step_rad=0.035,
        diagnostics=diagnostics,
    )
    print(json.dumps({
        "source": path.name,
        "logged_reason": item.get("reason"),
        "diagnostic_timeout_s": DIAGNOSTIC_TIMEOUT_S,
        "validated_plan": plan is not None,
        "plan_id": None if plan is None else plan.plan_id,
        "planning_time_s": None if plan is None else plan.planning_time_s,
        "queries": diagnostics,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
