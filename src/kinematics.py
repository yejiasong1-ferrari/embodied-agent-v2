"""Numerical inverse kinematics for Cartesian Panda hand targets."""

from __future__ import annotations

import mujoco
import numpy as np

from joint_control import LEFT_GRASP


ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))


def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 0.5 * sum(
        (np.cross(current[:, index], target[:, index]) for index in range(3)),
        start=np.zeros(3),
    )


def solve_hand_ik(
    model: mujoco.MjModel,
    target_position: np.ndarray,
    seed: np.ndarray | None = None,
    max_iterations: int = 300,
    tolerance: float = 1e-4,
    yaw_rad: float = 0.0,
    target_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Solve a damped-least-squares IK target while keeping the hand downward.

    ``yaw_rad`` rotates the grasp about the world vertical.  Zero keeps the
    finger-closing axis on world Y, which is the historical Panda wrist pose.
    """
    target_position = np.asarray(target_position, dtype=float)
    if target_position.shape != (3,):
        raise ValueError("target_position must have shape (3,)")

    q = LEFT_GRASP.copy() if seed is None else np.asarray(seed, dtype=float).copy()
    # Do not snap joint 1 and do not pull posture in the nullspace.
    # Snapping the base made joint 7 unwind by a turn; a posture pull
    # stopped the hover above the green block from converging at all.
    ik_data = mujoco.MjData(model)
    qpos_addresses = [int(model.joint(name).qposadr[0]) for name in ARM_JOINTS]
    dof_addresses = [int(model.joint(name).dofadr[0]) for name in ARM_JOINTS]
    for address, value in zip(qpos_addresses, q, strict=True):
        ik_data.qpos[address] = value
    mujoco.mj_forward(model, ik_data)
    hand_id = model.body("hand").id
    # Keep the palm level.  Yaw 0 closes the fingers along world Y; other yaw
    # values turn that axis about the vertical.  Copying the seed orientation
    # instead made the fingers swing with the base joint and miss axis-aligned
    # blocks once the arm left the left side of the table.
    if target_rotation is None:
        closing = np.array((-np.sin(yaw_rad), -np.cos(yaw_rad), 0.0))
        down = np.array((0.0, 0.0, -1.0))
        target_rotation = np.column_stack((np.cross(closing, down), closing, down))
    else:
        target_rotation = np.asarray(target_rotation, dtype=float)
        if target_rotation.shape != (3, 3):
            raise ValueError("target_rotation must have shape (3, 3)")

    jac_position = np.zeros((3, model.nv))
    jac_rotation = np.zeros((3, model.nv))
    damping = 0.04

    for _ in range(max_iterations):
        for address, value in zip(qpos_addresses, q, strict=True):
            ik_data.qpos[address] = value
        mujoco.mj_forward(model, ik_data)
        current_position = ik_data.xpos[hand_id]
        current_rotation = ik_data.xmat[hand_id].reshape(3, 3)
        position_error = target_position - current_position
        rotation_error = _orientation_error(current_rotation, target_rotation)
        error = np.concatenate((position_error, 0.35 * rotation_error))
        if np.linalg.norm(position_error) < tolerance and np.linalg.norm(rotation_error) < 2e-3:
            return q

        mujoco.mj_jacBody(model, ik_data, jac_position, jac_rotation, hand_id)
        jacobian = np.vstack((jac_position[:, dof_addresses], 0.35 * jac_rotation[:, dof_addresses]))
        jjt = jacobian @ jacobian.T + damping**2 * np.eye(6)
        delta = jacobian.T @ np.linalg.solve(jjt, error)
        max_step = np.max(np.abs(delta))
        if max_step > 0.12:
            delta *= 0.12 / max_step
        q += delta
        for index, joint_name in enumerate(ARM_JOINTS):
            joint_id = model.joint(joint_name).id
            if model.jnt_limited[joint_id]:
                low, high = model.jnt_range[joint_id]
                q[index] = np.clip(q[index], low + 1e-4, high - 1e-4)

    raise RuntimeError(
        f"IK did not converge for base target {np.round(target_position, 3).tolist()}"
    )
