"""Create the measured rigid payload relation after micro-lift succeeds."""

from __future__ import annotations

import math
import numpy as np
import mujoco

from ..domain import AABB, GraspEvidence, ObjectObservation, PayloadState, vec3


def build_payload_state(
    entity_id: str,
    observation: ObjectObservation,
    hand_position_m: np.ndarray,
    hand_rotation: np.ndarray,
    object_position_m: np.ndarray,
    object_rotation: np.ndarray,
    evidence: GraspEvidence,
    tracking_margin_m: float = 0.002,
) -> PayloadState:
    relative_position = hand_rotation.T @ (object_position_m - hand_position_m)
    relative_rotation = hand_rotation.T @ object_rotation
    quaternion_wxyz = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion_wxyz, relative_rotation.reshape(-1))
    quaternion_xyzw = (
        float(quaternion_wxyz[1]),
        float(quaternion_wxyz[2]),
        float(quaternion_wxyz[3]),
        float(quaternion_wxyz[0]),
    )
    # Re-measure attitude every time this state is rebuilt.  After transport,
    # compliant contacts may leave a payload a few degrees away from the
    # micro-lift attitude.  Reusing the old evidence made sample zero of every
    # pre-place path violate its own attitude contract, so no path could ever
    # guide the object upright again.
    measured_tilt_deg = math.degrees(math.acos(float(np.clip(
        object_rotation[2, 2], -1.0, 1.0
    ))))
    pivot_radius = 0.5 * float(np.linalg.norm(np.asarray(observation.bounds.size)))
    tilt_margin = pivot_radius * np.sin(np.deg2rad(max(measured_tilt_deg, 0.0)))
    envelope = float(tracking_margin_m + evidence.relative_slip_m + tilt_margin)
    return PayloadState(
        entity_id=entity_id,
        geometry=AABB(observation.bounds.minimum, observation.bounds.maximum),
        gripper_to_object_translation_m=vec3(relative_position),
        gripper_to_object_quaternion=quaternion_xyzw,
        bottom_offset_m=float(hand_position_m[2] - observation.bottom_z_m),
        measured_slip_m=float(evidence.relative_slip_m),
        measured_tilt_deg=float(measured_tilt_deg),
        envelope_margin_m=envelope,
    )
