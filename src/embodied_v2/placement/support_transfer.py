"""Support-contact confirmation, load transfer and gradual release."""

from __future__ import annotations

import numpy as np

from joint_control import GRIPPER_OPEN

from ..domain import PlacementEvidence, Relation


class SupportTransferController:
    def __init__(self, runtime, release_steps: int = 5):
        self.runtime = runtime
        self.release_steps = max(2, int(release_steps))

    def support_contact(self, payload_id: str, support_id: str) -> bool:
        payload_body = int(self.runtime.model.body(payload_id).id)
        support_body = int(self.runtime.model.body(support_id).id)
        for index in range(self.runtime.data.ncon):
            contact = self.runtime.data.contact[index]
            bodies = {
                int(self.runtime.model.geom_bodyid[contact.geom1]),
                int(self.runtime.model.geom_bodyid[contact.geom2]),
            }
            if bodies == {payload_body, support_body}:
                return True
        return False

    def gradual_release(self, payload_id: str, support_id: str) -> bool:
        if not self.support_contact(payload_id, support_id):
            return False
        start = float(self.runtime.data.ctrl[-1])
        for command in np.linspace(start, GRIPPER_OPEN, self.release_steps + 1)[1:]:
            if not self.runtime.robot.grasp_to(float(command), duration=0.12):
                return False
            # Contact must persist until the final opening; this prevents an
            # unsupported payload from being silently dropped.
            if command < GRIPPER_OPEN and not self.support_contact(payload_id, support_id):
                return False
        return True


def verify_placement(
    *,
    payload_center_m: tuple[float, float, float],
    destination_center_m: tuple[float, float, float],
    destination_radius_m: float,
    support_contact: bool,
    displacement_m: float,
    tilt_deg: float,
    speed_m_s: float,
    relation: Relation,
) -> PlacementEvidence:
    payload_xy = np.asarray(payload_center_m[:2], dtype=float)
    destination_xy = np.asarray(destination_center_m[:2], dtype=float)
    separation = float(np.linalg.norm(payload_xy - destination_xy))
    if relation is Relation.ON:
        relation_satisfied = support_contact and separation <= destination_radius_m
    else:
        relation_satisfied = separation <= max(0.20, 2.0 * destination_radius_m)
    return PlacementEvidence(
        support_contact=support_contact,
        displacement_m=float(displacement_m),
        tilt_deg=float(tilt_deg),
        still_moving=float(speed_m_s) > 0.012,
        relation_satisfied=relation_satisfied,
    )
