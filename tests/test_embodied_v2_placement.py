from __future__ import annotations

import unittest

from embodied_v2.domain import (
    AABB,
    ObjectObservation,
    PayloadState,
    Relation,
)
from embodied_v2.placement import PrePlacePlanner, verify_placement


class V2PlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.destination = ObjectObservation(
            "support", (0.3, 0.2, 0.02),
            AABB((0.2, 0.1, 0.0), (0.4, 0.3, 0.04)),
            0.0, 0.2, 0.04, 0.04, 0.0, None, (), "fresh_rgbd", 1.0,
        )
        self.payload = PayloadState(
            "payload", AABB((-0.03, -0.03, 0.0), (0.03, 0.03, 0.10)),
            (0.0, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0),
            0.12, 0.001, 1.0, 0.006,
        )

    def test_multiple_preplace_candidates_are_above_contact(self) -> None:
        candidates = PrePlacePlanner().candidates(self.destination, self.payload)
        self.assertGreaterEqual(len(candidates), 5)
        contact_z = self.destination.top_z_m + self.payload.bottom_offset_m
        self.assertTrue(all(item.hand_position_m[2] > contact_z for item in candidates))

    def test_preplace_compensates_measured_payload_offset(self) -> None:
        candidates = PrePlacePlanner().candidates(
            self.destination,
            self.payload,
            payload_world_offset_m=(0.03, -0.01, -0.12),
        )
        center = next(item for item in candidates if item.name == "center")
        self.assertAlmostEqual(center.hand_position_m[0] + 0.03, 0.3)
        self.assertAlmostEqual(center.hand_position_m[1] - 0.01, 0.2)

    def test_final_verification_checks_relation_and_motion(self) -> None:
        evidence = verify_placement(
            payload_center_m=(0.31, 0.19, 0.09),
            destination_center_m=(0.3, 0.2, 0.02),
            destination_radius_m=0.10,
            support_contact=True,
            displacement_m=0.002,
            tilt_deg=2.0,
            speed_m_s=0.001,
            relation=Relation.ON,
        )
        self.assertTrue(evidence.relation_satisfied)
        self.assertFalse(evidence.still_moving)


if __name__ == "__main__":
    unittest.main()
