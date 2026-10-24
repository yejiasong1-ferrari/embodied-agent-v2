from __future__ import annotations

import unittest

import mujoco
import numpy as np

from embodied_v2.adapters import MujocoRuntime
from embodied_v2.domain import ContactState, DistanceReading, GraspEvidence
from embodied_v2.grasp import (
    GeometryGraspPlanner,
    contact_gap_correction,
    evaluate_micro_lift,
    is_grasp_hypothesis_failure,
)
from embodied_v2.perception import (
    MujocoCloseRangePerception,
    MujocoSceneScanner,
    fuse_distance_measurements,
)


class V2PerceptionGraspTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = MujocoRuntime(width=160, height=120)
        cls.scanner = MujocoSceneScanner(cls.runtime)
        cls.scene = cls.scanner.scan()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.runtime.close()

    def test_scene_scan_discovers_geometry_without_catalog_names(self) -> None:
        movable = [entity for entity in self.scene.entities if entity.movable]
        self.assertGreaterEqual(len(movable), 5)
        self.assertTrue(any(entity.geometry == "cylinder" for entity in movable))
        self.assertTrue(any(entity.geometry == "box" for entity in movable))
        self.assertTrue(any(support.geometry == "plate" for support in self.scene.supports))
        self.assertGreaterEqual(len(self.scene.gap_map.regions), 3)

    def test_close_observation_and_insertion_depth(self) -> None:
        target = next(
            entity for entity in self.scene.entities
            if entity.movable and entity.geometry == "cylinder"
        )
        observed = MujocoCloseRangePerception(self.runtime).observe(
            target.entity_id, self.scene
        )
        candidates = GeometryGraspPlanner().candidates(observed, self.scene)
        self.assertEqual(len(candidates), 4)
        self.assertTrue(all(0.0 < item.insertion_depth_m < observed.height_m for item in candidates))
        self.assertTrue(all(item.palm_height_m > item.finger_target_height_m for item in candidates))

    def test_distance_conflict_forces_reobserve(self) -> None:
        reading = DistanceReading(
            0.10, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), "tof", True
        )
        fused, verdict = fuse_distance_measurements(0.14, reading)
        self.assertIsNone(fused)
        self.assertEqual(verdict, "sensor_conflict_reobserve")

    def test_missed_contact_is_a_local_correction_not_a_new_grasp(self) -> None:
        closing = np.array((1.0, 0.0))
        # Fingers are beside the measured center, 3 cm short along the approach.
        adjustment = contact_gap_correction(
            object_center_m=np.array((0.40, 0.03, 0.04)),
            top_z_m=0.08,
            bottom_z_m=0.0,
            finger_midpoint_m=np.array((0.40, 0.0, 0.04)),
            closing_axis_xy=closing,
            contact_state=ContactState.NO_CONTACT,
        )
        self.assertIsNotNone(adjustment)
        self.assertEqual(adjustment.reason, "too_far")
        self.assertFalse(is_grasp_hypothesis_failure(adjustment.reason))
        self.assertGreater(adjustment.delta_m[1], 0.0)
        self.assertLessEqual(abs(adjustment.delta_m[1]), 0.008)
        self.assertTrue(is_grasp_hypothesis_failure("local_exhausted"))
        self.assertTrue(is_grasp_hypothesis_failure("ik_unreachable"))
        self.assertTrue(is_grasp_hypothesis_failure("lift_unstable"))

    def test_force_asymmetry_does_not_fail_stable_micro_lift(self) -> None:
        evidence = GraspEvidence(
            contact_state=ContactState.BILATERAL,
            left_force_n=0.7,
            right_force_n=1.8,
            object_rise_m=0.004,
            relative_slip_m=0.002,
            lateral_displacement_m=0.003,
            tilt_deg=2.0,
            contact_persisted=True,
            hard_force_exceeded=False,
        )
        stable, reason = evaluate_micro_lift(evidence)
        self.assertTrue(stable)
        self.assertEqual(reason, "stable_grasp")
        self.assertGreater(evidence.force_ratio_diagnostic, 2.0)

    def test_v2_payload_and_finger_pad_physics_are_visible_and_bounded(self) -> None:
        cup_body = int(self.runtime.model.body("red_cup").id)
        self.assertGreater(float(self.runtime.model.body_mass[cup_body]), 0.10)
        pad_ids = [
            geom_id for geom_id in range(self.runtime.model.ngeom)
            if (mujoco.mj_id2name(
                self.runtime.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            ) or "").endswith("_grip_pad")
        ]
        self.assertEqual(len(pad_ids), 2)
        for geom_id in pad_ids:
            self.assertEqual(int(self.runtime.model.geom_group[geom_id]), 0)
            self.assertLessEqual(float(self.runtime.model.geom_friction[geom_id, 0]), 1.2)
            self.assertEqual(int(self.runtime.model.geom_condim[geom_id]), 4)


if __name__ == "__main__":
    unittest.main()
