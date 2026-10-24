from __future__ import annotations

import unittest

from embodied_v2.agent import (
    NaturalLanguageTaskManager,
    RouteStrategist,
    RouteSupervisor,
    SceneMemory,
    TypedRecoveryManager,
)
from embodied_v2.domain import (
    FailureType,
    PayloadState,
    Phase,
    SafetyReport,
    SafetyVerdict,
)
from embodied_v2.adapters import MujocoRuntime
from embodied_v2.perception import MujocoSceneScanner


class _OfflineManager:
    available = False

    def json_completion(self, system: str, user: str, timeout_s: float = 20.0) -> dict:
        raise AssertionError("offline posture planning must not call the model")


class V2ManagerRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = MujocoRuntime(width=120, height=90)
        cls.scene = MujocoSceneScanner(cls.runtime).scan()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.runtime.close()

    def test_task_manager_outputs_semantics_not_coordinates(self) -> None:
        task = NaturalLanguageTaskManager().understand("把红色杯子拿到黄色盘子上")
        self.assertEqual(task.object_color, "red")
        self.assertEqual(task.object_category, "cup")
        self.assertEqual(task.destination_color, "yellow")
        self.assertEqual(task.destination_category, "plate")

    def test_camera_inventory_grounds_terse_language_and_persists_tracks(self) -> None:
        memory = SceneMemory()
        remembered = memory.update(self.scene)
        task = NaturalLanguageTaskManager(scene_memory=memory).understand(
            "把蓝色的放到盘子上", self.scene
        )
        self.assertGreaterEqual(len(remembered), 10)
        self.assertEqual(memory.revision, 1)
        self.assertEqual(task.object_color, "blue")
        self.assertEqual(task.object_category, "box")
        self.assertEqual(task.destination_color, "yellow")
        self.assertEqual(task.destination_category, "plate")
        self.assertIn("blue_box", memory.prompt_context())

    def test_route_strategy_proposes_alternatives_and_safety_has_veto(self) -> None:
        task = NaturalLanguageTaskManager().understand("把红色杯子拿到黄色盘子上")
        entity = next(item for item in self.scene.entities if item.movable)
        payload = PayloadState(
            entity.entity_id, entity.bounds, (0, 0, 0.1), (0, 0, 0, 1),
            0.1, 0.0, 0.0, 0.006,
        )
        intents = RouteStrategist().propose(
            task, self.scene, payload, entity.center_m, self.scene.supports[0].center_m
        )
        self.assertGreaterEqual(len(intents), 2)
        failed = SafetyReport(SafetyVerdict.FAIL, "collision", -0.01, 0.02, 4)
        self.assertEqual(RouteSupervisor().review(intents[0], failed), "SAFETY_REJECT")
        postures = RouteStrategist(client=_OfflineManager(), use_online=False).propose_reconfiguration(
            task, self.scene, payload, tuple(item.corridor for item in intents),
        )
        self.assertGreaterEqual(len(postures), 2)
        self.assertTrue(all(item.posture in {"elbow_up", "elbow_left", "elbow_right"} for item in postures))
        self.assertTrue(all(item.high_clearance for item in postures))

    def test_recovery_changes_search_dimensions(self) -> None:
        directive = TypedRecoveryManager().decide(
            FailureType.UNILATERAL_CONTACT, Phase.CONTACT_SEARCH, 1
        )
        self.assertEqual(directive.restart_phase, Phase.PRE_ALIGN)
        self.assertIn("yaw", directive.changed_search_dimensions)
        exhausted = TypedRecoveryManager(maximum_attempts=1).decide(
            FailureType.LIFT_SLIP, Phase.MICRO_LIFT, 1
        )
        self.assertEqual(exhausted.restart_phase, Phase.STOPPED)


if __name__ == "__main__":
    unittest.main()
