from __future__ import annotations

import unittest

from embodied_v2.adapters import MujocoRuntime
from embodied_v2.domain import Phase
from embodied_v2.embodied_agent import EmbodiedAgentV2


class V2EndToEndSimulationTest(unittest.TestCase):
    def test_incomplete_instruction_stops_cleanly_without_state_machine_crash(self) -> None:
        with MujocoRuntime(width=64, height=48) as runtime:
            result = EmbodiedAgentV2(runtime).run("帮我拿一下")
        self.assertFalse(result.success)
        self.assertEqual(result.final_phase, Phase.STOPPED)
        self.assertIn("camera inventory", result.message)

    def test_natural_language_to_verified_placement(self) -> None:
        with MujocoRuntime(width=96, height=72) as runtime:
            result = EmbodiedAgentV2(runtime).run("把红色杯子拿到黄色盘子上")
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.final_phase, Phase.DONE)
        visited = {event.phase for event in result.events}
        self.assertTrue({
            Phase.CLOSE_REOBSERVE,
            Phase.MICRO_LIFT,
            Phase.PLAN_TRANSPORT,
            Phase.DESCENT,
            Phase.LOAD_TRANSFER,
            Phase.GRADUAL_RELEASE,
            Phase.VERIFY_RESULT,
        }.issubset(visited))


if __name__ == "__main__":
    unittest.main()
