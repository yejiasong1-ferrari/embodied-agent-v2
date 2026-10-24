from __future__ import annotations

import unittest

import numpy as np

from embodied_v2.adapters import MujocoRuntime
from embodied_v2.motion import generate_joint_trajectory
from embodied_v2.ui import app


class V2DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_dashboard_renders_v2_safety_console(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("Transport Safety Console", page)
        self.assertIn("Third-person live", page)
        self.assertIn("完整多物体场景", page)
        self.assertIn("Actual trajectory", page)
        self.assertIn("Safety veto", page)

    def test_status_and_health_are_machine_readable(self) -> None:
        health = self.client.get("/api/health").get_json()
        status = self.client.get("/api/status").get_json()
        self.assertEqual(health, {"ok": True, "system": "embodied_v2"})
        self.assertIn("phase", status)
        self.assertIn("events", status)

    def test_third_person_stream_emits_execution_frames(self) -> None:
        frames = []
        with MujocoRuntime(width=120, height=90) as runtime:
            runtime.attach_frame_sink(
                lambda frame: frames.append(frame.copy()),
                playback_speed=1000.0,
                frame_interval_s=0.04,
            )
            q = runtime.arm_q()
            trajectory = generate_joint_trajectory((
                q, q + np.array((0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            ))
            success, reason = runtime.execute_trajectory(trajectory)
        self.assertTrue(success, reason)
        self.assertGreaterEqual(len(frames), 2)
        self.assertEqual(frames[-1].shape, (90, 120, 3))

    def test_native_state_stream_uses_copied_simulation_snapshots(self) -> None:
        snapshots = []
        with MujocoRuntime(width=120, height=90) as runtime:
            runtime.attach_state_sink(
                snapshots.append,
                playback_speed=1000.0,
                frame_interval_s=0.02,
            )
            q = runtime.arm_q()
            trajectory = generate_joint_trajectory((
                q, q + np.array((0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            ))
            success, reason = runtime.execute_trajectory(trajectory)
            terminal_qpos = runtime.data.qpos.copy()

        self.assertTrue(success, reason)
        self.assertGreaterEqual(len(snapshots), 2)
        np.testing.assert_allclose(snapshots[-1].qpos, terminal_qpos)
        self.assertFalse(np.shares_memory(snapshots[-1].qpos, terminal_qpos))


if __name__ == "__main__":
    unittest.main()
