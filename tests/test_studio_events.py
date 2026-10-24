"""User-facing execution claims must follow the actual V2 evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from config import AgentConfig
from agent.task_log import TaskLog
from simulation.studio_presenter import human_event


class StudioEvidenceTests(unittest.TestCase):
    def test_live_event_matches_persisted_event(self):
        with tempfile.TemporaryDirectory() as root:
            events = []
            log = TaskLog(root, 'test', AgentConfig(), event_sink=events.append)
            log.event('state', {'state':'FAILED', 'failure':'USER_STOP'})
            saved = json.loads((log.path/'events.jsonl').read_text())
            self.assertEqual(events, [saved])

    def test_measurement_is_not_announced_as_success(self):
        _, text = human_event({'type':'final_validation', 'data':{
            'stable_displacement_m':.1, 'object_released':False,
            'relation_satisfied':False}})
        self.assertNotIn('通过', text)
        self.assertIn('仍被夹持', text)

    def test_grasp_zero_force_remains_a_measurement(self):
        tone, text = human_event({'type':'grasp_validation', 'data':{
            'left_force':0, 'right_force':0, 'width':0}})
        self.assertEqual(tone, 'info')
        self.assertNotIn('成功', text)

    def test_failure_and_fallback_are_visible(self):
        tone, text = human_event({'type':'state', 'data':{
            'state':'FAILED', 'failure':'COLLISION_RISK'}})
        self.assertEqual(tone, 'error')
        self.assertIn('碰撞', text)
        _, text = human_event({'type':'llm_fallback', 'data':{}})
        self.assertIn('没有使用 LLM', text)


if __name__ == '__main__': unittest.main()
