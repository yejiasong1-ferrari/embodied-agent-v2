import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from config import AgentConfig
from manipulation.motion_planner import MotionPlanner
from robot.robot_interface import Pose


class FlatRobot:
    def __init__(self):
        self.q=np.array([0.,0.])
        self.joint_limits=(np.array([-2.,-2.]),np.array([2.,2.]))

    def get_joint_positions(self):return self.q.copy()
    def get_ee_pose(self):return self.forward_kinematics(self.q)
    def forward_kinematics(self,q):return Pose(np.array([q[0],q[1],.3]),np.eye(3))
    def solve_ik(self,pose,seed):return [np.asarray(pose.position[:2]).copy()]
    def check_collision(self,q):
        return 'robot_collision:wall' if .48<q[0]<.49 and abs(q[1])<.1 else None


class TransportPreflightTests(unittest.TestCase):
    def test_dense_direct_rejection_searches_a_real_detour(self):
        robot=FlatRobot();planner=MotionPlanner(robot,AgentConfig())
        goal=Pose(np.array([.7,0.,.3]),np.eye(3))
        route=[np.array([0.,0.]),np.array([0.,.3]),np.array([.7,.3]),np.array([.7,0.])]
        with patch('motion_planning.plan_rrt_connect',return_value=SimpleNamespace(as_arrays=lambda:route)) as rrt:
            path=planner.plan_to_pose(goal)
        rrt.assert_called_once()
        self.assertIsNotNone(path)
        self.assertEqual(len(path),4)
        self.assertIsNone(planner.last_failure)
        self.assertIn({'route':'direct','reason':'dense_rejected'},planner.last_diagnostics)

    def test_preflight_rejects_a_later_blocked_segment_before_execution(self):
        robot=FlatRobot();planner=MotionPlanner(robot,AgentConfig())
        waypoints=[Pose(np.array([.2,0.,.3]),np.eye(3)),Pose(np.array([.7,0.,.3]),np.eye(3))]
        with patch('motion_planning.plan_rrt_connect',return_value=None):
            valid,evidence=planner.preview_route(waypoints,step_m=.2)
        self.assertFalse(valid)
        self.assertEqual(evidence['waypoint'],1)
        np.testing.assert_allclose(robot.get_joint_positions(),[0.,0.])

    def test_invalid_start_is_reported_without_rrt(self):
        robot=FlatRobot();robot.q=np.array([.485,0.])
        planner=MotionPlanner(robot,AgentConfig())
        with patch('motion_planning.plan_rrt_connect') as rrt:
            self.assertIsNone(planner.plan_to_pose(Pose(np.array([.7,0.,.3]),np.eye(3))))
        rrt.assert_not_called()
        self.assertEqual(planner.last_diagnostics,[{'route':'start','reason':'robot_collision:wall'}])


if __name__=='__main__':unittest.main()
