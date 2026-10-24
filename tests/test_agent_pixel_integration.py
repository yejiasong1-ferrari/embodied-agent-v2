"""Negative control against simulator-truth leakage, using a real rendered frame."""
import tempfile
import unittest
from dataclasses import replace
import numpy as np
from config import AgentConfig
from simulation.environment import Simulation
from simulation.mujoco_camera import MujocoCamera
from robot.mujoco_robot import MuJoCoRobot
from world.world_model import WorldModel
from perception.active_perception import ActivePerception
from agent.task_log import TaskLog
from llm.llm_client import LLMClient


class ReplayCamera:
    viewpoint_count=1
    def __init__(self,frame): self.frame=frame
    def capture(self): return self.frame
    def move_viewpoint(self,index): return index==0
    def set_manipulation_active(self,active): pass
    def update_collision_world(self,world): pass


class PixelIntegrationTests(unittest.TestCase):
    def test_verified_grasp_uses_physical_carry_friction_until_release(self):
        with Simulation(AgentConfig()) as sim:
            robot=MuJoCoRobot(sim)
            self.assertTrue(robot._control.one_sided_friction)
            robot.confirm_grasp()
            self.assertTrue(robot._bilateral_grasp_verified)
            self.assertFalse(robot._control.one_sided_friction)
            robot.open_gripper()
            self.assertFalse(robot._bilateral_grasp_verified)
            self.assertTrue(robot._control.one_sided_friction)

    def test_black_rgb_cannot_use_scene_objects_or_depth_as_semantic_truth(self):
        config=AgentConfig(); config.llm.mode='offline'
        with tempfile.TemporaryDirectory() as directory:
            config.log_root=directory
            with Simulation(config,seed=728) as sim:
                robot=MuJoCoRobot(sim)
                frame=MujocoCamera(sim,robot).capture()
                for black in (False,True):
                    observed=replace(frame,rgb=np.zeros_like(frame.rgb)) if black else frame
                    world=WorldModel(config.perception)
                    log=TaskLog(directory,'pixel negative control',config)
                    pipeline=ActivePerception(ReplayCamera(observed),world,config,LLMClient(config.llm),log)
                    pipeline.observe(['red cup'])
                    if black: self.assertEqual(world.objects,{})
                    else: self.assertGreaterEqual(len(world.objects),3)

if __name__=='__main__': unittest.main()
