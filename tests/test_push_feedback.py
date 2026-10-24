"""A sagging measured hand must not lower the commanded push plane."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from config import AgentConfig
from robot.robot_interface import Pose,ContactState
from skills.push_correction import PushCorrectionSkill
from world.world_model import WorldObject

class PushFeedbackTests(unittest.TestCase):
 def context(self):
  config=AgentConfig();obj=WorldObject('box','blue box','blue','box',np.array([.4,.2,.04]),
    np.array([0,0,0,1.]),np.array([.06,.06,.08]),.95,1.)
  pose=Pose(np.array([.4,.2,.4]),np.eye(3));moves=[];reads=[]
  robot=Mock();robot.get_ee_pose.side_effect=lambda:pose;robot.close_gripper.return_value=True
  robot.solve_ik.return_value=[np.zeros(7)];robot.check_collision.return_value=None
  def contact():
   reads.append(1);return ContactState(.5 if len(reads)>=10 else 0.,0.,False)
  robot.get_contact_state.side_effect=contact
  def move(position,rotation,**kw):
   nonlocal pose
   moves.append(np.asarray(position).copy());pose=Pose(np.asarray(position)-np.array([0,0,.003]),rotation)
  c=SimpleNamespace(config=config,held=False,source=obj,source_id=obj.id,world=object(),robot=robot,log=Mock(),
    safety=SimpleNamespace(check=lambda:None,force_limit_override=None),cartesian_speed_limit=None,budget=lambda:None,move=move)
  c.refresh=lambda:setattr(obj,'position',obj.position+np.array([.0015,0,0]))
  return c,moves
 def test_contact_search_does_not_accumulate_vertical_servo_error(self):
  c,moves=self.context()
  with patch('skills.push_correction.travel_to_observation'):
   self.assertTrue(PushCorrectionSkill().execute(c,np.array([.01,0])))
  expected=.035+c.config.planner.palm_to_contact_m
  np.testing.assert_allclose(np.array(moves[:-1])[:,2],expected,atol=1e-9)
  self.assertIsNone(c.safety.force_limit_override);self.assertIsNone(c.cartesian_speed_limit)
 def test_unreachable_push_rejected_before_robot_travel(self):
  from recovery.failure_classifier import SkillFailure
  c,moves=self.context();c.robot.check_collision.return_value='robot_collision:reference'
  with patch('skills.push_correction.travel_to_observation') as travel:
   with self.assertRaises(SkillFailure):PushCorrectionSkill().execute(c,np.array([.01,0]))
   travel.assert_not_called();self.assertEqual(moves,[])

if __name__=='__main__':unittest.main()

class PushBoundaryTests(unittest.TestCase):
 def test_diagonal_push_uses_ray_intersection_not_extreme_corner(self):
  from spatial.geometry import ray_exit_distance
  polygon=np.array([[-.04,-.03],[.04,-.03],[.04,.03],[-.04,.03]])
  direction=np.array([1.,1.])/np.sqrt(2)
  actual=ray_exit_distance(polygon,np.zeros(2),direction)
  self.assertAlmostEqual(actual,.03*np.sqrt(2))
  self.assertLess(actual,np.max(polygon@direction))
