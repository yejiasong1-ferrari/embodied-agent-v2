import unittest
import numpy as np
from config import AgentConfig
from simulation.planning_backend import PlanningBackend
from manipulation.payload_geometry import merge_observed_fragment
from world.world_model import WorldModel,WorldObject

class ObservedCollisionTests(unittest.TestCase):
 def test_resized_perceived_obstacle_participates_in_collision(self):
  backend=PlanningBackend();world=WorldModel(AgentConfig().perception)
  q=backend.model.key_qpos[0,backend.addresses].copy();pose=backend.forward(q)
  obstacle=WorldObject('obstacle','observed obstacle','gray','obstacle',pose.position+np.array([.16,0,0]),
    np.array([0,0,0,1.]),np.array([.002,.002,.002]),1.,0.,movable=False)
  world.objects[obstacle.id]=obstacle;backend.configure(world,None,None,None)
  self.assertIsNone(backend.collision(q))
  obstacle.size=np.array([.40,.20,.20]);backend.configure(world,None,None,None)
  self.assertEqual(backend.collision(q),'robot_collision:obstacle')
  obstacle.position+=np.array([2.,0,0]);backend.configure(world,None,None,None)
  self.assertIsNone(backend.collision(q))
 def test_side_release_exempts_only_the_released_source_hand_overlap(self):
  backend=PlanningBackend();world=WorldModel(AgentConfig().perception)
  q=backend.model.key_qpos[0,backend.addresses].copy();pose=backend.forward(q)
  source=WorldObject('source','blue object','blue','box',pose.position+np.array([0,0,-.02]),
    np.array([0,0,0,1.]),np.array([.02,.02,.02]),1.,0.)
  world.objects[source.id]=source
  backend.configure(world,source.id,None,'observed_table')
  self.assertEqual(backend.collision(q),'robot_collision:source')
  backend.configure(world,source.id,None,'side_release')
  self.assertIsNone(backend.collision(q))
  backend.configure(world,source.id,None,'pregrasp')
  self.assertIsNone(backend.collision(q))
  backend.configure(world,None,None,'side_release')
  self.assertEqual(backend.collision(q),'robot_collision:source')

 def test_confirmed_carried_fragment_expands_payload_instead_of_becoming_an_obstacle(self):
  backend=PlanningBackend();world=WorldModel(AgentConfig().perception)
  q=backend.model.key_qpos[0,backend.addresses].copy();pose=backend.forward(q)
  source=WorldObject('source','red cup','red','cup',pose.position+np.array([0,0,-.05]),
    np.array([0,0,0,1.]),np.array([.06,.06,.08]),1.,0.)
  fragment=WorldObject('fragment','red handle','red','bottle',source.position+np.array([.07,0,0]),
    np.array([0,0,0,1.]),np.array([.04,.02,.04]),1.,0.)
  world.objects.update(source=source,fragment=fragment)
  offset=pose.rotation.T@(source.position-pose.position)
  payload={'id':'source','size':source.size,'offset':offset,
    'collision_offset':offset,'relative_rotation':pose.rotation.T}
  expanded=merge_observed_fragment(payload,pose,fragment)
  backend.configure(world,'source',expanded,'observed_table')
  self.assertNotIn('fragment',backend.obstacles.values())
  self.assertGreater(backend.payload['collision_size'][0],payload['size'][0]+.025)
  self.assertAlmostEqual(payload['size'][0],.06)

if __name__=='__main__':unittest.main()
