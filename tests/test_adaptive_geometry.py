import unittest
import numpy as np
from config import AgentConfig
from world.world_model import WorldModel,WorldObject
from perception.geometry_analyzer import GeometryAnalyzer,ObjectCloudFusion
from manipulation.grasp_planner import GraspPlanner


def surface(tapered=False):
 # Synthetic point-cloud unit fixture, never supplied to integration runs.
 theta=np.linspace(0,2*np.pi,100,endpoint=False);points=[]
 for z in np.linspace(.015,.15,35):
  radius=.055 if tapered and z>.12 else .025
  points.extend(np.column_stack([.4+radius*np.cos(theta),.1+radius*np.sin(theta),np.full_like(theta,z)]))
 for radius in np.linspace(.002,.055 if tapered else .025,20):
  points.extend(np.column_stack([.4+radius*np.cos(theta),.1+radius*np.sin(theta),np.full_like(theta,.15)]))
 return np.asarray(points)

class AdaptiveGeometry(unittest.TestCase):
 def setUp(self):self.config=AgentConfig();self.world=WorldModel(self.config.perception)
 def object(self,points,size):
  obj=WorldObject('target','blue object','blue','object',np.array([.4,.1,.075]),np.array([0,0,0,1]),np.array(size),.95,1,points=points,uncertainty_m=.004)
  self.world.objects[obj.id]=obj;return obj
 def test_wide_top_lower_side_from_points(self):
  obj=self.object(surface(True),[.11,.11,.15]);model=GeometryAnalyzer(self.config).analyze(obj,self.world)
  self.assertGreater(model['top_width'],.10);self.assertLess(model['lower_width'],.055)
  self.assertFalse(model['gripper_fit_top']);self.assertGreater(model['complexity_score'],.55)
  candidates=GraspPlanner(self.config).candidates(obj,self.world)
  self.assertTrue(any(c.grasp_type=='side_grasp' and c.contact_region=='lower_body' for c in candidates))
  self.assertFalse(any(c.grasp_type=='top_grasp' for c in candidates))
  for c in candidates:np.testing.assert_allclose(c.orientation.T@c.orientation,np.eye(3),atol=1e-7)
 def test_blocked_side_is_excluded(self):
  obj=self.object(surface(True),[.11,.11,.15]);obstacle=WorldObject('wall','wall','gray','obstacle',np.array([.30,.1,.10]),np.array([0,0,0,1]),np.array([.05,.15,.20]),.9,1,movable=False)
  self.world.objects['wall']=obstacle;GeometryAnalyzer(self.config).analyze(obj,self.world)
  candidates=GraspPlanner(self.config).candidates(obj,self.world)
  self.assertFalse(any(c.approach_direction=='left' for c in candidates));self.assertTrue(any(c.approach_direction=='right' for c in candidates))
 def test_fusion_bounded_and_does_not_join_moved_object(self):
  obj=self.object(surface(True),[.11,.11,.15]);fusion=ObjectCloudFusion(self.config)
  a,_=fusion.add(obj);obj.position=obj.position+np.array([.2,0,0]);obj.points=obj.points+np.array([.2,0,0]);b,_=fusion.add(obj)
  self.assertLessEqual(len(b),self.config.grasp.max_cloud_points);self.assertGreater(b[:,0].min(),.54)
 def test_geometry_never_claims_missing_profile(self):
  obj=self.object(np.empty((0,3)),[.05,.05,.1]);model=GeometryAnalyzer(self.config).analyze(obj,self.world)
  self.assertEqual(model['geometry_completeness'],0);self.assertEqual(model['graspable_regions'],[])
 def test_rotated_box_uses_full_measured_jaw_width(self):
  grid=np.mgrid[-.03:.0301:.002,-.03:.0301:.002].reshape(2,-1).T
  angle=np.pi/4;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  xy=grid@rotation.T+np.array([.4,.1]);points=np.column_stack([xy,np.full(len(xy),.08)])
  obj=self.object(points,[.085,.085,.08]);obj.position[2]=.04
  candidates=GraspPlanner(self.config).candidates(obj,self.world)
  self.assertFalse(any(c.id in {'top_grasp:0','top_grasp:1','top_grasp:4','top_grasp:5'} for c in candidates))
  self.assertTrue(any(c.id in {'top_grasp:2','top_grasp:3'} for c in candidates))
 def test_multiview_footprint_discards_sparse_stray_pixels(self):
  clean=surface(True);stray=np.array([[.48,.1,.15]]*5)
  obj=self.object(np.concatenate([clean,stray]),[.11,.11,.15])
  model=GeometryAnalyzer(self.config).analyze(obj,self.world)
  edge=max(x[0] for x in model['footprint_offsets'])+obj.position[0]
  self.assertLess(edge,.46)

if __name__=='__main__':unittest.main()
