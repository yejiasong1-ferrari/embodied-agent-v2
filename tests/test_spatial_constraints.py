import unittest
import numpy as np
from config import AgentConfig
from world.world_model import WorldModel,WorldObject
from agent.task_interpreter import reference
from spatial.constraint_parser import parse_local
from spatial.spatial_planner import SpatialPlanner
from spatial.relation_validator import RelationValidator
from spatial.operator_frame import OperatorFrame
from spatial.correction_planner import PlacementCorrectionPlanner
from llm.schemas import Goal,SpatialConstraint,GraspStrategyDecision,GroundedTarget
from studio.camera import CameraState
from skills.manipulation import placement_planning_goal,short_arc_angles,near_visual_gap_limit,fuse_visual_gap_samples
from types import SimpleNamespace
from spatial.geometry import supported_by,footprint


def object_(identity,position,size,color,category='box'):
 return WorldObject(identity,f'{color} {category}',color,category,np.array(position,float),np.array([0,0,0,1.]),np.array(size,float),.95,1.,uncertainty_m=.005)

class SpatialContracts(unittest.TestCase):
 def setUp(self):
  self.config=AgentConfig();self.world=WorldModel(self.config.perception)
  self.source=object_('cup',(-.4,-.3,.04),(.06,.06,.08),'red','cup')
  self.target=object_('plate',(.25,.25,.02),(.20,.20,.04),'yellow','plate')
  self.world.objects.update(cup=self.source,plate=self.target)
  xy=np.mgrid[-.74:.741:.012,-.74:.741:.012].reshape(2,-1).T;self.world.support_points=np.column_stack([xy,np.zeros(len(xy))])
 def test_near_tolerance_visual_gap_uses_consistent_second_view(self):
  def sample(gap):
   error=abs(gap-.1)
   return {'constraints':[{'type':'relative_position','requested_gap_m':.1,'actual_gap_m':gap,
     'tolerance_m':.01,'alignment_error_m':.003,'direction_correct':True,'satisfied':error<=.01}],
     'fresh_visual_evidence':True,'stable_displacement_m':.0002,'object_released':True,
     'accepted':error<=.01,'relation_satisfied':error<=.01}
  first=sample(.089673);second=sample(.09237)
  self.assertTrue(near_visual_gap_limit(first))
  ok,evidence=fuse_visual_gap_samples([(False,first),(True,second)],self.config)
  self.assertTrue(ok)
  self.assertAlmostEqual(evidence['constraints'][0]['actual_gap_m'],(.089673+.09237)/2)
  self.assertEqual(first['constraints'][0]['actual_gap_m'],.089673)
  bad=sample(.10)
  self.assertFalse(fuse_visual_gap_samples([(False,first),(True,bad)],self.config)[0])
 def test_metric_gap_and_fixed_operator_frame(self):
  goal=parse_local('把红杯放到黄盘右边10cm',reference)
  planner=SpatialPlanner(self.config);a=planner.plan(goal,self.source,self.target,self.world)
  camera=CameraState().orbit(180/.35,0)
  b=planner.plan(goal,self.source,self.target,self.world)
  np.testing.assert_allclose(a.position,b.position);self.assertAlmostEqual(a.position[0],.25+.10+.03+.10)
  self.assertEqual(goal.spatial_constraints[0].reference_frame,'operator')
 def test_blocked_route_tries_a_distinct_valid_placement(self):
  goal=parse_local('把红杯放到黄盘右边10cm',reference)
  planner=SpatialPlanner(self.config)
  first=planner.plan(goal,self.source,self.target,self.world)
  second=planner.plan(goal,self.source,self.target,self.world,excluded_positions=[first.position])
  self.assertGreater(np.linalg.norm(second.position[:2]-first.position[:2]),.008)
  self.assertTrue(RelationValidator(self.config).evaluate(goal,self.source,self.target,self.world,second.position,require_visible=False)[0])
 def test_unreachable_candidates_are_reported_in_placement_failure(self):
  goal=parse_local('把红杯放到黄盘右边10cm',reference)
  with self.assertRaisesRegex(ValueError,'REACHABILITY'):
   SpatialPlanner(self.config).plan(goal,self.source,self.target,self.world,reachable=lambda position:False)
 def test_confirmed_carried_fragment_is_not_a_separate_placement_obstacle(self):
  goal=parse_local('把红杯放到黄盘右边10cm',reference)
  planner=SpatialPlanner(self.config)
  position=planner.plan(goal,self.source,self.target,self.world).position
  fragment=object_('fragment',position,(.025,.025,.04),'red','bottle')
  self.world.objects[fragment.id]=fragment
  self.assertFalse(planner.feasibility(self.source,self.target,self.world,position,goal.relation)[0])
  self.world.memory.held_id=self.source.id
  self.world.memory.held_fragments.add(fragment.id)
  self.assertTrue(planner.feasibility(self.source,self.target,self.world,position,goal.relation)[0])
 def test_upper_and_above_differ(self):
  upper=parse_local('把红杯放到黄盘上边8cm',reference);above=parse_local('把红杯放到黄盘上方8cm',reference)
  self.assertEqual(upper.relation,'upper_of');self.assertEqual(above.relation,'above')
  p=SpatialPlanner(self.config).plan(upper,self.source,self.target,self.world);self.assertAlmostEqual(p.position[1],.25+.10+.03+.08)
  with self.assertRaisesRegex(ValueError,'NO_SUPPORT'):SpatialPlanner(self.config).plan(above,self.source,self.target,self.world)
 def test_distance_validation_rejects_six_cm_for_ten(self):
  goal=parse_local('把红杯放到黄盘右边10cm',reference);self.source.position=np.array([.25+.10+.03+.06,.25,.04])
  ok,e=RelationValidator(self.config).evaluate(goal,self.source,self.target,self.world)
  self.assertFalse(ok);self.assertAlmostEqual(e['constraints'][0]['actual_gap_m'],.06);self.assertAlmostEqual(e['constraints'][0]['distance_error_m'],.04)
 def test_release_feedback_changes_only_next_plan(self):
  goal=parse_local('把红杯放到黄盘右边10cm',reference)
  context=SimpleNamespace(goal=goal,placement_gap_compensation_m=.016)
  planned=placement_planning_goal(context)
  self.assertAlmostEqual(goal.spatial_constraints[0].distance_m,.10)
  self.assertAlmostEqual(planned.spatial_constraints[0].distance_m,.116)
  self.assertAlmostEqual(SpatialPlanner(self.config).plan(planned,self.source,self.target,self.world).position[0],.25+.10+.03+.116)
 def test_transfer_angles_cross_minus_pi_by_short_arc(self):
  angles=short_arc_angles(2.7,-2.7)
  self.assertLess(abs(angles[-1]-angles[0]),1.)
  self.assertAlmostEqual(np.sin(angles[-1]),np.sin(-2.7))
  self.assertAlmostEqual(np.cos(angles[-1]),np.cos(-2.7))
 def test_partial_plate_view_keeps_measured_support_extent(self):
  self.source.position=np.array([.25,.25,.078])
  self.target.points=np.array([[.24+dx,.25+dy,.04] for dx in (-.01,.01) for dy in (-.04,.04)]*4)
  self.assertTrue(supported_by(self.source,self.target,self.source.position))
 def test_released_multiview_footprint_prevents_false_gap_success(self):
  goal=parse_local('把红杯放到黄盘左边10cm',reference)
  self.source.position=np.array([.02,.25,.04]);self.source.points=np.array([[.0,.25,.08],[.04,.25,.08],[.02,.24,.08]]*4)
  self.source.manipulation_geometry['verified_footprint_offsets']=[[-.05,-.03],[.05,-.03],[.05,.03],[-.05,.03]]
  ok,e=RelationValidator(self.config).evaluate(goal,self.source,self.target,self.world)
  self.assertFalse(ok)
  self.assertAlmostEqual(e['constraints'][0]['actual_gap_m'],.08)
 def test_verified_world_footprint_survives_centroid_shift(self):
  goal=parse_local('把红杯放到黄盘左边10cm',reference)
  self.source.position=np.array([.02,.25,.04])
  self.source.manipulation_geometry.update(verified_footprint_world=[[-.03,.22],[.07,.22],[.07,.28],[-.03,.28]],
      verified_footprint_anchor=[.02,.25])
  original=footprint(self.source).copy()
  self.source.position[0]+=.008
  np.testing.assert_allclose(footprint(self.source),original)
  _,e=RelationValidator(self.config).evaluate(goal,self.source,self.target,self.world)
  self.assertAlmostEqual(e['constraints'][0]['actual_gap_m'],.08)
 def test_center_distance_is_not_edge_gap(self):
  goal=parse_local('把红杯放到黄盘右边中心距离30cm',reference)
  desired=SpatialPlanner(self.config).plan(goal,self.source,self.target,self.world);self.assertAlmostEqual(desired.position[0],.55)
 def test_multiconstraint_rejection(self):
  obstacle=object_('blue',(.49,.25,.04),(.06,.06,.08),'blue');self.world.objects['blue']=obstacle
  goal=parse_local('把红杯放到黄盘右边10cm，离蓝盒至少8cm',reference)
  self.assertEqual(len(goal.spatial_constraints),2)
  with self.assertRaisesRegex(ValueError,'NO_VALID_PLACEMENT'):SpatialPlanner(self.config).plan(goal,self.source,self.target,self.world)
 def test_correction_limits_and_stability(self):
  goal=parse_local('把红杯放到黄盘右边10cm',reference);desired=SpatialPlanner(self.config).plan(goal,self.source,self.target,self.world)
  self.source.position=desired.position-np.array([.035,0,0]);self.source.stability_displacement_m=.001
  planner=PlacementCorrectionPlanner(self.config)
  self.assertEqual(planner.decide(goal,self.source,self.target,self.world,desired).strategy,'PUSH')
  self.assertEqual(planner.decide(goal,self.source,self.target,self.world,desired,push_attempts=2).reason,'INSUFFICIENT_PUSH_REACH')
  self.assertEqual(planner.decide(goal,self.source,self.target,self.world,desired,push_attempts=3).strategy,'REGRASP')
  self.source.position[0]-=.15;self.assertEqual(planner.decide(goal,self.source,self.target,self.world,desired).strategy,'REGRASP')
  self.source.position=desired.position-np.array([.035,0,0]);self.source.size[2]=.3
  self.assertEqual(planner.decide(goal,self.source,self.target,self.world,desired).reason,'TIPPING_RISK')
 def test_regrasp_feedback_precedes_multi_step_push(self):
  goal=parse_local('把红杯放到黄盘右边10cm',reference)
  desired=SpatialPlanner(self.config).plan(goal,self.source,self.target,self.world)
  self.source.position=desired.position-np.array([.022,0,0]);self.source.stability_displacement_m=.001
  decision=PlacementCorrectionPlanner(self.config).decide(goal,self.source,self.target,self.world,desired,regrasps=1)
  self.assertEqual((decision.strategy,decision.reason),('REGRASP','RELEASE_FEEDBACK_REGRASP'))
 def test_no_push_outside_observed_support(self):
  goal=parse_local('把红杯放到黄盘右边10cm',reference);desired=SpatialPlanner(self.config).plan(goal,self.source,self.target,self.world)
  self.source.position=desired.position-np.array([.025,0,0]);self.source.stability_displacement_m=0.
  self.world.support_points=self.world.support_points[self.world.support_points[:,0]<desired.position[0]]
  decision=PlacementCorrectionPlanner(self.config).decide(goal,self.source,self.target,self.world,desired)
  self.assertEqual(decision.strategy,'REGRASP')
 def test_explicit_constraints_and_unknown_object_front(self):
  goal=parse_local('把红杯放到黄盘右边10cm，保持竖直，不滑动',reference)
  self.assertEqual(set(goal.manipulation.explicit),{'keep_upright','no_sliding'})
  self.assertEqual(parse_local('把红杯放到黄盘右边10cm',reference).manipulation.explicit,[])
  ambiguous=parse_local('把红杯放到黄盘正前方10cm',reference)
  with self.assertRaisesRegex(ValueError,'AMBIGUOUS'):SpatialPlanner(self.config).plan(ambiguous,self.source,self.target,self.world)
 def test_llm_cannot_emit_coordinates_or_invalid_bbox(self):
  with self.assertRaises(ValueError):GraspStrategyDecision.model_validate({'selected_candidate_id':'x','strategy':'top_grasp','preferred_region':'top','approach_direction':'top','reason_code':'TOP_FEASIBLE','xyz':[0,0,1]})
  with self.assertRaises(ValueError):GroundedTarget(found=True,label='cup',bbox=(.1,.1,1.2,.4),confidence=.9)
 def test_frame_calibration_roundtrip(self):
  frame=OperatorFrame((.1,.2,0),90);point=np.array([.3,.4,.05]);np.testing.assert_allclose(frame.to_operator(frame.to_world(point)),point)

if __name__=='__main__':unittest.main()
