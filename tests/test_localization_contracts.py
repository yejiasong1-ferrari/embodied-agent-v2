"""Fresh RGB-D and persistent identity are requirements, never optional labels."""
import unittest
from types import SimpleNamespace
import numpy as np
from config import AgentConfig
from perception.object_localizer import ObjectLocalizer
from perception.camera import CameraFrame
from world.world_model import WorldObject
from world.object_memory import ObjectMemory

class LocalizationContracts(unittest.TestCase):
 def setUp(self):
  self.config=AgentConfig();self.local=ObjectLocalizer(self.config,SimpleNamespace(event=lambda *a:None))
  self.mask=np.ones((10,10),bool);self.detection=SimpleNamespace(bbox=(1,1,9,9),confidence=.95)
 def obj(self,t=1.,category='box',quality=.99):
  return WorldObject('target','green '+category,'green',category,np.array([.4,.2,.04]),np.array([0,0,0,1.]),np.array([.06,.06,.08]),.95,t,uncertainty_m=.004,view_quality=quality)
 def observe(self,t,depth=.35,phase='local'):
  o=self.obj(t);f=CameraFrame(np.zeros((10,10,3),np.uint8),np.full((10,10),depth),np.eye(3),np.eye(4),t,'wrist:0')
  return self.local.update(o,f,self.detection,self.mask,'green box',phase)
 def test_two_fresh_local_frames_required(self):
  self.assertEqual(self.observe(1,phase='coarse').localization_state,'COARSE_LOCKED')
  self.assertEqual(self.observe(2).localization_state,'COARSE_LOCKED')
  o=self.observe(2.1);self.assertTrue(self.local.precise(o,2.2));self.assertFalse(self.local.precise(o,8.))
 def test_stale_or_far_frame_cannot_raise_precision(self):
  self.observe(1);self.assertEqual(self.observe(10).localization_state,'COARSE_LOCKED')
  self.observe(11,depth=1.);self.assertEqual(self.observe(11.1).localization_state,'COARSE_LOCKED')
 def test_invalid_depth_never_locks(self):
  self.observe(1);o=self.observe(1.1,depth=float('nan'));self.assertEqual(o.valid_depth_ratio,0.);self.assertFalse(self.local.precise(o,1.1))
 def test_retained_identity_is_not_retained_visibility(self):
  memory=ObjectMemory(self.config.perception);o=self.obj();memory.objects[o.id]=o;memory.retained_ids.add(o.id)
  memory.update([],100.);self.assertIn(o.id,memory.objects);self.assertFalse(o.visible)
 def test_tiny_camera_tilt_does_not_change_semantic_class(self):
  memory=ObjectMemory(self.config.perception);o=self.obj();o.query='green box';memory.objects[o.id]=o
  other=self.obj(2,'bottle',.991);memory.update([other],2.)
  self.assertEqual(other.id,'target');self.assertEqual(other.category,'box')
 def test_fixed_obstacle_preserves_unseen_volume(self):
  memory=ObjectMemory(self.config.perception);o=self.obj();o.category='obstacle';o.movable=False;o.size=np.array([.28,.28,.25]);o.position=np.array([0.,.48,.125]);memory.objects[o.id]=o
  top=self.obj(2,category='obstacle');top.movable=False;top.position=np.array([0.,.50,.245]);top.size=np.array([.20,.20,.01]);memory.update([top],2.)
  self.assertLessEqual(top.low[2],.001);self.assertLessEqual(top.low[1],.34);self.assertGreaterEqual(top.high[2],.25)
 def test_full_plate_footprint_does_not_grow_from_small_view_jitter(self):
  memory=ObjectMemory(self.config.perception);plate=self.obj(category='plate');plate.movable=False
  plate.position=np.array([.34,.30,.018]);plate.size=np.array([.238,.238,.036]);plate.id='yellow_plate_001'
  memory.objects[plate.id]=plate
  for index in range(8):
   seen=self.obj(2+index,category='plate');seen.movable=False
   seen.position=np.array([.342,.299,.018]);seen.size=np.array([.244,.243,.036])
   memory.update([seen],2+index)
  self.assertEqual(list(memory.objects),[plate.id])
  np.testing.assert_allclose(memory.objects[plate.id].size[:2],[.238,.238])
  np.testing.assert_allclose(memory.objects[plate.id].position[:2],[.34,.30])
 def test_cropped_face_cannot_sink_remembered_obstacle(self):
  memory=ObjectMemory(self.config.perception);old=self.obj();memory.objects[old.id]=old
  face=self.obj(2);face.position[2]=.025;face.size[2]=.05;face.observation_clipped=True
  memory.update([face],2.)
  self.assertAlmostEqual(face.high[2],.08);self.assertAlmostEqual(face.low[2],0.)
  face.localization_state='PRECISE_LOCKED';face.valid_depth_ratio=1.
  self.assertFalse(self.local.precise(face,2.1))
 def test_side_view_can_reveal_previously_hidden_height(self):
  memory=ObjectMemory(self.config.perception);old=self.obj();old.position[2]=.12;memory.objects[old.id]=old
  side=self.obj(2);side.size[2]=.16;side.position[2]=.08
  memory.update([side],2.)
  self.assertAlmostEqual(side.size[2],.16);self.assertAlmostEqual(side.low[2],0.)
 def test_release_above_support_retains_pregrasp_height(self):
  memory=ObjectMemory(self.config.perception);old=self.obj();old.position[2]=.05;old.size[2]=.10
  memory.objects[old.id]=old;memory.grasped_heights[old.id]=.10
  after=self.obj(2);after.position[2]=.068;after.size[2]=.136
  memory.update([after],2.)
  self.assertAlmostEqual(after.size[2],.10)
  self.assertAlmostEqual(after.low[2],.036)
 def test_held_prediction_associates_pixels_but_never_creates_visibility(self):
  memory=ObjectMemory(self.config.perception);old=self.obj();memory.objects[old.id]=old;memory.held_id=old.id
  memory.held_expected_position=old.position+np.array([.15,0,0])
  observed=self.obj(2);observed.position+=np.array([.15,0,0]);memory.update([observed],2.)
  self.assertEqual(observed.id,'target');self.assertTrue(observed.visible)
  memory.held_expected_position+=np.array([.10,0,0]);memory.update([],3.)
  self.assertFalse(memory.objects['target'].visible)
  np.testing.assert_allclose(memory.objects['target'].position,[.55,.2,.04])
 def test_nested_rgb_fragments_keep_one_physical_object_id(self):
  memory=ObjectMemory(self.config.perception);main=self.obj();main.id='green_box_012';memory.objects[main.id]=main;memory.retained_ids.add(main.id)
  full=self.obj(2);part=self.obj(2);part.position[:2]+=[-.018,.003];part.size[:2]=[.022,.041]
  memory.update([full,part],2.)
  matches=[o for o in memory.objects.values() if o.color=='green' and o.movable]
  self.assertEqual([o.id for o in matches],[main.id])
  self.assertTrue(matches[0].visible)
 def test_nearby_distinct_objects_do_not_coalesce(self):
  memory=ObjectMemory(self.config.perception)
  left=self.obj();right=self.obj();right.position[0]+=.075
  memory.update([left,right],2.)
  self.assertEqual(len(memory.objects),2)
 def test_old_fragment_track_is_removed_before_collision_planning(self):
  memory=ObjectMemory(self.config.perception)
  source=self.obj();source.id='green_box_012';fragment=self.obj();fragment.id='green_box_060'
  fragment.position[0]-=.015;fragment.size[:2]=[.025,.04]
  memory.objects[source.id]=source;memory.objects[fragment.id]=fragment
  memory.retained_ids.add(source.id)
  memory.update([],2.)
  self.assertEqual(list(memory.objects),[source.id])
 def test_partial_tall_top_does_not_create_a_second_obstacle(self):
  memory=ObjectMemory(self.config.perception)
  source=self.obj();source.id='blue_cup_020';source.color='blue';source.category='cup';source.semantic_label='blue cup'
  source.position=np.array([-.5,-.23,.08]);source.size=np.array([.10,.10,.16]);memory.objects[source.id]=source
  memory.retained_ids.add(source.id)
  partial=self.obj(2,category='bottle');partial.color='blue';partial.semantic_label='blue bottle'
  partial.position=np.array([-.51,-.19,.08]);partial.size=np.array([.08,.04,.16])
  full=self.obj(2,category='cup');full.color='blue';full.semantic_label='blue cup'
  full.position=np.array([-.5,-.23,.054]);full.size=np.array([.10,.10,.16])
  memory.update([full,partial],2.)
  self.assertEqual(list(memory.objects),[source.id])
  self.assertAlmostEqual(memory.objects[source.id].high[2],.16)
 def test_nested_fixed_obstacle_strip_does_not_expand_collision_volume(self):
  memory=ObjectMemory(self.config.perception)
  full=self.obj();full.id='red_obstacle_002';full.color='red';full.category='obstacle';full.movable=False
  full.position=np.array([.496,-.383,.0375]);full.size=np.array([.157,.209,.075]);memory.objects[full.id]=full
  whole=self.obj(2);whole.color='red';whole.category='obstacle';whole.movable=False
  whole.position=full.position.copy();whole.size=full.size.copy()
  strip=self.obj(2);strip.color='red';strip.category='obstacle';strip.movable=False
  strip.position=np.array([.45,-.39,.0375]);strip.size=np.array([.048,.19,.075])
  memory.update([whole,strip],2.)
  self.assertEqual(list(memory.objects),[full.id])
  np.testing.assert_allclose(memory.objects[full.id].size,full.size)
 def test_fixed_obstacle_top_slice_is_not_a_second_plate(self):
  memory=ObjectMemory(self.config.perception)
  obstacle=self.obj();obstacle.id='red_obstacle_002';obstacle.color='red';obstacle.category='obstacle';obstacle.movable=False
  obstacle.position=np.array([.496,-.383,.0375]);obstacle.size=np.array([.157,.209,.075]);memory.objects[obstacle.id]=obstacle
  whole=self.obj(2);whole.color='red';whole.category='obstacle';whole.movable=False
  whole.position=obstacle.position.copy();whole.size=obstacle.size.copy()
  top=self.obj(2);top.color='red';top.category='plate';top.movable=False
  top.position=np.array([.443,-.389,.058]);top.size=np.array([.203,.203,.034])
  memory.update([whole,top],2.)
  self.assertEqual(list(memory.objects),[obstacle.id])
  np.testing.assert_allclose(memory.objects[obstacle.id].size,obstacle.size)
 def test_fixed_obstacle_fragment_misclassified_as_movable_is_merged(self):
  memory=ObjectMemory(self.config.perception)
  obstacle=self.obj();obstacle.id='red_obstacle_002';obstacle.color='red';obstacle.category='obstacle';obstacle.movable=False
  obstacle.position=np.array([.499,-.401,.0375]);obstacle.size=np.array([.149,.198,.075]);memory.objects[obstacle.id]=obstacle
  whole=self.obj(2);whole.color='red';whole.category='obstacle';whole.movable=False
  whole.position=obstacle.position.copy();whole.size=obstacle.size.copy()
  fragment=self.obj(2,category='bottle');fragment.color='red';fragment.movable=True
  fragment.position=np.array([.45,-.401,.0375]);fragment.size=np.array([.105,.075,.075])
  memory.update([whole,fragment],2.)
  self.assertEqual(list(memory.objects),[obstacle.id])
  self.assertFalse(memory.objects[obstacle.id].movable)

if __name__=='__main__':unittest.main()
