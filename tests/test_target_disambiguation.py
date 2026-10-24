"""The camera and image sent for ambiguity review use actual detail pixels."""
import base64
import io
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from config import AgentConfig
from simulation.environment import Simulation
from simulation.mujoco_camera import MujocoCamera
from robot.mujoco_robot import MuJoCoRobot
from world.world_model import WorldModel,WorldObject,AmbiguousReferenceError,ReferenceNotFoundError
from agent.task_log import TaskLog
from perception.active_perception import ActivePerception
from perception.visual_grounder import VisualGrounder
from llm.llm_client import LLMClient
from llm.schemas import Reference,VisualDisambiguation,CandidateJudgement,GroundedTarget
from skills.observation import FindObjectSkill
from recovery.failure_classifier import SkillFailure


class ReviewClient:
    online=False
    def __init__(self):self.calls=0
    def complete(self,schema,purpose,payload,image=None):
        self.calls+=1
        assert schema is VisualDisambiguation
        assert purpose=='visual_disambiguation'
        assert image.shape==(960,1280,3)
        assert payload['candidate_ids']==['A','B']
        return VisualDisambiguation(candidates=[
            CandidateJudgement(candidate_id='A',verdict='no',confidence=.95,evidence='shallow dish'),
            CandidateJudgement(candidate_id='B',verdict='yes',confidence=.95,evidence='vessel with handle')])


class TargetDisambiguationTests(unittest.TestCase):
    def test_detail_camera_and_measured_candidate_review(self):
        cfg=AgentConfig();cfg.localization.grounding_mode='local'
        with tempfile.TemporaryDirectory() as directory:
            cfg.log_root=directory
            with Simulation(cfg) as sim:
                camera=MujocoCamera(sim,MuJoCoRobot(sim))
                coarse=camera.capture();detail=camera.capture_detail()
                self.assertEqual(coarse.rgb.shape,(480,640,3))
                self.assertEqual(detail.rgb.shape,(960,1280,3))
                np.testing.assert_allclose(coarse.camera_to_world,detail.camera_to_world)
                self.assertAlmostEqual(detail.intrinsics[0,0],2*coarse.intrinsics[0,0])

                world=WorldModel(cfg.perception);client=ReviewClient()
                log=TaskLog(directory,'white cup detail review',cfg)
                perception=ActivePerception(camera,world,cfg,client,log)
                reference=Reference(description='白色杯子')
                perception.scan([reference],force_all=False)
                scans=[e for e in log.events if e['type']=='multiview_fusion']
                self.assertEqual(scans[-1]['data']['views'],6)
                with self.assertRaises(AmbiguousReferenceError):world.resolve(reference,require_visible=True)
                candidates=[o for o in world.objects.values() if o.visible and o.color=='white' and o.category=='cup']
                self.assertEqual(len(candidates),2)
                cfg.localization.grounding_mode='auto';client.online=True
                self.assertTrue(perception.disambiguate(reference,candidates))
                self.assertEqual(world.resolve(reference,require_visible=True).id,sorted(o.id for o in candidates)[1])
                self.assertEqual(client.calls,1)
                self.assertEqual(len([e for e in log.events if e['type']=='detail_capture']),1)

    def test_no_candidate_and_multiple_candidates_are_distinct(self):
        world=WorldModel(AgentConfig().perception)
        with self.assertRaises(ReferenceNotFoundError):world.resolve(Reference(description='white cup'))

    def test_measured_bottle_shape_can_correct_previous_cup_memory(self):
        world=WorldModel(AgentConfig().perception)
        def observed(category,stamp):
            return WorldObject('',f'white {category}','white',category,np.array([.25,-.44,.10]),
                np.array([0,0,0,1]),np.array([.055,.058,.20]),.95,stamp,view_quality=.9)
        world.update([observed('cup',1)],1)
        identity=next(iter(world.objects))
        world.update([observed('bottle',2)],2)
        self.assertEqual(world.objects[identity].category,'bottle')

    def test_single_shallow_round_candidate_requires_detail_confirmation(self):
        cfg=AgentConfig();world=WorldModel(cfg.perception)
        obj=WorldObject('', 'white cup','white','cup',np.array([.2,.1,.022]),
            np.array([0,0,0,1]),np.array([.10,.10,.044]),.95,1.)
        world.update([obj],1.)
        reference=Reference(description='白色杯子')
        calls=[]
        perception=SimpleNamespace(identify=lambda ref:world.resolve(ref,require_visible=True),
                                   disambiguate=lambda ref,objects:calls.append(objects) or False)
        context=SimpleNamespace(world=world,perception=perception,
            goal=SimpleNamespace(object=reference,target=None),config=cfg)
        with self.assertRaises(SkillFailure) as failure:FindObjectSkill().execute(context)
        self.assertEqual(failure.exception.failure.value,'LOW_CONFIDENCE')
        self.assertTrue(failure.exception.evidence['detail_reviewed'])
        self.assertEqual(len(calls),1)

    def test_user_reference_can_bind_a_differently_classified_object(self):
        cfg=AgentConfig();world=WorldModel(cfg.perception)
        red_vessel=WorldObject('', 'red bottle','red','bottle',np.array([.38,-.18,.05]),
            np.array([0,0,0,1]),np.array([.07,.07,.10]),.95,1.)
        red_box=WorldObject('', 'red box','red','box',np.array([.50,-.39,.04]),
            np.array([0,0,0,1]),np.array([.09,.09,.08]),.95,1.)
        world.update([red_vessel,red_box],1.)
        red_vessel.query='红色杯子';red_vessel.semantic_confidence=.95
        reference=Reference(description='红色杯子')
        events=[]
        perception=ActivePerception.__new__(ActivePerception)
        perception.world=world;perception.log=SimpleNamespace(event=lambda name,data:events.append((name,data)))
        def review(ref,candidates):
            self.assertEqual({o.id for o in candidates},{red_vessel.id,red_box.id})
            world.bind_reference(ref,red_vessel.id,{'source':'high_resolution_visual_review'})
            return True
        perception.disambiguate=review
        chosen=perception.identify(reference)
        self.assertEqual(chosen.id,red_vessel.id)
        self.assertEqual(chosen.category,'bottle')
        self.assertEqual(world.resolve(reference,require_visible=True).id,red_vessel.id)
        self.assertEqual(events[0][1]['reason'],'category_conflict')

    def test_related_colour_without_visual_confirmation_never_autobinds(self):
        world=WorldModel(AgentConfig().perception)
        only_box=WorldObject('', 'red box','red','box',np.array([.3,.1,.04]),
            np.array([0,0,0,1]),np.array([.08,.08,.08]),.95,1.)
        world.update([only_box],1.)
        perception=ActivePerception.__new__(ActivePerception)
        perception.world=world;perception.log=SimpleNamespace(event=lambda *args:None)
        perception.disambiguate=lambda ref,candidates:False
        with self.assertRaises(AmbiguousReferenceError):perception.identify(Reference(description='红色杯子'))
        self.assertFalse(world.reference_bindings)

    def test_find_skill_uses_verified_identity_across_category_difference(self):
        cfg=AgentConfig();world=WorldModel(cfg.perception)
        vessel=WorldObject('', 'red bottle','red','bottle',np.array([.38,-.18,.05]),
            np.array([0,0,0,1]),np.array([.07,.07,.10]),.95,1.)
        plate=WorldObject('', 'yellow plate','yellow','plate',np.array([.34,.30,.018]),
            np.array([0,0,0,1]),np.array([.23,.23,.036]),.95,1.,movable=False)
        world.update([vessel,plate],1.)
        perception=ActivePerception.__new__(ActivePerception)
        perception.world=world;perception.log=SimpleNamespace(event=lambda *args:None)
        calls=[]
        def review(ref,candidates):
            calls.append(ref.description)
            world.bind_reference(ref,vessel.id,{'source':'high_resolution_visual_review'})
            return True
        perception.disambiguate=review
        goal=SimpleNamespace(object=Reference(description='红色杯子'),
                             target=Reference(description='黄色盘子'),relation='on')
        context=SimpleNamespace(world=world,perception=perception,goal=goal,config=cfg,
                                source=None,target=None)
        self.assertTrue(FindObjectSkill().execute(context))
        self.assertEqual(calls,['红色杯子'])
        self.assertEqual((context.source_id,context.target_id),(vessel.id,plate.id))
        self.assertEqual(context.source.category,'bottle')
        self.assertTrue(world.memory.mission_active)

    def test_exact_category_does_not_hide_conflicting_visual_grounding(self):
        world=WorldModel(AgentConfig().perception)
        cup=WorldObject('', 'red cup','red','cup',np.array([.2,.1,.05]),
            np.array([0,0,0,1]),np.array([.07,.07,.10]),.95,1.)
        vessel=WorldObject('', 'red bottle','red','bottle',np.array([.4,.1,.05]),
            np.array([0,0,0,1]),np.array([.07,.07,.10]),.95,1.)
        world.update([cup,vessel],1.)
        vessel.query='红色杯子';vessel.semantic_confidence=.95
        perception=ActivePerception.__new__(ActivePerception)
        perception.world=world;perception.log=SimpleNamespace(event=lambda *args:None)
        seen=[]
        perception.disambiguate=lambda ref,candidates:seen.append({o.id for o in candidates}) or False
        with self.assertRaises(AmbiguousReferenceError):perception.identify(Reference(description='红色杯子'))
        self.assertEqual(seen,[{cup.id,vessel.id}])

    def test_multiview_category_evidence_also_triggers_review(self):
        world=WorldModel(AgentConfig().perception)
        cup=WorldObject('', 'white cup','white','cup',np.array([.2,.1,.05]),
            np.array([0,0,0,1]),np.array([.07,.07,.10]),.95,1.)
        vessel=WorldObject('', 'white bottle','white','bottle',np.array([.4,.1,.05]),
            np.array([0,0,0,1]),np.array([.07,.07,.10]),.95,1.,
            category_observations={'view:0':{'cup':.6},'view:1':{'cup':.6,'bottle':1.5}})
        world.update([cup,vessel],1.)
        perception=ActivePerception.__new__(ActivePerception)
        perception.world=world;perception.log=SimpleNamespace(event=lambda *args:None)
        seen=[];perception.disambiguate=lambda ref,candidates:seen.append({o.id for o in candidates}) or False
        with self.assertRaises(AmbiguousReferenceError):perception.identify(Reference(description='白色杯子'))
        self.assertEqual(seen,[{cup.id,vessel.id}])

    def test_verified_binding_tracks_identity_after_reclassification(self):
        world=WorldModel(AgentConfig().perception)
        vessel=WorldObject('', 'red bottle','red','bottle',np.array([.38,-.18,.05]),
            np.array([0,0,0,1]),np.array([.07,.07,.10]),.95,1.)
        world.update([vessel],1.)
        reference=Reference(description='红色杯子')
        world.bind_reference(reference,vessel.id,{'source':'high_resolution_visual_review'})
        vessel.category='box';vessel.color='orange'
        self.assertIs(world.resolve(reference,require_visible=True),vessel)
        vessel.visible=False
        with self.assertRaises(ReferenceNotFoundError):world.resolve(reference,require_visible=True)
        self.assertFalse(world.reference_bindings)

    def test_independent_views_can_correct_a_provisional_category(self):
        world=WorldModel(AgentConfig().perception)
        def seen(category,view,scores,width):
            return WorldObject('',f'red {category}','red',category,np.array([.38,-.18,.05]),
                np.array([0,0,0,1]),np.array([width,width,.10]),.95,float(view+1),
                view_quality=.99,category_observations={f'gripper_camera:{view}':scores})
        first=seen('bottle',0,{'bottle':.6,'cup':.855},.049)
        world.update([first],1.)
        identity=first.id
        world.update([seen('cup',1,{'cup':.6},.065)],2.)
        world.update([seen('cup',2,{'cup':1.455},.070)],3.)
        self.assertEqual(world.objects[identity].category,'cup')
        self.assertEqual(len(world.objects[identity].category_observations),3)

    def test_review_image_is_transmitted_without_coarse_downsampling(self):
        cfg=AgentConfig().llm;cfg.mode='required'
        client=LLMClient(cfg);client.api_key='test-key'
        observed={}
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return json.dumps({'choices':[{'message':{'content':json.dumps({
                'candidates':[{'candidate_id':'A','verdict':'uncertain','confidence':.5,'evidence':'occluded'}]})}}]}).encode()
        def fake_urlopen(request,timeout):
            payload=json.loads(request.data)
            data=payload['messages'][1]['content'][1]['image_url']['url'].split(',',1)[1]
            observed['size']=Image.open(io.BytesIO(base64.b64decode(data))).size
            return Response()
        with patch('llm.llm_client.urlopen',fake_urlopen):
            client.complete(VisualDisambiguation,'visual_disambiguation',{'candidate_ids':['A']},
                            image=np.zeros((960,1280,3),dtype=np.uint8))
        self.assertEqual(observed['size'],(1280,960))

    def test_out_of_frame_model_box_gets_one_bounded_coordinate_retry(self):
        cfg=AgentConfig();events=[]
        class Client:
            online=True
            def __init__(self):self.calls=0
            def complete(self,schema,purpose,payload,image=None):
                self.calls+=1
                if self.calls==1:
                    return GroundedTarget(found=True,label='红色杯子',color='red',category='cup',
                                          bbox=(550,390,638,510),coordinate_space='pixel',confidence=.95)
                self.assert_retry=payload['instruction'].startswith('Retry with coordinate_space')
                return GroundedTarget(found=True,label='红色杯子',color='red',category='cup',
                                      bbox=(.28,.35,.38,.48),coordinate_space='normalized',confidence=.95)
        client=Client()
        grounder=VisualGrounder(client,cfg,SimpleNamespace(event=lambda kind,data:events.append(kind)))
        result=grounder.ground(np.zeros((480,640,3),dtype=np.uint8),'红色杯子')
        self.assertTrue(result.found)
        self.assertEqual(result.bbox,(179,168,243,230))
        self.assertEqual((client.calls,grounder.calls),(2,2))
        self.assertTrue(client.assert_retry)
        self.assertIn('visual_grounding_retry',events)


if __name__=='__main__':unittest.main()
