import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from config import AgentConfig
from llm.schemas import Reference,SymbolicPlan,PlanStep,Goal
from agent.task_planner import TaskPlanner,MANIPULATION_SEQUENCE
from perception.camera import CameraFrame
from perception.depth_components import split_mask_3d
from recovery.failure_classifier import SkillFailure
from recovery.recovery_manager import RecoveryManager
from validation.safety_monitor import SafetyMonitor
from robot.robot_interface import Pose,ContactState
from world.world_model import WorldModel
from test_agent_geometry import obj


class Contracts(unittest.TestCase):
    def test_grasp_clearance_distinguishes_neighbouring_finger_lanes(self):
        from manipulation.grasp_planner import GraspPlanner
        world=WorldModel(AgentConfig().perception)
        source=obj('box',(.38,-.095,.04),(.068,.058,.08),'blue','box')
        neighbour=obj(position=(.38,-.18,.05))
        world.objects={source.id:source,neighbour.id:neighbour}
        candidates={c.id:c for c in GraspPlanner(AgentConfig()).candidates(source,world)}
        self.assertLess(candidates['top_grasp:0'].clearance,candidates['top_grasp:1'].clearance)

    def test_grasp_force_failure_is_terminal_collision_risk(self):
        from skills.manipulation import GraspSkill
        config=AgentConfig();config.localization.enabled=False
        context=SimpleNamespace(config=config,log=Mock(),safety=SimpleNamespace(allow_finger_contact=False,last_failure='force_limit',check=lambda:None),
                                robot=Mock(),candidate=SimpleNamespace(id='test-candidate',grasp_width=.06),grasp_attempts=0)
        context.robot.close_gripper.return_value=False
        with self.assertRaises(SkillFailure) as raised: GraspSkill().execute(context)
        self.assertEqual(raised.exception.failure.value,'COLLISION_RISK')

    def test_grasp_stability_needs_persistent_bilateral_contact(self):
        from validation.grasp_validator import GraspValidator
        validator=GraspValidator(AgentConfig())
        samples=[{'left_force':2.,'right_force':2.,'width':.059},
                 {'left_force':2.,'right_force':0.,'width':.059},
                 {'left_force':2.,'right_force':2.,'width':.059}]
        self.assertFalse(validator.stable_contact(samples)[0])
        samples[1]['right_force']=1.5
        self.assertTrue(validator.stable_contact(samples)[0])
        samples[2]['width']=.063
        self.assertFalse(validator.stable_contact(samples)[0])

    def test_probe_requires_visible_object_to_follow_small_lift(self):
        from validation.grasp_validator import GraspValidator
        validator=GraspValidator(AgentConfig())
        source=obj(position=(.3,0,.04));before=source.position.copy()
        source.position[2]+=.006
        self.assertTrue(validator.probe_lifted(before,source,np.array([0,0,.007]))[0])
        source.position[2]=before[2]
        self.assertFalse(validator.probe_lifted(before,source,np.array([0,0,.007]))[0])

    def test_final_validation_needs_two_detections_in_one_view(self):
        from skills.manipulation import VerifyTaskSkill
        source=obj(position=(.3,0,.09))
        target=obj('plate',(.3,0,.02),(.24,.24,.04),'yellow','plate')
        goal=Goal(goal='place_on',object=Reference(description='red cup'),target=Reference(description='yellow plate'))
        camera=SimpleNamespace(viewpoint_count=2,move_viewpoint=Mock(return_value=True))
        context=SimpleNamespace(source=source,target=target,source_id=source.id,target_id=target.id,
            world=SimpleNamespace(regions={}),goal=goal,held=False,config=AgentConfig(),robot=Mock(),log=Mock(),
            perception=SimpleNamespace(camera=camera),final_validated=False)
        context.refresh=Mock(side_effect=[[],[source,target],[source],[source,target],[source,target]])
        accepted,evidence=VerifyTaskSkill().measure(context)
        self.assertTrue(accepted)
        self.assertTrue(evidence["accepted"])
        self.assertEqual(camera.move_viewpoint.call_count,2)

    def test_observed_table_region_uses_fresh_depth_support(self):
        from skills.manipulation import VerifyTaskSkill
        source=obj(position=(.3,0,.04),size=(.06,.06,.08))
        target=obj('staging',(.3,0,-.002),(.13,.13,.004),'region','region')
        goal=Goal(goal='place_on',object=Reference(description='red cup'),target=Reference(description='free table region'))
        camera=SimpleNamespace(viewpoint_count=1,move_viewpoint=Mock(return_value=True))
        context=SimpleNamespace(source=source,target=target,source_id=source.id,target_id=target.id,
            world=SimpleNamespace(regions={target.id:target}),goal=goal,held=False,config=AgentConfig(),
            robot=Mock(),log=Mock(),perception=SimpleNamespace(camera=camera),final_validated=False)
        context.refresh=Mock(side_effect=[[],[source],[source]])
        accepted,evidence=VerifyTaskSkill().measure(context)
        self.assertTrue(accepted)
        self.assertTrue(evidence["fresh_visual_evidence"])

    def test_llm_chinese_labels_are_normalized(self):
        ref=Reference(description='红色杯子',color='红色',category='杯子')
        self.assertEqual((ref.color,ref.category),('red','cup'))
        self.assertEqual((Reference(description='red cup').color,Reference(description='red cup').category),('red','cup'))
        self.assertEqual((Reference(description='黄色盘子').color,Reference(description='黄色盘子').category),('yellow','plate'))

    def test_llm_cannot_insert_unverified_manipulation(self):
        client=Mock(online=True); client.config.mode='required'
        client.complete.return_value=SymbolicPlan(steps=[PlanStep(skill='MOVE_TO_TARGET')]+[PlanStep(skill=n) for n in MANIPULATION_SEQUENCE])
        planner=TaskPlanner(client,Mock())
        with self.assertRaises(ValueError):
            planner.plan(Goal(goal='place_on',object=Reference(description='cup'),target=Reference(description='plate')),WorldModel(AgentConfig().perception))

    def test_llm_inferred_handle_is_not_a_required_grasp_region(self):
        from agent.task_interpreter import TaskInterpreter
        goal=Goal(goal='place_right',object=Reference(description='红色杯子'),target=Reference(description='黄色盘子'))
        goal.manipulation.grasp_specific_region='handle'
        client=Mock(online=True);client.complete.return_value=goal
        log=Mock()
        interpreted=TaskInterpreter(client,log).understand('把红色杯子放到黄色盘子右边10cm')
        self.assertIsNone(interpreted.manipulation.grasp_specific_region)
        log.event.assert_called_once_with('unrequested_grasp_region_ignored',{'region':'handle'})

        goal.manipulation.grasp_specific_region='handle'
        interpreted=TaskInterpreter(client,Mock()).understand('抓住红色杯子的把手，放到黄色盘子右边10cm')
        self.assertEqual(interpreted.manipulation.grasp_specific_region,'handle')

    def test_llm_cannot_invent_placement_distance(self):
        from agent.task_interpreter import TaskInterpreter
        from llm.schemas import SpatialConstraint
        goal=Goal(goal='place_right',object=Reference(description='蓝色盒子'),
            target=Reference(description='绿色积木'),
            spatial_constraints=[SpatialConstraint(relation='right_of',distance_m=.10)])
        client=Mock(online=True);client.complete.return_value=goal
        log=Mock()
        interpreted=TaskInterpreter(client,log).understand('把蓝色盒子放到绿色积木右边')
        self.assertIsNone(interpreted.spatial_constraints[0].distance_m)
        log.event.assert_any_call('unrequested_distance_ignored',{
            'distance_m':.10,'constraint_type':'relative_position'})
        client.complete.return_value=goal.model_copy(deep=True)
        client.complete.return_value.spatial_constraints[0].distance_m=.10
        interpreted=TaskInterpreter(client,Mock()).understand('把蓝色盒子放到绿色积木右边10厘米')
        self.assertAlmostEqual(interpreted.spatial_constraints[0].distance_m,.10)

    def test_no_detections_does_not_mean_observed_free_space(self):
        world=WorldModel(AgentConfig().perception)
        with self.assertRaises(LookupError): TaskPlanner(Mock(),Mock()).free_region(obj(),world,AgentConfig().safety.workspace)
        x,y=np.meshgrid(np.arange(-.7,.71,.01),np.arange(-.7,.71,.01))
        world.update_support(np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size))),1)
        region=TaskPlanner(Mock(),Mock()).free_region(obj(),world,AgentConfig().safety.workspace)
        self.assertEqual(region.evidence_source,'observed_support_region')
        self.assertAlmostEqual(region.high[2],0)
        world.regions[region.id]=region
        world.update_support(np.empty((0,3)),2)
        self.assertFalse(region.visible)

    def test_touching_rgb_regions_separate_using_depth(self):
        depth=np.ones((30,80)); depth[:,40:]=2
        k=np.array([[100,0,40],[0,100,15],[0,0,1.]])
        frame=CameraFrame(np.zeros((30,80,3),np.uint8),depth,k,np.eye(4),1,'test')
        masks=split_mask_3d(frame,np.ones_like(depth,bool))
        self.assertEqual(len(masks),2)
        self.assertEqual(sorted(np.count_nonzero(m) for m in masks),[1200,1200])

    def test_close_differently_colored_large_surfaces_do_not_merge(self):
        world=WorldModel(AgentConfig().perception)
        world.update([obj(),obj('plate',(.31,0,.05),(.24,.24,.01),'yellow','plate')],1)
        self.assertEqual(len(world.objects),2)

    def test_user_stop_is_terminal_and_never_sent_to_llm(self):
        replanner=Mock(); replanner.client.online=True
        recovery=RecoveryManager(AgentConfig(),replanner,Mock())
        context=SimpleNamespace(candidate=None,current_skill='MOVE_TO_TARGET',source=None,robot=Mock())
        context.robot.get_contact_state.return_value=ContactState()
        self.assertEqual(recovery.decide(SkillFailure('USER_STOP'),context),'ABORT')
        replanner.reason.assert_not_called()

    def test_force_and_carried_tilt_are_runtime_vetoes(self):
        config=AgentConfig(); robot=Mock()
        robot.joint_limits=(np.full(7,-3.),np.full(7,3.)); robot.get_joint_positions.return_value=np.zeros(7)
        robot.get_ee_pose.return_value=Pose(np.array([.3,0,.4]),np.diag([1,-1,-1]))
        robot.get_contact_state.return_value=ContactState(25,2)
        monitor=SafetyMonitor(robot,config); monitor.allow_finger_contact=True
        self.assertEqual(monitor.check(),'force_limit')
        robot.get_contact_state.return_value=ContactState(2,2)
        robot.get_ee_pose.return_value=Pose(np.array([.3,0,.4]),np.eye(3))
        self.assertEqual(monitor.check(carrying=True),'carrying_tilt_limit')
        self.assertEqual(SafetyMonitor(robot,config,lambda:True).check(),'USER_STOP')

    def test_truth_facade_rejects_reads_during_execution(self):
        from evaluation.ground_truth import SimulationGroundTruth
        with self.assertRaises(PermissionError): SimulationGroundTruth(SimpleNamespace(execution_finished=False))

    def test_partial_face_before_grasp_keeps_complete_visual_shape(self):
        world=WorldModel(AgentConfig().perception)
        full=obj(position=(.3,0,.04),size=(.07,.06,.08),category='box')
        world.update([full],1)
        patch=obj(position=(.275,0,.04),size=(.02,.06,.08),category='box',stamp=2)
        world.update([patch],2)
        measured=next(iter(world.objects.values()))
        np.testing.assert_allclose(measured.position,[.3,0,.04])
        np.testing.assert_allclose(measured.size,[.07,.06,.08])

    def test_surface_attachment_rejects_shifted_geometry(self):
        from skills.skill_base import SkillContext
        robot=Mock(); robot.get_contact_state.return_value=ContactState(2,2)
        robot.get_ee_pose.return_value=Pose(np.zeros(3),np.eye(3))
        points=np.array([[x,y,.04] for x in np.linspace(-.03,.03,8) for y in np.linspace(-.025,.025,8)])
        source=obj(position=(0,0,0),size=(.07,.06,.08),category='box'); source.points=points
        context=SimpleNamespace(source=source,robot=robot,payload={'size':source.size,'relative_rotation':np.eye(3)},log=Mock())
        self.assertTrue(SkillContext.attachment_visible(context,np.zeros(3)))
        self.assertFalse(SkillContext.attachment_visible(context,np.array([.08,0,0])))

if __name__=='__main__': unittest.main()
