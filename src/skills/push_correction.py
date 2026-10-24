"""Bounded physical contact correction, observed after every short segment."""
import numpy as np
from .skill_base import Skill
from .localization import travel_to_observation
from robot.robot_interface import Pose
from manipulation.grasp_planner import down_rotation
from spatial.geometry import footprint,ray_exit_distance
from recovery.failure_classifier import SkillFailure


class PushCorrectionSkill(Skill):
    def execute(self,c,delta_xy=None):
        if c.held:raise SkillFailure('PLACEMENT_FAILED',{'reason':'push requires released object'})
        delta=np.asarray(delta_xy,float);length=float(np.linalg.norm(delta));cfg=c.config.placement_correction
        if length>cfg.max_push_distance_m or length<1e-6:raise SkillFailure('PLACEMENT_FAILED',{'reason':'invalid push distance'})
        direction=np.array([delta[0]/length,delta[1]/length,0.]);yaw=-np.arctan2(direction[1],direction[0]);rotation=down_rotation(yaw)
        half=ray_exit_distance(footprint(c.source),c.source.position[:2],-direction[:2])
        if half is None:raise SkillFailure('PLACEMENT_FAILED',{'reason':'no observed push contact boundary'})
        # Tool face offset is robot calibration; object extent is measured RGB-D.
        tool_radius=c.config.planner.push_tool_radius_m
        contact=c.source.position-direction*(half+tool_radius)
        contact[2]=c.source.low[2]+min(.035,c.source.size[2]*.45)+c.config.planner.palm_to_contact_m
        pre=contact-direction*.035;hover=pre+np.array([0,0,.10])
        c.log.event('stage',{'stage':'PUSH_CORRECTION','object_id':c.source_id,'contact_side':(-direction).tolist()})
        c.robot.resume()
        if not c.robot.close_gripper(0,lambda:c.safety.check()):
            raise SkillFailure('PLACEMENT_FAILED',{'reason':'push tool closure failed'})
        c.robot.configure_collision_world(c.world,c.source_id,None,'observed_table')
        for point in (contact,contact+direction*min(length,cfg.push_step_m)):
            c.budget()
            if not any(c.robot.check_collision(q) is None for q in c.robot.solve_ik(Pose(point,rotation),c.robot.get_joint_positions())):
                raise SkillFailure('PATH_BLOCKED',{'reason':'push tool clearance or reachability','position':point.tolist()})
        for retreat in (.035,.020,.010,.005):
            pre=contact-direction*retreat;hover=pre+np.array([0,0,.10])
            if all(any(c.robot.check_collision(q) is None for q in c.robot.solve_ik(Pose(point,rotation),c.robot.get_joint_positions())) for point in (pre,hover)):
                break
        else:raise SkillFailure('PATH_BLOCKED',{'reason':'no clear push entry'})
        travel_to_observation(c,Pose(hover,rotation));c.move(pre,rotation,allow_contact=True)
        before=c.source.position.copy();old_limit=c.safety.force_limit_override;old_speed=c.cartesian_speed_limit
        c.safety.force_limit_override=cfg.push_force_limit_n;c.cartesian_speed_limit=cfg.push_velocity_m_s
        try:
            touched=False
            # The edge from an oblique RGB-D footprint can lie a centimetre
            # outside the actual side face. Search a bounded extra 22 mm;
            # every 2 mm segment still passes collision and 6 N checks.
            for index in range(int(np.ceil((retreat+.022)/.002))):
                # Keep the calibrated contact height fixed. Adding each step
                # to the measured (slightly sagging) hand pose accumulates
                # servo error and can lower the palm into a neighbour.
                commanded=pre+direction*(.002*(index+1))
                c.budget();c.move(commanded,rotation,allow_contact=True)
                contact_state=c.robot.get_contact_state()
                if max(contact_state.left_force,contact_state.right_force)>=c.config.safety.min_grasp_force_n:
                    touched=True;break
            if not touched:raise SkillFailure('NO_CONTACT',{'reason':'push contact search exhausted'})
            step=min(length,cfg.push_step_m)
            commanded=commanded+direction*step
            c.move(commanded,rotation,allow_contact=True)
            commanded=commanded-direction*retreat
            c.move(commanded,rotation,allow_contact=True)
            c.move(commanded+np.array([0,0,.10]),rotation,allow_contact=True)
            c.robot.hold(.2,lambda:c.safety.check());c.refresh();c.refresh()
            if not c.source.visible:raise SkillFailure('TARGET_LOST',{'reason':'object lost after push'})
            actual=c.source.position-before
            c.log.event('push_observation',{'commanded_m':step,'observed_delta_m':actual.tolist(),'object_id':c.source_id,
                                          'contact_side':(-direction).tolist(),'source':'fresh_rgbd'})
            if actual[:2]@direction[:2]<cfg.min_observed_motion_m:raise SkillFailure('PLACEMENT_FAILED',{'reason':'push made no measured progress'})
            if abs(actual[2])>.015:raise SkillFailure('PLACEMENT_FAILED',{'reason':'possible tipping after push'})
            return True
        except SkillFailure as exc:
            if exc.failure.value not in {'USER_STOP','COLLISION_RISK'}:
                # A failed contact search can leave the closed tool beside
                # the source. Clear it before the caller tries to regrasp.
                try:
                    pose=c.robot.get_ee_pose()
                    c.move(pose.position-direction*.025,rotation,allow_contact=True)
                    pose=c.robot.get_ee_pose()
                    c.move(pose.position+np.array([0,0,.10]),rotation,allow_contact=True)
                    c.log.event('push_safe_retreat',{'reason':exc.failure.value})
                except SkillFailure as retreat_failure:
                    c.log.event('push_safe_retreat_failed',{'reason':retreat_failure.failure.value})
            raise
        finally:
            c.safety.force_limit_override=old_limit;c.cartesian_speed_limit=old_speed
