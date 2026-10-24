"""Coarse targets authorize observation motion only; closing requires a fresh lock."""
import numpy as np
import time
from robot.robot_interface import Pose
from manipulation.viewpoint_planner import ViewpointPlanner
from recovery.failure_classifier import SkillFailure


def travel_to_observation(c,pose):
    current=c.robot.get_ee_pose();highest=max((o.high[2] for o in c.world.objects.values()),default=.15)
    height=max(current.position[2],pose.position[2],highest+c.config.planner.palm_to_contact_m+.08)
    c.move(np.array([*current.position[:2],height]),current.rotation)
    delta=pose.position[:2]-current.position[:2];t=np.clip(-current.position[:2]@delta/max(delta@delta,1e-9),0,1)
    if np.linalg.norm(current.position[:2]+t*delta)<c.config.planner.base_clearance_radius_m:
        a0=np.arctan2(current.position[1],current.position[0]);a1=np.arctan2(pose.position[1],pose.position[0]);radius=max(np.linalg.norm(current.position[:2]),np.linalg.norm(pose.position[:2]),c.config.planner.transfer_radius_m)
        for angle in np.linspace(a0,a1,max(2,int(abs(a1-a0)/.20)+1)):
            c.move(np.array([radius*np.cos(angle),radius*np.sin(angle),height]),current.rotation)
    c.move(np.array([*pose.position[:2],height]),pose.rotation)
    c.move(pose.position,pose.rotation)


def localize_target(c,identity,move=True):
    cfg=c.config.localization
    if not cfg.enabled:return c.world.objects[identity]
    obj=c.world.objects[identity];camera=c.perception.camera;c.perception.phase='local'
    c.log.event('stage',{'stage':'APPROACHING_FOR_OBSERVATION' if move else 'TARGET_REOBSERVATION','object_id':identity})
    poses=list(ViewpointPlanner(c.config).candidates(obj,camera)) if camera.config.mode=='wrist' and move else [None]
    last=None
    for attempt,pose in enumerate(poses[:cfg.max_local_observations]):
        c.budget()
        if pose is not None:
            try:travel_to_observation(c,pose)
            except SkillFailure as exc:
                if exc.failure.value in {'USER_STOP','COLLISION_RISK'}:raise
                last=exc;c.log.event('viewpoint_rejected',{'reason':exc.failure.value});continue
        elif move and camera.config.mode=='gantry':
            camera.move_observation(obj.position+np.array([0,0,cfg.observation_distance_m]),obj.position)
        c.log.event('stage',{'stage':'LOCAL_REOBSERVATION','object_id':identity,'attempt':attempt+1})
        c.perception.observe(reground=True);c.refresh();obj=c.world.objects.get(identity)
        if obj and c.perception.localizer.precise(obj,time.monotonic()):
            c.log.event('stage',{'stage':'PRECISE_LOCKED','object_id':identity});return obj
        last=SkillFailure('TARGET_LOST' if obj is None or not obj.visible else 'LOW_CONFIDENCE',{'reason':'fresh close-range precision gate','object_id':identity})
    raise last or SkillFailure('LOW_CONFIDENCE',{'reason':'no safe local observation pose'})
