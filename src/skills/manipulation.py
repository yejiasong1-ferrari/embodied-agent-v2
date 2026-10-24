import numpy as np
from dataclasses import replace
from .skill_base import Skill
from recovery.failure_classifier import SkillFailure,FailureClassifier
from validation.grasp_validator import GraspValidator
from validation.task_validator import TaskValidator


def placement_planning_goal(c):
    """Apply visual release feedback to the next physical placement plan."""
    correction=float(getattr(c,'placement_gap_compensation_m',0.))
    if abs(correction)<1e-9:return c.goal
    goal=c.goal.model_copy(deep=True)
    for constraint in goal.spatial_constraints:
        if constraint.type=='relative_position' and constraint.distance_m is not None:
            constraint.distance_m=max(0.,constraint.distance_m+correction)
    return goal


def short_arc_angles(start,end,max_step=.20):
    delta=np.arctan2(np.sin(end-start),np.cos(end-start))
    return np.linspace(start,start+delta,max(2,int(abs(delta)/max_step)+1))


def transport_obstacle_height(world,waypoints,source_id,payload_width):
    """Only observed obstacles near the carried object's swept XY route set its altitude."""
    segments=list(zip(waypoints,waypoints[1:]))
    highest=.1
    carried={source_id}|set(getattr(getattr(world,'memory',None),'held_fragments',()))
    for obstacle in world.objects.values():
        if obstacle.id in carried:continue
        center=np.asarray(obstacle.position[:2]);radius=(
            .5*np.linalg.norm(obstacle.size[:2])+.5*payload_width+.06)
        for first,last in segments:
            delta=np.asarray(last)-np.asarray(first)
            alpha=np.clip(np.dot(center-first,delta)/max(np.dot(delta,delta),1e-9),0.,1.)
            if np.linalg.norm(center-(first+alpha*delta))<=radius:
                highest=max(highest,float(obstacle.high[2]));break
    return highest


def transport_route(c,object_position,placement_rotation,variant):
    """Build a complete route before executing any segment of it."""
    from robot.robot_interface import Pose
    side=c.candidate.grasp_type=='side_grasp'
    preserve='orientation_preservation' in c.goal.hard_constraints
    current=c.robot.get_ee_pose()
    initial=c.candidate.orientation
    a0=np.arctan2(current.position[1],current.position[0])
    hand_target=object_position-placement_rotation@np.asarray(c.payload['offset'])
    a1=np.arctan2(hand_target[1],hand_target[0])
    delta=hand_target[:2]-current.position[:2]
    alpha=np.clip(-np.dot(current.position[:2],delta)/max(np.dot(delta,delta),1e-9),0,1)
    near_base=np.linalg.norm(current.position[:2]+alpha*delta)<c.config.planner.base_clearance_radius_m
    use_arc=near_base or variant>0
    radius=(c.config.planner.side_transfer_radius_m if side else
            c.config.planner.transfer_radius_m+(.0,.05,-.05)[variant])
    angles=short_arc_angles(a0,a1) if use_arc else []
    if variant==2 and use_arc:
        shortest=np.arctan2(np.sin(a1-a0),np.cos(a1-a0))
        long_delta=shortest-np.copysign(2*np.pi,shortest)
        angles=np.linspace(a0,a0+long_delta,max(2,int(abs(long_delta)/.20)+1))
    last_rotation=initial
    if use_arc and not preserve and variant!=2:
        yaw=angles[-1]-a0
        rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        last_rotation=rz@initial
    if side or not preserve:last_rotation=placement_rotation
    hand_target=object_position-last_rotation@np.asarray(c.payload['offset'])
    route_xy=[current.position[:2]]
    if use_arc:route_xy.extend(np.array([radius*np.cos(angle),radius*np.sin(angle)]) for angle in angles)
    route_xy.append(hand_target[:2])
    highest=transport_obstacle_height(c.world,route_xy,c.source_id,max(c.payload.get('collision_size',c.payload['size'])[:2]))
    clearance=c.config.planner.side_transport_clearance_m if side else c.config.planner.transport_clearance_m
    z=max(current.position[2],hand_target[2]+clearance,
          highest+c.config.planner.palm_to_contact_m+.06)
    if variant==1:z+=.025
    waypoints=[Pose(np.array([*current.position[:2],z]),initial)]
    if use_arc:
        waypoints.append(Pose(np.array([radius*np.cos(a0),radius*np.sin(a0),z]),initial))
        for angle in angles[1:]:
            yaw=angle-a0
            rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
            rotation=initial if preserve or variant==2 else rz@initial
            waypoints.append(Pose(np.array([radius*np.cos(angle),radius*np.sin(angle),z]),rotation))
    if side:waypoints.append(Pose(waypoints[-1].position.copy(),placement_rotation))
    waypoints.append(Pose(np.array([*hand_target[:2],z]),last_rotation))
    view_height=hand_target[2]+.04 if side else max(hand_target[2]+.06,c.target.high[2]+.34)
    waypoints.append(Pose(np.array([*hand_target[:2],view_height]),last_rotation))
    return waypoints,last_rotation,{'variant':variant,'placement':object_position.tolist(),
        'radius_m':radius if use_arc else None,'height_m':float(z),
        'observed_route_obstacle_top_m':highest,'waypoints':len(waypoints)}


def near_visual_gap_limit(evidence,band_m=.004):
    rows=evidence.get('constraints',[])
    if len(rows)!=1:return False
    row=rows[0]
    requested,actual,tolerance=(row.get(key) for key in ('requested_gap_m','actual_gap_m','tolerance_m'))
    return bool(row.get('type')=='relative_position' and requested is not None and actual is not None
                and tolerance is not None and abs(abs(actual-requested)-tolerance)<=band_m)


def fuse_visual_gap_samples(samples,config):
    """Fuse independent post-release views only when their gaps agree."""
    if len(samples)<2:return samples[0]
    gaps=[float(e['constraints'][0]['actual_gap_m']) for _,e in samples]
    if max(gaps)-min(gaps)>.006:return samples[0]
    import copy
    evidence=copy.deepcopy(samples[-1][1]);row=evidence['constraints'][0]
    actual=float(np.median(gaps));requested=float(row['requested_gap_m']);tolerance=float(row['tolerance_m'])
    alignment=float(np.median([e['constraints'][0]['alignment_error_m'] for _,e in samples]))
    direction=all(e['constraints'][0].get('direction_correct',False) for _,e in samples)
    row.update(actual_gap_m=actual,distance_error_m=abs(actual-requested),alignment_error_m=alignment,
               direction_correct=direction,satisfied=bool(direction and abs(actual-requested)<=tolerance
                   and alignment<=config.spatial.direction_alignment_tolerance_m))
    evidence['relation_satisfied']=row['satisfied']
    evidence['fresh_visual_evidence']=all(e.get('fresh_visual_evidence',False) for _,e in samples)
    evidence['stable_displacement_m']=max(e.get('stable_displacement_m',float('inf')) for _,e in samples)
    evidence['object_released']=all(e.get('object_released',False) for _,e in samples)
    evidence['accepted']=bool(row['satisfied'] and evidence['fresh_visual_evidence'] and evidence['object_released']
                              and evidence['stable_displacement_m']<=config.safety.stable_displacement_m
                              and all(e.get('upright_verified',True) for _,e in samples))
    evidence['visual_gap_samples_m']=gaps
    return evidence['accepted'],evidence


class ApproachSkill(Skill):
    preconditions=("source","target")
    success_conditions=("fresh target pose","validated approach")
    failure_conditions=("TARGET_LOST","IK_FAILED","PATH_BLOCKED","COLLISION_RISK")
    def execute(self,c):
        c.perception.camera.set_manipulation_active(True)
        c.robot.resume(); c.robot.open_gripper()
        from .localization import localize_target
        from manipulation.adaptive_grasp import AdaptiveGraspPlanner
        localize_target(c,c.source_id)
        if not hasattr(c,'adaptive_grasp'):c.adaptive_grasp=AdaptiveGraspPlanner(c.config)
        if len(c.recovery.failed_candidates)>=c.config.recovery.max_grasp_attempts:
            raise SkillFailure('GRASP_FAILED',{'reason':'distinct grasp candidate budget exhausted'})
        c.candidate=c.adaptive_grasp.choose(c)
        distance=c.config.planner.side_approach_clearance_m if c.candidate.grasp_type=='side_grasp' else c.config.planner.approach_clearance_m
        hover=c.candidate.position-c.candidate.approach_vector*distance
        if c.candidate.grasp_type=='side_grasp':
            from robot.robot_interface import Pose
            c.robot.configure_collision_world(c.world,None,None,None)
            path=getattr(c,'pregrasp_path',None)
            if path is None:path=c.planner.plan_to_pose(Pose(hover,c.candidate.orientation))
            if path is None:raise SkillFailure(c.planner.last_failure or 'PATH_BLOCKED',{'reason':'side pregrasp route'})
            ok,reason=c.executor.execute(path)
            if not ok:raise SkillFailure(FailureClassifier.classify(reason),{'reason':reason})
            c.refresh()
            return self.refine(c)
        # Raise before lateral travel using observed obstacle tops.
        current=c.robot.get_ee_pose()
        height=max(hover[2],current.position[2],max((o.high[2] for o in c.world.objects.values()),default=.1)+c.config.planner.palm_to_contact_m+.06)
        delta=hover[:2]-current.position[:2]
        alpha=np.clip(-np.dot(current.position[:2],delta)/max(np.dot(delta,delta),1e-9),0,1)
        around_base=np.linalg.norm(current.position[:2]+alpha*delta)<c.config.planner.base_clearance_radius_m
        try:
            c.move(np.array([*current.position[:2],height]),current.rotation if around_base else c.candidate.orientation)
            if around_base:
                a0=np.arctan2(current.position[1],current.position[0]); a1=np.arctan2(hover[1],hover[0])
                arc=short_arc_angles(a0,a1)
                radius=max(np.linalg.norm(current.position[:2]),np.linalg.norm(hover[:2]),c.config.planner.transfer_radius_m)
                start_yaw=np.arctan2(current.rotation[1,0],current.rotation[0,0])
                end_yaw=np.arctan2(c.candidate.orientation[1,0],c.candidate.orientation[0,0])
                yaw_delta=np.arctan2(np.sin(end_yaw-start_yaw),np.cos(end_yaw-start_yaw))
                if abs(abs(yaw_delta)-np.pi)<.01: yaw_delta=np.copysign(np.pi,arc[-1]-a0)
                c.move(np.array([radius*np.cos(a0),radius*np.sin(a0),height]),current.rotation)
                from manipulation.grasp_planner import down_rotation
                for t,angle in zip(np.linspace(0,1,len(arc))[1:],arc[1:]):
                    c.move(np.array([radius*np.cos(angle),radius*np.sin(angle),height]),down_rotation(-(start_yaw+t*yaw_delta)))
            c.move(np.array([*hover[:2],height]),c.candidate.orientation)
            c.move(hover,c.candidate.orientation)
        except SkillFailure as exc:
            if exc.failure.value not in {'IK_FAILED','PATH_BLOCKED'}:raise
            # A straight lift beside the base can leave the IK workspace.
            # Ask the collision planner for another route to the observed
            # pregrasp pose before spending a grasp recovery attempt.
            from robot.robot_interface import Pose
            c.robot.configure_collision_world(c.world,None,None,None)
            path=c.planner.plan_to_pose(Pose(hover,c.candidate.orientation))
            if path is None:raise exc
            c.log.event('approach_route_fallback',{'candidate':c.candidate.id,'reason':exc.failure.value})
            ok,reason=c.executor.execute(path)
            if not ok:raise SkillFailure(FailureClassifier.classify(reason),{'reason':reason})
            c.refresh()
        return self.refine(c)

    def refine(self,c):
        c.log.event('stage',{'stage':'PREGRASP','candidate':c.candidate.id})
        c.perception.phase='local';c.perception.observe(reground=True);c.refresh()
        for attempt in range(c.config.localization.max_corrections):
            c.budget()
            if not c.source.visible:raise SkillFailure('TARGET_LOST',{'reason':'target disappeared during local approach'})
            refreshed=c.grasp_planner.candidates(c.source,c.world)
            candidate=next((x for x in refreshed if x.id==c.candidate.id),None)
            if candidate is None:raise SkillFailure('GRASP_FAILED',{'reason':'selected geometry region no longer visible'})
            c.candidate=candidate
            delta=candidate.position-c.robot.get_ee_pose().position
            distance=float(np.linalg.norm(delta))
            c.log.event('visual_correction',{'attempt':attempt+1,'error_xyz_m':delta.tolist(),'distance_m':distance,'source_id':c.source_id})
            if distance<=c.config.localization.correction_tolerance_m:break
            c.move(candidate.position,candidate.orientation,allow_contact=True,support_id='pregrasp')
        if c.config.localization.enabled:
            import time
            if not c.perception.localizer.precise(c.source,time.monotonic()):
                raise SkillFailure('TARGET_LOST' if not c.source.visible else 'LOW_CONFIDENCE',{'reason':'grasp requires fresh PRECISE_LOCKED'})
        c.prelift_position=c.source.position.copy()
        return True


class GraspSkill(Skill):
    preconditions=("source","candidate")
    success_conditions=("valid closure","bilateral contact")
    failure_conditions=("NO_CONTACT","GRASP_FAILED","COLLISION_RISK")
    def execute(self,c):
        if c.config.localization.enabled:
            import time
            if not c.perception.localizer.precise(c.source,time.monotonic()):raise SkillFailure('LOW_CONFIDENCE',{'reason':'precise lock expired before closure'})
        c.log.event('stage',{'stage':'GRASPING','candidate':c.candidate.id})
        c.grasp_attempts+=1
        c.safety.allow_finger_contact=True
        force=(c.config.grasp.side_grasp_force_target_n if getattr(c.candidate,'grasp_type',None)=='side_grasp' else None)
        if not c.robot.close_gripper(c.candidate.grasp_width,lambda:c.safety.check(),target_force_n=force):
            reason=c.safety.last_failure or getattr(c.robot,"last_error",None)
            raise SkillFailure(FailureClassifier.classify(reason),{"reason":reason})
        return True


class VerifyGraspSkill(Skill):
    preconditions=("candidate",)
    success_conditions=("bilateral contact and nonzero jaw gap",)
    failure_conditions=("NO_CONTACT",)
    def execute(self,c):
        c.refresh()
        contact=c.robot.get_ee_pose().position+c.robot.get_ee_pose().rotation[:,2]*c.config.planner.palm_to_contact_m
        distance=float(np.linalg.norm(c.source.position[:2]-contact[:2]))
        if distance>.02:
            raise SkillFailure("NO_CONTACT",{"reason":"visual target outside gripper after closure","distance_m":distance})
        validator=GraspValidator(c.config)
        ok,evidence=validator.contact(c.robot)
        c.log.event("grasp_validation",evidence)
        if not ok: raise SkillFailure("NO_CONTACT",evidence)
        for attempt in range(2):
            samples=[]
            for _ in range(3):
                c.budget()
                if not c.robot.hold(.08,lambda:c.safety.check()):
                    reason=c.safety.last_failure or getattr(c.robot,'last_error',None)
                    raise SkillFailure(FailureClassifier.classify(reason),{'reason':reason,'phase':'grasp_settle'})
                _,sample=validator.contact(c.robot)
                samples.append(sample)
            stable,evidence=validator.stable_contact(samples)
            c.log.event('grasp_stability',{'candidate':c.candidate.id,'attempt':attempt+1,**evidence})
            if stable:break
            if attempt==0:
                # The object is still supported. Let the fingers settle and
                # try the same measured aperture once before changing pose.
                force=(c.config.grasp.side_grasp_force_target_n if c.candidate.grasp_type=='side_grasp' else None)
                if not c.robot.close_gripper(c.candidate.grasp_width,lambda:c.safety.check(),target_force_n=force):
                    reason=c.safety.last_failure or getattr(c.robot,'last_error',None)
                    raise SkillFailure(FailureClassifier.classify(reason),{'reason':reason,'phase':'local_regrip'})
        if not stable:raise SkillFailure('NO_CONTACT',{'reason':'contact did not remain bilateral before lift',**evidence})
        confirm=getattr(c.robot,'confirm_grasp',None)
        if confirm is not None:confirm()
        c.grasp_verified=True
        c.any_grasp_verified=True
        if c.grasp_attempts==1:c.first_grasp_verified=True
        return True


class LiftSkill(Skill):
    preconditions=("grasp_verified","source")
    success_conditions=("object follows gripper in fresh RGB-D",)
    failure_conditions=("OBJECT_SLIPPED","TARGET_LOST")
    def execute(self,c):
        before=c.source.position.copy(); ee=c.robot.get_ee_pose()
        c.logged_held_fragments.clear()
        c.held=True; c.world.memory.held_id=c.source_id
        c.world.memory.grasped_heights[c.source_id]=float(c.source.size[2])
        c.safety.carry_reference_rotation=ee.rotation.copy()
        geometry=c.source.manipulation_geometry
        c.payload={"id":c.source_id,"size":np.asarray(geometry.get('collision_size',c.source.size)),"offset":ee.rotation.T@(before-ee.position),
                   "collision_offset":ee.rotation.T@(before+np.asarray(geometry.get('collision_center_offset',[0,0,0]))-ee.position),
                   "relative_rotation":ee.rotation.T.copy()}
        offsets=geometry.get('footprint_offsets')
        if offsets is not None:
            c.payload['footprint_local']=np.column_stack([offsets,np.zeros(len(offsets))])@ee.rotation
        surface=getattr(c,'grasp_surface_offsets',None)
        if c.candidate.grasp_type=='side_grasp' and surface is not None and len(surface)>=30:c.payload['surface_local']=surface@ee.rotation
        c.log.event("attachment_hypothesis",c.payload)
        from spatial.geometry import supported_by
        support='observed_table' if c.source.low[2]<.015 else next((o.id for o in c.world.objects.values()
            if o.id!=c.source_id and supported_by(c.source,o,c.source.position)),None)
        try:c.move(ee.position+np.array([0,0,.007]),c.candidate.orientation,allow_contact=True,support_id=support)
        except SkillFailure as exc:
            exc.evidence.setdefault('phase','probe_lift')
            raise
        if not c.source.visible:c.refresh(scan=True)
        probe_ok,probe_evidence=GraspValidator(c.config).probe_lifted(
            before,c.source,c.robot.get_ee_pose().position-ee.position)
        c.log.event('probe_lift_validation',probe_evidence)
        if not probe_ok:raise SkillFailure('OBJECT_SLIPPED',{'phase':'probe_lift',**probe_evidence})
        c.move(ee.position+np.array([0,0,.045]),c.candidate.orientation,allow_contact=True,support_id=support)
        # Scan multiple viewpoints if the hand occludes the lifted object.
        if not c.source.visible: c.refresh(scan=True)
        ok,evidence=GraspValidator(c.config).lifted(before,c.source,c.robot.get_ee_pose().position-ee.position)
        c.log.event("lift_validation",evidence)
        if not ok: raise SkillFailure("OBJECT_SLIPPED",evidence)
        c.lift_verified=True
        c.any_lift_verified=True
        if c.grasp_attempts==1:c.first_lift_success=True
        return True


class MoveObjectSkill(Skill):
    preconditions=("held","lift_verified","target")
    success_conditions=("above visually observed destination",)
    failure_conditions=("PATH_BLOCKED","OBJECT_SLIPPED","TARGET_LOST")
    def execute(self,c):
        from spatial.spatial_planner import SpatialPlanner
        target=c.target;relation=c.goal.relation
        c.log.event('stage',{'stage':'PLAN_PLACEMENT_REGION'})
        c.robot.configure_collision_world(c.world,c.source_id,c.payload,c.target_id if relation=='on' else 'observed_table')
        side=c.candidate.grasp_type=='side_grasp'
        preserve='orientation_preservation' in c.goal.hard_constraints
        orientations={}
        def reachable(position):
            from robot.robot_interface import Pose
            initial=c.candidate.orientation
            rotations=[initial]
            if side and not preserve:
                forward=initial@c.perception.camera.ee_to_camera[:3,2]
                heading=np.arctan2(forward[1],forward[0])
                to_reference=target.position-position
                toward=np.arctan2(to_reference[1],to_reference[0])
                rotations=[]
                for offset in (np.pi/3,-np.pi/3,np.pi/2,-np.pi/2,0.,np.pi):
                    yaw=toward+offset-heading
                    rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
                    rotations.append(rz@initial)
                rotations.append(initial)
            elif not preserve:
                for yaw in (np.pi,-np.pi/2,np.pi/2):
                    rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
                    rotations.append(rz@initial)
            valid=[]
            for rotation in rotations:
                pose=Pose(position-rotation@np.asarray(c.payload['offset'])+np.array([0,0,.025]),rotation)
                if any(c.robot.check_collision(q) is None for q in c.robot.solve_ik(pose,c.robot.get_joint_positions())):
                    valid.append(rotation)
            if valid:orientations[tuple(position)]=valid
            return bool(valid)
        rejected=[];selected=None;spatial_rejection=None
        exclusions=list(c.blocked_placements)
        for _ in range(3):
            try:plan=SpatialPlanner(c.config).plan(
                placement_planning_goal(c),c.source,c.target,c.world,reachable,
                excluded_positions=exclusions)
            except ValueError as exc:
                if 'AMBIGUOUS_REFERENCE_FRAME' in str(exc):
                    raise SkillFailure('AMBIGUOUS_REFERENCE_FRAME',{'reason':str(exc)}) from exc
                spatial_rejection=str(exc)
                break
            position=plan.position
            viable=[]
            for placement_rotation in orientations[tuple(position)]:
                for variant in range(3):
                    waypoints,final_rotation,description=transport_route(c,position,placement_rotation,variant)
                    from robot.robot_interface import Pose
                    hand_target=position-final_rotation@np.asarray(c.payload['offset'])
                    landing=Pose(hand_target+np.array([0,0,.025]),final_rotation)
                    corridor=.10 if side else None
                    valid,evidence=c.planner.preview_route([*waypoints,landing],carrying=True,
                        cartesian_corridor_m=corridor,step_m=.04,budget=c.budget)
                    if valid:
                        description.update(evidence)
                        description['final_descent_preflight']=True
                        viable.append((plan,waypoints,final_rotation,description))
                        continue
                    rejected.append({'placement':position.tolist(),**description,'failure':evidence})
                    c.log.event('transport_route_rejected',rejected[-1])
            if viable:
                selected=min(viable,key=lambda choice:(
                    choice[3]['near_joint_limit_segments'],
                    -choice[3]['minimum_joint_margin_rad'],
                    choice[3]['waypoints']))
                break
            exclusions.append(position.copy())
        if selected is None:
            raise SkillFailure('NO_VALID_PLACEMENT',{
                'reason':'no collision-free complete transport route to a valid placement',
                'spatial_rejection':spatial_rejection,
                'route_attempts':rejected[-9:]})
        c.placement_plan,waypoints,final_rotation,description=selected
        c.placement_position=c.placement_plan.position
        c.placement_support_id=c.placement_plan.support_id
        c.log.event('placement_plan',{'position':c.placement_position.tolist(),'relation':relation,
            'tolerance_m':c.placement_plan.tolerance_m,
            'constraints':[x.model_dump() for x in c.goal.spatial_constraints],
            'score':c.placement_plan.score})
        c.log.event('transport_route',description)
        for waypoint in waypoints:
            c.move(waypoint.position,waypoint.rotation,True)
        c.candidate=replace(c.candidate,orientation=final_rotation)
        return True


class PlaceSkill(Skill):
    preconditions=("held","placement_position")
    success_conditions=("controlled descent","released object")
    failure_conditions=("PLACEMENT_FAILED","OBJECT_SLIPPED")
    def check_preconditions(self,c):
        if not c.held or c.placement_position is None: raise SkillFailure("PLACEMENT_FAILED")
    def execute(self,c):
        from .localization import localize_target
        from spatial.spatial_planner import SpatialPlanner
        if c.target_id not in c.world.objects and c.target_id not in c.world.regions:
            raise SkillFailure('TARGET_LOST',{'reason':'placement reference missing from current world'})
        if c.target_id in c.world.objects:
            try:localize_target(c,c.target_id,move=False)
            except SkillFailure as initial_failure:
                if initial_failure.failure.value not in {'TARGET_LOST','LOW_CONFIDENCE'}:raise
                # The carried object can hide the support directly beneath
                # the wrist. Shift laterally while retaining the measured
                # payload and ask the same RGB-D precision gate again.
                start=c.robot.get_ee_pose()
                found=False
                for shift in ([0.,-.09,0.],[.09,0.,0.],[-.09,0.,0.],[0.,.09,0.]):
                    try:
                        for fraction in (.25,.5,.75,1.):
                            # The carried object may hide the reference only
                            # at the current angle. Stop the lateral motion as
                            # soon as two fresh views give a precise lock;
                            # travelling the full shift can lose the grip.
                            offset=(fraction*np.asarray(shift)).tolist()
                            c.move(start.position+np.asarray(offset),start.rotation,True)
                            try:localize_target(c,c.target_id,move=False)
                            except SkillFailure as exc:
                                if exc.failure.value in {'USER_STOP','COLLISION_RISK'}:raise
                                continue
                            c.log.event('target_view_reposition',{'shift_m':offset,'reference_id':c.target_id})
                            found=True;break
                        if found:break
                    except SkillFailure as exc:
                        if exc.failure.value in {'USER_STOP','COLLISION_RISK'}:raise
                if not found:
                    if 'orientation_preservation' in c.goal.hard_constraints:raise initial_failure
                    # Reorient the wrist around the measured attachment centre
                    # to reveal a reference hidden behind a finger. Every short
                    # step still requires payload collision and RGB-D checks.
                    pose=c.robot.get_ee_pose();initial=pose.rotation.copy()
                    pivot=pose.position+initial@np.asarray(c.payload['offset'])
                    for yaw in np.linspace(np.pi/12,np.pi,12):
                        rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
                        rotation=rz@initial
                        c.move(pivot-rotation@np.asarray(c.payload['offset']),rotation,True)
                        c.candidate=replace(c.candidate,orientation=rotation)
                        if not c.target.visible:continue
                        try:localize_target(c,c.target_id,move=False);found=True;break
                        except SkillFailure as exc:
                            if exc.failure.value not in {'TARGET_LOST','LOW_CONFIDENCE'}:raise
                    if not found:raise initial_failure
        try:
            spatial=SpatialPlanner(c.config);goal=placement_planning_goal(c)
            reuse=False
            if c.placement_position is not None and c.placement_plan is not None:
                safe,_,_=spatial.feasibility(c.source,c.target,c.world,c.placement_position,
                    c.goal.relation,support_id=c.placement_support_id)
                satisfies,_=spatial.validator.evaluate(goal,c.source,c.target,c.world,
                    c.placement_position,require_visible=False)
                reuse=bool(safe and satisfies)
            if not reuse:
                c.placement_plan=spatial.plan(goal,c.source,c.target,c.world,
                    excluded_positions=c.blocked_placements)
                c.placement_position=c.placement_plan.position
                c.placement_support_id=c.placement_plan.support_id
            else:c.log.event('placement_plan_reused',{'position':c.placement_position.tolist(),
                'basis':'fresh_visual_reference_and_verified_transport_route'})
        except ValueError as exc:raise SkillFailure('AMBIGUOUS_REFERENCE_FRAME' if 'AMBIGUOUS_REFERENCE_FRAME' in str(exc) else 'NO_VALID_PLACEMENT',{'reason':str(exc)}) from exc
        c.log.event('stage',{'stage':'PLACING'})
        hand_target=c.placement_position-c.candidate.orientation@np.asarray(c.payload["offset"])
        c.log.event('placement_plan',{'position':c.placement_position.tolist(),'hand_target':hand_target.tolist(),
            'reference_id':c.target_id,'reference_position':c.target.position.tolist(),'phase':'local_refinement'})
        # Finish lateral correction above the support, then descend vertically.
        current=c.robot.get_ee_pose()
        c.move(np.array([*hand_target[:2],current.position[2]]),c.candidate.orientation,True)
        try:
            c.move(hand_target+np.array([0,0,.025]),c.candidate.orientation,True,c.placement_support_id)
            # Correct from the freshly observed payload, including compliance
            # accumulated during transport. Leave a small, bounded clearance
            # for release rather than pressing the payload into its support.
            for attempt in range(c.config.localization.max_corrections):
                if not c.source.visible:raise SkillFailure('TARGET_LOST',{'reason':'payload invisible during final descent'})
                support_error=c.source.position-c.placement_position
                if (abs(support_error[2])<=.007 and np.linalg.norm(support_error[:2])<=.012):
                    # A side-held object may already rest on the support.
                    # Requiring the nominal 8 mm air gap would drive the
                    # hand through the table while the object cannot descend.
                    c.log.event('support_transfer',{'source':'fresh_rgbd','observed_position':c.source.position.tolist(),
                        'intended_position':c.placement_position.tolist(),'height_error_m':float(support_error[2]),
                        'horizontal_error_m':float(np.linalg.norm(support_error[:2]))})
                    break
                error=c.placement_position+np.array([0,0,c.config.planner.release_clearance_m])-c.source.position
                c.log.event('placement_visual_correction',{'attempt':attempt+1,'height_error_m':float(error[2]),'position_error_m':error.tolist(),
                    'observed_position':c.source.position.tolist(),'release_clearance_m':c.config.planner.release_clearance_m})
                if np.linalg.norm(error)<=c.config.localization.correction_tolerance_m:break
                pose=c.robot.get_ee_pose();step=np.clip(error,-c.config.localization.correction_step_m,c.config.localization.correction_step_m)
                c.move(pose.position+step,pose.rotation,True,c.placement_support_id)
            else:
                if abs(c.placement_position[2]+c.config.planner.release_clearance_m-c.source.position[2])>.006:
                    raise SkillFailure('PLACEMENT_FAILED',{'reason':'descent correction budget exhausted'})
        except SkillFailure as exc:
            if exc.failure.value!="OBJECT_SLIPPED": raise
            # Contact may unload when the support takes the object's weight.
            # The placement relation may intentionally be wrong in a recovery
            # trial. Confirm physical support here; VerifyTask evaluates the
            # relation after release and selects a correction if needed.
            c.refresh(scan=True)
            observed=c.source.position.copy()
            c.robot.hold(.3)
            c.refresh(scan=True)
            stable=bool(c.source.visible and np.linalg.norm(c.source.position-observed)<.004)
            near_target=bool(np.linalg.norm(c.source.position[:2]-c.placement_position[:2])<.022)
            intended_support_z=c.placement_position[2]-c.source.size[2]/2
            supported=bool(abs(c.source.low[2]-intended_support_z)<.012)
            # A dropped object can settle on the intended support several
            # centimetres from the plan. Release the unloaded gripper and let
            # the fresh relation check choose a push or regrasp correction.
            if not (stable and supported):raise
            c.log.event("support_transfer",{"source":"fresh_rgbd","stable":stable,"near_target":near_target,
                "support_bottom_z_m":float(c.source.low[2]),"observed_position":c.source.position.tolist(),
                "intended_position":c.placement_position.tolist(),"reason":"object stopped at support during descent"})
            c.robot.resume(); c.safety.last_failure=None; c.safety._lost_count=0
        if not c.robot.open_gripper(lambda:c.safety.check()):
            reason=c.safety.last_failure or getattr(c.robot,'last_error',None)
            raise SkillFailure(FailureClassifier.classify(reason),{'reason':reason,'phase':'release'})
        c.held=False; c.payload=None; c.world.memory.held_id=None
        c.robot.hold(.3)
        c.refresh()
        if not c.source.visible: c.refresh(scan=True)
        if not c.source.visible: raise SkillFailure("TARGET_LOST",{"reason":"object not visible after release"})
        # The open command can finish while the fingers are still moving.
        # Stay put until the measured jaw is at the open stop, then rise.
        open_width=float(c.config.planner.max_gripper_width_m)
        for _ in range(40):
            if c.robot.get_gripper_state()["width"]>=open_width-5e-4: break
            if not c.robot.hold(.05,lambda:c.safety.check()): raise SkillFailure("PLACEMENT_FAILED")
        else: raise SkillFailure("PLACEMENT_FAILED")
        before=c.source.position.copy()
        pose=c.robot.get_ee_pose()
        # Probe for an object hooked on one finger before a full retreat.
        side=c.candidate.grasp_type=='side_grasp'
        if side:
            outward=pose.position-c.source.position;outward[2]=0
            outward/=max(np.linalg.norm(outward),1e-9)
            probe_delta=.025*outward
            c.move(pose.position+probe_delta,pose.rotation,allow_contact=True,support_id='side_release')
            moved=float(np.dot(c.source.position-before,outward))
        else:
            probe_delta=np.array([0,0,.02])
            c.move(pose.position+probe_delta,pose.rotation,allow_contact=True,support_id=c.placement_support_id or c.target_id)
            moved=float(c.source.position[2]-before[2])
        followed=c.source.visible and moved>.003
        c.log.event("release_probe",{"followed_gripper":bool(followed),"probe_axis":"outward" if side else "up",
            "object_motion_m":moved})
        if followed:
            pose=c.robot.get_ee_pose()
            closing=c.candidate.orientation[:,1]
            offset=c.source.position[:2]-pose.position[:2]
            # Re-centre the fully open jaw around the measured payload, then
            # test whether it is still being carried. A lateral shove can
            # drag the object across the table instead of freeing it.
            correction=np.clip(np.dot(offset,closing[:2]),-.01,.01)
            c.move(pose.position+correction*closing,pose.rotation,allow_contact=True,support_id=c.placement_support_id or c.target_id)
            before=c.source.position.copy(); pose=c.robot.get_ee_pose()
            c.move(pose.position+np.array([0,0,.012]),pose.rotation,allow_contact=True,support_id=c.placement_support_id or c.target_id)
            if c.source.visible and c.source.position[2]-before[2]>.003:
                raise SkillFailure("OBJECT_SLIPPED",{"reason":"object followed gripper after jaw recentering"})
            c.robot.hold(.3); c.refresh(scan=True)
            from validation.placement_validator import PlacementValidator
            supported,evidence=PlacementValidator(c.config).validate(c.source,c.target,c.goal.relation)
            c.log.event("release_disengagement",evidence)
            if not supported: raise SkillFailure("OBJECT_SLIPPED",{"reason":"object remained attached after release","evidence":evidence})
        return True


class RetreatSkill(Skill):
    preconditions=("candidate",)
    success_conditions=("camera view unobstructed",)
    failure_conditions=("PATH_BLOCKED",)
    def execute(self,c):
        pose=c.robot.get_ee_pose()
        if c.candidate.grasp_type=='side_grasp':
            outward=pose.position-c.source.position;outward[2]=0
            outward/=max(np.linalg.norm(outward),1e-9)
            c.move(pose.position+.035*outward,pose.rotation,True,'side_release')
            pose=c.robot.get_ee_pose()
        c.move(pose.position+np.array([0,0,.12]),pose.rotation,True)
        # Leave the object's vertical sightline before final visual checking.
        pose=c.robot.get_ee_pose()
        c.move(np.array([pose.position[0]*.5,pose.position[1]*.5,max(.50,pose.position[2])]),pose.rotation)
        c.perception.camera.set_manipulation_active(False)
        return True


class VerifyTaskSkill(Skill):
    preconditions=("source","target")
    success_conditions=("fresh visual relation","stable object","released gripper")
    failure_conditions=("RELATION_NOT_SATISFIED","TARGET_LOST")
    def measure(self,c):
        candidate=getattr(c,'candidate',None)
        if (not c.held and candidate is not None and candidate.grasp_type=='side_grasp'
                and not getattr(c,'final_footprint_scanned',False)):
            # A wide cap may rotate while side-held. Its final single wrist
            # image can see only a central patch and understate the true edge
            # gap. Reconstruct the released footprint from fresh views before
            # using centimetre-scale relation validation.
            from perception.multiview_scanner import MultiViewObjectScanner
            model,points,views=MultiViewObjectScanner(c.config).scan(c)
            if views<2 or len(points)<c.config.grasp.min_slice_points:
                raise SkillFailure('LOW_CONFIDENCE',{'reason':'released side-grasp footprint needs two fresh views','views':views})
            c.source.points=points
            c.source.manipulation_geometry['verified_footprint_offsets']=model['footprint_offsets']
            c.source.manipulation_geometry['verified_footprint_anchor']=c.source.position[:2].tolist()
            c.source.manipulation_geometry['verified_footprint_world']=(np.asarray(model['footprint_offsets'])+c.source.position[:2]).tolist()
            c.final_footprint_scanned=True
            c.log.event('postrelease_footprint',{'views':views,'points':len(points),
                'vertices':len(model['footprint_offsets']),'source':'fresh_rgbd_multiview'})
        if getattr(c.perception.camera,'config',None) is not None and c.perception.camera.config.mode=='wrist' and not c.held:
            from manipulation.viewpoint_planner import ViewpointPlanner
            from types import SimpleNamespace
            references=[c.world.resolve(r) for r in c.goal.references]
            required={c.source_id,*[o.id for o in references if o.id not in c.world.regions]}
            pivot=SimpleNamespace(position=np.mean([c.source.position,*[o.position for o in references]],axis=0))
            poses=list(ViewpointPlanner(c.config).candidates(pivot,c.perception.camera))[:2]
            delta=c.target.position-c.source.position;delta[2]=0
            if np.linalg.norm(delta)>.01:
                # Keep both objects along the gap between the fingers in the
                # wrist image, rather than behind its two vertical finger bars.
                from robot.robot_interface import Pose
                down=delta/np.linalg.norm(delta);forward=np.array([0.,0.,-1.])
                camera_pose=np.eye(4);camera_pose[:3,:3]=np.column_stack([np.cross(down,forward),down,forward])
                camera_pose[:3,3]=pivot.position+np.array([0,0,c.config.localization.observation_distance_m])
                hand=camera_pose@np.linalg.inv(c.perception.camera.ee_to_camera)
                poses.insert(0,Pose(hand[:3,3],hand[:3,:3]))
            marginal=[]
            for pose in poses:
                c.budget();c.robot.configure_collision_world(c.world,None,None,None)
                path=c.planner.plan_to_pose(pose)
                if path is None:continue
                ok,reason=c.executor.execute(path)
                if not ok:raise SkillFailure(FailureClassifier.classify(reason),{'reason':reason})
                c.perception.phase='local';c.perception.observe(reground=True)
                first=c.refresh()
                if not required.issubset({o.id for o in first}):continue
                before=c.source.position.copy()
                if not c.robot.hold(.3,lambda:c.safety.check()):raise SkillFailure('COLLISION_RISK')
                second=c.refresh()
                if not required.issubset({o.id for o in second}):continue
                ok,evidence=TaskValidator(c.config).validate(c.goal,c.source,c.target,before,c.held,world=c.world)
                if marginal or (not ok and near_visual_gap_limit(evidence)):
                    marginal.append((ok,evidence))
                    if len(marginal)<3:continue
                    ok,evidence=fuse_visual_gap_samples(marginal,c.config)
                    c.log.event('visual_gap_consensus',{'samples_m':evidence.get('visual_gap_samples_m'),
                        'accepted':ok})
                    c.log.event('final_validation',evidence);return ok,evidence
                c.log.event('final_validation',evidence);return ok,evidence
            if marginal:
                ok,evidence=fuse_visual_gap_samples(marginal,c.config)
                c.log.event('visual_gap_consensus',{'samples_m':evidence.get('visual_gap_samples_m'),
                    'accepted':ok})
                c.log.event('final_validation',evidence);return ok,evidence
        c.refresh(scan=True)
        # A fused scan can mark objects visible even if its final camera view
        # cannot see both. Require two actual pixel detections in one view.
        region_target=c.target_id in c.world.regions
        required={c.source_id} if region_target else {c.source_id,c.target_id}
        if getattr(c.world,'objects',None) is not None:
            required.update(c.world.resolve(ref).id for ref in c.goal.references if c.world.resolve(ref).id not in c.world.regions)
        evidence={"reason":"no camera view showed both objects twice"}
        for view in range(c.perception.camera.viewpoint_count):
            if not c.perception.camera.move_viewpoint(view): continue
            first=c.refresh(); first_ids={o.id for o in first}
            if not required.issubset(first_ids) or (region_target and not c.target.visible): continue
            before=c.source.position.copy()
            c.robot.hold(.3)
            second=c.refresh(); second_ids={o.id for o in second}
            if not required.issubset(second_ids) or (region_target and not c.target.visible): continue
            ok,evidence=TaskValidator(c.config).validate(c.goal,c.source,c.target,before,c.held,world=c.world)
            c.log.event("final_validation",evidence)
            return ok,evidence
        c.log.event("final_validation",evidence)
        raise SkillFailure("TARGET_LOST",evidence)

    def execute(self,c):
        from spatial.spatial_planner import SpatialPlanner
        from spatial.correction_planner import PlacementCorrectionPlanner
        from .push_correction import PushCorrectionSkill
        from .skill_registry import SkillRegistry
        budget=c.config.placement_correction.max_push_attempts+c.config.placement_correction.max_regrasps+1
        for _ in range(budget):
            c.budget();c.log.event('stage',{'stage':'VALIDATE_RELATION'})
            ok,evidence=self.measure(c)
            c.log.event('relation_validation',evidence)
            if ok:c.final_validated=True;return True
            try:desired=SpatialPlanner(c.config).plan(c.goal,c.source,c.target,c.world)
            except ValueError as exc:raise SkillFailure('AMBIGUOUS_REFERENCE_FRAME' if 'AMBIGUOUS_REFERENCE_FRAME' in str(exc) else 'NO_VALID_PLACEMENT',{'reason':str(exc)}) from exc
            decision=PlacementCorrectionPlanner(c.config).decide(c.goal,c.source,c.target,c.world,desired,c.push_attempts,c.regrasp_attempts,held=c.held)
            c.log.event('placement_correction',{'strategy':decision.strategy,'reason':decision.reason,'delta_xy':decision.delta_xy.tolist(),
                                              'push_attempts':c.push_attempts,'regrasp_attempts':c.regrasp_attempts})
            c.correction_history.append({'strategy':decision.strategy,'reason':decision.reason})
            if decision.strategy=='ABORT':raise SkillFailure('RELATION_NOT_SATISFIED',evidence)
            if decision.strategy=='REPLACE':
                if c.regrasp_attempts>=c.config.placement_correction.max_regrasps:raise SkillFailure('RELATION_NOT_SATISFIED',evidence)
                c.regrasp_attempts+=1
                c.final_footprint_scanned=False
                c.source.manipulation_geometry.pop('verified_footprint_offsets',None)
                c.source.manipulation_geometry.pop('verified_footprint_world',None)
                c.source.manipulation_geometry.pop('verified_footprint_anchor',None)
                for skill in (MoveObjectSkill(),PlaceSkill(),RetreatSkill()):
                    c.steps+=1;c.budget();skill.execute(c)
                continue
            if decision.strategy=='PUSH':
                c.push_attempts+=1
                try:PushCorrectionSkill().execute(c,decision.delta_xy);continue
                except SkillFailure as exc:
                    c.log.event('push_failure',{'failure':exc.failure.value,'evidence':exc.evidence})
                    if exc.failure.value in {'COLLISION_RISK','USER_STOP'}:raise
                    c.push_attempts=c.config.placement_correction.max_push_attempts
            if c.regrasp_attempts>=c.config.placement_correction.max_regrasps:raise SkillFailure('RELATION_NOT_SATISFIED',evidence)
            if c.regrasp_attempts>=1:
                measured=next((x for x in evidence.get('constraints',[]) if x.get('requested_gap_m') is not None
                    and x.get('actual_gap_m') is not None),None)
                if measured is not None:
                    residual=float(measured['requested_gap_m']-measured['actual_gap_m'])
                    c.placement_gap_compensation_m=float(np.clip(residual,-.025,.025))
                    c.log.event('placement_feedback',{'measured_gap_m':measured['actual_gap_m'],
                        'requested_gap_m':measured['requested_gap_m'],'next_plan_compensation_m':c.placement_gap_compensation_m})
            c.regrasp_attempts+=1;c.log.event('stage',{'stage':'REGRASP','attempt':c.regrasp_attempts})
            c.final_footprint_scanned=False
            c.source.manipulation_geometry.pop('verified_footprint_offsets',None)
            c.source.manipulation_geometry.pop('verified_footprint_world',None)
            c.source.manipulation_geometry.pop('verified_footprint_anchor',None)
            c.grasp_verified=c.lift_verified=False;c.candidate=None;c.recovery.failed_candidates.clear()
            c.perception.camera.set_manipulation_active(False)
            c.refresh(scan=True)
            registry=SkillRegistry()
            for name in ['APPROACH_OBJECT','GRASP_OBJECT','VERIFY_GRASP','LIFT_OBJECT','MOVE_TO_TARGET','PLACE_OBJECT','RETREAT']:
                c.steps+=1;c.budget();c.log.event('correction_step',{'skill':name,'attempt':c.regrasp_attempts})
                if not registry.execute(name,c):raise SkillFailure('PLACEMENT_FAILED',{'correction_skill':name})
            c.current_skill='VERIFY_TASK'
        raise SkillFailure('RELATION_NOT_SATISFIED',{'reason':'placement correction budget exhausted'})
