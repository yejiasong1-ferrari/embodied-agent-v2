import time
import numpy as np
from dataclasses import replace
from llm.schemas import GraspStrategyDecision
from perception.geometry_analyzer import GeometryAnalyzer
from perception.multiview_scanner import MultiViewObjectScanner
from recovery.failure_classifier import SkillFailure
from robot.robot_interface import Pose


class AdaptiveGraspPlanner:
    def __init__(self,config):self.config=config;self.analyzer=GeometryAnalyzer(config);self.scanner=MultiViewObjectScanner(config);self.reasoner_calls=0
    def choose(self,c):
        start=time.monotonic();c.log.event('stage',{'stage':'ANALYZE_GEOMETRY'})
        force_scan=bool(getattr(c,'force_multiview_grasp',False))
        model=self.analyzer.analyze(c.source,c.world);constraints=c.goal.hard_constraints
        fast=(model['top_grasp_score']>=self.config.grasp.fast_top_score_threshold and model.get('gripper_fit_top')
              and c.source.stability_displacement_m is not None and c.source.stability_displacement_m<=.004
              and c.source.uncertainty_m<=self.config.localization.precise_uncertainty_m
              and not constraints&{'avoid_top_contact','prefer_handle','avoid_squeezing'} and c.goal.manipulation.grasp_specific_region in {None,'top'}
              and not force_scan)
        views=0
        if not fast and (force_scan or model['complexity_score']>=self.config.grasp.complexity_threshold or not model['graspable_regions']):
            c.log.event('stage',{'stage':'NEED_MORE_GEOMETRY'});model,points,views=self.scanner.scan(c);c.source.points=points
        c.force_multiview_grasp=False
        c.log.event('manipulation_geometry',{'object_id':c.source.id,'model':model})
        candidates=c.grasp_planner.candidates(c.source,c.world)
        if fast:candidates=[x for x in candidates if x.grasp_type=='top_grasp']
        if 'avoid_top_contact' in constraints:candidates=[x for x in candidates if x.contact_region!='top']
        if 'prefer_handle' in constraints:candidates=[x for x in candidates if x.contact_region=='handle']
        if c.goal.manipulation.grasp_specific_region:candidates=[x for x in candidates if x.contact_region==c.goal.manipulation.grasp_specific_region]
        candidates=[x for x in candidates if x.id not in c.recovery.failed_candidates and x.clearance>=c.config.safety.collision_margin_m]
        heading=np.arctan2(c.source.position[1],c.source.position[0])
        def cost(candidate):
            yaw=np.arctan2(candidate.orientation[1,0],candidate.orientation[0,0]);angle=abs(np.arctan2(np.sin(yaw-heading),np.cos(yaw-heading)))
            return ((candidate.grasp_type!='top_grasp',-candidate.score,-candidate.clearance,candidate.grasp_width,angle)
                    if fast else (-candidate.score,angle))
        candidates.sort(key=cost)
        feasible=[];pregrasp_paths={};lift_margins={}
        c.robot.configure_collision_world(c.world,c.source_id,None,None)
        for candidate in candidates:
            c.budget();distance=c.config.planner.side_approach_clearance_m if candidate.grasp_type=='side_grasp' else c.config.planner.approach_clearance_m
            hover=candidate.position-candidate.approach_vector*distance
            contact_solutions=[q for q in c.robot.solve_ik(Pose(candidate.position,candidate.orientation),c.robot.get_joint_positions())
                if c.robot.check_collision(q) is None]
            contact_ok=bool(contact_solutions)
            hover_solutions=c.robot.solve_ik(Pose(hover,candidate.orientation),c.robot.get_joint_positions()) if contact_ok else []
            hover_ok=any(c.robot.check_collision(q) is None for q in hover_solutions)
            if contact_ok and hover_ok and candidate.grasp_type=='side_grasp':
                c.robot.configure_collision_world(c.world,None,None,None)
                path=c.planner.plan_to_pose(Pose(hover,candidate.orientation))
                hover_ok=path is not None
                if hover_ok:pregrasp_paths[candidate.id]=path
                c.robot.configure_collision_world(c.world,c.source_id,None,None)
            lift_solutions=[]
            if contact_ok and hover_ok:
                lift_pose=Pose(candidate.position+np.array([0,0,.045]),candidate.orientation)
                lift_solutions=[q for q in c.robot.solve_ik(lift_pose,contact_solutions[0])
                    if c.robot.check_collision(q) is None]
            if contact_ok and hover_ok and lift_solutions:
                lower,upper=c.robot.joint_limits
                lift_margins[candidate.id]=max(float(np.min(np.minimum(q-lower,upper-q))) for q in lift_solutions)
                feasible.append(replace(candidate,reachability=True,collision_free=True))
            if len(feasible)>=6:break
        feasible.sort(key=lambda candidate:(-candidate.score,-lift_margins[candidate.id],-candidate.clearance,candidate.grasp_width))
        c.log.event('grasp_candidates',[{'id':x.id,'type':x.grasp_type,'position':x.position.tolist(),'width':x.grasp_width,'score':x.score,
            'clearance':x.clearance,'lift_joint_margin_rad':lift_margins[x.id],
            'region':x.contact_region,'approach':x.approach_direction,'reachability':x.reachability,'collision_free':x.collision_free} for x in feasible])
        if not feasible:
            if views==0 and not force_scan:
                c.force_multiview_grasp=True
                c.log.event('grasp_retry_multiview',{'reason':'single-view candidate width or route unsafe'})
                return self.choose(c)
            raise SkillFailure('GRASP_FAILED',{'reason':'no feasible grasp candidate for measured geometry/constraints'})
        chosen=feasible[0];client=c.recovery.replanner.client;reasoned=False;reasoner_called=False
        margin=abs(feasible[0].score-feasible[1].score) if len(feasible)>1 else 1.
        semantic=bool(constraints-{'avoid_collision'}) or c.source.category in {'cup','bottle'}
        if (not fast and self.config.grasp.llm_reasoner_enabled and client.online and self.reasoner_calls<self.config.grasp.max_reasoner_calls
            and (semantic or margin<self.config.grasp.reasoner_candidate_margin or len(c.recovery.history)>1)):
            self.reasoner_calls+=1
            reasoner_called=True
            try:
                decision=client.complete(GraspStrategyDecision,'grasp_strategy',{'goal':c.goal.model_dump(),'geometry':model,
                    'candidates':[{'id':x.id,'strategy':x.grasp_type,'region':x.contact_region,'approach':x.approach_direction,'score':x.score} for x in feasible]})
                selected=next((x for x in feasible if x.id==decision.selected_candidate_id and x.grasp_type==decision.strategy and x.contact_region==decision.preferred_region),None)
                if selected is None:raise ValueError('reasoner selected unavailable candidate')
                chosen=selected;reasoned=True;c.log.event('grasp_strategy',decision.model_dump())
            except (RuntimeError,ValueError) as exc:c.log.event('grasp_reasoner_fallback',{'reason':str(exc),'backend':'local_geometry'})
        c.log.event('adaptive_grasp',{'fast_path_used':fast,'multiview_used':views>0,'llm_grasp_reasoner_called':reasoner_called,'llm_selection_used':reasoned,
            'scan_views':views,'planning_time_s':time.monotonic()-start,'candidate':chosen.id,'strategy':chosen.grasp_type,
            'region':chosen.contact_region,'approach':chosen.approach_direction})
        c.pregrasp_path=pregrasp_paths.get(chosen.id)
        c.grasp_surface_offsets=c.source.points-c.source.position
        return chosen
