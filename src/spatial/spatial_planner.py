from dataclasses import dataclass
from llm.schemas import SpatialConstraint
import numpy as np
from .operator_frame import relation_axis
from .relation_validator import RelationValidator,goal_constraints
from .geometry import footprint,projected_edges,polygon_gap,support_covered,supported_by


@dataclass(frozen=True)
class DesiredPlacement:
    position: np.ndarray
    orientation: np.ndarray
    tolerance_m: float
    relation: str
    score: float
    evidence: dict
    support_id: str = 'observed_table'


class SpatialPlanner:
    def __init__(self,config):self.config=config;self.validator=RelationValidator(config)
    def feasibility(self,source,target,world,position,relation,margin=None,support_id=None):
        p=np.asarray(position);cfg=self.config.spatial;poly=footprint(source,p);bounds=np.asarray(self.config.safety.workspace)
        if np.any(poly<bounds[:2,0]) or np.any(poly>bounds[:2,1]):return False,'WORKSPACE_EDGE',0.
        if np.linalg.norm(p[:2])<self.config.planner.base_clearance_radius_m:return False,'BASE_CLEARANCE',0.
        support=world.objects.get(support_id) if support_id else target if relation in {'on','centered_with'} else None
        if support is not None:
            if not supported_by(source,support,p,0. if margin is None else margin):return False,'SUPPORT_EXTENT',0.
        elif not support_covered(world,source,p,cfg.min_table_edge_margin_m if margin is None else margin):return False,'UNOBSERVED_SUPPORT_OR_EDGE',0.
        clearance=1.
        memory=getattr(world,'memory',None)
        carried_fragments=(set(memory.held_fragments)
            if memory is not None and memory.held_id==source.id else set())
        for other in world.objects.values():
            if other.id==source.id or other.id in carried_fragments or support is not None and other.id==support.id:continue
            if p[2]-source.size[2]/2>other.high[2]+.015:continue
            gap=polygon_gap(poly,footprint(other));clearance=min(clearance,gap)
            if gap<cfg.min_obstacle_margin_m:return False,'OBSTACLE_CLEARANCE',clearance
        return True,None,clearance
    def plan(self,goal,source,target,world,reachable=None,excluded_positions=()):
        cfg=self.config.spatial;primary=next((c for c in goal_constraints(goal) if c.type=='relative_position'),None)
        if primary is None:raise ValueError('NO_VALID_PLACEMENT: missing primary relation')
        reference=world.resolve(primary.reference) if primary.reference else target
        axis=relation_axis(primary.relation,primary.reference_frame,cfg,reference);positions=[]
        if primary.relation in {'inside'}:raise ValueError('NO_VALID_PLACEMENT: cavity geometry unavailable')
        if primary.relation in {'above','below'}:
            # A released object cannot float. Candidate must have measured support.
            sign=axis[2];gap=primary.distance_m or cfg.near_min_gap_m
            z=reference.position[2]+sign*((source.size[2]+reference.size[2])/2+gap)
            positions=[np.array([*reference.position[:2],z])]
        elif primary.relation in {'on','centered_with'}:
            positions=[reference.position.copy()+np.array([0,0,(reference.size[2]+source.size[2])/2])]
        elif primary.relation=='between':
            refs=[world.resolve(r) for r in primary.references]
            positions=[(1-t)*refs[0].position+t*refs[1].position for t in [.5,.4,.6,.3,.7]]
        else:
            axes=[axis] if axis is not None else [np.array([np.cos(a),np.sin(a),0]) for a in np.linspace(0,2*np.pi,16,endpoint=False)]
            gaps=[primary.distance_m] if primary.distance_m is not None else np.linspace(cfg.near_min_gap_m,cfg.near_max_gap_m,4)
            for direction in axes:
                _,ref_high=projected_edges(reference,direction);src_low,_=projected_edges(source,direction)
                half_source=float(source.position@direction)-src_low
                extent=ref_high-float(reference.position@direction)+half_source
                for gap in gaps:
                    distance=gap if primary.distance_semantics=='center_distance' else extent+gap
                    for offset in [0,-cfg.candidate_step_m,cfg.candidate_step_m]:
                        positions.append(reference.position+direction*distance+np.array([-direction[1],direction[0],0])*offset)
        candidates=[];rejected=[]
        for point in positions:
            point=point.copy()
            if any(np.linalg.norm(point[:2]-np.asarray(old)[:2])<.008 for old in excluded_positions):
                rejected.append('PREVIOUS_ROUTE_BLOCKED');continue
            support_id=reference.id if primary.relation in {'on','centered_with'} else 'observed_table'
            if primary.relation not in {'on','above','below','centered_with'}:
                patch=world.support_points[np.linalg.norm(world.support_points[:,:2]-point[:2],axis=1)<.10]
                if not len(patch):rejected.append('UNOBSERVED_SUPPORT');continue
                point[2]=float(np.median(patch[:,2]))+source.size[2]/2
            if primary.relation in {'above','below'}:
                supports=[o for o in world.objects.values() if o.id!=source.id and supported_by(source,o,point)]
                if not supports:rejected.append('NO_SUPPORT_AT_REQUESTED_HEIGHT');continue
                support_id=supports[0].id
            ok,reason,clearance=self.feasibility(source,reference,world,point,primary.relation,support_id=support_id)
            if not ok:rejected.append(reason);continue
            accepted,evidence=self.validator.evaluate(goal,source,reference,world,point,require_visible=False)
            if not accepted:rejected.append('SPATIAL_CONSTRAINT');continue
            alignment=0.
            if axis is not None and abs(axis[2])<.9:
                delta=point-reference.position;alignment=float(np.linalg.norm((delta-axis*(delta@axis))[:2]))
            score=float(min(clearance,.15)*2-np.linalg.norm(point-source.position)*.1-alignment)
            candidates.append(DesiredPlacement(point,source.orientation.copy(),cfg.default_tolerance_m,primary.relation,score,evidence,support_id))
        for candidate in sorted(candidates,key=lambda c:c.score,reverse=True):
            if reachable is None or reachable(candidate.position):return candidate
            rejected.append('REACHABILITY')
        raise ValueError('NO_VALID_PLACEMENT: '+','.join(sorted(set(rejected or ['REACHABILITY']))))
