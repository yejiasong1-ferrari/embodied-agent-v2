import numpy as np
from llm.schemas import SpatialConstraint
from .operator_frame import relation_axis
from .geometry import footprint,projected_edges,polygon_gap,overlap


def goal_constraints(goal):
    return goal.spatial_constraints or [SpatialConstraint(relation=goal.relation)]


class RelationValidator:
    def __init__(self,config):self.config=config
    def evaluate(self,goal,source,target,world,position=None,require_visible=True):
        p=np.asarray(source.position if position is None else position);rows=[]
        for constraint in goal_constraints(goal):
            reference=world.resolve(constraint.reference) if constraint.reference else target
            row=self.constraint(constraint,source,reference,world,p,measured=position is None)
            if require_visible and (not source.visible or not reference.visible):row['satisfied']=False;row['reason']='TARGET_LOST'
            rows.append(row)
        accepted=bool(rows and all(row['satisfied'] for row in rows))
        return accepted,{'relation_satisfied':accepted,'fresh_visual_evidence':bool(source.visible and target.visible),
                         'constraints':rows,'requested_position':p.tolist() if position is not None else None,
                         'source':source.to_dict(),'target':target.to_dict()}
    def constraint(self,c,source,reference,world,p,measured=False):
        cfg=self.config.spatial;tol=c.tolerance_m or cfg.default_tolerance_m
        a=footprint(source,None if measured else p);b=footprint(reference)
        axis=relation_axis(c.relation,c.reference_frame,cfg,reference)
        center=float(np.linalg.norm(p-reference.position));gap=polygon_gap(a,b)
        row={'type':c.type,'relation':c.relation,'reference':reference.id,'reference_frame':c.reference_frame,
             'distance_semantics':c.distance_semantics,'requested_gap_m':c.distance_m,'tolerance_m':tol}
        if c.type!='relative_position':
            actual=center if c.distance_semantics=='center_distance' else gap
            error=actual-c.distance_m
            ok=error>=-tol if c.type=='min_distance' else error<=tol if c.type=='max_distance' else abs(error)<=tol
            row.update(actual_distance_m=actual,distance_error_m=abs(error),satisfied=bool(ok));return row
        relation=c.relation
        if axis is not None:
            low,_=projected_edges(source,axis,None if measured else p);_,high=projected_edges(reference,axis)
            signed=float((p-reference.position)@axis)
            actual=signed if c.distance_semantics=='center_distance' else low-high
            residual=(p-reference.position)-axis*signed
            alignment=float(np.linalg.norm(residual[:2] if abs(axis[2])<.9 else residual))
            correct=signed>0 and actual>=-tol
            distance_ok=abs(actual-c.distance_m)<=tol if c.distance_m is not None else actual>=-tol
            row.update(actual_gap_m=actual,distance_error_m=None if c.distance_m is None else abs(actual-c.distance_m),
                       direction_correct=bool(correct),alignment_error_m=alignment,
                       satisfied=bool(correct and distance_ok and alignment<=cfg.direction_alignment_tolerance_m))
        elif relation=='on':
            height=abs(p[2]-source.size[2]/2-reference.high[2])
            inside=all(__import__('cv2').pointPolygonTest(b.astype(np.float32),tuple(map(float,point)),True)>=-tol for point in a)
            row.update(satisfied=bool(inside and height<=self.config.safety.placement_z_tolerance_m),height_error_m=float(height))
        elif relation=='inside':
            row.update(satisfied=False,reason='CAVITY_GEOMETRY_REQUIRED')
        elif relation=='near':
            requested=c.distance_m;ok=abs(gap-requested)<=tol if requested is not None else cfg.near_min_gap_m-tol<=gap<=cfg.near_max_gap_m+tol
            row.update(satisfied=bool(ok and not overlap(a,b)),actual_gap_m=gap,distance_error_m=None if requested is None else abs(gap-requested))
        elif relation=='centered_with':
            error=float(np.linalg.norm(p[:2]-reference.position[:2]));row.update(satisfied=error<=tol,distance_error_m=error)
        elif relation=='between':
            refs=[world.resolve(ref) for ref in c.references];start,end=[r.position[:2] for r in refs]
            delta=end-start;t=float((p[:2]-start)@delta/max(delta@delta,1e-12))
            error=float(np.linalg.norm(p[:2]-(start+np.clip(t,0,1)*delta)))
            clear=all(not overlap(a,footprint(r)) for r in refs)
            row.update(satisfied=bool(.05<t<.95 and clear and error<=tol),segment_fraction=t,distance_error_m=error)
        else:row.update(satisfied=False,reason='UNSUPPORTED_RELATION')
        return row
