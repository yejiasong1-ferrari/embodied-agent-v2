from dataclasses import dataclass
import numpy as np
from .spatial_planner import SpatialPlanner


@dataclass(frozen=True)
class CorrectionDecision:
    strategy: str
    delta_xy: np.ndarray
    reason: str


class PlacementCorrectionPlanner:
    def __init__(self,config):self.config=config;self.planner=SpatialPlanner(config)
    def decide(self,goal,source,target,world,desired,push_attempts=0,regrasps=0,held=False):
        cfg=self.config.placement_correction;delta=desired.position[:2]-source.position[:2]
        if held:return CorrectionDecision('REPLACE',delta,'OBJECT_STILL_HELD')
        def regrasp(reason):return CorrectionDecision('REGRASP' if regrasps<cfg.max_regrasps else 'ABORT',delta,reason)
        if push_attempts>=cfg.max_push_attempts:return regrasp('PUSH_ATTEMPT_LIMIT')
        # A second release can use the last measured gap as feedback. After
        # one regrasp, a residual larger than one safe push step is better
        # corrected by another controlled placement while budget remains.
        if regrasps and regrasps<cfg.max_regrasps and np.linalg.norm(delta)>cfg.push_step_m:
            return regrasp('RELEASE_FEEDBACK_REGRASP')
        remaining=cfg.max_push_attempts-push_attempts
        # A bounded push cannot reliably remove more than one push_step per
        # attempt. Switch while the tool is safely clear if the remaining
        # attempts cannot reach the task tolerance.
        if np.linalg.norm(delta)>remaining*cfg.push_step_m+desired.tolerance_m:
            return regrasp('INSUFFICIENT_PUSH_REACH')
        if not cfg.enable_push:return regrasp('PUSH_DISABLED')
        if goal.hard_constraints&{'fragile','no_sliding','avoid_squeezing'}:return regrasp('TASK_FORBIDS_PUSH')
        if np.linalg.norm(delta)>cfg.max_push_distance_m:return regrasp('CORRECTION_TOO_LARGE')
        if abs(desired.position[2]-source.position[2])>.01:return regrasp('HEIGHT_ERROR')
        profile=getattr(source,'manipulation_geometry',{})
        width=profile.get('lower_width') or float(min(source.size[:2]))
        if source.size[2]/max(width,.001)>cfg.max_height_width_ratio:return regrasp('TIPPING_RISK')
        stable=getattr(source,'stability_displacement_m',None)
        if stable is None or stable>.003:return regrasp('STABILITY_NOT_OBSERVED')
        if not source.visible:return CorrectionDecision('ABORT',delta,'TARGET_LOST')
        for t in np.linspace(0,1,8):
            point=source.position.copy();point[:2]+=delta*t
            valid,reason,_=self.planner.feasibility(source,target,world,point,desired.relation,cfg.min_table_edge_margin_m,desired.support_id)
            if not valid:return regrasp(reason)
        return CorrectionDecision('PUSH',delta,'SMALL_STABLE_CLEAR_XY_ERROR')
