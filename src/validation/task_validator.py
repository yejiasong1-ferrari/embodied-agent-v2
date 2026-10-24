import numpy as np
from .placement_validator import PlacementValidator


class TaskValidator:
    def __init__(self,config): self.config=config; self.placement=PlacementValidator(config)
    def validate(self,goal,source,target,previous_position,held,world=None):
        if world is not None and goal.spatial_constraints:
            from spatial.relation_validator import RelationValidator
            ok,evidence=RelationValidator(self.config).evaluate(goal,source,target,world)
        else:ok,evidence=self.placement.validate(source,target,goal.relation)
        displacement=float(np.linalg.norm(source.position-previous_position))
        evidence.update(stable_displacement_m=displacement,object_released=not held)
        upright=True
        if goal.hard_constraints&{'keep_upright','keep_object_upright'}:
            normal=source.manipulation_geometry.get('top_surface_normal')
            normal=np.asarray(normal,float) if normal is not None else np.empty(0)
            upright=bool(normal.shape==(3,) and np.linalg.norm(normal)>.9 and
                         normal[2]/np.linalg.norm(normal)>=np.cos(np.deg2rad(12)))
            evidence['upright_surface_normal']=normal.tolist() if len(normal) else None
            evidence['upright_verified']=upright
        accepted=bool(ok and upright and not held and displacement<=self.config.safety.stable_displacement_m)
        evidence['accepted']=accepted
        return accepted,evidence
