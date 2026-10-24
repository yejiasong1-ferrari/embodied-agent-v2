import numpy as np
from llm.schemas import SymbolicPlan,PlanStep,Goal,Reference
from world.world_model import WorldObject


MANIPULATION_SEQUENCE=("OBSERVE_SCENE","FIND_OBJECT","APPROACH_OBJECT","GRASP_OBJECT","VERIFY_GRASP","LIFT_OBJECT","MOVE_TO_TARGET","PLACE_OBJECT","RETREAT","VERIFY_TASK")


class TaskPlanner:
    def __init__(self,client,log): self.client,self.log=client,log
    def plan(self,goal,world):
        if self.client.online:
            try:
                plan=self.client.complete(SymbolicPlan,"plan",{"goal":goal.model_dump(),"world":world.to_dict()})
                names=[s.skill for s in plan.steps]
                # Mandatory order is a local contract; a model cannot skip
                # grasp verification, safety, release or final perception.
                filtered=[name for name in names if name not in {"SCAN_SCENE","MOVE_CAMERA"}]
                if tuple(filtered)!=MANIPULATION_SEQUENCE: raise ValueError("unsafe/incomplete skill sequence")
                return plan
            except (RuntimeError,ValueError):
                if self.client.config.mode=="required": raise
                self.log.event("llm_fallback",{"purpose":"plan","backend":"validated_skill_sequence"})
        return SymbolicPlan(steps=[PlanStep(skill=s) for s in MANIPULATION_SEQUENCE],reason="local verified manipulation sequence")

    def blockers(self,source,world):
        result=[]
        for obj in world.objects.values():
            if obj.id==source.id or not obj.movable: continue
            overlap=np.minimum(source.high[:2]+.035,obj.high[:2])-np.maximum(source.low[:2]-.035,obj.low[:2])
            if np.all(overlap>0) and obj.high[2]>source.low[2]+.02: result.append(obj)
        return sorted(result,key=lambda o:o.high[2],reverse=True)

    def free_region(self,source,world,workspace):
        bounds=np.asarray(workspace)
        candidates=[]
        for x in np.linspace(max(bounds[0,0]+.12,-.55),min(bounds[0,1]-.12,.55),11):
            for y in np.linspace(max(bounds[1,0]+.12,-.55),min(bounds[1,1]-.12,.55),11):
                p=np.array([x,y])
                if np.linalg.norm(p)<.40 or np.linalg.norm(p)>.65: continue
                patch=world.support_points[np.all(np.abs(world.support_points[:,:2]-p)<.065,axis=1)]
                if len(patch)<50: continue
                # Require depth samples in all quadrants of the placement footprint.
                if any(np.count_nonzero(((patch[:,0]-x)*sx>0) & ((patch[:,1]-y)*sy>0))<5 for sx in (-1,1) for sy in (-1,1)): continue
                # Reserve the gripper's footprint as well as the payload.
                # An empty patch fitting the cup alone can trap the fingers
                # against a neighbouring tall object during release.
                clearance=min((np.linalg.norm(p-o.position[:2])-max(.10,max(source.size[:2])/2)-max(o.size[:2])/2
                               for o in [*world.objects.values(),*world.regions.values()]),default=.5)
                if clearance>.025: candidates.append((np.linalg.norm(p-source.position[:2]),p))
        if not candidates: raise LookupError("no visually observed free staging region")
        # The supporting plane comes from the source's measured lower surface.
        p=min(candidates,key=lambda c:c[0])[1]
        patch=world.support_points[np.all(np.abs(world.support_points[:,:2]-p)<.065,axis=1)]
        region=WorldObject("staging_region","free table region","region","region",np.array([*p,float(np.median(patch[:,2]))-.002]),
                          np.array([0,0,0,1]),np.array([.13,.13,.004]),.8,world.timestamp,movable=False,
                          evidence_source="observed_support_region")
        return region
