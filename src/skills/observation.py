from .skill_base import Skill
from recovery.failure_classifier import SkillFailure
from world.world_model import AmbiguousReferenceError,ReferenceNotFoundError


class ObserveSkill(Skill):
    success_conditions=("fresh RGB-D observation",)
    failure_conditions=("DEPTH_INVALID","TARGET_NOT_FOUND")
    def execute(self,c): c.refresh(scan=True); return True


class ScanSceneSkill(ObserveSkill): pass


class MoveCameraSkill(Skill):
    success_conditions=("changed viewpoint","fresh observation")
    failure_conditions=("PATH_BLOCKED",)
    def execute(self,c):
        camera=c.perception.camera
        if not camera.move_viewpoint((getattr(camera,"index",0)+1)%camera.viewpoint_count): raise SkillFailure("PATH_BLOCKED")
        c.refresh(); return True


class FindObjectSkill(Skill):
    success_conditions=("unambiguous visible source and target",)
    failure_conditions=("TARGET_NOT_FOUND","AMBIGUOUS_TARGET","LOW_CONFIDENCE")
    def execute(self,c):
        def find(reference):
            try:
                obj=c.perception.identify(reference)
                key=(reference.description,reference.color,reference.category)
                if (reference.category=='cup' and key not in c.world.reference_bindings
                        and obj.size[2]/max(min(obj.size[:2]),.01)<.65):
                    # A shallow circular component can be a bowl or puck.
                    # Its lone presence must not certify it as the cup.
                    if not c.perception.disambiguate(reference,[obj]):
                        raise SkillFailure('LOW_CONFIDENCE',{'reason':'shallow circular object is not visually confirmed as a cup',
                            'reference':reference.description,'candidate_ids':[obj.id],'detail_reviewed':True})
                return c.world.resolve(reference,require_visible=True)
            except AmbiguousReferenceError as exc:
                candidates=c.world.related_candidates(reference)
                raise SkillFailure('AMBIGUOUS_TARGET',{'reason':str(exc),'reference':reference.description,
                    'candidate_ids':[o.id for o in candidates]}) from exc
            except ReferenceNotFoundError as exc:
                raise SkillFailure('TARGET_NOT_FOUND',{'reason':str(exc),'reference':reference.description}) from exc
        c.source=find(c.goal.object)
        if c.goal.target:c.target=find(c.goal.target)
        if not c.source.movable: raise SkillFailure("GRASP_FAILED",{"reason":"source not visually graspable"})
        if c.source.confidence<c.config.perception.confidence_threshold: raise SkillFailure("LOW_CONFIDENCE")
        if c.source.uncertainty_m>c.config.perception.max_uncertainty_m: raise SkillFailure("LOW_CONFIDENCE",{"reason":"pose uncertainty"})
        if c.goal.relation=="inside":
            raise SkillFailure("PLACEMENT_FAILED",{"reason":"insertion requires visually reconstructed cavity; bounding box alone is insufficient","unsupported":True})
        c.source_id=c.source.id; c.target_id=c.target.id if c.target is not None else None
        c.world.memory.retained_ids.update([identity for identity in [c.source_id,c.target_id] if identity])
        c.world.memory.mission_active=True
        return True
