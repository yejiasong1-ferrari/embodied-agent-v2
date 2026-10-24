from .failure_classifier import FailureType


class RecoveryManager:
    def __init__(self,config,replanner,log):
        self.config,self.replanner,self.log=config,replanner,log
        self.history=[]; self.failed_candidates=set()

    def decide(self,failure,context):
        kind=failure.failure
        candidate=context.candidate.id if context.candidate else None
        row={"failure":kind.value,"candidate":candidate,"evidence":failure.evidence,"skill":context.current_skill,
             "target_pose":None if context.source is None else context.source.position.tolist(),
             "contact":context.robot.get_contact_state().__dict__,
             "strategy":None if context.candidate is None else getattr(context.candidate,"grasp_type",None),
             "region":None if context.candidate is None else getattr(context.candidate,"contact_region",None),
             "approach":None if context.candidate is None else getattr(context.candidate,"approach_direction",None)}
        self.history.append(row)
        def recorded(action):
            row['action']=action
            row['attempt']=len(self.history)
            self.log.event('recovery',row)
            return action
        if candidate and kind in {FailureType.NO_CONTACT,FailureType.GRASP_FAILED,FailureType.IK_FAILED,FailureType.OBJECT_SLIPPED,FailureType.PATH_BLOCKED}:
            self.failed_candidates.add(candidate)
        if len(self.history)>self.config.recovery.max_recoveries: return recorded("ABORT")
        if kind in {FailureType.COLLISION_RISK,FailureType.USER_STOP,FailureType.NO_VALID_PLACEMENT,
                    FailureType.AMBIGUOUS_REFERENCE_FRAME,FailureType.AMBIGUOUS_TARGET}: return recorded("ABORT")
        if failure.evidence.get("unsupported"): return recorded("ABORT")
        if failure.evidence.get("detail_reviewed"): return recorded("ABORT")
        if failure.evidence.get("reason")=="distinct grasp candidate budget exhausted": return recorded("ABORT")
        if sum(h["failure"]==kind.value and h["candidate"]==candidate for h in self.history)>2: return recorded("ABORT")
        if kind in {FailureType.TARGET_NOT_FOUND,FailureType.TARGET_LOST,FailureType.LOW_CONFIDENCE,FailureType.DEPTH_INVALID}: action="REOBSERVE"
        elif kind in {FailureType.NO_CONTACT,FailureType.GRASP_FAILED,FailureType.IK_FAILED,FailureType.OBJECT_SLIPPED}: action="CHANGE_GRASP"
        else: action="REPLAN"
        if context.held and kind in {FailureType.IK_FAILED,FailureType.PATH_BLOCKED}: action="REPLAN"
        if len(self.history)>=self.config.recovery.llm_after_failures and self.replanner.client.online:
            try:
                result=self.replanner.reason(context.goal,context.world,kind.value,self.history,
                    {**failure.evidence,"grasp_strategy":candidate,"distance":context.target_distance(),
                     "relevant_image_description":[o.semantic_label for o in context.world.objects.values() if o.visible]},
                    {"last_failure":context.safety.last_failure,"carrying":context.held})
                action=result.action
            except RuntimeError:
                if self.replanner.client.config.mode=="required": action="ABORT"
        if context.held and kind not in {FailureType.OBJECT_SLIPPED} and action in {"CHANGE_GRASP","CHANGE_APPROACH"}:
            action="REPLAN"
        return recorded(action)
