from enum import Enum


class FailureType(str,Enum):
    TARGET_NOT_FOUND="TARGET_NOT_FOUND"
    AMBIGUOUS_TARGET="AMBIGUOUS_TARGET"
    LOW_CONFIDENCE="LOW_CONFIDENCE"
    DEPTH_INVALID="DEPTH_INVALID"
    IK_FAILED="IK_FAILED"
    PATH_BLOCKED="PATH_BLOCKED"
    COLLISION_RISK="COLLISION_RISK"
    NO_CONTACT="NO_CONTACT"
    GRASP_FAILED="GRASP_FAILED"
    OBJECT_SLIPPED="OBJECT_SLIPPED"
    TARGET_LOST="TARGET_LOST"
    PLACEMENT_FAILED="PLACEMENT_FAILED"
    RELATION_NOT_SATISFIED="RELATION_NOT_SATISFIED"
    USER_STOP="USER_STOP"
    NO_VALID_PLACEMENT="NO_VALID_PLACEMENT"
    AMBIGUOUS_REFERENCE_FRAME="AMBIGUOUS_REFERENCE_FRAME"


class SkillFailure(RuntimeError):
    def __init__(self,failure,evidence=None):
        self.failure=FailureType(failure); self.evidence=evidence or {}
        super().__init__(self.failure.value)


class FailureClassifier:
    @staticmethod
    def classify(reason):
        if reason in FailureType._value2member_map_: return FailureType(reason)
        text=str(reason).lower()
        if "user_stop" in text: return FailureType.USER_STOP
        if "collision" in text or "limit" in text or "unexpected" in text or "deviation" in text: return FailureType.COLLISION_RISK
        if "contact" in text: return FailureType.NO_CONTACT
        if "slip" in text: return FailureType.OBJECT_SLIPPED
        if "ik" in text: return FailureType.IK_FAILED
        if "depth" in text: return FailureType.DEPTH_INVALID
        if "place" in text: return FailureType.PLACEMENT_FAILED
        if "path" in text: return FailureType.PATH_BLOCKED
        return FailureType.GRASP_FAILED
