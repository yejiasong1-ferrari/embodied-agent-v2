"""Failure-typed local recovery; each retry changes its search dimensions."""

from __future__ import annotations

from ..domain import FailureType, Phase, RecoveryDirective


_POLICY = {
    FailureType.PERCEPTION: ("rescan", Phase.SCAN_SCENE, ("viewpoint", "segmentation")),
    FailureType.SENSOR_CONFLICT: ("reobserve", Phase.CLOSE_REOBSERVE, ("sensor_fusion", "viewpoint")),
    # Depth and centering already ran inside the same grasp. A remaining
    # no-contact result means that axis is exhausted, so the next hypothesis
    # uses a different closing direction.
    FailureType.NO_CONTACT: ("change_axis", Phase.PRE_ALIGN, ("axis",)),
    FailureType.UNILATERAL_CONTACT: ("recenter_reorient", Phase.PRE_ALIGN, ("xy", "yaw")),
    FailureType.OBJECT_PUSHED: ("retreat_reobserve", Phase.CLOSE_REOBSERVE, ("approach", "centering")),
    FailureType.LIFT_SLIP: ("change_grasp", Phase.PRE_ALIGN, ("region", "depth", "yaw")),
    FailureType.APPROACH_COLLISION: ("change_approach", Phase.APPROACH, ("approach", "ik_branch")),
    FailureType.IK_UNREACHABLE: ("change_pose", Phase.APPROACH, ("pose", "ik_branch")),
    FailureType.TRANSPORT_BLOCKED: ("new_corridor", Phase.PLAN_TRANSPORT, ("corridor", "escape", "ik_branch")),
    FailureType.RUNTIME_INSTABILITY: ("stop_and_regrasp", Phase.CLOSE_REOBSERVE, ("grasp", "speed")),
    FailureType.PLACEMENT_UNSTABLE: ("reposition", Phase.PRE_PLACE, ("offset", "wrist", "height")),
    FailureType.VERIFICATION: ("inspect_and_recover", Phase.REOBSERVE_DESTINATION, ("viewpoint", "relation")),
}


class TypedRecoveryManager:
    def __init__(self, maximum_attempts: int = 3):
        self.maximum_attempts = int(maximum_attempts)

    def decide(self, failure: FailureType | str, phase: Phase | str, attempt: int) -> RecoveryDirective:
        failure_type = failure if isinstance(failure, FailureType) else FailureType(failure)
        if attempt >= self.maximum_attempts or failure_type not in _POLICY:
            return RecoveryDirective(
                "safe_stop", Phase.STOPPED, ("operator_review",),
                f"retry budget exhausted after {attempt} attempts",
            )
        action, restart, dimensions = _POLICY[failure_type]
        phase_name = phase.value if isinstance(phase, Phase) else phase
        return RecoveryDirective(
            action, restart, dimensions,
            f"{failure_type.value} during {phase_name}",
        )
