"""Result-based micro-lift verification; force symmetry is diagnostic only."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain import ContactState, GraspEvidence


@dataclass(frozen=True)
class StabilityThresholds:
    minimum_rise_m: float = 0.003
    maximum_relative_slip_m: float = 0.010
    maximum_lateral_displacement_m: float = 0.010
    maximum_tilt_deg: float = 8.0


def evaluate_micro_lift(
    evidence: GraspEvidence,
    thresholds: StabilityThresholds = StabilityThresholds(),
) -> tuple[bool, str]:
    if evidence.hard_force_exceeded:
        return False, "hard_force_limit"
    if evidence.contact_state is not ContactState.BILATERAL or not evidence.contact_persisted:
        return False, "contact_not_persistent"
    if evidence.relative_slip_m > thresholds.maximum_relative_slip_m:
        return False, "relative_slip"
    if evidence.lateral_displacement_m > thresholds.maximum_lateral_displacement_m:
        return False, "large_displacement"
    if evidence.tilt_deg > thresholds.maximum_tilt_deg:
        return False, "unstable_tilt"
    if evidence.object_rise_m < thresholds.minimum_rise_m:
        return False, "object_did_not_follow"
    return True, "stable_grasp"
