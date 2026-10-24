"""Shared immutable domain types for the V2 manipulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
JointVector = tuple[float, ...]


class Relation(str, Enum):
    ON = "on"
    NEXT_TO = "next_to"


class Phase(str, Enum):
    IDLE = "idle"
    UNDERSTAND_TASK = "understand_task"
    SCAN_SCENE = "scan_scene"
    APPROACH = "approach"
    CLOSE_REOBSERVE = "close_reobserve"
    PRE_ALIGN = "pre_align"
    CONTACT_SEARCH = "contact_search"
    MICRO_LIFT = "micro_lift"
    STABLE_GRASP = "stable_grasp"
    PLAN_TRANSPORT = "plan_transport"
    TRANSPORT = "transport"
    REOBSERVE_DESTINATION = "reobserve_destination"
    PRE_PLACE = "pre_place"
    DESCENT = "descent"
    FIRST_CONTACT = "first_contact"
    LOAD_TRANSFER = "load_transfer"
    GRADUAL_RELEASE = "gradual_release"
    VERIFY_RESULT = "verify_result"
    RECOVERY = "recovery"
    DONE = "done"
    STOPPED = "stopped"


class ContactState(str, Enum):
    NO_CONTACT = "no_contact"
    LEFT_ONLY = "left_only"
    RIGHT_ONLY = "right_only"
    BILATERAL = "bilateral_contact"
    LOST = "contact_lost"


class FailureType(str, Enum):
    PERCEPTION = "perception_failure"
    SENSOR_CONFLICT = "sensor_conflict"
    NO_CONTACT = "no_contact"
    UNILATERAL_CONTACT = "unilateral_contact"
    OBJECT_PUSHED = "object_pushed"
    LIFT_SLIP = "lift_slip"
    APPROACH_COLLISION = "approach_collision"
    IK_UNREACHABLE = "ik_unreachable"
    TRANSPORT_BLOCKED = "transport_blocked"
    RUNTIME_INSTABILITY = "runtime_instability"
    PLACEMENT_UNSTABLE = "placement_unstable"
    VERIFICATION = "verification_failure"
    RETRY_EXHAUSTED = "retry_exhausted"


class SafetyVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True)
class TaskSemantics:
    object_color: str
    object_category: str
    destination_color: str
    destination_category: str
    relation: Relation = Relation.ON
    fragile: bool = False
    source_text: str = ""


@dataclass(frozen=True)
class AABB:
    minimum: Vec3
    maximum: Vec3

    @property
    def center(self) -> Vec3:
        return tuple(
            (low + high) * 0.5
            for low, high in zip(self.minimum, self.maximum, strict=True)
        )  # type: ignore[return-value]

    @property
    def size(self) -> Vec3:
        return tuple(
            high - low
            for low, high in zip(self.minimum, self.maximum, strict=True)
        )  # type: ignore[return-value]

    def expanded(self, margin_m: float) -> "AABB":
        margin = float(margin_m)
        return AABB(
            tuple(value - margin for value in self.minimum),  # type: ignore[arg-type]
            tuple(value + margin for value in self.maximum),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class SceneEntity:
    entity_id: str
    center_m: Vec3
    bounds: AABB
    color: str
    geometry: str
    movable: bool
    support_candidate: bool = False
    orientation_xyzw: Quat = (0.0, 0.0, 0.0, 1.0)
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GapRegion:
    name: str
    description: str
    clearance: str
    centroid_m: Vec3


@dataclass(frozen=True)
class SemanticGapMap:
    regions: tuple[GapRegion, ...]
    suggested_corridors: tuple[str, ...]
    blocked_regions: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneModel:
    timestamp_s: float
    entities: tuple[SceneEntity, ...]
    obstacles: tuple[SceneEntity, ...]
    supports: tuple[SceneEntity, ...]
    occupancy_resolution_m: float
    gap_map: SemanticGapMap

    def entity(self, entity_id: str) -> SceneEntity:
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        raise KeyError(entity_id)


@dataclass(frozen=True)
class ObjectObservation:
    entity_id: str
    center_m: Vec3
    bounds: AABB
    yaw_rad: float
    width_m: float
    height_m: float
    top_z_m: float
    bottom_z_m: float
    protrusion_direction_rad: float | None
    local_obstacle_ids: tuple[str, ...]
    source: str
    confidence: float


@dataclass(frozen=True)
class DistanceReading:
    distance_m: float | None
    origin_m: Vec3
    direction: Vec3
    source: str
    valid: bool
    hit_entity_id: str | None = None


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    center_m: Vec3
    yaw_rad: float
    approach_direction: Vec3
    finger_target_height_m: float
    palm_height_m: float
    insertion_depth_m: float
    expected_width_m: float
    score: float
    rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraspEvidence:
    contact_state: ContactState
    left_force_n: float
    right_force_n: float
    object_rise_m: float
    relative_slip_m: float
    lateral_displacement_m: float
    tilt_deg: float
    contact_persisted: bool
    hard_force_exceeded: bool

    @property
    def force_ratio_diagnostic(self) -> float:
        low = min(self.left_force_n, self.right_force_n)
        return max(self.left_force_n, self.right_force_n) / max(low, 0.05)


@dataclass(frozen=True)
class PayloadState:
    entity_id: str
    geometry: AABB
    gripper_to_object_translation_m: Vec3
    gripper_to_object_quaternion: Quat
    bottom_offset_m: float
    measured_slip_m: float
    measured_tilt_deg: float
    envelope_margin_m: float


@dataclass(frozen=True)
class RouteIntent:
    corridor: str
    high_clearance: bool
    approach_side: str
    avoid_regions: tuple[str, ...] = ()
    rationale: str = ""
    # Symbolic arm shape only. Joint values stay in the local IK layer.
    posture: str = "keep"


@dataclass(frozen=True)
class JointTrajectory:
    positions: tuple[JointVector, ...]
    times_s: tuple[float, ...]
    source: str

    def __post_init__(self) -> None:
        if len(self.positions) != len(self.times_s) or not self.positions:
            raise ValueError("trajectory positions/times must be non-empty and aligned")
        if any(later <= earlier for earlier, later in zip(self.times_s, self.times_s[1:])):
            raise ValueError("trajectory times must be strictly increasing")


@dataclass(frozen=True)
class SafetyReport:
    verdict: SafetyVerdict
    reason: str
    minimum_robot_clearance_m: float
    minimum_payload_clearance_m: float
    samples_checked: int
    collision_pair: str | None = None


@dataclass(frozen=True)
class PlacementEvidence:
    support_contact: bool
    displacement_m: float
    tilt_deg: float
    still_moving: bool
    relation_satisfied: bool


@dataclass(frozen=True)
class RecoveryDirective:
    action: str
    restart_phase: Phase
    changed_search_dimensions: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class PhaseEvent:
    phase: Phase
    verdict: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    final_phase: Phase
    events: tuple[PhaseEvent, ...]
    failure: FailureType | None = None
    message: str = ""


def vec3(values: Sequence[float]) -> Vec3:
    if len(values) != 3:
        raise ValueError("expected three coordinates")
    return (float(values[0]), float(values[1]), float(values[2]))
