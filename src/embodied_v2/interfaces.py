"""Dependency-inversion boundaries for V2 components."""

from __future__ import annotations

from typing import Protocol, Sequence

from .domain import (
    DistanceReading,
    GraspCandidate,
    GraspEvidence,
    JointTrajectory,
    ObjectObservation,
    PayloadState,
    PlacementEvidence,
    RecoveryDirective,
    RouteIntent,
    SafetyReport,
    SceneModel,
    TaskSemantics,
    Vec3,
)


class TaskManager(Protocol):
    def understand(self, instruction: str, scene: SceneModel | None = None) -> TaskSemantics: ...


class SceneScanner(Protocol):
    def scan(self) -> SceneModel: ...


class CloseRangePerception(Protocol):
    def observe(self, entity_id: str, scene: SceneModel) -> ObjectObservation: ...


class DistanceSensor(Protocol):
    def measure(self, origin_m: Vec3, direction: Vec3, max_distance_m: float) -> DistanceReading: ...


class GraspPlanner(Protocol):
    def candidates(
        self, observation: ObjectObservation, scene: SceneModel
    ) -> Sequence[GraspCandidate]: ...


class GraspController(Protocol):
    def execute(self, candidate: GraspCandidate, entity_id: str) -> GraspEvidence: ...


class RouteStrategist(Protocol):
    def propose(
        self,
        task: TaskSemantics,
        scene: SceneModel,
        payload: PayloadState,
        start_m: Vec3,
        destination_m: Vec3,
    ) -> Sequence[RouteIntent]: ...


class MotionPlanner(Protocol):
    def plan_transport(
        self,
        payload: PayloadState,
        intent: RouteIntent,
        target_m: Vec3,
        scene: SceneModel,
        *,
        yaw_rad: float = 0.0,
    ) -> JointTrajectory | None: ...


class TrajectoryValidator(Protocol):
    def validate(
        self,
        trajectory: JointTrajectory,
        scene: SceneModel,
        payload: PayloadState | None,
        phase: str,
    ) -> SafetyReport: ...


class PlacementController(Protocol):
    def place(
        self,
        payload: PayloadState,
        destination: ObjectObservation,
        scene: SceneModel,
    ) -> PlacementEvidence: ...


class RecoveryManager(Protocol):
    def decide(self, failure: str, phase: str, attempt: int) -> RecoveryDirective: ...
