"""Explicit phase machine; illegal shortcuts are rejected locally."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .domain import Phase, PhaseEvent


_ALLOWED: dict[Phase, frozenset[Phase]] = {
    Phase.IDLE: frozenset({Phase.UNDERSTAND_TASK}),
    Phase.UNDERSTAND_TASK: frozenset({Phase.SCAN_SCENE, Phase.STOPPED}),
    Phase.SCAN_SCENE: frozenset({Phase.APPROACH, Phase.RECOVERY, Phase.STOPPED}),
    Phase.APPROACH: frozenset({Phase.CLOSE_REOBSERVE, Phase.RECOVERY, Phase.STOPPED}),
    Phase.CLOSE_REOBSERVE: frozenset({Phase.PRE_ALIGN, Phase.RECOVERY, Phase.STOPPED}),
    Phase.PRE_ALIGN: frozenset({Phase.CONTACT_SEARCH, Phase.RECOVERY, Phase.STOPPED}),
    Phase.CONTACT_SEARCH: frozenset({Phase.MICRO_LIFT, Phase.RECOVERY, Phase.STOPPED}),
    Phase.MICRO_LIFT: frozenset({Phase.STABLE_GRASP, Phase.RECOVERY, Phase.STOPPED}),
    Phase.STABLE_GRASP: frozenset({Phase.PLAN_TRANSPORT, Phase.RECOVERY, Phase.STOPPED}),
    Phase.PLAN_TRANSPORT: frozenset({Phase.TRANSPORT, Phase.RECOVERY, Phase.STOPPED}),
    Phase.TRANSPORT: frozenset({Phase.REOBSERVE_DESTINATION, Phase.RECOVERY, Phase.STOPPED}),
    Phase.REOBSERVE_DESTINATION: frozenset({Phase.PRE_PLACE, Phase.RECOVERY, Phase.STOPPED}),
    Phase.PRE_PLACE: frozenset({Phase.DESCENT, Phase.RECOVERY, Phase.STOPPED}),
    Phase.DESCENT: frozenset({Phase.FIRST_CONTACT, Phase.RECOVERY, Phase.STOPPED}),
    Phase.FIRST_CONTACT: frozenset({Phase.LOAD_TRANSFER, Phase.RECOVERY, Phase.STOPPED}),
    Phase.LOAD_TRANSFER: frozenset({Phase.GRADUAL_RELEASE, Phase.RECOVERY, Phase.STOPPED}),
    Phase.GRADUAL_RELEASE: frozenset({Phase.VERIFY_RESULT, Phase.RECOVERY, Phase.STOPPED}),
    Phase.VERIFY_RESULT: frozenset({Phase.DONE, Phase.RECOVERY, Phase.STOPPED}),
    Phase.RECOVERY: frozenset({
        Phase.SCAN_SCENE,
        Phase.APPROACH,
        Phase.CLOSE_REOBSERVE,
        Phase.PRE_ALIGN,
        Phase.PLAN_TRANSPORT,
        Phase.REOBSERVE_DESTINATION,
        Phase.PRE_PLACE,
        Phase.VERIFY_RESULT,
        Phase.STOPPED,
    }),
    Phase.DONE: frozenset(),
    Phase.STOPPED: frozenset(),
}


@dataclass
class ManipulationStateMachine:
    phase: Phase = Phase.IDLE
    events: list[PhaseEvent] = field(default_factory=list)
    on_event: Callable[[PhaseEvent], None] | None = field(
        default=None, repr=False, compare=False
    )

    def _emit(self, event: PhaseEvent) -> None:
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def transition(self, next_phase: Phase, verdict: str = "ok", **evidence: object) -> None:
        if next_phase not in _ALLOWED[self.phase]:
            raise ValueError(f"illegal V2 transition: {self.phase.value} -> {next_phase.value}")
        self.phase = next_phase
        self._emit(PhaseEvent(next_phase, verdict, dict(evidence)))

    def record(self, verdict: str = "evidence", **evidence: object) -> None:
        """Attach fresh evidence to the current phase without a fake transition."""
        self._emit(PhaseEvent(self.phase, verdict, dict(evidence)))

    def recover_to(self, next_phase: Phase, reason: str, **evidence: object) -> None:
        if self.phase != Phase.RECOVERY:
            self.transition(Phase.RECOVERY, "recover", reason=reason)
        self.transition(next_phase, "retry", reason=reason, **evidence)
