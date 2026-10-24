"""Persistent scene inventory shared by perception and language grounding."""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading

from ..domain import SceneEntity, SceneModel


@dataclass(frozen=True)
class RememberedEntity:
    track_id: str
    color: str
    category: str
    movable: bool
    support_candidate: bool
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    confidence: float
    last_seen_s: float


class SceneMemory:
    """Latest camera-derived object tracks, retained across UI commands."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entities: dict[str, RememberedEntity] = {}
        self._revision = 0

    def update(self, scene: SceneModel) -> tuple[RememberedEntity, ...]:
        observed = {
            entity.entity_id: self._remember(entity, scene.timestamp_s)
            for entity in scene.entities
        }
        with self._lock:
            # Keep tracks that are temporarily occluded. A later observation
            # replaces the old pose under the same stable track id.
            self._entities.update(observed)
            self._revision += 1
            return tuple(self._entities.values())

    @staticmethod
    def _remember(entity: SceneEntity, timestamp_s: float) -> RememberedEntity:
        size = entity.bounds.size
        category = entity.geometry
        if category == "cylinder":
            category = "bottle" if size[2] > 2.2 * max(size[0], size[1]) else "cup"
        return RememberedEntity(
            track_id=entity.entity_id,
            color=entity.color,
            category=category,
            movable=entity.movable,
            support_candidate=entity.support_candidate,
            center_m=entity.center_m,
            size_m=entity.bounds.size,
            confidence=entity.confidence,
            last_seen_s=float(timestamp_s),
        )

    def entities(self) -> tuple[RememberedEntity, ...]:
        with self._lock:
            return tuple(self._entities.values())

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def prompt_context(self) -> str:
        """Compact symbolic inventory supplied to the optional LLM grounder."""
        rows = [
            {
                "track_id": item.track_id,
                "color": item.color,
                "category": item.category,
                "movable": item.movable,
                "support": item.support_candidate,
                "size_m": [round(value, 4) for value in item.size_m],
            }
            for item in self.entities()
        ]
        return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))

    def event_inventory(self) -> tuple[dict[str, object], ...]:
        """Human-readable identifiers for the native evidence stream."""
        return tuple(
            {
                "id": item.track_id,
                "label": f"{item.color} {item.category}",
                "role": "movable" if item.movable else (
                    "support" if item.support_candidate else "obstacle"
                ),
            }
            for item in self.entities()
        )
