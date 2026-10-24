"""High-level route proposals based only on the semantic GAP map."""

from __future__ import annotations

import json

from ..domain import PayloadState, RouteIntent, SceneModel, TaskSemantics, Vec3
from .manager_client import ManagerClient

POSTURES = ("elbow_up", "elbow_left", "elbow_right")


class RouteStrategist:
    def __init__(self, client: ManagerClient | None = None, use_online: bool = False):
        self.client = client or ManagerClient()
        self.use_online = bool(use_online)

    def propose(
        self,
        task: TaskSemantics,
        scene: SceneModel,
        payload: PayloadState,
        start_m: Vec3,
        destination_m: Vec3,
    ) -> tuple[RouteIntent, ...]:
        allowed_corridors = set(scene.gap_map.suggested_corridors) | {"rear"}
        if self.use_online and self.client.available:
            try:
                response = self.client.json_completion(
                    "Propose 2-4 symbolic transport routes as JSON {routes:[...]}. "
                    "Each route may contain only corridor, high_clearance, "
                    "approach_side, avoid_regions, rationale. Do not emit any "
                    "coordinate, distance, waypoint, joint angle, or motion command.",
                    json.dumps({
                        "task": {
                            "relation": task.relation.value,
                            "fragile": task.fragile,
                        },
                        "gap_regions": [
                            {
                                "name": item.name,
                                "description": item.description,
                                "clearance": item.clearance,
                            }
                            for item in scene.gap_map.regions
                        ],
                        "allowed_corridors": sorted(allowed_corridors),
                        "payload_size_class": (
                            "large" if max(payload.geometry.size) > 0.12 else "compact"
                        ),
                    }),
                )
                online: list[RouteIntent] = []
                for item in response.get("routes", []):
                    corridor = str(item.get("corridor", ""))
                    if corridor not in allowed_corridors:
                        continue
                    online.append(RouteIntent(
                        corridor,
                        bool(item.get("high_clearance", False)),
                        str(item.get("approach_side", "front")),
                        tuple(
                            value for value in item.get("avoid_regions", [])
                            if value in allowed_corridors
                        ),
                        str(item.get("rationale", "online symbolic manager")),
                    ))
                if online:
                    return tuple(online)
            except (KeyError, TypeError, ValueError, RuntimeError, OSError):
                pass
        # Offline is deliberately first-class. The Manager selects symbolic
        # regions only; metric conversion remains inside CorridorPlanner.
        intents = [
            RouteIntent(
                corridor=name,
                high_clearance=(name in scene.gap_map.blocked_regions or task.fragile),
                approach_side="outer" if "outer" in name else "front",
                avoid_regions=scene.gap_map.blocked_regions,
                rationale="semantic gap-map ordering",
            )
            for name in scene.gap_map.suggested_corridors
        ]
        if all("rear" not in item.corridor for item in intents):
            intents.append(RouteIntent(
                "rear", True, "rear", scene.gap_map.blocked_regions,
                "fallback alternative corridor",
            ))
        return tuple(intents)

    def propose_reconfiguration(
        self,
        task: TaskSemantics,
        scene: SceneModel,
        payload: PayloadState,
        rejected_corridors: tuple[str, ...],
    ) -> tuple[RouteIntent, ...]:
        """Ask for a different arm shape after the object is already lifted.

        The model may name a posture and a corridor. A colliding result is
        still rejected by the local checker; this method never returns a
        joint angle or a distance.
        """
        allowed_corridors = set(scene.gap_map.suggested_corridors) | {"rear"}
        if self.client.available:
            try:
                response = self.client.json_completion(
                    "The object is already lifted and the previous transport "
                    "corridors collide. Propose 2 or 3 alternative arm postures "
                    "as JSON {routes:[...]}. Each route may contain only posture, "
                    "corridor, high_clearance, and rationale. posture must be one "
                    "of elbow_up, elbow_left, elbow_right. Do not emit coordinates, "
                    "distances, waypoints, joint angles, or motion commands. You "
                    "cannot approve a path the local collision check rejects.",
                    json.dumps({
                        "task": {"relation": task.relation.value, "fragile": task.fragile},
                        "rejected_corridors": list(rejected_corridors),
                        "allowed_corridors": sorted(allowed_corridors),
                        "allowed_postures": list(POSTURES),
                        "gap_regions": [
                            {
                                "name": item.name,
                                "description": item.description,
                                "clearance": item.clearance,
                            }
                            for item in scene.gap_map.regions
                        ],
                        "payload_size_class": (
                            "large" if max(payload.geometry.size) > 0.12 else "compact"
                        ),
                    }),
                )
                online: list[RouteIntent] = []
                for item in response.get("routes", []):
                    posture = str(item.get("posture", ""))
                    corridor = str(item.get("corridor", ""))
                    if posture not in POSTURES or corridor not in allowed_corridors:
                        continue
                    online.append(RouteIntent(
                        corridor,
                        True,
                        "outer" if "outer" in corridor else "front",
                        tuple(
                            value for value in item.get("avoid_regions", [])
                            if value in allowed_corridors
                        ),
                        str(item.get("rationale", "lifted posture change")),
                        posture,
                    ))
                if online:
                    return tuple(online)
            except (KeyError, TypeError, ValueError, RuntimeError, OSError):
                pass
        open_corridors = [
            name for name in (*scene.gap_map.suggested_corridors, "rear")
            if name not in rejected_corridors
        ] or ["rear"]
        return tuple(
            RouteIntent(
                corridor=open_corridors[index % len(open_corridors)],
                high_clearance=True,
                approach_side="outer",
                avoid_regions=scene.gap_map.blocked_regions,
                rationale="lifted arm posture change",
                posture=posture,
            )
            for index, posture in enumerate(POSTURES)
        )
