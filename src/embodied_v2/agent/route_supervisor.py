"""High-level route review that can reject but never override safety."""

from __future__ import annotations

import json

from ..domain import RouteIntent, SafetyReport, SafetyVerdict
from .manager_client import ManagerClient


class RouteSupervisor:
    def __init__(self, client: ManagerClient | None = None, use_online: bool = False):
        self.client = client or ManagerClient()
        self.use_online = bool(use_online)

    def review(self, intent: RouteIntent, report: SafetyReport) -> str:
        if report.verdict is SafetyVerdict.FAIL:
            return "SAFETY_REJECT"
        minimum = min(report.minimum_robot_clearance_m, report.minimum_payload_clearance_m)
        if self.use_online and self.client.available:
            try:
                result = self.client.json_completion(
                    "Review a symbolic route summary. Return JSON decision equal "
                    "to ACCEPT or REQUEST_ALTERNATIVE. Safety PASS is already a "
                    "local prerequisite and you cannot override safety.",
                    json.dumps({
                        "corridor": intent.corridor,
                        "high_clearance": intent.high_clearance,
                        "approach_side": intent.approach_side,
                        "minimum_clearance_class": (
                            "small" if minimum < 0.02 else "comfortable"
                        ),
                    }),
                )
                decision = str(result.get("decision", ""))
                if decision in {"ACCEPT", "REQUEST_ALTERNATIVE"}:
                    return decision
            except (KeyError, TypeError, ValueError, RuntimeError, OSError):
                pass
        # The local report already includes robot collision, payload collision,
        # and the measured dynamic envelope at every execution sample. An
        # extra name/clearance heuristic can trade a short stable route for a
        # long unstable one, so offline review accepts the first hard PASS.
        return "ACCEPT"
