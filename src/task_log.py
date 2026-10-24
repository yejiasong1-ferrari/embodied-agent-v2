"""Structured per-task evidence log.

One JSON file per task run, with seven fixed sections.  Everything the
execution layer measured or decided ends up here, so a failed run can be
audited offline without re-running the simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
import re
import time
from typing import Any

from embodied_agent import _json_default

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
SECTIONS = (
    "instruction", "strategy", "plan_briefs", "checkpoints",
    "failures", "recovery", "delivery",
)


def _plain(value: Any) -> Any:
    """Dataclasses and numpy become plain JSON-able Python."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def _log_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return _json_default(value)


def task_short_code(task: Any) -> str:
    """``red_cup_to_yellow_plate`` style tag used in evidence file names."""
    if task is None:
        return "task"
    parts = [str(getattr(task, "object_color", "") or ""), str(getattr(task, "object_name", "") or "")]
    reference = getattr(task, "reference", None)
    if reference:
        parts += ["to", str(getattr(task, "reference_color", "") or ""), str(reference)]
    code = "_".join(part for part in parts if part)
    code = re.sub(r"[^A-Za-z0-9_]+", "_", code).strip("_").lower()
    return code or "task"


@dataclass
class TaskLog:
    instruction: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    plan_briefs: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    recovery: list[dict[str, Any]] = field(default_factory=list)
    delivery: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    short_code: str = "task"

    # --- construction -----------------------------------------------------
    @classmethod
    def start(cls, instruction: str, task: Any = None, planner_backend: str = "") -> "TaskLog":
        log = cls(short_code=task_short_code(task))
        log.instruction = {
            "text": instruction,
            "planner_backend": planner_backend,
            "task": None if task is None else _plain(task.to_dict() if hasattr(task, "to_dict") else task),
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(log.started_at)),
        }
        return log

    def _stamp(self) -> float:
        return round(time.time() - self.started_at, 3)

    # --- sections ---------------------------------------------------------
    def set_strategy(self, strategy: Any, **extra: Any) -> None:
        payload = dict(_plain(strategy)) if isinstance(_plain(strategy), dict) else {"value": _plain(strategy)}
        payload.update(extra)
        self.strategy = payload

    def add_plan_brief(self, brief: Any, verdict: Any = None) -> None:
        self.plan_briefs.append({
            "t": self._stamp(),
            "brief": _plain(brief),
            "verdict": _plain(verdict),
        })

    def add_checkpoint(self, stage: str, evidence: dict[str, Any], verdict: Any = None) -> None:
        self.checkpoints.append({
            "t": self._stamp(),
            "stage": stage,
            "evidence": evidence,
            "verdict": _plain(verdict),
        })

    def add_failure(self, reason: str, diagnosis: Any = None, **extra: Any) -> None:
        entry: dict[str, Any] = {"t": self._stamp(), "reason": reason, "diagnosis": _plain(diagnosis)}
        entry.update(extra)
        self.failures.append(entry)

    def add_recovery_round(self, entry: dict[str, Any]) -> None:
        self.recovery.append({"t": self._stamp(), **entry})

    def set_delivery(self, **fields: Any) -> None:
        self.delivery.update({key: _plain(value) for key, value in fields.items()})

    # --- output -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {section: getattr(self, section) for section in SECTIONS}

    def write(self, output_dir: Path | None = None) -> Path:
        directory = output_dir or OUTPUT_DIR
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.started_at))
        path = directory / f"tasklog_{stamp}_{self.short_code}.json"
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=_log_default),
            encoding="utf-8",
        )
        return path


def grasp_case_path(output_dir: Path | None = None) -> Path:
    return (output_dir or OUTPUT_DIR) / "grasp_cases.json"


def load_grasp_cases(output_dir: Path | None = None) -> list[dict[str, Any]]:
    path = grasp_case_path(output_dir)
    if not path.exists():
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


def append_grasp_case(case: dict[str, Any], output_dir: Path | None = None) -> None:
    """Add one success case. Existing case fields are not rewritten."""
    directory = output_dir or OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    cases = load_grasp_cases(directory)
    cases.append(case)
    grasp_case_path(directory).write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8",
    )
