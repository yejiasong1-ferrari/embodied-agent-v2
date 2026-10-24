"""Version 2 embodied manipulation system.

This package is intentionally independent from the V1 orchestration code.
Only low-level MuJoCo/Panda infrastructure is adapted at the boundary.
"""

from .domain import ExecutionResult, Phase, TaskSemantics
from .state_machine import ManipulationStateMachine

__all__ = [
    "ExecutionResult",
    "ManipulationStateMachine",
    "Phase",
    "TaskSemantics",
]
