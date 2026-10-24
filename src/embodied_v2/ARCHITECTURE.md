# Embodied Manipulation V2

V2 is a separate package. It does not import V1 orchestration, task recovery,
grasp policy, or transport business logic. The MuJoCo Panda model/controller
and environment/API configuration are boundary adapters only.

## Dependency graph

```text
agent/task_manager ───────────────┐
perception/scene_scanner ─────────┼─> embodied_agent (orchestrator)
perception/close_range ───────────┤
perception/distance_sensor ───────┤
grasp/grasp_geometry ─────────────┤
grasp/grasp_planner ──────────────┤
grasp/grasp_controller ───────────┤
payload/payload_state ────────────┤
agent/route_strategist ───────────┤
motion/{escape,ik,corridor} ──────┤
motion/{planner,trajectory} ──────┤
safety/{collision,trajectory} ────┤
safety/safety_monitor ────────────┤
placement/{preplace,descent} ─────┤
placement/support_transfer ───────┤
agent/recovery_manager ───────────┘

all modules -> domain + interfaces
MuJoCo adapter -> V1 low-level model/controller only
```

The safety validator is below both deterministic and LLM managers. A manager
may request an alternative but cannot convert a safety failure into a pass.

## Phases

1. Core types, protocols and state machine.
2. Wide RGB-D scene scan, 3D scene model and semantic GAP map.
3. Close re-observation, distance-sensor fusion and grasp geometry.
4. Contact-state controller, micro-lift verification and PayloadState.
5. Escape/multi-IK/corridor search with OMPL RRT-Connect fallback, generated
   execution trajectory and exact clone validation for robot, gripper,
   dynamic payload envelope and payload geometry.
6. Destination re-observation, pre-place search, controlled descent, load
   transfer and gradual release.
7. High-level route strategy and route supervision (LLM optional).
8. Failure-typed local recovery and complete orchestration.

Every phase has tests. The headless MuJoCo smoke test exercises the assembled
system without requiring an online model.
