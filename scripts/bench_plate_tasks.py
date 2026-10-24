#!/usr/bin/env python3
"""Headless bench: place each movable object onto the yellow plate."""

from __future__ import annotations

import io
import os
import re
import sys
import time
from contextlib import redirect_stdout

import mujoco
import numpy as np

# repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from camera_demo import (  # noqa: E402
    _body_bottom_z,
    _body_xyz,
    _holding_check,
    reset_scene,
)
from embodied_agent import LLMRobotLeader  # noqa: E402
from joint_control import PandaController, create_model  # noqa: E402
from task_planner import plan_task  # noqa: E402
from task_runner import run_task  # noqa: E402
from viewer_controller import scan_scene_once  # noqa: E402

PLATE_XY = np.array([0.34, 0.30])
PLATE_TOP = 0.036
PLATE_R = 0.12

CASES = [
    ("把红色杯子拿到黄色盘子上", "red_cup"),
    ("把蓝色方块拿到黄色盘子上", "blue_box"),
    ("把绿色方块拿到黄色盘子上", "green_block"),
    ("把橙色方块拿到黄色盘子上", "cube"),
]

os.environ.pop("DASHSCOPE_API_KEY", None)


def count_patterns(log: str) -> dict[str, int]:
    keys = [
        "主动观察",
        "标注高障碍",
        "到不了目标",
        "绕开",
        "外圈",
        "转运中物体已滑落",
        "没有放到终点",
        "放置已核对",
        "一次夹紧",
        "阶段一完成",
        "第 2 次抓取",
        "第 3 次抓取",
        "抬高并绕开",
        "腕部却要转",
        "空转",
    ]
    return {k: len(re.findall(re.escape(k), log)) for k in keys}


def run_one(
    instruction: str,
    body: str,
    speed: float = 800.0,
    include_log: bool = False,
) -> dict:
    buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(buf):
        model = create_model()
        data = mujoco.MjData(model)
        reset_scene(model, data)
        robot = PandaController(model, data, speed=speed)
        renderer = mujoco.Renderer(model, height=480, width=640)
        # The bench is the offline regression path; .env must not re-enable
        # the online leader here.
        leader = LLMRobotLeader(mode="offline")
        task, backend = plan_task(instruction, use_llm=False)

        def scan(current_task, ceiling, obstacles, captures):
            return scan_scene_once(
                current_task, model, data, renderer, robot, ceiling, obstacles,
                captures=captures, leader=leader, instruction=instruction,
            )

        outcome = run_task(
            instruction, leader, model, data, renderer, robot,
            scan=scan, task=task, planner_backend=backend,
        )
        ok = outcome.success
        xyz = _body_xyz(model, data, body)
        bottom = _body_bottom_z(model, data, body)
    elapsed = time.perf_counter() - t0
    log = buf.getvalue()
    xy_err = float(np.linalg.norm(xyz[:2] - PLATE_XY))
    on_disk = xy_err <= PLATE_R + 0.03
    on_height = PLATE_TOP - 0.012 <= bottom <= PLATE_TOP + 0.04
    result = {
        "instruction": instruction,
        "body": body,
        "backend": backend,
        "ok": ok,
        "elapsed_s": round(elapsed, 2),
        "xyz": np.round(xyz, 4).tolist(),
        "bottom_m": round(bottom, 4),
        "xy_err_m": round(xy_err, 4),
        "on_plate_geom": on_disk and on_height,
        "obstacles_marked": outcome.obstacles_marked,
        "recovery_rounds": outcome.recovery_rounds,
        "outcome_reason": outcome.reason,
        "last_abort_reason": robot.last_abort_reason,
        "counts": count_patterns(log),
        "log_lines": len(log.splitlines()),
        "log_tail": "\n".join(log.splitlines()[-25:]),
    }
    if include_log:
        result["log"] = log
    return result


def run_white_mug_demo(speed: float = 800.0, include_log: bool = False) -> dict:
    """P2 registry demo: data-added white mug must localize and lift."""
    instruction = "拿起白色马克杯"
    buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(buf):
        model = create_model()
        data = mujoco.MjData(model)
        reset_scene(model, data)
        robot = PandaController(model, data, speed=speed)
        renderer = mujoco.Renderer(model, height=480, width=640)
        leader = LLMRobotLeader(mode="offline")
        task, backend = plan_task(instruction, use_llm=False)

        def scan(current_task, ceiling, obstacles, captures):
            return scan_scene_once(
                current_task, model, data, renderer, robot, ceiling, obstacles,
                captures=captures, leader=leader, instruction=instruction,
            )

        outcome = run_task(
            instruction, leader, model, data, renderer, robot,
            scan=scan, task=task, planner_backend=backend,
        )
        held_raw, holding_report = _holding_check(model, data, "white_mug")
        held = bool(held_raw)
        located = bool(outcome.target is not None)
    log = buf.getvalue()
    result = {
        "instruction": instruction,
        "backend": backend,
        "located": located,
        "held": held,
        "ok": bool(outcome.success and located and held),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "recovery_rounds": outcome.recovery_rounds,
        "outcome_reason": outcome.reason,
        "holding_report": holding_report,
        "log_lines": len(log.splitlines()),
        "log_tail": "\n".join(log.splitlines()[-25:]),
    }
    if include_log:
        result["log"] = log
    return result


def main() -> None:
    results = [run_one(inst, body) for inst, body in CASES]
    white = run_white_mug_demo()
    print("=" * 72)
    print("黄盘放置基准测试 (speed=800, 无界面)")
    print("=" * 72)
    for r in results:
        status = "PASS" if r["ok"] and r["on_plate_geom"] else ("FAIL" if not r["ok"] else "WARN")
        print(f"\n[{status}] {r['instruction']}")
        print(
            f"  耗时 {r['elapsed_s']}s | 返回 ok={r['ok']} | 几何在盘上={r['on_plate_geom']}"
            f" | 恢复轮次={r['recovery_rounds']} | 结论={r['outcome_reason']}"
        )
        print(f"  xyz={r['xyz']} bottom={r['bottom_m']} xy误差={r['xy_err_m']}m")
        print(f"  扫描标注障碍 {r['obstacles_marked']} 个 | 日志 {r['log_lines']} 行")
        hot = {k: v for k, v in r["counts"].items() if v > 0}
        print(f"  日志计数: {hot}")
    print("\n" + "=" * 72)
    fails = [r for r in results if not r["ok"] or not r["on_plate_geom"]]
    if fails:
        print("\n--- 失败/异常任务日志末尾 ---")
        for r in fails:
            print(f"\n### {r['instruction']}\n{r['log_tail']}")
    print("\n" + "=" * 72)
    print("P2 物体注册表数据新增演示")
    print("=" * 72)
    white_status = "PASS" if white["ok"] else "FAIL"
    print(
        f"[{white_status}] {white['instruction']} | 定位={white['located']} | "
        f"抓取保持={white['held']} | 耗时={white['elapsed_s']}s | "
        f"恢复轮次={white['recovery_rounds']}"
    )
    if not white["ok"]:
        print(white["log_tail"])


if __name__ == "__main__":
    main()
