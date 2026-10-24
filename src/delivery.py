"""Delivery photo and hand-off stub.

Every task run, successful or not, ends with one annotated external-camera
image so a human can see what happened without reading logs.  ``deliver`` is
intentionally a stub this round: no external channel is wired yet.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from camera_demo import render_result_overview
from task_log import OUTPUT_DIR, task_short_code


_CJK_FONT_CANDIDATES = (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def _font(size: int = 14) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in _CJK_FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_lines(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = _font(14)
    pad, line_h = 8, 18
    box_h = pad * 2 + line_h * len(lines)
    draw.rectangle((0, 0, canvas.width, box_h), fill=(0, 0, 0, 160))
    for index, line in enumerate(lines):
        draw.text((pad, pad + index * line_h), line, fill=(245, 245, 245, 255), font=font)
    return np.asarray(canvas)


def render_delivery(
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    task: Any,
    outcome: Any,
) -> np.ndarray:
    """External-camera proof of the final scene, annotated with the verdict."""
    if outcome.support_xy_m is not None:
        look_xy = np.asarray(outcome.support_xy_m, dtype=float)
    elif outcome.final_object_xyz_m is not None:
        look_xy = np.asarray(outcome.final_object_xyz_m[:2], dtype=float)
    else:
        look_xy = np.array([0.35, 0.0])
    frame = render_result_overview(model, renderer, data, look_xy)
    verdict = "PASS" if outcome.success else ("STOPPED" if outcome.stopped else "FAIL")
    lines = [
        f"{outcome.instruction}",
        f"{verdict}  reason={outcome.reason}",
    ]
    if outcome.xy_error_m is not None:
        if getattr(outcome, "error_vs_locked_reference_m", None) is not None:
            lines.append(f"xy vs locked {outcome.xy_error_m * 1000:.0f} mm")
            used = outcome.error_vs_used_support_m
            if used is not None:
                lines.append(f"xy vs used support {used * 1000:.0f} mm")
        else:
            lines.append(f"xy error {outcome.xy_error_m * 1000:.0f} mm")
    if outcome.final_object_bottom_m is not None:
        support = outcome.support_top_m
        support_text = "" if support is None else f" vs support {support:.3f} m"
        lines.append(f"bottom {outcome.final_object_bottom_m:.3f} m{support_text}")
    lines.append(
        f"{outcome.elapsed_s:.1f} s   recovery rounds {outcome.recovery_rounds}"
        f"   route {outcome.route_policy}"
    )
    if task is not None and getattr(outcome, "pick_strategy", None) is not None:
        strategy = outcome.pick_strategy
        lines.append(
            f"strategy {strategy.approach}/{strategy.grip_width}/{strategy.lift_height}/"
            f"{strategy.transport}/{strategy.speed_profile} ({strategy.backend})"
        )
    return _draw_lines(frame, lines)


def save_delivery(image: np.ndarray, task: Any, output_dir: Path | None = None) -> Path:
    directory = output_dir or OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = directory / f"delivery_{stamp}_{task_short_code(task)}.png"
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)
    return path


def copy_delivery(path: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
    return destination


def deliver(image_path: Path | str | None, tasklog_path: Path | str | None, summary: dict[str, Any]) -> None:
    """Hand-off stub.  No external channel this round; one line of evidence."""
    print(
        "deliver() stub: "
        + json.dumps({
            "image": None if image_path is None else str(image_path),
            "tasklog": None if tasklog_path is None else str(tasklog_path),
            "success": summary.get("success"),
            "reason": summary.get("reason"),
        }, ensure_ascii=False)
    )
