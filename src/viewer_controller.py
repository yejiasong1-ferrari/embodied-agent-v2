"""Native one-window MuJoCo controller with editable commands and copyable logs.

Run on macOS with:
    .venv/bin/python src/viewer_controller.py
"""

from __future__ import annotations

import argparse
import atexit
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import traceback

import mujoco
import numpy as np
import PySide6

# On some macOS installations, downloaded wheel directories carry provenance
# metadata that makes Qt's directory scanner skip the Cocoa platform plugin.
# Loading the exact same signed plugin from a clean temporary directory works.
# This local, automatically-cleaned copy avoids changing any system security
# setting and must be configured before QApplication is constructed.
_qt_runtime_dir = Path(tempfile.mkdtemp(prefix="panda-qt-"))
_qt_platform_dir = _qt_runtime_dir / "platforms"
_qt_platform_dir.mkdir()
_qt_cocoa_source = Path(PySide6.__file__).parent / "Qt" / "plugins" / "platforms" / "libqcocoa.dylib"
shutil.copyfile(_qt_cocoa_source, _qt_platform_dir / "libqcocoa.dylib")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_qt_platform_dir)
atexit.register(shutil.rmtree, _qt_runtime_dir, ignore_errors=True)

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QPushButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)
from PIL import Image

from camera_demo import (
    CAMERA_NAME, IMAGE_HEIGHT, IMAGE_WIDTH, _note_obstacles,
    apply_support_point_snap, reference_point_within_color_gate,
    render_rgbd, reset_scene,
)
from embodied_agent import LLMRobotLeader, LeaderUnavailableError
from task_planner import NotFoundReport
from joint_control import LEFT_ABOVE, PandaController, create_model
from task_planner import TaskSpec
from task_runner import run_task
from scene_layout import OBJECT_BEARING_DEG
from vision_detector import (
    Detection, detect_colored_object, detection_from_image_point, workspace_ceiling_m,
)

# Evidence frames (failed VLM pointing, task logs, delivery photos) live here.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


# One monotonic sweep. The old implementation swept 13 target angles and
# repeated all 13 for the reference; now one RGB-D frame serves both queries.
SCAN_ONCE_ANGLES_DEG = (
    135.0, 112.5, 90.0, 67.5, 45.0, 22.5, 0.0,
    -22.5, -45.0, -67.5, -90.0, -112.5, -135.0,
)


def _scan_schedule(task: TaskSpec) -> tuple[float, ...]:
    """Look near likely bearings first, then fall back to the full sweep."""
    priorities = [OBJECT_BEARING_DEG.get(task.object_color, 0.0)]
    if task.reference:
        priorities.append(OBJECT_BEARING_DEG.get(task.reference_color or "yellow", 45.0))
    ordered: list[float] = []
    for center in priorities:
        for offset in (0.0, -22.5, 22.5):
            angle = float(np.clip(center + offset, -135.0, 135.0))
            if angle not in ordered:
                ordered.append(angle)
    ordered.extend(angle for angle in SCAN_ONCE_ANGLES_DEG if angle not in ordered)
    return tuple(ordered)


class InteractiveSceneLabel(QLabel):
    """MuJoCo-like orbit camera: drag to rotate, wheel to zoom."""

    def __init__(self, camera: mujoco.MjvCamera, reset_camera) -> None:
        super().__init__()
        self.camera = camera
        self.reset_camera = reset_camera
        self.last_position = None
        self.drag_button = Qt.MouseButton.NoButton
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setMouseTracking(True)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.last_position = event.position()
        self.drag_button = event.button()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.last_position = None
        self.drag_button = Qt.MouseButton.NoButton
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.last_position is None:
            return
        delta = event.position() - self.last_position
        self.last_position = event.position()
        if self.drag_button == Qt.MouseButton.LeftButton:
            self.camera.azimuth -= float(delta.x()) * 0.45
            self.camera.elevation = float(np.clip(
                self.camera.elevation + float(delta.y()) * 0.35, -89.0, 35.0
            ))
        elif self.drag_button in {Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton}:
            azimuth = np.deg2rad(float(self.camera.azimuth))
            scale = max(0.0005, float(self.camera.distance) / max(self.width(), 1))
            right = np.array([np.cos(azimuth), np.sin(azimuth)])
            forward = np.array([-np.sin(azimuth), np.cos(azimuth)])
            self.camera.lookat[:2] += (-float(delta.x()) * right + float(delta.y()) * forward) * scale
        event.accept()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        factor = float(np.exp(-event.angleDelta().y() * 0.0012))
        self.camera.distance = float(np.clip(self.camera.distance * factor, 0.45, 4.5))
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.reset_camera()
        event.accept()


class _LiveViewerAdapter:
    """Give PandaController real-time pacing without a second window."""

    def __init__(self, stop_event: threading.Event, closed_event: threading.Event):
        self.stop_event = stop_event
        self.closed_event = closed_event

    def is_running(self) -> bool:
        return not self.stop_event.is_set() and not self.closed_event.is_set()

    def sync(self) -> None:
        pass


class _LogStream(io.TextIOBase):
    """Forward print() lines from the worker into the Qt log box."""

    def __init__(self, window: "ViewerController") -> None:
        self.window = window
        self.pending = ""

    def write(self, value: str) -> int:
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.strip():
                self.window.log_signal.emit(line.strip())
        return len(value)

    def flush(self) -> None:
        if self.pending.strip():
            self.window.log_signal.emit(self.pending.strip())
        self.pending = ""


def _center_error(detection: Detection) -> float:
    u, v = detection.center_uv
    return float(np.hypot(u - IMAGE_WIDTH / 2, v - IMAGE_HEIGHT / 2))


def _priority_view_angles(task: TaskSpec) -> tuple[float, ...]:
    """Two wide views: one toward the named object, one toward the destination."""
    angles = [float(OBJECT_BEARING_DEG.get(task.object_color, 0.0))]
    if task.reference:
        destination = float(OBJECT_BEARING_DEG.get(task.reference_color or "yellow", 45.0))
        if abs(destination - angles[0]) > 12.0:
            angles.append(destination)
    return tuple(angles)


def _snapshot_wide_view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    angle_deg: float,
) -> dict[str, object]:
    """RGB-D plus the camera pose, so the point can be grounded after the arm moves."""
    rgb, depth = render_rgbd(renderer, data)
    camera_id = model.camera(CAMERA_NAME).id
    base_id = model.body("link0").id
    return {
        "angle_deg": float(angle_deg),
        "rgb": rgb.copy(),
        "depth": depth.copy(),
        "cam_xpos": np.array(data.cam_xpos[camera_id], dtype=float, copy=True),
        "cam_xmat": np.array(data.cam_xmat[camera_id], dtype=float, copy=True),
        "fovy_deg": float(model.cam_fovy[camera_id]),
        "base_xpos": np.array(data.xpos[base_id], dtype=float, copy=True),
        "base_xmat": np.array(data.xmat[base_id], dtype=float, copy=True),
    }


def _ground_pointing(
    view: dict[str, object],
    u_norm: float,
    v_norm: float,
    label: str,
    model: mujoco.MjModel,
) -> Detection | None:
    pose = (
        view["cam_xpos"], view["cam_xmat"], view["fovy_deg"],
        view["base_xpos"], view["base_xmat"],
    )
    return detection_from_image_point(
        view["rgb"], view["depth"], u_norm, v_norm, label, model,
        camera_name=CAMERA_NAME, camera_pose=pose,  # type: ignore[arg-type]
    )


def _scan_by_color(
    task: TaskSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    robot: PandaController,
    ceiling: list[float] | None = None,
    obstacles: list[dict] | None = None,
    captures: dict[str, np.ndarray] | None = None,
) -> tuple[Detection | None, Detection | None, float | None]:
    """Offline fallback: the largest same-color region is the object."""
    target_best: tuple[float, Detection, float] | None = None
    reference_best: tuple[float, Detection, float] | None = None

    for angle_deg in _scan_schedule(task):
        print(f"→ 主动观察：base {angle_deg:+.0f}°")
        pose = LEFT_ABOVE.copy()
        pose[0] = np.deg2rad(angle_deg)
        if not robot.move_to(pose, duration=0.22):
            break

        rgb, depth = render_rgbd(renderer, data)
        if captures is not None:
            captures["latest"] = rgb.copy()
        if ceiling is not None:
            ceiling[0] = max(
                ceiling[0], workspace_ceiling_m(model, data, CAMERA_NAME, depth)
            )
        _note_obstacles(obstacles, model, data, depth)
        target = detect_colored_object(
            rgb, depth, task.object_color, task.object_name,
            model, data, CAMERA_NAME,
        )
        if target is not None:
            error = _center_error(target)
            if target_best is None or error < target_best[0]:
                target_best = (error, target, angle_deg)
                if captures is not None:
                    captures["target"] = rgb.copy()
                print(f"  ✓ 锁定拿取物 {target.label}，中心误差 {error:.0f}px")

        if task.reference:
            reference = detect_colored_object(
                rgb, depth, task.reference_color or "yellow", task.reference,
                model, data, CAMERA_NAME,
            )
            if reference is not None and target_best is not None:
                same_place = float(np.linalg.norm(
                    reference.base_xyz_m[:2] - target_best[1].base_xyz_m[:2]
                )) < 0.12
                if same_place:
                    # Orange and yellow ranges can overlap, so a glancing view
                    # sometimes labels the cube itself as the plate.
                    reference = None
            if reference is not None:
                error = _center_error(reference)
                if reference_best is None or error < reference_best[0]:
                    apply_support_point_snap(
                        reference, rgb, depth, task.reference_color or "yellow",
                        model, data,
                    )
                    reference_best = (error, reference, angle_deg)
                    if captures is not None:
                        captures["destination"] = rgb.copy()
                    print(f"  ✓ 锁定放置物 {reference.label}，中心误差 {error:.0f}px")

        # Centering happens after the fast move to directly above the object.
        # The first clear sighting of every named object is enough to leave.
        target_clear = target_best is not None
        reference_clear = not task.reference or reference_best is not None
        if target_clear and reference_clear:
            marked = 0 if obstacles is None else len(obstacles)
            print(f"✓ 指令中的物体已锁定，停止扫描。已标注 {marked} 个高障碍")
            break

    return (
        target_best[1] if target_best else None,
        reference_best[1] if reference_best else None,
        target_best[2] if target_best else None,
    )


def scan_scene_once(
    task: TaskSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    robot: PandaController,
    ceiling: list[float] | None = None,
    obstacles: list[dict] | None = None,
    captures: dict[str, np.ndarray] | None = None,
    leader: LLMRobotLeader | None = None,
    instruction: str | None = None,
) -> tuple[Detection | None, Detection | None, float | None]:
    """Confirm the object in a wide photo, then convert that pixel with depth.

    Color blobs are used only when the vision-language model is offline.
    """
    if leader is None or not leader.online or not instruction:
        return _scan_by_color(
            task, model, data, renderer, robot,
            ceiling=ceiling, obstacles=obstacles, captures=captures,
        )

    views: list[dict[str, object]] = []
    for angle_deg in _priority_view_angles(task):
        print(f"→ 广角拍摄：base {angle_deg:+.0f}°")
        pose = LEFT_ABOVE.copy()
        pose[0] = np.deg2rad(angle_deg)
        if not robot.move_to(pose, duration=0.22):
            break
        view = _snapshot_wide_view(model, data, renderer, angle_deg)
        views.append(view)
        if captures is not None:
            captures["latest"] = view["rgb"]  # type: ignore[assignment]
        if ceiling is not None:
            ceiling[0] = max(
                ceiling[0],
                workspace_ceiling_m(model, data, CAMERA_NAME, view["depth"]),  # type: ignore[arg-type]
            )
        _note_obstacles(obstacles, model, data, view["depth"])  # type: ignore[arg-type]
    if not views:
        return None, None, None

    print("→ 把广角照片交给 LLM，先确认要拿的物体，再用深度换算坐标")

    def color_fallback(reason: str):
        """Only the pixel choice failed; the RGB-D color path is still safe."""
        print(f"  定位兜底=color_rgbd（{reason}）")
        _save_locate_failure(views, reason)
        target, reference, angle = _scan_by_color(
            task, model, data, renderer, robot,
            ceiling=ceiling, obstacles=obstacles, captures=captures,
        )
        for found in (target, reference):
            if found is not None:
                found.source = "color_fallback"
        return target, reference, angle

    try:
        pointing = leader.locate_in_views(instruction, task, views)
    except LeaderUnavailableError:
        raise
    except Exception as error:
        return color_fallback(f"locate_in_views_error:{type(error).__name__}:{error}")

    target_view = views[pointing.target_view]
    target = _ground_pointing(
        target_view, pointing.target_u, pointing.target_v,
        f"{task.object_color} {task.object_name}", model,
    )
    print(
        f"  LLM 确认拿取物：画面 {pointing.target_view}，"
        f"像素 ({pointing.target_u:.3f}, {pointing.target_v:.3f})"
    )
    print("  理由:", pointing.reason)
    if target is None:
        print("  ✗ 该图像点没有可用深度")
        return color_fallback("target_point_no_depth")
    target.source = "llm_point"
    print("  深度反投影基座坐标:", np.round(target.base_xyz_m, 3), "m")
    if captures is not None:
        captures["target"] = target_view["rgb"]  # type: ignore[assignment]

    reference = None
    if task.reference is not None and pointing.destination_view is not None:
        destination_view = views[pointing.destination_view]
        reference = _ground_pointing(
            destination_view,
            float(pointing.destination_u),
            float(pointing.destination_v),
            f"{task.reference_color or 'yellow'} {task.reference}",
            model,
        )
        print(
            f"  LLM 确认放置物：画面 {pointing.destination_view}，"
            f"像素 ({float(pointing.destination_u):.3f}, {float(pointing.destination_v):.3f})"
        )
        if reference is None:
            print("  ✗ 放置点没有可用深度")
            return color_fallback("destination_point_no_depth")
        same_place = float(np.linalg.norm(
            reference.base_xyz_m[:2] - target.base_xyz_m[:2]
        )) < 0.12
        if same_place:
            print("  ✗ 放置点和拿取物落在同一位置，不采用")
            return color_fallback("destination_same_as_target")
        reference.source = "llm_point"
        # The arm is still parked on the destination bearing (the last view).
        # An LLM point outside the existing 6 cm gate of that colour detection
        # is not the support; local colour owns the coordinate.
        plate_view = views[-1]
        color_hit = detect_colored_object(
            plate_view["rgb"],  # type: ignore[arg-type]
            plate_view["depth"],  # type: ignore[arg-type]
            task.reference_color or "yellow",
            task.reference,
            model,
            data,
            CAMERA_NAME,
        )
        if color_hit is not None and not reference_point_within_color_gate(
            reference.base_xyz_m[:2], color_hit.base_xyz_m[:2],
        ):
            print(
                "  放置点距本地色块超过 6 cm，改用色块："
                f"{np.round(reference.base_xyz_m[:2], 3)} → "
                f"{np.round(color_hit.base_xyz_m[:2], 3)}"
            )
            reference = color_hit
            reference.source = "color_fallback"
            destination_view = plate_view
        apply_support_point_snap(
            reference,
            destination_view["rgb"],  # type: ignore[arg-type]
            destination_view["depth"],  # type: ignore[arg-type]
            task.reference_color or "yellow",
            model,
            camera_pose=(
                destination_view["cam_xpos"], destination_view["cam_xmat"],
                destination_view["fovy_deg"], destination_view["base_xpos"],
                destination_view["base_xmat"],
            ),
        )
        print("  放置物基座坐标:", np.round(reference.base_xyz_m, 3), "m")
        if captures is not None:
            captures["destination"] = destination_view["rgb"]  # type: ignore[assignment]
    elif task.reference is not None:
        print("  ✗ LLM 没有给出放置点")
        return color_fallback("destination_point_missing")
    marked = 0 if obstacles is None else len(obstacles)
    print(f"✓ 物体由广角画面确认（llm_point），已标注 {marked} 个高障碍")
    # Park where the color scan would have stopped: facing the pick target.
    # The wide sweep ends at the destination view, and starting the grasp from
    # that wrist branch gave one-sided finger contact on the cup; the color
    # path, which parks over the target, grasps reliably.
    target_angle = float(target_view["angle_deg"])  # type: ignore[arg-type]
    if abs(float(views[-1]["angle_deg"]) - target_angle) > 1.0:  # type: ignore[arg-type]
        print(f"→ 回到拿取物视角：base {target_angle:+.0f}°")
        pose = LEFT_ABOVE.copy()
        pose[0] = np.deg2rad(target_angle)
        robot.move_to(pose, duration=0.22)
    return target, reference, target_angle


def _save_locate_failure(
    views: list[dict[str, object]], reason: str, output_dir: Path | None = None
) -> Path | None:
    """Keep the wide frames the VLM saw so a bad pointing can be audited."""
    directory = output_dir or OUTPUT_DIR
    try:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        frames: list[str] = []
        for index, view in enumerate(views):
            path = directory / f"locate_fail_{stamp}_{index}.png"
            Image.fromarray(np.asarray(view["rgb"], dtype=np.uint8), mode="RGB").save(path)
            frames.append(path.name)
        record = directory / f"locate_fail_{stamp}.json"
        record.write_text(
            json.dumps({
                "time": stamp,
                "reason": reason,
                "frames": frames,
                "angles_deg": [float(view["angle_deg"]) for view in views],  # type: ignore[arg-type]
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  失败帧已存: {record}")
        return record
    except OSError as error:
        print(f"  失败帧保存失败: {error}")
        return None


class ViewerController(QMainWindow):
    """Qt main window containing live MuJoCo and real native controls."""

    log_signal = Signal(str)
    state_signal = Signal(str, str)

    def __init__(
        self,
        default_instruction: str = "把红色杯子拿到黄色盘子上",
    ) -> None:
        super().__init__()
        self.default_instruction = default_instruction
        self.model = create_model()
        self.data = mujoco.MjData(self.model)
        reset_scene(self.model, self.data)
        self.data_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.closed_event = threading.Event()
        self.worker: threading.Thread | None = None

        self.viewer_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer_camera)
        self.viewer_camera.lookat[:] = (0.0, 0.0, 0.30)
        self.viewer_camera.distance = 1.70
        self.viewer_camera.azimuth = 135
        self.viewer_camera.elevation = -27
        # The wrist camera stays 640×480 for detection.  The window view is a
        # separate 1920-wide buffer so the on-screen image is not an upscaled thumbnail.
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), 1920)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), 1350)
        self.model.vis.quality.offsamples = 8
        self.model.vis.quality.numslices = 48
        self.model.vis.quality.numstacks = 32
        self.view_width = 1920
        self.view_height = 1350
        self._view_data = mujoco.MjData(self.model)
        self.live_renderer = mujoco.Renderer(
            self.model, height=self.view_height, width=self.view_width
        )

        self._build_ui()
        self.log_signal.connect(self._append_log)
        self.state_signal.connect(self._set_state)
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._refresh_live_view)
        self.render_timer.start(33)
        self._refresh_live_view()

    def _reset_view_camera(self) -> None:
        self.viewer_camera.lookat[:] = (0.0, 0.0, 0.30)
        self.viewer_camera.distance = 1.70
        self.viewer_camera.azimuth = 135
        self.viewer_camera.elevation = -27

    def _build_ui(self) -> None:
        self.setWindowTitle("Panda")
        self.resize(1560, 940)
        self.setMinimumSize(1180, 720)
        QApplication.instance().setFont(QFont("PingFang SC", 13))

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(16)

        scene_card = QFrame()
        scene_card.setObjectName("card")
        scene_layout = QVBoxLayout(scene_card)
        scene_layout.setContentsMargins(16, 14, 16, 14)
        scene_layout.setSpacing(10)
        scene_header = QHBoxLayout()
        view_title = QLabel("场景")
        view_title.setObjectName("sectionTitle")
        resolution = QLabel(f"{self.view_width} × {self.view_height}")
        resolution.setObjectName("badge")
        scene_header.addWidget(view_title)
        scene_header.addStretch(1)
        scene_header.addWidget(resolution)
        scene_layout.addLayout(scene_header)
        self.scene_label = InteractiveSceneLabel(
            self.viewer_camera, self._reset_view_camera
        )
        self.scene_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scene_label.setMinimumSize(860, 540)
        self.scene_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.scene_label.setObjectName("scene")
        scene_layout.addWidget(self.scene_label, 1)
        view_help = QLabel("拖动旋转    右键平移    滚轮缩放    双击复位")
        view_help.setObjectName("footer")
        scene_layout.addWidget(view_help)
        root_layout.addWidget(scene_card, 1)

        panel = QVBoxLayout()
        panel.setSpacing(8)
        eyebrow = QLabel("CONTROL")
        eyebrow.setObjectName("eyebrow")
        panel.addWidget(eyebrow)
        title = QLabel("Panda")
        title.setObjectName("title")
        panel.addWidget(title)
        subtitle = QLabel("输入自然语言指令，然后点击“运行”开始")
        subtitle.setObjectName("muted")
        panel.addWidget(subtitle)
        panel.addSpacing(8)

        self.instruction_input = QLineEdit(self.default_instruction)
        self.instruction_input.setPlaceholderText("输入新的机器人指令")
        self.instruction_input.setClearButtonEnabled(True)
        self.instruction_input.returnPressed.connect(self.start_task)
        panel.addWidget(self.instruction_input)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.run_button = QPushButton("运行")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.start_task)
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("danger")
        self.stop_button.clicked.connect(self.stop_task)
        buttons.addWidget(self.run_button, 2)
        buttons.addWidget(self.stop_button, 1)
        panel.addLayout(buttons)
        panel.addSpacing(6)

        status_row = QHBoxLayout()
        self.status_label = QLabel("●  READY")
        self.status_label.setObjectName("ready")
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        status_row.addWidget(self.status_label)
        status_row.addStretch(1)
        panel.addLayout(status_row)
        self.detail_label = QLabel("等待指令 · 当前不会自动运行")
        self.detail_label.setWordWrap(True)
        self.detail_label.setObjectName("detail")
        panel.addWidget(self.detail_label)
        panel.addSpacing(4)

        log_header = QHBoxLayout()
        log_title = QLabel("日志")
        log_title.setObjectName("sectionTitle")
        copy_button = QPushButton("复制")
        copy_button.setObjectName("small")
        copy_button.clicked.connect(self.copy_logs)
        log_header.addWidget(log_title, 1)
        log_header.addWidget(copy_button)
        panel.addLayout(log_header)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.log_box.setPlainText("就绪。输入指令后按回车，或点击运行。")
        panel.addWidget(self.log_box, 1)

        footer = QLabel("回车运行  ·  可选中日志后复制  ·  夹爪相机仍用于识别")
        footer.setObjectName("footer")
        footer.setWordWrap(True)
        panel.addWidget(footer)

        panel_widget = QFrame()
        panel_widget.setObjectName("card")
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(380)
        panel.setContentsMargins(18, 16, 18, 16)
        root_layout.addWidget(panel_widget)
        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0c0e12; color: #e7edf4; }
            QFrame#card {
                background: #14181f;
                border: 1px solid #2a3140;
                border-radius: 18px;
            }
            QLabel#scene {
                background: #090b0e;
                border: 1px solid #232a36;
                border-radius: 12px;
            }
            QLabel#eyebrow {
                color: #8ea2ff;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 1.4px;
            }
            QLabel#title { font-size: 28px; font-weight: 600; color: #f4f7fb; }
            QLabel#sectionTitle { color: #d5deea; font-size: 14px; font-weight: 600; }
            QLabel#muted { color: #8b97a8; font-size: 13px; }
            QLabel#badge {
                color: #c5d0de;
                background: #1c2430;
                border: 1px solid #334055;
                border-radius: 9px;
                padding: 3px 8px;
                font-size: 11px;
                font-family: Menlo;
            }
            QLabel#ready, QLabel#running, QLabel#done, QLabel#failed {
                border-radius: 10px;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#ready { color: #9ee0b8; background: #173226; border: 1px solid #2d6a49; }
            QLabel#running { color: #b9d0ff; background: #1a2740; border: 1px solid #3d5f99; }
            QLabel#done { color: #9ee0b8; background: #173226; border: 1px solid #2d6a49; }
            QLabel#failed { color: #ffb3c0; background: #3a2028; border: 1px solid #7a4452; }
            QLabel#detail { color: #c5d0de; padding: 6px 0 2px 0; font-size: 13px; }
            QLabel#footer { color: #6d7b8d; font-size: 12px; }
            QLineEdit {
                background: #0e1218;
                border: 1px solid #3a4558;
                border-radius: 10px;
                padding: 11px 12px;
                font-size: 15px;
                selection-background-color: #3d6fe0;
            }
            QLineEdit:focus { border: 1px solid #8ea2ff; }
            QPushButton { min-height: 40px; border-radius: 10px; font-weight: 600; font-size: 14px; }
            QPushButton#primary { background: #3d6fe0; color: white; border: none; }
            QPushButton#primary:hover { background: #5482ea; }
            QPushButton#primary:pressed { background: #2f5bc4; }
            QPushButton#danger {
                background: #24181c;
                color: #f0b7c2;
                border: 1px solid #6a424c;
            }
            QPushButton#danger:hover { background: #322026; }
            QPushButton#small {
                min-height: 28px;
                padding: 0 12px;
                background: transparent;
                color: #c5d0de;
                border: 1px solid #3a4558;
                border-radius: 8px;
                font-size: 12px;
            }
            QPushButton#small:hover { background: #1c2430; }
            QTextEdit {
                background: #0e1218;
                border: 1px solid #2a3140;
                border-radius: 12px;
                padding: 10px;
                color: #d5deea;
                font-family: Menlo;
                font-size: 12px;
                selection-background-color: #2c4f96;
            }
            """
        )

    def _render_frame(self) -> np.ndarray:
        # Copy the pose under the lock, then draw outside it.  A high-resolution
        # frame takes longer than one physics step and must not stall the arm.
        with self.data_lock:
            self._view_data.qpos[:] = self.data.qpos
            self._view_data.qvel[:] = self.data.qvel
            self._view_data.ctrl[:] = self.data.ctrl
            if self.model.nmocap:
                self._view_data.mocap_pos[:] = self.data.mocap_pos
                self._view_data.mocap_quat[:] = self.data.mocap_quat
        mujoco.mj_forward(self.model, self._view_data)
        self.live_renderer.update_scene(self._view_data, camera=self.viewer_camera)
        return self.live_renderer.render().copy()

    def _refresh_live_view(self) -> None:
        if self.closed_event.is_set() or self.scene_label.width() < 2:
            return
        rgb = self._render_frame()
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(image)
        ratio = max(1.0, float(self.scene_label.devicePixelRatioF()))
        target = self.scene_label.size() * ratio
        scaled = pixmap.scaled(
            target, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(ratio)
        self.scene_label.setPixmap(scaled)

    def _append_log(self, message: str) -> None:
        self.log_box.append(f"› {message}")
        bar = self.log_box.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _set_state(self, status: str, detail: str) -> None:
        self.status_label.setText(f"●  {status}")
        names = {
            "READY": "ready", "RUNNING": "running", "DONE": "done",
            "STOPPING": "running", "STOPPED": "ready", "FAILED": "failed",
        }
        self.status_label.setObjectName(names.get(status, "ready"))
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.detail_label.setText(detail)

    def copy_logs(self) -> None:
        selected = self.log_box.textCursor().selectedText()
        QApplication.clipboard().setText(selected if selected else self.log_box.toPlainText())
        self.detail_label.setText("日志已复制到剪贴板")

    def start_task(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.log_signal.emit("已有任务正在运行，请先点击停止")
            return
        instruction = self.instruction_input.text().strip()
        if not instruction:
            self.state_signal.emit("FAILED", "请先输入指令")
            self.instruction_input.setFocus()
            return
        self.log_box.clear()
        self.log_signal.emit(f"指令：{instruction}")
        self.stop_event.clear()
        self.state_signal.emit("RUNNING", "正在理解指令…")
        self.worker = threading.Thread(
            target=self._run_pipeline, args=(instruction,), daemon=True
        )
        self.worker.start()

    def stop_task(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.stop_event.set()
            self.state_signal.emit("STOPPING", "正在安全停止并保留当前画面…")
            self.log_signal.emit("收到停止请求")
        else:
            self.log_signal.emit("当前没有运行中的任务")

    def _run_pipeline(self, instruction: str) -> None:
        stream = _LogStream(self)
        renderer = None
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                leader = LLMRobotLeader()
                with self.data_lock:
                    reset_scene(self.model, self.data)
                renderer = mujoco.Renderer(
                    self.model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH
                )
                robot = PandaController(
                    self.model, self.data,
                    viewer=_LiveViewerAdapter(self.stop_event, self.closed_event),
                    speed=1.0, data_lock=self.data_lock,
                )
                inventory = None
                if leader.online:
                    from scene_perception import collect_open_inventory

                    print("→ 开放清点：每条指令重新扫描，不沿用上一条的物体清单")
                    inventory = collect_open_inventory(
                        robot, self.model, self.data, renderer, leader,
                    )
                    print(f"  清点 {len(inventory)} 条")
                try:
                    task, backend = leader.interpret_instruction(
                        instruction, inventory=inventory,
                    )
                except NotFoundReport as report:
                    from task_log import TaskLog

                    log = TaskLog.start(instruction, None, "not_found")
                    log.add_checkpoint("not_found", {
                        "message": report.message, **report.details,
                    }, "not_found")
                    print(report.message)
                    print("任务日志:", log.write())
                    self.state_signal.emit("FAILED", report.message)
                    return
                self.state_signal.emit(
                    "RUNNING",
                    f"规划：{task.action} {task.object_color} {task.object_name}\n任务理解：{backend}",
                )

                def scan(
                    current_task: TaskSpec,
                    ceiling: list[float],
                    obstacles: list[dict],
                    captures: dict[str, np.ndarray],
                ):
                    return scan_scene_once(
                        current_task, self.model, self.data, renderer, robot,
                        ceiling=ceiling, obstacles=obstacles, captures=captures,
                        leader=leader, instruction=instruction,
                    )

                outcome = run_task(
                    instruction, leader, self.model, self.data, renderer, robot,
                    scan=scan,
                    should_stop=self.stop_event.is_set,
                    status=lambda text: self.state_signal.emit("RUNNING", text),
                    task=task, planner_backend=backend,
                )
                proof = (
                    f"\n交差图 {Path(outcome.delivery_path).name}"
                    if outcome.delivery_path else ""
                )
                if outcome.stopped:
                    self.state_signal.emit("STOPPED", f"任务已停止，当前画面已保留{proof}")
                elif outcome.success:
                    angle = outcome.observation_angle_deg or 0.0
                    self.state_signal.emit(
                        "DONE",
                        f"任务完成 · 恢复 {outcome.recovery_rounds} 轮 · "
                        f"观察角度 {angle:+.1f}°\n"
                        f"水平偏差 {(outcome.xy_error_m or 0.0) * 1000:.0f} mm{proof}",
                    )
                    print("完成。可以修改输入框并运行下一条指令。")
                else:
                    self.state_signal.emit(
                        "FAILED",
                        f"动作未完成：{outcome.reason}\n恢复 {outcome.recovery_rounds} 轮{proof}",
                    )
                print("任务结果:", json.dumps(outcome.summary(), ensure_ascii=False))
                print("LLM token 用量:", json.dumps(leader.usage_summary()))
        except Exception as error:
            self.log_signal.emit(traceback.format_exc().splitlines()[-1])
            self.state_signal.emit("FAILED", str(error))
        finally:
            stream.flush()
            if renderer is not None:
                renderer.close()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.stop_event.set()
        self.closed_event.set()
        self.render_timer.stop()
        self.live_renderer.close()
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autorun", action="store_true", help="打开后自动执行默认指令")
    parser.add_argument(
        "--instruction",
        default="把红色杯子拿到黄色盘子上",
        help="窗口中预填的自然语言指令；配合 --autorun 可直接执行",
    )
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Panda")
    app.setStyle("Fusion")
    window = ViewerController(default_instruction=args.instruction)
    window.show()
    if args.autorun:
        QTimer.singleShot(300, window.start_task)
    app.exec()


if __name__ == "__main__":
    main()
