"""Native PySide6 desktop studio for Embodied Manipulation V2."""

from __future__ import annotations

import argparse
import atexit
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import traceback

import mujoco
import numpy as np
import PySide6

# Keep the proven local Cocoa-plugin workaround isolated inside V2. It does
# not alter system security settings or the V1 application.
_qt_runtime_dir = Path(tempfile.mkdtemp(prefix="embodied-v2-qt-"))
_qt_platform_dir = _qt_runtime_dir / "platforms"
_qt_platform_dir.mkdir()
_qt_cocoa_source = (
    Path(PySide6.__file__).parent
    / "Qt" / "plugins" / "platforms" / "libqcocoa.dylib"
)
if _qt_cocoa_source.exists():
    shutil.copyfile(_qt_cocoa_source, _qt_platform_dir / "libqcocoa.dylib")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_qt_platform_dir)
atexit.register(shutil.rmtree, _qt_runtime_dir, ignore_errors=True)

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .adapters import MujocoRuntime, SimulationSnapshot
from .agent import SceneMemory
from .domain import PhaseEvent
from .embodied_agent import EmbodiedAgentV2


PHASES = (
    "understand_task", "scan_scene", "approach", "close_reobserve",
    "pre_align", "contact_search", "micro_lift", "stable_grasp",
    "plan_transport", "transport", "reobserve_destination", "pre_place",
    "descent", "first_contact", "load_transfer", "gradual_release",
    "verify_result", "done",
)
PHASE_LABELS = {
    "idle": "待机", "understand_task": "理解任务", "scan_scene": "扫描场景",
    "approach": "粗接近", "close_reobserve": "近距重观测",
    "pre_align": "抓取对准", "contact_search": "接触搜索",
    "micro_lift": "试抬验证", "stable_grasp": "稳定抓取",
    "plan_transport": "规划运输", "transport": "安全运输",
    "reobserve_destination": "目标重观测", "pre_place": "预放置",
    "descent": "受控下降", "first_contact": "首次接触",
    "load_transfer": "载荷转移", "gradual_release": "渐进释放",
    "verify_result": "最终验证", "recovery": "局部恢复",
    "done": "完成", "stopped": "已停止",
}


class InteractiveSceneLabel(QLabel):
    """Native orbit/pan/zoom interaction for the third-person scene."""

    def __init__(self, camera: mujoco.MjvCamera, reset_camera) -> None:
        super().__init__()
        self.camera = camera
        self.reset_camera = reset_camera
        self.last_position = None
        self.drag_button = Qt.MouseButton.NoButton
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setMouseTracking(True)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.last_position = event.position()
        self.drag_button = event.button()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self.last_position = None
        self.drag_button = Qt.MouseButton.NoButton
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.last_position is None:
            return
        delta = event.position() - self.last_position
        self.last_position = event.position()
        if self.drag_button == Qt.MouseButton.LeftButton:
            self.camera.azimuth -= float(delta.x()) * 0.40
            self.camera.elevation = float(np.clip(
                self.camera.elevation + float(delta.y()) * 0.32, -88.0, 25.0
            ))
        elif self.drag_button in {Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton}:
            angle = np.deg2rad(float(self.camera.azimuth))
            scale = max(0.0005, float(self.camera.distance) / max(self.width(), 1))
            right = np.array((np.cos(angle), np.sin(angle)))
            forward = np.array((-np.sin(angle), np.cos(angle)))
            self.camera.lookat[:2] += (
                -float(delta.x()) * right + float(delta.y()) * forward
            ) * scale
        event.accept()

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = float(np.exp(-event.angleDelta().y() * 0.0011))
        self.camera.distance = float(np.clip(self.camera.distance * factor, 0.55, 4.0))
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.reset_camera()
        event.accept()


class _LogStream(io.TextIOBase):
    def __init__(self, signal) -> None:
        self.signal = signal
        self.pending = ""

    def write(self, value: str) -> int:
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.strip():
                self.signal.emit(line.strip())
        return len(value)

    def flush(self) -> None:
        if self.pending.strip():
            self.signal.emit(self.pending.strip())
        self.pending = ""


class MetricTile(QFrame):
    def __init__(self, title: str, value: str = "—") -> None:
        super().__init__()
        self.setObjectName("metric")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 9, 11, 9)
        layout.setSpacing(3)
        label = QLabel(title.upper())
        label.setObjectName("metricLabel")
        self.value = QLabel(value)
        self.value.setObjectName("metricValue")
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label)
        layout.addWidget(self.value)

    def set_value(self, value: str, tone: str = "normal") -> None:
        self.value.setText(value)
        colors = {"good": "#67e8b6", "warn": "#f6c667", "bad": "#ff7b82"}
        self.value.setStyleSheet(f"color:{colors.get(tone, '#e8f5f0')};")


class NativeV2Window(QMainWindow):
    event_signal = Signal(dict)
    result_signal = Signal(dict)
    log_signal = Signal(str)

    def __init__(self, default_instruction: str) -> None:
        super().__init__()
        # The display runtime and its OpenGL resources always stay on the Qt
        # main thread. A separate task runtime is created inside the worker.
        self.runtime = MujocoRuntime(width=640, height=480, speed=2.5)
        self.worker: threading.Thread | None = None
        self._active_runtime: MujocoRuntime | None = None
        self._active_runtime_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._latest_snapshot = self.runtime.state_snapshot()
        self.scene_memory = SceneMemory()

        self.camera = mujoco.MjvCamera()
        self._reset_camera()
        self.runtime.model.vis.global_.offwidth = max(
            int(self.runtime.model.vis.global_.offwidth), 1280
        )
        self.runtime.model.vis.global_.offheight = max(
            int(self.runtime.model.vis.global_.offheight), 800
        )
        self.runtime.model.vis.quality.offsamples = 8
        self.view_data = mujoco.MjData(self.runtime.model)
        self.live_renderer = mujoco.Renderer(
            self.runtime.model, width=1280, height=800
        )
        self.phase_chips: dict[str, QLabel] = {}
        self.metrics: dict[str, MetricTile] = {}
        self.latest_evidence: dict[str, object] = {}
        self._build_ui(default_instruction)
        self.event_signal.connect(self._on_event)
        self.result_signal.connect(self._on_result)
        self.log_signal.connect(self._append_log)
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._refresh_scene)
        self.render_timer.start(40)
        self._refresh_scene()

    def _reset_camera(self) -> None:
        mujoco.mjv_defaultFreeCamera(self.runtime.model, self.camera)
        self.camera.lookat[:] = (0.0, 0.0, 0.20)
        self.camera.distance = 1.55
        self.camera.azimuth = 135.0
        self.camera.elevation = -30.0

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 13, 15, 15)
        layout.setSpacing(10)
        heading = QLabel(title.upper())
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        return card, layout

    def _build_ui(self, default_instruction: str) -> None:
        self.setWindowTitle("Embodied V2 · Manipulation Studio")
        self.resize(1580, 970)
        self.setMinimumSize(1220, 760)
        app = QApplication.instance()
        if app is not None:
            app.setFont(QFont("PingFang SC", 12))

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 16, 18, 18)
        outer.setSpacing(13)

        header = QHBoxLayout()
        logo = QLabel("V2")
        logo.setObjectName("logo")
        header.addWidget(logo)
        heading = QVBoxLayout()
        eyebrow = QLabel("EMBODIED MANIPULATION")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Manipulation Studio")
        title.setObjectName("title")
        heading.addWidget(eyebrow)
        heading.addWidget(title)
        header.addLayout(heading)
        header.addStretch(1)
        self.scene_count = QLabel("18 ENTITIES · LOCAL")
        self.scene_count.setObjectName("badge")
        self.status = QLabel("●  READY")
        self.status.setObjectName("ready")
        header.addWidget(self.scene_count)
        header.addWidget(self.status)
        outer.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(14)
        scene_card, scene_layout = self._card("Third-person scene · live")
        scene_header = QHBoxLayout()
        self.phase_title = QLabel("待机 · 等待自然语言任务")
        self.phase_title.setObjectName("scenePhase")
        scene_header.addWidget(self.phase_title)
        scene_header.addStretch(1)
        camera_badge = QLabel("1280 × 800  /  INTERACTIVE")
        camera_badge.setObjectName("badge")
        scene_header.addWidget(camera_badge)
        scene_layout.addLayout(scene_header)
        self.scene_label = InteractiveSceneLabel(self.camera, self._reset_camera)
        self.scene_label.setObjectName("scene")
        self.scene_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scene_label.setMinimumSize(760, 520)
        self.scene_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        scene_layout.addWidget(self.scene_label, 1)
        help_text = QLabel("左键旋转  ·  右键平移  ·  滚轮缩放  ·  双击复位第三视角")
        help_text.setObjectName("muted")
        scene_layout.addWidget(help_text)
        content.addWidget(scene_card, 1)

        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFrameShape(QFrame.Shape.NoFrame)
        panel_scroll.setFixedWidth(465)
        panel_body = QWidget()
        panel = QVBoxLayout(panel_body)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.setSpacing(11)

        command_card, command_layout = self._card("Mission command")
        self.instruction = QLineEdit(default_instruction)
        self.instruction.setPlaceholderText("例如：把红色杯子拿到黄色盘子上")
        self.instruction.setClearButtonEnabled(True)
        self.instruction.returnPressed.connect(self.start_task)
        command_layout.addWidget(self.instruction)
        quick = QComboBox()
        quick.addItems((
            "快速选择任务…",
            "把红色杯子拿到黄色盘子上",
            "把蓝色方块拿到黄色盘子上",
            "把白色杯子拿到黄色盘子上",
            "把紫色瓶子放到黄色盘子旁边",
        ))
        quick.currentTextChanged.connect(
            lambda value: self.instruction.setText(value)
            if value and not value.endswith("…") else None
        )
        command_layout.addWidget(quick)
        buttons = QHBoxLayout()
        self.run_button = QPushButton("开始执行")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.start_task)
        self.stop_button = QPushButton("安全停止")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_task)
        buttons.addWidget(self.run_button, 2)
        buttons.addWidget(self.stop_button, 1)
        command_layout.addLayout(buttons)
        panel.addWidget(command_card)

        phase_card, phase_layout = self._card("Execution phases")
        self.progress = QProgressBar()
        self.progress.setRange(0, len(PHASES))
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        phase_layout.addWidget(self.progress)
        chip_grid = QGridLayout()
        chip_grid.setHorizontalSpacing(5)
        chip_grid.setVerticalSpacing(5)
        for index, phase in enumerate(PHASES):
            chip = QLabel(PHASE_LABELS[phase])
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setProperty("state", "pending")
            chip.setObjectName("phaseChip")
            self.phase_chips[phase] = chip
            chip_grid.addWidget(chip, index // 3, index % 3)
        phase_layout.addLayout(chip_grid)
        panel.addWidget(phase_card)

        metric_card, metric_layout = self._card("Measured safety evidence")
        metric_grid = QGridLayout()
        metric_grid.setSpacing(7)
        definitions = (
            ("contact", "Contact"), ("rise", "Object rise"),
            ("slip", "Relative slip"), ("tilt", "Tilt"),
            ("route", "Route"), ("clearance", "Min clearance"),
            ("samples", "Trajectory"), ("safety", "Safety veto"),
        )
        for index, (key, label) in enumerate(definitions):
            tile = MetricTile(label)
            self.metrics[key] = tile
            metric_grid.addWidget(tile, index // 2, index % 2)
        metric_layout.addLayout(metric_grid)
        panel.addWidget(metric_card)

        event_card, event_layout = self._card("Evidence stream")
        self.events = QTextEdit()
        self.events.setReadOnly(True)
        self.events.setMinimumHeight(180)
        self.events.setPlaceholderText("运行后显示状态机与安全证据")
        event_layout.addWidget(self.events)
        panel.addWidget(event_card)
        panel.addStretch(1)
        panel_scroll.setWidget(panel_body)
        content.addWidget(panel_scroll)
        outer.addLayout(content, 1)
        self.setCentralWidget(root)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#07100f; color:#edf8f4; }
            QFrame#card { background:#0d1917; border:1px solid #20332f; border-radius:18px; }
            QLabel#logo { min-width:46px; min-height:46px; max-width:46px; max-height:46px; border-radius:14px; background:#62e6b4; color:#062019; font-size:16px; font-weight:900; qproperty-alignment:AlignCenter; }
            QLabel#eyebrow { color:#67e8b6; font-size:10px; font-weight:700; letter-spacing:2px; }
            QLabel#title { color:#f1faf6; font-size:25px; font-weight:750; }
            QLabel#sectionTitle { color:#a9c2ba; font-size:10px; font-weight:700; letter-spacing:1.5px; }
            QLabel#scenePhase { color:#dcebe6; font-size:14px; font-weight:650; }
            QLabel#muted { color:#718d84; font-size:11px; }
            QLabel#badge { color:#9bb2aa; background:#091411; border:1px solid #294039; border-radius:10px; padding:5px 9px; font-family:Menlo; font-size:10px; }
            QLabel#ready, QLabel#running, QLabel#done, QLabel#failed { border-radius:11px; padding:6px 11px; font-size:11px; font-weight:750; }
            QLabel#ready { color:#a7bbb5; background:#0a1513; border:1px solid #263b35; }
            QLabel#running { color:#071711; background:#67e8b6; border:1px solid #67e8b6; }
            QLabel#done { color:#071711; background:#67e8b6; border:1px solid #67e8b6; }
            QLabel#failed { color:#ffc1c4; background:#32181c; border:1px solid #71353c; }
            QLabel#scene { background:#050b0a; border:1px solid #1d302b; border-radius:13px; }
            QScrollArea { background:transparent; border:none; }
            QScrollBar:vertical { width:7px; background:transparent; }
            QScrollBar::handle:vertical { background:#2b4941; border-radius:3px; min-height:30px; }
            QLineEdit, QComboBox { min-height:42px; background:#07120f; border:1px solid #29443d; border-radius:10px; padding:0 11px; color:#edf8f4; font-size:13px; }
            QLineEdit:focus, QComboBox:focus { border:1px solid #56dbaa; }
            QComboBox QAbstractItemView { background:#0d1917; color:#edf8f4; border:1px solid #29443d; selection-background-color:#245747; }
            QPushButton { min-height:42px; border-radius:11px; font-size:13px; font-weight:750; }
            QPushButton#primary { background:#62e6b4; color:#062019; border:none; }
            QPushButton#primary:hover { background:#7bf0c4; }
            QPushButton#primary:pressed { background:#45c995; }
            QPushButton#danger { background:#241316; color:#ffabb0; border:1px solid #653038; }
            QPushButton#danger:hover { background:#351a1f; }
            QPushButton:disabled { color:#53665f; background:#101b18; border:1px solid #20312c; }
            QProgressBar { min-height:6px; max-height:6px; border:none; border-radius:3px; background:#14241f; }
            QProgressBar::chunk { border-radius:3px; background:#62e6b4; }
            QLabel#phaseChip { min-height:25px; border-radius:7px; background:#091411; border:1px solid #1d302b; color:#5e766f; font-size:9px; padding:1px 3px; }
            QLabel#phaseChip[state="done"] { color:#86ac9f; border:1px solid #285644; background:#10231d; }
            QLabel#phaseChip[state="active"] { color:#052019; border:1px solid #62e6b4; background:#62e6b4; font-weight:800; }
            QFrame#metric { background:#081310; border:1px solid #1c302b; border-radius:11px; }
            QLabel#metricLabel { color:#718a82; font-size:9px; letter-spacing:1px; }
            QLabel#metricValue { color:#e8f5f0; font-family:Menlo; font-size:12px; font-weight:650; }
            QTextEdit { background:#07120f; border:1px solid #1d302b; border-radius:11px; color:#9bb5ad; padding:8px; font-family:Menlo; font-size:10px; selection-background-color:#23604c; }
        """)

    def _render_frame(self) -> np.ndarray:
        # Take an immutable snapshot reference only; never wait for the task
        # runtime or invoke its renderer from the GUI thread.
        with self._snapshot_lock:
            snapshot = self._latest_snapshot
        self.view_data.qpos[:] = snapshot.qpos
        self.view_data.qvel[:] = snapshot.qvel
        self.view_data.ctrl[:] = snapshot.ctrl
        if self.runtime.model.nmocap:
            self.view_data.mocap_pos[:] = snapshot.mocap_pos
            self.view_data.mocap_quat[:] = snapshot.mocap_quat
        mujoco.mj_forward(self.runtime.model, self.view_data)
        self.live_renderer.update_scene(self.view_data, camera=self.camera)
        return self.live_renderer.render().copy()

    def _publish_snapshot(self, snapshot: SimulationSnapshot) -> None:
        """Publish a complete state copy without touching Qt or OpenGL."""
        with self._snapshot_lock:
            self._latest_snapshot = snapshot

    def _refresh_scene(self) -> None:
        if self.scene_label.width() < 2:
            return
        rgb = self._render_frame()
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(image)
        ratio = max(1.0, float(self.scene_label.devicePixelRatioF()))
        scaled = pixmap.scaled(
            self.scene_label.size() * ratio,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(ratio)
        self.scene_label.setPixmap(scaled)

    def _set_status(self, name: str, detail: str) -> None:
        self.status.setText(f"●  {name}")
        object_name = {
            "READY": "ready", "RUNNING": "running", "DONE": "done",
            "STOPPING": "running", "STOPPED": "ready", "FAILED": "failed",
        }.get(name, "ready")
        self.status.setObjectName(object_name)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.phase_title.setText(detail)

    def _set_phase(self, current: str) -> None:
        current_index = PHASES.index(current) if current in PHASES else -1
        for index, (phase, chip) in enumerate(self.phase_chips.items()):
            state = "done" if current == "done" or index < current_index else (
                "active" if index == current_index else "pending"
            )
            chip.setProperty("state", state)
            chip.style().unpolish(chip)
            chip.style().polish(chip)
        self.progress.setValue(
            len(PHASES) if current == "done" else max(0, current_index + 1)
        )

    @staticmethod
    def _mm(value) -> str:
        return "—" if value is None else f"{float(value) * 1000:.1f} mm"

    def _on_event(self, event: dict) -> None:
        phase = str(event.get("phase", "idle"))
        evidence = dict(event.get("evidence") or {})
        self.latest_evidence.update(evidence)
        self._set_phase(phase)
        self._set_status("RUNNING", PHASE_LABELS.get(phase, phase))
        self.events.append(
            f"[{PHASE_LABELS.get(phase, phase)}] "
            + (json.dumps(evidence, ensure_ascii=False) if evidence else event.get("verdict", ""))
        )
        bar = self.events.verticalScrollBar()
        bar.setValue(bar.maximum())
        values = self.latest_evidence
        self.metrics["contact"].set_value(str(values.get("contact_state", "—")))
        self.metrics["rise"].set_value(self._mm(values.get("object_rise_m")))
        self.metrics["slip"].set_value(self._mm(values.get("relative_slip_m")))
        tilt = values.get("tilt_deg")
        self.metrics["tilt"].set_value("—" if tilt is None else f"{float(tilt):.2f}°")
        self.metrics["route"].set_value(str(values.get("corridor", "—")))
        clearances = [
            value for value in (
                values.get("robot_clearance_m"), values.get("payload_clearance_m")
            ) if value is not None
        ]
        self.metrics["clearance"].set_value(
            self._mm(min(clearances)) if clearances else "—",
            "good" if clearances else "normal",
        )
        samples = values.get("trajectory_samples")
        self.metrics["samples"].set_value("—" if samples is None else f"{samples} samples")
        safety = str(values.get("safety_verdict", "LOCAL"))
        self.metrics["safety"].set_value(safety.upper(), "good" if safety != "fail" else "bad")

    def _on_result(self, result: dict) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if result.get("success"):
            self._set_phase("done")
            self._set_status("DONE", "任务完成 · 放置关系与稳定性已验证")
        elif result.get("final_phase") == "stopped":
            self._set_status("STOPPED", "任务已安全停止，保留当前场景")
        else:
            self._set_status("FAILED", str(result.get("message", "任务失败")))

    def _append_log(self, line: str) -> None:
        if line:
            self.events.append(f"› {line}")

    def start_task(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        instruction = self.instruction.text().strip()
        if not instruction:
            self._set_status("FAILED", "请先输入自然语言指令")
            return
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._stop_requested.clear()
        self.events.clear()
        self.latest_evidence.clear()
        for metric in self.metrics.values():
            metric.set_value("—")
        self._set_phase("idle")
        self._set_status("RUNNING", "正在初始化完整多物体场景…")
        self.worker = threading.Thread(
            target=self._run_agent,
            args=(instruction,),
            daemon=True,
        )
        self.worker.start()

    def _run_agent(self, instruction: str) -> None:
        stream = _LogStream(self.log_signal)
        task_runtime: MujocoRuntime | None = None
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                # MuJoCo's offscreen renderer is thread-affine on macOS. Build
                # and close this runtime in the worker that performs perception.
                task_runtime = MujocoRuntime(width=640, height=480, speed=2.5)
                with self._active_runtime_lock:
                    self._active_runtime = task_runtime
                if self._stop_requested.is_set():
                    task_runtime.request_stop()
                task_runtime.attach_state_sink(
                    self._publish_snapshot,
                    playback_speed=2.5,
                    frame_interval_s=0.04,
                )

                def emit(event: PhaseEvent) -> None:
                    payload = asdict(event)
                    payload["phase"] = event.phase.value
                    self.event_signal.emit(payload)

                result = EmbodiedAgentV2(
                    task_runtime,
                    use_online_grounding=True,
                    scene_memory=self.scene_memory,
                    event_sink=emit,
                ).run(instruction)
                payload = asdict(result)
                payload["final_phase"] = result.final_phase.value
                payload["failure"] = result.failure.value if result.failure else None
                self.result_signal.emit(payload)
        except Exception as error:
            self.log_signal.emit(traceback.format_exc())
            self.result_signal.emit({
                "success": False,
                "final_phase": "stopped",
                "message": str(error),
            })
        finally:
            stream.flush()
            with self._active_runtime_lock:
                if self._active_runtime is task_runtime:
                    self._active_runtime = None
            if task_runtime is not None:
                task_runtime.close()

    def stop_task(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self._stop_requested.set()
            with self._active_runtime_lock:
                task_runtime = self._active_runtime
            if task_runtime is not None:
                task_runtime.request_stop()
            self.stop_button.setEnabled(False)
            self._set_status("STOPPING", "正在安全停止并冻结当前位置…")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_requested.set()
        with self._active_runtime_lock:
            task_runtime = self._active_runtime
        if task_runtime is not None:
            task_runtime.request_stop()
        self.render_timer.stop()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=1.5)
        self.live_renderer.close()
        self.runtime.close()
        event.accept()


def legacy_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instruction",
        default="把红色杯子拿到黄色盘子上",
    )
    parser.add_argument("--autorun", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Embodied V2")
    app.setStyle("Fusion")
    window = NativeV2Window(args.instruction)
    if not args.no_show:
        window.show()
    if args.autorun:
        QTimer.singleShot(400, window.start_task)
    if args.screenshot is not None:
        def save_screenshot() -> None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))

        QTimer.singleShot(650, save_screenshot)
    if args.smoke:
        QTimer.singleShot(1000, app.quit)
    app.exec()


def main() -> None:
    """Compatibility launcher for the visual-agent studio."""
    from simulation.studio import main as visual_studio
    visual_studio()


if __name__ == "__main__":
    main()
