> 迁移说明：新版启动入口已切换到 RGB-D 像素驱动的 Agent。请先阅读根目录 README.md 和 docs/architecture-upgrade.md。以下保留的是原 V2 历史说明，其中第三人称窗口、旧 Manager 和真值适配器描述不代表新版执行链路。

# Embodied Manipulation V2

V2 is an independent package. It reuses only the Panda/MuJoCo low-level
model/controller boundary and does not import V1 task, grasp, transport, or
recovery orchestration.

Run the headless end-to-end simulation from the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m embodied_v2.embodied_agent \
  '把红色杯子拿到黄色盘子上'
```

Open the native V2 desktop studio (recommended):

```bash
PYTHONPATH=src .venv/bin/python -m embodied_v2
```

The native PySide6 window loads the complete multi-object scene and renders a
continuous interactive third-person view while the robot executes the
instruction. Drag to orbit, right-drag to pan, use the wheel to zoom and
double-click to reset the camera. Phase, grasp, transport-safety and placement
evidence update beside the scene.

Run all V2 tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover \
  -s tests -p 'test_embodied_v2_*.py'
```

The online Manager is optional and is enabled with `--online-manager`. It
uses the existing `DASHSCOPE_API_KEY`/`OPENAI_API_KEY`, `VLM_BASE_URL`, and
`VLM_MODEL` environment variables. Its output remains symbolic; local
geometry, IK, force/contact handling, trajectory generation and the safety
veto remain authoritative.
