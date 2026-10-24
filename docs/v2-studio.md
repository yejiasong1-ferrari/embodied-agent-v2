# V2 Studio · 本地机器人工作区

## 启动与操作

```bash
.venv/bin/python run_studio.py
# 或双击「启动V2控制台.command」
# 兼容原入口：
PYTHONPATH=src .venv/bin/python -m embodied_v2
```

等待顶部 `Simulation · connected` 和 `READY`。在底部输入完整任务，例如「把蓝色盒子放到绿色积木右边」，选择语言模式，点击 **Run Task** 或按 Enter。指令会交给真实 `agent.embodied_agent.EmbodiedAgent.execute()`，当前接的是本地 MuJoCo 仿真。

### 观察场景

- 左键拖动：围绕观察中心旋转。
- 滚轮、触控板滚动或缩放手势：拉近/拉远，距离限制 0.55–4 m。
- Shift + 左键、右键或中键拖动：平移观察中心。
- 视角菜单：Perspective / Top / Front / Left / Right。
- 右下角 XYZ：指示方向，点击轴切换标准视角。
- **R**：Reset View；只复位用户镜头。
- **F**：专注模式。保留任务摘要和运行时 Stop；右边缘展开 Inspector，底边缘展开命令栏。
- **Esc**：退出专注模式。R/F 仅在场景获得键盘焦点时生效，在输入框打字不会触发。
- 物体选择：点击视觉世界模型中物体的投影中心附近，或在「视觉」物体列表中选择；右侧显示视觉估计与关系，可聚焦观察镜头。点击空白处返回任务页。

场景默认只在悬停、选中时显示物体名称。图像和世界模型的旧观测会标记 `STALE`，位置不是持续查询仿真真值。

### Inspector

右侧顶部的 **当前任务** 始终显示指令、真实阶段、当前动作和最新事件。下面的 **任务详情** 显示实际技能计划、目标和抓取候选、失败恢复记录、最终视觉验收；**事件**是默认打开的完整中文日志。恢复重试时进度允许退回；「重试技能已通过」只表示该重试步骤通过。只有明确的 `final_validation.accepted=true` 加上 Agent 成功状态才显示完成。

**视觉**提供实际 RGB、Detection、Depth 与用户 Scene 图像。点击图像可在独立窗口中拖动、滚轮放大。深度来自真实 RGB-D depth，固定 0–3 m 灰度映射，近亮远暗，无效值黑色；缺失数据显示 Unavailable。超过 3 秒的观测标记 STALE。

**事件**提供中文时间线，可筛选异常与恢复。用户向上阅读时，新事件不会强制滚回底部，使用「有新事件 ↓」继续跟随。Developer Mode 可展开每项原始证据。

**Runs**打开最近 100 个本地任务的只读历史：状态、UTC 任务时间、耗时、事件、RGB/检测图片和结果。始终标记 **REPLAY**。这是已保存证据浏览，不是机械臂轨迹重新执行。

### 停止与重置

Stop 直接设置取消标志并停止控制器，不排在任务队列或相机请求后面。正在进行的网络请求或计算需要返回后完成收尾，界面在此期间显示 STOPPING。专注模式保留停止按钮。

设置中的 **Reset View** 只复位观察镜头；**Reset Scene** 重建仿真、Agent 与视觉记忆，运行时禁用。停止后如仍夹着物体，应 Reset Scene 再开始。正常连续任务沿用现场。

### 三种语言模式

- 自动：优先在线 LLM，回退明确记入日志。
- 在线 LLM 必须成功：调用失败即失败，不静默改用规则。
- 本地规则：有限语法解析；仍有实际 RGB-D 感知、运动与验收。

默认检测器是 HSV / 深度几何的有限词汇视觉。LLM 负责语言理解、高层计划和恢复建议，未把此系统标成任意物体视觉识别。设置可查看检测器，任务详情可查看实际语言后端和 LLM 次数。

## 线程与数据边界

1. **GUI 线程**：Qt 事件、交互、布局、事件展示；约 30 Hz 从单槽邮箱取最新渲染帧，250 ms 刷新计时和数据年龄。
2. **AgentWorker**：拥有执行 Simulation、感知 renderer 和真实 Agent，处理任务及物理步。最高约 25 Hz 复制 `mjSTATE_INTEGRATION`。模拟锁只保护短暂状态复制/物理步/停止控制写入。
3. **RenderWorker**：拥有独立复制的 `MjModel`、`MjData`、GL renderer 和用户 `MjvCamera`。从快照重建可视状态；不在原模拟数据上渲染。LLM 网络等待时依然能处理镜头请求。观察画面按场景尺寸与屏幕像素密度请求像素，最高 1920×1200；目标上限 30 fps，实际显示测量值；静止时停止冗余渲染，慢设备降低分辨率。感知 RGB-D 参数不受观察画面像素设置影响。

`studio/gateway.py` 是 UI 的命令与观测边界：界面向它提交任务、停止、重建和观察镜头请求，向上只接收事件、图像与状态快照。它仍与正式 Agent 运行在同一个本地进程内；UI 展示层没有单独重写机器人任务逻辑。

渲染侧的地板颜色与 grid 只修改复制模型/可视几何；不会影响感知图像、碰撞或摩擦。GUI 不把渲染拾取结果、物体 body/geom ID 或仿真位置传给 Agent。选择完全使用 WorldModel 估计位置和渲染相机投影；Focus 只写用户相机中心。

## 开发者模式

设置 → Developer Mode：打开 Raw WorldModel、机器人关节/末端/夹爪/仿真时间、最近的实际抓握力测量（带采样时间）、原始事件及 LLM 请求/响应。密钥不写入事件；请求记录只含实际 payload、模型、用途、调用编号、endpoint 和是否含图像。高频完整轨迹仍在任务文件中。

## 文件

- `run_studio.py` / `simulation/studio.py`：正式启动与兼容 CLI。
- `studio/window.py`：窗口与信号接线。
- `studio/runtime.py`：任务/渲染线程与快照邮箱。
- `studio/camera.py` / `scene.py`：纯镜头数学、场景、工具栏与 gizmo。
- `studio/bridge.py`：事件驱动的 Mission 状态。
- `studio/components.py`：任务、视觉、事件、物体详情与命令栏。
- `studio/history.py`：异步读取历史证据。
- `studio/theme.py`：统一样式与设计变量。
- `studio/gateway.py`：UI 与任务/渲染线程之间的适配接口。

旧的单文件 `simulation/studio.py` 窗口/同线程渲染实现已由上述组件替代，保留同一导入路径和 CLI。`design/v2_ui_preview.py` 是此前设计预览，不是正式入口。

## 可复现检查

```bash
# 启动 / 截图 / 干净关闭
.venv/bin/python run_studio.py --smoke --screenshot outputs/studio-ready.png
# 完整在线执行
.venv/bin/python run_studio.py --autorun --required-llm --quit-after-task
# 自动化实际 Qt 操作与真实 Agent 任务
.venv/bin/python scripts/check_studio_workspace.py --case blue
.venv/bin/python scripts/check_studio_workspace.py --case red
.venv/bin/python scripts/check_studio_workspace.py --case failure
.venv/bin/python scripts/check_studio_workspace.py --case stop-llm
.venv/bin/python scripts/check_studio_workspace.py --case stop-motion --offline
# 证据和镜头契约
PYTHONPATH=src:tests .venv/bin/python -m unittest test_studio_workspace test_studio_events test_agent_contracts test_agent_geometry test_agent_ground_truth_guard
```

检查脚本会调用已配置的在线服务（`--offline` 可禁用），保存真实任务日志和 `outputs/studio_checks/` 下的检查报告/窗口截图。截图名称不代表成功，应以报告和 `result.json` 为准。完整升级与本轮验收记录见 [UI 重构验收报告](v2-studio-ui-acceptance.md)。
