# V2 架构升级与验收记录

> 历史记录说明：本文保留升级过程中当时的检查结果。下文“五项 5/5”属于早期实现，不是当前版本的总体成功率。当前小规模随机完整任务结果及限制见 [公开证据](../evidence/README.md) 和仓库首页 README。

## 1. 旧系统审计

根目录原先没有 README 或统一 requirements。V1 的 `camera_demo.py` 集成了任务解释、颜色/VLM 观察、物体登记、抓取和运输。原 V2 已分离 manager、executor、motion、validation，但扫描器仍枚举仿真对象，读取对象位置、类型和颜色；最终验证和碰撞环境克隆同样使用场景真值。因此“有相机画面”不代表执行依赖真实视觉。

本次先审阅了原有场景、XML、控制器、视觉、任务/运动规划、测试和脚本。复用了 PandaController 的物理接触控制、`solve_hand_ik` 和现有通用 `plan_rrt_connect`。没有通过直接移动对象坐标、固定焊接或改写物理对象状态实现抓放。

## 2. 新架构

自然语言 → Pydantic Goal → 多视角 RGB-D → WorldModel/SceneGraph → 符号技能计划 → 短段运动 → 传感反馈 → 分类恢复 → 新观测下的最终关系/稳定性验证。

统一 API 是 `EmbodiedAgent.execute(task)`；状态包括 IDLE、UNDERSTAND、OBSERVE、PLAN、EXECUTE、MONITOR、VERIFY、RECOVER、REPLAN、SUCCESS、FAILED。取消、技能数、恢复数、不同抓取数、LLM 调用数和总时长都有限制。

## 3. 新增文件

完整清单见本目录 `upgrade-files.txt`。主要新增：`run_agent.py`、`requirements-agent.txt`、`config/agent.json`、`scripts/benchmark_agent.py`，以及 `src/agent`、`perception`、`world`、`skills`、`manipulation`、`validation`、`recovery`、`robot`、`simulation`、`llm`、`config`、`evaluation`。核心技能按职责合并在 `observation.py` 和 `manipulation.py` 中，没有为了逐字匹配目录示例而复制实现。

## 4. 修改文件

- `src/joint_control.py`：增加编译前的可选 `spec_hook`，复用旧模型创建和控制器。
- `src/embodied_v2/__main__.py`：默认桌面入口切换至新视觉 Agent。
- `src/embodied_v2/embodied_agent.py`：CLI main 转发新 Agent；旧类保留用于回归参考。
- `src/embodied_v2/native_ui.py`：main 转发新 UI；旧窗口实现保留为 legacy_main。
- 新根 README、V2 迁移说明、配置和测试。

开始工作时 `camera_demo.py`、`motion_planning.py`、`test_embodied_agent.py` 已有用户未提交修改；没有回滚或将这些修改冒称为本次新增。原 `embodied_v2/`、多份批次说明、artifacts 和旧 V2 测试原先也已存在但未被 Git 跟踪。

## 5. 删除/废弃

没有删除旧文件。旧 `EmbodiedAgentV2`、`MujocoRuntime` 与旧 native 窗口属于历史回归参考，不作为新的像素感知 API。原始 V1 演示仍保留原有行为。所有文档推荐的新版 CLI/native 启动命令均进入新 Agent；直接导入旧类仍可能使用旧真值逻辑，不能据此宣称旧代码整体已消除真值依赖。

## 6. Perception

相机输出 RGB、深度、K、光学外参和时间戳。自身网格只用于去除机器人像素。默认 fallback 基于颜色与图像连通域，再用三维连通域拆开同色前后重叠物体。有效 mask 深度形成点云，通过鲁棒范围、可见顶面、圆弧/杯壁与邻近支撑面估计几何。

多视角更新位置。遮挡时保留过去实际测得的尺寸，不把剩余表面中心当成完整物体中心。抓取后的点云还要与“先前视觉形状随夹爪移动”的假设相符。DINO/SAM2 为真实 adapter，未安装权重、未声称运行；默认回退确实使用像素，不查场景登记库。

## 7. LLM

在线调用沿用现有兼容 API，任务理解、规划、恢复有独立提示词和 schema。中文颜色/类别统一到视觉词汇。模型只能选择白名单目标与技能，不能提供关节角、轨迹、力和 PID。技能顺序在本地强制验证。`auto` 和 `offline` 的回退会记录在日志中；`required` 不静默回退。

已实际运行在线 LLM 中文任务，存在真实 API interpretation/plan 事件及完整抓放证据。默认视觉并不是每帧调用 LLM：语言模型根据视觉构建的世界模型选择对象和步骤。

## 8. WorldModel

维护稳定对象 ID、语义、颜色、类别、中心、尺寸、姿态、置信度、可见性、时间戳、可移动性和不确定度。遮挡后保留短期记忆并衰减置信度；不把记忆当成新观测。SceneGraph 支持 on、inside、near、overlapping、前后左右和上下关系。坐标方向在 README 中明确。

“空桌面”必须由实际深度覆盖证明，不能仅凭没有检测到物体就认定安全。暂存区域单独标注为 observed_support_region，不伪装成识别出的物体。

## 9. Skills 与执行

每个注册技能声明前提、execute、monitor、成功/失败条件。抓取有多种旋转方向、顶部进入和侧面进入候选。夹爪根据双侧接触力做有次数限制的渐进闭合。抬起必须有新图像中物体随夹爪运动的证据。

路径使用旧 IK 与 OMPL，碰撞世界仅含机器人模型、标定桌面和视觉障碍代理。携带时保持竖直、限制速度；基座后方穿行困难时用前侧绕行路径并逐步调整 yaw。每段后更新视觉，每个物理步监控关节、力、非预期接触、姿态及轨迹误差。

## 10. Recovery

保留 TARGET_NOT_FOUND、LOW_CONFIDENCE、DEPTH_INVALID、IK_FAILED、PATH_BLOCKED、COLLISION_RISK、NO_CONTACT、GRASP_FAILED、OBJECT_SLIPPED、TARGET_LOST、PLACEMENT_FAILED、RELATION_NOT_SATISFIED，另有 USER_STOP。

先本地换视角/换抓取候选/重新估计目标/改变运输高度与路线。连续失败才提供任务、世界状态、失败类型、接触力、距离、候选与历史给 LLM。危险状态和用户停止不交给 LLM 覆盖。尚有双侧接触但附着无法确认时会停住持物，不会在空中直接松手重试。放置时的力卸载必须有新的视觉支撑证据才能继续。

## 11. 真值隔离

执行层已取消：物体目录枚举、body 名称定位、rgba 作为识别、直接读取目标 xpos/body_pos、根据真实对象几何抓取、克隆全部真实场景做碰撞、根据真实目标位置宣告完成。

允许使用：机器人关节/末端位置、接触传感、相机内外参、渲染 RGB-D。这些是本体感知与传感器标定，不是环境物体真值。`simulation/environment.py` 仍有场景构建和随机初始位置；它们不传入 Agent。碰撞代理中的 body_pos 写入来自视觉估计，不是读取物理对象位置。

`SimulationGroundTruth` 位于 evaluation，仅任务结束后可读取；每次读取也检查状态。AST 测试禁止纯运行层导入仿真、旧登记/视觉系统或访问 xpos/qpos 等属性。该守卫是工程边界与回归机制，不是恶意代码的安全沙箱。

## 12. 运行验证

现有 V1：158 项通过。原 V2：27 项通过。新测试：25 项通过，覆盖投影、深度、三维拆分、记忆、关系、schema、计划顺序、可见空区、用户停止、接触/姿态、真值守卫、黑图负对照、同视角双次验收与遮挡形状。

本机 `pip check` 通过。MuJoCo 实际渲染、离线抓放、在线 LLM 抓放、native Qt 窗口及窗口内完整任务均已运行。窗口证据：`outputs/agent_studio_smoke.png`、`outputs/agent_studio_task.png`。

额外的两步清障演示使用 `--scenario blocked`：先把视觉识别的青色遮挡物移到观察过的空桌面，再取出红杯放到另一个观察过的空桌面。两步均在同一视角连续新观测下满足关系和稳定性，整项任务返回成功；证据在 `outputs/agent/task_20261014T152310_69cb52fa`。这项额外演示未单独运行仿真真值评分，不能混入五项 benchmark 的成功率。

最新五项 benchmark（seed 23）：Agent 视觉判定和任务结束后的独立真值均为 5/5；抓取 5/5、放置 5/5；平均位置误差 0.000910 米；恢复 0 次；碰撞 0 次；离线运行 LLM 调用 0 次；平均执行 61.15 秒。机器可读结果与每个任务证据路径记录在 `outputs/agent_benchmark_acceptance.json`，并汇总到 `acceptance-results.json`。早期失败日志保留，不冒充全通过。

## 13. 已知限制

- HSV fallback 是有限颜色/几何词汇，不能据此声称识别任意未知物体；复杂纹理、透明/反光物体、同色拥挤物体仍可能失败。
- 几何估计与抓取主要针对当前桌面上的直立刚体。侧面抓取是从侧方进入的夹取，尚未验证通用水平腕姿抓取。
- inside 关系有几何实现，但未重建空腔时放入动作明确拒绝；不能把实心包围盒当成容器内部。
- 遮挡搬运仍可能因不确定度/附着检验失败而安全停止。边界姿态、障碍密度和可达性会限制任务成功率。
- 清障/取出使用实际可见的暂存区域并逐目标验收，已验证上述青色遮挡物实例。更贴靠的物体、较宽的盒子抓取或小物体被单侧指尖带起时仍可能安全停止；不承诺任意堆叠的通用清障。
- 大模型 adapter、真实机械臂、通用 6D 姿态和真实力反馈尚未完成硬件验证。
- macOS 渲染器报告 ARB_clip_control 不可用，因此深度精度有限；最终 benchmark 会报告实测误差。
- 旧历史类为兼容测试保留，直接调用旧类不享有新视觉边界保证。

## 14. 如何运行

```bash
.venv/bin/python run_agent.py
.venv/bin/python run_agent.py '把红色杯子放到黄色盘子上' --offline
.venv/bin/python run_agent.py '把红色杯子放到黄色盘子上' --required-llm
PYTHONPATH=src .venv/bin/python -m embodied_v2
.venv/bin/python scripts/benchmark_agent.py --trials 5 --seed 23
```

完整配置、日志、可选模型与扩展方式见根 README。

## 15. 真机迁移

替换 RobotInterface 的 MuJoCo 实现、CameraInterface 实现和启动层；提供真实驱动、FK/IK/碰撞模型、时间同步、相机/手眼标定、接触/力传感、实体急停和控制周期。Agent、符号任务、WorldModel、技能及验证逻辑继续复用。PolicyInterface 预留 learned policy，不依赖 VLA 才能运行。
