# Selected public evidence / 精简公开证据

这里仅保留少量演示和原始结果，便于核对系统能力和限制。逐帧 RGB-D、接触与运动日志仍在作者本机 `outputs/`，没有上传。这些是分别录制或运行的案例，**不能代表总体成功率或实体机器人性能**。

- [`v2-studio-blue-box-demo.mp4`](v2-studio-blue-box-demo.mp4) 和 [`v2-studio-blue-box-demo.png`](v2-studio-blue-box-demo.png)：2026-10-24 的 V2 Studio 真实任务录屏，指令为“把蓝色盒子放到绿色积木右边”。画面展示机械臂抓取、放置、释放和最终视觉验收通过。公开 MP4 只做了缩小分辨率、去除静音音轨、裁掉末尾录入的 macOS 菜单；封面截自这段视频。它与下面的红杯和随机试验不是同一次运行。此视频没有配套公开的逐帧任务日志，因此只能据画面核对该次演示，不能用于计算成功率。

- [`online-red-cup-task.json`](online-red-cup-task.json) 和 [`online-red-cup-result.json`](online-red-cup-result.json)：同一次在线 LLM 运行，指令是“把红色杯子放到黄色盘子上”。`llm.mode=required`，最终视觉关系与稳定性验收通过；用时约 177 秒，11 次 LLM 调用，1 次恢复。结果 JSON 中的 `log_path` 指向未上传的本地完整日志。
- [`offline-randomized-three-trials.json`](offline-randomized-three-trials.json)：一组三项随机完整任务，离线语言解析，2/3 任务成功、3/3 抓取成功、2/3 放置成功。第三项出现 `PATH_BLOCKED`，Agent 未通过最终验收；其中 `ground_truth_success=true` 只表示任务结束后某项真值关系成立，不能代替完整任务成功。文件中的 `log_path` 也指向未上传的本地日志。
- [`studio-local-red-cup-report.json`](studio-local-red-cup-report.json) 和 [`studio-local-red-cup-success.png`](studio-local-red-cup-success.png)：另一项本地规则模式的 Studio UI 检查及结束画面。它与上面的在线 LLM 任务**不是同一次运行**。

仿真物体真值仅由执行完成后的评估代码读取；Agent 执行依据 RGB-D 观测、机器人自身状态和接触反馈。截图是给人看的 Studio 观察视角，不是 Agent 的视觉输入。
