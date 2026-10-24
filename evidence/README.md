# Selected public evidence / 精简公开证据

这里仅保留少量原始结果与一张界面截图，便于核对系统能力和限制。逐帧 RGB-D、接触与运动日志仍在作者本机 `outputs/`，没有上传。下列案例发生于 2026-10-17，**不能代表总体成功率或实体机器人性能**。

- [`online-red-cup-task.json`](online-red-cup-task.json) 和 [`online-red-cup-result.json`](online-red-cup-result.json)：同一次在线 LLM 运行，指令是“把红色杯子放到黄色盘子上”。`llm.mode=required`，最终视觉关系与稳定性验收通过；用时约 177 秒，11 次 LLM 调用，1 次恢复。结果 JSON 中的 `log_path` 指向未上传的本地完整日志。
- [`offline-randomized-three-trials.json`](offline-randomized-three-trials.json)：一组三项随机完整任务，离线语言解析，2/3 任务成功、3/3 抓取成功、2/3 放置成功。第三项出现 `PATH_BLOCKED`，Agent 未通过最终验收；其中 `ground_truth_success=true` 只表示任务结束后某项真值关系成立，不能代替完整任务成功。文件中的 `log_path` 也指向未上传的本地日志。
- [`studio-local-red-cup-report.json`](studio-local-red-cup-report.json) 和 [`studio-local-red-cup-success.png`](studio-local-red-cup-success.png)：另一项本地规则模式的 Studio UI 检查及结束画面。它与上面的在线 LLM 任务**不是同一次运行**。

仿真物体真值仅由执行完成后的评估代码读取；Agent 执行依据 RGB-D 观测、机器人自身状态和接触反馈。截图是给人看的 Studio 观察视角，不是 Agent 的视觉输入。
