# Selected public evidence

This folder contains a small set of checkable demonstrations and result files. Full per-frame RGB-D, contact, and motion logs remain in the author's local `outputs/` directory and are not included here. These items come from **different runs**; they do not establish an overall success rate or performance on physical hardware.

## Blue-box V2 Studio demonstration

**Instruction:** `把蓝色盒子放到绿色积木右边` — “Place the blue box to the right of the green block.” [Watch the 30-second video on the README page](../README.md#watch-a-real-v2-run), or [download the MP4](v2-studio-blue-box-demo.mp4). The [still frame](v2-studio-blue-box-demo.png) shows the final Studio view.

This is a screen recording of an actual V2 Studio task on 2026-10-24. The footage shows the arm grasping, placing, releasing, and then receiving a successful visual task check. The public MP4 was resized, its silent audio track was removed, and the trailing macOS menu capture was trimmed. The still was taken from the video. No complete per-frame task log is published for this recording, so its outcome can be inspected in the footage but it cannot be used to calculate a success rate.

## Other, independent runs

- [`online-red-cup-task.json`](online-red-cup-task.json) and [`online-red-cup-result.json`](online-red-cup-result.json) are from one online LLM run of “place the red cup on the yellow plate.” The required LLM mode was active; final visual relation and stability validation passed. The result records about 177 seconds, 11 LLM calls, and one recovery. Its `log_path` points to an unpublished local full log.
- [`offline-randomized-three-trials.json`](offline-randomized-three-trials.json) contains three randomized complete tasks using local language rules. Two of three tasks passed; three of three grasps and two of three placements passed. The third task encountered `PATH_BLOCKED` and did **not** pass final agent validation. Its `ground_truth_success=true` field only says a post-run truth relation was satisfied; it is not equivalent to complete task success. Listed log paths refer to local unpublished files.
- [`studio-local-red-cup-report.json`](studio-local-red-cup-report.json) and [`studio-local-red-cup-success.png`](studio-local-red-cup-success.png) document a separate local-rules Studio UI check. They are not screenshots or logs from the online red-cup run above.

During execution, the agent uses RGB-D observations, robot state, and contact feedback. Simulator object truth is accessed for evaluation **after** execution. Studio screenshots show the human observer camera, not the agent's visual input.
