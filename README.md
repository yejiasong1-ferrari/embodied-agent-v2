# V2 Studio — a robot that checks its own work

Give a simulated Franka Panda a tabletop instruction. It observes the scene through RGB-D vision, plans a grasp and placement, moves the object, then looks again to decide whether the requested relationship is actually true.

**One-line demo:** `把蓝色盒子放到绿色积木右边` — “Place the blue box to the right of the green block” → **see → locate → grasp → place → verify**.

[![V2 Studio after placing the blue box to the right of the green block](evidence/v2-studio-blue-box-demo.png)](https://github.com/yejiasong1-ferrari/embodied-agent-v2/raw/refs/heads/main/evidence/v2-studio-blue-box-demo.mp4)

**[Open the V2 Studio demonstration (MP4, 30 seconds)](https://github.com/yejiasong1-ferrari/embodied-agent-v2/raw/refs/heads/main/evidence/v2-studio-blue-box-demo.mp4).** This is a screen recording of an actual MuJoCo task. The final frame shows the blue box released beside the green block and V2's fresh-image validation reporting success. The Studio scene is the *human observer's* movable camera; the agent makes decisions from its separate wrist-mounted RGB-D camera.

## How the robot works

1. **Understand the goal.** The agent turns the instruction into a target object, a reference object, and a spatial condition: *blue box* must end up *right of* *green block*. A language model can help with task interpretation and ambiguous visual candidates, but it does not send joint angles or bypass the robot's local checks.
2. **Look for measured objects.** A simulated wrist RGB-D camera supplies color, depth, and camera calibration. The default visual pipeline finds candidate regions in pixels and projects valid depth into 3D positions and sizes. Routine scans use 640 × 480; when candidates remain hard to distinguish and visual review is available, it can take a calibrated 1280 × 960 detail image. If the view is insufficient, the robot can observe again from another angle.
3. **Maintain a world model.** V2 records what it has seen, where it estimates each object to be, and how uncertain or stale that estimate is. During a task, the agent does **not** look up object names or positions from MuJoCo's scene registry. Simulator ground truth is reserved for evaluation after execution.
4. **Choose and execute a safe motion.** Local skills generate grasp candidates and a placement target. Inverse kinematics, collision checks, workspace limits, and short monitored motion segments decide what can actually be executed. Contact and new observations determine whether the object was picked up or whether a bounded recovery is needed.
5. **Verify the result.** After release, the arm clears the view and the camera observes again. A validator checks that the object is released, stable, and in the requested spatial relationship. Studio shows **Complete** only after this final check passes.

The visible loop is therefore **language → RGB-D evidence → world model → local skills and motion → fresh visual validation**. [V2 Studio](docs/v2-studio.md) exposes the same agent's live task state, visual evidence, and readable event log; its rotatable, zoomable 3D view is for inspection, not a shortcut to simulation truth.

## Try it locally

Tested with Python 3.12 and MuJoCo 3.14 on macOS. The repository includes the complete V2 code, tests, and the pinned Panda model submodule.

```bash
git clone --recurse-submodules https://github.com/yejiasong1-ferrari/embodied-agent-v2.git
cd embodied-agent-v2
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-agent.txt
.venv/bin/python run_studio.py
```

In Studio, enter `把蓝色盒子放到绿色积木右边` and run the task. Drag to rotate the scene, scroll to zoom, and use Shift-drag to pan. The right panel shows the current action, camera evidence, recovery attempts, and final validation. For a terminal-only run using local language rules: `.venv/bin/python run_agent.py '把蓝色盒子放到绿色积木右边' --offline`.

## Evidence and scope

- [Demo video and evidence notes](evidence/README.md) describe the recording and selected run artifacts. The [online red-cup task](evidence/online-red-cup-result.json), [local Studio screenshot](evidence/studio-local-red-cup-success.png), and [three randomized offline trials](evidence/offline-randomized-three-trials.json) are **separate runs**; they are not a single experiment or a general success-rate estimate.
- The current system is a **MuJoCo research prototype**. Default visual detection relies mainly on limited color and depth cues; an optional vision model can help resolve semantic ambiguity. It is not a general open-world object recognizer. Heavy occlusion, some container operations, and transfer to a physical arm remain open problems.
- The UI shows an observer view, while robot actions depend on RGB-D observations, robot state, and contact feedback. Recorded outcomes should be read together with their validation evidence. More detail is in the [architecture notes](docs/architecture-upgrade.md) and [Studio guide](docs/v2-studio.md).

The next research steps are uncertainty-aware active perception, more robust contact-rich grasping, and validation on real hardware. The project is intended as a reproducible base for studying how language, vision, and manipulation can form a closed loop.
