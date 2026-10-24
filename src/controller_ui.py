"""Local interactive dashboard for the language-to-robot pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time
import webbrowser

from flask import Flask, jsonify, request, send_from_directory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "controller"
MJPYTHON = PROJECT_ROOT / ".venv" / "bin" / "mjpython"
CAMERA_DEMO = PROJECT_ROOT / "src" / "camera_demo.py"

app = Flask(__name__)
state_lock = threading.Lock()
state = {
    "status": "idle",
    "instruction": "",
    "logs": [],
    "report": None,
    "run_id": None,
    "process": None,
}


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Panda Active Vision Controller</title>
<style>
:root{--bg:#0b1020;--card:#141b2d;--line:#29324a;--text:#edf2ff;--muted:#94a3bd;--blue:#68a7ff;--green:#4ade80;--red:#fb7185}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#090d18,#111a30);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.shell{max-width:1440px;margin:auto;padding:24px}.top{display:flex;align-items:end;justify-content:space-between;gap:16px;margin-bottom:18px}h1{margin:0;font-size:28px}.sub{color:var(--muted)}
.command{display:grid;grid-template-columns:1fr auto auto;gap:10px;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px;margin-bottom:16px}
input{width:100%;background:#0b1120;border:1px solid #39445f;border-radius:10px;padding:12px 14px;color:white;font-size:16px}button{border:0;border-radius:10px;padding:0 20px;font-weight:700;cursor:pointer}.run{background:var(--blue)}.stop{background:#34202a;color:var(--red);border:1px solid #6f3443}
.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:16px}.stack{display:grid;gap:16px}.card{background:rgba(20,27,45,.96);border:1px solid var(--line);border-radius:16px;padding:16px;min-width:0}.card h2{font-size:15px;margin:0 0 12px;color:#c9d7f2}.badge{padding:5px 10px;border-radius:99px;background:#222c42;color:var(--muted)}.running{color:#ffe08a}.completed{color:var(--green)}.failed{color:var(--red)}
pre{margin:0;white-space:pre-wrap;word-break:break-word;background:#090e1b;border-radius:10px;padding:12px;color:#cfe0ff;max-height:310px;overflow:auto;font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}
.preview{width:100%;min-height:320px;object-fit:contain;background:#090e1b;border-radius:12px;border:1px solid #222c42}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.metric{background:#0d1424;border-radius:10px;padding:10px}.metric span{display:block;color:var(--muted);font-size:11px}.metric strong{font-family:ui-monospace,monospace;font-size:12px}.logs{height:300px}
@media(max-width:900px){.grid{grid-template-columns:1fr}.command{grid-template-columns:1fr 1fr}.command input{grid-column:1/-1}.metrics{grid-template-columns:1fr}}
</style></head><body><div class="shell">
<div class="top"><div><h1>Panda Active Vision Controller</h1><div class="sub">Natural language → LLM plan → RGB-D detector → base coordinates → IK → execution</div></div><div id="status" class="badge">idle</div></div>
<div class="command"><input id="instruction" value="把红色杯子拿到黄色盘子上" placeholder="输入机器人指令"><button class="run" onclick="runTask()">运行指令</button><button class="stop" onclick="stopTask()">停止</button></div>
<div class="grid"><div class="stack">
  <section class="card"><h2>Detector 输出 · 框 / 轮廓 / 中心</h2><img id="preview" class="preview" alt="等待检测结果"></section>
  <section class="card"><h2>坐标与执行</h2><div class="metrics"><div class="metric"><span>Target base XYZ</span><strong id="targetXYZ">—</strong></div><div class="metric"><span>Reference base XYZ</span><strong id="referenceXYZ">—</strong></div><div class="metric"><span>Depth</span><strong id="depth">—</strong></div></div><pre id="execution">等待运行</pre></section>
</div><div class="stack">
  <section class="card"><h2>LLM 输入</h2><pre id="llmInput">等待指令</pre></section>
  <section class="card"><h2>LLM 输出 · Task JSON</h2><pre id="llmOutput">等待规划</pre></section>
  <section class="card"><h2>Detector 输入 / 输出</h2><pre id="detector">等待图像</pre></section>
  <section class="card"><h2>实时日志</h2><pre id="logs" class="logs"></pre></section>
</div></div></div>
<script>
const pretty=x=>x?JSON.stringify(x,null,2):'—';
async function runTask(){const instruction=document.getElementById('instruction').value.trim();if(!instruction)return;await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({instruction})});}
async function stopTask(){await fetch('/api/stop',{method:'POST'});}
async function refresh(){let r=await fetch('/api/status');let s=await r.json();let st=document.getElementById('status');st.textContent=s.status;st.className='badge '+s.status;document.getElementById('logs').textContent=(s.logs||[]).join('');let p=s.report||{};document.getElementById('llmInput').textContent=pretty(p.llm_input||{instruction:s.instruction});document.getElementById('llmOutput').textContent=pretty({backend:p.planner_backend,task:p.llm_output});document.getElementById('detector').textContent=pretty({input:p.detector_input,target:p.target_detection,reference:p.reference_detection});document.getElementById('execution').textContent=pretty(p.execution);let t=p.target_detection,rp=p.reference_detection;document.getElementById('targetXYZ').textContent=t?pretty(t.base_xyz_m):'—';document.getElementById('referenceXYZ').textContent=rp?pretty(rp.base_xyz_m):'—';document.getElementById('depth').textContent=t?(t.depth_m.toFixed(3)+' m'):'—';if(s.image_url)document.getElementById('preview').src=s.image_url+'?v='+Date.now();}
setInterval(refresh,700);refresh();
</script></body></html>"""


def _run_worker(instruction: str, run_id: str) -> None:
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    image_path = run_dir / "detection.png"
    report_path = run_dir / "report.json"
    command = [
        str(MJPYTHON), str(CAMERA_DEMO), "--instruction", instruction,
        "--output", str(image_path), "--report", str(report_path),
        "--no-viewer", "--no-show",
    ]
    process = subprocess.Popen(
        command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, env=os.environ.copy(),
    )
    with state_lock:
        state["process"] = process
    assert process.stdout is not None
    for line in process.stdout:
        with state_lock:
            state["logs"].append(line)
    return_code = process.wait()
    report = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    with state_lock:
        state["report"] = report
        state["process"] = None
        state["status"] = "completed" if return_code == 0 else "failed"


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def run_task():
    payload = request.get_json(force=True)
    instruction = str(payload.get("instruction", "")).strip()
    if not instruction:
        return jsonify({"error": "instruction is required"}), 400
    with state_lock:
        if state["status"] == "running":
            return jsonify({"error": "a task is already running"}), 409
        run_id = str(int(time.time() * 1000))
        state.update(status="running", instruction=instruction, logs=[], report=None, run_id=run_id, process=None)
    threading.Thread(target=_run_worker, args=(instruction, run_id), daemon=True).start()
    return jsonify({"ok": True, "run_id": run_id})


@app.post("/api/stop")
def stop_task():
    with state_lock:
        process = state.get("process")
        if process and process.poll() is None:
            process.terminate()
            state["status"] = "stopping"
    return jsonify({"ok": True})


@app.get("/api/status")
def status():
    with state_lock:
        payload = {key: value for key, value in state.items() if key != "process"}
    run_id = payload.get("run_id")
    image = OUTPUT_ROOT / str(run_id) / "detection.png" if run_id else None
    payload["image_url"] = f"/artifacts/{run_id}/detection.png" if image and image.exists() else None
    return jsonify(payload)


@app.get("/artifacts/<run_id>/<path:filename>")
def artifact(run_id: str, filename: str):
    return send_from_directory(OUTPUT_ROOT / run_id, filename)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    url = "http://127.0.0.1:7860"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print("Controller:", url)
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)


if __name__ == "__main__":
    main()
