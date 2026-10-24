"""Independent local operations console for Embodied Manipulation V2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time
import webbrowser

from flask import Flask, jsonify, request, send_from_directory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v2_controller"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
EVENT_PREFIX = "V2_EVENT "

app = Flask(__name__)
state_lock = threading.Lock()
state: dict[str, object] = {
    "status": "idle",
    "phase": "idle",
    "instruction": "",
    "events": [],
    "logs": [],
    "report": None,
    "run_id": None,
    "process": None,
    "started_at": None,
    "finished_at": None,
    "idle_entities": 0,
}


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Embodied V2 · Operations Console</title>
<style>
:root{--bg:#07100f;--surface:#0d1917;--surface2:#101f1c;--line:#20332f;--text:#ecf8f3;--muted:#8da49d;--mint:#5ee9b5;--mint2:#29bd8b;--amber:#f2be5c;--red:#ff6f75;--blue:#78a9ff;--radius:18px}
*{box-sizing:border-box}body{margin:0;color:var(--text);background:radial-gradient(circle at 12% 0,#15352d 0,transparent 28%),radial-gradient(circle at 92% 12%,#152942 0,transparent 25%),var(--bg);font:14px/1.5 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:100vh}
.shell{max-width:1540px;margin:auto;padding:24px}.top{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:20px}.brand{display:flex;align-items:center;gap:14px}.mark{width:44px;height:44px;border-radius:14px;background:linear-gradient(145deg,var(--mint),#1d8c6c);box-shadow:0 0 34px #44dba840;display:grid;place-items:center;color:#062019;font-weight:900}.eyebrow{font-size:11px;letter-spacing:.17em;text-transform:uppercase;color:var(--mint)}h1{font-size:24px;margin:1px 0 0;letter-spacing:-.03em}.topmeta{display:flex;gap:9px;align-items:center}.pill{padding:7px 11px;border:1px solid var(--line);border-radius:999px;color:var(--muted);background:#0a1513}.status{display:flex;align-items:center;gap:7px;color:var(--text)}.dot{width:8px;height:8px;border-radius:50%;background:#667}.status.running .dot{background:var(--mint);box-shadow:0 0 0 5px #5ee9b518;animation:pulse 1.5s infinite}.status.completed .dot{background:var(--mint)}.status.failed .dot{background:var(--red)}@keyframes pulse{50%{opacity:.45}}
.command{display:grid;grid-template-columns:minmax(260px,1fr) auto auto;gap:10px;padding:12px;background:rgba(13,25,23,.92);border:1px solid var(--line);border-radius:var(--radius);box-shadow:0 18px 55px #0004;margin-bottom:14px}.inputwrap{position:relative}.inputwrap span{position:absolute;left:14px;top:10px;font-size:10px;letter-spacing:.12em;color:var(--muted);text-transform:uppercase}.inputwrap input{width:100%;height:58px;padding:24px 14px 8px;background:#081310;border:1px solid #29443d;border-radius:12px;color:var(--text);font-size:16px;outline:none}.inputwrap input:focus{border-color:var(--mint2);box-shadow:0 0 0 3px #5ee9b512}button{border:0;border-radius:12px;padding:0 20px;font-weight:750;cursor:pointer;transition:.18s}button:hover{transform:translateY(-1px)}button:disabled{opacity:.4;cursor:not-allowed;transform:none}.run{background:linear-gradient(135deg,var(--mint),#2dc28f);color:#062019}.stop{background:#281519;color:#ff9a9f;border:1px solid #65343a}.quick{display:flex;gap:8px;align-items:center;margin:0 0 18px 4px;color:var(--muted);font-size:12px}.quick button{padding:6px 10px;border:1px solid var(--line);background:#0c1816;color:#b7cbc5;font-weight:500;border-radius:999px}
.phasebar{display:grid;grid-template-columns:repeat(10,1fr);gap:5px;margin-bottom:16px}.phase{min-width:0;border:1px solid var(--line);background:#0a1513;border-radius:9px;padding:7px 6px;color:#668078;text-align:center;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.phase.done{color:#8cb9aa;border-color:#275846;background:#10221d}.phase.active{color:#071510;background:var(--mint);border-color:var(--mint);font-weight:800;box-shadow:0 0 20px #5ee9b530}.phase.error{background:#431e23;color:#ffb5b8;border-color:#8d3b45}
.grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(360px,.65fr);gap:14px}.stack{display:grid;gap:14px}.card{background:linear-gradient(155deg,rgba(16,31,28,.96),rgba(10,21,19,.96));border:1px solid var(--line);border-radius:var(--radius);overflow:hidden;box-shadow:0 15px 45px #0003}.cardhead{display:flex;justify-content:space-between;align-items:center;padding:14px 16px;border-bottom:1px solid var(--line)}.cardhead h2{font-size:12px;letter-spacing:.1em;text-transform:uppercase;margin:0;color:#bad0c9}.cardbody{padding:14px 16px}.viewport{position:relative;min-height:430px;background:linear-gradient(135deg,#06100e,#0b1715);display:grid;place-items:center;overflow:hidden}.viewport img{width:100%;height:430px;object-fit:contain;image-rendering:auto}.empty{color:#587068;text-align:center}.empty b{display:block;color:#91aaa2;margin-bottom:4px}.hud{position:absolute;inset:14px 14px auto;display:flex;justify-content:space-between;pointer-events:none}.hud span{padding:6px 9px;border:1px solid #ffffff24;border-radius:9px;background:#06100dcc;backdrop-filter:blur(8px);font:11px ui-monospace,monospace}.scanline{position:absolute;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--mint),transparent);box-shadow:0 0 14px var(--mint);animation:scan 4s linear infinite;opacity:0}.running .scanline{opacity:.65}@keyframes scan{from{top:8%}to{top:92%}}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.metric{padding:12px;background:#081411;border:1px solid #1d302c;border-radius:12px;min-width:0}.metric label{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px}.metric strong{display:block;font:600 13px ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.good{color:var(--mint)}.warn{color:var(--amber)}.danger{color:var(--red)}
.contracts{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.contract{padding:10px;border-radius:11px;background:#0a1614;border:1px solid var(--line)}.contract i{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--mint);margin-right:6px}.contract b{display:block;font-size:11px}.contract span{font-size:10px;color:var(--muted)}
.eventlist{height:340px;overflow:auto;padding:6px 14px 12px}.event{display:grid;grid-template-columns:9px 118px 1fr;gap:10px;padding:10px 2px;border-bottom:1px solid #1c2d29}.event:last-child{border-bottom:0}.eventdot{width:7px;height:7px;border-radius:50%;background:var(--mint);margin-top:6px}.event .name{font:600 11px ui-monospace,monospace;color:#cce0da}.event .detail{font:11px ui-monospace,monospace;color:var(--muted);word-break:break-word}.log{height:190px;overflow:auto;margin:0;padding:12px;background:#06100e;color:#8ba69d;font:11px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap}.result{padding:12px;border-radius:12px;background:#081411;border:1px solid var(--line)}.result strong{display:block;margin-bottom:3px}.result small{color:var(--muted)}
@media(max-width:1050px){.grid{grid-template-columns:1fr}.phasebar{grid-template-columns:repeat(5,1fr)}}@media(max-width:680px){.shell{padding:12px}.top{align-items:flex-start}.topmeta{flex-direction:column;align-items:flex-end}.command{grid-template-columns:1fr 1fr}.inputwrap{grid-column:1/-1}.phasebar{grid-template-columns:repeat(4,1fr)}.metrics,.contracts{grid-template-columns:repeat(2,1fr)}.viewport,.viewport img{height:300px;min-height:300px}}
</style></head><body><main class="shell">
<header class="top"><div class="brand"><div class="mark">V2</div><div><div class="eyebrow">Embodied Manipulation</div><h1>Transport Safety Console</h1></div></div><div class="topmeta"><div id="timer" class="pill">00:00</div><div id="status" class="pill status"><i class="dot"></i><span>IDLE</span></div></div></header>
<section class="command"><div class="inputwrap"><span>Natural language mission</span><input id="instruction" value="把红色杯子拿到黄色盘子上" autocomplete="off"></div><button id="run" class="run" onclick="runTask()">开始执行</button><button id="stop" class="stop" onclick="stopTask()">安全停止</button></section>
<div class="quick">快速任务 <button onclick="setPrompt('把蓝色方块拿到黄色盘子上')">蓝色方块 → 盘子</button><button onclick="setPrompt('把白色杯子拿到黄色盘子上')">白色杯子 → 盘子</button><button onclick="setPrompt('把紫色瓶子放到黄色盘子旁边')">紫色瓶子 → 旁边</button></div>
<section id="phases" class="phasebar"></section>
<div class="grid"><div class="stack">
 <section class="card"><div class="cardhead"><h2>Scene camera · Third-person live</h2><span id="frameState" class="pill">等待任务</span></div><div id="viewport" class="viewport"><div id="empty" class="empty"><b>第三视角尚未启动</b>运行任务后实时显示完整多物体场景</div><img id="preview" hidden><div class="hud"><span id="hudPhase">PHASE · IDLE</span><span id="hudRun">RUN · —</span></div><div class="scanline"></div></div></section>
 <section class="card"><div class="cardhead"><h2>Measured evidence</h2><span class="pill">LOCAL AUTHORITY</span></div><div class="cardbody"><div class="metrics">
  <div class="metric"><label>Contact</label><strong id="contact">—</strong></div><div class="metric"><label>Object rise</label><strong id="rise">—</strong></div><div class="metric"><label>Relative slip</label><strong id="slip">—</strong></div><div class="metric"><label>Tilt</label><strong id="tilt">—</strong></div>
  <div class="metric"><label>Route</label><strong id="route">—</strong></div><div class="metric"><label>Robot clearance</label><strong id="robotClearance">—</strong></div><div class="metric"><label>Payload clearance</label><strong id="payloadClearance">—</strong></div><div class="metric"><label>Trajectory samples</label><strong id="samples">—</strong></div>
 </div></div></section>
 <section class="card"><div class="cardhead"><h2>Transport safety contract</h2></div><div class="cardbody"><div class="contracts"><div class="contract"><i></i><b>Whole robot</b><span>clone collision rollout</span></div><div class="contract"><i></i><b>Dynamic payload</b><span>measured rigid transform</span></div><div class="contract"><i></i><b>Actual trajectory</b><span>dense execution samples</span></div><div class="contract"><i></i><b>Safety veto</b><span>manager cannot override</span></div></div></div></section>
</div><aside class="stack">
 <section class="card"><div class="cardhead"><h2>Mission state</h2><span id="scene" class="pill">0 entities</span></div><div class="cardbody"><div id="result" class="result"><strong>系统就绪</strong><small>输入自然语言任务开始运行。</small></div></div></section>
 <section class="card"><div class="cardhead"><h2>Phase evidence stream</h2><span id="eventCount" class="pill">0 events</span></div><div id="events" class="eventlist"></div></section>
 <section class="card"><div class="cardhead"><h2>Runtime log</h2></div><pre id="logs" class="log"></pre></section>
</aside></div></main>
<script>
const phaseOrder=['understand_task','scan_scene','approach','close_reobserve','pre_align','contact_search','micro_lift','stable_grasp','plan_transport','transport','reobserve_destination','pre_place','descent','first_contact','load_transfer','gradual_release','verify_result','done'];
const phaseLabel={understand_task:'任务理解',scan_scene:'场景扫描',approach:'粗接近',close_reobserve:'近距观测',pre_align:'抓取对准',contact_search:'接触搜索',micro_lift:'试抬验证',stable_grasp:'稳定抓取',plan_transport:'运输规划',transport:'安全运输',reobserve_destination:'目标重观测',pre_place:'预放置',descent:'受控下降',first_contact:'首次接触',load_transfer:'载荷转移',gradual_release:'渐进释放',verify_result:'最终验证',done:'完成',recovery:'恢复',stopped:'停止',idle:'待机'};
const phases=document.getElementById('phases');phaseOrder.forEach(p=>{let e=document.createElement('div');e.className='phase';e.id='phase-'+p;e.textContent=phaseLabel[p];phases.appendChild(e)});
const fmtM=v=>v==null?'—':(Number(v)*1000).toFixed(1)+' mm';const fmtDeg=v=>v==null?'—':Number(v).toFixed(2)+'°';
function setPrompt(v){document.getElementById('instruction').value=v}
function latest(events,key){for(let i=events.length-1;i>=0;i--){if(events[i].evidence&&events[i].evidence[key]!=null)return events[i].evidence[key]}return null}
async function runTask(){let instruction=document.getElementById('instruction').value.trim();if(!instruction)return;let r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({instruction})});if(!r.ok){let x=await r.json();alert(x.error||'启动失败')}}
async function stopTask(){await fetch('/api/stop',{method:'POST'})}
function render(s){let running=s.status==='running'||s.status==='stopping';document.getElementById('run').disabled=running;document.getElementById('stop').disabled=!running;let badge=document.getElementById('status');badge.className='pill status '+s.status;badge.querySelector('span').textContent=s.status.toUpperCase();document.getElementById('timer').textContent=formatTime(s.elapsed_s||0);document.getElementById('hudPhase').textContent='PHASE · '+(phaseLabel[s.phase]||s.phase).toUpperCase();document.getElementById('hudRun').textContent='RUN · '+(s.run_id||'—');document.getElementById('viewport').className='viewport '+(running?'running':'');
 let idx=phaseOrder.indexOf(s.phase);phaseOrder.forEach((p,i)=>{let el=document.getElementById('phase-'+p);el.className='phase '+(i<idx?'done':i===idx?'active':'')});if(s.status==='failed'||s.status==='stopped'){if(idx>=0)document.getElementById('phase-'+s.phase)?.classList.add('error')}
 if(s.image_url){let img=document.getElementById('preview');img.src=s.image_url+'?v='+(s.image_version||Date.now());img.hidden=false;document.getElementById('empty').hidden=true;document.getElementById('frameState').textContent=running?'LIVE':s.status==='idle'?'SCENE READY':'FINAL FRAME'}
 let ev=s.events||[];document.getElementById('eventCount').textContent=ev.length+' events';let entities=latest(ev,'entities');document.getElementById('scene').textContent=(entities==null?(s.idle_entities||0):entities)+' entities';
 document.getElementById('contact').textContent=latest(ev,'contact_state')||'—';document.getElementById('rise').textContent=fmtM(latest(ev,'object_rise_m'));document.getElementById('slip').textContent=fmtM(latest(ev,'relative_slip_m'));document.getElementById('tilt').textContent=fmtDeg(latest(ev,'tilt_deg'));document.getElementById('route').textContent=latest(ev,'corridor')||'—';document.getElementById('robotClearance').textContent=fmtM(latest(ev,'robot_clearance_m'));document.getElementById('payloadClearance').textContent=fmtM(latest(ev,'payload_clearance_m'));document.getElementById('samples').textContent=latest(ev,'trajectory_samples')??'—';
 let box=document.getElementById('result'),report=s.report||{};if(s.status==='completed'){box.innerHTML='<strong class="good">任务完成并验证</strong><small>'+escapeHtml(report.message||'关系与稳定性检查通过')+'</small>'}else if(s.status==='failed'){box.innerHTML='<strong class="danger">任务未完成</strong><small>'+escapeHtml(report.message||'查看恢复证据')+'</small>'}else if(running){box.innerHTML='<strong class="warn">'+escapeHtml(phaseLabel[s.phase]||s.phase)+'</strong><small>本地安全层正在验证当前阶段。</small>'}else{box.innerHTML='<strong>系统就绪</strong><small>输入自然语言任务开始运行。</small>'}
 let list=document.getElementById('events');list.innerHTML=ev.slice().reverse().map(e=>'<div class="event"><i class="eventdot"></i><span class="name">'+escapeHtml(phaseLabel[e.phase]||e.phase)+'</span><span class="detail">'+escapeHtml(JSON.stringify(e.evidence||{}))+'</span></div>').join('');document.getElementById('logs').textContent=(s.logs||[]).join('');}
function formatTime(v){v=Math.floor(v);return String(Math.floor(v/60)).padStart(2,'0')+':'+String(v%60).padStart(2,'0')}function escapeHtml(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]))}
async function refresh(){try{let r=await fetch('/api/status');render(await r.json())}catch(e){}}setInterval(refresh,500);refresh();
</script></body></html>"""


def _run_worker(instruction: str, run_id: str, online_manager: bool) -> None:
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    snapshot_path = run_dir / "live.png"
    command = [
        str(PYTHON), "-u", "-m", "embodied_v2.embodied_agent", instruction,
        "--report", str(report_path), "--snapshot", str(snapshot_path),
        "--live-view",
    ]
    if online_manager:
        command.append("--online-manager")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    with state_lock:
        state["process"] = process
    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.strip()
        if stripped.startswith(EVENT_PREFIX):
            try:
                event = json.loads(stripped[len(EVENT_PREFIX):])
            except json.JSONDecodeError:
                event = None
            if event is not None:
                with state_lock:
                    cast_events = state["events"]
                    assert isinstance(cast_events, list)
                    cast_events.append(event)
                    state["phase"] = event["phase"]
                continue
        with state_lock:
            cast_logs = state["logs"]
            assert isinstance(cast_logs, list)
            cast_logs.append(line)
            del cast_logs[:-250]
    return_code = process.wait()
    report = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = None
    with state_lock:
        state["report"] = report
        state["process"] = None
        state["finished_at"] = time.time()
        if state["status"] == "stopping":
            state["status"] = "stopped"
            state["phase"] = "stopped"
        else:
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
        if state["status"] in {"running", "stopping"}:
            return jsonify({"error": "a V2 task is already running"}), 409
        run_id = str(int(time.time() * 1000))
        state.update(
            status="running",
            phase="understand_task",
            instruction=instruction,
            events=[],
            logs=[],
            report=None,
            run_id=run_id,
            process=None,
            started_at=time.time(),
            finished_at=None,
        )
    thread = threading.Thread(
        target=_run_worker,
        args=(instruction, run_id, bool(payload.get("online_manager", False))),
        daemon=True,
    )
    thread.start()
    return jsonify({"ok": True, "run_id": run_id})


@app.post("/api/stop")
def stop_task():
    with state_lock:
        process = state.get("process")
        if isinstance(process, subprocess.Popen) and process.poll() is None:
            process.terminate()
            state["status"] = "stopping"
    return jsonify({"ok": True})


@app.get("/api/status")
def status():
    with state_lock:
        payload = {key: value for key, value in state.items() if key != "process"}
    started = payload.get("started_at")
    finished = payload.get("finished_at")
    payload["elapsed_s"] = (
        max(0.0, float(finished or time.time()) - float(started)) if started else 0.0
    )
    run_id = payload.get("run_id")
    image = OUTPUT_ROOT / str(run_id) / "live.png" if run_id else None
    if image is not None and image.exists():
        payload["image_url"] = f"/artifacts/{run_id}/live.png"
        payload["image_version"] = image.stat().st_mtime_ns
    else:
        idle_image = OUTPUT_ROOT / "scene" / "idle.png"
        if idle_image.exists():
            payload["image_url"] = "/artifacts/scene/idle.png"
            payload["image_version"] = idle_image.stat().st_mtime_ns
        else:
            payload["image_url"] = None
            payload["image_version"] = 0
    return jsonify(payload)


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "system": "embodied_v2"})


@app.get("/artifacts/<run_id>/<path:filename>")
def artifact(run_id: str, filename: str):
    return send_from_directory(OUTPUT_ROOT / run_id, filename)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    idle_image = OUTPUT_ROOT / "scene" / "idle.png"
    if not idle_image.exists():
        from PIL import Image

        from .adapters import MujocoRuntime

        idle_image.parent.mkdir(parents=True, exist_ok=True)
        with MujocoRuntime(width=640, height=480) as runtime:
            Image.fromarray(runtime.render_third_person()).save(idle_image)
            state["idle_entities"] = (
                len(runtime.movable_body_names())
                + len(tuple(runtime.iter_static_body_names()))
            )
    else:
        from .adapters import MujocoRuntime

        with MujocoRuntime(width=32, height=24) as runtime:
            state["idle_entities"] = (
                len(runtime.movable_body_names())
                + len(tuple(runtime.iter_static_body_names()))
            )
    url = "http://127.0.0.1:7862"
    if os.getenv("EMBODIED_V2_NO_BROWSER") != "1":
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    print(f"Embodied V2 UI: {url}", flush=True)
    app.run(host="127.0.0.1", port=7862, debug=False, threaded=True)


if __name__ == "__main__":
    main()
