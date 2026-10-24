"""Native GUI acceptance checks against the real V2 agent (no mock frames/events).
Run from repo root with .venv/bin/python scripts/check_studio_workspace.py --case blue.
"""
import argparse,json,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from simulation.studio import Studio
from config import AgentConfig
from studio.camera import CameraState
from PySide6.QtCore import QTimer,Qt,QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QPushButton
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--case',choices=['blue','red','failure','stop-llm','stop-motion','reset','metric'],default='blue');p.add_argument('--offline',action='store_true');p.add_argument('--wrist',action='store_true');args=p.parse_args()
root=Path('outputs/studio_checks');root.mkdir(parents=True,exist_ok=True)
app=QApplication([]);config=AgentConfig()
if args.wrist:config.camera.mode='wrist'
w=Studio(config,'');w.show()
report={'case':args.case,'checks':{},'errors':[],'camera_frames_during_llm':0,'camera_frames_during_motion':0,'max_gui_timer_gap_s':0.}
def exception_hook(typ,value,tb):
    report['errors'].append(''.join(traceback.format_exception(typ,value,tb)));print(report['errors'][-1],flush=True);w.close()
sys.excepthook=exception_hook

phase={'llm':False,'motion':False,'run':False,'last_frame':0,'last_tick':time.monotonic(),'started':time.monotonic(),'result':False,'selected':False}

def check(name,value):
    report['checks'][name]=bool(value)
    if not value:report['errors'].append(name)

def guarded(fn):
    def inner(*a):
        try:fn(*a)
        except Exception:
            report['errors'].append(traceback.format_exc());print(report['errors'][-1],flush=True);w.close()
    return inner

@guarded
def setup():
    w.resize(1440,900);QTest.qWait(200)
    w.grab().save(str(root/f'{args.case}_1440.png'))
    check('scene_majority_1440',w.scene.width()/w.width()>.70)
    w.resize(1280,800);QTest.qWait(200);w.grab().save(str(root/f'{args.case}_1280.png'))
    check('layout_1280',w.inspector.width()>=260 and w.bar.start.geometry().right()<w.bar.width() and w.scene.height()>500)
    sim=w.worker.sim;state=sim.data.qpos.copy();ctrl=sim.data.ctrl.copy();mocap=sim.data.mocap_pos.copy();quat=sim.data.mocap_quat.copy();camera=sim.model.cam_pos.copy();world=json.dumps(w.world,sort_keys=True)
    original=w.scene.state;center=w.scene.rect().center()
    QTest.mousePress(w.scene,Qt.MouseButton.LeftButton,pos=center);QTest.mouseMove(w.scene,center+QPoint(100,30),50);QTest.mouseRelease(w.scene,Qt.MouseButton.LeftButton,pos=center+QPoint(100,30))
    check('orbit',w.scene.state.azimuth!=original.azimuth)
    before=w.scene.state;QTest.mousePress(w.scene,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ShiftModifier,center)
    QTest.mouseMove(w.scene,center+QPoint(25,20),50);QTest.mouseRelease(w.scene,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ShiftModifier,center+QPoint(25,20))
    check('pan',w.scene.state.lookat!=before.lookat)
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtCore import QPointF
    before=w.scene.state.distance
    event=QWheelEvent(QPointF(center),QPointF(w.scene.mapToGlobal(center)),QPoint(),QPoint(0,120),Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
    app.sendEvent(w.scene,event);check('wheel_zoom',w.scene.state.distance<before)
    w.command.setFocus();w.command.clear();QTest.keyClicks(w.command,'rf');check('typing_rf_does_not_trigger_shortcuts',w.command.text()=='rf' and not w.focused)
    QTest.keyClick(w.scene,Qt.Key.Key_R);check('reset_view',w.scene.state==CameraState())
    QTest.keyClick(w.scene,Qt.Key.Key_F);check('focus_mode',w.focused and not w.inspector.isVisible())
    w.grab().save(str(root/f'{args.case}_focus.png'));QTest.keyClick(w.scene,Qt.Key.Key_Escape);check('escape_focus',not w.focused)
    for name in ['Top','Front','Left','Right','Perspective']:w.toolbar.views.setCurrentText(name);w.toolbar.preset.emit(name);check('preset_'+name,w.scene.state==CameraState.preset(name))
    QTest.qWait(300)
    check('camera_physics_isolation',np.array_equal(state,sim.data.qpos) and np.array_equal(ctrl,sim.data.ctrl))
    check('camera_perception_isolation',np.array_equal(mocap,sim.data.mocap_pos) and np.array_equal(quat,sim.data.mocap_quat) and np.array_equal(camera,sim.model.cam_pos))
    check('camera_world_isolation',world==json.dumps(w.world,sort_keys=True))
    check('separate_render_model',w.mailbox.model is not sim.model and not np.shares_memory(w.mailbox.model.geom_rgba,sim.model.geom_rgba))
    check('render_output',w.last_frame>0 and not w.scene.image.isNull())
    if args.case=='reset':
        phase['old_sim']=sim;w.open_settings();dialog=w.dialogs[-1]
        button=dialog.findChild(QPushButton,'reset_scene');check('idle_reset_enabled',button.isEnabled())
        QTest.mouseClick(button,Qt.MouseButton.LeftButton);check('reset_disables_start',not w.start.isEnabled())
        phase['reset_pending']=True;return
    command='把蓝色盒子放到绿色积木左边8cm' if args.case=='metric' else '把红色杯子放到黄色盘子上' if args.case=='red' else '跳舞' if args.case=='failure' else '把蓝色盒子放到绿色积木右边'
    w.command.setText(command);w.mode.setCurrentIndex(w.mode.findData('offline' if args.offline or args.case=='failure' else 'required'))
    phase['run']=True;w.command.setFocus();QTest.keyClick(w.command,Qt.Key.Key_Return);check('input_enter_starts_task',w.busy)

@guarded
def trace(e):
    kind,d=e['type'],e['data']
    if kind=='llm_request':phase['llm']=True
    if kind in {'llm','llm_error'}:phase['llm']=False
    if kind=='state':
        phase['motion']=d.get('state')=='EXECUTE'
        if d.get('state')=='EXECUTE' and not (root/f'{args.case}_running.png').exists():QTimer.singleShot(350,lambda:w.grab().save(str(root/f'{args.case}_running.png')))
    if kind=='recovery':
        check('actual_recovery_visible',w.inspector.mission.recovery.isVisible());w.grab().save(str(root/f'{args.case}_recovery.png'))
    if kind=='final_validation':check('validator_matches_ui',w.mission.validation.get('accepted')==d.get('accepted'))
    if args.case=='stop-llm' and kind=='llm_request':QTimer.singleShot(700,stop)
    if args.case=='stop-motion' and kind=='state' and d.get('skill')=='MOVE_TO_TARGET' and not w.stop_requested:QTimer.singleShot(250,stop)

@guarded
def stop():
    if w.stop_requested:return
    if not w.focused:w.toggle_focus()
    check('stop_visible_in_focus',w.bar.stop_button.isVisible());before=time.monotonic();QTest.mouseClick(w.bar.stop_button,Qt.MouseButton.LeftButton)
    report['stop_dispatch_s']=time.monotonic()-before;check('stop_priority',w.worker.cancel_requested.is_set() and w.worker.sim.stopped)

@guarded
def tick():
    now=time.monotonic();gap=now-phase['last_tick'];phase['last_tick']=now
    if phase['run'] and not phase['result']:
        report['max_gui_timer_gap_s']=max(report['max_gui_timer_gap_s'],gap)
        if w.last_frame!=phase['last_frame']:
            if phase['llm']:report['camera_frames_during_llm']+=1
            if phase['motion']:report['camera_frames_during_motion']+=1
            phase['last_frame']=w.last_frame
        w.scene.change(w.scene.state.orbit(2,0))
        if w.world.get('objects') and not phase['selected'] and w.scene.anchors():
            obj,pos,_=w.scene.anchors()[0];QTest.mouseClick(w.scene,Qt.MouseButton.LeftButton,pos=pos.toPoint());phase['selected']=True
            check('world_object_selection',w.scene.selected_id is not None and w.inspector.currentIndex()==1)
            before=w.scene.state.lookat;QTest.mouseClick(w.inspector.object.focus,Qt.MouseButton.LeftButton)
            check('world_object_focus',np.allclose(w.scene.state.lookat,obj['position']))
            w.select_object(None)
    if now-phase['started']>390 and not phase['result']:report['errors'].append('GUI test timed out');w.close()

@guarded
def done(result):
    phase['result']=True;report['result']=result
    check('task_result_matches_ui',w.mission.result==result and not w.busy)
    if args.case in {'stop-llm','stop-motion'}:check('user_stop_result',result.get('metrics',{}).get('failure')=='USER_STOP')
    if args.case=='failure':check('unsupported_task_failure',result['success'] is False)
    if result['success']:check('completion_requires_validator',w.mission.complete)
    if args.case in {'blue','red','metric'}:
        check('camera_during_motion',report['camera_frames_during_motion']>2)
        if not args.offline:check('camera_during_real_llm_wait',report['camera_frames_during_llm']>2)
    w.scene.reset();QTimer.singleShot(500,finish)

@guarded
def finish():
    w.grab().save(str(root/f'{args.case}_result.png'));w.inspector.tabs.setCurrentIndex(2);w.grab().save(str(root/f'{args.case}_events.png'))
    print(json.dumps(report,ensure_ascii=False),flush=True);w.close()

@guarded
def ready():
    if phase.get('reset_pending'):
        check('reset_recreated_simulation',w.worker.sim is not phase['old_sim'])
        check('reset_ready_for_new_task',w.initialized and w.start.isEnabled() and not w.busy)
        check('reset_cleared_previous_state',not w.mission.complete and not w.raw_events)
        w.command.setText('跳舞');w.mode.setCurrentIndex(w.mode.findData('offline'))
        QTest.keyClick(w.command,Qt.Key.Key_Return);check('new_task_after_reset',w.busy)
        w.open_settings();button=w.dialogs[-1].findChild(QPushButton,'reset_scene')
        check('active_task_reset_disabled',not button.isEnabled());w.dialogs[-1].close()
    else:QTimer.singleShot(700,setup)

w.worker.ready.connect(ready);w.worker.trace.connect(trace);w.worker.finished_task.connect(done)
w.worker.fatal.connect(lambda text:report['errors'].append(text));w.renderer.failed.connect(lambda text:report['errors'].append(text))
timer=QTimer();timer.setInterval(150);timer.timeout.connect(tick);timer.start();app.exec()
report['checks']['clean_shutdown']=not w.worker.isRunning() and not w.renderer.isRunning()
(root/f'{args.case}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('REPORT',root/f'{args.case}.json',flush=True)
sys.exit(1 if report['errors'] or not phase['result'] else 0)
