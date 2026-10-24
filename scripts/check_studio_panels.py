"""Replay actual saved evidence through presentation components; no robot commands."""
import json,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from simulation.qt_runtime import prepare_qt
prepare_qt()
from PySide6.QtCore import Qt,QPoint,QPointF,QTimer,QEventLoop
from PySide6.QtGui import QImage,QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout
import numpy as np
from studio.components import InspectorPanel
from studio.history import RunsDialog
from studio.runtime import depth_image
from studio.bridge import MissionState
from studio.theme import STYLE

root=Path('outputs/studio_checks');record=json.loads((root/'blue.json').read_text());path=Path(record['result']['log_path']);events=[json.loads(s) for s in (path/'events.jsonl').read_text().splitlines()]
app=QApplication([]);w=QWidget();w.setWindowTitle('V2 Studio · REPLAY component checks');w.setStyleSheet(STYLE);w.resize(440,800)
panel=InspectorPanel();QVBoxLayout(w).addWidget(panel);w.show();report={'source':str(path),'checks':{},'errors':[]}
def exception_hook(t,v,tb):report['errors'].append(''.join(traceback.format_exception(t,v,tb)))
sys.excepthook=exception_hook
def check(name,ok):
 report['checks'][name]=bool(ok)
 if not ok:report['errors'].append(name)

def wait(ms):
 loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()

def until(condition,limit=10000):
 end=time.monotonic()+limit/1000
 while not condition() and time.monotonic()<end:wait(50)
 return condition()

state=MissionState()
for e in events:
 state.apply(e)
 if e['type'] not in {'trajectory','observation','visual_attachment','attachment_hypothesis'}:panel.events.add_event(e,allow_success=state.complete)
panel.mission.update_state(state,json.loads((path/'task.json').read_text())['user_command']);wait(200)
check('real_replay_validator',state.complete)
check('recovery_history',len(state.recoveries)==sum(e['type']=='recovery' for e in events))
panel.tabs.setCurrentIndex(2);wait(200);bar=panel.events.scroll.verticalScrollBar();bar.setValue(0);wait(50)
print('SCROLL',bar.minimum(),bar.maximum(),bar.value(),panel.events.follow,panel.events.content.size().toTuple(),flush=True)
check('manual_scroll_disables_follow',not panel.events.follow)
panel.events.add_event(events[-1]);wait(100)
check('new_event_preserves_scroll',bar.value()==0)
# Use an existing human-readable event from the same recorded run to test append behavior.
panel.events.add_event(next(e for e in reversed(events) if e['type']=='state'));wait(100)
check('new_event_badge',panel.events.new.isVisible());QTest.mouseClick(panel.events.new,Qt.MouseButton.LeftButton);wait(80)
check('resume_follow',bar.value()==bar.maximum())
panel.events.add_event(next(e for e in reversed(events) if e['type']=='state'));wait(100)
check('append_follows_latest',bar.value()==bar.maximum())
panel.events.developer(True);item=panel.events.items[-1][0];item.expand.setChecked(True);check('developer_event_evidence',item.detail.isVisible());panel.events.developer(False);check('ordinary_mode_hides_raw',not item.detail.isVisible())
panel.events.filter.setCurrentIndex(1);wait(50);check('anomaly_filter',all(not item.isVisible() or tone in {'error','warning'} or event['type']=='recovery' for item,tone,event in panel.events.items))
last=sorted((path/'images').glob('*_rgb.png'))[-1];prefix=last.name.split('_')[0];npz=np.load(path/'images'/f'{prefix}_depth.npz')
world=json.loads(sorted((path/'world_states').glob('*.json'))[-1].read_text())
payload={'RGB':QImage(str(last)),'Detection':QImage(str(path/'images'/f'{prefix}_vision.png')),'Depth':depth_image(npz['depth']),'timestamp':float(npz['timestamp']),'revision':world['revision'],'depth_range':'0–3 m / recorded RGB-D'}
panel.tabs.setCurrentIndex(1);panel.vision.update_world(world);panel.vision.update_vision(payload)
for mode in ['RGB','Detection','Depth']:
 panel.vision.mode.setCurrentText(mode);check('actual_'+mode,not panel.vision.tile.image.isNull())
check('old_vision_stale',any(tag in panel.vision.age.text() for tag in ['STALE','画面滞后']));check('actual_object_list',panel.vision.objects.count()==len(world['objects']))
QTest.mouseClick(panel.vision.tile,Qt.MouseButton.LeftButton);wait(100);viewer=panel.vision.viewers[-1];before=viewer.transform().m11()
event=QWheelEvent(QPointF(100,100),QPointF(100,100),QPoint(),QPoint(0,120),Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
app.sendEvent(viewer.viewport(),event);check('image_zoom',viewer.transform().m11()>before);viewer.close()
panel.vision.mode.setCurrentText('Scene');check('missing_scene_unavailable',panel.vision.tile.image.isNull())
runs=RunsDialog('outputs/agent',w);runs.show();check('history_list_loaded',until(lambda:runs.runs.count()>0))
print('HISTORY',runs.note.text(),[(t.isRunning(),t.isFinished()) for t in runs.loaders],flush=True)
rows=[runs.runs.item(i) for i in range(runs.runs.count())]
row=next((item for item in rows if item.data(Qt.ItemDataRole.UserRole)==str(path)),None)
if row is None:
 row=next((item for item in rows if (Path(item.data(Qt.ItemDataRole.UserRole))/'result.json').exists()
           and any((Path(item.data(Qt.ItemDataRole.UserRole))/'images').glob('*.png'))),None)
check('history_has_recent_task',row is not None)
if row:
 history_path=Path(row.data(Qt.ItemDataRole.UserRole));report['history_source']=str(history_path)
 runs.choose(row);check('history_result_loaded',until(lambda:runs.images.count()>0))
 check('history_result_matches_disk',json.loads(runs.result.toPlainText())==json.loads((history_path/'result.json').read_text()))
 check('history_marked_replay','REPLAY' in runs.windowTitle() and 'REPLAY' in runs.note.text())
 runs.grab().save(str(root/'history_replay.png'));check('history_images_real',not runs.image.image.isNull())
for loader in runs.loaders:loader.wait(1000)
runs.close();w.close();app.processEvents()
(root/'panels.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False));sys.exit(bool(report['errors']))
