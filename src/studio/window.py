"""Production workspace composition. Only AgentWorker can execute a task."""
import json
import sys
import time
from pathlib import Path
from PySide6.QtCore import Qt,QTimer,QPropertyAnimation,QEasingCurve
from PySide6.QtGui import QDesktopServices,QFont,QShortcut,QKeySequence
from PySide6.QtWidgets import (QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QPushButton,
    QSplitter,QDialog,QCheckBox,QPlainTextEdit,QTabWidget,QGraphicsOpacityEffect,QFrame)
from . import theme
from .gateway import StudioGateway
from .bridge import MissionState
from .scene import OrbitScene,SceneToolbar
from .components import label,StatusCapsule,InspectorPanel,CommandBar
from .history import RunsDialog

class Studio(QMainWindow):
    def __init__(self,config,instruction=''):
        super().__init__();self.config=config;self.busy=False;self.initialized=False;self.closing=False
        self.focused=False;self.started_at=None;self.last_frame=0;self.last_packet={};self.world={};self.mission=MissionState()
        self.dialogs=[];self.result_path=None;self.developer_mode=False;self.raw_events=[];self.stop_requested=False
        self.setWindowTitle('V2 Studio · 真实机械臂工作区');self.resize(1440,900);self.setMinimumSize(1000,700);self.setStyleSheet(theme.STYLE)
        central=QWidget();self.setCentralWidget(central);layout=QVBoxLayout(central);layout.setContentsMargins(0,0,0,0);layout.setSpacing(0)
        self.top=QWidget();self.top.setObjectName('topbar');self.top.setFixedHeight(62);top=QHBoxLayout(self.top);top.setContentsMargins(24,0,24,0);top.setSpacing(10)
        brand=label('V2  STUDIO','brand');font=brand.font();font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing,1.2);font.setWeight(QFont.Weight.DemiBold);brand.setFont(font);top.addWidget(brand)
        self.product=label('机械臂工作区','product',False);top.addWidget(self.product)
        self.workspace=QPushButton('工作区');self.workspace.setObjectName('quiet');self.workspace.setCheckable(True);self.workspace.setChecked(True);top.addWidget(self.workspace)
        self.runs=QPushButton('记录');self.runs.setObjectName('quiet');self.runs.clicked.connect(self.open_runs);top.addWidget(self.runs);top.addStretch()
        self.status=StatusCapsule();top.addWidget(self.status);self.connection=label('正在连接','muted',False);top.addWidget(self.connection)
        self.fps=label('—','section',False);top.addWidget(self.fps)
        self.settings=QPushButton('设置');self.settings.setObjectName('quiet');self.settings.clicked.connect(self.open_settings);top.addWidget(self.settings);layout.addWidget(self.top)
        self.split=QSplitter();self.split.setChildrenCollapsible(False);layout.addWidget(self.split,1)
        scene_page=QWidget();scene_page.setObjectName('stage');sl=QVBoxLayout(scene_page);sl.setContentsMargins(16,12,12,12);sl.setSpacing(9)
        self.toolbar=SceneToolbar();sl.addWidget(self.toolbar);self.scene=OrbitScene();self.scene.setObjectName('viewport');sl.addWidget(self.scene,1)
        footer=QFrame();footer.setObjectName('scene_footer');fl=QHBoxLayout(footer);fl.setContentsMargins(12,3,12,3);fl.setSpacing(16)
        self.scene_action=label('等待任务','scene_meta',False);fl.addWidget(self.scene_action,1)
        self.frame_age=label('等待画面','scene_meta',False);fl.addWidget(self.frame_age);sl.addWidget(footer)
        self.split.addWidget(scene_page)
        self.inspector=InspectorPanel();self.inspector.setObjectName('inspector');self.inspector.setMaximumWidth(420);self.split.addWidget(self.inspector)
        self.split.setSizes([1090,350]);self.split.setStretchFactor(0,1);self.split.setStretchFactor(1,0)
        self.bar=CommandBar(instruction);layout.addWidget(self.bar);self.command=self.bar.command;self.mode=self.bar.mode;self.start=self.bar.start;self.start.setEnabled(False)
        self.gateway=StudioGateway(config)
        self.mailbox=self.gateway.mailbox;self.renderer=self.gateway.renderer;self.worker=self.gateway.worker
        self.worker.ready.connect(self.ready);self.worker.trace.connect(self.on_trace);self.worker.world.connect(self.on_world)
        self.worker.vision.connect(self.inspector.vision.update_vision);self.worker.finished_task.connect(self.done);self.worker.fatal.connect(self.on_fatal);self.renderer.failed.connect(self.on_fatal)
        self.bar.run.connect(self.run_task);self.bar.stop.connect(self.stop_task)
        self.scene.camera_changed.connect(self.camera_changed);self.scene.selected.connect(self.select_object);self.scene.focus_mode.connect(self.toggle_focus);self.scene.escape_requested.connect(self.escape_focus);self.scene.edge.connect(self.reveal_edge)
        self.toolbar.preset.connect(self.scene.preset);self.toolbar.reset.connect(self.scene.reset);self.toolbar.zoom.connect(self.scene.zoom);self.toolbar.focus_mode.connect(self.toggle_focus)
        self.inspector.object_selected.connect(self.select_object);self.inspector.focus_requested.connect(self.scene.focus_object)
        self.inspector.object.back.connect(lambda:self.select_object(None))
        self.fade_effect=QGraphicsOpacityEffect(self.inspector);self.inspector.setGraphicsEffect(self.fade_effect);self.fade_effect.setOpacity(1.)
        self.fade=QPropertyAnimation(self.fade_effect,b'opacity',self);self.fade.setDuration(180);self.fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.inspector.tabs.currentChanged.connect(self.fade_inspector)
        self.escape=QShortcut(QKeySequence(Qt.Key.Key_Escape),self);self.escape.activated.connect(self.escape_focus)
        self.poll=QTimer(self);self.poll.setInterval(33);self.poll.timeout.connect(self.render_tick);self.poll.start()
        self.clock=QTimer(self);self.clock.setInterval(250);self.clock.timeout.connect(self.tick);self.clock.start()
        self.edge_timer=QTimer(self);self.edge_timer.setSingleShot(True);self.edge_timer.timeout.connect(self.hide_edges)
        self.gateway.start();QTimer.singleShot(0,self.scene.setFocus)
    def ready(self):
        self.initialized=True;self.stop_requested=False;self.status.set_state('IDLE');self.connection.setText('MuJoCo 已连接');self.start.setEnabled(True)
        self.inspector.mission.update_state(self.mission,'');self.inspector.live.update_state(self.mission,'');self.update_reset_buttons()
    def camera_changed(self,state):
        self.gateway.camera(state)
        presets=[n for n in ['Perspective','Top','Front','Left','Right'] if state==type(state).preset(n)]
        self.toolbar.views.setCurrentText(presets[0] if presets else 'Custom')
    def render_tick(self):
        ratio=max(1.,min(2.,self.scene.devicePixelRatioF()))
        self.gateway.size(max(200,round(self.scene.width()*ratio)),max(200,round(self.scene.height()*ratio)))
        revision,packet=self.gateway.frame(self.last_frame)
        if packet is not None:
            self.last_frame=revision;self.last_packet=packet;self.scene.set_frame(packet);self.inspector.vision.update_scene(packet)
    def tick(self):
        self.inspector.vision.tick()
        if self.last_packet:
            age=max(0.,time.monotonic()-self.last_packet['rendered_at'])
            rate=self.last_packet['fps']
            self.fps.setText(f'{rate:.0f} FPS' if age<1 and rate>=1 else '实时画面' if age<1 else '静止')
            if age<3:
                self.frame_age.setText(f'观察画面 {age:.1f} 秒前')
            else:
                self.frame_age.setText('等待场景更新' if self.busy else '场景静止 · 可旋转观察')
        if self.busy and self.started_at:
            seconds=int(time.monotonic()-self.started_at);self.bar.hint.setText(('停止中' if self.stop_requested else '正在执行')+f'  {seconds//60:02d}:{seconds%60:02d}')
        if self.inspector.currentIndex()==1:
            obj=next((o for o in self.world.get('objects',[]) if o['id']==self.scene.selected_id),None)
            if obj:self.inspector.object.update_object(obj,self.world)
        if getattr(self,'dev',None) and self.dev.isVisible():self.refresh_developer()
    def run_task(self):
        if self.busy or not self.initialized or self.closing:return
        command=self.command.text().strip()
        if not command:self.command.setFocus();return
        self.mission=MissionState();self.raw_events=[];self.inspector.events.clear();self.result_path=None
        self.inspector.live.latest.setText('等待真实执行事件')
        self.started_at=time.monotonic();self.busy=True;self.stop_requested=False;self.bar.set_running(True)
        if self.focused:self.bar.focus_mode(True,True)
        self.status.set_state('UNDERSTAND');self.inspector.select(None,self.world);self.inspector.tabs.setCurrentIndex(2)
        self.inspector.mission.update_state(self.mission,command);self.inspector.live.update_state(self.mission,command);self.update_reset_buttons()
        self.scene_action.setText('任务已提交 · 等待执行事件')
        self.gateway.submit(command,self.mode.currentData())
    def stop_task(self):
        if not self.busy:return
        self.stop_requested=True;self.gateway.stop();self.status.set_state('STOPPING');self.inspector.live.status.set_state('STOPPING')
        self.inspector.mission.message.setText('已请求停止，控制器已停止；等待当前计算或模型请求返回。')
        self.bar.stop_button.setText('正在停止');self.bar.stop_button.setEnabled(False)
    def on_trace(self,event):
        self.mission.apply(event)
        if event['type']!='context':self.raw_events.append(event)
        shown=self.mission.state
        if shown=='SUCCESS' and not self.mission.complete:shown='VERIFY'
        self.status.set_state('STOPPING' if self.stop_requested else shown)
        self.inspector.events.add_event(event,allow_success=self.mission.complete)
        self.inspector.live.add_event(event,allow_success=self.mission.complete)
        self.inspector.mission.update_state(self.mission,self.command.text());self.inspector.live.update_state(self.mission,self.command.text())
        if self.stop_requested:self.inspector.live.status.set_state('STOPPING')
        if event['type']=='state' and event['data'].get('state') in {'RECOVER','REPLAN'}:
            self.inspector.tabs.setCurrentIndex(0)
        if event['type']=='state':self.scene_action.setText(self.inspector.live.action.text())
    def on_world(self,world):
        self.world=world;self.scene.set_world(world);self.inspector.vision.update_world(world)
        if self.scene.selected_id:
            obj=next((o for o in world.get('objects',[]) if o['id']==self.scene.selected_id),None)
            if obj:self.inspector.select(obj,world)
    def select_object(self,obj):
        self.scene.selected_id=obj['id'] if obj else None;self.scene.update();self.inspector.select(obj,self.world)
    def done(self,result):
        self.busy=False;self.mission.result=result
        self.mission.state='SUCCESS' if result.get('success') else 'FAILED'
        stopped=result.get('metrics',{}).get('failure')=='USER_STOP'
        self.status.set_state('STOPPED' if stopped else 'SUCCESS' if self.mission.complete else 'FAILED')
        self.bar.set_running(False);self.bar.stop_button.setEnabled(True);self.bar.stop_button.setText('停止')
        if self.focused:self.bar.focus_mode(True,False)
        self.inspector.mission.update_state(self.mission,self.command.text());self.inspector.live.update_state(self.mission,self.command.text());self.update_reset_buttons()
        self.scene_action.setText('任务完成 · 视觉验收通过' if self.mission.complete else '任务已停止' if stopped else '任务未完成 · 查看事件证据')
        self.result_path=Path(result['log_path']).resolve() if result.get('log_path') else None
        if result.get('success') and not self.mission.complete:self.inspector.mission.message.setText('执行器返回成功，但缺少明确验收事件；请检查证据。')
    def on_fatal(self,message):
        self.initialized=False;self.worker.stop();self.busy=False;self.bar.set_running(False);self.start.setEnabled(False);self.update_reset_buttons()
        self.status.set_state('FAILED');self.connection.setText('模拟已断开')
        self.inspector.mission.message.setText('运行服务异常，请查看事件详情并重新启动。')
        self.inspector.events.add_event({'type':'failure','elapsed_s':0,'data':{'skill':'runtime','failure':'RUNTIME_ERROR','evidence':message}})
        self.inspector.live.status.set_state('FAILED');self.inspector.live.latest.setText('运行服务异常，请查看事件详情。')
        self.raw_events.append({'type':'runtime_error','data':message});print(message,file=sys.stderr,flush=True)
    def fade_inspector(self,*args):
        self.fade.stop();self.fade.setStartValue(.7);self.fade.setEndValue(1.);self.fade.start()
    def toggle_focus(self):
        self.focused=not self.focused;self.inspector.setVisible(not self.focused)
        for w in [self.workspace,self.runs,self.connection,self.fps,self.settings,self.product]:w.setVisible(not self.focused)
        self.bar.focus_mode(self.focused,self.busy);self.scene.setFocus()
        if not self.focused:self.fade_inspector()
    def escape_focus(self):
        if self.focused:self.toggle_focus()
    def reveal_edge(self,which):
        if not self.focused:return
        if which=='inspector':self.inspector.show();self.split.setSizes([max(600,self.width()-320),320]);self.fade_inspector()
        else:self.bar.focus_mode(False,self.busy)
        self.edge_timer.start(1800)
    def hide_edges(self):
        if not self.focused:return
        if self.inspector.underMouse() or self.bar.underMouse() or self.command.hasFocus():self.edge_timer.start(1200);return
        self.inspector.hide();self.bar.focus_mode(True,self.busy)
    def open_runs(self):
        dialog=RunsDialog(self.config.log_root,self);self.dialogs.append(dialog);dialog.show()
    def open_settings(self):
        dialog=QDialog(self);dialog.setWindowTitle('Robot Studio');dialog.resize(460,380);l=QVBoxLayout(dialog);l.setContentsMargins(28,24,28,24);l.setSpacing(16)
        l.addWidget(label('设置','title'))
        l.addWidget(label(f'感知相机  {self.config.camera.mode} / RGB-D\n检测器  {self.config.perception.detector}\n语言模式  {self.mode.currentText()}','muted'))
        l.addWidget(label('复位视角只移动观察镜头。重建场景会重新创建仿真，任务进行时不可用。','muted'))
        view=QPushButton('复位视角');view.clicked.connect(self.scene.reset);l.addWidget(view)
        reset=QPushButton('重建场景');reset.setObjectName('reset_scene');reset.setEnabled(not self.busy and self.initialized);reset.clicked.connect(lambda:self.reset_scene(dialog));l.addWidget(reset)
        dev=QCheckBox('开发者数据');dev.setChecked(self.developer_mode);dev.toggled.connect(self.set_developer);l.addWidget(dev)
        close=QPushButton('关闭');close.clicked.connect(dialog.close);l.addWidget(close);self.dialogs.append(dialog);dialog.show()
    def update_reset_buttons(self):
        for d in self.dialogs:
            for button in d.findChildren(QPushButton,'reset_scene'):button.setEnabled(not self.busy and self.initialized)
    def reset_scene(self,dialog=None):
        if self.busy or not self.initialized:return
        self.initialized=False;self.start.setEnabled(False);self.mission=MissionState();self.inspector.events.clear();self.raw_events=[]
        self.status.set_state('CONNECTING');self.connection.setText('正在重建场景');self.scene.reset();self.gateway.reset();self.update_reset_buttons()
        self.inspector.live.latest.setText('正在重建场景');self.inspector.live.update_state(self.mission,'');self.scene_action.setText('等待任务')
        if dialog:dialog.close()
    def set_developer(self,enabled):
        self.developer_mode=enabled;self.inspector.events.developer(enabled)
        if enabled:
            if not getattr(self,'dev',None):
                self.dev=QDialog(self);self.dev.setWindowTitle('Robot Studio');self.dev.resize(840,650);l=QVBoxLayout(self.dev);l.setContentsMargins(24,20,24,20)
                l.addWidget(label('执行数据','title'));tabs=QTabWidget();l.addWidget(tabs);self.dev_text={}
                for name in ['WorldModel','Telemetry','Events / LLM']:
                    box=QPlainTextEdit();box.setReadOnly(True);tabs.addTab(box,name);self.dev_text[name]=box
            self.refresh_developer();self.dev.show()
        elif getattr(self,'dev',None):self.dev.hide()
    def refresh_developer(self):
        values={'WorldModel':self.world,'Telemetry':self.last_packet.get('telemetry',{}),'Events / LLM':self.raw_events[-150:]}
        for name,value in values.items():
            box=self.dev_text[name];text=json.dumps(value,ensure_ascii=False,indent=2)
            if box.toPlainText()!=text:
                scrollbar=box.verticalScrollBar();pos=scrollbar.value();box.setPlainText(text);scrollbar.setValue(pos)
    def closeEvent(self,event):
        self.closing=True;self.gateway.close()
        if self.gateway.is_running() or any(isinstance(d,RunsDialog) and any(t.isRunning() for t in d.loaders) for d in self.dialogs):
            self.start.setEnabled(False);self.status.set_state('STOPPING');self.bar.hint.setText('正在关闭 · 等待当前计算或请求返回')
            QTimer.singleShot(150,self.close);event.ignore()
        else:
            for d in self.dialogs:d.close()
            if getattr(self,'dev',None):self.dev.close()
            for viewer in self.inspector.vision.viewers:viewer.close()
            event.accept()
