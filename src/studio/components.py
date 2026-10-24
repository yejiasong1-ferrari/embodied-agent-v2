"""Reusable workspace widgets. Inputs are immutable agent evidence only."""
import html
import json
from pathlib import Path
import time
from datetime import datetime
from PySide6.QtCore import Qt,Signal,QTimer,QUrl
from PySide6.QtGui import QColor,QPixmap,QImage,QPainter
from PySide6.QtWidgets import (QWidget,QLabel,QVBoxLayout,QHBoxLayout,QPushButton,QFrame,
    QScrollArea,QListWidget,QListWidgetItem,QTextBrowser,QPlainTextEdit,QTabWidget,
    QComboBox,QLineEdit,QStackedWidget,QSizePolicy,QDialog,QGraphicsView,QGraphicsScene,QGridLayout,QSpacerItem)
from simulation.studio_presenter import SKILLS,STATES,FAILURES,ACTIONS,human_event,object_name
from . import theme


def label(value='',name=None,wrap=True):
    w=QLabel(value);w.setWordWrap(wrap)
    if name:w.setObjectName(name)
    w.setSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Fixed)
    return w

def divider():
    w=QFrame();w.setObjectName('line');w.setFixedHeight(1);return w

def scroll(widget):
    s=QScrollArea();s.setWidgetResizable(True);s.setWidget(widget);return s

class StatusCapsule(QLabel):
    NAMES={'IDLE':'就绪','UNDERSTAND':'理解','OBSERVE':'观察','PLAN':'规划','EXECUTE':'执行',
           'MONITOR':'监测','VERIFY':'验收','RECOVER':'恢复','REPLAN':'恢复',
           'SUCCESS':'完成','FAILED':'失败','STOPPING':'停止中','STOPPED':'已停止','CONNECTING':'连接中'}
    def __init__(self):
        super().__init__();self.setObjectName('status');self.setMaximumHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Fixed);self.set_state('CONNECTING')
    def set_state(self,state):
        self.setText(('✓ ' if state=='SUCCESS' else '● ')+self.NAMES.get(state,state))
        self.setProperty('tone','success' if state=='SUCCESS' else 'failure' if state=='FAILED' else 'warning' if state in {'RECOVER','REPLAN','STOPPING','STOPPED'} else 'info')
        self.style().unpolish(self);self.style().polish(self)

class LiveFlowPanel(QFrame):
    """Persistent summary made only from task events and observed state."""
    PHASES=(('UNDERSTAND','理解'),('OBSERVE','观察'),('PLAN','规划'),
            ('EXECUTE','执行'),('RECOVER','恢复'),('VERIFY','验收'))
    def __init__(self):
        super().__init__();self.setObjectName('live_flow')
        layout=QVBoxLayout(self);layout.setContentsMargins(20,17,20,16);layout.setSpacing(9)
        layout.addWidget(label('当前任务  /  LIVE MISSION','section'))
        self.title=label('等待指令','flow_title');self.title.setMaximumHeight(58);layout.addWidget(self.title)
        row=QHBoxLayout();row.setSpacing(8);self.status=StatusCapsule();row.addWidget(self.status)
        self.backend=label('语言来源待确认','backend',False);self.backend.setMaximumWidth(166)
        row.addWidget(self.backend);row.addStretch();layout.addLayout(row)
        phases=QGridLayout();phases.setContentsMargins(0,5,0,4);phases.setHorizontalSpacing(6);phases.setVerticalSpacing(6)
        self.phase_labels={}
        for i,(state,name) in enumerate(self.PHASES):
            phase=label(name,'phase',False);phase.setAlignment(Qt.AlignmentFlag.AlignCenter)
            phase.setProperty('phase_state','idle');phases.addWidget(phase,i//3,i%3)
            self.phase_labels[state]=phase
        layout.addLayout(phases)
        self.action=label('等待任务','flow_action');layout.addWidget(self.action)
        self.latest=label('执行后显示真实事件','muted');self.latest.setMaximumHeight(38);layout.addWidget(self.latest)

    def update_state(self,state,command):
        self.title.setText(command or '等待指令')
        shown='VERIFY' if state.state=='SUCCESS' and not state.complete else state.state
        if state.result.get('metrics',{}).get('failure')=='USER_STOP':shown='STOPPED'
        self.status.set_state(shown)
        backend=state.backend if state.backend!='Unavailable' else '语言来源待确认'
        self.backend.setText(backend);self.backend.setToolTip(backend)
        active='RECOVER' if state.state=='REPLAN' else 'VERIFY' if state.state=='SUCCESS' else state.state
        for key,phase in self.phase_labels.items():
            value='active' if key==active else 'seen' if key in state.seen_states else 'idle'
            if phase.property('phase_state')!=value:
                phase.setProperty('phase_state',value);phase.style().unpolish(phase);phase.style().polish(phase)
        skill=state.plan[state.current] if state.current is not None and state.current<len(state.plan) else None
        self.action.setText('当前动作  ·  '+SKILLS.get(skill,(STATES.get(state.state,'等待任务'),))[0])

    def add_event(self,event,allow_success=True):
        if event.get('type')=='context':return
        human=human_event(event)
        if event.get('type')=='state' and event.get('data',{}).get('state')=='SUCCESS' and not allow_success:
            human=('warning','执行器已返回，正在等待最终视觉验收。')
        if human:self.latest.setText(human[1])

class MissionTimeline(QListWidget):
    def __init__(self):
        super().__init__();self.setMaximumHeight(300);self.setMinimumHeight(100)
    def update_state(self,state):
        self.clear()
        for i,skill in enumerate(state.plan):
            completed=i in state.completed;active=i==state.current and not completed
            prefix='✓' if completed else '●' if active else '○'
            item=QListWidgetItem(f'{prefix}  {SKILLS.get(skill,(skill,))[0]}')
            item.setForeground(QColor(theme.ACCENT if completed else theme.WARNING if active and state.recovering else theme.TEXT_PRIMARY if active else theme.TEXT_MUTED))
            self.addItem(item)
        if state.current is not None:self.scrollToItem(self.item(state.current))
        self.setVisible(bool(state.plan))

class MissionPanel(QWidget):
    def __init__(self):
        super().__init__();        l=QVBoxLayout(self);l.setContentsMargins(22,24,22,20);l.setSpacing(16)
        self.title=label('等待指令','title');l.addWidget(self.title)
        self.status=StatusCapsule();l.addWidget(self.status)
        self.message=label('先观察场景，再规划并执行。','muted');l.addWidget(self.message)
        self.timeline=MissionTimeline();self.timeline.hide();l.addWidget(self.timeline)
        l.addWidget(divider());l.addWidget(label('当前动作','section'))
        self.skill=label('等待任务');l.addWidget(self.skill)
        self.target=label('');self.target.hide();l.addWidget(self.target)
        self.grasp=label('','muted');self.grasp.hide();l.addWidget(self.grasp)
        self.precision=label('','muted');self.precision.hide();l.addWidget(self.precision)
        self.distance=label('','muted');self.distance.hide();l.addWidget(self.distance)
        self.recovery=QFrame();self.recovery.setObjectName('recovery');rl=QVBoxLayout(self.recovery);rl.setSpacing(8)
        self.recovery_title=label('恢复','warning');rl.addWidget(self.recovery_title)
        self.recovery_rows=label();rl.addWidget(self.recovery_rows);l.addWidget(self.recovery);self.recovery.hide()
        self.verification=label();l.addWidget(self.verification);self.verification.hide()
        self.details_button=QPushButton('详情');self.details_button.setObjectName('quiet');self.details_button.setCheckable(True)
        self.details_button.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Fixed)
        self.details=QPlainTextEdit();self.details.setReadOnly(True);self.details.setMaximumHeight(160);self.details.hide()
        self.details_button.toggled.connect(self.details.setVisible);l.addWidget(self.details_button);l.addWidget(self.details)
        l.addStretch()
    def update_state(self,state,command):
        self.title.setText(command or '等待指令')
        shown=state.state
        if shown=='SUCCESS' and not state.complete:shown='VERIFY'
        if shown=='FAILED' and state.result.get('metrics',{}).get('failure')=='USER_STOP':shown='STOPPED'
        self.status.set_state(shown);self.timeline.update_state(state)
        name=state.plan[state.current] if state.current is not None else None
        self.skill.setText(SKILLS.get(name,(name or '等待任务',))[0])
        source=state.context.get('source');candidate=state.context.get('candidate')
        self.target.setVisible(bool(source))
        if source:self.target.setText(f"目标  {object_name(source)}\n置信度  {source['confidence']:.0%}")
        self.grasp.setVisible(bool(candidate))
        if candidate:self.grasp.setText(f"{'侧向抓取' if candidate['type']=='side_grasp' else '顶部抓取'} · 计划夹爪宽度 {candidate['width']*1000:.1f} mm")
        lock=state.localization.get(source.get('id')) if source else None
        self.precision.setVisible(bool(lock))
        if lock:self.precision.setText(f"{'精定位已锁定' if lock['state']=='PRECISE_LOCKED' else '粗定位，等待近距离确认'} · 估计误差 {lock['uncertainty_m']*1000:.1f} mm")
        rows=state.spatial.get('constraints',[])
        distances=[f"目标间距 {r['requested_gap_m']*100:.1f} cm · 实测 {r['actual_gap_m']*100:.1f} cm" for r in rows if r.get('requested_gap_m') is not None and r.get('actual_gap_m') is not None]
        self.distance.setVisible(bool(distances));self.distance.setText('\n'.join(distances))
        self.recovery.setVisible(bool(state.recoveries) or state.recovering)
        self.recovery_title.setText('恢复完成' if state.recovery_resolved else '恢复')
        self.recovery_rows.setText('\n\n'.join(f"尝试 {r.get('attempt',i+1):02d} · {r.get('candidate') or '尚无抓取候选'}\n{FAILURES.get(r['failure'],r['failure'])}\n→ {ACTIONS.get(r.get('action'),r.get('action','正在分析'))}" for i,r in enumerate(state.recoveries)))
        if state.recovering and not state.recoveries:self.recovery_rows.setText('正在分析失败证据…')
        self.verification.setVisible(bool(state.validation))
        if state.validation:
            ok=state.validation.get('accepted') is True
            self.verification.setText('重新观察：验收通过。\n位置关系、释放状态与稳定性符合条件。' if ok else '验收尚未通过，查看事件中的证据。')
        self.message.setText('验收完成。' if state.complete else FAILURES.get(state.result.get('metrics',{}).get('failure'),'任务未完成，可在事件里查看原因。') if state.result else '先观察场景，再规划并执行。' if state.state=='IDLE' else SKILLS.get(name,('', '状态来自本次执行。'))[1])
        self.details.setPlainText(json.dumps({'language_backend':state.backend,'llm_calls':state.llm_calls,
            'source':source,'candidate':candidate,'localization':lock,'grasp_strategy':state.grasp_strategy,
            'spatial':state.spatial,'corrections':state.corrections,'validation':state.validation,'result':state.result},ensure_ascii=False,indent=2))

class EventItem(QFrame):
    def __init__(self,event,tone,text):
        super().__init__();self.record=event
        l=QVBoxLayout(self);l.setContentsMargins(0,8,0,9);l.setSpacing(5)
        stamp=datetime.fromtimestamp(event['wall_time']).strftime('%H:%M:%S') if event.get('wall_time') else f"+{event.get('elapsed_s',0):.1f}s"
        l.addWidget(label(stamp,'section'))
        title=label(text);title.setObjectName({'error':'failure','warning':'warning','success':'success'}.get(tone,''));l.addWidget(title)
        self.expand=QPushButton('查看事件证据');self.expand.setObjectName('quiet');self.expand.setCheckable(True);l.addWidget(self.expand)
        self.detail=QPlainTextEdit();self.detail.setReadOnly(True);self.detail.setMaximumHeight(180);self.detail.hide()
        self.detail.setPlainText(json.dumps(event,ensure_ascii=False,indent=2));l.addWidget(self.detail)
        self.expand.toggled.connect(self.detail.setVisible);l.addWidget(divider())
    def developer(self,enabled):
        self.expand.setVisible(enabled)
        if not enabled:self.expand.setChecked(False)

class EventsPanel(QWidget):
    def __init__(self):
        super().__init__();self.items=[];self.developer_mode=False;self.all_events=[]
        l=QVBoxLayout(self);l.setContentsMargins(16,12,16,12)
        self.filter=QComboBox();self.filter.addItems(['所有事件','异常与恢复']);self.filter.currentIndexChanged.connect(self.apply_filter);l.addWidget(self.filter)
        self.content=QWidget();self.rows=QVBoxLayout(self.content);self.rows.setContentsMargins(0,0,0,0);self.rows.setSpacing(0)
        self.scroll=scroll(self.content);l.addWidget(self.scroll,1)
        self.new=QPushButton('有新事件 ↓');self.new.hide();self.new.clicked.connect(self.to_bottom);l.addWidget(self.new)
        self.empty=label('执行任务后，这里显示真实事件。','muted');self.rows.addWidget(self.empty)
        self.empty_spacer=QSpacerItem(0,0,QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Expanding)
        self.rows.addItem(self.empty_spacer)
        self.follow=True
        self.scroll.verticalScrollBar().valueChanged.connect(self.scrolled)
        self.scroll.verticalScrollBar().rangeChanged.connect(lambda *_:QTimer.singleShot(0,self.follow_latest))
    def scrolled(self,value):
        s=self.scroll.verticalScrollBar();self.follow=value>=s.maximum()-15
    def follow_latest(self):
        if self.follow:self.to_bottom()
    def to_bottom(self):
        self.follow=True;self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum());self.new.hide()
    def clear(self):
        for item,_,_ in self.items:self.rows.removeWidget(item);item.deleteLater()
        self.items=[];self.all_events=[];self.empty.show();self.new.hide();self.follow=True
        if self.empty_spacer is None:
            self.empty_spacer=QSpacerItem(0,0,QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Expanding)
            self.rows.addItem(self.empty_spacer)
    def developer(self,enabled):
        self.developer_mode=enabled
        for item,_,_ in self.items:item.developer(enabled)
    def add_event(self,event,allow_success=True):
        if event['type']=='context':return
        self.all_events.append(event)
        human=human_event(event)
        if event['type']=='state' and event['data'].get('state')=='SUCCESS' and not allow_success:
            human=('warning','收到完成状态，等待明确的 Validator 验收证据。')
        if event['type']=='final_validation' and event['data'].get('accepted') is False:
            human=('warning','最终视觉验收未通过，尚不能确认任务完成。')
        if not human:return
        tone,text=human;follow=self.follow;self.empty.hide()
        if self.empty_spacer is not None:
            self.rows.removeItem(self.empty_spacer);self.empty_spacer=None
        item=EventItem(event,tone,text);item.developer(self.developer_mode)
        self.rows.addWidget(item);self.items.append((item,tone,event))
        if len(self.items)>200:
            old,_,_=self.items.pop(0);self.rows.removeWidget(old);old.deleteLater()
        self.apply_filter()
        if follow:QTimer.singleShot(0,self.to_bottom)
        else:self.new.show()
    def apply_filter(self):
        for item,tone,event in self.items:item.setVisible(self.filter.currentIndex()==0 or tone in {'warning','error'} or event['type']=='recovery')

class VisionViewer(QGraphicsView):
    def __init__(self,image,title,parent=None):
        super().__init__(parent);self.setWindowTitle(title);self.setScene(QGraphicsScene(self));self.item=self.scene().addPixmap(QPixmap.fromImage(image))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag);self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.resize(960,720);self.setWindowFlag(Qt.WindowType.Window);self.level=1.
        QTimer.singleShot(0,lambda:self.fitInView(self.item,Qt.AspectRatioMode.KeepAspectRatio))
    def wheelEvent(self,e):
        f=1.15 if e.angleDelta().y()>0 else 1/1.15
        if .2<self.level*f<20:self.scale(f,f);self.level*=f
        e.accept()

class ImageTile(QWidget):
    clicked=Signal()
    def __init__(self):
        super().__init__();self.image=QImage();self.message='Unavailable';self.setMinimumHeight(160)
    def set_image(self,image):self.image=image;self.update()
    def mouseReleaseEvent(self,e):
        if not self.image.isNull():self.clicked.emit()
    def paintEvent(self,e):
        p=QPainter(self);p.fillRect(self.rect(),QColor(theme.SCENE_BG))
        if self.image.isNull():p.setPen(QColor(theme.TEXT_SECONDARY));p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,self.message)
        else:
            size=self.image.size().scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio)
            p.drawImage((self.width()-size.width())//2,(self.height()-size.height())//2,self.image.scaled(size,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))

class VisionPanel(QWidget):
    object_selected=Signal(object)
    def __init__(self):
        super().__init__();self.payload={};self.scene_packet={};self.world={};self.viewers=[]
        l=QVBoxLayout(self);l.setContentsMargins(20,18,20,18);l.setSpacing(14)
        self.mode=QComboBox();self.mode.addItems(['RGB','Detection','Depth','Scene']);self.mode.currentTextChanged.connect(self.refresh);l.addWidget(self.mode)
        self.tile=ImageTile();self.tile.clicked.connect(self.enlarge);l.addWidget(self.tile)
        self.age=label('等待观测','muted');l.addWidget(self.age)
        self.caption=label('','muted');l.addWidget(self.caption)
        l.addWidget(divider());l.addWidget(label('场景','section'))
        self.objects=QListWidget();self.objects.itemClicked.connect(lambda item:self.object_selected.emit(item.data(Qt.ItemDataRole.UserRole)));l.addWidget(self.objects,1)
        self.empty=label('等待观测','muted');l.addWidget(self.empty)
    def update_vision(self,payload):self.payload=payload;self.refresh()
    def update_world(self,world):
        self.world=world;self.objects.clear();objects=world.get('objects',[]);self.empty.setVisible(not objects)
        for obj in objects:
            item=QListWidgetItem(f"{'●' if obj['visible'] else '○'} {object_name(obj)}   {obj['confidence']:.0%}")
            item.setData(Qt.ItemDataRole.UserRole,obj);self.objects.addItem(item)
    def update_scene(self,packet):
        self.scene_packet=packet
        if self.mode.currentText()=='Scene':self.refresh()
    def refresh(self,*args):
        mode=self.mode.currentText()
        im=self.scene_packet.get('image',QImage()) if mode=='Scene' else self.payload.get(mode,QImage())
        if mode=='Detection' and not self.payload.get('revision'):im=QImage()
        self.tile.set_image(im)
        self.caption.setText('用户观察相机 · 不作为感知输入' if mode=='Scene' else self.payload.get('depth_range','Unavailable') if mode=='Depth' else '点击图像可缩放查看')
        self.tick()
    def tick(self):
        scene=self.mode.currentText()=='Scene';ts=self.scene_packet.get('rendered_at') if scene else self.payload.get('timestamp')
        if ts is None:self.age.setText('等待观测');return
        if self.payload.get('replay') and not scene:
            self.age.setText('历史观测 / REPLAY · 非实时画面');return
        if ts>time.monotonic()+1 and not scene:
            self.age.setText('画面滞后 · 观测时钟与当前会话不一致');return
        age=max(0,time.monotonic()-ts)
        self.age.setText(('画面滞后  ' if age>3 and not scene else '')+f"{'渲染' if scene else '观测'}于 {age:.1f} 秒前")
    def enlarge(self):
        if self.tile.image.isNull():return
        view=VisionViewer(self.tile.image,f"{self.mode.currentText()} · {self.age.text()} · 滚轮缩放 / 拖动平移",self)
        self.viewers.append(view);view.show()

class ObjectInspector(QWidget):
    focus_requested=Signal(dict);back=Signal()
    def __init__(self):
        super().__init__();self.object=None
        l=QVBoxLayout(self);l.setContentsMargins(18,18,18,18);l.setSpacing(12)
        b=QPushButton('← 返回任务');b.setObjectName('quiet');b.clicked.connect(self.back.emit);l.addWidget(b)
        l.addWidget(label('物体','section'));self.title=label('','title');l.addWidget(self.title)
        self.summary=label();l.addWidget(self.summary);l.addWidget(divider())
        self.position=label('','mono');l.addWidget(self.position)
        self.relations=label('','muted');l.addWidget(self.relations)
        self.focus=QPushButton('聚焦观察镜头');self.focus.clicked.connect(lambda:self.focus_requested.emit(self.object) if self.object else None);l.addWidget(self.focus);l.addStretch()
    def update_object(self,obj,world):
        self.object=obj;self.title.setText(object_name(obj));age=max(0,time.monotonic()-obj['last_seen'])
        self.summary.setText(f"{'可见' if obj['visible'] and age<5 else '历史观测'}\n置信度  {obj['confidence']:.0%}\n{'可移动' if obj['movable'] else '固定'}")
        self.position.setText('位置  m\n'+ '\n'.join(f'{axis}  {value:.3f}' for axis,value in zip('XYZ',obj['position'])))
        names={o['id']:object_name(o) for o in world.get('objects',[])}
        translations={'on':'位于上面','left_of':'在左侧','right_of':'在右侧','front_of':'在前面','behind':'在后面','near':'靠近','above':'在上方','below':'在下方','inside':'位于内部','overlapping':'重叠'}
        relations=[f"{translations.get(r['relation'],r['relation'])} · {names.get(r['target'],r['target'])}" for r in world.get('relationships',[]) if r['source']==obj['id']]
        self.relations.setText('关系\n'+('\n'.join(relations) if relations else '暂无'))

class InspectorPanel(QStackedWidget):
    focus_requested=Signal(dict);object_selected=Signal(object)
    def __init__(self):
        super().__init__();self.setMinimumWidth(300)
        self.tabs=QTabWidget();self.mission=MissionPanel();self.vision=VisionPanel();self.events=EventsPanel()
        self.tabs.addTab(scroll(self.mission),'任务详情');self.tabs.addTab(self.vision,'视觉');self.tabs.addTab(self.events,'事件')
        for widget in (self.mission.title,self.mission.status,self.mission.message):widget.hide()
        self.tabs.setCurrentIndex(2)
        self.live=LiveFlowPanel();page=QWidget();layout=QVBoxLayout(page)
        layout.setContentsMargins(0,0,0,0);layout.setSpacing(0);layout.addWidget(self.live);layout.addWidget(self.tabs,1)
        self.addWidget(page)
        self.object=ObjectInspector();self.addWidget(scroll(self.object));self.object.back.connect(lambda:self.select(None,{}))
        self.object.focus_requested.connect(self.focus_requested);self.vision.object_selected.connect(self.object_selected)
    def select(self,obj,world):
        if obj:self.object.update_object(obj,world);self.setCurrentIndex(1)
        else:self.setCurrentIndex(0)

class CommandBar(QFrame):
    run=Signal();stop=Signal()
    def __init__(self,instruction):
        super().__init__();self.setObjectName('command');self.setFixedHeight(88)
        l=QHBoxLayout(self);l.setContentsMargins(28,14,24,14);l.setSpacing(16)
        fields=QVBoxLayout();fields.setSpacing(4)
        self.hint=label('输入指令','muted');fields.addWidget(self.hint)
        self.command=QLineEdit(instruction);self.command.setPlaceholderText('例如：把红色杯子放到黄色盘子上');self.command.returnPressed.connect(self.run.emit);fields.addWidget(self.command)
        self.running=label('','title');self.running.hide();fields.addWidget(self.running);l.addLayout(fields,1)
        self.mode=QComboBox();self.mode.addItem('自动','auto');self.mode.addItem('在线','required');self.mode.addItem('本地','offline');l.addWidget(self.mode)
        self.start=QPushButton('执行');self.start.setObjectName('run');self.start.setMinimumHeight(40);self.start.setMinimumWidth(88);self.start.clicked.connect(self.run.emit);l.addWidget(self.start)
        self.stop_button=QPushButton('停止');self.stop_button.setObjectName('stop');self.stop_button.setMinimumHeight(40);self.stop_button.clicked.connect(self.stop.emit);self.stop_button.hide();l.addWidget(self.stop_button)
    def set_running(self,busy):
        self.command.setVisible(not busy);self.mode.setVisible(not busy);self.start.setVisible(not busy)
        self.running.setVisible(busy);self.running.setText(self.command.text());self.stop_button.setVisible(busy)
        self.hint.setText('正在执行' if busy else '输入指令')
    def focus_mode(self,enabled,busy):
        self.setFixedHeight(68 if enabled else 88)
        if enabled:
            self.command.hide();self.mode.hide();self.start.hide();self.running.show()
            self.running.setText(self.command.text() if busy else '移到底部可展开输入');self.hint.setText('专注')
        else:self.set_running(busy)
