"""Interactive spectator viewport; semantic picking uses WorldModel only."""
import math
import time
import numpy as np
from PySide6.QtCore import Qt,Signal,QRectF,QPointF,QEvent
from PySide6.QtGui import QPainter,QColor,QFont,QImage
from PySide6.QtWidgets import QWidget,QHBoxLayout,QPushButton,QComboBox,QLabel
from simulation.studio_presenter import object_name
from .camera import CameraState,project
from . import theme

class SceneToolbar(QWidget):
    preset=Signal(str);reset=Signal();zoom=Signal(float);focus_mode=Signal()
    def __init__(self):
        super().__init__();self.setFixedHeight(43);l=QHBoxLayout(self);l.setContentsMargins(4,0,4,0);l.setSpacing(8)
        title=QLabel('实时工作场景');title.setObjectName('stage_title');l.addWidget(title)
        source=QLabel('LIVE · MUJOCO');source.setObjectName('stage_source');l.addWidget(source)
        l.addStretch()
        self.views=QComboBox();self.views.addItems(['Perspective','Top','Front','Left','Right','Custom'])
        self.views.setMinimumWidth(132)
        self.views.activated.connect(lambda i:self.preset.emit(self.views.itemText(i)))
        l.addWidget(self.views)
        for text,action in [('−',lambda:self.zoom.emit(-100)),('+',lambda:self.zoom.emit(100)),('视角',self.reset.emit),('专注',self.focus_mode.emit)]:
            b=QPushButton(text);b.setObjectName('scene_tool');b.clicked.connect(action);l.addWidget(b)
    def custom(self):
        self.views.blockSignals(True);self.views.setCurrentText('Custom');self.views.blockSignals(False)

class OrientationGizmo:
    def __init__(self):self.points={}
    def paint(self,p,state,width,height):
        a=math.radians(state.azimuth);e=math.radians(state.elevation)
        origin=np.array([width-51.,height-49.])
        right=np.array([-math.sin(a),math.cos(a),0.]);up=np.array([-math.cos(a)*math.sin(e),-math.sin(a)*math.sin(e),math.cos(e)])
        for axis,name,col in [(np.eye(3)[0],'X',theme.TEXT_SECONDARY),(np.eye(3)[1],'Y',theme.ACCENT),(np.eye(3)[2],'Z',theme.INFO)]:
            end=origin+np.array([axis@right,-axis@up])*27
            self.points[name]=QPointF(*end)
            p.setPen(QColor(col));p.drawLine(QPointF(*origin),QPointF(*end));p.drawText(QPointF(end[0]+4,end[1]-4),name)
    def hit(self,pos):return next((k for k,v in self.points.items() if (pos-v).manhattanLength()<17),None)

class OrbitScene(QWidget):
    camera_changed=Signal(object);selected=Signal(object);focus_mode=Signal();escape_requested=Signal();edge=Signal(str)
    def __init__(self):
        super().__init__();self.state=CameraState();self.render_state=self.state
        self.image=QImage();self.projection=None;self.world={};self.selected_id=None;self.hover_id=None
        self.anchor=None;self.origin=None;self.panning=False;self.moved=False;self.gizmo=OrientationGizmo()
        self.setMinimumSize(200,200);self.setFocusPolicy(Qt.FocusPolicy.StrongFocus);self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
    def set_frame(self,packet):
        self.image=packet['image'];self.projection=packet['projection'];self.render_state=packet['camera'];self.update()
    def set_world(self,world):
        self.world=world
        if self.selected_id not in {o['id'] for o in world.get('objects',[])}:self.selected_id=None;self.selected.emit(None)
        self.update()
    def change(self,state):self.state=state;self.camera_changed.emit(state)
    def reset(self):self.change(CameraState())
    def zoom(self,delta):self.change(self.state.zoom(delta))
    def preset(self,name):self.change(CameraState.preset(name))
    def focus_object(self,obj):self.change(self.state.focus(obj['position']))
    def image_rect(self):
        if self.image.isNull():return QRectF(self.rect())
        size=self.image.size().scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio)
        return QRectF((self.width()-size.width())/2,(self.height()-size.height())/2,size.width(),size.height())
    def anchors(self):
        rect=self.image_rect();rows=[]
        for obj in self.world.get('objects',[]):
            if not obj.get('visible'):continue
            p=project(obj['position'],self.projection)
            if p:rows.append((obj,QPointF(rect.x()+p[0]*rect.width(),rect.y()+p[1]*rect.height()),p[2]))
        return rows
    def pick(self,pos):
        hits=[((point-pos).manhattanLength(),depth,obj) for obj,point,depth in self.anchors() if (point-pos).manhattanLength()<35]
        return min(hits,key=lambda x:(x[0],x[1]))[2] if hits else None
    def mousePressEvent(self,e):
        self.setFocus();self.anchor=e.position();self.origin=e.position();self.moved=False
        self.panning=e.button()!=Qt.MouseButton.LeftButton or bool(e.modifiers()&Qt.KeyboardModifier.ShiftModifier)
        self.setCursor(Qt.CursorShape.ClosedHandCursor);e.accept()
    def mouseMoveEvent(self,e):
        pos=e.position()
        if self.anchor is None:
            obj=self.pick(pos);self.hover_id=obj['id'] if obj else None;self.update()
            if pos.x()>self.width()-16:self.edge.emit('inspector')
            if pos.y()>self.height()-16:self.edge.emit('command')
            return
        delta=pos-self.anchor;self.anchor=pos
        if (pos-self.origin).manhattanLength()>4:self.moved=True
        if self.moved:self.change(self.state.pan(delta.x(),delta.y(),self.height()) if self.panning else self.state.orbit(delta.x(),delta.y()))
        e.accept()
    def mouseReleaseEvent(self,e):
        if not self.moved and e.button()==Qt.MouseButton.LeftButton:
            axis=self.gizmo.hit(e.position())
            if axis:self.preset({'X':'Right','Y':'Front','Z':'Top'}[axis])
            else:
                obj=self.pick(e.position());self.selected_id=obj['id'] if obj else None;self.selected.emit(obj)
        self.anchor=None;self.setCursor(Qt.CursorShape.OpenHandCursor);self.update();e.accept()
    def wheelEvent(self,e):
        delta=e.pixelDelta().y() if not e.pixelDelta().isNull() else e.angleDelta().y()
        self.zoom(delta);e.accept()
    def event(self,e):
        if e.type()==QEvent.Type.NativeGesture and e.gestureType()==Qt.NativeGestureType.ZoomNativeGesture:
            self.zoom(e.value()*500);return True
        return super().event(e)
    def keyPressEvent(self,e):
        if e.key()==Qt.Key.Key_R:self.reset()
        elif e.key()==Qt.Key.Key_F:self.focus_mode.emit()
        elif e.key()==Qt.Key.Key_Escape:self.escape_requested.emit()
        else:super().keyPressEvent(e)
    def paintEvent(self,e):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,True)
        p.setRenderHint(QPainter.RenderHint.Antialiasing,True)
        p.fillRect(self.rect(),QColor(theme.SCENE_BG))
        if not self.image.isNull():p.drawImage(self.image_rect(),self.image)
        else:p.setPen(QColor(theme.TEXT_SECONDARY));p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,'正在连接')
        p.setFont(QFont('PingFang SC',11));p.setPen(QColor(theme.TEXT_MUTED))
        p.drawText(22,self.height()-22,'拖动旋转    滚轮缩放    Shift 平移')
        self.gizmo.paint(p,self.render_state,self.width(),self.height())
        for obj,point,_ in self.anchors():
            if obj['id'] not in {self.selected_id,self.hover_id}:continue
            selected=obj['id']==self.selected_id
            p.setPen(QColor(theme.ACCENT));p.drawEllipse(point,5,5)
            label=object_name(obj)+(f"  {obj['confidence']:.0%}" if selected else '')
            if time.monotonic()-obj.get('last_seen',0)>5:label+=' · 未更新'
            end=point+QPointF(15,-18);p.drawLine(point,end)
            box=QRectF(end.x(),end.y()-20,max(100,p.fontMetrics().horizontalAdvance(label)+16),25)
            box.moveLeft(min(max(0,box.x()),self.width()-box.width()));box.moveTop(max(0,box.y()))
            p.fillRect(box,QColor(theme.PANEL));p.drawText(box.adjusted(8,0,-2,0),Qt.AlignmentFlag.AlignVCenter,label)
        p.setPen(QColor(theme.BORDER));p.drawRect(self.rect().adjusted(0,0,-1,-1))
