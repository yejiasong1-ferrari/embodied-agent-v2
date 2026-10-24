"""Read-only saved evidence browser. Never drives the simulator."""
import json
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import QThread,Signal,QUrl,Qt
from PySide6.QtGui import QImage,QDesktopServices
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QListWidget,QListWidgetItem,QTabWidget,QPlainTextEdit,QPushButton,QSplitter,QWidget
from .components import label,EventsPanel,ImageTile,VisionViewer

class RunLoader(QThread):
    loaded=Signal(object)
    def __init__(self,path,detail=False):super().__init__();self.path=Path(path);self.detail=detail
    def run(self):
        try:
            if self.detail:
                events=[]
                for line in (self.path/'events.jsonl').read_text().splitlines():
                    try:
                        event=json.loads(line)
                        if event['type'] not in {'trajectory','observation','visual_attachment','attachment_hypothesis'}:events.append(event)
                    except (ValueError,KeyError):continue
                result=json.loads((self.path/'result.json').read_text()) if (self.path/'result.json').exists() else {}
                self.loaded.emit({'path':str(self.path),'events':events,'result':result,
                                  'images':[str(p) for p in sorted((self.path/'images').glob('*.png'))]})
            else:
                rows=[]
                for p in sorted(self.path.glob('task_*'),reverse=True)[:100]:
                    try:
                        task=json.loads((p/'task.json').read_text())
                        result=json.loads((p/'result.json').read_text()) if (p/'result.json').exists() else {}
                        rows.append({'path':str(p),'task':task,'result':result})
                    except (OSError,ValueError):continue
                self.loaded.emit(rows)
        except Exception as exc:self.loaded.emit({'error':str(exc)})

class RunsDialog(QDialog):
    def __init__(self,root,parent):
        super().__init__(parent);self.setWindowTitle('Robot Studio · 历史回放 / REPLAY');self.resize(1040,740);self.loaders=[];self.viewers=[];self.run_path=None
        l=QVBoxLayout(self);l.setContentsMargins(24,20,24,20);l.setSpacing(12);l.addWidget(label('历史回放 / REPLAY','title'))
        l.addWidget(label('已保存的画面和事件，不会控制当前场景。','muted'))
        split=QSplitter();l.addWidget(split,1);self.runs=QListWidget();self.runs.setMaximumWidth(350);split.addWidget(self.runs)
        self.tabs=QTabWidget();split.addWidget(self.tabs);self.events=EventsPanel();self.tabs.addTab(self.events,'事件')
        page=QWidget();pl=QVBoxLayout(page);self.image=ImageTile();self.image.setMinimumHeight(280);pl.addWidget(self.image,1)
        self.images=QListWidget();self.images.setMaximumHeight(190);pl.addWidget(self.images);self.tabs.addTab(page,'图片')
        self.result=QPlainTextEdit();self.result.setReadOnly(True);self.tabs.addTab(self.result,'结果')
        self.note=label('正在读取本地任务…','muted');l.addWidget(self.note)
        actions=QHBoxLayout();actions.addStretch();b=QPushButton('打开证据目录');b.clicked.connect(self.open_path);actions.addWidget(b)
        close=QPushButton('返回实时工作区');close.clicked.connect(self.close);actions.addWidget(close);l.addLayout(actions)
        self.runs.itemClicked.connect(self.choose);self.images.itemClicked.connect(self.show_image);self.image.clicked.connect(self.enlarge)
        self.load(root,False,self.populate)
    def load(self,path,detail,callback):
        loader=RunLoader(path,detail);self.loaders.append(loader);loader.loaded.connect(callback);loader.start()
    def populate(self,rows):
        if isinstance(rows,dict):self.note.setText(rows.get('error','无法读取'));return
        self.note.setText(f'{len(rows)} 个最近任务')
        for row in rows:
            result=row['result'];status='已完成' if result.get('success') else '未完成' if result else '未结束 / 中断'
            duration=result.get('metrics',{}).get('execution_time_s');duration=f'{duration:.1f}s' if duration is not None else '—'
            try:stamp=datetime.strptime(Path(row['path']).name.split('_')[1],'%Y%m%dT%H%M%S').strftime('%Y-%m-%d %H:%M:%S UTC')
            except (ValueError,IndexError):stamp=Path(row['path']).name
            item=QListWidgetItem(f"{row['task']['user_command']}\n{status} · {duration}\n{stamp}")
            item.setData(Qt.ItemDataRole.UserRole,row['path']);self.runs.addItem(item)
    def choose(self,item):
        self.run_path=item.data(Qt.ItemDataRole.UserRole);self.note.setText('正在读取…')
        self.load(self.run_path,True,self.loaded)
    def loaded(self,row):
        if 'error' in row:self.note.setText(row['error']);return
        if row['path']!=self.run_path:return
        self.events.clear()
        for event in row['events']:self.events.add_event(event)
        self.result.setPlainText(json.dumps(row['result'],ensure_ascii=False,indent=2));self.images.clear();self.image.set_image(QImage())
        for path in row['images']:
            item=QListWidgetItem(Path(path).name);item.setData(Qt.ItemDataRole.UserRole,path);self.images.addItem(item)
        if self.images.count():self.images.setCurrentRow(self.images.count()-1);self.show_image(self.images.currentItem())
        self.note.setText('REPLAY · '+Path(row['path']).name)
    def show_image(self,item):self.image.set_image(QImage(item.data(Qt.ItemDataRole.UserRole)))
    def enlarge(self):
        if self.image.image.isNull():return
        viewer=VisionViewer(self.image.image,self.images.currentItem().text(),self);self.viewers.append(viewer);viewer.show()
    def open_path(self):
        if self.run_path:QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.run_path).resolve())))
    def closeEvent(self,event):
        if any(loader.isRunning() for loader in self.loaders):
            from PySide6.QtCore import QTimer
            QTimer.singleShot(100,self.close);event.ignore()
        else:event.accept()
