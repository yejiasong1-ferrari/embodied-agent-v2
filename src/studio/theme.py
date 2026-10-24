BACKGROUND = '#080B0F'
SCENE_BG = '#080C10'
PANEL = '#11171D'
ELEVATED = '#1A2027'
BORDER = '#303942'
TEXT_PRIMARY = '#F1F5F7'
TEXT_SECONDARY = '#BBC5CC'
TEXT_MUTED = '#84919B'
ACCENT = '#A6CADB'
SUCCESS = '#D6E8DC'
WARNING = '#D6B66D'
FAILURE = '#E08A8A'
INFO = '#9BB0C4'
SPACE_1, SPACE_2, SPACE_3, SPACE_4, SPACE_5 = 4, 8, 12, 16, 24
RADIUS_SMALL, RADIUS_MEDIUM = 8, 14

STYLE = f'''
QWidget {{
  background:{BACKGROUND};
  color:{TEXT_PRIMARY};
  font-family:"PingFang SC",".AppleSystemUIFont";
  font-size:14px;
}}
QLabel {{ background:transparent; }}
QLabel#brand {{ font-size:15px; font-weight:600; color:{TEXT_PRIMARY}; }}
QLabel#product {{ color:{TEXT_MUTED}; font-size:12px; padding-left:12px; padding-right:12px; border-left:1px solid rgba(255,255,255,0.13); }}
QLabel#title {{ font-size:19px; font-weight:600; letter-spacing:0px; }}
QLabel#muted {{ color:{TEXT_SECONDARY}; font-size:12px; line-height:1.4; }}
QLabel#section {{ color:{TEXT_MUTED}; font-size:11px; font-weight:600; }}
QLabel#flow_title {{ color:{TEXT_PRIMARY}; font-size:17px; font-weight:600; }}
QLabel#flow_action {{ color:{TEXT_PRIMARY}; font-size:13px; font-weight:500; }}
QLabel#backend {{ color:{TEXT_MUTED}; font-size:11px; }}
QLabel#stage_title {{ color:{TEXT_PRIMARY}; font-size:15px; font-weight:600; }}
QLabel#stage_source {{ color:{TEXT_MUTED}; font-size:10px; letter-spacing:1px; padding-left:6px; }}
QLabel#scene_meta {{ color:{TEXT_SECONDARY}; font-size:11px; }}
QLabel#mono, QPlainTextEdit {{ font-family:Menlo,monospace; font-size:11px; }}
QLabel#warning {{ color:{WARNING}; }}
QLabel#failure {{ color:{FAILURE}; }}
QLabel#success {{ color:{TEXT_PRIMARY}; }}
QLabel#status {{
  padding:5px 11px;
  border-radius:11px;
  background:rgba(166,202,219,0.07);
  border:1px solid rgba(166,202,219,0.18);
  color:{TEXT_SECONDARY};
  font-size:11px;
}}
QLabel#status[tone="success"] {{ color:{SUCCESS}; border-color:rgba(214,232,220,0.28); }}
QLabel#status[tone="warning"] {{ color:{WARNING}; }}
QLabel#status[tone="failure"] {{ color:{FAILURE}; }}
QWidget#topbar {{
  background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 rgba(27,34,41,0.96),stop:1 rgba(10,14,18,0.98));
  border-bottom:1px solid rgba(255,255,255,0.09);
}}
QWidget#stage {{ background:rgba(8,11,15,0.98); }}
QWidget#viewport {{ border:1px solid rgba(255,255,255,0.11); }}
QFrame#scene_footer {{ background:rgba(18,25,31,0.84); border:1px solid rgba(255,255,255,0.08); border-radius:8px; }}
QFrame#inspector, QStackedWidget#inspector {{
  background:rgba(13,18,23,0.97);
  border-left:1px solid rgba(255,255,255,0.10);
}}
QFrame#live_flow {{
  background:rgba(29,37,44,0.82);
  border-bottom:1px solid rgba(255,255,255,0.12);
}}
QLabel#phase {{
  color:{TEXT_MUTED};
  background:rgba(255,255,255,0.035);
  border:1px solid rgba(255,255,255,0.07);
  border-radius:7px;
  padding:5px 0;
  font-size:11px;
}}
QLabel#phase[phase_state="seen"] {{ color:{TEXT_SECONDARY}; border-color:rgba(255,255,255,0.15); }}
QLabel#phase[phase_state="active"] {{ color:{ACCENT}; background:rgba(166,202,219,0.10); border-color:rgba(166,202,219,0.35); font-weight:600; }}
QTabWidget {{
  background:rgba(12,17,21,0.97);
}}
QFrame#line {{ background:rgba(255,255,255,0.08); max-height:1px; }}
QFrame#command {{
  background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 rgba(28,35,42,0.97),stop:1 rgba(10,14,18,0.98));
  border-top:1px solid rgba(255,255,255,0.11);
}}
QFrame#recovery {{
  background:rgba(214,182,109,0.08);
  border:1px solid rgba(214,182,109,0.28);
  border-radius:12px;
}}
QPushButton {{
  background:rgba(255,255,255,0.045);
  border:1px solid rgba(255,255,255,0.11);
  border-radius:9px;
  padding:8px 14px;
  color:{TEXT_SECONDARY};
  font-size:13px;
}}
QPushButton:hover {{
  background:rgba(255,255,255,0.08);
  color:{TEXT_PRIMARY};
  border-color:rgba(255,255,255,0.18);
}}
QPushButton:checked {{ color:{TEXT_PRIMARY}; border-color:rgba(255,255,255,0.28); }}
QPushButton#quiet {{
  background:transparent;
  border:0;
  color:{TEXT_SECONDARY};
  padding:6px 10px;
}}
QPushButton#quiet:hover {{ color:{TEXT_PRIMARY}; background:transparent; }}
QPushButton#quiet:checked {{ color:{TEXT_PRIMARY}; }}
QPushButton#scene_tool {{ padding:7px 11px; min-width:18px; }}
QPushButton#run {{
  background:{TEXT_PRIMARY};
  color:#111214;
  border:0;
  font-weight:600;
  padding:8px 20px;
}}
QPushButton#run:hover {{ background:#FFFFFF; color:#111214; }}
QPushButton#stop {{
  background:rgba(224,138,138,0.10);
  color:{FAILURE};
  border:1px solid rgba(224,138,138,0.28);
}}
QPushButton:disabled, QPushButton#run:disabled, QPushButton#stop:disabled {{
  background:rgba(255,255,255,0.03);
  color:{TEXT_MUTED};
  border:1px solid rgba(255,255,255,0.06);
}}
QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget, QTableWidget {{
  background:rgba(255,255,255,0.035);
  border:1px solid rgba(255,255,255,0.08);
  border-radius:10px;
  selection-background-color:rgba(255,255,255,0.14);
  selection-color:{TEXT_PRIMARY};
}}
QLineEdit {{ padding:10px 14px; font-size:15px; }}
QLineEdit:focus {{ border-color:rgba(166,202,219,0.48); }}
QComboBox {{
  background:rgba(255,255,255,0.04);
  border:1px solid rgba(255,255,255,0.10);
  border-radius:10px;
  padding:8px 12px;
  color:{TEXT_SECONDARY};
}}
QComboBox::drop-down {{ border:0; width:22px; }}
QComboBox QAbstractItemView {{
  background:{PANEL};
  color:{TEXT_PRIMARY};
  border:1px solid rgba(255,255,255,0.10);
  selection-background-color:rgba(255,255,255,0.10);
}}
QTabWidget::pane {{ border:0; background:transparent; }}
QTabBar::tab {{
  background:transparent;
  color:{TEXT_SECONDARY};
  padding:13px 13px 11px 13px;
  border:0;
  border-bottom:1px solid transparent;
  font-size:12px;
}}
QTabBar::tab:selected {{
  color:{TEXT_PRIMARY};
  border-bottom:2px solid {ACCENT};
}}
QScrollArea {{ border:0; background:transparent; }}
QScrollBar:vertical {{ background:transparent; width:8px; margin:8px 0; }}
QScrollBar::handle:vertical {{
  background:rgba(255,255,255,0.16);
  border-radius:4px;
  min-height:28px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QListWidget::item {{
  padding:10px 8px;
  border:0;
  border-bottom:1px solid rgba(255,255,255,0.06);
}}
QListWidget::item:selected {{
  background:rgba(255,255,255,0.08);
  color:{TEXT_PRIMARY};
}}
QHeaderView::section {{
  background:transparent;
  color:{TEXT_SECONDARY};
  border:0;
  border-bottom:1px solid rgba(255,255,255,0.08);
  padding:8px;
}}
QSplitter::handle {{ background:rgba(255,255,255,0.06); width:1px; }}
QToolTip {{
  background:{ELEVATED};
  color:{TEXT_PRIMARY};
  border:1px solid rgba(255,255,255,0.12);
  padding:6px 8px;
}}
QCheckBox {{ background:transparent; spacing:8px; }}
'''
