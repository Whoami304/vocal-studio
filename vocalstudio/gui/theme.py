"""Тёмная тема в стиле студийного софта."""
from PyQt6.QtGui import QColor

BG = "#14161b"
PANEL = "#1c1f26"
PANEL2 = "#232731"
EDGE = "#2f3441"
TEXT = "#d7dbe6"
MUTED = "#7f8799"
ACCENT = "#2fd0c0"      # бирюзовый — автотюн/активное
ACCENT2 = "#ff9f43"     # оранжевый — скорректированный тон
REC = "#ff4d5e"
BLUE = "#6aa8ff"
GREEN = "#6be38a"
WARN = "#ffcc4d"


def c(hex_, a=255):
    col = QColor(hex_)
    col.setAlpha(a)
    return col


QSS = f"""
* {{ font-family: "Segoe UI", "Inter", "Noto Sans", sans-serif; font-size: 12px; color: {TEXT}; }}
QMainWindow, QDialog, QWidget#root {{ background: {BG}; }}
QWidget {{ background: transparent; }}
QDialog, QMessageBox, QInputDialog, QFileDialog {{ background: {BG}; }}
QToolTip {{ background: {PANEL2}; color: {TEXT}; border: 1px solid {EDGE}; padding: 4px 6px; }}

QFrame#panel {{ background: {PANEL}; border: 1px solid {EDGE}; border-radius: 8px; }}
QFrame#transport {{ background: {PANEL}; border-bottom: 1px solid {EDGE}; }}
QLabel#title {{ font-size: 11px; font-weight: 700; letter-spacing: 2px; color: {MUTED}; }}
QLabel#logo {{ font-size: 15px; font-weight: 800; letter-spacing: 3px; color: {ACCENT}; }}
QLabel#time {{ font-family: "Consolas", "DejaVu Sans Mono", monospace; font-size: 22px; font-weight: 600; color: {TEXT}; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#status {{ color: {MUTED}; padding-left: 6px; }}

QPushButton {{
    background: {PANEL2}; border: 1px solid {EDGE}; border-radius: 6px; padding: 6px 12px; color: {TEXT};
}}
QPushButton:hover {{ background: #2b3040; border-color: #3d4456; }}
QPushButton:pressed {{ background: #1a1d25; }}
QPushButton:disabled {{ color: #566; background: #1a1d24; }}
QPushButton:checked {{ background: #1f3f3b; border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton#rec {{ color: {REC}; font-weight: 700; }}
QPushButton#rec:checked {{ background: #4a1a20; border-color: {REC}; color: #fff; }}
QPushButton#play:checked {{ background: #1f3f3b; border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton#nav {{ background: transparent; border: none; border-bottom: 2px solid transparent;
    border-radius: 0; padding: 8px 16px; font-weight: 700; letter-spacing: 1px; color: {MUTED}; }}
QPushButton#nav:hover {{ color: {TEXT}; }}
QPushButton#nav:checked {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; background: transparent; }}
QPushButton#power {{ border-radius: 9px; padding: 0; min-width: 18px; max-width: 18px; min-height: 18px; max-height: 18px;
    background: #2a2f3b; border: 1px solid #3b4254; }}
QPushButton#power:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QPushButton#abtn:checked {{ background: #4a3a14; border-color: {WARN}; color: {WARN}; }}

QComboBox {{ background: {PANEL2}; border: 1px solid {EDGE}; border-radius: 6px; padding: 4px 8px; min-height: 20px; }}
QComboBox:hover {{ border-color: #3d4456; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {PANEL2}; border: 1px solid {EDGE}; selection-background-color: #2c5a57; outline: 0; }}

QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid #3b4254; background: #232731; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QSlider::groove:horizontal {{ height: 4px; background: #2a2f3b; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 12px; margin: -5px 0; border-radius: 6px; }}

QScrollBar:horizontal {{ background: {PANEL}; height: 12px; margin: 0; border-radius: 6px; }}
QScrollBar::handle:horizontal {{ background: #333a4b; min-width: 30px; border-radius: 6px; }}
QScrollBar::handle:horizontal:hover {{ background: #434c62; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar:vertical {{ background: {PANEL}; width: 12px; border-radius: 6px; }}
QScrollBar::handle:vertical {{ background: #333a4b; min-height: 30px; border-radius: 6px; }}

QProgressBar {{ background: #232731; border: 1px solid {EDGE}; border-radius: 4px; height: 8px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QMenu {{ background: {PANEL2}; border: 1px solid {EDGE}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: #2c5a57; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {EDGE}; }}
QGroupBox {{ border: 1px solid {EDGE}; border-radius: 6px; margin-top: 10px; padding-top: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: {MUTED}; }}
QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {PANEL2}; border: 1px solid {EDGE}; border-radius: 6px; padding: 4px 6px; }}

"""

_EXTRA = """
QFrame#trackheader { background: #181b23; border: none; border-right: 1px solid #2a2f3b; border-bottom: 1px solid #12151c; }
QFrame#trackheader[sel="true"] { background: #1d2a2c; border-left: 3px solid %(ACCENT)s; }
QLabel#trackname { font-size: 13px; font-weight: 700; color: %(TEXT)s; padding-left: 2px; }
QPushButton#mini, QPushButton#miniarm { background: #232731; border: 1px solid #333a4a; border-radius: 4px; padding: 0; font-size: 11px; font-weight: 700; color: #aeb6c8; }
QPushButton#mini:hover, QPushButton#miniarm:hover { background: #2d3342; }
QPushButton#mini:checked, QPushButton#miniarm:checked { background: #3a4660; color: #fff; border-color: #55648a; }
QPushButton#mini[has="true"] { color: %(ACCENT)s; border-color: #2a5f5a; }
QPushButton#miniarm:checked { background: #5a1c25; color: #ff6b7a; border-color: %(REC)s; }
QFrame#pluginrow { background: #20242e; border: 1px solid #2c3140; border-radius: 6px; }
QFrame#pluginrow:hover { border-color: #3d4456; }
QFrame#pluginrow[sel="true"] { background: #1f3433; border: 1px solid %(ACCENT)s; }
QFrame#pluginrow[off="true"] QLabel { color: #5c6478; }
QLabel#pname { font-size: 12px; font-weight: 700; }
QPushButton#primary2 { background: #1f3f3b; border: 1px solid %(ACCENT)s; color: %(ACCENT)s; font-weight: 700; padding: 8px 12px; }
QPushButton#primary2:hover { background: #26524c; }
QPushButton#magic { background: #3b2a5e; border: 1px solid #a98bff; color: #e4d9ff; font-weight: 700; padding: 8px 12px; }
QPushButton#magic:hover { background: #4a3578; }
QPushButton#magic:disabled { background: #1a1d24; border-color: #2a2f3b; color: #566; }
QPushButton#addtrack { background: transparent; border: 1px dashed #3a4152; color: #8b94a8; padding: 10px 16px; font-weight: 600; }
QPushButton#addtrack:hover { border-color: %(ACCENT)s; color: %(ACCENT)s; background: #172426; }
QFrame#banner { background: #1a2230; border: 1px solid #2c3b57; border-radius: 8px; }
QPushButton#step { background: transparent; border: 1px solid #333a4a; border-radius: 14px; padding: 5px 14px; color: #8b94a8; }
QPushButton#step:hover { border-color: #55648a; color: %(TEXT)s; }
QPushButton#step[cur="true"] { background: #1f3f3b; border-color: %(ACCENT)s; color: %(ACCENT)s; font-weight: 700; }
QPushButton#step[done="true"] { color: %(GREEN)s; border-color: #2c5f3b; }
QPushButton#close { background: transparent; border: none; color: #7f8799; padding: 2px 8px; }
QPushButton#close:hover { color: %(TEXT)s; }
QSplitter::handle { background: #14161b; }
QSplitter::handle:vertical { height: 6px; }
QMenuBar { background: %(PANEL)s; border-bottom: 1px solid %(EDGE)s; }
QMenuBar::item { padding: 5px 10px; background: transparent; }
QMenuBar::item:selected { background: #2b3040; border-radius: 4px; }
QTextEdit { background: #10131a; border: 1px solid %(EDGE)s; border-radius: 6px; }
QRadioButton { spacing: 8px; }
QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px; border: 1px solid #3b4254; background: #232731; }
QRadioButton::indicator:checked { background: %(ACCENT)s; border-color: %(ACCENT)s; }
""" % dict(ACCENT=ACCENT, TEXT=TEXT, REC=REC, GREEN=GREEN, PANEL=PANEL, EDGE=EDGE)
QSS += _EXTRA
