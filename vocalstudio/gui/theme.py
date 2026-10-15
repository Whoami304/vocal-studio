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
