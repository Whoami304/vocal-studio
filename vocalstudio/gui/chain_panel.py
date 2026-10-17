"""Строка плагина в цепочке (используется окном FX)."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout

from ..plugins import PLUGINS


class PluginRow(QFrame):
    clicked = pyqtSignal(str)
    toggled = pyqtSignal(str, bool)
    moved = pyqtSignal(str, int)
    removed = pyqtSignal(str)

    def __init__(self, plugin, index, selected, parent=None):
        super().__init__(parent)
        self.pid = plugin["id"]
        self.setObjectName("pluginrow")
        self.setProperty("sel", bool(selected))
        self.setProperty("off", not plugin["on"])
        self.setFixedHeight(50)
        d = PLUGINS[plugin["type"]]
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 6, 4)
        lay.setSpacing(6)
        num = QLabel(str(index + 1))
        num.setObjectName("muted")
        num.setFixedWidth(14)
        lay.addWidget(num)
        pw = QPushButton()
        pw.setObjectName("power")
        pw.setCheckable(True)
        pw.setChecked(bool(plugin["on"]))
        pw.setToolTip("Включить / выключить плагин")
        pw.setStatusTip("Включить / выключить плагин, чтобы сравнить звук с ним и без него")
        pw.toggled.connect(lambda on: self.toggled.emit(self.pid, bool(on)))
        lay.addWidget(pw)
        col = QVBoxLayout()
        col.setSpacing(0)
        name = QLabel(f"{d.icon}  {d.title}")
        name.setObjectName("pname")
        sub = QLabel(d.tagline)
        sub.setObjectName("muted")
        sub.setStyleSheet("font-size: 10px;")
        sub.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        col.addWidget(name)
        col.addWidget(sub)
        lay.addLayout(col, 1)
        for text, tip, fn in (("▲", "Переместить выше (раньше в цепочке)", lambda: self.moved.emit(self.pid, -1)),
                              ("▼", "Переместить ниже (позже в цепочке)", lambda: self.moved.emit(self.pid, 1)),
                              ("✕", "Удалить плагин", lambda: self.removed.emit(self.pid))):
            b = QPushButton(text)
            b.setObjectName("mini")
            b.setFixedSize(22, 22)
            b.setToolTip(tip)
            b.setStatusTip(tip)
            b.clicked.connect(fn)
            lay.addWidget(b)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.pid)
