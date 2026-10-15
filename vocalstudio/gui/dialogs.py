"""Диалоги: настройки аудио, экспорт."""
from __future__ import annotations

import os
import subprocess
import sys

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                             QLabel, QPushButton, QVBoxLayout)

from .. import audio_io


def open_folder(path: str):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


class SettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки аудио")
        self.setMinimumWidth(520)
        self.settings = dict(settings)
        ins, outs, d_in, d_out = audio_io.list_devices()
        form = QFormLayout()
        self.cb_in = QComboBox()
        self.cb_out = QComboBox()
        self.cb_in.addItem("Системное по умолчанию", None)
        self.cb_out.addItem("Системное по умолчанию", None)
        for i, n in ins:
            self.cb_in.addItem(n, n)
        for i, n in outs:
            self.cb_out.addItem(n, n)
        self._ins = dict((n, i) for i, n in ins)
        self._outs = dict((n, i) for i, n in outs)
        for cb, key in ((self.cb_in, "in_dev"), (self.cb_out, "out_dev")):
            name = settings.get(key)
            if name:
                idx = cb.findData(name)
                if idx >= 0:
                    cb.setCurrentIndex(idx)
        form.addRow("Микрофон (вход):", self.cb_in)
        form.addRow("Наушники / колонки (выход):", self.cb_out)
        self.sp_lat = QDoubleSpinBox()
        self.sp_lat.setRange(-300.0, 500.0)
        self.sp_lat.setSuffix(" мс")
        self.sp_lat.setDecimals(1)
        self.sp_lat.setValue(float(settings.get("latency_ms", 0.0)))
        self.sp_lat.setToolTip("Дополнительная подстройка синхронизации записи с подложкой. "
                               "Если вокал слышится позже подложки — увеличьте значение.")
        form.addRow("Подстройка задержки записи:", self.sp_lat)
        self.cb_mon = QCheckBox("Слышать себя во время записи (мониторинг без эффектов)")
        self.cb_mon.setChecked(bool(settings.get("monitor", False)))
        form.addRow("", self.cb_mon)
        note = QLabel("Используйте наушники при включённом мониторинге, иначе возникнет обратная связь.\n"
                      "Мониторинг идёт без автотюна — эффекты применяются после записи.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(note)
        row = QHBoxLayout()
        b = QPushButton("Открыть папку с дублями")
        b.clicked.connect(lambda: open_folder(audio_io.takes_dir()))
        row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def result_settings(self) -> dict:
        s = dict(self.settings)
        s["in_dev"] = self.cb_in.currentData()
        s["out_dev"] = self.cb_out.currentData()
        s["in_idx"] = self._ins.get(s["in_dev"]) if s["in_dev"] else None
        s["out_idx"] = self._outs.get(s["out_dev"]) if s["out_dev"] else None
        s["latency_ms"] = self.sp_lat.value()
        s["monitor"] = self.cb_mon.isChecked()
        return s


class ExportDialog(QDialog):
    def __init__(self, has_backing: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Экспорт")
        self.setMinimumWidth(380)
        form = QFormLayout()
        self.cb_fmt = QComboBox()
        self.cb_fmt.addItems(list(audio_io.EXPORT_FORMATS.keys()))
        form.addRow("Формат:", self.cb_fmt)
        self.chk_backing = QCheckBox("Включить подложку (бит)")
        self.chk_backing.setChecked(has_backing)
        self.chk_backing.setEnabled(has_backing)
        form.addRow("", self.chk_backing)
        self.chk_norm = QCheckBox("Нормализовать пик до −1 дБFS")
        self.chk_norm.setChecked(False)
        form.addRow("", self.chk_norm)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
