"""Главное окно Vocal Studio."""
from __future__ import annotations

import os
import time

import numpy as np
from PyQt6.QtCore import QTimer, Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel,
                             QMainWindow, QMessageBox, QProgressBar, QPushButton, QStackedWidget, QVBoxLayout,
                             QWidget)
from scipy.signal import welch

from .. import audio_io
from ..audio_io import AudioEngine, SR, to_mono, to_stereo
from ..params import FACTORY_ORDER
from ..session import (Session, delete_user_preset, list_user_presets, load_settings, load_user_preset,
                       save_settings, save_user_preset)
from . import theme
from .dialogs import ExportDialog, SettingsDialog, open_folder
from .panels import Binder, DynamicsPage, EQPage, OutputPage, SpacePage, TunePage
from .pitch_view import PitchPanel
from .widgets import LevelMeter

APP_NAME = "Vocal Studio"
AUDIO_FILTER = "Аудио (*.wav *.flac *.mp3 *.ogg *.aiff *.aif *.m4a);;Все файлы (*.*)"


def fmt_time(t: float) -> str:
    t = max(0.0, t)
    m, s = divmod(t, 60.0)
    return f"{int(m):02d}:{s:05.2f}"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1380, 880)
        self.setMinimumSize(1120, 720)
        self.setAcceptDrops(True)

        self.settings = load_settings()
        try:
            gw, gh = self.settings.get("geometry", [1380, 880])
            self.resize(max(1120, int(gw)), max(720, int(gh)))
        except Exception:
            pass
        self.session = Session(self)
        self.engine = AudioEngine(SR)
        self._apply_device_settings()
        self.playhead = 0.0
        self.rec_start = 0
        self._rec_had_backing = False
        self._render_t0 = 0.0
        self._armed = False

        self._build_ui()
        self._connect()
        self._refresh_presets()
        self._load_last_preset()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        if not self.engine.available:
            self.statusBar().showMessage("Аудио-подсистема недоступна — запись и воспроизведение отключены, "
                                         "но обработка файлов работает.", 15000)
        else:
            self.statusBar().showMessage("Готово. Нажмите REC для записи или откройте аудиофайл.", 8000)
        self.session.request_render(immediate=True)

    # ------------------------------------------------------------- UI ----
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # --- верхняя панель
        top = QFrame()
        top.setObjectName("transport")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(14, 8, 14, 8)
        tl.setSpacing(8)
        logo = QLabel("VOCAL STUDIO")
        logo.setObjectName("logo")
        tl.addWidget(logo)
        tl.addSpacing(16)
        tl.addWidget(QLabel("Preset:"))
        self.cb_preset = QComboBox()
        self.cb_preset.setMinimumWidth(210)
        tl.addWidget(self.cb_preset)
        self.btn_save_preset = QPushButton("Save")
        self.btn_del_preset = QPushButton("Delete")
        for b in (self.btn_save_preset, self.btn_del_preset):
            b.setFixedHeight(28)
            tl.addWidget(b)
        tl.addStretch(1)
        self.lbl_files = QLabel("")
        self.lbl_files.setObjectName("muted")
        tl.addWidget(self.lbl_files)
        tl.addSpacing(12)
        self.btn_bypass = QPushButton("Original (A/B)")
        self.btn_bypass.setObjectName("abtn")
        self.btn_bypass.setCheckable(True)
        self.btn_bypass.setToolTip("Сравнить: слушать без обработки (Ctrl+B)")
        self.btn_undo = QPushButton("Undo take")
        self.btn_undo.setToolTip("Вернуть предыдущий дубль (Ctrl+Z)")
        self.btn_settings = QPushButton("Audio Settings")
        for b in (self.btn_bypass, self.btn_undo, self.btn_settings):
            b.setFixedHeight(28)
            tl.addWidget(b)
        v.addWidget(top)

        # --- транспорт
        tr = QFrame()
        tr.setObjectName("transport")
        trl = QHBoxLayout(tr)
        trl.setContentsMargins(14, 8, 14, 8)
        trl.setSpacing(8)
        self.btn_rec = QPushButton("●  REC")
        self.btn_rec.setObjectName("rec")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setToolTip("Запись с микрофона от текущей позиции (R). Если подложка загружена — она играет.")
        self.btn_play = QPushButton("▶  PLAY")
        self.btn_play.setObjectName("play")
        self.btn_play.setCheckable(True)
        self.btn_play.setToolTip("Воспроизвести / остановить (Пробел)")
        self.btn_stop = QPushButton("■  STOP")
        self.btn_home = QPushButton("|◀")
        self.btn_home.setToolTip("В начало (Home)")
        for b in (self.btn_rec, self.btn_play, self.btn_stop, self.btn_home):
            b.setFixedHeight(34)
            b.setMinimumWidth(92 if b is not self.btn_home else 44)
            trl.addWidget(b)
        self.lbl_time = QLabel("00:00.00")
        self.lbl_time.setObjectName("time")
        self.lbl_time.setMinimumWidth(120)
        trl.addSpacing(8)
        trl.addWidget(self.lbl_time)
        self.btn_loop = QPushButton("Loop")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setFixedHeight(28)
        trl.addWidget(self.btn_loop)
        trl.addSpacing(10)
        self.btn_arm = QPushButton("Mic Check")
        self.btn_arm.setCheckable(True)
        self.btn_arm.setFixedHeight(28)
        self.btn_arm.setToolTip("Включить вход микрофона, чтобы настроить уровень до записи")
        trl.addWidget(self.btn_arm)
        lab = QLabel("IN")
        lab.setObjectName("muted")
        trl.addWidget(lab)
        self.meter_in = LevelMeter()
        self.meter_in.setFixedWidth(130)
        self.meter_in.setFixedHeight(12)
        trl.addWidget(self.meter_in)
        lab = QLabel("OUT")
        lab.setObjectName("muted")
        trl.addWidget(lab)
        self.meter_out = LevelMeter()
        self.meter_out.setFixedWidth(130)
        self.meter_out.setFixedHeight(12)
        trl.addWidget(self.meter_out)
        trl.addStretch(1)
        self.btn_open = QPushButton("Open Vocal…")
        self.btn_backing = QPushButton("Load Backing…")
        self.btn_clear_backing = QPushButton("✕")
        self.btn_clear_backing.setFixedWidth(30)
        self.btn_clear_backing.setToolTip("Убрать подложку")
        self.btn_export = QPushButton("Export…")
        self.btn_export.setStyleSheet(f"QPushButton {{ background: #1f3f3b; border-color: {theme.ACCENT}; color: {theme.ACCENT}; font-weight: 700; }}")
        for b in (self.btn_open, self.btn_backing, self.btn_clear_backing, self.btn_export):
            b.setFixedHeight(30)
            if b is self.btn_backing:
                b.setMinimumWidth(118)
            if b is self.btn_open:
                b.setMinimumWidth(104)
            trl.addWidget(b)
        v.addWidget(tr)

        # --- график
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(12, 10, 12, 6)
        self.pitch = PitchPanel()
        ml.addWidget(self.pitch)
        v.addWidget(mid, 5)

        # --- навигация по страницам
        nav = QFrame()
        nav.setObjectName("transport")
        nl = QHBoxLayout(nav)
        nl.setContentsMargins(12, 0, 12, 0)
        nl.setSpacing(0)
        self.nav_btns = []
        self.stack = QStackedWidget()
        self.binder = Binder(self.session)
        self.page_tune = TunePage(self.binder)
        self.page_eq = EQPage(self.binder)
        self.page_dyn = DynamicsPage(self.binder)
        self.page_space = SpacePage(self.binder)
        self.page_out = OutputPage(self.binder)
        for name, page in (("AUTO-TUNE", self.page_tune), ("EQ", self.page_eq), ("DYNAMICS", self.page_dyn),
                           ("DELAY / REVERB", self.page_space), ("INPUT / OUTPUT", self.page_out)):
            b = QPushButton(name)
            b.setObjectName("nav")
            b.setCheckable(True)
            idx = self.stack.addWidget(page)
            b.clicked.connect(lambda _=False, i=idx: self._show_page(i))
            nl.addWidget(b)
            self.nav_btns.append(b)
        nl.addStretch(1)
        v.addWidget(nav)
        pages = QWidget()
        pl = QVBoxLayout(pages)
        pl.setContentsMargins(12, 10, 12, 10)
        pl.addWidget(self.stack)
        v.addWidget(pages, 4)
        self._show_page(0)

        # --- статусбар
        self.prog = QProgressBar()
        self.prog.setFixedWidth(160)
        self.prog.setFixedHeight(10)
        self.prog.setTextVisible(False)
        self.prog.setRange(0, 0)
        self.prog.hide()
        self.lbl_render = QLabel("")
        self.lbl_render.setObjectName("muted")
        self.statusBar().addPermanentWidget(self.lbl_render)
        self.statusBar().addPermanentWidget(self.prog)

    def _show_page(self, i):
        self.stack.setCurrentIndex(i)
        for k, b in enumerate(self.nav_btns):
            b.setChecked(k == i)

    # ------------------------------------------------------- подключение ----
    def _connect(self):
        s = self.session
        self.btn_rec.clicked.connect(self.toggle_record)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_home.clicked.connect(lambda: self.seek(0.0))
        self.btn_open.clicked.connect(self.open_vocal)
        self.btn_backing.clicked.connect(self.open_backing)
        self.btn_clear_backing.clicked.connect(lambda: s.set_backing(None, ""))
        self.btn_export.clicked.connect(self.export)
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_undo.clicked.connect(self.undo_take)
        self.btn_bypass.toggled.connect(s.set_bypass_all)
        self.btn_arm.toggled.connect(self.toggle_arm)
        self.btn_loop.toggled.connect(self._loop_toggled)
        self.cb_preset.activated.connect(self._preset_chosen)
        self.btn_save_preset.clicked.connect(self.save_preset)
        self.btn_del_preset.clicked.connect(self.delete_preset)

        s.render_started.connect(self._render_started)
        s.render_done.connect(self._render_done)
        s.render_failed.connect(self._render_failed)
        s.render_progress.connect(self._render_progress)
        s.take_changed.connect(self._take_changed)
        s.backing_changed.connect(self._backing_changed)
        s.params_replaced.connect(self._params_replaced)
        s.param_set.connect(self._param_set)

        pv = self.pitch.view
        pv.seek_requested.connect(self.seek)
        pv.overrides_changed.connect(s.set_overrides)

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.toggle_play)
        QShortcut(QKeySequence("R"), self, activated=self.toggle_record)
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, activated=lambda: self.seek(0.0))
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.open_vocal)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self.export)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo_take)
        QShortcut(QKeySequence("Ctrl+B"), self, activated=self.btn_bypass.toggle)

    # -------------------------------------------------------- устройства ----
    def _apply_device_settings(self):
        s = self.settings
        ins, outs, _, _ = audio_io.list_devices()
        self.engine.in_dev = None
        self.engine.out_dev = None
        for i, n in ins:
            if s.get("in_dev") == n:
                self.engine.in_dev = i
        for i, n in outs:
            if s.get("out_dev") == n:
                self.engine.out_dev = i
        self.engine.monitor = bool(s.get("monitor", False))

    def open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec():
            self.settings = dlg.result_settings()
            save_settings(self.settings)
            was_play = self.engine.mode == "play"
            self.stop_all()
            self._apply_device_settings()
            self.statusBar().showMessage("Настройки применены", 4000)

    # --------------------------------------------------------- пресеты ----
    def _refresh_presets(self, select: str | None = None):
        self.cb_preset.blockSignals(True)
        self.cb_preset.clear()
        for n in FACTORY_ORDER:
            self.cb_preset.addItem(n, ("f", n))
        users = list_user_presets()
        if users:
            self.cb_preset.insertSeparator(self.cb_preset.count())
            for n in users:
                self.cb_preset.addItem("★ " + n, ("u", n))
        if select:
            idx = self.cb_preset.findText(select)
            if idx >= 0:
                self.cb_preset.setCurrentIndex(idx)
        self.cb_preset.blockSignals(False)

    def _preset_chosen(self, i):
        data = self.cb_preset.itemData(i)
        if not data:
            return
        kind, name = data
        if kind == "f":
            self.session.load_preset(name)
        else:
            try:
                self.session.load_params_dict(load_user_preset(name))
            except Exception as e:
                QMessageBox.warning(self, APP_NAME, f"Не удалось загрузить пресет: {e}")
                return
        self.settings["last_preset"] = self.cb_preset.itemText(i)
        save_settings(self.settings)

    def _load_last_preset(self):
        last = self.settings.get("last_preset")
        if last:
            idx = self.cb_preset.findText(last)
            if idx >= 0:
                self.cb_preset.setCurrentIndex(idx)
                self._preset_chosen(idx)

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Сохранить пресет", "Название пресета:")
        if ok and name.strip():
            save_user_preset(name.strip(), self.session.params)
            self._refresh_presets("★ " + name.strip())
            self.statusBar().showMessage(f"Пресет «{name.strip()}» сохранён", 4000)

    def delete_preset(self):
        data = self.cb_preset.currentData()
        if data and data[0] == "u":
            if QMessageBox.question(self, APP_NAME, f"Удалить пресет «{data[1]}»?") == QMessageBox.StandardButton.Yes:
                delete_user_preset(data[1])
                self._refresh_presets()
        else:
            self.statusBar().showMessage("Встроенные пресеты удалить нельзя", 3000)

    # --------------------------------------------------------- файлы -------
    def open_vocal(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть вокал", self.settings.get("last_dir", ""), AUDIO_FILTER)
        if path:
            self.load_vocal(path)

    def load_vocal(self, path):
        try:
            self.stop_all()
            data = audio_io.load_audio(path, SR)
            mono = to_mono(data)
            self.session.set_take(mono, os.path.basename(path))
            self.settings["last_dir"] = os.path.dirname(path)
            self.seek(0.0)
            self.statusBar().showMessage(f"Загружен вокал: {os.path.basename(path)}", 5000)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Не удалось открыть файл:\n{e}")

    def open_backing(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить подложку", self.settings.get("last_dir", ""), AUDIO_FILTER)
        if path:
            self.load_backing(path)

    def load_backing(self, path):
        try:
            data = audio_io.load_audio(path, SR)
            self.session.set_backing(to_stereo(data), os.path.basename(path))
            self.settings["last_dir"] = os.path.dirname(path)
            self.statusBar().showMessage(f"Подложка: {os.path.basename(path)}", 5000)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Не удалось открыть файл:\n{e}")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.load_backing(urls[0])
            else:
                self.load_vocal(urls[0])

    def export(self):
        s = self.session
        if s.take is None and s.backing is None:
            self.statusBar().showMessage("Нечего экспортировать", 3000)
            return
        dlg = ExportDialog(s.backing is not None, self)
        if not dlg.exec():
            return
        fmt = dlg.cb_fmt.currentText()
        ext = audio_io.EXPORT_FORMATS[fmt][2]
        base = os.path.splitext(s.take_name or "vocal")[0] + "_processed"
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт", os.path.join(self.settings.get("last_dir", ""), base + ext),
                                              f"{fmt} (*{ext})")
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        self.stop_all()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            backing_keep = s.backing
            if not dlg.chk_backing.isChecked():
                s.backing = None
            try:
                res = s.render_blocking()
            finally:
                s.backing = backing_keep
            out = res.out.astype(np.float64)
            if dlg.chk_norm.isChecked() and out.size:
                pk = np.abs(out).max()
                if pk > 1e-6:
                    out = out * (10 ** (-1 / 20) / pk)
            audio_io.save_audio(path, out, SR, fmt)
            self.settings["last_dir"] = os.path.dirname(path)
            save_settings(self.settings)
            self.statusBar().showMessage(f"Сохранено: {path}", 8000)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Ошибка экспорта:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()
            s.request_render(immediate=True)

    # --------------------------------------------------- воспроизведение ----
    def seek(self, t: float):
        self.playhead = max(0.0, t)
        if self.engine.mode == "play":
            self.engine.pos = int(self.playhead * SR)
        self._update_time()
        self.pitch.view.set_playhead(self.playhead)
        self.page_dyn.set_playhead(self.playhead)

    def toggle_play(self):
        if self.engine.mode == "record":
            self.toggle_record()
            return
        if self.engine.mode == "play":
            self.stop_all()
        else:
            self.start_play()

    def start_play(self):
        if not self.engine.available:
            self._no_audio()
            self.btn_play.setChecked(False)
            return
        res = self.session.result
        if res is None or res.out.shape[1] == 0:
            self.statusBar().showMessage("Нечего воспроизводить — запишите вокал или откройте файл", 4000)
            self.btn_play.setChecked(False)
            return
        self._stop_arm()
        self.engine.set_buffer(res.out)
        pos = int(self.playhead * SR)
        if pos >= res.out.shape[1] - 10:
            pos = 0
            self.playhead = 0.0
        self._apply_loop()
        try:
            self.engine.start_play(pos)
        except Exception as e:
            self.btn_play.setChecked(False)
            QMessageBox.warning(self, APP_NAME, f"Не удалось открыть устройство вывода:\n{e}")
            return
        self.btn_play.setChecked(True)
        self.btn_play.setText("❚❚  PAUSE")

    def _apply_loop(self):
        res = self.session.result
        if self.btn_loop.isChecked() and res is not None and res.out.shape[1] > 0:
            self.engine.loop = (0, res.out.shape[1])
        else:
            self.engine.loop = None

    def _loop_toggled(self, on):
        self._apply_loop()

    def stop_all(self):
        if self.engine.mode == "record":
            self._finish_recording()
        elif self.engine.mode in ("play", "monitor"):
            self.engine.stop()
        self.btn_play.setChecked(False)
        self.btn_play.setText("▶  PLAY")
        if self.btn_arm.isChecked():
            self.btn_arm.blockSignals(True)
            self.btn_arm.setChecked(False)
            self.btn_arm.blockSignals(False)

    def _no_audio(self):
        QMessageBox.information(self, APP_NAME,
                                "Аудио-подсистема недоступна (PortAudio не загрузился).\n"
                                "Открытие файлов и экспорт работают, но запись/воспроизведение — нет.\n\n" +
                                str(audio_io.SD_ERROR or ""))

    # -------------------------------------------------------------- запись ----
    def toggle_arm(self, on):
        if on:
            if not self.engine.available:
                self._no_audio()
                self.btn_arm.setChecked(False)
                return
            if self.engine.mode in ("play", "record"):
                self.btn_arm.setChecked(False)
                return
            try:
                self.engine.monitor = bool(self.settings.get("monitor", False))
                self.engine.start_record(0, None, record=False)
            except Exception as e:
                self.btn_arm.setChecked(False)
                QMessageBox.warning(self, APP_NAME, f"Не удалось открыть микрофон:\n{e}")
        else:
            self._stop_arm()

    def _stop_arm(self):
        if self.engine.mode == "monitor":
            self.engine.stop()
        if self.btn_arm.isChecked():
            self.btn_arm.blockSignals(True)
            self.btn_arm.setChecked(False)
            self.btn_arm.blockSignals(False)

    def toggle_record(self):
        if self.engine.mode == "record":
            self._finish_recording()
            return
        if not self.engine.available:
            self._no_audio()
            self.btn_rec.setChecked(False)
            return
        if self.engine.mode in ("play", "monitor"):
            self.engine.stop()
            self.btn_play.setChecked(False)
            self.btn_play.setText("▶  PLAY")
            self.btn_arm.blockSignals(True)
            self.btn_arm.setChecked(False)
            self.btn_arm.blockSignals(False)
        s = self.session
        backing = None
        if s.backing is not None:
            g = 10 ** (s.params["master"]["backing_db"] / 20.0)
            backing = (s.backing * g).astype(np.float32)
        self.rec_start = int(self.playhead * SR)
        self._rec_had_backing = backing is not None
        try:
            self.engine.monitor = bool(self.settings.get("monitor", False))
            self.engine.start_record(self.rec_start, backing, record=True)
        except Exception as e:
            self.btn_rec.setChecked(False)
            QMessageBox.warning(self, APP_NAME, f"Не удалось начать запись:\n{e}")
            return
        self.btn_rec.setChecked(True)
        self.btn_rec.setText("■  STOP REC")
        self.pitch.view.follow = True
        if self.engine.warning:
            self.statusBar().showMessage(self.engine.warning, 8000)
        else:
            self.statusBar().showMessage("Идёт запись… нажмите STOP, чтобы закончить", 0)

    def _finish_recording(self):
        eng = self.engine
        rec, lat = eng.stop_record()
        self.btn_rec.setChecked(False)
        self.btn_rec.setText("●  REC")
        self.pitch.view.set_live(None, 0.0)
        if len(rec) < SR // 20:
            self.statusBar().showMessage("Запись слишком короткая — отброшена", 4000)
            return
        if self._rec_had_backing:
            comp = int(max(0.0, lat + float(self.settings.get("latency_ms", 0.0)) / 1000.0) * SR)
            if comp < len(rec):
                rec = rec[comp:]
        # автосохранение дубля
        try:
            fn = os.path.join(audio_io.takes_dir(), time.strftime("take_%Y%m%d_%H%M%S.wav"))
            audio_io.save_audio(fn, rec[None, :], SR, "WAV 24-bit")
            self.statusBar().showMessage(f"Дубль записан и сохранён: {fn}", 8000)
        except Exception:
            self.statusBar().showMessage("Дубль записан (не удалось сохранить файл)", 6000)
        s = self.session
        start = self.rec_start
        if s.take is None or start < int(0.05 * SR):
            new = rec
            start = 0
        else:
            old = s.take
            if len(old) < start:
                old = np.pad(old, (0, start - len(old)))
            tail = old[start + len(rec):] if len(old) > start + len(rec) else np.zeros(0, np.float32)
            new = np.concatenate([old[:start], rec, tail])
        s.set_take(new, time.strftime("take_%H%M%S"))
        self.seek(start / SR)

    def undo_take(self):
        if self.engine.mode == "record":
            return
        if self.session.undo_take():
            self.statusBar().showMessage("Предыдущий дубль восстановлен", 3000)

    # ------------------------------------------------------------ слоты ----
    def _take_changed(self):
        s = self.session
        pv = self.pitch.view
        pv.set_wave(s.take, SR)
        pv.set_overrides(s.overrides)
        pv.set_duration(s.total_samples / SR)
        if s.take is None:
            pv.set_result(None, None, SR)
            self.page_eq.view.set_spectrum(None, None)
        else:
            self._compute_spectrum(s.take)
        self.lbl_files.setText(self._files_label())
        self.btn_undo.setEnabled(s._undo_take is not None)

    def _files_label(self):
        s = self.session
        parts = []
        if s.take_name:
            parts.append(f"Vocal: {s.take_name}")
        if s.backing_name:
            parts.append(f"Backing: {s.backing_name}")
        return "   ".join(parts)

    def _backing_changed(self):
        self.lbl_files.setText(self._files_label())
        self.pitch.view.set_duration(self.session.total_samples / SR)

    def _compute_spectrum(self, x):
        try:
            seg = x[: SR * 90]
            f, pxx = welch(seg, fs=SR, nperseg=4096)
            db = 10 * np.log10(pxx + 1e-14)
            lf = np.geomspace(20, 20000, 160)
            out = np.interp(lf, f, db)
            out = np.convolve(out, np.ones(5) / 5, mode="same")
            out = (out - out.max() + 70.0) / 70.0
            self.page_eq.view.set_spectrum(lf, out)
        except Exception:
            pass

    def _params_replaced(self):
        self.binder.refresh()
        self.page_tune.refresh()
        self._sync_mask()

    def _param_set(self, module, key):
        if module == "tune" and key in ("transpose", "detune"):
            self._sync_mask()

    def _sync_mask(self):
        t = self.session.params["tune"]
        self.pitch.view.set_mask(t["notes"], t["transpose"] + t["detune"] / 100.0)

    def _render_started(self):
        self._render_t0 = time.time()
        self.prog.show()
        self.lbl_render.setText("Обработка…")

    def _render_progress(self, stage, f):
        names = {"pitch": "анализ тона", "tune": "автотюн", "comp": "компрессор", "reverb": "реверб", "delay": "дилей"}
        self.lbl_render.setText("Обработка: " + names.get(stage, stage))

    def _render_done(self, res):
        if not self.session._running or True:
            self.prog.hide()
            self.lbl_render.setText(f"Готово за {time.time() - self._render_t0:.1f} с")
        if self.engine.mode == "play":
            self.engine.set_buffer(res.out)
            self._apply_loop()
        pv = self.pitch.view
        self._sync_mask()
        pv.set_result(res.track, res.correction, SR)
        pv.set_overrides(self.session.overrides)
        pv.set_duration(max(self.session.total_samples, res.out.shape[1]) / SR)
        if not getattr(self, "_fitted", False) and res.track is not None and np.isfinite(res.correction.in_midi).any():
            pv.fit_vertical()
            pv.fit_horizontal()
            self._fitted = True
        self.page_dyn.set_result(res, SR)
        self.btn_undo.setEnabled(self.session._undo_take is not None)

    def _render_failed(self, msg):
        self.prog.hide()
        self.lbl_render.setText("")
        self.statusBar().showMessage("Ошибка обработки: " + msg, 10000)

    # --------------------------------------------------------- тики --------
    def _update_time(self):
        self.lbl_time.setText(fmt_time(self.playhead))

    def _tick(self):
        eng = self.engine
        mode = eng.mode
        if mode == "play":
            self.playhead = eng.pos / float(SR)
            if eng.finished:
                self.stop_all()
        elif mode == "record":
            self.playhead = eng.pos / float(eng.fs)
            arr, off = eng.live_tail(14.0)
            if len(arr):
                t0 = self.rec_start / SR + off / eng.rec_sr
                # живая волна рисуется на частоте rec_sr; масштаб времени считает PitchView через sr
                self.pitch.view.set_live(arr if eng.rec_sr == SR else audio_io.resample(arr, eng.rec_sr, SR), t0)
            if eng.in_peak >= 0.99:
                self.statusBar().showMessage("Клиппинг на входе! Уменьшите громкость микрофона.", 1500)
        if mode in ("play", "record", "monitor"):
            self.meter_in.set_level(eng.in_peak)
            self.meter_out.set_level(eng.out_peak)
            self._update_time()
            self.pitch.view.set_playhead(self.playhead)
            self.page_dyn.set_playhead(self.playhead)
        else:
            self.meter_in.set_level(0.0)
            self.meter_out.set_level(0.0)

    def closeEvent(self, e):
        try:
            self.engine.stop()
            self.settings["geometry"] = [self.width(), self.height()]
            save_settings(self.settings)
        finally:
            super().closeEvent(e)
