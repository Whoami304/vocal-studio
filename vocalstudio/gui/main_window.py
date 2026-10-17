"""Главное окно Vocal Studio: дорожки как в DAW, окно плагинов (FX), запись, правка аудио, экспорт."""
from __future__ import annotations

import copy
import dataclasses
import os
import threading
import time

import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QColorDialog, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout, QInputDialog,
                             QLabel, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QSizePolicy, QVBoxLayout,
                             QWidget)

from .. import audio_io
from ..audio_io import AudioEngine, AudioError, SR, to_mono, to_stereo
from ..project import Project
from ..session import MASTER, Session, config_dir, load_settings, save_settings
from . import safety, theme
from .dialogs import (AutoMixDialog, DiagnosticsDialog, ExportDialog, SettingsDialog, ShortcutsDialog, TourDialog,
                      WelcomeDialog)
from .fx_window import FxWindow
from .safety import guard
from .timeline import TimelineState, TracksPanel
from .widgets import LevelMeter

APP_NAME = "Vocal Studio"
AUDIO_FILTER = "Аудио (*.wav *.flac *.mp3 *.ogg *.aiff *.aif);;Все файлы (*.*)"
PROJECT_FILTER = "Проект Vocal Studio (*.vsp)"
AUTOSAVE_SECONDS = 90


def fmt_time(t: float) -> str:
    t = max(0.0, t)
    m, s = divmod(t, 60.0)
    return f"{int(m):02d}:{s:05.2f}"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.settings = load_settings()
        try:
            gw, gh = self.settings.get("geometry", [1440, 900])
            self.resize(max(1320, int(gw)), max(760, int(gh)))
        except Exception:
            self.resize(1440, 900)
        self.setMinimumSize(1320, 720)
        self.setAcceptDrops(True)

        self.session = Session(self)
        self.engine = AudioEngine(SR)
        self._apply_device_settings()
        self.state = TimelineState()
        self.playhead = 0.0
        self.rec_start = 0
        self.rec_tid = None
        self._rec_had_backing = False
        self._render_t0 = 0.0
        self._track_ids = ()
        self._played = False
        self._exported = False
        self._tick_errors = 0
        self.fx_windows: dict[str, FxWindow] = {}
        self._autosave_busy = False

        self._build_ui()
        self._build_menu()
        self._connect()
        self._ensure_default_track()
        self._structure_changed()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(AUTOSAVE_SECONDS * 1000)
        self.autosave_timer.timeout.connect(self._autosave)
        self.autosave_timer.start()

        if not self.engine.available:
            self.statusBar().showMessage("Аудио-подсистема недоступна — запись и воспроизведение отключены, "
                                         "но обработка файлов работает.", 15000)
        self._update_title()
        self._update_quickstart()
        self._update_edit_buttons()
        QTimer.singleShot(500, self._startup_dialogs)

    # ================================================================ UI ====
    def _btn(self, text, tip, slot=None, checkable=False, w=None, h=None, name=None) -> QPushButton:
        b = QPushButton(text)
        b.setToolTip(tip)
        b.setStatusTip(tip)
        b.setCheckable(checkable)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)        # пробел не должен «нажимать» кнопку
        if w:
            b.setFixedWidth(w)
        if h:
            b.setFixedHeight(h)
        if name:
            b.setObjectName(name)
        elif h and h <= 30:
            b.setStyleSheet("padding: 2px 8px;")
        if slot is not None:
            b.clicked.connect(lambda _=False: slot())
        return b

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # --- логотип/проект в углу меню
        corner = QWidget()
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 0, 12, 0)
        self.lbl_project = QLabel("")
        self.lbl_project.setStyleSheet("font-weight: 600;")
        cl.addWidget(self.lbl_project)
        corner.setMinimumWidth(220)
        self.menuBar().setCornerWidget(corner, Qt.Corner.TopRightCorner)

        # --- транспорт
        tr = QFrame()
        tr.setObjectName("transport")
        trl = QHBoxLayout(tr)
        trl.setContentsMargins(10, 8, 10, 8)
        trl.setSpacing(6)
        self.btn_rec = self._btn("●  REC", "Запись с микрофона на выбранную (красная REC) дорожку (R)", self.toggle_record,
                                 True, 92, 34, "rec")
        self.btn_rec.setStatusTip("REC: записать голос на дорожку с красной кнопкой REC. Бит и другие дорожки играют в наушниках.")
        self.btn_play = self._btn("▶  PLAY", "Воспроизвести / остановить (Пробел). Shift+Пробел — только выделение", self.toggle_play,
                                  True, 86, 34, "play")
        self.btn_stop = self._btn("■  STOP", "Остановить", self.stop_all, False, 76, 34)
        self.btn_home = self._btn("|◀", "В начало (Home)", lambda: self.seek(0.0), False, 38, 34)
        for b in (self.btn_rec, self.btn_play, self.btn_stop, self.btn_home):
            trl.addWidget(b)
        self.lbl_time = QLabel("00:00.00")
        self.lbl_time.setObjectName("time")
        self.lbl_time.setMinimumWidth(100)
        trl.addSpacing(6)
        trl.addWidget(self.lbl_time)
        self.btn_loop = self._btn("Loop", "Зациклить: выделение или весь проект (L)", self._apply_loop, True, None, 28)
        self.btn_loop.toggled.connect(lambda _: self._apply_loop())
        trl.addWidget(self.btn_loop)
        trl.addSpacing(6)
        self.sp_bpm = QDoubleSpinBox()
        self.sp_bpm.setRange(30, 300)
        self.sp_bpm.setDecimals(0)
        self.sp_bpm.setSuffix(" BPM")
        self.sp_bpm.setValue(self.session.project.bpm)
        self.sp_bpm.setFixedWidth(84)
        self.sp_bpm.setToolTip("Темп проекта: для метронома и синхронизации эха (Delay)")
        self.sp_bpm.setStatusTip("Темп проекта в ударах в минуту: используется метрономом и плагином Delay")
        self.sp_bpm.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.sp_bpm.valueChanged.connect(self._bpm_changed)
        trl.addWidget(self.sp_bpm)
        self.btn_metro = self._btn("Метроном", "Включить щелчки метронома при воспроизведении и записи", self._metro_changed, True, None, 28)
        self.btn_metro.toggled.connect(lambda _: self._metro_changed())
        self.btn_count = self._btn("Отсчёт", "Перед записью сыграть один такт «1-2-3-4»", None, True, None, 28)
        trl.addWidget(self.btn_metro)
        trl.addWidget(self.btn_count)
        trl.addStretch(1)
        for text, attr in (("IN", "meter_in"), ("OUT", "meter_out")):
            lab = QLabel(text)
            lab.setObjectName("muted")
            trl.addWidget(lab)
            m = LevelMeter()
            m.setFixedWidth(50)
            m.setFixedHeight(12)
            setattr(self, attr, m)
            trl.addWidget(m)
        self.btn_arm = self._btn("Тест микрофона", "Включить вход микрофона, чтобы выставить уровень до записи", self.toggle_arm,
                                 True, None, 28)
        self.btn_arm.toggled.connect(self.toggle_arm)
        self.btn_arm.setStatusTip("Тест микрофона: показывает уровень сигнала до записи. Пики не должны касаться красной зоны.")
        trl.addWidget(self.btn_arm)
        self.btn_bypass = self._btn("Original (A/B)", "Сравнить: слушать без всех плагинов (Ctrl+B)", None, True, None, 30, "abtn")
        self.btn_bypass.setMinimumWidth(100)
        self.btn_bypass.setStatusTip("Original (A/B): послушать запись без обработки, чтобы сравнить с результатом")
        trl.addWidget(self.btn_bypass)
        self.btn_settings = self._btn("Звук…", "Выбор микрофона и наушников, задержка записи", self.open_settings, False, None, 30)
        self.btn_settings.setMinimumWidth(60)
        trl.addWidget(self.btn_settings)
        self.btn_export = self._btn("Экспорт…", "Сохранить готовый микс в файл (Ctrl+E)", self.export, False, None, 30)
        self.btn_export.setStyleSheet(f"QPushButton {{ background: #1f3f3b; border-color: {theme.ACCENT}; color: {theme.ACCENT}; font-weight: 700; }}")
        self.btn_export.setMinimumWidth(92)
        trl.addWidget(self.btn_export)
        v.addWidget(tr)

        # --- панель правки
        eb = QFrame()
        eb.setObjectName("transport")
        el = QHBoxLayout(eb)
        el.setContentsMargins(14, 4, 14, 4)
        el.setSpacing(6)
        cap = QLabel("ПРАВКА")
        cap.setObjectName("title")
        el.addWidget(cap)
        self.edit_btns = {}

        def eb_btn(key, text, tip, slot, w=None):
            b = self._btn(text, tip, slot, False, w, 26)
            b.setStyleSheet("padding: 2px 7px; font-size: 11px;")
            self.edit_btns[key] = b
            el.addWidget(b)
            return b

        def sep():
            f = QFrame()
            f.setFrameShape(QFrame.Shape.VLine)
            f.setStyleSheet("color: #2f3441;")
            el.addWidget(f)

        eb_btn("undo", "↶ Отменить", "Отменить последнее действие (Ctrl+Z)", self.undo)
        eb_btn("redo", "↷ Повторить", "Вернуть отменённое действие (Ctrl+Y)", self.redo)
        sep()
        eb_btn("selall", "Всё", "Выделить всю запись на дорожке (Ctrl+A). Выделять можно и мышью: проведите по дорожке", lambda: self.edit("selectall"))
        eb_btn("copy", "Копировать", "Копировать выделенный участок (Ctrl+C)", lambda: self.edit("copy"))
        eb_btn("cut", "Вырезать", "Вырезать выделенное в буфер: на месте останется тишина (Ctrl+X)", lambda: self.edit("cut"))
        eb_btn("paste", "Вставить", "Вставить скопированное в позицию курсора (Ctrl+V)", lambda: self.edit("paste"))
        sep()
        eb_btn("delete", "Удалить", "Удалить выделенный участок — на его месте тишина, остальное не сдвигается (Delete)", lambda: self.edit("silence"))
        eb_btn("ripple", "Удалить со сдвигом", "Удалить участок и сдвинуть остальное, закрыв пустоту (Ctrl+Delete)", lambda: self.edit("delete"))
        eb_btn("trim", "Обрезать", "Оставить только выделенное, остальное отрезать (Ctrl+T)", lambda: self.edit("trim"))
        sep()
        eb_btn("fadein", "Fade In", "Плавное нарастание в начале выделения (или записи)", lambda: self.edit("fade_in"))
        eb_btn("fadeout", "Fade Out", "Плавное затухание в конце выделения (или записи)", lambda: self.edit("fade_out"))
        eb_btn("norm", "Нормализовать", "Довести выделение (или запись) до максимальной громкости без перегруза", lambda: self.edit("normalize"))
        sep()
        eb_btn("zoomsel", "Приблизить", "Приблизить выделенный участок на экране", self.zoom_selection)
        self.lbl_sel = QLabel("")
        self.lbl_sel.setObjectName("muted")
        self.lbl_sel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        el.addSpacing(8)
        el.addWidget(self.lbl_sel, 1)
        v.addWidget(eb)

        # --- баннер «Быстрый старт»
        self.banner = QFrame()
        self.banner.setObjectName("banner")
        bl = QHBoxLayout(self.banner)
        bl.setContentsMargins(12, 3, 8, 3)
        bl.setSpacing(8)
        t = QLabel("Быстрый старт:")
        t.setStyleSheet("font-weight: 700;")
        bl.addWidget(t)
        self.steps = []
        self.step_labels = ["①  Добавьте голос", "②  ✨ Auto-Mix", "③  Послушайте  ▶", "④  Сохраните"]
        tips = ["Запишите голос с микрофона или откройте готовый файл", "Программа сама настроит автотюн, эквалайзер, компрессор и эффекты",
                "Нажмите Play и сравните с кнопкой Original", "Экспортируйте готовый трек в WAV или MP3"]
        for i, (tx, tp) in enumerate(zip(self.step_labels, tips)):
            b = self._btn(tx, tp, lambda k=i: self._step_clicked(k), False, None, None, "step")
            bl.addWidget(b)
            self.steps.append(b)
        bl.addStretch(1)
        self.btn_tour = self._btn("Тур по программе", "Короткий рассказ, как всё устроено", self.tour, False, None, 26)
        bl.addWidget(self.btn_tour)
        self.btn_banner_close = self._btn("✕", "Скрыть подсказки (вернуть: Помощь → Быстрый старт)", self.hide_quickstart, False, None, None, "close")
        bl.addWidget(self.btn_banner_close)
        bw = QWidget()
        bwl = QVBoxLayout(bw)
        bwl.setContentsMargins(12, 6, 12, 0)
        bwl.addWidget(self.banner)
        v.addWidget(bw)
        self.banner_wrap = bw
        if self.settings.get("hide_quickstart"):
            bw.hide()

        # --- дорожки
        self.tracks = TracksPanel(self.session, self.state)
        up = QWidget()
        ul = QVBoxLayout(up)
        ul.setContentsMargins(12, 8, 12, 8)
        ul.addWidget(self.tracks)
        v.addWidget(up, 1)

        # --- статусбар
        self.prog = QProgressBar()
        self.prog.setFixedWidth(150)
        self.prog.setFixedHeight(10)
        self.prog.setTextVisible(False)
        self.prog.setRange(0, 0)
        self.prog.hide()
        self.lbl_render = QLabel("")
        self.lbl_render.setObjectName("muted")
        self.statusBar().addPermanentWidget(self.lbl_render)
        self.statusBar().addPermanentWidget(self.prog)
        self.statusBar().showMessage("Наведите курсор на любую кнопку или ручку — здесь появится подсказка.")
        self.btn_bypass.toggled.connect(lambda on: self.session.set_bypass_all(on))
        for b in self.findChildren(QPushButton):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # кнопки не должны сжиматься до обрезанного текста
        for bar in (tr, eb):
            for b in bar.findChildren(QPushButton):
                b.ensurePolished()
                b.setMinimumWidth(max(b.minimumWidth(), b.sizeHint().width()))

    def _build_menu(self):
        mb = self.menuBar()
        self.actions = {}

        def act(menu, key, text, fn, shortcut=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(lambda _=False: fn())
            menu.addAction(a)
            self.actions[key] = a
            return a

        m = mb.addMenu("Файл")
        act(m, "new", "Новый проект", self.new_project, "Ctrl+N")
        act(m, "open", "Открыть проект…", self.open_project, "Ctrl+O")
        self.recent_menu = m.addMenu("Недавние проекты")
        act(m, "save", "Сохранить проект", self.save_project, "Ctrl+S")
        act(m, "saveas", "Сохранить проект как…", lambda: self.save_project(True), "Ctrl+Shift+S")
        m.addSeparator()
        act(m, "import_vocal", "Открыть вокал из файла…", self.open_vocal, "Ctrl+I")
        act(m, "import_beat", "Загрузить бит / инструментал…", self.open_backing)
        m.addSeparator()
        act(m, "export", "Экспорт…", self.export, "Ctrl+E")
        act(m, "audio", "Микрофон и звук…", self.open_settings)
        m.addSeparator()
        act(m, "quit", "Выход", self.close, "Ctrl+Q")
        self._fill_recent()

        m = mb.addMenu("Правка")
        act(m, "undo", "Отменить", self.undo, "Ctrl+Z")
        act(m, "redo", "Повторить", self.redo, "Ctrl+Y")
        m.addSeparator()
        act(m, "cut", "Вырезать", lambda: self.edit("cut"), "Ctrl+X")
        act(m, "copy", "Копировать", lambda: self.edit("copy"), "Ctrl+C")
        act(m, "paste", "Вставить в позицию курсора", lambda: self.edit("paste"), "Ctrl+V")
        act(m, "selall", "Выделить всю запись", lambda: self.edit("selectall"), "Ctrl+A")
        m.addSeparator()
        act(m, "delete", "Удалить выделенное (тишина)", lambda: self.edit("silence"), "Delete")
        act(m, "ripple", "Удалить и сомкнуть края", lambda: self.edit("delete"), "Ctrl+Delete")
        act(m, "trim", "Обрезать до выделения", lambda: self.edit("trim"), "Ctrl+T")
        m.addSeparator()
        act(m, "fadein", "Fade In", lambda: self.edit("fade_in"))
        act(m, "fadeout", "Fade Out", lambda: self.edit("fade_out"))
        act(m, "norm", "Нормализовать", lambda: self.edit("normalize"))
        act(m, "gain", "Усилить / ослабить…", self.edit_gain)
        act(m, "reverse", "Развернуть задом наперёд", lambda: self.edit("reverse"))
        act(m, "trimsil", "Убрать тишину по краям записи", lambda: self.edit("trim_silence"))
        m.addSeparator()
        act(m, "shift", "Сдвинуть запись по времени…", self.shift_clip)
        act(m, "ab", "Original (A/B): послушать без обработки", self.btn_bypass.toggle, "Ctrl+B")

        m = mb.addMenu("Дорожка")
        act(m, "add_vocal", "Добавить дорожку для голоса", lambda: self.add_vocal_track(), "Ctrl+Shift+T")
        act(m, "add_beat", "Добавить дорожку с битом…", self.open_backing)
        m.addSeparator()
        act(m, "fx", "Плагины (FX) выбранной дорожки…", lambda: self.open_fx(self.session.sel_track), "F")
        act(m, "automix", "✨ Auto-Mix выбранной дорожки…", lambda: self.automix(self.session.sel_track), "Ctrl+M")
        m.addSeparator()
        act(m, "rename", "Переименовать…", lambda: self.track_action("rename", self.session.sel_track))
        act(m, "dup", "Дублировать дорожку", lambda: self.track_action("duplicate", self.session.sel_track))
        act(m, "clear", "Удалить запись с дорожки", lambda: self.edit("clear"))
        act(m, "remove", "Удалить дорожку", lambda: self.track_action("remove", self.session.sel_track))

        m = mb.addMenu("Воспроизведение")
        act(m, "play", "Играть / стоп", self.toggle_play)
        act(m, "play_sel", "Играть выделение", lambda: self.start_play(True), "Shift+Space")
        act(m, "record", "Запись", self.toggle_record, "R")
        act(m, "loop", "Зациклить", self.btn_loop.toggle, "L")
        act(m, "home", "В начало", lambda: self.seek(0.0), "Home")
        act(m, "end", "В конец", self.goto_end, "End")
        act(m, "metro", "Метроном", self.btn_metro.toggle, "M")

        m = mb.addMenu("Вид")
        act(m, "zin", "Приблизить", lambda: self.state.zoom(1.3, self.state.view_w / 2), "Ctrl+=")
        act(m, "zout", "Отдалить", lambda: self.state.zoom(1 / 1.3, self.state.view_w / 2), "Ctrl+-")
        act(m, "fit", "Показать весь проект", self.state.fit, "Ctrl+0")
        act(m, "zsel", "Приблизить выделение", self.zoom_selection)
        m.addSeparator()
        act(m, "qs", "Быстрый старт (подсказки)", self.show_quickstart)

        m = mb.addMenu("Помощь")
        act(m, "tour", "Тур по программе", self.tour)
        act(m, "keys", "Горячие клавиши", lambda: ShortcutsDialog(self).exec())
        act(m, "welcome", "Приветствие", self._welcome)
        act(m, "diag", "Диагностика (если что-то не работает)…", self.diagnostics)
        act(m, "about", "О программе", lambda: QMessageBox.about(
            self, APP_NAME, "<b>Vocal Studio</b><br>Запись и обработка вокала: автотюн, EQ, компрессор, де-эссер, "
                            "дилей, реверберация, гармония и другие плагины. Всё работает локально на вашем компьютере."))

        # Shift+пробел и пробел — отдельные QShortcut (пробел не должен срабатывать из полей ввода)
        sc = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._space)

    def _space(self):
        w = QApplication.focusWidget()
        from PyQt6.QtWidgets import QAbstractSpinBox, QLineEdit
        if isinstance(w, (QAbstractSpinBox, QLineEdit)):
            return
        self.toggle_play()

    def _connect(self):
        s = self.session
        self.tracks.request.connect(self.track_action)
        self.tracks.files_dropped.connect(self._lane_drop)
        self.tracks.lane_menu.connect(self.lane_menu)
        self.tracks.clip_moved.connect(lambda tid, ts: self._move_clip(tid, ts))
        self.tracks.btn_add_vocal.clicked.connect(lambda: self.add_vocal_track())
        self.tracks.btn_add_back.clicked.connect(self.open_backing)
        self.tracks.btn_add_file.clicked.connect(self.open_vocal)
        self.state.seek_requested.connect(self.seek)
        self.state.changed.connect(self._update_edit_buttons)

        s.structure_changed.connect(self._structure_changed)
        s.track_changed.connect(self._track_changed)
        s.selection_changed.connect(self.tracks.update_selection)
        s.audio_changed.connect(self.tracks.update_audio)
        s.audio_changed.connect(lambda _tid: self._update_duration())
        s.project_replaced.connect(self._project_replaced)
        s.dirty_changed.connect(lambda _d: self._update_title())
        s.history_changed.connect(self._update_edit_buttons)
        s.render_started.connect(self._render_started)
        s.render_done.connect(self._render_done)
        s.render_failed.connect(self._render_failed)
        s.render_progress.connect(self._render_progress)

    # ======================================================== окна плагинов ==
    @guard
    def open_fx(self, scope):
        if not scope:
            self.statusBar().showMessage("Сначала выберите дорожку", 3000)
            return
        if scope != MASTER and self.session.track(scope) is None:
            return
        win = self.fx_windows.get(scope)
        if win is None:
            win = FxWindow(self.session, scope)
            win.seek_requested.connect(self.seek)
            win.automix_requested.connect(self.automix)
            win.closed.connect(self._fx_closed)
            self.fx_windows[scope] = win
            if self.session.result is not None:
                win.set_result(self.session.result)
            geo = self.geometry()
            win.move(geo.x() + 60 + 24 * (len(self.fx_windows) - 1), geo.y() + 90 + 24 * (len(self.fx_windows) - 1))
        self.session.select(scope)
        win.show()
        win.raise_()
        win.activateWindow()

    def _fx_closed(self, scope):
        w = self.fx_windows.pop(scope, None)
        if w is not None:
            try:
                self.session.structure_changed.disconnect(w.rebuild)
                self.session.plugins_replaced.disconnect(w._replaced)
            except Exception:
                pass
            w.deleteLater()

    # ======================================================== быстрый старт ==
    def _step_clicked(self, k):
        if k == 0:
            m = QMenu(self)
            a1 = m.addAction("●  Записать голос с микрофона")
            a2 = m.addAction("Открыть вокал из файла…")
            a3 = m.addAction("Загрузить бит / инструментал…")
            a = m.exec(self.steps[0].mapToGlobal(self.steps[0].rect().bottomLeft()))
            if a == a1:
                self.prepare_record_track()
                self.statusBar().showMessage("Дорожка готова. Нажмите ● REC и пойте!", 8000)
            elif a == a2:
                self.open_vocal()
            elif a == a3:
                self.open_backing()
        elif k == 1:
            self.automix(self._best_vocal_track())
        elif k == 2:
            self.toggle_play()
        elif k == 3:
            self.export()

    def _best_vocal_track(self):
        s = self.session
        t = s.track(s.sel_track) if s.sel_track and s.sel_track != MASTER else None
        if t is not None and t.kind == "vocal" and t.audio is not None:
            return t.id
        for q in s.project.tracks:
            if q.kind == "vocal" and q.audio is not None:
                return q.id
        return t.id if t is not None else None

    def _update_quickstart(self):
        s = self.session
        has_audio = any(t.audio is not None for t in s.project.tracks)
        has_fx = any(t.plugins for t in s.project.tracks)
        done = [has_audio, has_fx and has_audio, self._played, self._exported]
        cur = next((i for i, d in enumerate(done) if not d), len(done))
        for i, b in enumerate(self.steps):
            b.setProperty("done", bool(done[i]))
            b.setProperty("cur", i == cur)
            b.setText(("✓  " + self.step_labels[i][3:]) if done[i] else self.step_labels[i])
            b.style().unpolish(b)
            b.style().polish(b)

    def hide_quickstart(self):
        self.banner_wrap.hide()
        self.settings["hide_quickstart"] = True
        save_settings(self.settings)

    def show_quickstart(self):
        self.banner_wrap.show()
        self.settings["hide_quickstart"] = False
        save_settings(self.settings)

    def tour(self):
        TourDialog(self).exec()

    def _startup_dialogs(self):
        try:
            if self._offer_recovery():
                return
            if not self.settings.get("welcome_done"):
                self._welcome()
        except Exception:
            import sys
            safety.report(*sys.exc_info(), where="startup")

    def _welcome(self):
        dlg = WelcomeDialog(self)
        dlg.exec()
        self.settings["welcome_done"] = True
        save_settings(self.settings)
        c = dlg.choice
        if c == "record":
            self.prepare_record_track()
            self.statusBar().showMessage("Дорожка для голоса готова. Нажмите ● REC (или клавишу R) и пойте!", 10000)
        elif c == "vocal":
            self.open_vocal()
        elif c == "backing":
            self.open_backing()
        elif c == "tour":
            self.tour()

    @guard
    def diagnostics(self):
        DiagnosticsDialog(self.engine, self).exec()

    # ======================================================== устройства =====
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

    @guard
    def open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec():
            self.settings = dlg.result_settings()
            save_settings(self.settings)
            self.stop_all()
            self._apply_device_settings()
            self.statusBar().showMessage("Настройки применены", 4000)

    # ============================================================= проект ====
    def _confirm_discard(self) -> bool:
        s = self.session
        if not s.project.dirty or not any(t.audio is not None or t.plugins for t in s.project.tracks):
            return True
        r = QMessageBox.question(self, APP_NAME, "Сохранить изменения в проекте?",
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                 QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.save_project()
        return r == QMessageBox.StandardButton.Discard

    def _ensure_default_track(self):
        """Пустой проект сразу содержит готовую дорожку для записи — так понятнее, что делать."""
        s = self.session
        if not s.project.tracks:
            t = s.add_track("vocal")
            s.set_track_prop(t.id, "arm", True)
            s.mark_dirty(False)
            s._undo_stack.clear()
            s.history_changed.emit()

    @guard
    def new_project(self):
        self.stop_all()
        if self._confirm_discard():
            self._close_fx_windows()
            self.session.new_project()
            self._ensure_default_track()
            self.seek(0.0)

    def _close_fx_windows(self):
        for w in list(self.fx_windows.values()):
            w.close()

    @guard
    def open_project(self):
        self.stop_all()
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", self.settings.get("last_dir", ""), PROJECT_FILTER)
        if path:
            self._open_project_path(path)

    def _open_project_path(self, path):
        try:
            self._close_fx_windows()
            self.session.open_project(path)
            self.settings["last_dir"] = os.path.dirname(path)
            self._add_recent(path)
            self.seek(0.0)
            self.statusBar().showMessage(f"Проект открыт: {os.path.basename(path)}", 5000)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Не удалось открыть проект:\n{e}")

    def open_path(self, path):
        if path.lower().endswith(".vsp"):
            self._open_project_path(path)
        else:
            self.load_audio_file(path, "vocal")

    def _fill_recent(self):
        self.recent_menu.clear()
        recent = [p for p in self.settings.get("recent", []) if os.path.isfile(p)]
        if not recent:
            a = self.recent_menu.addAction("(пусто)")
            a.setEnabled(False)
        for p in recent[:8]:
            a = self.recent_menu.addAction(os.path.basename(p))
            a.setToolTip(p)
            a.triggered.connect(lambda _=False, pp=p: (self.stop_all(), self._confirm_discard() and self._open_project_path(pp)))

    def _add_recent(self, path):
        r = [p for p in self.settings.get("recent", []) if p != path]
        self.settings["recent"] = ([path] + r)[:8]
        save_settings(self.settings)
        self._fill_recent()

    @guard
    def save_project(self, as_new=False) -> bool:
        s = self.session
        path = s.project.path
        if as_new or not path:
            base = (s.project.name if s.project.name != "Untitled" else "my_song") + ".vsp"
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", os.path.join(self.settings.get("last_dir", ""), base),
                                                  PROJECT_FILTER)
            if not path:
                return False
            if not path.lower().endswith(".vsp"):
                path += ".vsp"
        try:
            s.save_project(path)
            self.settings["last_dir"] = os.path.dirname(path)
            self._add_recent(path)
            self.statusBar().showMessage(f"Проект сохранён: {path}", 6000)
            self._update_title()
            self._remove_autosave()
            return True
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Не удалось сохранить проект:\n{e}")
            return False

    def _project_replaced(self):
        self._track_ids = ()
        self._played = False
        self._exported = False
        self.state.clear_selection()
        self.sp_bpm.blockSignals(True)
        self.sp_bpm.setValue(self.session.project.bpm)
        self.sp_bpm.blockSignals(False)
        self._update_title()

    def _update_title(self):
        p = self.session.project
        self.setWindowTitle(f"{p.name}{' *' if p.dirty else ''} — {APP_NAME}")
        self.lbl_project.setText(f"{p.name}{' *' if p.dirty else ''}")

    # --- автосохранение и восстановление --------------------------------
    def _autosave_path(self):
        return os.path.join(config_dir(), "autosave.vsp")

    def _remove_autosave(self):
        try:
            os.remove(self._autosave_path())
        except OSError:
            pass

    def _autosave(self):
        s = self.session
        if self._autosave_busy or not s.project.dirty or self.engine.mode == "record":
            return
        if not any(t.audio is not None for t in s.project.tracks):
            return
        snap = Project()
        snap.sr, snap.name, snap.bpm = s.project.sr, s.project.name, s.project.bpm
        snap.master = copy.deepcopy(s.project.master)
        snap.tracks = [dataclasses.replace(t, plugins=copy.deepcopy(t.plugins)) for t in s.project.tracks]
        self._autosave_busy = True

        def work():
            try:
                tmp = self._autosave_path() + ".tmp"
                snap.save(tmp)
                os.replace(tmp, self._autosave_path())
            except Exception:
                pass
            finally:
                self._autosave_busy = False

        threading.Thread(target=work, daemon=True).start()

    def _offer_recovery(self) -> bool:
        p = self._autosave_path()
        if not os.path.isfile(p) or any(t.audio is not None for t in self.session.project.tracks):
            return False
        age = time.time() - os.path.getmtime(p)
        r = QMessageBox.question(self, "Восстановить проект?",
                                 f"Найдена автосохранённая копия проекта ({time.strftime('%d.%m %H:%M', time.localtime(os.path.getmtime(p)))}).\n"
                                 "Вероятно, программа была закрыта без сохранения. Восстановить?")
        if r == QMessageBox.StandardButton.Yes:
            try:
                self.session.open_project(p)
                self.session.project.path = None
                self.session.mark_dirty(True)
                self.statusBar().showMessage("Проект восстановлен из автосохранения. Не забудьте сохранить его (Ctrl+S).", 12000)
                return True
            except Exception as e:
                QMessageBox.warning(self, APP_NAME, f"Не удалось восстановить: {e}")
        else:
            self._remove_autosave()
        return False

    # ============================================================ дорожки =====
    def _structure_changed(self):
        ids = tuple((t.id, t.kind) for t in self.session.project.tracks)
        if ids != self._track_ids:
            self._track_ids = ids
            self.tracks.rebuild()
            for tid in list(self.state.__dict__.get("sel_tid") and [self.state.sel_tid] or []):
                if self.session.track(tid) is None:
                    self.state.clear_selection()
        else:
            self.tracks.refresh_headers()
            self.tracks.update_selection()
        self._update_quickstart()
        self._update_edit_buttons()
        self._update_title()

    def _track_changed(self, tid):
        self.tracks.refresh_headers()
        w = self.fx_windows.get(tid)
        if w is not None:
            w.rebuild()

    def add_vocal_track(self, select=True):
        return self.session.add_track("vocal")

    def prepare_record_track(self):
        """Выбирает (или создаёт) дорожку для записи и помечает её красной кнопкой REC."""
        s = self.session
        t = None
        for q in s.project.tracks:
            if q.arm and q.kind == "vocal":
                t = q
        if t is None:
            sel = s.track(s.sel_track) if s.sel_track and s.sel_track != MASTER else None
            if sel is not None and sel.kind == "vocal":
                t = sel
            else:
                t = next((q for q in s.project.tracks if q.kind == "vocal" and q.audio is None), None)
        if t is None:
            t = s.add_track("vocal")
        s.set_track_prop(t.id, "arm", True)
        s.select(t.id)
        return t

    @guard
    def track_action(self, action, tid):
        s = self.session
        if tid is None:
            return
        if action == "select":
            s.select(tid)
        elif action == "fx":
            self.open_fx(tid)
        elif action == "rename":
            t = s.track(tid)
            if t:
                name, ok = QInputDialog.getText(self, "Переименовать дорожку", "Название:", text=t.name)
                if ok and name.strip():
                    s.set_track_prop(tid, "name", name.strip())
        elif action == "color":
            t = s.track(tid)
            if t:
                from PyQt6.QtGui import QColor
                c = QColorDialog.getColor(QColor(t.color), self, "Цвет дорожки")
                if c.isValid():
                    s.set_track_prop(tid, "color", c.name())
        elif action == "duplicate":
            s.duplicate_track(tid)
        elif action == "clear":
            self.edit("clear", tid=tid)
        elif action == "remove":
            t = s.track(tid)
            if t is None:
                return
            if t.audio is not None or t.plugins:
                if QMessageBox.question(self, APP_NAME, f"Удалить дорожку «{t.name}» вместе с записью и плагинами?\n"
                                        "(Можно отменить: Ctrl+Z)") != QMessageBox.StandardButton.Yes:
                    return
            if self.engine.mode != "idle":
                self.stop_all()
            w = self.fx_windows.get(tid)
            if w is not None:
                w.close()
            s.remove_track(tid)
        elif action == "automix":
            self.automix(tid)

    @guard
    def automix(self, tid):
        s = self.session
        if not tid or tid == MASTER or s.track(tid) is None:
            QMessageBox.information(self, "Auto-Mix", "Выберите вокальную дорожку с записью (нажмите на неё), затем ✨ Auto-Mix.")
            return
        t = s.track(tid)
        if t.audio is None:
            QMessageBox.information(self, "Auto-Mix", "На этой дорожке пока нет записи. Сначала запишите голос (● REC) или откройте файл.")
            return
        if t.kind != "vocal":
            QMessageBox.information(self, "Auto-Mix", "Auto-Mix предназначен для вокальных дорожек. Выберите дорожку с голосом.")
            return
        self.stop_all()
        AutoMixDialog(s, tid, self).exec()
        self._update_quickstart()
        if s.track(tid) is not None and s.track(tid).plugins:
            self.open_fx(tid)

    # ============================================================== файлы =====
    @guard
    def open_vocal(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть вокал", self.settings.get("last_dir", ""), AUDIO_FILTER)
        if path:
            self.load_audio_file(path, "vocal")

    @guard
    def open_backing(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить бит / инструментал", self.settings.get("last_dir", ""), AUDIO_FILTER)
        if path:
            self.load_audio_file(path, "audio")

    def load_audio_file(self, path, kind, tid=None):
        try:
            self.stop_all()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                data = audio_io.load_audio(path, SR)
            finally:
                QApplication.restoreOverrideCursor()
            name = os.path.splitext(os.path.basename(path))[0]
            audio = to_mono(data)[None, :] if kind == "vocal" else to_stereo(data)
            s = self.session
            if tid is not None and s.track(tid) is not None:
                s.set_track_audio(tid, audio, os.path.basename(path), start=0)
                t = s.track(tid)
            else:
                t = s.add_track(kind, name=name[:24] or None, audio=audio, source=os.path.basename(path))
            self.settings["last_dir"] = os.path.dirname(path)
            self.seek(0.0)
            self.statusBar().showMessage(f"Загружено: {os.path.basename(path)}  →  дорожка «{t.name}»", 6000)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, str(e))

    @guard
    def _lane_drop(self, tid, urls):
        t = self.session.track(tid)
        if t is None:
            return
        self.load_audio_file(urls[0], t.kind, tid=tid)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    @guard
    def dropEvent(self, e):
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        for u in urls:
            if u.lower().endswith(".vsp"):
                self.stop_all()
                if self._confirm_discard():
                    self._open_project_path(u)
                return
        kind = "audio" if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else "vocal"
        for u in urls:
            self.load_audio_file(u, kind)

    @guard
    def export(self):
        s = self.session
        if not any(t.audio is not None for t in s.project.tracks):
            self.statusBar().showMessage("Нечего экспортировать — сначала добавьте звук", 4000)
            return
        dlg = ExportDialog(sum(1 for t in s.project.tracks if t.audio is not None), self)
        if not dlg.exec():
            return
        fmt = dlg.cb_fmt.currentText()
        ext = audio_io.EXPORT_FORMATS[fmt][2]
        stems = dlg.rb_stems.isChecked()
        base = s.project.name if s.project.name != "Untitled" else "my_song"
        self.stop_all()
        if stems:
            folder = QFileDialog.getExistingDirectory(self, "Папка для файлов дорожек", self.settings.get("last_dir", ""))
            if not folder:
                return
            target = folder
        else:
            target, _ = QFileDialog.getSaveFileName(self, "Экспорт", os.path.join(self.settings.get("last_dir", ""), base + ext), f"{fmt} (*{ext})")
            if not target:
                return
            if not target.lower().endswith(ext):
                target += ext
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            res = s.render_blocking()
            if stems:
                n = 0
                for t in s.project.tracks:
                    tr = res.tracks.get(t.id)
                    if tr is None:
                        continue
                    safe = "".join(ch for ch in t.name if ch.isalnum() or ch in " -_").strip() or "track"
                    audio_io.save_audio(os.path.join(target, f"{safe}{ext}"), tr.out.astype(np.float64), SR, fmt)
                    n += 1
                msg = f"Сохранено дорожек: {n} → {target}"
            else:
                out = res.out.astype(np.float64)
                if dlg.chk_norm.isChecked() and out.size:
                    pk = np.abs(out).max()
                    if pk > 1e-6:
                        out = out * (10 ** (-1 / 20) / pk)
                audio_io.save_audio(target, out, SR, fmt)
                msg = f"Сохранено: {target}"
            self.settings["last_dir"] = os.path.dirname(target) if not stems else target
            save_settings(self.settings)
            self._exported = True
            self._update_quickstart()
            self.statusBar().showMessage(msg, 10000)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Ошибка экспорта:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()
            s.request_render(immediate=True)

    # ============================================================== правка ====
    def _selection_target(self):
        """(tid, t0, t1) — выделение или (выбранная дорожка, None, None)."""
        s, st = self.session, self.state
        if st.has_selection and s.track(st.sel_tid) is not None:
            return st.sel_tid, st.sel_t0, st.sel_t1
        tid = s.sel_track
        if tid and tid != MASTER and s.track(tid) is not None:
            return tid, None, None
        return None, None, None

    @guard
    def edit(self, op, tid=None, **kw):
        s, st = self.session, self.state
        if self.engine.mode == "record":
            return
        sel_tid, t0, t1 = self._selection_target()
        tid = tid or sel_tid
        if tid is None:
            self.statusBar().showMessage("Сначала выберите дорожку (нажмите на неё)", 4000)
            return
        t = s.track(tid)
        if op == "selectall":
            if t.audio is None:
                self.statusBar().showMessage("На дорожке нет записи", 3000)
                return
            st.set_selection(tid, t.start / s.sr, t.end / s.sr)
            return
        if tid != sel_tid:
            t0 = t1 = None
        if op == "paste":
            kw["at"] = kw.get("at", self.playhead)
        if op == "clear":
            if t.audio is None:
                self.statusBar().showMessage("На дорожке нет записи", 3000)
                return
            if QMessageBox.question(self, APP_NAME, f"Удалить запись с дорожки «{t.name}»?\n(Можно отменить: Ctrl+Z)") != QMessageBox.StandardButton.Yes:
                return
            st.clear_selection()
        if op in ("silence", "cut", "delete", "trim", "copy") and t0 is None:
            if op == "silence" and t.audio is not None:
                # Delete без выделения: удаляем запись целиком (по запросу)
                self.edit("clear", tid=tid)
                return
            self.statusBar().showMessage("Сначала выделите участок: проведите мышью по дорожке", 5000)
            return
        msg = s.edit_audio(tid, op, t0, t1, **kw)
        if msg:
            self.statusBar().showMessage(msg, 6000)
        if op == "delete":
            st.clear_selection()
        elif op == "trim":
            t = s.track(tid)
            if t is not None and t.audio is not None:
                st.set_selection(tid, t.start / s.sr, t.end / s.sr)
        self._update_edit_buttons()

    @guard
    def edit_gain(self):
        db, ok = QInputDialog.getDouble(self, "Усилить / ослабить", "Изменение громкости, дБ:", 0.0, -40.0, 24.0, 1)
        if ok:
            self.edit("gain", db=db)

    @guard
    def shift_clip(self):
        tid, _, _ = self._selection_target()
        t = self.session.track(tid) if tid else None
        if t is None or t.audio is None:
            self.statusBar().showMessage("Выберите дорожку с записью", 3000)
            return
        ms, ok = QInputDialog.getDouble(self, "Сдвинуть запись", "Сдвиг в миллисекундах (+ позже, − раньше):", 0.0, -60000.0, 60000.0, 0)
        if ok and ms:
            self._move_clip(tid, t.start / SR + ms / 1000.0)

    @guard
    def _move_clip(self, tid, new_start_sec):
        self.session.move_clip(tid, int(round(max(0.0, new_start_sec) * SR)))
        self.statusBar().showMessage("Запись сдвинута", 3000)

    @guard
    def undo(self):
        if self.engine.mode == "record":
            return
        if self.session.undo():
            self.statusBar().showMessage("Отменено", 2500)
        else:
            self.statusBar().showMessage("Нечего отменять", 2500)

    @guard
    def redo(self):
        if self.engine.mode == "record":
            return
        if self.session.redo():
            self.statusBar().showMessage("Повторено", 2500)
        else:
            self.statusBar().showMessage("Нечего повторять", 2500)

    def zoom_selection(self):
        st = self.state
        if not st.has_selection:
            self.statusBar().showMessage("Сначала выделите участок", 3000)
            return
        dur = st.sel_t1 - st.sel_t0
        st.pps = max(4.0, (st.view_w - 40) / max(dur, 0.05))
        st.t0 = max(0.0, st.sel_t0 - 20 / st.pps)
        st.changed.emit()

    def goto_end(self):
        self.seek(self.state.duration - 2.0 if self.state.duration > 2 else 0.0)

    def _update_edit_buttons(self):
        s, st = self.session, self.state
        sel = st.has_selection
        rec = self.engine.mode == "record"
        have_audio = any(t.audio is not None for t in s.project.tracks)
        eb = self.edit_btns
        for k in ("copy", "cut", "delete", "ripple", "trim", "zoomsel"):
            eb[k].setEnabled(sel and not rec)
        eb["paste"].setEnabled(s.clipboard is not None and not rec)
        eb["undo"].setEnabled(s.can_undo and not rec)
        eb["redo"].setEnabled(s.can_redo and not rec)
        eb["selall"].setEnabled(have_audio and not rec)
        for k in ("fadein", "fadeout", "norm"):
            eb[k].setEnabled(have_audio and not rec)
        if sel:
            self.lbl_sel.setText(f"Выделено: {st.sel_t1 - st.sel_t0:.2f} с  ({fmt_time(st.sel_t0)} – {fmt_time(st.sel_t1)})")
        else:
            self.lbl_sel.setText("Чтобы выделить участок — проведите мышью по дорожке")

    @guard
    def lane_menu(self, tid, gpos, t):
        s, st = self.session, self.state
        tr = s.track(tid)
        if tr is None or self.engine.mode == "record":
            return
        has_sel = st.has_selection and st.sel_tid == tid
        m = QMenu(self)
        a = {}

        def add(key, text, enabled=True):
            act = m.addAction(text)
            act.setEnabled(enabled)
            a[act] = key
            return act

        add("cut", "Вырезать  (Ctrl+X)", has_sel)
        add("copy", "Копировать  (Ctrl+C)", has_sel)
        add("paste", f"Вставить сюда  (Ctrl+V)", s.clipboard is not None)
        m.addSeparator()
        add("silence", "Удалить выделенное (останется тишина)  (Del)", has_sel)
        add("delete", "Удалить и сомкнуть края  (Ctrl+Del)", has_sel)
        add("trim", "Обрезать до выделения  (Ctrl+T)", has_sel)
        m.addSeparator()
        add("fade_in", "Fade In", tr.audio is not None)
        add("fade_out", "Fade Out", tr.audio is not None)
        add("normalize", "Нормализовать", tr.audio is not None)
        add("reverse", "Развернуть выделение", has_sel)
        add("trim_silence", "Убрать тишину по краям записи", tr.audio is not None)
        m.addSeparator()
        add("selectall", "Выделить всю запись  (Ctrl+A)", tr.audio is not None)
        add("shift", "Сдвинуть запись по времени…", tr.audio is not None)
        m.addSeparator()
        add("fx", "Плагины (FX)…")
        add("automix", "✨ Auto-Mix…", tr.kind == "vocal" and tr.audio is not None)
        add("clear", "Удалить запись с дорожки", tr.audio is not None)
        act = m.exec(gpos)
        key = a.get(act)
        if key is None:
            return
        if key == "paste":
            self.edit("paste", tid=tid, at=t)
        elif key == "fx":
            self.open_fx(tid)
        elif key == "automix":
            self.automix(tid)
        elif key == "shift":
            self.shift_clip()
        else:
            self.edit(key, tid=tid)

    # ====================================================== воспроизведение ===
    def seek(self, t: float):
        self.playhead = max(0.0, t)
        if self.engine.mode == "play":
            self.engine.seek_seconds(self.playhead)
        self._update_time()
        self.state.set_playhead(self.playhead)
        for w in self.fx_windows.values():
            w.set_playhead(self.playhead)

    @guard
    def toggle_play(self):
        if self.engine.mode == "record":
            self.toggle_record()
        elif self.engine.mode == "play":
            self.stop_all()
        else:
            self.start_play()

    def _sync_click(self):
        self.engine.click_on = self.btn_metro.isChecked()
        self.engine.click_bpm = self.session.project.bpm

    @guard
    def start_play(self, from_selection=False):
        if not self.engine.available:
            self._no_audio()
            self.btn_play.setChecked(False)
            return
        res = self.session.result
        if res is None or res.out.shape[1] == 0:
            self.statusBar().showMessage("Нечего воспроизводить — запишите голос (● REC) или откройте файл", 5000)
            self.btn_play.setChecked(False)
            return
        self._stop_arm()
        self.engine.set_buffer(res.out)
        if from_selection and self.state.has_selection:
            self.playhead = self.state.sel_t0
        pos = int(self.playhead * SR)
        if pos >= res.out.shape[1] - 10:
            pos = 0
            self.playhead = 0.0
        self._sync_click()
        try:
            self.engine.start_play(pos)
        except AudioError as e:
            self.btn_play.setChecked(False)
            QMessageBox.warning(self, "Воспроизведение", str(e))
            return
        self._apply_loop()
        self.btn_play.setChecked(True)
        self.btn_play.setText("❚❚  PAUSE")
        self._played = True
        self._update_quickstart()

    def _apply_loop(self):
        res = self.session.result
        eng = self.engine
        if self.btn_loop.isChecked() and res is not None and res.out.shape[1] > 0:
            if self.state.has_selection:
                eng.set_loop_seconds(self.state.sel_t0, self.state.sel_t1)
            else:
                eng.set_loop_seconds(0.0, res.out.shape[1] / SR)
        else:
            eng.set_loop_seconds(None)

    def _bpm_changed(self, v):
        self.session.set_bpm(v)
        self._sync_click()

    def _metro_changed(self):
        self._sync_click()

    def stop_all(self):
        try:
            if self.engine.mode == "record":
                self._finish_recording()
            elif self.engine.mode in ("play", "monitor"):
                self.engine.stop()
        finally:
            self.btn_play.setChecked(False)
            self.btn_play.setText("▶  PLAY")
            if self.btn_arm.isChecked():
                self.btn_arm.blockSignals(True)
                self.btn_arm.setChecked(False)
                self.btn_arm.blockSignals(False)
            self._update_edit_buttons()

    def _no_audio(self):
        QMessageBox.information(self, APP_NAME,
                                "Аудио-подсистема недоступна (PortAudio не загрузился).\n"
                                "Открытие файлов и экспорт работают, но запись/воспроизведение — нет.\n\n" +
                                str(audio_io.SD_ERROR or "") + "\n\nПомощь → Диагностика покажет подробности.")

    # ================================================================ запись ==
    @guard
    def toggle_arm(self, on=None):
        on = self.btn_arm.isChecked() if on is None else on
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
                self.statusBar().showMessage("Говорите/пойте — уровень отображается на шкале IN. Не должен касаться красной зоны.", 8000)
            except AudioError as e:
                self.btn_arm.blockSignals(True)
                self.btn_arm.setChecked(False)
                self.btn_arm.blockSignals(False)
                QMessageBox.warning(self, "Микрофон", str(e))
        else:
            self._stop_arm()

    def _stop_arm(self):
        if self.engine.mode == "monitor":
            self.engine.stop()
        if self.btn_arm.isChecked():
            self.btn_arm.blockSignals(True)
            self.btn_arm.setChecked(False)
            self.btn_arm.blockSignals(False)

    def _monitor_mix(self, exclude_tid):
        """Что слышит исполнитель во время записи: все остальные дорожки (после фейдеров и плагинов)."""
        res = self.session.result
        if res is None or res.out.shape[1] == 0:
            return None
        acc = None
        for tid, tr in res.tracks.items():
            if tid == exclude_tid or not tr.audible:
                continue
            acc = tr.out.copy() if acc is None else acc + tr.out
        if acc is None:
            return None
        g = 10 ** (self.session.project.master["vol_db"] / 20.0)
        return (acc * g).astype(np.float32)

    @guard
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
        t = self.prepare_record_track()
        backing = self._monitor_mix(t.id)
        self.rec_tid = t.id
        total = self.session.project.total_samples()
        if total > SR // 2 and self.playhead * SR >= total - SR * 0.3:
            self.seek(0.0)      # курсор стоит в самом конце — начинаем сначала, а не после песни
        self.rec_start = int(self.playhead * SR)
        bk = None
        if backing is not None and backing.shape[1] > self.rec_start:
            bk = np.ascontiguousarray(backing[:, self.rec_start:])
        self._sync_click()
        count_in = self.engine.count_in_samples(1) if self.btn_count.isChecked() else 0
        self._rec_had_backing = bk is not None or self.btn_metro.isChecked() or count_in > 0
        try:
            self.engine.monitor = bool(self.settings.get("monitor", False))
            self.engine.start_record(self.rec_start, bk, record=True, count_in=count_in)
        except AudioError as e:
            self.btn_rec.setChecked(False)
            QMessageBox.warning(self, "Запись", str(e))
            return
        self.btn_rec.setChecked(True)
        self.btn_rec.setText("■  STOP REC")
        self.state.follow = True
        self._update_edit_buttons()
        if self.engine.warning:
            self.statusBar().showMessage(self.engine.warning, 10000)
        else:
            self.statusBar().showMessage(f"Идёт запись на дорожку «{t.name}»… нажмите STOP (или R), чтобы закончить", 0)

    def _finish_recording(self):
        eng = self.engine
        rec, lat = eng.stop_record()
        self.btn_rec.setChecked(False)
        self.btn_rec.setText("●  REC")
        tid = self.rec_tid
        self.tracks.set_live(tid, None, 0.0)
        s = self.session
        t = s.track(tid) if tid else None
        if t is not None and self._rec_had_backing:
            comp = int(max(0.0, lat + float(self.settings.get("latency_ms", 0.0)) / 1000.0) * SR)
            if comp < len(rec):
                rec = rec[comp:]
        if t is None or len(rec) < SR // 5:
            self.statusBar().showMessage("Запись слишком короткая — отброшена", 4000)
            self._update_edit_buttons()
            return
        try:
            fn = os.path.join(audio_io.takes_dir(), time.strftime("take_%Y%m%d_%H%M%S.wav"))
            audio_io.save_audio(fn, rec[None, :], SR, "WAV 24-bit")
            self.statusBar().showMessage(f"Запись сохранена на дорожку «{t.name}» (копия: {fn}). Нажмите ✨ Auto-Mix или ▶ Play.", 10000)
        except Exception:
            self.statusBar().showMessage("Запись добавлена на дорожку (копию в файл сохранить не удалось)", 6000)
        start = self.rec_start
        s.record_into_track(tid, rec, start, replace_all=(t.audio is None or start < int(0.05 * SR)))
        self.seek(start / SR)
        self._update_edit_buttons()

    # =============================================================== рендер ===
    def _render_started(self):
        self._render_t0 = time.time()
        self.prog.show()
        self.lbl_render.setText("Обработка…")

    def _render_progress(self, stage, f):
        names = {"pitch": "анализ тона", "tune": "автотюн", "comp": "компрессор", "reverb": "реверберация",
                 "delay": "дилей", "eq": "эквалайзер", "deess": "де-эссер", "gate": "гейт", "sat": "насыщение",
                 "limiter": "лимитер", "gain": "утилита", "breath": "вдохи", "rider": "rider", "exciter": "exciter",
                 "doubler": "doubler", "harmony": "гармония", "unmask": "unmasker"}
        self.lbl_render.setText("Обработка: " + names.get(stage, stage))

    def _update_duration(self):
        res = self.session.result
        n = max(self.session.project.total_samples(), 0 if res is None else res.out.shape[1])
        self.state.set_duration(n / SR)

    @guard
    def _render_done(self, res):
        self.prog.hide()
        self.lbl_render.setText(f"Готово за {time.time() - self._render_t0:.1f} с")
        if self.engine.mode == "play":
            self.engine.set_buffer(res.out)
            self._apply_loop()
        self.tracks.set_mix_wave(res.out)
        for w in self.fx_windows.values():
            w.set_result(res)
        self._update_duration()
        if not getattr(self, "_fit_done", False) and res.out.shape[1] > 0:
            self._fit_done = True
            self.state.fit()

    def _render_failed(self, msg):
        self.prog.hide()
        self.lbl_render.setText("")
        self.statusBar().showMessage("Ошибка обработки: " + msg, 12000)

    # ================================================================ тики ====
    def _update_time(self):
        self.lbl_time.setText(fmt_time(self.playhead))

    def _tick(self):
        try:
            eng = self.engine
            mode = eng.mode
            if mode == "play":
                self.playhead = eng.pos_seconds
                if eng.finished:
                    self.stop_all()
            elif mode == "record":
                self.playhead = max(0.0, self.rec_start / SR + (eng.pos - eng._count_in) / float(eng.fs))
                arr, off = eng.live_tail(30.0)
                if len(arr) and self.rec_tid:
                    cut = eng._count_in
                    if off + len(arr) > cut:
                        a0 = max(0, cut - off)
                        arr2 = arr[a0:]
                        t0 = self.rec_start / SR + max(0, off - cut) / eng.rec_sr
                        if len(arr2):
                            self.tracks.set_live(self.rec_tid, arr2 if eng.rec_sr == SR else audio_io.resample(arr2, eng.rec_sr, SR), t0)
                if eng.in_peak >= 0.99:
                    self.statusBar().showMessage("Клиппинг на входе! Уменьшите громкость микрофона.", 1500)
            if mode in ("play", "record", "monitor"):
                self.meter_in.set_level(eng.in_peak)
                self.meter_out.set_level(eng.out_peak)
                self._update_time()
                self.state.set_playhead(self.playhead)
                for w in self.fx_windows.values():
                    w.set_playhead(self.playhead)
                self.tracks.update_meters(self.session.result, int(self.playhead * SR), eng.out_peak)
                if eng.last_error:
                    self.statusBar().showMessage("Ошибка звукового потока: " + eng.last_error, 6000)
                    eng.last_error = ""
            else:
                self.meter_in.set_level(0.0)
                self.meter_out.set_level(0.0)
                self.tracks.update_meters(None, 0, 0.0)
            self._tick_errors = 0
        except Exception:
            import sys
            self._tick_errors += 1
            if self._tick_errors <= 3:
                safety.report(*sys.exc_info(), where="tick", parent=self, status=self.statusBar().showMessage)
            if self._tick_errors > 5 and self.engine.mode != "idle":
                try:
                    self.engine.stop()
                except Exception:
                    pass

    def closeEvent(self, e):
        try:
            self.stop_all()
        except Exception:
            pass
        if not self._confirm_discard():
            e.ignore()
            return
        try:
            self._close_fx_windows()
            self.engine.stop()
            self.settings["geometry"] = [self.width(), self.height()]
            save_settings(self.settings)
            self._remove_autosave()
        finally:
            super().closeEvent(e)
