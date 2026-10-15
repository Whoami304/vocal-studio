"""Главное окно Vocal Studio: дорожки + плагины (как в DAW), запись, воспроизведение, экспорт."""
from __future__ import annotations

import os
import time

import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QColorDialog, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QMainWindow,
                             QMenu, QMessageBox, QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget)

from .. import audio_io
from ..audio_io import AudioEngine, SR, to_mono, to_stereo
from ..session import MASTER, Session, load_settings, save_settings
from . import theme
from .chain_panel import ChainPanel
from .dialogs import (AutoMixDialog, ExportDialog, SettingsDialog, ShortcutsDialog, TourDialog, WelcomeDialog)
from .timeline import TimelineState, TracksPanel
from .widgets import LevelMeter

APP_NAME = "Vocal Studio"
AUDIO_FILTER = "Аудио (*.wav *.flac *.mp3 *.ogg *.aiff *.aif *.m4a);;Все файлы (*.*)"
PROJECT_FILTER = "Проект Vocal Studio (*.vsp)"


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
            self.resize(max(1180, int(gw)), max(760, int(gh)))
        except Exception:
            self.resize(1440, 900)
        self.setMinimumSize(1200, 720)
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

        self._build_ui()
        self._build_menu()
        self._connect()
        self._structure_changed()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        if not self.engine.available:
            self.statusBar().showMessage("Аудио-подсистема недоступна — запись и воспроизведение отключены, "
                                         "но обработка файлов работает.", 15000)
        self._update_title()
        self._update_quickstart()
        if not self.settings.get("welcome_done"):
            QTimer.singleShot(400, self._welcome)

    # ================================================================ UI ====
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # --- кнопки, вынесенные из верхней панели (логотип и проект — в углу меню)
        self.btn_new = QPushButton("Новый")
        self.btn_open_proj = QPushButton("Открыть проект")
        self.btn_save = QPushButton("Сохранить")
        self.btn_bypass = QPushButton("Original (A/B)")
        self.btn_bypass.setObjectName("abtn")
        self.btn_bypass.setMinimumWidth(116)
        self.btn_bypass.setCheckable(True)
        self.btn_bypass.setToolTip("Сравнить: слушать без всех плагинов (Ctrl+B)")
        self.btn_bypass.setStatusTip("Original (A/B): послушать запись без обработки, чтобы сравнить с результатом")
        self.btn_undo = QPushButton("Отменить запись")
        self.btn_undo.setToolTip("Вернуть предыдущую версию записи на выбранной дорожке (Ctrl+Z)")
        self.btn_settings = QPushButton("Настройки аудио")
        self.btn_help = QPushButton("?  Помощь")
        for b in (self.btn_new, self.btn_open_proj, self.btn_save, self.btn_undo, self.btn_help):
            b.hide()
        corner = QWidget()
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 0, 12, 0)
        cl.setSpacing(12)
        logo = QLabel("VOCAL STUDIO")
        logo.setObjectName("logo")
        self.lbl_project = QLabel("")
        self.lbl_project.setStyleSheet("font-weight: 600;")
        cl.addWidget(self.lbl_project)
        cl.addWidget(logo)
        corner.setMinimumWidth(corner.sizeHint().width() + 90)
        self.menuBar().setCornerWidget(corner, Qt.Corner.TopRightCorner)

        # --- транспорт
        tr = QFrame()
        tr.setObjectName("transport")
        trl = QHBoxLayout(tr)
        trl.setContentsMargins(14, 8, 14, 8)
        trl.setSpacing(8)
        self.btn_rec = QPushButton("●  REC")
        self.btn_rec.setObjectName("rec")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setToolTip("Запись с микрофона на выбранную дорожку (R)")
        self.btn_rec.setStatusTip("REC: записать голос с микрофона на выбранную (красная точка) дорожку. Бит играет в наушниках.")
        self.btn_play = QPushButton("▶  PLAY")
        self.btn_play.setObjectName("play")
        self.btn_play.setCheckable(True)
        self.btn_play.setToolTip("Воспроизвести / остановить (Пробел)")
        self.btn_play.setStatusTip("Воспроизвести весь проект (пробел)")
        self.btn_stop = QPushButton("■  STOP")
        self.btn_home = QPushButton("|◀")
        self.btn_home.setToolTip("В начало (Home)")
        for b in (self.btn_rec, self.btn_play, self.btn_stop, self.btn_home):
            b.setFixedHeight(34)
            b.setFixedWidth(104 if b is self.btn_rec else (92 if b is self.btn_play else (82 if b is self.btn_stop else 44)))
            trl.addWidget(b)
        self.lbl_time = QLabel("00:00.00")
        self.lbl_time.setObjectName("time")
        self.lbl_time.setMinimumWidth(120)
        trl.addSpacing(8)
        trl.addWidget(self.lbl_time)
        self.btn_loop = QPushButton("Loop")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setFixedHeight(28)
        self.btn_loop.setToolTip("Зациклить воспроизведение")
        trl.addWidget(self.btn_loop)
        trl.addSpacing(10)
        self.btn_arm = QPushButton("Тест микрофона")
        self.btn_arm.setCheckable(True)
        self.btn_arm.setFixedHeight(28)
        self.btn_arm.setToolTip("Включить вход микрофона, чтобы выставить уровень до записи")
        self.btn_arm.setStatusTip("Проверка микрофона: показывает уровень сигнала до записи. Пики не должны касаться красной зоны.")
        trl.addWidget(self.btn_arm)
        for text, attr in (("IN", "meter_in"), ("OUT", "meter_out")):
            lab = QLabel(text)
            lab.setObjectName("muted")
            trl.addWidget(lab)
            m = LevelMeter()
            m.setFixedWidth(84)
            m.setFixedHeight(12)
            setattr(self, attr, m)
            trl.addWidget(m)
        trl.addStretch(1)
        trl.addWidget(self.btn_bypass)
        self.btn_bypass.setFixedHeight(30)
        self.btn_settings.setText("Микрофон / звук…")
        self.btn_settings.setMinimumWidth(138)
        self.btn_settings.setToolTip("Выбор микрофона и наушников, задержка записи")
        self.btn_settings.setStatusTip("Настройки аудио: выбрать микрофон и наушники, подстроить задержку записи")
        self.btn_settings.setFixedHeight(30)
        trl.addWidget(self.btn_settings)
        self.btn_open = QPushButton("Открыть вокал…")
        self.btn_backing = QPushButton("Загрузить бит…")
        self.btn_export = QPushButton("Экспорт…")
        self.btn_export.setStyleSheet(f"QPushButton {{ background: #1f3f3b; border-color: {theme.ACCENT}; color: {theme.ACCENT}; font-weight: 700; }}")
        self.btn_open.hide()
        self.btn_backing.hide()
        self.btn_export.setFixedHeight(30)
        self.btn_export.setMinimumWidth(96)
        trl.addWidget(self.btn_export)
        v.addWidget(tr)

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
        texts = ["①  Добавьте голос", "②  ✨ Auto-Mix", "③  Послушайте  ▶", "④  Сохраните"]
        tips = ["Запишите голос с микрофона или откройте готовый файл", "Программа сама настроит автотюн, эквалайзер, компрессор и эффекты",
                "Нажмите Play и сравните с кнопкой Original", "Экспортируйте готовый трек в WAV или MP3"]
        for i, (tx, tp) in enumerate(zip(texts, tips)):
            b = QPushButton(tx)
            b.setObjectName("step")
            b.setToolTip(tp)
            b.setStatusTip(tp)
            b.clicked.connect(lambda _=False, k=i: self._step_clicked(k))
            bl.addWidget(b)
            self.steps.append(b)
        bl.addStretch(1)
        self.btn_tour = QPushButton("Тур по программе")
        self.btn_tour.setFixedHeight(26)
        bl.addWidget(self.btn_tour)
        self.btn_banner_close = QPushButton("✕")
        self.btn_banner_close.setObjectName("close")
        self.btn_banner_close.setToolTip("Скрыть подсказки (вернуть: Помощь → Быстрый старт)")
        bl.addWidget(self.btn_banner_close)
        bw = QWidget()
        bwl = QVBoxLayout(bw)
        bwl.setContentsMargins(12, 6, 12, 0)
        bwl.addWidget(self.banner)
        v.addWidget(bw)
        self.banner_wrap = bw
        if self.settings.get("hide_quickstart"):
            bw.hide()

        # --- дорожки / плагины
        self.tracks = TracksPanel(self.session, self.state)
        self.chain = ChainPanel(self.session)
        self.split = QSplitter(Qt.Orientation.Vertical)
        up = QWidget()
        ul = QVBoxLayout(up)
        ul.setContentsMargins(12, 8, 12, 4)
        ul.addWidget(self.tracks)
        dn = QWidget()
        dl = QVBoxLayout(dn)
        dl.setContentsMargins(12, 4, 12, 10)
        dl.addWidget(self.chain)
        self.split.addWidget(up)
        self.split.addWidget(dn)
        self.split.setStretchFactor(0, 4)
        self.split.setStretchFactor(1, 6)
        self.split.setChildrenCollapsible(False)
        self.split.setSizes([330, 470])
        v.addWidget(self.split, 1)

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

    def _build_menu(self):
        mb = self.menuBar()

        def act(menu, text, fn, shortcut=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(lambda _=False: fn())
            menu.addAction(a)
            return a

        m = mb.addMenu("Файл")
        act(m, "Новый проект", self.new_project, "Ctrl+N")
        act(m, "Открыть проект…", self.open_project, "Ctrl+O")
        act(m, "Сохранить проект", self.save_project, "Ctrl+S")
        act(m, "Сохранить проект как…", lambda: self.save_project(True), "Ctrl+Shift+S")
        m.addSeparator()
        act(m, "Открыть вокал из файла…", self.open_vocal, "Ctrl+I")
        act(m, "Загрузить бит / инструментал…", self.open_backing)
        m.addSeparator()
        act(m, "Экспорт…", self.export, "Ctrl+E")
        act(m, "Микрофон и звук…", self.open_settings)
        m.addSeparator()
        act(m, "Выход", self.close, "Ctrl+Q")
        m = mb.addMenu("Правка")
        act(m, "Отменить последнюю запись на дорожке", self.undo_take, "Ctrl+Z")
        act(m, "Original (A/B): послушать без обработки", self.btn_bypass.toggle, "Ctrl+B")
        m = mb.addMenu("Дорожка")
        act(m, "Добавить вокальную дорожку", lambda: self.add_vocal_track(), "Ctrl+T")
        act(m, "Добавить дорожку для бита", lambda: self.open_backing())
        act(m, "✨ Auto-Mix выбранной дорожки…", lambda: self.automix(self.session.sel_track), "Ctrl+M")
        m.addSeparator()
        act(m, "Удалить выбранную дорожку", lambda: self.track_action("remove", self.session.sel_track))
        act(m, "Очистить запись на выбранной дорожке", lambda: self.track_action("clear", self.session.sel_track))
        m = mb.addMenu("Помощь")
        act(m, "Быстрый старт (показать подсказки)", self.show_quickstart)
        act(m, "Тур по программе", self.tour)
        act(m, "Горячие клавиши", lambda: ShortcutsDialog(self).exec())
        act(m, "Приветствие", self._welcome)
        act(m, "О программе", lambda: QMessageBox.about(
            self, APP_NAME, "<b>Vocal Studio</b><br>Запись и обработка вокала: автотюн, EQ, компрессор, де-эссер, "
                            "дилей, реверберация. Всё работает локально на вашем компьютере."))

    def _connect(self):
        s = self.session
        self.btn_rec.clicked.connect(self.toggle_record)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_home.clicked.connect(lambda: self.seek(0.0))
        self.btn_open.clicked.connect(self.open_vocal)
        self.btn_backing.clicked.connect(self.open_backing)
        self.btn_export.clicked.connect(self.export)
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_undo.clicked.connect(self.undo_take)
        self.btn_bypass.toggled.connect(s.set_bypass_all)
        self.btn_arm.toggled.connect(self.toggle_arm)
        self.btn_loop.toggled.connect(lambda _: self._apply_loop())
        self.btn_new.clicked.connect(self.new_project)
        self.btn_open_proj.clicked.connect(self.open_project)
        self.btn_save.clicked.connect(lambda: self.save_project())
        self.btn_help.clicked.connect(self._help_menu)
        self.btn_tour.clicked.connect(self.tour)
        self.btn_banner_close.clicked.connect(self.hide_quickstart)

        self.tracks.request.connect(self.track_action)
        self.tracks.files_dropped.connect(self._lane_drop)
        self.tracks.btn_add_vocal.clicked.connect(lambda: self.add_vocal_track())
        self.tracks.btn_add_back.clicked.connect(self.open_backing)
        self.tracks.btn_add_file.clicked.connect(self.open_vocal)
        self.state.seek_requested.connect(self.seek)
        self.chain.seek_requested.connect(self.seek)
        self.chain.automix_requested.connect(self.automix)

        s.structure_changed.connect(self._structure_changed)
        s.track_changed.connect(lambda tid: self.tracks.refresh_headers())
        s.selection_changed.connect(self.tracks.update_selection)
        s.audio_changed.connect(self.tracks.update_audio)
        s.audio_changed.connect(lambda _tid: self._update_duration())
        s.project_replaced.connect(self._project_replaced)
        s.dirty_changed.connect(lambda _d: self._update_title())
        s.render_started.connect(self._render_started)
        s.render_done.connect(self._render_done)
        s.render_failed.connect(self._render_failed)
        s.render_progress.connect(self._render_progress)

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.toggle_play)
        QShortcut(QKeySequence("R"), self, activated=self.toggle_record)
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, activated=lambda: self.seek(0.0))

    def _help_menu(self):
        m = QMenu(self)
        a1 = m.addAction("Быстрый старт (подсказки)")
        a2 = m.addAction("Тур по программе")
        a3 = m.addAction("Горячие клавиши")
        a = m.exec(self.btn_help.mapToGlobal(self.btn_help.rect().bottomLeft()))
        if a == a1:
            self.show_quickstart()
        elif a == a2:
            self.tour()
        elif a == a3:
            ShortcutsDialog(self).exec()

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
        has_audio = any(t.audio is not None and t.kind == "vocal" for t in s.project.tracks) or \
            any(t.audio is not None for t in s.project.tracks)
        has_fx = any(t.plugins for t in s.project.tracks)
        done = [has_audio, has_fx and has_audio, self._played, self._exported]
        cur = next((i for i, d in enumerate(done) if not d), len(done))
        for i, b in enumerate(self.steps):
            b.setProperty("done", bool(done[i]))
            b.setProperty("cur", i == cur)
            txt = b.text()
            if done[i] and not txt.startswith("✓"):
                b.setText("✓ " + txt[3:].strip() if txt[1:2] == " " or True else txt)
            elif not done[i] and txt.startswith("✓"):
                b.setText(["①  Добавьте голос", "②  ✨ Auto-Mix", "③  Послушайте  ▶", "④  Сохраните"][i])
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

    def new_project(self):
        self.stop_all()
        if self._confirm_discard():
            self.session.new_project()
            self.seek(0.0)

    def open_project(self):
        self.stop_all()
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", self.settings.get("last_dir", ""), PROJECT_FILTER)
        if path:
            try:
                self.session.open_project(path)
                self.settings["last_dir"] = os.path.dirname(path)
                self.seek(0.0)
                self.statusBar().showMessage(f"Проект открыт: {os.path.basename(path)}", 5000)
            except Exception as e:
                QMessageBox.warning(self, APP_NAME, f"Не удалось открыть проект:\n{e}")

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
            self.statusBar().showMessage(f"Проект сохранён: {path}", 6000)
            self._update_title()
            return True
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Не удалось сохранить проект:\n{e}")
            return False

    def _project_replaced(self):
        self._track_ids = ()
        self._played = False
        self._exported = False
        self._update_title()

    def _update_title(self):
        p = self.session.project
        self.setWindowTitle(f"{p.name}{' *' if p.dirty else ''} — {APP_NAME}")
        self.lbl_project.setText(f"{p.name}{' *' if p.dirty else ''}")

    # ============================================================ дорожки =====
    def _structure_changed(self):
        ids = tuple((t.id, t.kind) for t in self.session.project.tracks)
        if ids != self._track_ids:
            self._track_ids = ids
            self.tracks.rebuild()
        else:
            self.tracks.refresh_headers()
            self.tracks.update_selection()
        self._update_quickstart()
        self.btn_undo.setEnabled(any(t.undo for t in self.session.project.tracks))
        self._update_title()

    def add_vocal_track(self, select=True):
        t = self.session.add_track("vocal")
        return t

    def prepare_record_track(self):
        """Выбирает (или создаёт) дорожку для записи и помечает её красной точкой."""
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

    def track_action(self, action, tid):
        s = self.session
        if tid is None:
            return
        if action == "select":
            s.select(tid)
        elif action == "fx":
            s.select(tid)
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
            t = s.track(tid)
            if t and t.audio is not None:
                s.set_track_audio(tid, None)
        elif action == "remove":
            t = s.track(tid)
            if t is None:
                return
            if t.audio is not None or t.plugins:
                if QMessageBox.question(self, APP_NAME, f"Удалить дорожку «{t.name}» вместе с записью и плагинами?") != QMessageBox.StandardButton.Yes:
                    return
            if self.engine.mode != "idle":
                self.stop_all()
            s.remove_track(tid)
        elif action == "automix":
            self.automix(tid)

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

    # ============================================================== файлы =====
    def open_vocal(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть вокал", self.settings.get("last_dir", ""), AUDIO_FILTER)
        if path:
            self.load_audio_file(path, "vocal")

    def open_backing(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить бит / инструментал", self.settings.get("last_dir", ""), AUDIO_FILTER)
        if path:
            self.load_audio_file(path, "audio")

    def load_audio_file(self, path, kind, tid=None):
        try:
            self.stop_all()
            data = audio_io.load_audio(path, SR)
            name = os.path.splitext(os.path.basename(path))[0]
            audio = to_mono(data)[None, :] if kind == "vocal" else to_stereo(data)
            s = self.session
            if tid is not None and s.track(tid) is not None:
                s.set_track_audio(tid, audio, os.path.basename(path))
                t = s.track(tid)
            else:
                t = s.add_track(kind, name=name[:24] or None, audio=audio, source=os.path.basename(path))
            self.settings["last_dir"] = os.path.dirname(path)
            self.seek(0.0)
            self.statusBar().showMessage(f"Загружено: {os.path.basename(path)}  →  дорожка «{t.name}»", 6000)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Не удалось открыть файл:\n{e}")

    def _lane_drop(self, tid, urls):
        t = self.session.track(tid)
        if t is None:
            return
        self.load_audio_file(urls[0], t.kind, tid=tid)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            kind = "audio" if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else "vocal"
            for u in urls:
                self.load_audio_file(u, kind)

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

    # ====================================================== воспроизведение ===
    def seek(self, t: float):
        self.playhead = max(0.0, t)
        if self.engine.mode == "play":
            self.engine.pos = int(self.playhead * SR)
        self._update_time()
        self.state.set_playhead(self.playhead)
        self.chain.set_playhead(self.playhead)

    def toggle_play(self):
        if self.engine.mode == "record":
            self.toggle_record()
        elif self.engine.mode == "play":
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
            self.statusBar().showMessage("Нечего воспроизводить — запишите голос (● REC) или откройте файл", 5000)
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
        self._played = True
        self._update_quickstart()

    def _apply_loop(self):
        res = self.session.result
        if self.btn_loop.isChecked() and res is not None and res.out.shape[1] > 0:
            self.engine.loop = (0, res.out.shape[1])
        else:
            self.engine.loop = None

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

    # ================================================================ запись ==
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
                self.statusBar().showMessage("Говорите/пойте — уровень отображается на шкале IN. Не должен касаться красной зоны.", 8000)
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
        self.state.follow = True
        if self.engine.warning:
            self.statusBar().showMessage(self.engine.warning, 8000)
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
        if t is None or len(rec) < SR // 20:
            self.statusBar().showMessage("Запись слишком короткая — отброшена", 4000)
            return
        if self._rec_had_backing:
            comp = int(max(0.0, lat + float(self.settings.get("latency_ms", 0.0)) / 1000.0) * SR)
            if comp < len(rec):
                rec = rec[comp:]
        try:
            fn = os.path.join(audio_io.takes_dir(), time.strftime("take_%Y%m%d_%H%M%S.wav"))
            audio_io.save_audio(fn, rec[None, :], SR, "WAV 24-bit")
            self.statusBar().showMessage(f"Запись сохранена на дорожку «{t.name}» (копия: {fn}). Нажмите ✨ Auto-Mix или ▶ Play.", 10000)
        except Exception:
            self.statusBar().showMessage("Запись добавлена на дорожку (копию в файл сохранить не удалось)", 6000)
        start = self.rec_start
        if t.audio is None or start < int(0.05 * SR):
            new = rec[None, :]
        else:
            old = t.mono()
            if len(old) < start:
                old = np.pad(old, (0, start - len(old)))
            tail = old[start + len(rec):] if len(old) > start + len(rec) else np.zeros(0, np.float32)
            new = np.concatenate([old[:start], rec, tail])[None, :]
        s.set_track_audio(tid, new, time.strftime("запись %H:%M:%S"))
        self.seek(start / SR)

    def undo_take(self):
        if self.engine.mode == "record":
            return
        tid = self.session.sel_track
        if tid and tid != MASTER and self.session.undo_audio(tid):
            self.statusBar().showMessage("Предыдущая версия записи восстановлена", 3000)
        else:
            self.statusBar().showMessage("Нечего отменять на выбранной дорожке", 3000)

    # =============================================================== рендер ===
    def _render_started(self):
        self._render_t0 = time.time()
        self.prog.show()
        self.lbl_render.setText("Обработка…")

    def _render_progress(self, stage, f):
        names = {"pitch": "анализ тона", "tune": "автотюн", "comp": "компрессор", "reverb": "реверберация",
                 "delay": "дилей", "eq": "эквалайзер", "deess": "де-эссер", "gate": "гейт", "sat": "насыщение",
                 "limiter": "лимитер", "gain": "утилита"}
        self.lbl_render.setText("Обработка: " + names.get(stage, stage))

    def _update_duration(self):
        res = self.session.result
        n = max(self.session.project.total_samples(), 0 if res is None else res.out.shape[1])
        self.state.set_duration(n / SR)

    def _render_done(self, res):
        self.prog.hide()
        self.lbl_render.setText(f"Готово за {time.time() - self._render_t0:.1f} с")
        if self.engine.mode == "play":
            self.engine.set_buffer(res.out)
            self._apply_loop()
        self.tracks.set_mix_wave(res.out)
        self.chain.set_result(res)
        self._update_duration()
        if not getattr(self, "_fit_done", False) and res.out.shape[1] > 0:
            self._fit_done = True
            self.state.fit()

    def _render_failed(self, msg):
        self.prog.hide()
        self.lbl_render.setText("")
        self.statusBar().showMessage("Ошибка обработки: " + msg, 10000)

    # ================================================================ тики ====
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
            arr, off = eng.live_tail(30.0)
            if len(arr) and self.rec_tid:
                t0 = self.rec_start / SR + off / eng.rec_sr
                self.tracks.set_live(self.rec_tid, arr if eng.rec_sr == SR else audio_io.resample(arr, eng.rec_sr, SR), t0)
            if eng.in_peak >= 0.99:
                self.statusBar().showMessage("Клиппинг на входе! Уменьшите громкость микрофона.", 1500)
        if mode in ("play", "record", "monitor"):
            self.meter_in.set_level(eng.in_peak)
            self.meter_out.set_level(eng.out_peak)
            self._update_time()
            self.state.set_playhead(self.playhead)
            self.chain.set_playhead(self.playhead)
            self.tracks.update_meters(self.session.result, int(self.playhead * SR), eng.out_peak)
        else:
            self.meter_in.set_level(0.0)
            self.meter_out.set_level(0.0)
            self.tracks.update_meters(None, 0, 0.0)

    def closeEvent(self, e):
        self.stop_all()
        if not self._confirm_discard():
            e.ignore()
            return
        try:
            self.engine.stop()
            self.settings["geometry"] = [self.width(), self.height()]
            save_settings(self.settings)
        finally:
            super().closeEvent(e)
