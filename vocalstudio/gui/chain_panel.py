"""Панель плагинов выбранной дорожки: список (цепочка) слева, редактор плагина справа."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QInputDialog, QLabel, QMenu, QMessageBox, QPushButton, QScrollArea,
                             QSizePolicy, QVBoxLayout, QWidget)

from ..plugins import CATEGORY_ORDER, CATEGORY_TITLES, CHAIN_PRESET_ORDER, CHAIN_PRESETS, PLUGINS
from ..session import (MASTER, delete_user_chain, list_user_chains, load_user_chain, save_user_chain)
from . import theme
from .editors import make_editor


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


class ChainPanel(QWidget):
    seek_requested = pyqtSignal(float)
    automix_requested = pyqtSignal(str)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.s = session
        self.editor = None
        self.editor_key = None
        self.last_res = None
        self._rebuilding = False

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # ---- левая колонка
        left = QFrame()
        left.setObjectName("panel")
        left.setFixedWidth(372)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 10, 10)
        ll.setSpacing(6)
        self.title = QLabel("ПЛАГИНЫ")
        self.title.setObjectName("title")
        ll.addWidget(self.title)
        hint = QLabel("Звук идёт сверху вниз через каждый плагин. Нажмите на плагин, чтобы настроить его.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        ll.addWidget(hint)
        self.list_box = QVBoxLayout()
        self.list_box.setSpacing(3)
        lw = QWidget()
        lw.setLayout(self.list_box)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setWidget(lw)
        self.list_scroll = sc
        ll.addWidget(sc, 1)
        self.btn_add = QPushButton("＋  Добавить плагин  ▾")
        self.btn_add.setObjectName("primary2")
        self.btn_add.setToolTip("Добавить автотюн, эквалайзер, компрессор, реверберацию и другие плагины на эту дорожку")
        self.btn_add.setStatusTip("Добавить плагин на выбранную дорожку: автотюн, EQ, компрессор, реверберация, дилей…")
        self.btn_add.clicked.connect(self._add_menu)
        ll.addWidget(self.btn_add)
        row = QHBoxLayout()
        self.btn_chains = QPushButton("Готовые цепочки ▾")
        self.btn_chains.setToolTip("Применить готовый набор плагинов целиком (Natural Pop, Hard Tune, Rap…)")
        self.btn_chains.setStatusTip("Применить готовую цепочку плагинов: Natural Pop, Hard Tune, Rap, Ballad…")
        self.btn_chains.clicked.connect(self._chains_menu)
        self.btn_auto = QPushButton("✨  Auto-Mix")
        self.btn_auto.setObjectName("magic")
        self.btn_auto.setToolTip("Программа проанализирует ваш голос (тональность, тембр, динамику) и сама настроит все плагины")
        self.btn_auto.setStatusTip("Auto-Mix: анализ голоса и автоматическая настройка тональности, EQ, компрессора, реверберации и громкости")
        self.btn_auto.clicked.connect(lambda: self.automix_requested.emit(self.s.sel_track or ""))
        row.addWidget(self.btn_chains, 1)
        row.addWidget(self.btn_auto, 1)
        ll.addLayout(row)
        root.addWidget(left)

        # ---- правая часть: редактор
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.placeholder = QLabel("")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setObjectName("muted")
        self.placeholder.setWordWrap(True)
        self.scroll.setWidget(self.placeholder)
        root.addWidget(self.scroll, 1)

        session.structure_changed.connect(self.rebuild)
        session.selection_changed.connect(self.rebuild)
        session.plugins_replaced.connect(self._replaced)
        session.project_replaced.connect(self._reset_editor)
        self.rebuild()

    # ------------------------------------------------------------------
    def _scope(self):
        return self.s.sel_track

    def _plugins(self):
        sc = self._scope()
        return self.s.scope_plugins(sc) if sc else []

    def _name(self):
        sc = self._scope()
        if sc == MASTER:
            return "MASTER — общий выход"
        t = self.s.track(sc) if sc else None
        return t.name if t else ""

    def _reset_editor(self):
        self.editor = None
        self.editor_key = None

    def rebuild(self):
        sc = self._scope()
        # список
        while self.list_box.count():
            it = self.list_box.takeAt(0)
            w_ = it.widget()
            if w_ is not None:
                w_.setParent(None)
                w_.deleteLater()
        plugins = self._plugins()
        if sc is None:
            self.title.setText("ПЛАГИНЫ")
        else:
            self.title.setText(f"ПЛАГИНЫ  ·  {self._name()}")
        if sc and self.s.sel_plugin is None and plugins:
            self.s.sel_plugin = plugins[0]["id"]
        for i, p in enumerate(plugins):
            row = PluginRow(p, i, p["id"] == self.s.sel_plugin)
            row.clicked.connect(lambda pid: self.s.select(self._scope(), pid))
            row.toggled.connect(lambda pid, on: self.s.set_plugin_on(self._scope(), pid, on))
            row.moved.connect(lambda pid, d: self.s.move_plugin(self._scope(), pid, d))
            row.removed.connect(lambda pid: self.s.remove_plugin(self._scope(), pid))
            self.list_box.addWidget(row)
        if sc and not plugins:
            e = QLabel("Плагинов пока нет.\nНажмите «Добавить плагин» или выберите готовую цепочку — или доверьте всё кнопке ✨ Auto-Mix.")
            e.setObjectName("muted")
            e.setWordWrap(True)
            e.setAlignment(Qt.AlignmentFlag.AlignCenter)
            e.setMinimumHeight(110)
            self.list_box.addWidget(e)
        self.list_box.addStretch(1)
        self.btn_add.setEnabled(bool(sc))
        self.btn_chains.setEnabled(bool(sc))
        t = self.s.track(sc) if sc and sc != MASTER else None
        self.btn_auto.setEnabled(bool(t) and t.kind == "vocal")
        self._show_editor()

    def _show_editor(self):
        sc = self._scope()
        pl = self.s.find_plugin(sc, self.s.sel_plugin) if sc and self.s.sel_plugin else None
        key = (sc, self.s.sel_plugin) if pl else None
        if key == self.editor_key and self.editor is not None:
            self.editor.refresh()
            return
        self.editor_key = key
        if pl is None:
            self.editor = None
            if sc is None:
                self.placeholder.setText("Нет выбранной дорожки.\n\nДобавьте дорожку кнопкой «＋ Добавить вокальную дорожку» "
                                         "или нажмите ● REC — дорожка создастся сама.")
            elif not self._plugins():
                self.placeholder.setText("Здесь появится настройка плагина.\n\nДобавьте первый плагин слева.")
            else:
                self.placeholder.setText("Выберите плагин в списке слева.")
            w = self.scroll.takeWidget()
            if w is not None and w is not self.placeholder:
                w.deleteLater()
            self.placeholder = QLabel(self.placeholder.text())
            self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.placeholder.setObjectName("muted")
            self.placeholder.setWordWrap(True)
            self.scroll.setWidget(self.placeholder)
            return
        ed = make_editor(self.s, sc, pl)
        ed.seek_requested.connect(self.seek_requested.emit)
        self.editor = ed
        w = self.scroll.takeWidget()
        if w is not None:
            w.deleteLater()
        self.scroll.setWidget(ed)
        self._push_result(ed)

    def _push_result(self, ed):
        res = self.last_res
        if res is None:
            return
        tr = res.tracks.get(self._scope()) if self._scope() != MASTER else None
        ed.set_result(res, tr)

    def _replaced(self, scope):
        if self.editor is not None and scope == self._scope():
            self.editor.refresh()

    def set_result(self, res):
        self.last_res = res
        if self.editor is not None:
            self._push_result(self.editor)

    def set_playhead(self, t):
        if self.editor is not None:
            self.editor.set_playhead(t)

    # ---- меню ----------------------------------------------------------
    def _add_menu(self):
        sc = self._scope()
        if not sc:
            return
        m = QMenu(self)
        acts = {}
        for cat in CATEGORY_ORDER:
            items = [d for d in PLUGINS.values() if d.category == cat]
            if not items:
                continue
            m.addSection(CATEGORY_TITLES[cat])
            for d in items:
                a = m.addAction(f"{d.icon}   {d.title}  —  {d.tagline}")
                acts[a] = d.type
        a = m.exec(self.btn_add.mapToGlobal(self.btn_add.rect().bottomLeft()))
        if a in acts:
            self.s.add_plugin(sc, acts[a])

    def _chains_menu(self):
        sc = self._scope()
        if not sc:
            return
        m = QMenu(self)
        acts = {}
        m.addSection("Готовые цепочки")
        for name in CHAIN_PRESET_ORDER:
            acts[m.addAction(name)] = ("factory", name)
        users = list_user_chains()
        if users:
            m.addSection("Мои цепочки")
            for name in users:
                acts[m.addAction("★ " + name)] = ("user", name)
        m.addSeparator()
        acts[m.addAction("Сохранить текущую цепочку…")] = ("save", None)
        if users:
            sub = m.addMenu("Удалить мою цепочку")
            for name in users:
                acts[sub.addAction(name)] = ("del", name)
        a = m.exec(self.btn_chains.mapToGlobal(self.btn_chains.rect().bottomLeft()))
        if a not in acts:
            return
        kind, name = acts[a]
        if kind in ("factory", "user"):
            if self._plugins():
                ok = QMessageBox.question(self, "Заменить цепочку?",
                                          "Текущие плагины этой дорожки будут заменены выбранной цепочкой. Продолжить?")
                if ok != QMessageBox.StandardButton.Yes:
                    return
            if kind == "factory":
                self.s.apply_chain_preset(sc, name)
            else:
                try:
                    self.s.apply_chain(sc, load_user_chain(name))
                except Exception as e:
                    QMessageBox.warning(self, "Vocal Studio", f"Не удалось загрузить цепочку: {e}")
        elif kind == "save":
            n, ok = QInputDialog.getText(self, "Сохранить цепочку", "Название цепочки:")
            if ok and n.strip():
                import copy
                save_user_chain(n.strip(), copy.deepcopy(self._plugins()))
        elif kind == "del":
            delete_user_chain(name)
