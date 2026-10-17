"""Окно плагинов дорожки (кнопка FX): слева цепочка и список доступных плагинов, справа настройки выбранного."""
from __future__ import annotations

import copy

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
                             QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from ..plugins import CATEGORY_ORDER, CATEGORY_TITLES, CHAIN_PRESET_ORDER, PLUGINS
from ..session import MASTER, delete_user_chain, list_user_chains, load_user_chain, save_user_chain
from . import theme
from .chain_panel import PluginRow
from .editors import make_editor


class FxWindow(QWidget):
    seek_requested = pyqtSignal(float)
    automix_requested = pyqtSignal(str)
    closed = pyqtSignal(str)

    def __init__(self, session, scope: str, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.s = session
        self.scope = scope
        self.sel_plugin: str | None = None
        self.editor = None
        self.editor_key = None
        self.last_res = None
        self.resize(1320, 780)
        self.setMinimumSize(980, 560)

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ---------------- левая колонка ----------------
        left = QFrame()
        left.setObjectName("panel")
        left.setFixedWidth(392)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 10, 10)
        ll.setSpacing(6)
        self.title = QLabel("")
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
        self.list_scroll = QScrollArea()
        self.list_scroll.setWidgetResizable(True)
        self.list_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.list_scroll.setWidget(lw)
        self.list_scroll.setMinimumHeight(150)
        ll.addWidget(self.list_scroll, 3)

        row = QHBoxLayout()
        self.btn_chains = QPushButton("Готовые цепочки ▾")
        self.btn_chains.setToolTip("Применить готовый набор плагинов целиком (Natural Pop, Hard Tune, Rap, Studio…)")
        self.btn_chains.setStatusTip("Применить готовую цепочку плагинов целиком")
        self.btn_chains.clicked.connect(self._chains_menu)
        self.btn_auto = QPushButton("✨  Auto-Mix")
        self.btn_auto.setObjectName("magic")
        self.btn_auto.setToolTip("Программа проанализирует голос и сама настроит плагины")
        self.btn_auto.clicked.connect(lambda: self.automix_requested.emit(self.scope))
        row.addWidget(self.btn_chains, 1)
        row.addWidget(self.btn_auto, 1)
        ll.addLayout(row)

        cap = QLabel("ДОСТУПНЫЕ ПЛАГИНЫ")
        cap.setObjectName("title")
        ll.addSpacing(4)
        ll.addWidget(cap)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск: например «реверб», «автотюн», «шум»…")
        self.search.textChanged.connect(self._fill_available)
        ll.addWidget(self.search)
        self.avail = QListWidget()
        self.avail.setStyleSheet("QListWidget { background: #14171e; border: 1px solid #2f3441; border-radius: 6px; }"
                                 "QListWidget::item { padding: 5px 6px; } QListWidget::item:selected { background: #1f3f3b; }")
        self.avail.itemDoubleClicked.connect(self._avail_activated)
        self.avail.currentItemChanged.connect(self._avail_selected)
        ll.addWidget(self.avail, 4)
        self.avail_help = QLabel("Выберите плагин, чтобы прочитать, что он делает. Двойной клик — добавить в цепочку.")
        self.avail_help.setObjectName("muted")
        self.avail_help.setWordWrap(True)
        self.avail_help.setMinimumHeight(54)
        ll.addWidget(self.avail_help)
        self.btn_add = QPushButton("＋  Добавить выбранный плагин в цепочку")
        self.btn_add.setObjectName("primary2")
        self.btn_add.clicked.connect(self._add_selected)
        ll.addWidget(self.btn_add)
        root.addWidget(left)

        # ---------------- правая часть ----------------
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.placeholder = self._make_placeholder("")
        self.scroll.setWidget(self.placeholder)
        root.addWidget(self.scroll, 1)

        session.structure_changed.connect(self.rebuild)
        session.plugins_replaced.connect(self._replaced)
        self._fill_available()
        self.rebuild()

    # ---------------------------------------------------------------- helpers
    def _make_placeholder(self, text):
        l = QLabel(text)
        l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.setObjectName("muted")
        l.setWordWrap(True)
        return l

    def plugins(self) -> list:
        return self.s.scope_plugins(self.scope)

    def name(self) -> str:
        if self.scope == MASTER:
            return "MASTER — общий выход"
        t = self.s.track(self.scope)
        return t.name if t else ""

    def can_automix(self) -> bool:
        t = self.s.track(self.scope) if self.scope != MASTER else None
        return t is not None and t.kind == "vocal"

    def closeEvent(self, e):
        self.closed.emit(self.scope)
        super().closeEvent(e)

    # ------------------------------------------------------------- список
    def rebuild(self):
        if self.scope != MASTER and self.s.track(self.scope) is None:
            self.close()
            return
        nm = self.name()
        self.setWindowTitle(f"FX — {nm}")
        self.title.setText(f"ЦЕПОЧКА ПЛАГИНОВ  ·  {nm}")
        while self.list_box.count():
            it = self.list_box.takeAt(0)
            w_ = it.widget()
            if w_ is not None:
                w_.setParent(None)
                w_.deleteLater()
        plugins = self.plugins()
        if self.sel_plugin is None or self.s.find_plugin(self.scope, self.sel_plugin) is None:
            self.sel_plugin = plugins[0]["id"] if plugins else None
        for i, p in enumerate(plugins):
            row = PluginRow(p, i, p["id"] == self.sel_plugin)
            row.clicked.connect(self._select)
            row.toggled.connect(lambda pid, on: self.s.set_plugin_on(self.scope, pid, on))
            row.moved.connect(lambda pid, d: self.s.move_plugin(self.scope, pid, d))
            row.removed.connect(lambda pid: self.s.remove_plugin(self.scope, pid))
            self.list_box.addWidget(row)
        if not plugins:
            e = QLabel("Плагинов пока нет.\n\nДобавьте плагин из списка «Доступные плагины» ниже (двойной клик), "
                       "выберите готовую цепочку или нажмите ✨ Auto-Mix.")
            e.setObjectName("muted")
            e.setWordWrap(True)
            e.setAlignment(Qt.AlignmentFlag.AlignCenter)
            e.setMinimumHeight(110)
            self.list_box.addWidget(e)
        self.list_box.addStretch(1)
        self.btn_auto.setEnabled(self.can_automix())
        self._show_editor()

    def _select(self, pid):
        self.sel_plugin = pid
        self.rebuild()

    def _show_editor(self):
        pl = self.s.find_plugin(self.scope, self.sel_plugin) if self.sel_plugin else None
        key = (self.scope, self.sel_plugin) if pl else None
        if key == self.editor_key and self.editor is not None:
            self.editor.refresh()
            return
        self.editor_key = key
        w = self.scroll.takeWidget()
        if w is not None:
            w.deleteLater()
        if pl is None:
            self.editor = None
            self.scroll.setWidget(self._make_placeholder(
                "Здесь появятся настройки плагина.\n\nДобавьте первый плагин слева." if not self.plugins()
                else "Выберите плагин в списке слева."))
            return
        ed = make_editor(self.s, self.scope, pl)
        ed.seek_requested.connect(self.seek_requested.emit)
        self.editor = ed
        self.scroll.setWidget(ed)
        self._push_result(ed)

    def _push_result(self, ed):
        res = self.last_res
        if res is None:
            return
        tr = res.tracks.get(self.scope) if self.scope != MASTER else None
        ed.set_result(res, tr)

    def _replaced(self, scope):
        if self.editor is not None and scope == self.scope:
            self.editor.refresh()

    def set_result(self, res):
        self.last_res = res
        if self.editor is not None:
            self._push_result(self.editor)

    def set_playhead(self, t):
        if self.editor is not None and self.isVisible():
            self.editor.set_playhead(t)

    # ---------------------------------------------------- доступные плагины
    def _fill_available(self):
        q = self.search.text().strip().lower()
        self.avail.clear()
        for cat in CATEGORY_ORDER:
            items = [d for d in PLUGINS.values() if d.category == cat and
                     (not q or q in d.title.lower() or q in d.tagline.lower() or q in d.help.lower()
                      or q in CATEGORY_TITLES[cat].lower())]
            if not items:
                continue
            h = QListWidgetItem(CATEGORY_TITLES[cat].upper())
            h.setFlags(Qt.ItemFlag.NoItemFlags)
            f = QFont(self.font())
            f.setPointSizeF(8.0)
            f.setBold(True)
            h.setFont(f)
            h.setForeground(QColor(theme.MUTED))
            self.avail.addItem(h)
            for d in items:
                it = QListWidgetItem(f"{d.icon}   {d.title}  —  {d.tagline}")
                it.setData(Qt.ItemDataRole.UserRole, d.type)
                it.setToolTip(d.help)
                self.avail.addItem(it)

    def _avail_selected(self, cur, prev):
        t = cur.data(Qt.ItemDataRole.UserRole) if cur else None
        if t:
            d = PLUGINS[t]
            self.avail_help.setText(f"<b>{d.title}</b> — {d.help}")

    def _avail_activated(self, item):
        t = item.data(Qt.ItemDataRole.UserRole)
        if t:
            self._add(t)

    def _add_selected(self):
        it = self.avail.currentItem()
        t = it.data(Qt.ItemDataRole.UserRole) if it else None
        if t:
            self._add(t)
        else:
            self.avail_help.setText("Сначала выберите плагин в списке выше.")

    def _add(self, ptype):
        pl = self.s.add_plugin(self.scope, ptype)
        self.sel_plugin = pl["id"]
        self.rebuild()

    # ------------------------------------------------------------- цепочки
    def _chains_menu(self):
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
            if self.plugins():
                ok = QMessageBox.question(self, "Заменить цепочку?",
                                          "Текущие плагины будут заменены выбранной цепочкой (это можно отменить Ctrl+Z). Продолжить?")
                if ok != QMessageBox.StandardButton.Yes:
                    return
            try:
                if kind == "factory":
                    self.s.apply_chain_preset(self.scope, name)
                else:
                    self.s.apply_chain(self.scope, load_user_chain(name))
            except Exception as e:
                QMessageBox.warning(self, "Vocal Studio", f"Не удалось применить цепочку: {e}")
        elif kind == "save":
            n, ok = QInputDialog.getText(self, "Сохранить цепочку", "Название цепочки:")
            if ok and n.strip():
                save_user_chain(n.strip(), copy.deepcopy(self.plugins()))
        elif kind == "del":
            delete_user_chain(name)
