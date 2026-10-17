"""Редакторы плагинов: у каждого типа свой интерфейс, привязанный к экземпляру плагина на дорожке."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QMenu,
                             QPushButton, QVBoxLayout, QWidget, QMessageBox)

from ..plugins import PLUGINS, spec_map
from ..session import (delete_user_plugin_preset, list_user_plugin_presets, save_user_plugin_preset)
from . import theme
from .async_util import run_async
from .pitch_view import PitchPanel
from .plots import EQView, GRTrace, TransferCurve
from .widgets import Knob, NoteKeys


class ParamBinder:
    """Создаёт контролы, привязанные к параметрам конкретного плагина."""

    def __init__(self, session, scope, pid, ptype):
        self.s, self.scope, self.pid = session, scope, pid
        self.specs = spec_map(ptype)
        self.setters: dict = {}

    def value(self, key):
        p = self.s.find_plugin(self.scope, self.pid)
        return None if p is None else p["params"].get(key)

    def refresh(self):
        for key, fn in self.setters.items():
            v = self.value(key)
            if v is not None:
                fn(v)

    def knob(self, key, size=54, accent=theme.ACCENT) -> Knob:
        spec = self.specs[key]
        kn = Knob(spec, self.value(key), size=size, accent=accent)
        kn.valueChanged.connect(lambda v, k=key: self.s.set_param(self.scope, self.pid, k, v))
        self.setters[key] = lambda v, kn=kn: kn.set_value(v)
        return kn

    def combo(self, key, label=True, width=None) -> QWidget:
        spec = self.specs[key]
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        if label:
            lb = QLabel(spec.label.upper())
            lb.setObjectName("title")
            lay.addWidget(lb)
        cb = QComboBox()
        cb.addItems(spec.choices)
        cb.setToolTip(spec.tip or spec.label)
        cb.setStatusTip(spec.tip or spec.label)
        if width:
            cb.setMinimumWidth(width)
        cb.setCurrentText(str(self.value(key)))
        cb.activated.connect(lambda i, k=key, cb=cb: self.s.set_param(self.scope, self.pid, k, cb.itemText(i)))

        def setter(v, cb=cb):
            cb.blockSignals(True)
            cb.setCurrentText(str(v))
            cb.blockSignals(False)

        self.setters[key] = setter
        lay.addWidget(cb)
        w.combo = cb
        return w

    def check(self, key, text=None) -> QCheckBox:
        spec = self.specs[key]
        cb = QCheckBox(text or spec.label)
        cb.setToolTip(spec.tip or spec.label)
        cb.setStatusTip(spec.tip or spec.label)
        cb.setChecked(bool(self.value(key)))
        cb.toggled.connect(lambda on, k=key: self.s.set_param(self.scope, self.pid, k, bool(on)))

        def setter(v, cb=cb):
            cb.blockSignals(True)
            cb.setChecked(bool(v))
            cb.blockSignals(False)

        self.setters[key] = setter
        return cb


def _row(widgets, spacing=2) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for x in widgets:
        lay.addWidget(x)
    lay.addStretch(1)
    return w


class PluginEditor(QFrame):
    """Рамка редактора: заголовок (название, описание, пресеты, вкл/выкл) + тело."""
    seek_requested = pyqtSignal(float)

    ACCENT = theme.ACCENT

    def __init__(self, session, scope, plugin, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.s = session
        self.scope = scope
        self.pid = plugin["id"]
        self.ptype = plugin["type"]
        self.d = PLUGINS[self.ptype]
        self.b = ParamBinder(session, scope, self.pid, self.ptype)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 12)
        outer.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(10)
        self.power = QPushButton()
        self.power.setObjectName("power")
        self.power.setCheckable(True)
        self.power.setChecked(bool(plugin["on"]))
        self.power.setToolTip("Включить / выключить этот плагин (сравнить звук «с» и «без»)")
        self.power.setStatusTip("Включить / выключить этот плагин, чтобы сравнить звук с ним и без него")
        self.power.toggled.connect(lambda on: self.s.set_plugin_on(self.scope, self.pid, bool(on)))
        head.addWidget(self.power)
        title = QLabel(self.d.title.upper())
        title.setObjectName("title")
        head.addWidget(title)
        tag = QLabel("— " + self.d.tagline)
        tag.setObjectName("muted")
        head.addWidget(tag)
        head.addStretch(1)
        self.btn_presets = QPushButton("Пресеты ▾")
        self.btn_presets.setToolTip("Готовые настройки этого плагина и ваши сохранённые")
        self.btn_presets.setStatusTip("Готовые настройки этого плагина и ваши собственные пресеты")
        self.btn_presets.clicked.connect(self._presets_menu)
        head.addWidget(self.btn_presets)
        outer.addLayout(head)
        help_ = QLabel(self.d.help)
        help_.setObjectName("muted")
        help_.setWordWrap(True)
        outer.addWidget(help_)
        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        outer.addLayout(self.body, 1)
        self.build()
        self.b.refresh()

    # --- переопределяются ---------------------------------------------
    def build(self):
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        keys = [s.key for s in self.d.specs if s.kind == "float"]
        for i, k in enumerate(keys):
            grid.addWidget(self.b.knob(k, 56, self.ACCENT), 0, i)
        for s in self.d.specs:
            if s.kind == "choice":
                grid.addWidget(self.b.combo(s.key), 1, 0, 1, 2)
            elif s.kind == "bool":
                grid.addWidget(self.b.check(s.key), 1, 2, 1, 2)
        w = QWidget()
        w.setLayout(grid)
        self.body.addWidget(w, 0, Qt.AlignmentFlag.AlignLeft)
        self.body.addStretch(1)

    def set_result(self, res, track_result):
        pass

    def set_playhead(self, t):
        pass

    def refresh(self):
        p = self.s.find_plugin(self.scope, self.pid)
        if p is None:
            return
        self.power.blockSignals(True)
        self.power.setChecked(bool(p["on"]))
        self.power.blockSignals(False)
        self.b.refresh()

    # --- пресеты ------------------------------------------------------
    def _presets_menu(self):
        m = QMenu(self)
        acts = {}
        for name, params in self.d.presets.items():
            acts[m.addAction(name)] = ("apply", params)
        users = list_user_plugin_presets(self.ptype)
        if users:
            m.addSeparator()
            for name, params in users.items():
                acts[m.addAction("★ " + name)] = ("apply", params)
        m.addSeparator()
        acts[m.addAction("Сохранить текущие настройки как пресет…")] = ("save", None)
        if users:
            sub = m.addMenu("Удалить свой пресет")
            for name in users:
                acts[sub.addAction(name)] = ("delete", name)
        acts[m.addAction("Сбросить к настройкам по умолчанию")] = ("reset", None)
        a = m.exec(self.btn_presets.mapToGlobal(self.btn_presets.rect().bottomLeft()))
        if a not in acts:
            return
        kind, data = acts[a]
        if kind == "apply":
            self.s.apply_plugin_preset(self.scope, self.pid, data)
        elif kind == "save":
            p = self.s.find_plugin(self.scope, self.pid)
            name, ok = QInputDialog.getText(self, "Сохранить пресет", "Название пресета:")
            if ok and name.strip() and p is not None:
                save_user_plugin_preset(self.ptype, name.strip(), p["params"])
        elif kind == "delete":
            delete_user_plugin_preset(self.ptype, data)
        elif kind == "reset":
            self.s.apply_plugin_preset(self.scope, self.pid, {s.key: s.default for s in self.d.specs})


# --------------------------------------------------- ключ / лад / ноты ---

class KeyControls(QWidget):
    """Выбор тональности: Key, Scale, автоопределение, клавиатура нот (используют Auto-Tune и Harmony)."""

    def __init__(self, editor, with_input_type=False, parent=None):
        super().__init__(parent)
        self.ed = editor
        s, b = editor.s, editor.b
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        top = QHBoxLayout()
        self.key_w = b.combo("key", width=64)
        self.scale_w = b.combo("scale", width=140)
        self.key_w.combo.activated.connect(lambda i: s.set_scale(editor.scope, editor.pid, self.key_w.combo.itemText(i), b.value("scale")))
        self.scale_w.combo.activated.connect(lambda i: s.set_scale(editor.scope, editor.pid, b.value("key"), self.scale_w.combo.itemText(i)))
        top.addWidget(self.key_w)
        top.addWidget(self.scale_w)
        if with_input_type:
            top.addWidget(b.combo("input_type", width=110))
        top.addStretch(1)
        lay.addLayout(top)
        self.btn_detect = QPushButton("🎯  Определить тональность автоматически")
        self.btn_detect.setToolTip("Программа послушает ваш вокал (и подложку) и сама выберет ключ и лад")
        self.btn_detect.setStatusTip("Автоопределение тональности: анализ вашего пения и подложки")
        self.btn_detect.clicked.connect(self._detect)
        lay.addWidget(self.btn_detect)
        self.lbl_detect = QLabel("")
        self.lbl_detect.setObjectName("muted")
        self.lbl_detect.setWordWrap(True)
        lay.addWidget(self.lbl_detect)
        self.keys = NoteKeys()
        self.keys.changed.connect(lambda m: s.set_notes(editor.scope, editor.pid, m))
        lay.addWidget(self.keys)
        bb = QHBoxLayout()
        for text, fn in (("Все", lambda: s.set_notes(editor.scope, editor.pid, [True] * 12)),
                         ("Ничего", lambda: s.set_notes(editor.scope, editor.pid, [False] * 12)),
                         ("Инверсия", lambda: s.set_notes(editor.scope, editor.pid, [not x for x in b.value("notes")]))):
            btn = QPushButton(text)
            btn.setFixedHeight(24)
            btn.clicked.connect(fn)
            bb.addWidget(btn)
        bb.addStretch(1)
        lay.addLayout(bb)
        self.setFixedWidth(400)

    def refresh(self):
        notes = self.ed.b.value("notes")
        if notes is not None:
            self.keys.set_mask(notes)

    def _detect(self):
        ed = self.ed
        t = ed.s.track(ed.scope) if ed.scope != "master" else None
        if t is None or t.audio is None:
            self.lbl_detect.setText("На дорожке нет записи — сначала запишите вокал или откройте файл.")
            return
        self.btn_detect.setEnabled(False)
        self.lbl_detect.setText("Слушаю… определяю тональность")

        def done(res, err):
            self.btn_detect.setEnabled(True)
            if err is not None or res is None:
                self.lbl_detect.setText("Не удалось определить тональность — слишком мало пения. Выберите её вручную.")
                return
            ed.s.set_scale(ed.scope, ed.pid, res["key"], res["scale"])
            extra = ""
            if "ref_a" in ed.b.specs and res.get("ref_a") and abs(res["ref_a"] - 440.0) > 3:
                ed.s.set_param(ed.scope, ed.pid, "ref_a", res["ref_a"])
                extra = f" Строй подложки: {res['ref_a']} Гц — настроил."
            conf = "" if res["confident"] else " (уверенность невысокая — проверьте на слух)"
            self.lbl_detect.setText(f"Найдено: {res['name']} — по {res['source']}, уверенность {res['label']}{conf}.{extra}")

        run_async(lambda: ed.s.detect_key_for(ed.scope), done)


# ------------------------------------------------------------------ Auto-Tune --

class TuneEditor(PluginEditor):
    def build(self):
        s = self.s
        self.panel = PitchPanel()
        self.panel.setMinimumHeight(150)
        self.panel.view.seek_requested.connect(self.seek_requested.emit)
        self.panel.view.overrides_changed.connect(
            lambda ov: s.set_param(self.scope, self.pid, "overrides", [list(o) for o in ov]))
        self.body.addWidget(self.panel, 1)

        row = QHBoxLayout()
        row.setSpacing(18)
        self.kc = KeyControls(self, with_input_type=True)
        row.addWidget(self.kc)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.addWidget(self.b.knob("retune", 84), 0, 0, 2, 2, Qt.AlignmentFlag.AlignCenter)
        order = ["humanize", "flex", "vibrato", "amount", "transpose", "detune", "formant", "tracking", "ref_a"]
        pos = [(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]
        for key, (r, c) in zip(order, pos):
            grid.addWidget(self.b.knob(key, 48), r, c, Qt.AlignmentFlag.AlignCenter)
        gw = QWidget()
        gw.setLayout(grid)
        row.addWidget(gw, 1, Qt.AlignmentFlag.AlignTop)
        rw = QWidget()
        rw.setLayout(row)
        self.body.addWidget(rw)
        self.refresh_tune()

    def refresh_tune(self):
        p = self.s.find_plugin(self.scope, self.pid)
        if p is None:
            return
        self.kc.refresh()
        pr = p["params"]
        self.panel.view.set_mask(pr["notes"], pr["transpose"] + pr["detune"] / 100.0)

    def refresh(self):
        super().refresh()
        self.refresh_tune()

    def set_result(self, res, track_result):
        pv = self.panel.view
        t = self.s.track(self.scope)
        if t is not None and t.audio is not None:
            import numpy as np
            mono = t.mono()
            if t.start:
                mono = np.concatenate([np.zeros(t.start, np.float32), mono])
            pv.set_wave(mono, self.s.sr)
            pv.set_duration(max(t.end, 1) / self.s.sr)
        else:
            pv.set_wave(None, self.s.sr)
        got = track_result.tune.get(self.pid) if track_result is not None else None
        if got is None:
            pv.set_result(None, None, self.s.sr)
            return
        tr, corr = got
        p = self.s.find_plugin(self.scope, self.pid)
        pv.set_overrides([tuple(o) for o in (p["params"].get("overrides") or [])] if p else [])
        first = pv.in_midi is None
        pv.set_result(tr, corr, self.s.sr)
        if first:
            pv.fit_vertical()
            pv.fit_horizontal()

    def set_playhead(self, t):
        self.panel.view.set_playhead(t)


# ------------------------------------------------------------------------ EQ ---

class EQEditor(PluginEditor):
    def build(self):
        self.view = EQView()
        self.view.setMinimumHeight(170)
        self.view.param_edited.connect(lambda k, v: self.s.set_param(self.scope, self.pid, k, v))
        self.body.addWidget(self.view, 1)
        bands = QHBoxLayout()
        bands.setSpacing(10)

        def group(title, items):
            g = QFrame()
            g.setStyleSheet("QFrame { background: #20242e; border-radius: 6px; }")
            gl = QVBoxLayout(g)
            gl.setContentsMargins(8, 4, 8, 4)
            gl.setSpacing(0)
            t = QLabel(title)
            t.setObjectName("title")
            gl.addWidget(t)
            gl.addWidget(_row(items))
            return g

        b = self.b
        hp_box = QWidget()
        hl = QVBoxLayout(hp_box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(b.check("hpf_on", "On"))
        hl.addWidget(b.combo("hpf_slope", label=False))
        bands.addWidget(group("HIGH-PASS", [hp_box, b.knob("hpf_f", 42, "#8fb3ff")]))
        bands.addWidget(group("LOW SHELF", [b.knob("ls_f", 42, "#6be38a"), b.knob("ls_g", 42, "#6be38a")]))
        for n, col in (("b1", "#ffd36b"), ("b2", "#ff9f43"), ("b3", "#ff6b81")):
            bands.addWidget(group("BELL " + n[1], [b.knob(n + "_f", 42, col), b.knob(n + "_g", 42, col), b.knob(n + "_q", 42, col)]))
        bands.addWidget(group("HIGH SHELF", [b.knob("hs_f", 42, "#c58bff"), b.knob("hs_g", 42, "#c58bff")]))
        lp_box = QWidget()
        ll = QVBoxLayout(lp_box)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(b.check("lp_on", "On"))
        bands.addWidget(group("LOW-PASS", [lp_box, b.knob("lp_f", 42, "#8fb3ff")]))
        bands.addStretch(1)
        w = QWidget()
        w.setLayout(bands)
        self.body.addWidget(w)
        self._update_view()

    def _update_view(self):
        p = self.s.find_plugin(self.scope, self.pid)
        if p is not None:
            self.view.set_params(p["params"], self.s.sr)

    def refresh(self):
        super().refresh()
        self._update_view()

    def set_result(self, res, track_result):
        t = self.s.track(self.scope)
        if t is None or t.audio is None:
            self.view.set_spectrum(None, None)
            return
        key = t.audio_id
        if getattr(self, "_spec_key", None) != key:
            self._spec_key = key
            try:
                import numpy as np
                from scipy.signal import welch
                x = t.mono()[: self.s.sr * 90]
                f, pxx = welch(x, fs=self.s.sr, nperseg=4096)
                db = 10 * np.log10(pxx + 1e-14)
                lf = np.geomspace(20, 20000, 160)
                out = np.interp(lf, f, db)
                out = np.convolve(out, np.ones(5) / 5, mode="same")
                out = (out - out.max() + 70.0) / 70.0
                self.view.set_spectrum(lf, out)
            except Exception:
                pass


# ----------------------------------------------------------------- Dynamics ---

class CompEditor(PluginEditor):
    ACCENT = theme.ACCENT2

    def build(self):
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        for i, k in enumerate(["threshold", "ratio", "attack", "release", "knee", "makeup"]):
            grid.addWidget(self.b.knob(k, 56, self.ACCENT), 0, i)
        extra = QVBoxLayout()
        extra.addWidget(self.b.combo("detect"))
        extra.addWidget(self.b.check("auto_makeup"))
        ew = QWidget()
        ew.setLayout(extra)
        grid.addWidget(ew, 0, 6)
        grid.addWidget(self.b.knob("mix", 50, self.ACCENT), 0, 7)
        self.curve = TransferCurve()
        grid.addWidget(self.curve, 0, 8, 1, 1, Qt.AlignmentFlag.AlignTop)
        w = QWidget()
        w.setLayout(grid)
        self.body.addWidget(w, 0, Qt.AlignmentFlag.AlignLeft)
        self.gr = GRTrace("GAIN REDUCTION — насколько компрессор «прижал» звук", 18.0)
        self.gr.setMinimumHeight(70)
        self.body.addWidget(self.gr, 1)
        self.refresh_curve()

    def refresh_curve(self):
        p = self.s.find_plugin(self.scope, self.pid)
        if p:
            self.curve.set_params(p["params"])

    def refresh(self):
        super().refresh()
        self.refresh_curve()

    def set_result(self, res, track_result):
        gr = None
        if track_result is not None:
            gr = track_result.gr.get(self.pid)
        if gr is None and res is not None:
            gr = res.master_gr.get(self.pid)
        self.gr.set_data(gr, self.s.sr)

    def set_playhead(self, t):
        self.gr.set_playhead(t)


class DeessEditor(PluginEditor):
    def build(self):
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        for i, k in enumerate(["freq", "threshold", "range"]):
            grid.addWidget(self.b.knob(k, 56), 0, i)
        grid.addWidget(self.b.check("wide"), 0, 3)
        w = QWidget()
        w.setLayout(grid)
        self.body.addWidget(w, 0, Qt.AlignmentFlag.AlignLeft)
        self.gr = GRTrace("SIBILANCE REDUCTION — где подавлены «с»", 14.0)
        from PyQt6.QtGui import QColor
        self.gr.color = QColor(theme.REC)
        self.gr.setMinimumHeight(70)
        self.body.addWidget(self.gr, 1)

    def set_result(self, res, track_result):
        gr = track_result.gr.get(self.pid) if track_result is not None else None
        self.gr.set_data(gr, self.s.sr)

    def set_playhead(self, t):
        self.gr.set_playhead(t)


class DelayEditor(PluginEditor):
    ACCENT = "#6aa8ff"

    def build(self):
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.addWidget(self.b.combo("mode", width=90), 0, 0)
        grid.addWidget(self.b.combo("division", width=110), 0, 1, 1, 2)
        for i, k in enumerate(["bpm", "time_ms", "feedback", "pingpong", "lowcut", "highcut", "mix"]):
            grid.addWidget(self.b.knob(k, 56, self.ACCENT), 1, i)
        w = QWidget()
        w.setLayout(grid)
        self.body.addWidget(w, 0, Qt.AlignmentFlag.AlignLeft)
        self.body.addStretch(1)


class ReverbEditor(PluginEditor):
    ACCENT = "#c58bff"

    def build(self):
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.addWidget(self.b.combo("type", width=120), 0, 0, 1, 2)
        for i, k in enumerate(["decay", "predelay", "size", "damping", "width", "lowcut", "highcut", "mix"]):
            grid.addWidget(self.b.knob(k, 56, self.ACCENT), 1, i)
        w = QWidget()
        w.setLayout(grid)
        self.body.addWidget(w, 0, Qt.AlignmentFlag.AlignLeft)
        self.body.addStretch(1)


class RiderEditor(PluginEditor):
    def build(self):
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        for i, k in enumerate(["target", "range", "speed", "gate"]):
            grid.addWidget(self.b.knob(k, 56, self.ACCENT), 0, i)
        w = QWidget()
        w.setLayout(grid)
        self.body.addWidget(w, 0, Qt.AlignmentFlag.AlignLeft)
        self.gr = GRTrace("RIDER — как громкость подстраивалась по ходу песни (вверх/вниз)", 12.0)
        self.gr.setMinimumHeight(70)
        self.body.addWidget(self.gr, 1)

    def set_result(self, res, track_result):
        gr = track_result.gr.get(self.pid) if track_result is not None else None
        self.gr.set_data(gr, self.s.sr)

    def set_playhead(self, t):
        self.gr.set_playhead(t)


class UnmaskEditor(PluginEditor):
    def build(self):
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        for i, k in enumerate(["freq", "q", "reduction", "threshold", "release"]):
            grid.addWidget(self.b.knob(k, 56, self.ACCENT), 0, i)
        w = QWidget()
        w.setLayout(grid)
        self.body.addWidget(w, 0, Qt.AlignmentFlag.AlignLeft)
        tip = QLabel("Нужен вокал на другой дорожке: плагин реагирует на его громкость.")
        tip.setObjectName("muted")
        self.body.addWidget(tip)
        self.gr = GRTrace("UNMASK — где музыка приглушалась под голос", 8.0)
        self.gr.setMinimumHeight(70)
        self.body.addWidget(self.gr, 1)

    def set_result(self, res, track_result):
        gr = track_result.gr.get(self.pid) if track_result is not None else None
        self.gr.set_data(gr, self.s.sr)

    def set_playhead(self, t):
        self.gr.set_playhead(t)


class HarmonyEditor(PluginEditor):
    ACCENT = "#c58bff"

    def build(self):
        row = QHBoxLayout()
        row.setSpacing(18)
        self.kc = KeyControls(self)
        row.addWidget(self.kc, 0, Qt.AlignmentFlag.AlignTop)
        right = QVBoxLayout()
        right.setSpacing(6)
        cap = QLabel("ГОЛОСА ГАРМОНИИ")
        cap.setObjectName("title")
        right.addWidget(cap)
        for k in range(1, 5):
            r = QHBoxLayout()
            r.setSpacing(10)
            cb = self.b.combo(f"v{k}_int", label=False, width=130)
            lab = QLabel(f"Голос {k}")
            lab.setFixedWidth(54)
            r.addWidget(lab)
            r.addWidget(cb)
            r.addWidget(self.b.knob(f"v{k}_lvl", 40, self.ACCENT))
            r.addWidget(self.b.knob(f"v{k}_pan", 40, self.ACCENT))
            r.addStretch(1)
            w = QWidget()
            w.setLayout(r)
            right.addWidget(w)
        g = QGridLayout()
        for i, key in enumerate(["mix", "formant", "timing", "drift", "ref_a", "tracking"]):
            g.addWidget(self.b.knob(key, 46, self.ACCENT), 0, i)
        gw = QWidget()
        gw.setLayout(g)
        right.addWidget(gw)
        right.addStretch(1)
        rw = QWidget()
        rw.setLayout(right)
        row.addWidget(rw, 1)
        w = QWidget()
        w.setLayout(row)
        self.body.addWidget(w)
        self.kc.refresh()

    def refresh(self):
        super().refresh()
        self.kc.refresh()


EDITORS = {"tune": TuneEditor, "eq": EQEditor, "comp": CompEditor, "deess": DeessEditor,
           "delay": DelayEditor, "reverb": ReverbEditor, "rider": RiderEditor, "unmask": UnmaskEditor,
           "harmony": HarmonyEditor}


def make_editor(session, scope, plugin) -> PluginEditor:
    return EDITORS.get(plugin["type"], PluginEditor)(session, scope, plugin)
