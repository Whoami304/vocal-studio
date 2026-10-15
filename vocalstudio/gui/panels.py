"""Панели модулей (рэк): Auto-Tune, EQ, динамика, пространство, выход."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

from ..params import MODULES, Spec
from ..dsp.scales import NOTE_NAMES
from . import theme
from .plots import EQView, GRTrace, TransferCurve
from .widgets import Knob, LevelMeter, NoteKeys

SPEC = {(m, s.key): s for m, d in MODULES.items() for s in d["specs"]}


class Binder:
    """Создаёт контролы, привязанные к Session, и умеет обновлять их при смене пресета."""

    def __init__(self, session):
        self.s = session
        self.setters = {}

    def refresh(self):
        for (m, k), fn in self.setters.items():
            v = self.s.params[m].get(k)
            if v is not None:
                fn(v)

    def knob(self, module, key, size=54, accent=theme.ACCENT) -> Knob:
        spec = SPEC[(module, key)]
        kn = Knob(spec, self.s.params[module][key], size=size, accent=accent)
        kn.valueChanged.connect(lambda v, m=module, k=key: self.s.set_param(m, k, v))
        self.setters[(module, key)] = lambda v, kn=kn: kn.set_value(v)
        return kn

    def combo(self, module, key, label=True, width=None) -> QWidget:
        spec = SPEC[(module, key)]
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
        if width:
            cb.setMinimumWidth(width)
        cb.setCurrentText(str(self.s.params[module][key]))
        cb.activated.connect(lambda i, m=module, k=key, cb=cb: self.s.set_param(m, k, cb.itemText(i)))

        def setter(v, cb=cb):
            cb.blockSignals(True)
            cb.setCurrentText(str(v))
            cb.blockSignals(False)

        self.setters[(module, key)] = setter
        lay.addWidget(cb)
        w.combo = cb
        return w

    def check(self, module, key, text=None) -> QCheckBox:
        spec = SPEC[(module, key)]
        cb = QCheckBox(text or spec.label)
        cb.setToolTip(spec.tip or spec.label)
        cb.setChecked(bool(self.s.params[module][key]))
        cb.toggled.connect(lambda on, m=module, k=key: self.s.set_param(m, k, bool(on)))

        def setter(v, cb=cb):
            cb.blockSignals(True)
            cb.setChecked(bool(v))
            cb.blockSignals(False)

        self.setters[(module, key)] = setter
        return cb

    def frame(self, module, title=None, accent=theme.ACCENT) -> tuple[QFrame, QGridLayout]:
        """Рамка модуля с кнопкой-питанием в заголовке."""
        fr = QFrame()
        fr.setObjectName("panel")
        outer = QVBoxLayout(fr)
        outer.setContentsMargins(12, 8, 12, 10)
        outer.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(8)
        if MODULES[module]["has_bypass"]:
            pw = QPushButton()
            pw.setObjectName("power")
            pw.setCheckable(True)
            pw.setChecked(bool(self.s.params[module]["on"]))
            pw.setToolTip("Включить/выключить модуль")
            pw.toggled.connect(lambda on, m=module: self.s.set_param(m, "on", bool(on)))

            def setter(v, pw=pw):
                pw.blockSignals(True)
                pw.setChecked(bool(v))
                pw.blockSignals(False)

            self.setters[(module, "on")] = setter
            head.addWidget(pw)
        t = QLabel(title or MODULES[module]["title"])
        t.setObjectName("title")
        head.addWidget(t)
        head.addStretch(1)
        outer.addLayout(head)
        body = QGridLayout()
        body.setHorizontalSpacing(4)
        body.setVerticalSpacing(4)
        outer.addLayout(body)
        outer.addStretch(1)
        fr.head = head
        return fr, body


def _row(widgets, spacing=2) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for x in widgets:
        lay.addWidget(x)
    lay.addStretch(1)
    return w


# ------------------------------------------------------------- Auto-Tune ---

class TunePage(QWidget):
    def __init__(self, binder: Binder, parent=None):
        super().__init__(parent)
        self.b = binder
        s = binder.s
        fr, body = binder.frame("tune", "AUTO-TUNE  ·  PITCH CORRECTION")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(fr)

        # левая колонка: тональность
        left = QVBoxLayout()
        left.setSpacing(8)
        top = QHBoxLayout()
        self.key_w = binder.combo("tune", "key", width=70)
        self.scale_w = binder.combo("tune", "scale", width=150)
        self.in_w = binder.combo("tune", "input_type", width=120)
        # Key/Scale меняют маску нот, поэтому переопределяем обработчики
        self.key_w.combo.activated.connect(lambda i: s.set_scale(self.key_w.combo.itemText(i), s.params["tune"]["scale"]))
        self.scale_w.combo.activated.connect(lambda i: s.set_scale(s.params["tune"]["key"], self.scale_w.combo.itemText(i)))
        for w in (self.key_w, self.scale_w, self.in_w):
            top.addWidget(w)
        top.addStretch(1)
        left.addLayout(top)
        cap = QLabel("TARGET NOTES — клик по клавише включает/выключает ноту")
        cap.setObjectName("muted")
        left.addWidget(cap)
        self.keys = NoteKeys()
        self.keys.set_mask(s.params["tune"]["notes"])
        self.keys.changed.connect(s.set_notes)
        left.addWidget(self.keys)
        row = QHBoxLayout()
        for text, fn in (("All", lambda: s.set_notes([True] * 12)),
                         ("None", lambda: s.set_notes([False] * 12)),
                         ("Invert", lambda: s.set_notes([not x for x in s.params["tune"]["notes"]]))):
            bt = QPushButton(text)
            bt.setFixedHeight(24)
            bt.clicked.connect(fn)
            row.addWidget(bt)
        row.addStretch(1)
        left.addLayout(row)
        hint = QLabel("Retune Speed 0 — жёсткий «робот»-эффект.\n15–40 ms — современный поп.\n80+ ms — прозрачная лёгкая подстройка.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        left.addWidget(hint)
        left.addStretch(1)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMinimumWidth(380)
        lw.setMaximumWidth(460)
        body.addWidget(lw, 0, 0, 2, 1)

        # правая часть: ручки
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)
        main = binder.knob("tune", "retune", size=92)
        grid.addWidget(main, 0, 0, 2, 2, Qt.AlignmentFlag.AlignCenter)
        order = ["humanize", "flex", "vibrato", "amount", "transpose", "detune", "formant", "tracking", "ref_a"]
        pos = [(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]
        for key, (r, c) in zip(order, pos):
            grid.addWidget(binder.knob("tune", key, size=54), r, c, Qt.AlignmentFlag.AlignCenter)
        gw = QWidget()
        gw.setLayout(grid)
        body.addWidget(gw, 0, 1, 2, 1, Qt.AlignmentFlag.AlignTop)
        body.setColumnStretch(1, 1)

    def refresh(self):
        t = self.b.s.params["tune"]
        self.keys.set_mask(t["notes"])


# -------------------------------------------------------------------- EQ ---

class EQPage(QWidget):
    def __init__(self, binder: Binder, parent=None):
        super().__init__(parent)
        self.b = binder
        s = binder.s
        fr, body = binder.frame("eq", "PARAMETRIC EQUALIZER")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(fr)

        self.view = EQView()
        self.view.param_edited.connect(lambda k, v: s.set_param("eq", k, v))
        body.addWidget(self.view, 0, 0, 1, 1)

        bands = QHBoxLayout()
        bands.setSpacing(10)
        pos = binder.combo("eq", "pos", width=110)
        pos.setFixedWidth(120)

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

        hp_items = [binder.check("eq", "hpf_on", "On"), ]
        hp_box = QWidget()
        hl = QVBoxLayout(hp_box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(hp_items[0])
        hl.addWidget(binder.combo("eq", "hpf_slope", label=False))
        bands.addWidget(group("HIGH-PASS", [hp_box, binder.knob("eq", "hpf_f", 42, "#8fb3ff")]))
        bands.addWidget(group("LOW SHELF", [binder.knob("eq", "ls_f", 42, "#6be38a"), binder.knob("eq", "ls_g", 42, "#6be38a")]))
        for n, col in (("b1", "#ffd36b"), ("b2", "#ff9f43"), ("b3", "#ff6b81")):
            bands.addWidget(group("BELL " + n[1], [binder.knob("eq", n + "_f", 42, col), binder.knob("eq", n + "_g", 42, col),
                                                    binder.knob("eq", n + "_q", 42, col)]))
        bands.addWidget(group("HIGH SHELF", [binder.knob("eq", "hs_f", 42, "#c58bff"), binder.knob("eq", "hs_g", 42, "#c58bff")]))
        lp_box = QWidget()
        ll = QVBoxLayout(lp_box)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(binder.check("eq", "lp_on", "On"))
        bands.addWidget(group("LOW-PASS", [lp_box, binder.knob("eq", "lp_f", 42, "#8fb3ff")]))
        bands.addWidget(pos, 0, Qt.AlignmentFlag.AlignTop)
        bands.addStretch(1)
        body.addLayout(bands, 1, 0)
        body.setRowStretch(0, 1)
        s.param_set.connect(self._on_param)
        s.params_replaced.connect(self.refresh)
        self.refresh()

    def _on_param(self, module, key):
        if module == "eq":
            self.view.set_params(self.b.s.params["eq"], self.b.s.sr)

    def refresh(self):
        self.view.set_params(self.b.s.params["eq"], self.b.s.sr)


# ------------------------------------------------------------- Dynamics ---

class DynamicsPage(QWidget):
    def __init__(self, binder: Binder, parent=None):
        super().__init__(parent)
        self.b = binder
        s = binder.s
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        gf, gb = binder.frame("gate", "NOISE GATE")
        for i, k in enumerate(["threshold", "range", "attack", "release"]):
            gb.addWidget(binder.knob("gate", k, 48), i // 2, i % 2)
        lay.addWidget(gf)

        cf, cb = binder.frame("comp", "COMPRESSOR", theme.ACCENT2)
        keys = ["threshold", "ratio", "attack", "release", "knee", "makeup"]
        for i, k in enumerate(keys):
            cb.addWidget(binder.knob("comp", k, 52, theme.ACCENT2), i // 3, i % 3)
        extra = QVBoxLayout()
        extra.addWidget(binder.combo("comp", "detect"))
        extra.addWidget(binder.check("comp", "auto_makeup"))
        extra.addWidget(binder.knob("comp", "mix", 44, theme.ACCENT2))
        ew = QWidget()
        ew.setLayout(extra)
        cb.addWidget(ew, 0, 3, 2, 1)
        self.curve = TransferCurve()
        cb.addWidget(self.curve, 0, 4, 2, 1, Qt.AlignmentFlag.AlignCenter)
        self.gr_comp = GRTrace("GAIN REDUCTION", 18.0)
        cb.addWidget(self.gr_comp, 2, 0, 1, 5)
        lay.addWidget(cf, 3)

        df, db = binder.frame("deess", "DE-ESSER")
        for i, k in enumerate(["freq", "threshold", "range"]):
            db.addWidget(binder.knob("deess", k, 50), 0, i)
        db.addWidget(binder.check("deess", "wide"), 1, 0, 1, 3)
        self.gr_de = GRTrace("SIBILANCE REDUCTION", 14.0)
        self.gr_de.color = __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(theme.REC)
        db.addWidget(self.gr_de, 2, 0, 1, 3)
        lay.addWidget(df, 2)

        sf, sb = binder.frame("sat", "SATURATION")
        for i, k in enumerate(["drive", "warmth", "mix"]):
            sb.addWidget(binder.knob("sat", k, 48), 0, i)
        lay.addWidget(sf, 2)

        s.param_set.connect(self._on_param)
        s.params_replaced.connect(self.refresh)
        self.refresh()

    def _on_param(self, module, key):
        if module == "comp":
            self.curve.set_params(self.b.s.params["comp"])

    def refresh(self):
        self.curve.set_params(self.b.s.params["comp"])

    def set_result(self, res, sr):
        self.gr_comp.set_data(res.comp_gr, sr) if res is not None else self.gr_comp.set_data(None, sr)
        self.gr_de.set_data(res.deess_gr, sr) if res is not None else self.gr_de.set_data(None, sr)

    def set_playhead(self, t):
        self.gr_comp.set_playhead(t)
        self.gr_de.set_playhead(t)


# ---------------------------------------------------------------- Space ---

class SpacePage(QWidget):
    def __init__(self, binder: Binder, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        df, db = binder.frame("delay", "DELAY", "#6aa8ff")
        db.addWidget(binder.combo("delay", "mode", width=90), 0, 0)
        db.addWidget(binder.combo("delay", "division", width=100), 0, 1, 1, 2)
        keys = ["bpm", "time_ms", "feedback", "pingpong", "lowcut", "highcut", "mix"]
        for i, k in enumerate(keys):
            db.addWidget(binder.knob("delay", k, 50, "#6aa8ff"), 1 + i // 4, i % 4)
        lay.addWidget(df, 1)

        rf, rb = binder.frame("reverb", "REVERB", "#c58bff")
        rb.addWidget(binder.combo("reverb", "type", width=110), 0, 0, 1, 2)
        keys = ["decay", "predelay", "size", "damping", "width", "lowcut", "highcut", "mix"]
        for i, k in enumerate(keys):
            rb.addWidget(binder.knob("reverb", k, 50, "#c58bff"), 1 + i // 4, i % 4)
        lay.addWidget(rf, 1)


# --------------------------------------------------------------- Output ---

class OutputPage(QWidget):
    def __init__(self, binder: Binder, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        inf, ib = binder.frame("input", "INPUT")
        ib.addWidget(binder.knob("input", "gain", 56), 0, 0)
        lay.addWidget(inf)
        of, ob = binder.frame("master", "OUTPUT / MASTER")
        ob.addWidget(binder.knob("master", "vocal_db", 56), 0, 0)
        ob.addWidget(binder.knob("master", "backing_db", 56), 0, 1)
        ob.addWidget(binder.knob("master", "ceiling", 56), 0, 2)
        ob.addWidget(binder.check("master", "limiter_on", "Limiter"), 1, 0, 1, 3)
        lay.addWidget(of)
        lay.addStretch(1)
