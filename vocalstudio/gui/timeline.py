"""Дорожки и таймлайн: заголовки дорожек (громкость, панорама, arm/mute/solo), дорожки с волной, линейка."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QBrush, QLinearGradient
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu, QPushButton, QScrollArea,
                             QScrollBar, QSizePolicy, QSlider, QVBoxLayout, QWidget, QColorDialog)

from . import theme
from .pitch_view import WavePyramid
from .widgets import LevelMeter

HEADER_W = 292
ROW_H = 86
MASTER_H = 66
RULER_H = 24


class TimelineState(QObject):
    changed = pyqtSignal()
    seek_requested = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.t0 = 0.0
        self.pps = 70.0
        self.playhead = 0.0
        self.duration = 20.0
        self.view_w = 800
        self.follow = True
        self.sel_tid = None
        self.sel_t0 = 0.0
        self.sel_t1 = 0.0

    # --- выделение (участок записи на дорожке) ---
    @property
    def has_selection(self) -> bool:
        return self.sel_tid is not None and self.sel_t1 - self.sel_t0 > 0.005

    def set_selection(self, tid, t0, t1):
        t0, t1 = sorted((float(t0), float(t1)))
        self.sel_tid, self.sel_t0, self.sel_t1 = tid, max(0.0, t0), max(0.0, t1)
        self.changed.emit()

    def clear_selection(self):
        if self.sel_tid is not None:
            self.sel_tid = None
            self.changed.emit()

    def x(self, t):
        return (t - self.t0) * self.pps

    def t_at(self, x):
        return self.t0 + x / self.pps

    @property
    def view_dur(self):
        return max(0.1, self.view_w / self.pps)

    def set_duration(self, d):
        self.duration = max(8.0, d + 2.0)
        self.changed.emit()

    def set_playhead(self, t):
        self.playhead = t
        if self.follow and self.view_w > 50:
            x = self.x(t)
            if x > self.view_w - 40 or x < 0:
                self.t0 = max(0.0, t - 0.15 * self.view_dur)
        self.changed.emit()

    def zoom(self, factor, anchor_x):
        tc = self.t_at(anchor_x)
        self.pps = min(3000.0, max(4.0, self.pps * factor))
        self.t0 = max(0.0, tc - anchor_x / self.pps)
        self.changed.emit()

    def pan(self, dt):
        self.t0 = max(0.0, self.t0 + dt)
        self.changed.emit()

    def fit(self):
        self.pps = max(4.0, (self.view_w - 20) / max(1.0, self.duration))
        self.t0 = 0.0
        self.changed.emit()


def wave_columns(pyr: WavePyramid, t0_samp: float, spp: float, W: int):
    if pyr.n == 0:
        return None
    B = pyr.BLOCK
    cols = np.arange(W)
    s = t0_samp + cols * spp
    e = s + spp
    nb = len(pyr.mn)
    b0 = np.floor(s / B).astype(np.int64)
    b1 = np.maximum(np.ceil(e / B).astype(np.int64), b0 + 1)
    ok = (b1 > 0) & (b0 < nb)
    if not ok.any():
        return None
    mn = np.zeros(W, np.float32)
    mx = np.zeros(W, np.float32)
    b0c = np.clip(b0, 0, nb - 1)
    b1c = np.clip(b1, 1, nb)
    if spp <= B * 2:
        mn[ok] = pyr.mn[b0c[ok]]
        mx[ok] = pyr.mx[b0c[ok]]
    else:
        wb = int(np.max(b1c[ok] - b0c[ok]))
        idx = b0c[:, None] + np.arange(wb)[None, :]
        valid = idx < b1c[:, None]
        idx = np.clip(idx, 0, nb - 1)
        mx[ok] = np.where(valid, pyr.mx[idx], -np.inf).max(axis=1)[ok]
        mn[ok] = np.where(valid, pyr.mn[idx], np.inf).min(axis=1)[ok]
    return mn, mx


class Lane(QWidget):
    """Дорожка со звуковой волной: выделение мышью, перемещение записи (Alt+перетаскивание), контекстное меню."""
    clicked = pyqtSignal(str)
    files_dropped = pyqtSignal(str, list)
    context_requested = pyqtSignal(str, object, float)     # tid, глобальная точка, время
    clip_moved = pyqtSignal(str, float)                    # tid, новое начало (сек)
    select_all_requested = pyqtSignal(str)

    EDGE = 6

    def __init__(self, tid: str, state: TimelineState, color: str, master=False, parent=None):
        super().__init__(parent)
        self.tid = tid
        self.state = state
        self.color = QColor(color)
        self.master = master
        self.setFixedHeight(MASTER_H if master else ROW_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.pyr = WavePyramid(None)
        self.audio_id = None
        self.sr = 44100
        self.start = 0.0            # начало записи на таймлайне, сек
        self.selected = False
        self.armed = False
        self.muted = False
        self.empty_hint = ""
        self.live = None            # (массив, t0 сек) во время записи
        self._pix = None
        self._pkey = None
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self._mode = None
        self._press_x = 0.0
        self._press_t = 0.0
        self._dragged = False
        self._move_t0 = 0.0
        self.preview = 0.0

    def set_color(self, c):
        self.color = QColor(c)
        self._pkey = None
        self.update()

    def set_audio(self, mono, audio_id, sr, start_sec=0.0):
        if audio_id != self.audio_id or audio_id is None or abs(start_sec - self.start) > 1e-9:
            self.audio_id = audio_id
            if mono is not None or self.pyr.n:
                self.pyr = WavePyramid(mono)
            self.sr = sr
            self.start = start_sec
            self._pkey = None
            self.update()

    def invalidate(self):
        self._pkey = None
        self.update()

    def _clip_range(self):
        if self.pyr.n == 0:
            return None
        return self.start, self.start + self.pyr.n / self.sr

    # --- мышь ----------------------------------------------------------------
    def _near_edge(self, x):
        st = self.state
        if st.sel_tid != self.tid or not st.has_selection:
            return None
        x0, x1 = st.x(st.sel_t0), st.x(st.sel_t1)
        if abs(x - x0) <= self.EDGE:
            return "left"
        if abs(x - x1) <= self.EDGE:
            return "right"
        return None

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        self.clicked.emit(self.tid)
        if self.master:
            self._mode = "seek"
            self.state.seek_requested.emit(max(0.0, self.state.t_at(e.position().x())))
            return
        x = e.position().x()
        t = self.state.t_at(x)
        edge = self._near_edge(x)
        if (e.modifiers() & Qt.KeyboardModifier.AltModifier) and self._clip_range() is not None:
            self._mode = "move"
            self._move_t0 = t
            self.preview = 0.0
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif edge:
            self._mode = "edge_" + edge
        else:
            self._mode = "select"
            self._press_x, self._press_t, self._dragged = x, t, False

    def mouseMoveEvent(self, e):
        x = e.position().x()
        t = max(0.0, self.state.t_at(x))
        st = self.state
        if self._mode == "move":
            self.preview = max(t - self._move_t0, -self.start)
            self.update()
        elif self._mode == "select":
            if abs(x - self._press_x) > 4 or self._dragged:
                self._dragged = True
                st.set_selection(self.tid, self._press_t, t)
        elif self._mode == "edge_left":
            st.set_selection(self.tid, t, st.sel_t1)
        elif self._mode == "edge_right":
            st.set_selection(self.tid, st.sel_t0, t)
        elif self._mode == "seek":
            st.seek_requested.emit(t)
        else:
            if self._near_edge(x):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif e.modifiers() & Qt.KeyboardModifier.AltModifier and self._clip_range() is not None:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.IBeamCursor if not self.master else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e):
        st = self.state
        if self._mode == "move":
            self.clip_moved.emit(self.tid, self.start + self.preview)
            self.preview = 0.0
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
        elif self._mode == "select" and not self._dragged:
            st.clear_selection()
            st.seek_requested.emit(max(0.0, self._press_t))
        self._mode = None

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and not self.master and self._clip_range() is not None:
            a, b = self._clip_range()
            self.state.set_selection(self.tid, a, b)

    def contextMenuEvent(self, e):
        self.clicked.emit(self.tid)
        self.context_requested.emit(self.tid, e.globalPos(), max(0.0, self.state.t_at(e.pos().x())))

    def wheelEvent(self, e):
        d = e.angleDelta().y() or e.angleDelta().x()
        mod = e.modifiers()
        if mod & Qt.KeyboardModifier.ControlModifier:
            self.state.zoom(1.2 if d > 0 else 1 / 1.2, e.position().x())
            e.accept()
        elif mod & Qt.KeyboardModifier.ShiftModifier:
            self.state.pan(-self.state.view_dur * 0.12 * (1 if d > 0 else -1))
            e.accept()
        else:
            e.ignore()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.files_dropped.emit(self.tid, urls)

    # --- рисование ---------------------------------------------------------
    def _render_cache(self):
        W, H = max(1, self.width()), max(1, self.height())
        key = (W, H, round(self.state.t0, 4), round(self.state.pps, 3), self.audio_id, self.color.name(),
               self.pyr.n, self.live is not None and len(self.live[0]), self.live and self.live[1], self.muted,
               round(self.start, 4))
        if key == self._pkey and self._pix is not None:
            return self._pix
        pm = QPixmap(W, H)
        pm.fill(QColor("#12151c"))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        st = self.state
        step = _grid_step(st.pps)
        t = math.floor(st.t0 / step) * step
        p.setPen(QPen(QColor(255, 255, 255, 12), 1))
        while True:
            x = st.x(t)
            if x > W:
                break
            if x >= 0:
                p.drawLine(QPointF(x, 0), QPointF(x, H))
            t += step
        p.setPen(QPen(QColor(255, 255, 255, 18), 1))
        p.drawLine(QPointF(0, H - 0.5), QPointF(W, H - 0.5))
        col = QColor(self.color)
        if self.muted:
            col = QColor("#5a6072")
        if self.pyr.n > 0:
            x1, x2 = st.x(self.start), st.x(self.start + self.pyr.n / self.sr)
            r = QRectF(max(-5, x1), 5, min(W + 5, x2) - max(-5, x1), H - 10)
            if r.width() > 0:
                bg = QColor(col)
                bg.setAlpha(34)
                p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 140), 1))
                p.setBrush(bg)
                p.drawRoundedRect(r, 5, 5)
                self._draw_wave(p, self.pyr, (st.t0 - self.start) * self.sr, col, H, int(max(0, x1)), int(min(W, x2)))
        if self.live is not None:
            arr, lt0 = self.live
            if len(arr):
                pyr = WavePyramid(arr)
                self._draw_wave(p, pyr, (st.t0 - lt0) * self.sr, QColor(theme.REC), H, int(max(0, st.x(lt0))), W)
        p.end()
        self._pix, self._pkey = pm, key
        return pm

    def _draw_wave(self, p, pyr, t0_samp, col, H, xa, xb):
        st = self.state
        spp = self.sr / st.pps
        mid = H / 2
        half = H / 2 - 9
        width = int(self.width())
        res = wave_columns(pyr, t0_samp, spp, width)
        if res is None:
            return
        mn, mx = res
        p.setPen(QPen(col, 1))
        for x in range(max(0, xa), min(width, xb)):
            a, b = mn[x], mx[x]
            if b - a > 1e-5:
                p.drawLine(QPointF(x + 0.5, mid - b * half), QPointF(x + 0.5, mid - a * half))

    def paintEvent(self, ev):
        p = QPainter(self)
        p.drawPixmap(0, 0, self._render_cache())
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        st = self.state
        if self.selected:
            p.setPen(QPen(QColor(theme.ACCENT), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(0.75, 0.75, W - 1.5, H - 1.5))
        # выделение
        if st.sel_tid == self.tid and st.has_selection:
            x0, x1 = st.x(st.sel_t0), st.x(st.sel_t1)
            r = QRectF(max(-2.0, x0), 2, min(W + 2.0, x1) - max(-2.0, x0), H - 4)
            sel = QColor("#ffe27a")
            sel.setAlpha(46)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(sel)
            p.drawRect(r)
            p.setPen(QPen(QColor("#ffe27a"), 1.5))
            for xx in (x0, x1):
                if 0 <= xx <= W:
                    p.drawLine(QPointF(xx, 2), QPointF(xx, H - 2))
        if self.preview and self._clip_range() is not None:
            a, b = self._clip_range()
            r = QRectF(st.x(a + self.preview), 5, (b - a) * st.pps, H - 10)
            p.setPen(QPen(QColor("#ffffff"), 1.5, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r, 5, 5)
        if self.pyr.n == 0 and self.live is None and self.empty_hint:
            f = QFont(self.font())
            f.setPointSizeF(9.5)
            p.setFont(f)
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(14, 0, W - 28, H), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.empty_hint)
        if self.armed:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.REC))
            p.drawEllipse(QPointF(12, 12), 4, 4)
        x = st.x(st.playhead)
        if 0 <= x <= W:
            p.setPen(QPen(QColor("#ffffff"), 1.4))
            p.drawLine(QPointF(x, 0), QPointF(x, H))


def _grid_step(pps):
    target = 80.0 / pps
    for s in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300):
        if s >= target:
            return float(s)
    return 600.0


class Ruler(QWidget):
    def __init__(self, state: TimelineState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setFixedHeight(RULER_H)
        self._seek = False

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._seek = True
            self.state.seek_requested.emit(max(0.0, self.state.t_at(e.position().x())))

    def mouseMoveEvent(self, e):
        if self._seek:
            self.state.seek_requested.emit(max(0.0, self.state.t_at(e.position().x())))

    def mouseReleaseEvent(self, e):
        self._seek = False

    def wheelEvent(self, e):
        d = e.angleDelta().y()
        self.state.zoom(1.2 if d > 0 else 1 / 1.2, e.position().x())

    def paintEvent(self, ev):
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#161a22"))
        st = self.state
        f = QFont(self.font())
        f.setPointSizeF(8.0)
        p.setFont(f)
        step = _grid_step(st.pps)
        t = math.floor(st.t0 / step) * step
        while True:
            x = st.x(t)
            if x > W:
                break
            if x >= 0:
                p.setPen(QPen(QColor("#3a4152"), 1))
                p.drawLine(QPointF(x, H - 6), QPointF(x, H))
                p.setPen(QColor(theme.MUTED))
                mm, ss = divmod(t, 60.0)
                p.drawText(QPointF(x + 4, H - 9), f"{int(mm)}:{ss:04.1f}" if step < 1 else f"{int(mm)}:{int(ss):02d}")
            t += step
        x = st.x(st.playhead)
        if 0 <= x <= W:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#ffffff"))
            from PyQt6.QtGui import QPolygonF
            p.drawPolygon(QPolygonF([QPointF(x - 5, 0), QPointF(x + 5, 0), QPointF(x, 9)]))
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.drawLine(QPointF(x, 9), QPointF(x, H))


class ResetSlider(QSlider):
    def __init__(self, default=0, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.default = default

    def mouseDoubleClickEvent(self, e):
        self.setValue(self.default)


def _mini(text, tip, checkable=True, w=28):
    b = QPushButton(text)
    b.setObjectName("mini")
    b.setCheckable(checkable)
    b.setFixedSize(w, 24)
    b.setToolTip(tip)
    b.setStatusTip(tip)
    return b


class TrackHeader(QFrame):
    """Левая часть строки: имя, REC-arm, Mute, Solo, FX, громкость, панорама, индикатор уровня."""
    action = pyqtSignal(str, str)           # (действие, track_id)

    def __init__(self, session, track=None, master=False, parent=None):
        super().__init__(parent)
        self.s = session
        self.tid = "master" if master else track.id
        self.master = master
        self.setObjectName("trackheader")
        self.setFixedSize(HEADER_W, MASTER_H if master else ROW_H)
        self._building = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(2)
        top = QHBoxLayout()
        top.setContentsMargins(0, 6, 0, 0)
        top.setSpacing(4)
        self.bar = QFrame()
        self.bar.setFixedWidth(6)
        top.addWidget(self.bar)
        self.name = QLabel("MASTER" if master else track.name)
        self.name.setObjectName("trackname")
        self.name.setMinimumWidth(90)
        self.name.setToolTip("Двойной клик — переименовать. Правая кнопка — меню дорожки.")
        top.addWidget(self.name, 1)
        if not master:
            self.b_arm = _mini("REC", "Записывать на эту дорожку: красная кнопка REC наверху запишет голос сюда", w=38)
            self.b_arm.setObjectName("miniarm")
            self.b_mute = _mini("M", "Mute — выключить дорожку (заглушить)")
            self.b_solo = _mini("S", "Solo — слушать только эту дорожку")
            for b in (self.b_arm, self.b_mute, self.b_solo):
                top.addWidget(b)
            self.b_arm.toggled.connect(lambda on: None if self._building else self.s.set_track_prop(self.tid, "arm", on))
            self.b_mute.toggled.connect(lambda on: None if self._building else self.s.set_track_prop(self.tid, "mute", on))
            self.b_solo.toggled.connect(lambda on: None if self._building else self.s.set_track_prop(self.tid, "solo", on))
        self.b_fx = _mini("FX", "Плагины этой дорожки: автотюн, EQ, компрессор, реверберация…", checkable=False, w=46)
        self.b_fx.clicked.connect(lambda: self.action.emit("fx", self.tid))
        top.addWidget(self.b_fx)
        lay.addLayout(top)

        r2 = QHBoxLayout()
        r2.setContentsMargins(10, 0, 0, 0)
        r2.setSpacing(6)
        lv = QLabel("Громк.")
        lv.setObjectName("muted")
        lv.setFixedWidth(40)
        r2.addWidget(lv)
        self.vol = ResetSlider(0)
        self.vol.setRange(-600, 120)
        self.vol.setToolTip("Громкость дорожки. Двойной клик — 0 дБ")
        self.vol.setStatusTip("Громкость дорожки (двойной клик — сбросить на 0 дБ)")
        self.vol.valueChanged.connect(self._vol)
        r2.addWidget(self.vol, 1)
        self.vol_lbl = QLabel("0.0 dB")
        self.vol_lbl.setObjectName("muted")
        self.vol_lbl.setFixedWidth(52)
        r2.addWidget(self.vol_lbl)
        lay.addLayout(r2)

        r3 = QHBoxLayout()
        r3.setContentsMargins(10, 0, 0, 0)
        r3.setSpacing(6)
        lp = QLabel("Пан" if not master else "")
        lp.setObjectName("muted")
        lp.setFixedWidth(40)
        r3.addWidget(lp)
        if master:
            lp.hide()
        if not master:
            self.pan = ResetSlider(0)
            self.pan.setRange(-100, 100)
            self.pan.setToolTip("Панорама: влево/вправо. Двойной клик — по центру")
            self.pan.setStatusTip("Панорама: положение дорожки между левым и правым каналом (двойной клик — центр)")
            self.pan.valueChanged.connect(self._pan)
            r3.addWidget(self.pan, 1)
            self.pan_lbl = QLabel("C")
            self.pan_lbl.setObjectName("muted")
            self.pan_lbl.setFixedWidth(52)
            r3.addWidget(self.pan_lbl)
        if not master:
            lay.addLayout(r3)
        self.meter = LevelMeter()
        self.meter.setFixedHeight(7)
        lay.addWidget(self.meter)
        lay.addSpacing(3)
        self.refresh()

    # --- обновление из модели
    def refresh(self):
        self._building = True
        try:
            if self.master:
                self.bar.setStyleSheet("background:#8fb3ff; border-radius:3px;")
                vol = self.s.project.master["vol_db"]
                n = len([p for p in self.s.project.master["plugins"]])
            else:
                t = self.s.track(self.tid)
                if t is None:
                    return
                self.name.setText(t.name)
                self.bar.setStyleSheet(f"background:{t.color}; border-radius:3px;")
                self.b_arm.setChecked(t.arm)
                self.b_arm.setEnabled(t.kind == "vocal")
                self.b_mute.setChecked(t.mute)
                self.b_solo.setChecked(t.solo)
                self.pan.setValue(int(round(t.pan * 100)))
                self.pan_lbl.setText("C" if abs(t.pan) < 0.01 else (f"L{int(-t.pan * 100)}" if t.pan < 0 else f"R{int(t.pan * 100)}"))
                vol = t.vol_db
                n = len(t.plugins)
            self.vol.setValue(int(round(vol * 10)))
            self.vol_lbl.setText(f"{vol:+.1f} dB")
            self.b_fx.setText(f"FX {n}" if n else "FX")
            self.b_fx.setProperty("has", bool(n))
            self.b_fx.style().unpolish(self.b_fx)
            self.b_fx.style().polish(self.b_fx)
        finally:
            self._building = False

    def set_selected(self, on):
        self.setProperty("sel", bool(on))
        self.style().unpolish(self)
        self.style().polish(self)

    def _vol(self, v):
        if self._building:
            return
        db = v / 10.0
        self.vol_lbl.setText(f"{db:+.1f} dB")
        if self.master:
            self.s.set_master_vol(db)
        else:
            self.s.set_track_prop(self.tid, "vol_db", db)

    def _pan(self, v):
        if self._building:
            return
        p = v / 100.0
        self.pan_lbl.setText("C" if abs(p) < 0.01 else (f"L{int(-p * 100)}" if p < 0 else f"R{int(p * 100)}"))
        self.s.set_track_prop(self.tid, "pan", p)

    # --- мышь
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.action.emit("select", self.tid)

    def mouseDoubleClickEvent(self, e):
        if not self.master:
            self.action.emit("rename", self.tid)

    def contextMenuEvent(self, e):
        if self.master:
            return
        m = QMenu(self)
        acts = [
            ("rename", "Переименовать…"), ("automix", "✨ Auto-Mix этой дорожки…"), ("color", "Цвет дорожки…"),
            ("duplicate", "Дублировать дорожку"), ("clear", "Очистить запись"), ("remove", "Удалить дорожку"),
        ]
        mp = {}
        for key, text in acts:
            mp[m.addAction(text)] = key
            if key == "automix":
                m.addSeparator()
            if key == "color":
                m.addSeparator()
        a = m.exec(e.globalPos())
        if a in mp:
            self.action.emit(mp[a], self.tid)


class TracksPanel(QWidget):
    """Линейка + строки дорожек + мастер + скроллбар + кнопки добавления дорожек."""
    request = pyqtSignal(str, str)           # (действие, track_id)
    files_dropped = pyqtSignal(str, list)
    lane_menu = pyqtSignal(str, object, float)
    clip_moved = pyqtSignal(str, float)

    def __init__(self, session, state: TimelineState, parent=None):
        super().__init__(parent)
        self.s = session
        self.state = state
        self.rows: dict[str, tuple[TrackHeader, Lane]] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # верх: подпись + линейка
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(0)
        self.btn_menu = QPushButton("＋  Добавить дорожку  ▾")
        self.btn_menu.setObjectName("addtrack")
        self.btn_menu.setFixedSize(HEADER_W - 8, RULER_H)
        self.btn_menu.setStyleSheet("padding: 0 8px; font-size: 11px;")
        self.btn_menu.setToolTip("Новая дорожка для записи голоса, вокал из файла или бит")
        self.btn_menu.setStatusTip("Добавить дорожку: для записи голоса, из аудиофайла или для бита/инструментала")
        self.btn_menu.clicked.connect(self._add_menu)
        top.addWidget(self.btn_menu)
        top.addSpacing(8)
        self.ruler = Ruler(state)
        top.addWidget(self.ruler, 1)
        outer.addLayout(top)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inner = QWidget()
        self.vl = QVBoxLayout(self.inner)
        self.vl.setContentsMargins(0, 0, 0, 0)
        self.vl.setSpacing(1)
        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(1)
        self.vl.addLayout(self.rows_box)
        add = QHBoxLayout()
        add.setContentsMargins(10, 8, 10, 8)
        self.btn_add_vocal = QPushButton("＋  Новая дорожка для записи голоса")
        self.btn_add_vocal.setObjectName("addtrack")
        self.btn_add_vocal.setToolTip("Новая дорожка для записи голоса или импорта вокала")
        self.btn_add_vocal.setStatusTip("Создать новую дорожку для записи голоса. У каждой дорожки — свои плагины.")
        self.btn_add_file = QPushButton("＋  Открыть вокал из файла")
        self.btn_add_file.setObjectName("addtrack")
        self.btn_add_file.setToolTip("Добавить готовую запись голоса (wav, mp3, flac…) на новую дорожку")
        self.btn_add_file.setStatusTip("Открыть аудиофайл с вашим голосом на новую дорожку (можно и перетащить файл в окно)")
        self.btn_add_back = QPushButton("＋  Добавить бит / инструментал")
        self.btn_add_back.setObjectName("addtrack")
        self.btn_add_back.setToolTip("Загрузить музыку, под которую вы поёте (wav, mp3, flac…)")
        self.btn_add_back.setStatusTip("Загрузить готовый бит или инструментал на отдельную дорожку")
        add.addWidget(self.btn_add_vocal)
        add.addWidget(self.btn_add_file)
        add.addWidget(self.btn_add_back)
        add.addStretch(1)
        self.vl.addLayout(add)
        self.vl.addStretch(1)
        self.scroll.setWidget(self.inner)
        outer.addWidget(self.scroll, 1)

        # мастер
        self.master_header = TrackHeader(session, master=True)
        self.master_lane = Lane("master", state, "#8fb3ff", master=True)
        self.master_lane.empty_hint = ""
        mrow = QHBoxLayout()
        mrow.setContentsMargins(0, 0, 0, 0)
        mrow.setSpacing(0)
        mrow.addWidget(self.master_header)
        mrow.addWidget(self.master_lane, 1)
        outer.addLayout(mrow)
        self.master_header.action.connect(lambda a, t: self.request.emit(a, t))
        self.master_lane.clicked.connect(lambda t: self.request.emit("select", t))

        # нижняя панель: скролл по времени + масштаб
        bot = QHBoxLayout()
        bot.setContentsMargins(HEADER_W, 2, 4, 0)
        self.sb = QScrollBar(Qt.Orientation.Horizontal)
        bot.addWidget(self.sb, 1)
        for text, tip, fn in (("−", "Уменьшить масштаб (Ctrl+колесо)", lambda: state.zoom(1 / 1.3, state.view_w / 2)),
                              ("+", "Увеличить масштаб (Ctrl+колесо)", lambda: state.zoom(1.3, state.view_w / 2)),
                              ("Fit", "Показать весь проект", state.fit)):
            b = QPushButton(text)
            b.setFixedHeight(22)
            b.setMinimumWidth(30)
            b.setToolTip(tip)
            b.setStatusTip(tip)
            b.setStyleSheet("padding: 0 8px; font-size: 11px;")
            b.clicked.connect(lambda _=False, f=fn: f())
            bot.addWidget(b)
        self.btn_follow = QPushButton("Follow")
        self.btn_follow.setCheckable(True)
        self.btn_follow.setChecked(True)
        self.btn_follow.setFixedHeight(22)
        self.btn_follow.setToolTip("Прокручивать дорожки вслед за воспроизведением")
        self.btn_follow.setStyleSheet("padding: 0 8px; font-size: 11px;")
        self.btn_follow.toggled.connect(lambda on: setattr(state, "follow", on))
        bot.addWidget(self.btn_follow)
        outer.addLayout(bot)
        self._sb_block = False
        self.sb.valueChanged.connect(self._sb_changed)
        state.changed.connect(self._state_changed)
        self.ruler.update()

    def _add_menu(self):
        m = QMenu(self)
        a1 = m.addAction("●  Новая дорожка для записи голоса")
        a2 = m.addAction("Открыть вокал из аудиофайла…")
        a3 = m.addAction("Добавить бит / инструментал…")
        a = m.exec(self.btn_menu.mapToGlobal(self.btn_menu.rect().bottomLeft()))
        if a == a1:
            self.btn_add_vocal.click()
        elif a == a2:
            self.btn_add_file.click()
        elif a == a3:
            self.btn_add_back.click()

    # ----------------------------------------------------------------------
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.state.view_w = max(100, self.width() - HEADER_W - (self.scroll.verticalScrollBar().width() if self.scroll.verticalScrollBar().isVisible() else 0))
        self._sync_sb()

    def _state_changed(self):
        self.ruler.update()
        for h, l in self.rows.values():
            l.invalidate() if l._pkey is None or self._view_changed(l) else l.update()
        self.master_lane.update() if not self._view_changed(self.master_lane) else self.master_lane.invalidate()
        self._sync_sb()

    def _view_changed(self, lane):
        return lane._pkey is None or (lane._pkey[2] != round(self.state.t0, 4) or lane._pkey[3] != round(self.state.pps, 3))

    def _sync_sb(self):
        self._sb_block = True
        total = max(self.state.duration, self.state.view_dur)
        self.sb.setRange(0, int(max(0.0, total - self.state.view_dur * 0.4) * 1000))
        self.sb.setPageStep(int(self.state.view_dur * 1000))
        self.sb.setValue(int(self.state.t0 * 1000))
        self._sb_block = False

    def _sb_changed(self, v):
        if self._sb_block:
            return
        self.state.t0 = v / 1000.0
        self.state.changed.emit()

    # ----------------------------------------------------------------------
    def rebuild(self):
        """Пересоздаёт строки дорожек по модели проекта."""
        old_live = {tid: l.live for tid, (h, l) in self.rows.items()}
        while self.rows_box.count():
            it = self.rows_box.takeAt(0)
            w = it.widget() if it else None
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif it.layout() is not None:
                lay = it.layout()
                while lay.count():
                    x = lay.takeAt(0)
                    xw = x.widget()
                    if xw is not None:
                        xw.setParent(None)
                        xw.deleteLater()
        self.rows.clear()
        for t in self.s.project.tracks:
            h = TrackHeader(self.s, t)
            l = Lane(t.id, self.state, t.color)
            l.sr = self.s.sr
            l.set_audio(t.mono(), t.audio_id or None, self.s.sr, t.start / self.s.sr) if t.audio is not None else None
            l.empty_hint = ("Пусто. Нажмите ● чтобы записать голос, или перетащите сюда аудиофайл." if t.kind == "vocal"
                            else "Пусто. Перетащите сюда бит/инструментал или нажмите «Добавить бит».")
            h.action.connect(lambda a, tid: self.request.emit(a, tid))
            l.clicked.connect(lambda tid: self.request.emit("select", tid))
            l.files_dropped.connect(lambda tid, urls: self.files_dropped.emit(tid, urls))
            l.context_requested.connect(lambda tid, pos, tt: self.lane_menu.emit(tid, pos, tt))
            l.clip_moved.connect(lambda tid, ts: self.clip_moved.emit(tid, ts))
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(0)
            row.addWidget(h)
            row.addWidget(l, 1)
            self.rows_box.addLayout(row)
            self.rows[t.id] = (h, l)
            l.live = old_live.get(t.id)
        self.update_selection()
        self.refresh_headers()

    def refresh_headers(self):
        for tid, (h, l) in self.rows.items():
            h.refresh()
            t = self.s.track(tid)
            if t:
                l.armed = t.arm
                l.muted = t.mute
                l.set_color(t.color)
                l.update()
        self.master_header.refresh()

    def update_selection(self):
        sel = self.s.sel_track
        for tid, (h, l) in self.rows.items():
            h.set_selected(tid == sel)
            l.selected = (tid == sel)
            l.update()
        self.master_header.set_selected(sel == "master")
        self.master_lane.selected = (sel == "master")
        self.master_lane.update()

    def update_audio(self, tid):
        pair = self.rows.get(tid)
        t = self.s.track(tid)
        if pair and t:
            h, l = pair
            l.set_audio(t.mono(), t.audio_id or None, self.s.sr, t.start / self.s.sr)
            if t.audio is None:
                l.pyr = WavePyramid(None)
                l.audio_id = None
                l.invalidate()

    def set_mix_wave(self, mix):
        if mix is None or mix.shape[1] == 0:
            self.master_lane.set_audio(None, None, self.s.sr)
            return
        self.master_lane.set_audio(mix.mean(axis=0), f"mix{mix.shape[1]}:{float(np.abs(mix[:, ::997]).sum()):.3f}", self.s.sr)

    def set_live(self, tid, arr, t0):
        pair = self.rows.get(tid)
        if pair:
            pair[1].live = None if arr is None else (arr, t0)
            pair[1].update()

    def update_meters(self, result, pos_samples, mixer_peak):
        for tid, (h, l) in self.rows.items():
            tr = result.tracks.get(tid) if result is not None else None
            if tr is not None and tr.audible and 0 <= pos_samples < tr.out.shape[1]:
                a = max(0, pos_samples - 1800)
                seg = tr.out[:, a:pos_samples + 1]
                h.meter.set_level(float(np.abs(seg).max()) if seg.size else 0.0)
            else:
                h.meter.set_level(0.0)
        self.master_header.meter.set_level(mixer_peak)
