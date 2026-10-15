"""График высоты тона (Graph-режим): пиано-ролл, исходный и скорректированный
тон, блоки целевых нот (их можно перетаскивать), форма волны и плейхед."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF, QBrush
from PyQt6.QtWidgets import QHBoxLayout, QMenu, QPushButton, QScrollBar, QVBoxLayout, QWidget, QSizePolicy

from . import theme
from ..dsp.scales import NOTE_NAMES

GUTTER = 46
RULER = 20
WAVE_H = 58
_BLACK = {1, 3, 6, 8, 10}


def midi_name(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


class WavePyramid:
    BLOCK = 64

    def __init__(self, x: np.ndarray | None):
        self.n = 0 if x is None else len(x)
        if x is None or len(x) == 0:
            self.mn = self.mx = np.zeros(0, dtype=np.float32)
            return
        B = self.BLOCK
        nb = int(np.ceil(len(x) / B))
        pad = nb * B - len(x)
        xp = np.pad(x.astype(np.float32), (0, pad))
        r = xp.reshape(nb, B)
        self.mn = r.min(axis=1)
        self.mx = r.max(axis=1)


class PitchView(QWidget):
    seek_requested = pyqtSignal(float)
    overrides_changed = pyqtSignal(list)
    view_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.t0 = 0.0          # секунды в левой границе
        self.pps = 90.0        # пикселей на секунду
        self.m_lo = 48.0
        self.m_hi = 76.0
        self.sr = 44100
        self.wave = WavePyramid(None)
        self.live = None       # (массив, t0_sec) запись в процессе
        self.track = None
        self.corr = None
        self.mask = [True] * 12
        self.transpose = 0.0
        self.overrides: list = []
        self.playhead = 0.0
        self.duration = 10.0
        self.blocks: list = []
        self._drag = None
        self._seeking = False
        self._pan = None
        self._hover_note = None
        self.follow = True
        self.in_midi = None
        self.out_midi = None
        self.hop_t = 0.0058

    # ------------------------------------------------------------ данные --
    def set_wave(self, x, sr):
        self.sr = sr
        self.wave = WavePyramid(x)
        self.update()

    def set_result(self, track, corr, sr):
        self.track, self.corr, self.sr = track, corr, sr
        if track is not None and corr is not None:
            self.hop_t = track.hop / sr
            self.in_midi = corr.in_midi
            self.out_midi = corr.out_midi
            self._build_blocks()
        else:
            self.in_midi = self.out_midi = None
            self.blocks = []
        self.update()

    def set_mask(self, mask, transpose=0.0):
        self.mask = list(mask)
        self.transpose = transpose
        self.update()

    def set_overrides(self, ov):
        self.overrides = list(ov)
        self.update()

    def set_duration(self, d):
        self.duration = max(1.0, d)
        self.view_changed.emit()

    def set_playhead(self, t):
        self.playhead = t
        if self.follow:
            x = self._x(t)
            right = self.width() - 30
            if x > right or x < GUTTER:
                self.t0 = max(0.0, t - 0.2 * self._view_dur())
                self.view_changed.emit()
        self.update()

    def set_live(self, arr, t0):
        self.live = None if arr is None else (arr, t0)
        self.update()

    def fit_vertical(self):
        m = self.in_midi
        if m is not None and np.isfinite(m).any():
            lo, hi = float(np.nanmin(m)), float(np.nanmax(m))
            if self.out_midi is not None and np.isfinite(self.out_midi).any():
                lo = min(lo, float(np.nanmin(self.out_midi)))
                hi = max(hi, float(np.nanmax(self.out_midi)))
            lo, hi = math.floor(lo) - 3, math.ceil(hi) + 3
            if hi - lo < 16:
                mid = (hi + lo) / 2
                lo, hi = mid - 8, mid + 8
            self.m_lo, self.m_hi = lo, hi
        else:
            self.m_lo, self.m_hi = 48, 76
        self.update()

    def fit_horizontal(self):
        w = max(100, self.width() - GUTTER - 10)
        self.pps = max(8.0, w / max(self.duration, 1.0))
        self.t0 = 0.0
        self.view_changed.emit()
        self.update()

    def _view_dur(self):
        return max(0.1, (self.width() - GUTTER) / self.pps)

    # ----------------------------------------------------------- геометрия --
    def _pitch_rect(self):
        return QRectF(GUTTER, RULER, max(10, self.width() - GUTTER), max(10, self.height() - RULER - WAVE_H))

    def _wave_rect(self):
        return QRectF(GUTTER, self.height() - WAVE_H, max(10, self.width() - GUTTER), WAVE_H)

    def _x(self, t):
        return GUTTER + (t - self.t0) * self.pps

    def _t(self, x):
        return self.t0 + (x - GUTTER) / self.pps

    def _y(self, m):
        r = self._pitch_rect()
        return r.top() + (self.m_hi - m) / (self.m_hi - self.m_lo) * r.height()

    def _m(self, y):
        r = self._pitch_rect()
        return self.m_hi - (y - r.top()) / r.height() * (self.m_hi - self.m_lo)

    def _build_blocks(self):
        tg = self.corr.target_midi
        blocks = []
        n = len(tg)
        i = 0
        while i < n:
            if not np.isfinite(tg[i]):
                i += 1
                continue
            j = i
            while j + 1 < n and np.isfinite(tg[j + 1]) and abs(tg[j + 1] - tg[i]) < 1e-6:
                j += 1
            if j - i >= 2:
                blocks.append((i * self.hop_t, (j + 1) * self.hop_t, float(tg[i])))
            i = j + 1
        self.blocks = blocks

    def _block_at(self, pos):
        t = self._t(pos.x())
        for (a, b, note) in self.blocks:
            if a <= t <= b and abs(self._m(pos.y()) - note) <= 0.55:
                return (a, b, note)
        return None

    def _has_override(self, a, b):
        for (oa, ob, _n) in self.overrides:
            if oa < b - 1e-6 and ob > a + 1e-6:
                return True
        return False

    # ------------------------------------------------------------- события --
    def mousePressEvent(self, e):
        pos = e.position()
        if e.button() == Qt.MouseButton.MiddleButton:
            self._pan = (pos.x(), self.t0)
            return
        if e.button() == Qt.MouseButton.RightButton:
            blk = self._block_at(pos) if self._pitch_rect().contains(pos) else None
            self._context(e.globalPosition().toPoint(), blk)
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if self._pitch_rect().contains(pos):
            blk = self._block_at(pos)
            if blk is not None:
                self._drag = {"blk": blk, "y0": pos.y(), "note": blk[2]}
                self.setCursor(Qt.CursorShape.SizeVerCursor)
                return
        if pos.x() >= GUTTER:
            self._seeking = True
            self.seek_requested.emit(max(0.0, self._t(pos.x())))

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._pan is not None:
            dx = pos.x() - self._pan[0]
            self.t0 = max(0.0, self._pan[1] - dx / self.pps)
            self.view_changed.emit()
            self.update()
            return
        if self._drag is not None:
            dm = (self._drag["y0"] - pos.y()) / self._pitch_rect().height() * (self.m_hi - self.m_lo)
            self._drag["note"] = round(self._drag["blk"][2] + dm)
            self.update()
            return
        if self._seeking:
            self.seek_requested.emit(max(0.0, self._t(pos.x())))
            return
        # подсказка
        if self._pitch_rect().contains(pos):
            m = int(round(self._m(pos.y())))
            self._hover_note = m
            blk = self._block_at(pos)
            self.setCursor(Qt.CursorShape.OpenHandCursor if blk else Qt.CursorShape.ArrowCursor)
            self.setToolTip(f"{midi_name(m)}" + ("  — потяните блок, чтобы изменить целевую ноту" if blk else ""))
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e):
        self._pan = None
        self._seeking = False
        if self._drag is not None:
            a, b, note = self._drag["blk"]
            new = self._drag["note"]
            self._drag = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if abs(new - note) >= 1:
                shift = self.transpose
                ov = [o for o in self.overrides if not (o[0] < b - 1e-6 and o[1] > a + 1e-6)]
                ov.append((a, b, float(new - shift)))
                self.overrides = ov
                self.overrides_changed.emit(list(ov))
            self.update()

    def _context(self, gpos, blk):
        m = QMenu(self)
        a_rst = a_bp = None
        if blk is not None:
            a_bp = m.addAction("Не корректировать эту ноту")
            if self._has_override(blk[0], blk[1]):
                a_rst = m.addAction("Вернуть автоматическую цель")
        a_all = m.addAction("Сбросить все ручные правки")
        a_all.setEnabled(bool(self.overrides))
        act = m.exec(gpos)
        if act is None:
            return
        if act == a_all:
            self.overrides = []
            self.overrides_changed.emit([])
        elif blk is not None and act == a_rst:
            a, b, _ = blk
            self.overrides = [o for o in self.overrides if not (o[0] < b - 1e-6 and o[1] > a + 1e-6)]
            self.overrides_changed.emit(list(self.overrides))
        elif blk is not None and act == a_bp:
            a, b, _ = blk
            ov = [o for o in self.overrides if not (o[0] < b - 1e-6 and o[1] > a + 1e-6)]
            ov.append((a, b, None))
            self.overrides = ov
            self.overrides_changed.emit(list(ov))
        self.update()

    def wheelEvent(self, e):
        d = e.angleDelta().y()
        if d == 0:
            d = e.angleDelta().x()
        mod = e.modifiers()
        if mod & Qt.KeyboardModifier.ControlModifier:
            # вертикальный масштаб
            f = 0.88 if d > 0 else 1.14
            mc = self._m(e.position().y())
            self.m_lo = mc - (mc - self.m_lo) * f
            self.m_hi = mc + (self.m_hi - mc) * f
            if self.m_hi - self.m_lo < 6:
                c = (self.m_hi + self.m_lo) / 2
                self.m_lo, self.m_hi = c - 3, c + 3
        elif mod & Qt.KeyboardModifier.AltModifier:
            sh = (self.m_hi - self.m_lo) * 0.08 * (1 if d > 0 else -1)
            self.m_lo += sh
            self.m_hi += sh
        elif mod & Qt.KeyboardModifier.ShiftModifier:
            self.t0 = max(0.0, self.t0 - (self._view_dur() * 0.1) * (1 if d > 0 else -1))
            self.view_changed.emit()
        else:
            tc = self._t(e.position().x())
            f = 1.18 if d > 0 else 1 / 1.18
            self.pps = min(4000.0, max(4.0, self.pps * f))
            self.t0 = max(0.0, tc - (e.position().x() - GUTTER) / self.pps)
            self.view_changed.emit()
        self.update()
        e.accept()

    # ------------------------------------------------------------- рисование --
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#10131a"))
        pr = self._pitch_rect()
        wr = self._wave_rect()

        # --- строки полутонов
        p.save()
        p.setClipRect(QRectF(0, RULER, W, pr.height()))
        lo, hi = int(math.floor(self.m_lo)) - 1, int(math.ceil(self.m_hi)) + 1
        f = QFont(self.font())
        f.setPointSizeF(8.0)
        p.setFont(f)
        row_h = pr.height() / (self.m_hi - self.m_lo)
        for n in range(lo, hi + 1):
            yt, yb = self._y(n + 0.5), self._y(n - 0.5)
            pc = n % 12
            on = self.mask[pc]
            r = QRectF(GUTTER, yt, W - GUTTER, yb - yt)
            if on:
                p.fillRect(r, QColor(47, 208, 192, 22 if pc not in _BLACK else 30))
            elif pc in _BLACK:
                p.fillRect(r, QColor(0, 0, 0, 55))
            p.setPen(QPen(QColor(255, 255, 255, 26 if pc == 0 else 10), 1))
            p.drawLine(QPointF(GUTTER, yb), QPointF(W, yb))
            # клавиша
            kc = QColor("#272c38") if pc in _BLACK else QColor("#3a4152")
            if on:
                kc = QColor("#1c8a80") if pc in _BLACK else QColor("#2fd0c0")
            p.fillRect(QRectF(0, yt + 0.5, GUTTER - 2, max(1.0, yb - yt - 1)), kc)
            if row_h >= 12 and (pc == 0 or row_h >= 15):
                p.setPen(QColor("#0c1a18") if on and pc not in _BLACK else QColor("#aab2c4"))
                p.drawText(QRectF(2, yt, GUTTER - 8, yb - yt), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                           midi_name(n))
        p.restore()

        # --- сетка времени и линейка
        p.fillRect(QRectF(0, 0, W, RULER), QColor("#161a22"))
        step = self._grid_step()
        t_start = math.floor(self.t0 / step) * step
        t = t_start
        p.setClipRect(QRectF(GUTTER, 0, W - GUTTER, H))
        f2 = QFont(self.font())
        f2.setPointSizeF(8.0)
        p.setFont(f2)
        while True:
            x = self._x(t)
            if x > W:
                break
            if x >= GUTTER:
                p.setPen(QPen(QColor(255, 255, 255, 14), 1))
                p.drawLine(QPointF(x, RULER), QPointF(x, pr.bottom()))
                p.setPen(QColor(theme.MUTED))
                mm, ss = divmod(t, 60.0)
                p.drawText(QPointF(x + 4, 13), f"{int(mm)}:{ss:04.1f}" if step < 1 else f"{int(mm)}:{int(ss):02d}")
                p.setPen(QPen(QColor("#3a4152"), 1))
                p.drawLine(QPointF(x, RULER - 5), QPointF(x, RULER))
            t += step

        # --- блоки нот
        p.setClipRect(QRectF(GUTTER, RULER, W - GUTTER, pr.height()))
        for (a, b, note) in self.blocks:
            x1, x2 = self._x(a), self._x(b)
            if x2 < GUTTER or x1 > W:
                continue
            ov = self._has_override(a, b)
            cur = note
            if self._drag is not None and self._drag["blk"][0] == a:
                cur = self._drag["note"]
            r = QRectF(x1, self._y(cur + 0.42), max(2.0, x2 - x1), self._y(cur - 0.42) - self._y(cur + 0.42))
            col = QColor(theme.WARN) if ov or (self._drag is not None and self._drag["blk"][0] == a) else QColor(theme.ACCENT)
            p.setPen(QPen(col, 1.2))
            p.setBrush(QColor(col.red(), col.green(), col.blue(), 55))
            p.drawRoundedRect(r, 3, 3)
        # bypass-блоки (правка None)
        for (a, b, n) in self.overrides:
            if n is None:
                x1, x2 = self._x(a), self._x(b)
                p.setPen(QPen(QColor(theme.MUTED), 1, Qt.PenStyle.DashLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(QRectF(x1, pr.top() + 2, max(2.0, x2 - x1), pr.height() - 4))

        # --- кривые
        if self.in_midi is not None:
            self._draw_curve(p, self.in_midi, QColor(theme.BLUE), 1.4, 190)
            self._draw_curve(p, self.out_midi, QColor(theme.ACCENT2), 2.2, 255)

        # --- волна
        p.setClipRect(QRectF(GUTTER, wr.top(), wr.width(), wr.height()))
        p.fillRect(wr, QColor("#0d1016"))
        mid = wr.center().y()
        p.setPen(QPen(QColor(255, 255, 255, 18), 1))
        p.drawLine(QPointF(GUTTER, mid), QPointF(W, mid))
        self._draw_wave(p, wr)

        # --- плейхед
        p.setClipRect(QRectF(0, 0, W, H))
        x = self._x(self.playhead)
        if GUTTER <= x <= W:
            p.setPen(QPen(QColor("#ffffff"), 1.4))
            p.drawLine(QPointF(x, 0), QPointF(x, H))
            tri = QPolygonF([QPointF(x - 5, 0), QPointF(x + 5, 0), QPointF(x, 8)])
            p.setBrush(QColor("#ffffff"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(tri)

        # --- пустое состояние
        if self.in_midi is None and self.wave.n == 0 and self.live is None:
            p.setPen(QColor(theme.MUTED))
            f3 = QFont(self.font())
            f3.setPointSizeF(12.0)
            p.setFont(f3)
            p.drawText(QRectF(GUTTER, RULER, W - GUTTER, pr.height()), Qt.AlignmentFlag.AlignCenter,
                       "Нажмите REC, чтобы записать вокал, или откройте аудиофайл (Open Vocal)")
        p.setClipping(False)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(theme.EDGE), 1))
        p.drawRect(QRectF(0.5, 0.5, W - 1, H - 1))

    def _grid_step(self):
        target = 90.0 / self.pps
        for s in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300):
            if s >= target:
                return float(s)
        return 600.0

    def _draw_curve(self, p, midi, color, width, alpha):
        if midi is None or len(midi) == 0:
            return
        W = self.width()
        i0 = max(0, int(self._t(GUTTER) / self.hop_t) - 1)
        i1 = min(len(midi), int(self._t(W) / self.hop_t) + 2)
        if i1 <= i0:
            return
        seg = midi[i0:i1]
        stride = max(1, int(1.0 / (self.hop_t * self.pps)))
        idx = np.arange(i0, i1, stride)
        vals = midi[idx]
        pen = QPen(QColor(color.red(), color.green(), color.blue(), alpha), width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        run = []
        for i, v in zip(idx, vals):
            if np.isfinite(v):
                run.append(QPointF(self._x(i * self.hop_t), self._y(v)))
            else:
                if len(run) > 1:
                    p.drawPolyline(QPolygonF(run))
                run = []
        if len(run) > 1:
            p.drawPolyline(QPolygonF(run))

    def _draw_wave(self, p, wr):
        W = int(wr.width())
        if W <= 2:
            return
        col = QColor("#6c7690")
        half = wr.height() / 2 - 3
        mid = wr.center().y()
        spp = self.sr / self.pps
        t_left = self.t0
        # исходная
        if self.wave.n > 0:
            res = self._columns(self.wave, t_left, spp, W)
            if res is not None:
                mn, mx = res
                p.setPen(QPen(col, 1))
                xs = GUTTER + np.arange(W)
                for x, a, b in zip(xs, mn, mx):
                    if b - a > 1e-5:
                        p.drawLine(QPointF(x, mid - b * half), QPointF(x, mid - a * half))
        # живая запись
        if self.live is not None:
            arr, lt0 = self.live
            n = len(arr)
            if n > 0:
                pyr = WavePyramid(arr)
                off = (t_left - lt0) * self.sr
                res = self._columns(pyr, off, spp, W)
                if res is not None:
                    mn, mx = res
                    p.setPen(QPen(QColor(theme.REC), 1))
                    xs = GUTTER + np.arange(W)
                    for x, a, b in zip(xs, mn, mx):
                        if b - a > 1e-5:
                            p.drawLine(QPointF(x, mid - b * half), QPointF(x, mid - a * half))

    def _columns(self, pyr, t0_samp, spp, W):
        """min/max на колонку (простая реализация через блоки пирамиды)."""
        if pyr.n == 0:
            return None
        B = pyr.BLOCK
        cols = np.arange(W)
        s = t0_samp + cols * spp
        e = s + spp
        mn = np.zeros(W, dtype=np.float32)
        mx = np.zeros(W, dtype=np.float32)
        nb = len(pyr.mn)
        b0 = np.floor(s / B).astype(np.int64)
        b1 = np.ceil(e / B).astype(np.int64)
        b1 = np.maximum(b1, b0 + 1)
        ok = (b1 > 0) & (b0 < nb)
        if not ok.any():
            return None
        b0c = np.clip(b0, 0, nb - 1)
        b1c = np.clip(b1, 1, nb)
        if spp <= B * 2:
            # близкий масштаб: по одному блоку на колонку
            mn[ok] = pyr.mn[b0c[ok]]
            mx[ok] = pyr.mx[b0c[ok]]
        else:
            width_blocks = int(np.max(b1c[ok] - b0c[ok]))
            # векторное окно: берём максимум по фиксированному числу блоков
            idx = b0c[:, None] + np.arange(width_blocks)[None, :]
            valid = idx < b1c[:, None]
            idx = np.clip(idx, 0, nb - 1)
            mx_ = np.where(valid, pyr.mx[idx], -np.inf).max(axis=1)
            mn_ = np.where(valid, pyr.mn[idx], np.inf).min(axis=1)
            mx[ok] = mx_[ok]
            mn[ok] = mn_[ok]
        return mn, mx


class PitchPanel(QWidget):
    """PitchView + скроллбар + кнопки масштаба."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = PitchView(self)
        self.sb = QScrollBar(Qt.Orientation.Horizontal, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.sb, 1)
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setToolTip("Показать весь дубль и подобрать высотный диапазон")
        self.btn_follow = QPushButton("Follow")
        self.btn_follow.setCheckable(True)
        self.btn_follow.setChecked(True)
        self.btn_follow.setToolTip("Прокручивать график вслед за воспроизведением")
        for b in (self.btn_fit, self.btn_follow):
            b.setFixedHeight(22)
            b.setStyleSheet("padding: 0 10px; font-size: 11px;")
        row.addWidget(self.btn_fit)
        row.addWidget(self.btn_follow)
        lay.addLayout(row)
        self.btn_fit.clicked.connect(self._fit)
        self.btn_follow.toggled.connect(lambda on: setattr(self.view, "follow", on))
        self.sb.valueChanged.connect(self._sb_changed)
        self.view.view_changed.connect(self._sync_sb)
        self._block = False

    def _fit(self):
        self.view.fit_horizontal()
        self.view.fit_vertical()

    def _sb_changed(self, v):
        if self._block:
            return
        self.view.t0 = v / 1000.0
        self.view.update()

    def _sync_sb(self):
        self._block = True
        total = max(self.view.duration, self.view._view_dur())
        self.sb.setRange(0, int(max(0.0, total - self.view._view_dur() * 0.5) * 1000))
        self.sb.setPageStep(int(self.view._view_dur() * 1000))
        self.sb.setValue(int(self.view.t0 * 1000))
        self._block = False
