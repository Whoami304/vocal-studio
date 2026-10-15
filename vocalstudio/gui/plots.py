"""Графики: эквалайзер, кривая компрессора, трасса gain reduction."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QBrush, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QWidget

from . import theme
from ..dsp import eq as eqm
from ..plugins import spec_map

FMIN, FMAX = 20.0, 20000.0
DBR = 18.0

_BAND_DEFS = [
    # id, f-key, g-key, q-key, on-key, label, color
    ("hpf", "hpf_f", None, None, "hpf_on", "HP", "#8fb3ff"),
    ("ls", "ls_f", "ls_g", None, None, "LS", "#6be38a"),
    ("b1", "b1_f", "b1_g", "b1_q", None, "1", "#ffd36b"),
    ("b2", "b2_f", "b2_g", "b2_q", None, "2", "#ff9f43"),
    ("b3", "b3_f", "b3_g", "b3_q", None, "3", "#ff6b81"),
    ("hs", "hs_f", "hs_g", None, None, "HS", "#c58bff"),
    ("lp", "lp_f", None, None, "lp_on", "LP", "#8fb3ff"),
]


_EQ_SPECS = spec_map("eq")


def _spec(key):
    return _EQ_SPECS[key]


class EQView(QWidget):
    """Кривая эквалайзера с перетаскиваемыми узлами."""
    param_edited = pyqtSignal(str, object)     # (key, value)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.p: dict = {}
        self.sr = 44100
        self.spec_f = None
        self.spec_db = None
        self._drag = None
        self._hover = None

    def set_params(self, p, sr=44100):
        self.p = dict(p)
        self.sr = sr
        self.update()

    def set_spectrum(self, f, db):
        self.spec_f, self.spec_db = f, db
        self.update()

    def _fx(self, f):
        w = self.width() - 40
        return 30 + w * (math.log(f / FMIN) / math.log(FMAX / FMIN))

    def _xf(self, x):
        w = self.width() - 40
        return FMIN * (FMAX / FMIN) ** (min(max((x - 30) / w, 0.0), 1.0))

    def _gy(self, g):
        h = self.height() - 20
        return 8 + h * (0.5 - g / (2 * DBR))

    def _yg(self, y):
        h = self.height() - 20
        return (0.5 - (y - 8) / h) * 2 * DBR

    def _handle_pos(self, bid):
        for (b, fk, gk, qk, ok, label, col) in _BAND_DEFS:
            if b == bid:
                f = self.p.get(fk, 1000.0)
                g = self.p.get(gk, 0.0) if gk else 0.0
                return QPointF(self._fx(f), self._gy(g))

    def _hit(self, pos):
        best, bd = None, 14.0
        for (b, fk, gk, qk, ok, label, col) in _BAND_DEFS:
            hp = self._handle_pos(b)
            d = math.hypot(hp.x() - pos.x(), hp.y() - pos.y())
            if d < bd:
                best, bd = b, d
        return best

    def mousePressEvent(self, e):
        b = self._hit(e.position())
        if b is not None and e.button() == Qt.MouseButton.LeftButton:
            self._drag = b
        elif b is not None and e.button() == Qt.MouseButton.RightButton:
            for (bid, fk, gk, qk, ok, label, col) in _BAND_DEFS:
                if bid == b:
                    if ok:
                        self.param_edited.emit(ok, not self.p.get(ok, True))
                    elif gk:
                        self.param_edited.emit(gk, 0.0)

    def mouseDoubleClickEvent(self, e):
        b = self._hit(e.position())
        for (bid, fk, gk, qk, ok, label, col) in _BAND_DEFS:
            if bid == b and ok:
                self.param_edited.emit(ok, not self.p.get(ok, True))

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._drag is not None:
            for (bid, fk, gk, qk, ok, label, col) in _BAND_DEFS:
                if bid == self._drag:
                    sf = _spec(fk)
                    f = min(max(self._xf(pos.x()), sf.lo), sf.hi)
                    self.param_edited.emit(fk, round(f, 1))
                    if gk:
                        sg = _spec(gk)
                        g = min(max(self._yg(pos.y()), sg.lo), sg.hi)
                        self.param_edited.emit(gk, round(g, 1))
                    if ok and not self.p.get(ok, True):
                        self.param_edited.emit(ok, True)
            return
        h = self._hit(pos)
        if h != self._hover:
            self._hover = h
            self.setCursor(Qt.CursorShape.OpenHandCursor if h else Qt.CursorShape.ArrowCursor)
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def wheelEvent(self, e):
        b = self._hover or self._hit(e.position())
        for (bid, fk, gk, qk, ok, label, col) in _BAND_DEFS:
            if bid == b and qk:
                sq = _spec(qk)
                q = self.p.get(qk, 1.0)
                q = q * (1.12 if e.angleDelta().y() > 0 else 1 / 1.12)
                self.param_edited.emit(qk, round(min(max(q, sq.lo), sq.hi), 2))
                e.accept()
                return
        e.ignore()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#10131a"))
        f = QFont(self.font())
        f.setPointSizeF(8.0)
        p.setFont(f)
        plot = QRectF(30, 8, W - 40, H - 20)
        # сетка
        for fr in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000):
            x = self._fx(fr)
            p.setPen(QPen(QColor(255, 255, 255, 18), 1))
            p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            p.setPen(QColor(theme.MUTED))
            lab = f"{fr // 1000}k" if fr >= 1000 else str(fr)
            p.drawText(QRectF(x - 20, plot.bottom() + 2, 40, 12), Qt.AlignmentFlag.AlignCenter, lab)
        for g in range(-18, 19, 6):
            y = self._gy(g)
            p.setPen(QPen(QColor(255, 255, 255, 40 if g == 0 else 14), 1))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(0, y - 6, 26, 12), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{g:+d}" if g else "0")
        # спектр дубля
        if self.spec_f is not None:
            pts = [QPointF(plot.left(), plot.bottom())]
            for fr, db in zip(self.spec_f, self.spec_db):
                if fr < FMIN or fr > FMAX:
                    continue
                pts.append(QPointF(self._fx(fr), plot.bottom() - (plot.height() * 0.8) * min(max(db, 0.0), 1.0)))
            pts.append(QPointF(plot.right(), plot.bottom()))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(106, 168, 255, 40))
            p.drawPolygon(QPolygonF(pts))
        # кривая
        freqs = np.geomspace(FMIN, FMAX, 400)
        resp = eqm.eq_response(self.p, self.sr, freqs) if self.p else np.zeros_like(freqs)
        path = QPainterPath()
        for i, (fr, g) in enumerate(zip(freqs, resp)):
            pt = QPointF(self._fx(fr), self._gy(float(np.clip(g, -DBR, DBR))))
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        fill = QPainterPath(path)
        fill.lineTo(QPointF(self._fx(FMAX), self._gy(0)))
        fill.lineTo(QPointF(self._fx(FMIN), self._gy(0)))
        fill.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(47, 208, 192, 40))
        p.drawPath(fill)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(theme.ACCENT), 2.2))
        p.drawPath(path)
        # узлы
        for (bid, fk, gk, qk, ok, label, col) in _BAND_DEFS:
            hp = self._handle_pos(bid)
            on = self.p.get(ok, True) if ok else True
            c = QColor(col)
            if not on:
                c = QColor("#4a5163")
            p.setPen(QPen(QColor("#0b0d11"), 1.5))
            p.setBrush(c)
            r = 8 if bid == self._hover or bid == self._drag else 7
            p.drawEllipse(hp, r, r)
            p.setPen(QColor("#0b0d11"))
            ff = QFont(self.font())
            ff.setPointSizeF(6.5)
            ff.setBold(True)
            p.setFont(ff)
            p.drawText(QRectF(hp.x() - 9, hp.y() - 7, 18, 14), Qt.AlignmentFlag.AlignCenter, label)
            p.setFont(f)


class TransferCurve(QWidget):
    """Статическая кривая компрессора + рабочая точка."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(150, 150)
        self.p = {}

    def set_params(self, p):
        self.p = dict(p)
        self.update()

    def curve(self, x):
        thr = self.p.get("threshold", -18.0)
        ratio = max(1.0, self.p.get("ratio", 3.0))
        knee = self.p.get("knee", 6.0)
        over = x - thr
        if knee > 0 and abs(over) <= knee / 2:
            y = x + (1 / ratio - 1) * (over + knee / 2) ** 2 / (2 * knee)
        elif over > knee / 2:
            y = thr + over / ratio
        else:
            y = x
        mk = self.p.get("makeup", 0.0)
        if self.p.get("auto_makeup", False):
            mk += -thr * (1 - 1 / ratio) * 0.5
        return y + mk

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(8, 8, self.width() - 16, self.height() - 16)
        p.fillRect(self.rect(), QColor("#10131a"))
        p.setPen(QPen(QColor(255, 255, 255, 18), 1))
        for i in range(1, 6):
            p.drawLine(QPointF(r.left() + r.width() * i / 6, r.top()), QPointF(r.left() + r.width() * i / 6, r.bottom()))
            p.drawLine(QPointF(r.left(), r.top() + r.height() * i / 6), QPointF(r.right(), r.top() + r.height() * i / 6))
        p.setPen(QPen(QColor(255, 255, 255, 50), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.top()))
        lo, hi = -60.0, 0.0
        path = QPainterPath()
        for i, x in enumerate(np.linspace(lo, hi, 80)):
            y = min(hi, max(lo, self.curve(x)))
            pt = QPointF(r.left() + (x - lo) / (hi - lo) * r.width(), r.bottom() - (y - lo) / (hi - lo) * r.height())
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        p.setPen(QPen(QColor(theme.ACCENT), 2))
        p.drawPath(path)


class GRTrace(QWidget):
    """Кривая gain reduction по времени всего дубля."""

    def __init__(self, title="GR", maxdb=18.0, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(54)
        self.title = title
        self.maxdb = maxdb
        self.gr = None
        self.sr_div = 64
        self.playhead = 0.0
        self.duration = 1.0
        self.sr = 44100
        self.color = QColor(theme.ACCENT2)

    def set_data(self, gr, sr):
        self.gr, self.sr = gr, sr
        self.duration = 0 if gr is None else len(gr) * self.sr_div / sr
        self.update()

    def set_playhead(self, t):
        self.playhead = t
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(theme.EDGE), 1))
        p.setBrush(QColor("#10131a"))
        p.drawRoundedRect(r, 4, 4)
        f = QFont(self.font())
        f.setPointSizeF(7.5)
        p.setFont(f)
        p.setPen(QColor(theme.MUTED))
        p.drawText(QRectF(6, 2, 100, 12), Qt.AlignmentFlag.AlignLeft, self.title)
        if self.gr is None or len(self.gr) < 2:
            return
        W = int(r.width())
        n = len(self.gr)
        idx = np.linspace(0, n - 1, W).astype(int)
        # берём минимум (макс. редукцию) на колонку
        edges = np.linspace(0, n, W + 1).astype(int)
        vals = np.array([self.gr[a:max(a + 1, b)].min() for a, b in zip(edges[:-1], edges[1:])])
        vals = np.clip(vals, -self.maxdb, 0.0)
        path = QPainterPath()
        path.moveTo(QPointF(r.left(), r.top()))
        for i, v in enumerate(vals):
            path.lineTo(QPointF(r.left() + i, r.top() + (-v / self.maxdb) * r.height()))
        path.lineTo(QPointF(r.right(), r.top()))
        path.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        c = QColor(self.color)
        c.setAlpha(110)
        p.setBrush(c)
        p.drawPath(path)
        if self.duration > 0:
            x = r.left() + r.width() * min(1.0, self.playhead / self.duration)
            p.setPen(QPen(QColor(255, 255, 255, 160), 1))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
