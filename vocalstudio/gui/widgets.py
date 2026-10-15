"""Базовые виджеты: ручка, клавиатура нот, индикаторы."""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QBrush, QLinearGradient
from PyQt6.QtWidgets import QInputDialog, QMenu, QSizePolicy, QWidget

from ..params import Spec
from . import theme


def spec_to_pos(spec: Spec, v: float) -> float:
    lo, hi = spec.lo, spec.hi
    v = min(max(v, lo), hi)
    if hi == lo:
        return 0.0
    if spec.log and lo > 0:
        return math.log(v / lo) / math.log(hi / lo)
    t = (v - lo) / (hi - lo)
    return t ** (1.0 / spec.curve)


def pos_to_spec(spec: Spec, pos: float) -> float:
    pos = min(max(pos, 0.0), 1.0)
    lo, hi = spec.lo, spec.hi
    if spec.log and lo > 0:
        v = lo * (hi / lo) ** pos
    else:
        v = lo + (hi - lo) * pos ** spec.curve
    if spec.decimals == 0:
        v = round(v)
    else:
        v = round(v, spec.decimals + 1)
    return float(min(max(v, lo), hi))


def fmt_value(spec: Spec, v: float) -> str:
    if spec.decimals == 0:
        s = f"{int(round(v))}"
    else:
        s = f"{v:.{spec.decimals}f}"
    if spec.key in ("hpf_f", "ls_f", "b1_f", "b2_f", "b3_f", "hs_f", "lp_f", "freq", "lowcut", "highcut") and v >= 1000:
        s = f"{v / 1000:.2f}".rstrip("0").rstrip(".") + " k"
        return s + "Hz"
    return f"{s} {spec.unit}".strip()


class Knob(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, spec: Spec, value: float | None = None, size: int = 54, accent: str = theme.ACCENT, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.value = float(spec.default if value is None else value)
        self.size_px = size
        self.accent = QColor(accent)
        self._drag_y = None
        self._drag_pos = 0.0
        self.setFixedSize(max(size + 14, 70), size + 36)
        self.setToolTip(spec.tip or spec.label)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

    def set_value(self, v: float, emit: bool = False):
        v = min(max(float(v), self.spec.lo), self.spec.hi)
        if v != self.value:
            self.value = v
            self.update()
            if emit:
                self.valueChanged.emit(v)

    # --- события -------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_y = e.position().y()
            self._drag_pos = spec_to_pos(self.spec, self.value)
            self.setFocus()
        e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_y is None:
            return
        dy = self._drag_y - e.position().y()
        fine = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        pos = self._drag_pos + dy / (900.0 if fine else 180.0)
        self._set_pos(pos)

    def mouseReleaseEvent(self, e):
        self._drag_y = None

    def mouseDoubleClickEvent(self, e):
        self.set_value(self.spec.default, emit=True)

    def wheelEvent(self, e):
        step = 0.004 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0.02
        d = 1 if e.angleDelta().y() > 0 else -1
        self._set_pos(spec_to_pos(self.spec, self.value) + d * step)
        e.accept()

    def contextMenuEvent(self, e):
        m = QMenu(self)
        a1 = m.addAction("Сбросить по умолчанию")
        a2 = m.addAction("Ввести значение…")
        act = m.exec(e.globalPos())
        if act == a1:
            self.set_value(self.spec.default, emit=True)
        elif act == a2:
            v, ok = QInputDialog.getDouble(self, self.spec.label, f"{self.spec.label} ({self.spec.unit})",
                                           self.value, self.spec.lo, self.spec.hi, max(self.spec.decimals, 1))
            if ok:
                self.set_value(v, emit=True)

    def _set_pos(self, pos):
        v = pos_to_spec(self.spec, pos)
        if v != self.value:
            self.value = v
            self.update()
            self.valueChanged.emit(v)

    # --- рисование -----------------------------------------------------
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        f = QFont(self.font())
        f.setPointSizeF(8.0)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(theme.MUTED))
        fm = QFontMetrics(f)
        label = fm.elidedText(self.spec.label.upper(), Qt.TextElideMode.ElideRight, w - 2)
        p.drawText(QRectF(0, 0, w, 14), Qt.AlignmentFlag.AlignCenter, label)

        s = self.size_px
        cx, cy = w / 2.0, 16 + s / 2.0
        r = s / 2.0 - 4
        rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
        # корпус
        grad = QLinearGradient(cx, cy - r, cx, cy + r)
        grad.setColorAt(0, QColor("#2c313e"))
        grad.setColorAt(1, QColor("#1b1e26"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(QRectF(cx - r + 4, cy - r + 4, 2 * r - 8, 2 * r - 8))
        # дорожка
        pen = QPen(QColor("#2b303d"), 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(rect, int(225 * 16), int(-270 * 16))
        pos = spec_to_pos(self.spec, self.value)
        bip = self.spec.lo < 0 < self.spec.hi
        pen = QPen(self.accent, 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        if bip:
            zero = spec_to_pos(self.spec, 0.0)
            a0 = 225 - 270 * zero
            span = -270 * (pos - zero)
            if abs(span) > 0.5:
                p.drawArc(rect, int(a0 * 16), int(span * 16))
        else:
            if pos > 0.002:
                p.drawArc(rect, int(225 * 16), int(-270 * pos * 16))
        # указатель
        ang = math.radians(225 - 270 * pos)
        p1 = QPointF(cx + math.cos(ang) * (r - 12), cy - math.sin(ang) * (r - 12))
        p2 = QPointF(cx + math.cos(ang) * (r - 4), cy - math.sin(ang) * (r - 4))
        pen = QPen(QColor("#f0f3fa"), 2.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(p1, p2)
        # значение
        f2 = QFont(self.font())
        f2.setPointSizeF(9.0)
        p.setFont(f2)
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRectF(0, 16 + s + 1, w, 16), Qt.AlignmentFlag.AlignCenter, fmt_value(self.spec, self.value))


NOTE_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_BLACK = {1, 3, 6, 8, 10}


class NoteKeys(QWidget):
    """Одна октава клавиш: нажатием включаются/выключаются ноты, к которым тянет автотюн."""
    changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mask = [True] * 12
        self.setMinimumSize(340, 110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(110)
        self.setToolTip("Клик по клавише включает/выключает ноту. Автотюн подтягивает голос только к включённым нотам.")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hover = -1
        self.setMouseTracking(True)

    def set_mask(self, mask):
        self.mask = list(mask)
        self.update()

    def _geometry(self):
        w, h = self.width(), self.height()
        white = [0, 2, 4, 5, 7, 9, 11]
        kw = w / 7.0
        rects = {}
        for i, n in enumerate(white):
            rects[n] = QRectF(i * kw + 1, 0, kw - 2, h - 1)
        bw = kw * 0.62
        bpos = {1: 0, 3: 1, 6: 3, 8: 4, 10: 5}
        for n, i in bpos.items():
            rects[n] = QRectF((i + 1) * kw - bw / 2, 0, bw, h * 0.6)
        return rects

    def _hit(self, pt):
        rects = self._geometry()
        for n in sorted(rects, key=lambda k: (k not in _BLACK)):  # чёрные первыми
            if rects[n].contains(pt):
                return n
        return -1

    def mousePressEvent(self, e):
        n = self._hit(e.position())
        if n >= 0:
            self.mask[n] = not self.mask[n]
            self.update()
            self.changed.emit(list(self.mask))

    def mouseMoveEvent(self, e):
        n = self._hit(e.position())
        if n != self._hover:
            self._hover = n
            self.update()

    def leaveEvent(self, e):
        self._hover = -1
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rects = self._geometry()
        f = QFont(self.font())
        f.setPointSizeF(8.0)
        f.setBold(True)
        p.setFont(f)
        for n in [k for k in rects if k not in _BLACK] + [k for k in rects if k in _BLACK]:
            r = rects[n]
            on = self.mask[n]
            black = n in _BLACK
            if black:
                base = QColor("#1a1d24") if not on else QColor("#1c8a80")
            else:
                base = QColor("#c9ced9") if not on else QColor(theme.ACCENT)
                if not on:
                    base = QColor("#3a4050")
            if n == self._hover:
                base = base.lighter(125)
            p.setPen(QPen(QColor("#0e1014"), 1))
            p.setBrush(base)
            p.drawRoundedRect(r, 4, 4)
            p.setPen(QColor("#0b1c1a") if (on and not black) else QColor("#9aa3b5" if not on else "#e9fffb"))
            p.drawText(QRectF(r.x(), r.bottom() - 20, r.width(), 18) if not black else QRectF(r.x(), r.bottom() - 18, r.width(), 16),
                       Qt.AlignmentFlag.AlignCenter, NOTE_LABELS[n])


class LevelMeter(QWidget):
    """Горизонтальный пик-метр (дБ)."""

    def __init__(self, parent=None, vertical=False):
        super().__init__(parent)
        self.level = 0.0
        self.hold = 0.0
        self.vertical = vertical
        self.setMinimumSize(90, 10) if not vertical else self.setMinimumSize(10, 80)

    def set_level(self, lin: float):
        self.level = max(lin, self.level * 0.82)
        self.hold = max(lin, self.hold * 0.985)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(theme.EDGE), 1))
        p.setBrush(QColor("#10131a"))
        p.drawRoundedRect(r, 3, 3)

        def pos(lin):
            db = 20 * math.log10(max(lin, 1e-5))
            return min(max((db + 60.0) / 60.0, 0.0), 1.0)

        frac = pos(self.level)
        if self.vertical:
            h = r.height() * frac
            bar = QRectF(r.x(), r.bottom() - h, r.width(), h)
        else:
            bar = QRectF(r.x(), r.y(), r.width() * frac, r.height())
        g = QLinearGradient(r.topLeft(), r.topRight() if not self.vertical else r.bottomLeft())
        g.setColorAt(0.0, QColor(theme.GREEN))
        g.setColorAt(0.75, QColor(theme.WARN))
        g.setColorAt(1.0, QColor(theme.REC))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(g))
        p.drawRoundedRect(bar, 3, 3)
        hp = pos(self.hold)
        p.setPen(QPen(QColor("#e6e9f2"), 1.5))
        if self.vertical:
            y = r.bottom() - r.height() * hp
            p.drawLine(QPointF(r.x(), y), QPointF(r.right(), y))
        else:
            x = r.x() + r.width() * hp
            p.drawLine(QPointF(x, r.y()), QPointF(x, r.bottom()))
