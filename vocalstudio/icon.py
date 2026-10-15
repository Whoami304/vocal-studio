"""Иконка приложения, нарисованная программно."""
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPen, QPixmap, QBrush


def make_pixmap(size: int = 256) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    r = QRectF(size * 0.04, size * 0.04, size * 0.92, size * 0.92)
    g = QLinearGradient(r.topLeft(), r.bottomRight())
    g.setColorAt(0, QColor("#1d2230"))
    g.setColorAt(1, QColor("#0c0e13"))
    p.setPen(QPen(QColor("#2fd0c0"), size * 0.02))
    p.setBrush(QBrush(g))
    p.drawRoundedRect(r, size * 0.22, size * 0.22)
    # «волна → ступенька» (символ автотюна)
    pen = QPen(QColor("#6aa8ff"), size * 0.03)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    import math
    pts = [QPointF(size * (0.18 + 0.64 * i / 60), size * (0.5 + 0.17 * math.sin(i / 60 * 6.0 + 0.4) * (0.4 + i / 100))) for i in range(61)]
    for a, b in zip(pts[:-1], pts[1:]):
        p.drawLine(a, b)
    pen = QPen(QColor("#ff9f43"), size * 0.045)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    steps = [(0.18, 0.62), (0.40, 0.62), (0.40, 0.46), (0.62, 0.46), (0.62, 0.32), (0.82, 0.32)]
    for (x1, y1), (x2, y2) in zip(steps[:-1], steps[1:]):
        p.drawLine(QPointF(size * x1, size * y1), QPointF(size * x2, size * y2))
    p.end()
    return pm


def make_icon() -> QIcon:
    ic = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        ic.addPixmap(make_pixmap(s))
    return ic
