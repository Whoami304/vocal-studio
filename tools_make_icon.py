"""Генерирует assets/icon.ico (PNG-кадры внутри ICO) из программной иконки."""
import os, struct, sys, io
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QBuffer, QIODevice
from vocalstudio.icon import make_pixmap

app = QApplication([])
sizes = [16, 24, 32, 48, 64, 128, 256]
pngs = []
for s in sizes:
    pm = make_pixmap(s)
    buf = QBuffer(); buf.open(QIODevice.OpenModeFlag.WriteOnly)
    pm.save(buf, "PNG")
    pngs.append(bytes(buf.data()))
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"), exist_ok=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico")
header = struct.pack("<HHH", 0, 1, len(sizes))
offset = 6 + 16 * len(sizes)
entries, blobs = b"", b""
for s, data in zip(sizes, pngs):
    entries += struct.pack("<BBBBHHII", 0 if s >= 256 else s, 0 if s >= 256 else s, 0, 0, 1, 32, len(data), offset)
    offset += len(data)
    blobs += data
with open(out, "wb") as f:
    f.write(header + entries + blobs)
pm = make_pixmap(256); pm.save(os.path.join(os.path.dirname(out), "icon.png"))
print("written", out, os.path.getsize(out))
