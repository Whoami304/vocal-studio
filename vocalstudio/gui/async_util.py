"""Запуск тяжёлой функции в потоке с возвратом результата в GUI-поток."""
from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, pyqtSignal

_alive: list = []


class _Bridge(QObject):
    done = pyqtSignal(object, object)


def run_async(fn, callback):
    """fn() выполняется в потоке; callback(result, error) — в GUI-потоке."""
    b = _Bridge()
    _alive.append(b)

    def finish(res, err):
        try:
            callback(res, err)
        finally:
            if b in _alive:
                _alive.remove(b)

    b.done.connect(finish)

    def work():
        try:
            r = fn()
            b.done.emit(r, None)
        except Exception as e:  # pragma: no cover
            b.done.emit(None, e)

    threading.Thread(target=work, daemon=True).start()
