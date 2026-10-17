"""Защита от ошибок: журнал, подавление повторов, понятное сообщение вместо «падения»."""
from __future__ import annotations

import functools
import os
import time
import traceback

from ..session import config_dir

_last_dialog = 0.0
_seen: dict = {}
_recent: list = []


def log_path() -> str:
    return os.path.join(config_dir(), "error.log")


def log_exception(exc_type, exc, tb, where: str = "") -> str:
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {where}\n{text}\n"
    try:
        with open(log_path(), "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    _recent.append(line)
    del _recent[:-20]
    return text


def recent_errors() -> str:
    try:
        with open(log_path(), "r", encoding="utf-8") as f:
            data = f.read()
        return data[-6000:]
    except Exception:
        return "".join(_recent)


def report(exc_type, exc, tb, where: str = "", parent=None, status=None):
    """Пишет в журнал; показывает окно не чаще раза в 30 с и не для повторяющейся ошибки."""
    global _last_dialog
    log_exception(exc_type, exc, tb, where)
    key = (exc_type.__name__, str(exc)[:100], where)
    _seen[key] = _seen.get(key, 0) + 1
    msg = f"Ошибка ({where or exc_type.__name__}): {exc}"
    if status is not None:
        try:
            status(msg + f"  — подробности: {log_path()}", 15000)
        except Exception:
            pass
    now = time.time()
    if _seen[key] == 1 and now - _last_dialog > 30.0:
        _last_dialog = now
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(parent, "Vocal Studio — что-то пошло не так",
                                f"{exc_type.__name__}: {exc}\n\nПрограмма продолжает работу. Сведения записаны в журнал:\n{log_path()}\n"
                                "Если ошибка повторяется, отправьте этот файл разработчику (Помощь → Диагностика).")
        except Exception:
            pass


def guard(fn):
    """Декоратор для методов окна: исключение не роняет программу, а попадает в журнал и строку состояния."""
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        try:
            return fn(self, *a, **k)
        except Exception:
            import sys
            et, ev, tb = sys.exc_info()
            st = None
            try:
                st = self.statusBar().showMessage
            except Exception:
                pass
            report(et, ev, tb, where=fn.__name__, parent=self if hasattr(self, "isVisible") else None, status=st)
            return None
    return wrapper
