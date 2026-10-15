"""Состояние проекта + фоновый рендер (с отменой и дебаунсом)."""
from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from .audio_io import SR, to_mono, to_stereo
from .dsp.chain import Chain, RenderResult
from .params import default_params, merge_params, PRESETS, preset_params
from .dsp.scales import NOTE_NAMES, scale_mask


def config_dir() -> str:
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(base, "VocalStudio")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


class Session(QObject):
    params_changed = pyqtSignal()            # любое изменение параметров
    param_set = pyqtSignal(str, str)         # (module, key)
    params_replaced = pyqtSignal()           # пресет / сброс — UI надо обновить целиком
    render_started = pyqtSignal()
    render_progress = pyqtSignal(str, float)
    render_done = pyqtSignal(object)         # RenderResult
    render_failed = pyqtSignal(str)
    take_changed = pyqtSignal()
    backing_changed = pyqtSignal()
    _again = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sr = SR
        self.params = default_params()
        self.take: np.ndarray | None = None       # моно float32 (n,)
        self.take_id = "none"
        self.take_name = ""
        self.backing: np.ndarray | None = None    # (2, n) float32
        self.backing_name = ""
        self.overrides: list = []                 # [(t0, t1, midi|None)]
        self.bypass_all = False
        self.result: RenderResult | None = None
        self._undo_take = None
        self.chain = Chain()
        self._gen = 0
        self._running = False
        self._pending = False
        self._lock = threading.Lock()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(140)
        self._timer.timeout.connect(self._kick)
        self._dirty_since_render = True
        self._again.connect(self._kick)

    # ---- параметры ----------------------------------------------------
    def set_param(self, module: str, key: str, value):
        if self.params[module].get(key) == value:
            return
        self.params[module][key] = value
        self.param_set.emit(module, key)
        self.params_changed.emit()
        self.request_render()

    def set_params_bulk(self, module: str, values: dict):
        self.params[module].update(values)
        self.params_replaced.emit()
        self.params_changed.emit()
        self.request_render()

    def set_scale(self, key_name: str, scale: str):
        t = self.params["tune"]
        t["key"], t["scale"] = key_name, scale
        if scale != "Custom":
            t["notes"] = scale_mask(NOTE_NAMES.index(key_name), scale)
        self.params_replaced.emit()
        self.params_changed.emit()
        self.request_render()

    def set_notes(self, mask):
        t = self.params["tune"]
        t["notes"] = list(mask)
        # Подбираем имя лада, если маска совпала с известной
        t["scale"] = "Custom"
        from .dsp.scales import SCALES
        for name in SCALES:
            if scale_mask(NOTE_NAMES.index(t["key"]), name) == list(mask):
                t["scale"] = name
                break
        self.params_replaced.emit()
        self.params_changed.emit()
        self.request_render()

    def load_preset(self, name: str):
        self.params = preset_params(name)
        self.params_replaced.emit()
        self.params_changed.emit()
        self.request_render()

    def load_params_dict(self, d: dict):
        self.params = merge_params(default_params(), d)
        self.params_replaced.emit()
        self.params_changed.emit()
        self.request_render()

    def set_overrides(self, ov: list):
        self.overrides = ov
        self.params_changed.emit()
        self.request_render()

    def set_bypass_all(self, on: bool):
        self.bypass_all = on
        self.request_render()

    # ---- данные -------------------------------------------------------
    def set_take(self, mono: np.ndarray, name: str = "", keep_undo: bool = True):
        if keep_undo and self.take is not None:
            self._undo_take = (self.take, self.take_id, self.take_name, self.overrides)
        self.take = np.ascontiguousarray(mono, dtype=np.float32)
        self.take_id = f"{time.time_ns()}"
        self.take_name = name
        self.overrides = []
        self.chain.invalidate()
        self.take_changed.emit()
        self.request_render(immediate=True)

    def undo_take(self) -> bool:
        if self._undo_take is None:
            return False
        self.take, self.take_id, self.take_name, self.overrides = self._undo_take
        self._undo_take = None
        self.chain.invalidate()
        self.take_changed.emit()
        self.request_render(immediate=True)
        return True

    def clear_take(self):
        self.take = None
        self.take_name = ""
        self.result = None
        self.chain.invalidate()
        self.take_changed.emit()
        self.request_render(immediate=True)

    def set_backing(self, stereo: np.ndarray | None, name: str = ""):
        self.backing = None if stereo is None else np.ascontiguousarray(stereo, dtype=np.float32)
        self.backing_name = name
        self.backing_changed.emit()
        self.request_render(immediate=True)

    @property
    def total_samples(self) -> int:
        n = 0 if self.take is None else len(self.take)
        if self.result is not None:
            n = max(n, self.result.out.shape[1])
        if self.backing is not None:
            n = max(n, self.backing.shape[1])
        return n

    # ---- рендер -------------------------------------------------------
    def request_render(self, immediate: bool = False):
        self._dirty_since_render = True
        self._gen += 1
        if immediate:
            self._timer.stop()
            self._kick()
        else:
            self._timer.start()

    def _kick(self):
        with self._lock:
            if self._running:
                self._pending = True
                return
            self._running = True
            self._pending = False
        gen = self._gen
        params = copy.deepcopy(self.params)
        take = self.take
        backing = self.backing
        take_id = self.take_id
        overrides = copy.deepcopy(self.overrides)
        bypass = self.bypass_all
        sr = self.sr
        self.render_started.emit()
        t = threading.Thread(target=self._work, args=(gen, params, take, backing, take_id, overrides, bypass, sr),
                             daemon=True)
        t.start()

    def _work(self, gen, params, take, backing, take_id, overrides, bypass, sr):
        try:
            cancel = lambda: gen != self._gen
            res = self.chain.render(
                None if take is None else take.astype(np.float64), sr, params, backing=backing,
                take_id=take_id, overrides=overrides, bypass_all=bypass, cancel=cancel,
                progress=lambda st, f: self.render_progress.emit(st, f),
            )
            if gen == self._gen:
                self.result = res
                self._dirty_since_render = False
                self.render_done.emit(res)
        except InterruptedError:
            pass
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc()
            self.render_failed.emit(f"{type(e).__name__}: {e}")
        finally:
            with self._lock:
                self._running = False
                again = self._pending or (gen != self._gen)
            if again:
                # перезапуск из GUI-потока
                self._again.emit()

    def render_blocking(self) -> RenderResult:
        """Синхронный рендер (для экспорта). Ждёт завершения фонового воркера."""
        self._timer.stop()
        self._gen += 1
        while True:
            with self._lock:
                if not self._running:
                    self._running = True
                    break
            time.sleep(0.02)
        try:
            res = self.chain.render(
                None if self.take is None else self.take.astype(np.float64), self.sr, copy.deepcopy(self.params),
                backing=self.backing, take_id=self.take_id, overrides=copy.deepcopy(self.overrides),
                bypass_all=self.bypass_all,
            )
            self.result = res
            self._dirty_since_render = False
            return res
        finally:
            with self._lock:
                self._running = False


# ------------------------------------------------------------ пресеты ------

def user_presets_dir() -> str:
    d = os.path.join(config_dir(), "presets")
    os.makedirs(d, exist_ok=True)
    return d


def list_user_presets() -> list[str]:
    try:
        return sorted(f[:-5] for f in os.listdir(user_presets_dir()) if f.endswith(".json"))
    except OSError:
        return []


def save_user_preset(name: str, params: dict):
    safe = "".join(ch for ch in name if ch.isalnum() or ch in " -_()").strip() or "preset"
    with open(os.path.join(user_presets_dir(), safe + ".json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=1)


def load_user_preset(name: str) -> dict:
    with open(os.path.join(user_presets_dir(), name + ".json"), "r", encoding="utf-8") as f:
        return json.load(f)


def delete_user_preset(name: str):
    try:
        os.remove(os.path.join(user_presets_dir(), name + ".json"))
    except OSError:
        pass


# ---------- настройки приложения ----------

def load_settings() -> dict:
    try:
        with open(os.path.join(config_dir(), "settings.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d: dict):
    try:
        with open(os.path.join(config_dir(), "settings.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
    except Exception:
        pass
