"""Сессия: проект + микшер + фоновый рендер + Auto-Mix. Qt-слой между GUI и ядром."""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import sys
import threading
import time

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from . import analysis, editing
from .dsp.pitch import detect_pitch
from .dsp.scales import NOTE_NAMES, SCALES, scale_mask
from .mixer import MixResult, Mixer
from .plugins import (CHAIN_PRESETS, PLUGINS, apply_plugin_preset, new_plugin)
from .project import Project, Track, new_audio_id


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


MASTER = "master"


class Session(QObject):
    structure_changed = pyqtSignal()               # дорожки/плагины добавлены, удалены, переставлены
    track_changed = pyqtSignal(str)                # свойства дорожки (громкость, имя, mute...)
    plugin_changed = pyqtSignal(str, str, str)     # (scope, plugin_id, key) — параметр
    plugins_replaced = pyqtSignal(str)             # у scope заменены значения целиком (пресет)
    selection_changed = pyqtSignal()
    audio_changed = pyqtSignal(str)                # аудио дорожки изменилось
    project_replaced = pyqtSignal()
    dirty_changed = pyqtSignal(bool)
    render_started = pyqtSignal()
    render_progress = pyqtSignal(str, float)
    render_done = pyqtSignal(object)
    render_failed = pyqtSignal(str)
    automix_progress = pyqtSignal(str)
    automix_done = pyqtSignal(object)
    history_changed = pyqtSignal()
    _again = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = Project()
        self.mixer = Mixer()
        self._tmp_mixer = Mixer()
        self.result: MixResult | None = None
        self.bypass_all = False
        self.sel_track: str | None = None
        self.sel_plugin: str | None = None
        self._gen = 0
        self._running = False
        self._pending = False
        self._lock = threading.Lock()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(140)
        self._timer.timeout.connect(self._kick)
        self._again.connect(self._kick)
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._last_key = None
        self._last_t = 0.0
        self.clipboard = None            # (audio (C, n) float32)

    # ================================================================ история ==
    def _capture(self) -> dict:
        p = self.project
        return {
            "tracks": [dataclasses.replace(t, plugins=copy.deepcopy(t.plugins)) for t in p.tracks],
            "master": copy.deepcopy(p.master), "bpm": p.bpm,
        }

    def checkpoint(self, coalesce: str | None = None):
        """Сохраняет состояние проекта перед изменением (для Отменить/Повторить).
        Одинаковые подряд идущие изменения (ручка, ползунок) за ~1.2 с объединяются в один шаг."""
        now = time.monotonic()
        if coalesce is not None and coalesce == self._last_key and now - self._last_t < 1.2:
            self._last_t = now
            return
        self._undo_stack.append(self._capture())
        del self._undo_stack[:-80]
        self._redo_stack.clear()
        self._last_key, self._last_t = coalesce, now
        self.history_changed.emit()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._capture())
        self._restore(self._undo_stack.pop())
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._capture())
        self._restore(self._redo_stack.pop())
        return True

    def _restore(self, st: dict):
        p = self.project
        arms = {t.id: t.arm for t in p.tracks}
        old_audio = {t.id: (t.audio_id, t.start) for t in p.tracks}
        p.tracks = [dataclasses.replace(t, plugins=copy.deepcopy(t.plugins), arm=arms.get(t.id, False)) for t in st["tracks"]]
        p.master = copy.deepcopy(st["master"])
        p.bpm = st["bpm"]
        ids = {t.id for t in p.tracks}
        if self.sel_track not in ids and self.sel_track != MASTER:
            self.sel_track = p.tracks[0].id if p.tracks else None
        if self.sel_plugin is not None and self.find_plugin(self.sel_track, self.sel_plugin) is None:
            self.sel_plugin = None
        self._last_key = None
        self.mark_dirty()
        for t in p.tracks:
            if old_audio.get(t.id) != (t.audio_id, t.start):
                self.audio_changed.emit(t.id)
        self.structure_changed.emit()
        self.track_changed.emit("*")
        self.selection_changed.emit()
        for sc in [MASTER] + [t.id for t in p.tracks]:
            self.plugins_replaced.emit(sc)
        self.history_changed.emit()
        self.request_render(immediate=True)

    # ================================================================ доступ ==
    @property
    def sr(self) -> int:
        return self.project.sr

    def track(self, tid) -> Track | None:
        return self.project.get(tid)

    def scope_plugins(self, scope: str) -> list:
        if scope == MASTER:
            return self.project.master["plugins"]
        t = self.project.get(scope)
        return t.plugins if t else []

    def find_plugin(self, scope: str, pid: str):
        for p in self.scope_plugins(scope):
            if p["id"] == pid:
                return p
        return None

    def mark_dirty(self, d: bool = True):
        if self.project.dirty != d:
            self.project.dirty = d
            self.dirty_changed.emit(d)

    # =============================================================== дорожки ==
    def add_track(self, kind="vocal", name=None, audio=None, source="", select=True, start=0) -> Track:
        self.checkpoint()
        t = self.project.add_track(name, kind, audio, source)
        t.start = int(start)
        if select:
            self.sel_track, self.sel_plugin = t.id, None
        self.mark_dirty()
        self.structure_changed.emit()
        self.selection_changed.emit()
        if audio is not None:
            self.audio_changed.emit(t.id)
        self.request_render(immediate=True)
        return t

    def remove_track(self, tid):
        self.checkpoint()
        self.project.remove_track(tid)
        if self.sel_track == tid:
            self.sel_track = self.project.tracks[0].id if self.project.tracks else None
            self.sel_plugin = None
        self.mark_dirty()
        self.structure_changed.emit()
        self.selection_changed.emit()
        self.request_render(immediate=True)

    def duplicate_track(self, tid):
        t = self.project.get(tid)
        if t is None:
            return None
        self.checkpoint()
        n = self.project.add_track(t.name + " copy", t.kind, t.audio, t.source_name)
        n.audio_id = t.audio_id
        n.start = t.start
        n.vol_db, n.pan = t.vol_db, t.pan
        n.plugins = [dict(copy.deepcopy(p), id=new_plugin(p["type"])["id"]) for p in t.plugins]
        self.mark_dirty()
        self.structure_changed.emit()
        self.request_render(immediate=True)
        return n

    def select(self, tid, pid=None):
        if self.sel_track != tid or self.sel_plugin != pid:
            self.sel_track, self.sel_plugin = tid, pid
            self.selection_changed.emit()

    def set_track_prop(self, tid, key, value):
        t = self.project.get(tid)
        if t is None or getattr(t, key) == value:
            return
        if key == "arm":
            if value:
                for o in self.project.tracks:
                    o.arm = (o.id == tid)
            else:
                t.arm = False
            self.track_changed.emit("*")
            return
        self.checkpoint(f"prop:{tid}:{key}")
        setattr(t, key, value)
        self.track_changed.emit(tid)
        self.mark_dirty()
        if key in ("vol_db", "pan", "mute", "solo"):
            self.request_render()

    def set_master_vol(self, v):
        self.checkpoint("master_vol")
        self.project.master["vol_db"] = float(v)
        self.mark_dirty()
        self.track_changed.emit(MASTER)
        self.request_render()

    def set_bpm(self, bpm: float):
        bpm = float(np.clip(bpm, 30.0, 300.0))
        if abs(bpm - self.project.bpm) < 1e-6:
            return
        self.checkpoint("bpm")
        self.project.bpm = bpm
        self.mark_dirty()
        self.track_changed.emit("*")

    def set_track_audio(self, tid, audio: np.ndarray | None, source: str = "", start: int | None = None):
        t = self.project.get(tid)
        if t is None:
            return
        self.checkpoint()
        t.audio = None if audio is None else np.ascontiguousarray(audio, dtype=np.float32)
        t.audio_id = new_audio_id() if audio is not None else ""
        t.source_name = source
        if start is not None:
            t.start = int(start)
        elif audio is None:
            t.start = 0
        self.mark_dirty()
        self.audio_changed.emit(tid)
        self.structure_changed.emit()
        self.request_render(immediate=True)

    # ------------------------------------------------------------ запись/правка
    def record_into_track(self, tid, rec: np.ndarray, start_sample: int, replace_all: bool = False):
        """Помещает новую запись (моно float32) на дорожку: целиком (новый дубль) или поверх участка (punch-in)."""
        t = self.project.get(tid)
        if t is None:
            return
        self.checkpoint()
        clip = np.asarray(rec, dtype=np.float32)[None, :]
        if t.audio is None or replace_all:
            t.audio, t.start = np.ascontiguousarray(clip), int(start_sample)
        else:
            t.audio, t.start = editing.overwrite_into(t.audio, int(start_sample), clip, t.start)
        t.audio_id = new_audio_id()
        t.source_name = time.strftime("запись %H:%M:%S")
        self.mark_dirty()
        self.audio_changed.emit(tid)
        self.structure_changed.emit()
        self.request_render(immediate=True)

    def move_clip(self, tid, new_start: int):
        t = self.project.get(tid)
        if t is None or t.audio is None:
            return
        new_start = max(0, int(new_start))
        if new_start == t.start:
            return
        self.checkpoint()
        t.start = new_start
        t.audio_id = new_audio_id()      # новый ключ кэша — сдвиг меняет и анализ высоты
        self.mark_dirty()
        self.audio_changed.emit(tid)
        self.structure_changed.emit()
        self.request_render(immediate=True)

    def _buf_range(self, t: Track, t0: float | None, t1: float | None):
        """Секунды таймлайна → индексы буфера дорожки. None = вся запись."""
        n = t.length
        if t0 is None or t1 is None:
            return 0, n
        i0 = int(round(t0 * self.sr)) - t.start
        i1 = int(round(t1 * self.sr)) - t.start
        return editing.clamp_range(n, i0, i1)

    def edit_audio(self, tid, op: str, t0: float | None = None, t1: float | None = None, **kw) -> str:
        """Правка записи на дорожке. Возвращает сообщение для пользователя (пустое — ничего не изменилось).
        op: delete (со смыканием), silence, cut, copy, trim, fade_in, fade_out, normalize, gain, reverse,
            trim_silence, paste (kw: at — секунды), clear."""
        t = self.project.get(tid)
        if t is None:
            return "Дорожка не найдена"
        if op == "paste":
            if self.clipboard is None:
                return "Буфер обмена пуст — сначала скопируйте участок (Ctrl+C)"
            at = int(round(float(kw.get("at", 0.0)) * self.sr))
            self.checkpoint()
            t.audio, t.start = editing.mix_into(t.audio, at, self.clipboard, t.start)
            t.audio_id = new_audio_id()
            self._after_edit(tid)
            return f"Вставлено {self.clipboard.shape[1] / self.sr:.2f} с"
        if t.audio is None:
            return "На дорожке нет записи"
        i0, i1 = self._buf_range(t, t0, t1)
        whole = (t0 is None or t1 is None)
        if op in ("delete", "silence", "cut", "copy", "trim", "reverse", "gain") and i1 - i0 < 2:
            return "Сначала выделите участок: проведите мышью по дорожке"
        a = t.audio
        msg = ""
        if op == "copy":
            self.clipboard = editing.copy_range(a, i0, i1)
            return f"Скопировано {(i1 - i0) / self.sr:.2f} с"
        if op == "clear":
            self.set_track_audio(tid, None)
            return "Запись удалена с дорожки"
        self.checkpoint()
        if op == "cut":
            self.clipboard = editing.copy_range(a, i0, i1)
            new = editing.silence_range(a, i0, i1, fade=96)
            msg = f"Вырезано {(i1 - i0) / self.sr:.2f} с (вставьте Ctrl+V; на месте осталась тишина)"
        elif op == "silence":
            new = editing.silence_range(a, i0, i1, fade=96)
            msg = "Участок заглушён (длина не изменилась)"
        elif op == "delete":
            new = editing.delete_range(a, i0, i1)
            msg = f"Участок удалён, края сомкнуты (−{(i1 - i0) / self.sr:.2f} с)"
        elif op == "trim":
            new, off = editing.crop(a, i0, i1)
            t.start += off
            msg = "Запись обрезана до выделения"
        elif op == "fade_in":
            new = editing.fade(a, i0, i1 if not whole else min(a.shape[1], i0 + int(0.15 * self.sr)), "in")
            msg = "Нарастание (fade in) применено"
        elif op == "fade_out":
            lo = i0 if not whole else max(0, i1 - int(0.15 * self.sr))
            new = editing.fade(a, lo, i1, "out")
            msg = "Затухание (fade out) применено"
        elif op == "normalize":
            new, g = editing.normalize(a, i0, i1, float(kw.get("peak_db", -1.0)))
            msg = f"Нормализация: {g:+.1f} дБ"
        elif op == "gain":
            new = editing.gain_db(a, i0, i1, float(kw.get("db", 0.0)))
            msg = f"Усиление {float(kw.get('db', 0.0)):+.1f} дБ"
        elif op == "reverse":
            new = editing.reverse(a, i0, i1)
            msg = "Участок развёрнут"
        elif op == "trim_silence":
            new, off = editing.trim_silence(a, float(kw.get("thr_db", -55.0)), sr=self.sr)
            t.start += off
            msg = f"Тишина по краям убрана ({a.shape[1] - new.shape[1]} отсчётов)"
        else:
            self._undo_stack.pop()
            return f"Неизвестная операция {op}"
        t.audio = np.ascontiguousarray(new, dtype=np.float32)
        t.audio_id = new_audio_id()
        self._after_edit(tid)
        return msg

    def _after_edit(self, tid):
        self.mark_dirty()
        self.audio_changed.emit(tid)
        self.structure_changed.emit()
        self.request_render(immediate=True)

    # =============================================================== плагины ==
    def add_plugin(self, scope, ptype, index=None, overrides=None) -> dict:
        self.checkpoint()
        pl = new_plugin(ptype, overrides)
        lst = self.scope_plugins(scope)
        if index is None:
            lst.append(pl)
        else:
            lst.insert(index, pl)
        self.sel_plugin = pl["id"]
        self.mark_dirty()
        self.structure_changed.emit()
        self.selection_changed.emit()
        self.request_render()
        return pl

    def remove_plugin(self, scope, pid):
        lst = self.scope_plugins(scope)
        i = next((k for k, p in enumerate(lst) if p["id"] == pid), None)
        if i is None:
            return
        self.checkpoint()
        del lst[i]
        if self.sel_plugin == pid:
            self.sel_plugin = lst[min(i, len(lst) - 1)]["id"] if lst else None
        self.mark_dirty()
        self.structure_changed.emit()
        self.selection_changed.emit()
        self.request_render()

    def move_plugin(self, scope, pid, delta):
        lst = self.scope_plugins(scope)
        i = next((k for k, p in enumerate(lst) if p["id"] == pid), None)
        if i is None:
            return
        j = i + delta
        if 0 <= j < len(lst):
            self.checkpoint()
            lst[i], lst[j] = lst[j], lst[i]
            self.mark_dirty()
            self.structure_changed.emit()
            self.request_render()

    def set_plugin_on(self, scope, pid, on):
        p = self.find_plugin(scope, pid)
        if p is not None and p["on"] != bool(on):
            self.checkpoint()
            p["on"] = bool(on)
            self.mark_dirty()
            self.plugin_changed.emit(scope, pid, "on")
            self.structure_changed.emit()
            self.request_render()

    def set_param(self, scope, pid, key, value):
        p = self.find_plugin(scope, pid)
        if p is None or p["params"].get(key) == value:
            return
        self.checkpoint(f"param:{pid}:{key}")
        p["params"][key] = value
        self.mark_dirty()
        self.plugin_changed.emit(scope, pid, key)
        self.request_render()

    def set_scale(self, scope, pid, key_name, scale):
        p = self.find_plugin(scope, pid)
        if p is None:
            return
        self.checkpoint()
        pr = p["params"]
        pr["key"], pr["scale"] = key_name, scale
        if scale != "Custom":
            idx = key_name % 12 if isinstance(key_name, int) else NOTE_NAMES.index(key_name)
            pr["notes"] = scale_mask(idx, scale)
        self.mark_dirty()
        self.plugins_replaced.emit(scope)
        self.request_render()

    def set_notes(self, scope, pid, mask):
        p = self.find_plugin(scope, pid)
        if p is None:
            return
        self.checkpoint()
        pr = p["params"]
        pr["notes"] = list(mask)
        pr["scale"] = "Custom"
        for name in SCALES:
            if scale_mask(NOTE_NAMES.index(pr["key"]), name) == list(mask):
                pr["scale"] = name
                break
        self.mark_dirty()
        self.plugins_replaced.emit(scope)
        self.request_render()

    def apply_plugin_preset(self, scope, pid, preset: dict):
        p = self.find_plugin(scope, pid)
        if p is None:
            return
        self.checkpoint()
        apply_plugin_preset(p, preset)
        self.mark_dirty()
        self.plugins_replaced.emit(scope)
        self.request_render()

    def apply_chain(self, scope, plugins: list):
        self.checkpoint()
        lst = self.scope_plugins(scope)
        lst[:] = plugins
        self.sel_plugin = plugins[0]["id"] if plugins else None
        self.mark_dirty()
        self.structure_changed.emit()
        self.selection_changed.emit()
        self.request_render()

    def apply_chain_preset(self, scope, name):
        self.apply_chain(scope, CHAIN_PRESETS[name]())

    def set_bypass_all(self, on):
        self.bypass_all = bool(on)
        self.request_render()

    # ============================================================== проект ====
    def new_project(self):
        self.project = Project()
        self.mixer.invalidate()
        self.result = None
        self.sel_track = self.sel_plugin = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.history_changed.emit()
        self.project_replaced.emit()
        self.structure_changed.emit()
        self.selection_changed.emit()
        self.request_render(immediate=True)

    def open_project(self, path):
        pr = Project.load(path)
        self.project = pr
        self.mixer.invalidate()
        self.result = None
        self.sel_track = pr.tracks[0].id if pr.tracks else None
        self.sel_plugin = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.history_changed.emit()
        self.project_replaced.emit()
        self.structure_changed.emit()
        self.selection_changed.emit()
        self.request_render(immediate=True)

    def save_project(self, path):
        self.project.name = os.path.splitext(os.path.basename(path))[0]
        self.project.save(path)
        self.dirty_changed.emit(False)

    # =============================================================== рендер ===
    def request_render(self, immediate=False):
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
        snap = self.project.snapshot(self.bypass_all)
        self.render_started.emit()
        threading.Thread(target=self._work, args=(gen, snap), daemon=True).start()

    def _work(self, gen, snap):
        try:
            res = self.mixer.render(snap, cancel=lambda: gen != self._gen,
                                    progress=lambda st, f: self.render_progress.emit(st, f))
            if gen == self._gen:
                self.result = res
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
                try:
                    self._again.emit()
                except RuntimeError:   # объект уже удалён при закрытии
                    pass

    def render_blocking(self, snapshot=None) -> MixResult:
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
            snap = snapshot or self.project.snapshot(self.bypass_all)
            res = self.mixer.render(snap)
            if snapshot is None:
                self.result = res
            return res
        finally:
            with self._lock:
                self._running = False

    # ================================================================ анализ ==
    def other_audio_mono(self, exclude_tid=None):
        """Сумма аудио-дорожек (подложки) в моно — для определения тональности/темпа."""
        acc = None
        for t in self.project.tracks:
            if t.id == exclude_tid or t.kind != "audio" or t.audio is None:
                continue
            m = t.mono()
            acc = m.copy() if acc is None else (np.pad(acc, (0, max(0, len(m) - len(acc)))) +
                                                np.pad(m, (0, max(0, len(acc) - len(m)))))
        return acc

    def run_automix(self, tid, style, opts, apply_levels=True):
        """Запускает анализ + подбор цепочки в фоне. Результат — сигнал automix_done."""
        t = self.project.get(tid)
        if t is None or t.audio is None:
            self.automix_done.emit({"error": "На дорожке нет записи. Сначала запишите вокал или откройте файл."})
            return
        mono = t.mono().astype(np.float64)
        backing = self.other_audio_mono(tid)
        snap = self.project.snapshot(False)
        audio_tracks = [q for q in self.project.tracks if q.kind == "audio" and q.audio is not None and q.id != tid]
        bpm = self.project.bpm
        sr = self.sr

        def work():
            try:
                self.automix_progress.emit("Слушаю вокал: определяю высоту тона…")
                track = detect_pitch(mono, sr, 60, 1200, tracking=0.22)
                self.automix_progress.emit("Анализирую тональность, тембр и динамику…")
                va = analysis.analyze_vocal(mono, sr, track, backing=backing)
                if not va.has_voice:
                    self.automix_done.emit({"error": va.notes[0] if va.notes else "Не удалось проанализировать запись."})
                    return
                plugins, settings, report = analysis.build_vocal_chain(va, style, opts, bpm_default=bpm)
                result = {"analysis": va, "plugins": plugins, "settings": settings, "report": report,
                          "tid": tid, "style": style, "vol_db": None, "backing_vol_db": None,
                          "unmask": (2.5 if (audio_tracks and opts.get("space", True)) else None),
                          "bpm": (round(va.backing_tempo[0]) if va.backing_tempo and va.backing_tempo[1] > 0.1 else None)}
                if result["unmask"]:
                    result["report"].append("Unmasker: на бит добавлен — он будет слегка приглушаться там, где поёт вокал, чтобы голос был разборчивее.")
                if apply_levels:
                    self.automix_progress.emit("Балансирую громкость вокала и подложки…")
                    # громкость вокала после цепочки
                    sn = copy.deepcopy({k: v for k, v in snap.items() if k != "tracks"})
                    one = [dict(q, plugins=plugins, vol_db=0.0, pan=0.0, mute=False, solo=False)
                           for q in snap["tracks"] if q["id"] == tid]
                    sn["tracks"] = one
                    sn["master"]["plugins"] = []
                    pre = self._tmp_mixer.render_track_pre_fader(sn, tid)
                    loud_v = analysis.active_loudness_db(pre, sr) if pre is not None else -20.0
                    if audio_tracks:
                        acc = None
                        for q in audio_tracks:
                            a = q.audio * (10 ** (q.vol_db / 20.0))
                            acc = a.copy() if acc is None else (np.pad(acc, ((0, 0), (0, max(0, a.shape[1] - acc.shape[1])))) +
                                                                np.pad(a, ((0, 0), (0, max(0, acc.shape[1] - a.shape[1])))))
                        loud_b = analysis.active_loudness_db(acc, sr)
                        bk_adj = 0.0
                        if loud_b > -13.0:
                            bk_adj = float(np.clip(-15.0 - loud_b, -12.0, 0.0))
                            loud_b += bk_adj
                            result["backing_vol_db"] = bk_adj
                            result["report"].append(f"Подложка слишком громкая ({loud_b - bk_adj:.0f} дБ) — убавлена на {-bk_adj:.0f} дБ для запаса.")
                        vol = loud_b + settings["vs_backing"] - loud_v
                        result["report"].append(f"Баланс: вокал на {settings['vs_backing']:+.1f} дБ относительно подложки.")
                    else:
                        vol = -18.0 - loud_v
                        result["report"].append("Громкость вокала выставлена на комфортный уровень (−18 дБFS RMS).")
                    result["vol_db"] = float(np.clip(vol, -30.0, 12.0))
                    result["pan"] = 0.0
                self.automix_done.emit(result)
            except Exception as e:  # pragma: no cover
                import traceback
                traceback.print_exc()
                self.automix_done.emit({"error": f"Ошибка анализа: {e}"})

        threading.Thread(target=work, daemon=True).start()

    def apply_automix(self, res: dict):
        """Применяет результат Auto-Mix; возвращает данные для отмены."""
        tid = res["tid"]
        t = self.project.get(tid)
        self.checkpoint()
        undo = {"tid": tid, "plugins": copy.deepcopy(t.plugins), "vol_db": t.vol_db, "pan": t.pan, "bk": {}}
        t.plugins = res["plugins"]
        if res.get("vol_db") is not None:
            t.vol_db = res["vol_db"]
        if res.get("backing_vol_db") is not None:
            for q in self.project.tracks:
                if q.kind == "audio" and q.audio is not None:
                    undo["bk"][q.id] = q.vol_db
                    q.vol_db = q.vol_db + res["backing_vol_db"]
        if res.get("bpm"):
            self.project.bpm = float(res["bpm"])
        if res.get("unmask"):
            for q in self.project.tracks:
                if q.kind == "audio" and q.audio is not None and not any(p["type"] == "unmask" for p in q.plugins):
                    q.plugins.append(new_plugin("unmask", {"reduction": float(res["unmask"])}))
        self.sel_track = tid
        self.sel_plugin = t.plugins[0]["id"] if t.plugins else None
        self.mark_dirty()
        self.structure_changed.emit()
        self.selection_changed.emit()
        self.track_changed.emit("*")
        self.request_render(immediate=True)
        return undo

    def undo_automix(self, undo=None):
        """Отмена Auto-Mix = обычная отмена (шаг истории создан в apply_automix)."""
        self.undo()

    def detect_key_for(self, tid):
        """Определение тональности для кнопки в Auto-Tune: -> dict(key, scale, name, confident, source, ref_a)."""
        t = self.project.get(tid)
        if t is None or t.audio is None:
            return None
        mono = t.mono().astype(np.float64)
        track = detect_pitch(mono, self.sr, 60, 1200, tracking=0.22)
        kv = analysis.detect_key_from_pitch(track)
        backing = self.other_audio_mono(tid)
        kb = analysis.detect_key_from_audio(backing, self.sr) if backing is not None else None
        use = kb if (kb is not None and kb.confident) else kv
        if use is None:
            return None
        return {"key": NOTE_NAMES[use.key], "scale": use.scale, "name": use.name, "confident": use.confident,
                "source": "подложке" if use is kb else "вашему пению", "label": use.confidence_label,
                "vocal": kv.name if kv else None, "backing": kb.name if kb else None,
                "ref_a": round(440.0 * 2 ** (kb.tuning_cents / 1200.0), 1) if kb and kb.confident else None}


# ----------------------------------------------------- пользовательские пресеты

def _presets_dir(sub) -> str:
    d = os.path.join(config_dir(), sub)
    os.makedirs(d, exist_ok=True)
    return d


def _safe(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in " -_()").strip() or "preset"


def list_user_plugin_presets(ptype) -> dict:
    path = os.path.join(_presets_dir("plugin_presets"), ptype + ".json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_user_plugin_preset(ptype, name, params: dict):
    allp = list_user_plugin_presets(ptype)
    allp[name] = {k: v for k, v in params.items() if k not in ("overrides",)}
    with open(os.path.join(_presets_dir("plugin_presets"), ptype + ".json"), "w", encoding="utf-8") as f:
        json.dump(allp, f, ensure_ascii=False, indent=1)


def delete_user_plugin_preset(ptype, name):
    allp = list_user_plugin_presets(ptype)
    if name in allp:
        del allp[name]
        with open(os.path.join(_presets_dir("plugin_presets"), ptype + ".json"), "w", encoding="utf-8") as f:
            json.dump(allp, f, ensure_ascii=False, indent=1)


def list_user_chains() -> list[str]:
    try:
        return sorted(f[:-5] for f in os.listdir(_presets_dir("chains")) if f.endswith(".json"))
    except OSError:
        return []


def save_user_chain(name, plugins: list):
    with open(os.path.join(_presets_dir("chains"), _safe(name) + ".json"), "w", encoding="utf-8") as f:
        json.dump(plugins, f, ensure_ascii=False, indent=1)


def load_user_chain(name) -> list:
    with open(os.path.join(_presets_dir("chains"), name + ".json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for p in data:
        if p.get("type") in PLUGINS:
            q = new_plugin(p["type"], p.get("params"), p.get("on", True))
            out.append(q)
    return out


def delete_user_chain(name):
    try:
        os.remove(os.path.join(_presets_dir("chains"), name + ".json"))
    except OSError:
        pass


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
