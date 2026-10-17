"""Многодорожечный микшер с кэшированием по стадиям.

Для каждой дорожки: аудио → цепочка плагинов → громкость/панорама → сумма → цепочка мастера → мастер-громкость.
При изменении параметра пересчитываются только стадии, начиная с изменённой.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .dsp.pitch import INPUT_RANGES, PitchTrack, detect_pitch
from .plugins import PLUGINS, Env, plugin_tail


def _k(*parts) -> str:
    return json.dumps(parts, sort_keys=True, default=str)


@dataclass
class TrackResult:
    out: np.ndarray                                  # (2, N) float32, после фейдера
    peak: float = 0.0
    tune: dict = field(default_factory=dict)         # plugin_id -> (PitchTrack, Correction)
    gr: dict = field(default_factory=dict)           # plugin_id -> массив GR (дБ)
    audible: bool = True


@dataclass
class MixResult:
    out: np.ndarray                                  # (2, N) float32
    sr: int
    tracks: dict = field(default_factory=dict)       # track_id -> TrackResult
    master_gr: dict = field(default_factory=dict)


def _chain_tail(plugins, sr) -> int:
    """Хвост цепочки: плагины идут последовательно, хвосты суммируются (с ограничением 20 с)."""
    return int(min(sum(plugin_tail(p, sr) for p in plugins), 20 * sr))


class Mixer:
    def __init__(self):
        self._cache: dict = {}        # (scope, plugin_id) -> (key, array float32, results)
        self._pitch: dict = {}        # (scope, plugin_id) -> (key, PitchTrack)

    def invalidate(self):
        self._cache.clear()
        self._pitch.clear()

    # ------------------------------------------------------------------
    def _run_chain(self, scope, x, key, plugins, sr, bypass, cancel, progress, results, sidechain=None, sc_key=""):
        """Прогон цепочки плагинов с кэшем. Возвращает (x, key)."""
        for pl in plugins:
            if cancel is not None and cancel():
                raise InterruptedError
            if not pl["on"] or bypass:
                continue
            pid = pl["id"]
            key = _k(key, pl["type"], pl["params"])
            if pl["type"] == "unmask":
                key = _k(key, sc_key)
            hit = self._cache.get((scope, pid))
            if hit is not None and hit[0] == key:
                x = hit[1].astype(np.float64)
                results[pid] = hit[2]
                continue
            env = Env(sr, cancel=cancel, progress=progress, sidechain=sidechain)
            prev_key = key

            def get_pitch(mono, p, scope=scope, pid=pid, prev=prev_key, env=env):
                pk = _k(prev, p.get("input_type"), p.get("tracking"))
                ent = self._pitch.get((scope, pid))
                if ent is not None and ent[0] == pk:
                    return ent[1]
                fmin, fmax = INPUT_RANGES.get(p.get("input_type", "Auto"), INPUT_RANGES["Auto"])
                thr = 0.10 + 0.30 * float(p.get("tracking", 40.0)) / 100.0
                tr = detect_pitch(mono, sr, fmin=fmin, fmax=fmax, tracking=thr,
                                  progress=(lambda f: progress("pitch", f)) if progress else None, cancel=cancel)
                self._pitch[(scope, pid)] = (pk, tr)
                return tr

            env.get_pitch = get_pitch
            if progress:
                progress(pl["type"], 0.0)
            x = PLUGINS[pl["type"]].process(x, sr, pl["params"], env)
            self._cache[(scope, pid)] = (key, x.astype(np.float32), env.results)
            results[pid] = env.results
        return x, key

    # ------------------------------------------------------------------
    def render(self, snap: dict, cancel=None, progress=None) -> MixResult:
        sr = snap["sr"]
        bypass = snap["bypass_all"]
        tracks = snap["tracks"]
        master = snap["master"]
        n_audio = max([0 if t["audio"] is None else t["start"] + t["audio"].shape[1] for t in tracks] + [0])
        if n_audio == 0:
            return MixResult(out=np.zeros((2, 0), np.float32), sr=sr)
        tail = 0
        if not bypass:
            for t in tracks:
                tail = max(tail, _chain_tail(t["plugins"], sr))
            tail += _chain_tail(master["plugins"], sr)
        N = n_audio + tail

        any_solo = any(t["solo"] for t in tracks)
        live_ids = set()
        mix = np.zeros((2, N))
        mix_key_parts = []
        result = MixResult(out=np.zeros((2, 0), np.float32), sr=sr)

        sc_sum = np.zeros(N)           # голос для sidechain (Unmasker на дорожках с музыкой)
        sc_parts = []
        ordered = [t for t in tracks if t["kind"] != "audio"] + [t for t in tracks if t["kind"] == "audio"]
        for t in ordered:
            if cancel is not None and cancel():
                raise InterruptedError
            live_ids.update(p["id"] for p in t["plugins"])
            if t["audio"] is None:
                continue
            a = t["audio"]
            x = np.zeros((a.shape[0], N))
            x[:, t["start"]: t["start"] + a.shape[1]] = a
            key = _k("t", t["audio_id"], t["start"], N, sr)
            res: dict = {}
            x, key = self._run_chain(("t", t["id"]), x, key, t["plugins"], sr, bypass, cancel, progress, res,
                                     sidechain=sc_sum if t["kind"] == "audio" else None, sc_key=_k(sc_parts))
            audible = (not t["mute"]) and ((not any_solo) or t["solo"])
            g = 10.0 ** (t["vol_db"] / 20.0)
            pan = float(t["pan"])
            if x.shape[0] == 1:
                th = (pan + 1.0) * np.pi / 4.0
                gl, gr_ = np.cos(th) * np.sqrt(2.0), np.sin(th) * np.sqrt(2.0)
                y = np.vstack([x[0] * gl, x[0] * gr_]) * g
            else:
                gl = 1.0 if pan <= 0 else 1.0 - pan
                gr_ = 1.0 if pan >= 0 else 1.0 + pan
                y = np.vstack([x[0] * gl, x[1] * gr_]) * g
            tr = TrackResult(out=y.astype(np.float32), peak=float(np.abs(y).max()) if y.size else 0.0, audible=audible)
            for pid, r in res.items():
                if "tune" in r:
                    tr.tune[pid] = r["tune"]
                if "gr" in r:
                    tr.gr[pid] = r["gr"]
            result.tracks[t["id"]] = tr
            if audible:
                mix += y
                mix_key_parts.append((t["id"], key, t["vol_db"], t["pan"]))
                if t["kind"] != "audio":
                    sc_sum += y.mean(axis=0)
                    sc_parts.append((t["id"], key, t["vol_db"], t["pan"]))

        live_ids.update(p["id"] for p in master["plugins"])
        mkey = _k("m", N, mix_key_parts)
        res: dict = {}
        mix, mkey = self._run_chain(("m", "master"), mix, mkey, master["plugins"], sr, bypass, cancel, progress, res)
        for pid, r in res.items():
            if "gr" in r:
                result.master_gr[pid] = r["gr"]
        mix = mix * (10.0 ** (master["vol_db"] / 20.0))
        result.out = mix.astype(np.float32)

        # чистим кэш удалённых плагинов
        for k in [k for k in self._cache if k[1] not in live_ids]:
            del self._cache[k]
        for k in [k for k in self._pitch if k[1] not in live_ids]:
            del self._pitch[k]
        return result

    # ------------------------------------------------------------------
    def render_track_pre_fader(self, snap: dict, tid: str, cancel=None, progress=None) -> np.ndarray | None:
        """Результат цепочки дорожки до фейдера (для анализа громкости). Использует тот же кэш."""
        sr = snap["sr"]
        t = next((q for q in snap["tracks"] if q["id"] == tid), None)
        if t is None or t["audio"] is None:
            return None
        tail = 0 if snap["bypass_all"] else _chain_tail(t["plugins"], sr)
        n_audio = max([0 if q["audio"] is None else q["start"] + q["audio"].shape[1] for q in snap["tracks"]] + [0])
        N = n_audio + tail
        a = t["audio"]
        x = np.zeros((a.shape[0], N))
        x[:, t["start"]: t["start"] + a.shape[1]] = a
        key = _k("t", t["audio_id"], t["start"], N, sr)
        x, _ = self._run_chain(("t", t["id"]), x, key, t["plugins"], sr, snap["bypass_all"], cancel, progress, {})
        return x
