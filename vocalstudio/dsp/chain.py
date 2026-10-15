"""Цепочка обработки вокала с кэшированием по стадиям.

Порядок: Input → Auto-Tune → Gate → [EQ] → Compressor → De-esser → [EQ] →
Saturation → (Delay ‖ Reverb параллельно) → Output (Backing + Limiter).
При изменении параметра пересчитываются только стадии начиная с изменённой.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from . import dynamics, eq as eqm, timefx
from .autotune import autotune
from .pitch import PitchTrack, detect_pitch, INPUT_RANGES


@dataclass
class RenderResult:
    out: np.ndarray                     # (2, N) float32
    vocal_dry: np.ndarray               # исходный моно-вокал (после input)
    track: PitchTrack | None = None
    correction: object = None
    comp_gr: np.ndarray | None = None   # GR компрессора (дБ, замедленный)
    deess_gr: np.ndarray | None = None
    sr: int = 44100
    timings: dict = field(default_factory=dict)


def _k(*parts) -> str:
    return json.dumps(parts, sort_keys=True, default=str)


class Chain:
    def __init__(self):
        self._stage_cache: dict[str, tuple[str, object]] = {}
        self._pitch_key = None
        self._pitch = None

    def invalidate(self):
        self._stage_cache.clear()
        self._pitch_key = None
        self._pitch = None

    # ------------------------------------------------------------------
    def get_pitch(self, vocal, sr, take_id, tune_p, cancel=None, progress=None) -> PitchTrack:
        key = _k(take_id, tune_p.get("input_type"), tune_p.get("tracking"))
        if self._pitch_key == key and self._pitch is not None:
            return self._pitch
        fmin, fmax = INPUT_RANGES.get(tune_p.get("input_type", "Auto"), INPUT_RANGES["Auto"])
        thr = 0.10 + 0.30 * float(tune_p.get("tracking", 40.0)) / 100.0
        tr = detect_pitch(vocal, sr, fmin=fmin, fmax=fmax, tracking=thr, progress=progress, cancel=cancel)
        self._pitch_key, self._pitch = key, tr
        return tr

    # ------------------------------------------------------------------
    def render(self, vocal: np.ndarray | None, sr: int, params: dict, backing: np.ndarray | None = None,
               take_id: str = "", overrides=None, bypass_all: bool = False, cancel=None, progress=None) -> RenderResult:
        def chk():
            if cancel is not None and cancel():
                raise InterruptedError

        def prog(stage, f=0.0):
            if progress:
                progress(stage, f)

        P = params
        tail = 0
        if vocal is not None and not bypass_all:
            if P["delay"]["on"] and P["delay"]["mix"] > 0:
                tail = max(tail, timefx.delay_tail_samples(P["delay"], sr))
            if P["reverb"]["on"] and P["reverb"]["mix"] > 0:
                tail = max(tail, timefx.reverb_tail_samples(P["reverb"], sr))
        n_voc = 0 if vocal is None else len(vocal)
        N = n_voc + tail

        vg = 10.0 ** (P["master"]["vocal_db"] / 20.0)
        bg = 10.0 ** (P["master"]["backing_db"] / 20.0)

        if vocal is None or n_voc == 0:
            out = np.zeros((2, 0)) if backing is None else backing * bg
            return RenderResult(out=out.astype(np.float32), vocal_dry=np.zeros(0), sr=sr)

        res = RenderResult(out=np.zeros((2, 0), np.float32), vocal_dry=vocal, sr=sr)

        if bypass_all:
            mono = vocal * vg
            stereo = np.vstack([mono, mono])
            return self._finish(stereo, backing, bg, P, sr, res, apply_vocal_gain=False)

        # ---- стадии --------------------------------------------------
        cache = self._stage_cache
        prev_key = _k("take", take_id, n_voc, sr)
        x = vocal

        def stage(name, params_for_key, fn, enabled=True):
            nonlocal prev_key, x
            if not enabled:
                return
            key = _k(prev_key, name, params_for_key)
            hit = cache.get(name)
            if hit is not None and hit[0] == key:
                x = hit[1].astype(np.float64)
            else:
                chk()
                prog(name, 0.0)
                x = fn(x)
                cache[name] = (key, x.astype(np.float32))
            prev_key = key

        # input
        g_in = 10.0 ** (P["input"]["gain"] / 20.0)
        stage("input", P["input"], lambda v: v * g_in, abs(P["input"]["gain"]) > 0.01)
        res.vocal_dry = x

        # tune
        if P["tune"]["on"]:
            tp = P["tune"]
            res.track = self.get_pitch(x, sr, _k(prev_key), tp, cancel=cancel,
                                       progress=lambda f: prog("pitch", f))
            holder = {}

            def run_tune(v):
                y, corr = autotune(v, res.track, tp, overrides=overrides, cancel=cancel,
                                   progress=lambda f: prog("tune", f))
                holder["corr"] = corr
                return y

            stage("tune", [tp, overrides], run_tune)
            hit = cache.get("tune")
            if "corr" in holder:
                self._corr = holder["corr"]
            res.correction = getattr(self, "_corr", None)
            if res.correction is None:
                from .autotune import compute_correction
                res.correction = compute_correction(res.track, tp, overrides)

        stage("gate", P["gate"], lambda v: dynamics.gate(v, sr, P["gate"]), P["gate"]["on"])

        def do_eq():
            stage("eq", P["eq"], lambda v: eqm.eq_process(v, P["eq"], sr), P["eq"]["on"])

        eq_pre = P["eq"]["pos"] == "Pre-Comp"
        if eq_pre:
            do_eq()

        comp_holder = {}

        def run_comp(v):
            y, gr = dynamics.compressor(v, sr, P["comp"])
            comp_holder["gr"] = gr
            return y

        stage("comp", P["comp"], run_comp, P["comp"]["on"])
        if "gr" in comp_holder:
            res.comp_gr = comp_holder["gr"][::64]

        de_holder = {}

        def run_de(v):
            y, gr = dynamics.deesser(v, sr, P["deess"])
            de_holder["gr"] = gr
            return y

        stage("deess", P["deess"], run_de, P["deess"]["on"])
        if "gr" in de_holder:
            res.deess_gr = de_holder["gr"][::64]

        if not eq_pre:
            do_eq()

        stage("sat", P["sat"], lambda v: dynamics.saturate(v, sr, P["sat"]), P["sat"]["on"])

        # дальше работаем с хвостом
        if tail > 0:
            xt = np.concatenate([x, np.zeros(tail)])
        else:
            xt = x
        chk()
        stereo = np.vstack([xt, xt])
        sends_key = _k(prev_key, N)

        if P["delay"]["on"] and P["delay"]["mix"] > 0:
            key = _k(sends_key, "delay", P["delay"])
            hit = cache.get("delay")
            if hit is not None and hit[0] == key:
                wet = hit[1].astype(np.float64)
            else:
                prog("delay", 0.0)
                wet = timefx.delay_process(xt, sr, P["delay"])
                cache["delay"] = (key, wet.astype(np.float32))
            stereo = stereo + wet * (P["delay"]["mix"] / 100.0)
        chk()
        if P["reverb"]["on"] and P["reverb"]["mix"] > 0:
            key = _k(sends_key, "reverb", P["reverb"])
            hit = cache.get("reverb")
            if hit is not None and hit[0] == key:
                wet = hit[1].astype(np.float64)
            else:
                prog("reverb", 0.0)
                wet = timefx.reverb_process(xt, sr, P["reverb"])
                cache["reverb"] = (key, wet.astype(np.float32))
            stereo = stereo + wet * (P["reverb"]["mix"] / 100.0)

        stereo = stereo * vg
        return self._finish(stereo, backing, bg, P, sr, res, apply_vocal_gain=False)

    # ------------------------------------------------------------------
    def _finish(self, stereo, backing, bg, P, sr, res, apply_vocal_gain=False):
        if backing is not None and backing.shape[1] > 0:
            L = max(stereo.shape[1], backing.shape[1])
            mix = np.zeros((2, L))
            mix[:, : stereo.shape[1]] += stereo
            mix[:, : backing.shape[1]] += backing * bg
        else:
            mix = stereo
        if P["master"]["limiter_on"]:
            mix = dynamics.limiter(mix, sr, P["master"]["ceiling"])
        res.out = mix.astype(np.float32)
        return res
