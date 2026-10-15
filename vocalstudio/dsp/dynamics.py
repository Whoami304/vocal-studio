"""Динамическая обработка: гейт, компрессор, де-эссер, насыщение, лимитер."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import resample_poly, sosfilt

from .eq import bandpass_sos

CTRL_RATE = 2756.0  # частота управляющего сигнала, Гц


def _block(fs: int) -> int:
    return max(1, int(round(fs / CTRL_RATE)))


def _detector(x: np.ndarray, mode: str) -> np.ndarray:
    """Связанный (linked) детектор для моно/стерео: (ch, n) -> (n,)."""
    if x.ndim == 1:
        return x
    if mode == "rms":
        return np.sqrt(np.mean(x * x, axis=0))
    return np.max(np.abs(x), axis=0)


def _blocks_peak(x: np.ndarray, B: int):
    n = len(x)
    nb = int(np.ceil(n / B))
    pad = nb * B - n
    xa = np.abs(np.pad(x, (0, pad)))
    return xa.reshape(nb, B).max(axis=1), xa.reshape(nb, B)


def _level_db(x: np.ndarray, fs: int, mode: str = "peak"):
    B = _block(fs)
    pk, blocks = _blocks_peak(x, B)
    if mode == "rms":
        ms = (blocks ** 2).mean(axis=1)
        ms = uniform_filter1d(ms, size=max(1, int(round(0.012 * fs / B))), mode="nearest")
        lvl = np.sqrt(np.maximum(ms, 0.0))
    else:
        lvl = pk
    return 20.0 * np.log10(lvl + 1e-9), B


def _smooth(target: np.ndarray, c_down: float, c_up: float, init: float = 0.0) -> np.ndarray:
    """Односторонняя экспоненциальная релаксация с разными коэффициентами."""
    out = [0.0] * len(target)
    prev = init
    for i, t in enumerate(target.tolist()):
        c = c_down if t < prev else c_up
        prev = c * prev + (1.0 - c) * t
        out[i] = prev
    return np.asarray(out)


def _coef(ms: float, fs_ctrl: float) -> float:
    if ms <= 0.01:
        return 0.0
    return float(np.exp(-1.0 / (ms * 0.001 * fs_ctrl)))


def _to_audio_rate(g_db: np.ndarray, B: int, n: int) -> np.ndarray:
    centers = (np.arange(len(g_db)) + 0.5) * B
    return np.interp(np.arange(n), centers, g_db)


def compressor(x: np.ndarray, fs: int, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (выход, кривая GR в дБ на аудио-частоте для индикаторов)."""
    thr = float(p.get("threshold", -18.0))
    ratio = max(1.0, float(p.get("ratio", 3.0)))
    knee = max(0.0, float(p.get("knee", 6.0)))
    att = float(p.get("attack", 8.0))
    rel = float(p.get("release", 120.0))
    makeup = float(p.get("makeup", 0.0))
    if p.get("auto_makeup", False):
        makeup += -thr * (1.0 - 1.0 / ratio) * 0.5
    mix = float(p.get("mix", 100.0)) / 100.0
    mode = "rms" if p.get("detect", "RMS") == "RMS" else "peak"

    L, B = _level_db(_detector(x, mode), fs, mode)
    over = L - thr
    slope = 1.0 / ratio - 1.0
    gr = np.zeros_like(over)
    if knee > 0:
        in_knee = np.abs(over) <= knee / 2.0
        above = over > knee / 2.0
        gr[above] = slope * over[above]
        gr[in_knee] = slope * (over[in_knee] + knee / 2.0) ** 2 / (2.0 * knee)
    else:
        m = over > 0
        gr[m] = slope * over[m]

    fsc = fs / B
    sm = _smooth(gr, _coef(att, fsc), _coef(rel, fsc))
    g_db = _to_audio_rate(sm, B, x.shape[-1])
    g = 10.0 ** ((g_db + makeup) / 20.0)
    y = x * g
    if mix < 0.999:
        y = x * (1.0 - mix) + y * mix
    return y, g_db


def gate(x: np.ndarray, fs: int, p: dict) -> np.ndarray:
    thr = float(p.get("threshold", -55.0))
    rng = float(p.get("range", 30.0))
    att = float(p.get("attack", 2.0))
    rel = float(p.get("release", 120.0))
    L, B = _level_db(_detector(x, "peak"), fs, "peak")
    # удержание: открыт, пока в пределах 40 мс был сигнал
    hold = max(1, int(0.04 * fs / B))
    Lh = maximum_filter1d(L, size=hold, mode="nearest")
    gr = -np.minimum(rng, np.maximum(0.0, thr - Lh) * 3.0)
    fsc = fs / B
    # закрытие (gain вниз) — release, открытие — attack
    sm = _smooth(gr, _coef(rel, fsc), _coef(att, fsc))
    return x * 10.0 ** (_to_audio_rate(sm, B, x.shape[-1]) / 20.0)


def deesser(x: np.ndarray, fs: int, p: dict) -> tuple[np.ndarray, np.ndarray]:
    f = float(p.get("freq", 6500.0))
    thr = float(p.get("threshold", -30.0))
    rng = float(p.get("range", 8.0))
    sos = bandpass_sos(fs, f, 1.6)
    band = sosfilt(sos, x, axis=-1)
    L, B = _level_db(_detector(band, "peak"), fs, "peak")
    gr = -np.clip((L - thr) * 0.8, 0.0, rng)
    fsc = fs / B
    sm = _smooth(gr, _coef(0.4, fsc), _coef(25.0, fsc))
    g_db = _to_audio_rate(sm, B, x.shape[-1])
    g = 10.0 ** (g_db / 20.0)
    if p.get("wide", False):
        return x * g, g_db
    # вычитаем из полосы нужную долю: полоса ослабляется на g
    return x - (1.0 - g) * band, g_db


def saturate(x: np.ndarray, fs: int, p: dict) -> np.ndarray:
    drive = 10.0 ** (float(p.get("drive", 6.0)) / 20.0)
    mix = float(p.get("mix", 30.0)) / 100.0
    n = x.shape[-1]
    xo = resample_poly(x, 2, 1, axis=-1)
    # лёгкая асимметрия -> чётные гармоники ("ламповый" оттенок)
    bias = 0.08 * float(p.get("warmth", 1.0))
    y = np.tanh(drive * xo + bias) - np.tanh(bias)
    y = y / drive
    y = resample_poly(y, 1, 2, axis=-1)[..., :n]
    if y.shape[-1] < n:
        y = np.pad(y, [(0, 0)] * (y.ndim - 1) + [(0, n - y.shape[-1])])
    return x * (1.0 - mix) + y * mix


def limiter(x: np.ndarray, fs: int, ceiling_db: float = -1.0, lookahead_ms: float = 5.0,
            release_ms: float = 60.0) -> np.ndarray:
    """Лукахед-лимитер (оффлайн, без задержки). x: (ch, n) или (n,)."""
    ceil = 10.0 ** (ceiling_db / 20.0)
    mono = x.ndim == 1
    xx = x[None, :] if mono else x
    peak = np.abs(xx).max(axis=0)
    need = np.minimum(1.0, ceil / np.maximum(peak, 1e-9))
    la = max(2, int(lookahead_ms * 0.001 * fs))
    g = minimum_filter1d(need, size=2 * la + 1, mode="nearest")
    g = uniform_filter1d(g, size=2 * la + 1, mode="nearest")
    # плавный release: медленно возвращаем усиление вверх
    rc = _coef(release_ms, fs / 8.0)
    gd = g[::8]
    sm = _smooth(np.log(np.maximum(gd, 1e-6)), 0.0, rc)  # вниз — мгновенно, вверх — release
    sm = np.minimum(sm, np.log(np.maximum(gd, 1e-6)))
    g2 = np.exp(np.interp(np.arange(len(g)), (np.arange(len(sm))) * 8.0, sm))
    g2 = np.minimum(g2, g)
    y = xx * g2[None, :]
    # гарантия потолка
    y = np.clip(y, -ceil, ceil)
    return y[0] if mono else y
