"""Пространственные эффекты: дилей (с фильтрами в петле, ping-pong) и реверберация
(свёрточная, с синтезируемой импульсной характеристикой)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, oaconvolve, sosfilt, sosfilt_zi

DIVISIONS = {
    "1/4": 1.0,
    "1/8": 0.5,
    "1/8 dotted": 0.75,
    "1/8 triplet": 1.0 / 3.0,
    "1/16": 0.25,
    "1/2": 2.0,
}


def delay_samples(p: dict, fs: int) -> int:
    if p.get("mode", "Sync") == "Sync":
        beat = 60.0 / max(20.0, float(p.get("bpm", 120.0)))
        sec = beat * DIVISIONS.get(p.get("division", "1/8"), 0.5)
    else:
        sec = float(p.get("time_ms", 350.0)) / 1000.0
    return max(32, int(sec * fs))


def delay_tail_samples(p: dict, fs: int) -> int:
    D = delay_samples(p, fs)
    fb = min(0.95, float(p.get("feedback", 30.0)) / 100.0)
    if fb < 0.02:
        return D
    reps = int(np.ceil(np.log(0.001) / np.log(fb)))
    return int(min(reps * D + D, 10 * fs))


def delay_process(x: np.ndarray, fs: int, p: dict) -> np.ndarray:
    """x: mono (n,) с запасом нулей в конце. Возвращает wet стерео (2, n) (только эхо)."""
    n = len(x)
    D = delay_samples(p, fs)
    fb = min(0.95, float(p.get("feedback", 30.0)) / 100.0)
    pp = float(p.get("pingpong", 50.0)) / 100.0
    lc = float(p.get("lowcut", 200.0))
    hc = float(p.get("highcut", 6000.0))
    sos = np.vstack([
        butter(1, max(20.0, lc) / (0.5 * fs), btype="high", output="sos"),
        butter(1, min(hc, 0.45 * fs) / (0.5 * fs), btype="low", output="sos"),
    ])
    xin = sosfilt(sos, x)
    nch = 2
    w = np.zeros((nch, n))
    zi = [np.zeros((sos.shape[0], 2)) for _ in range(nch)]
    n_chunks = int(np.ceil(n / D))
    in_gain = np.array([1.0, 1.0 - pp])
    for c in range(1, n_chunks):
        a, b = c * D, min(n, (c + 1) * D)
        m = b - a
        pa = (c - 1) * D
        prev = w[:, pa:pa + m]
        mixed = np.empty_like(prev)
        mixed[0] = (1 - pp) * prev[0] + pp * prev[1]
        mixed[1] = (1 - pp) * prev[1] + pp * prev[0]
        for ch in range(nch):
            f, zi[ch] = sosfilt(sos, mixed[ch], zi=zi[ch])
            w[ch, a:b] = in_gain[ch] * xin[pa:pa + m] + fb * f
    return w


# ----------------------------------------------------------------- reverb --

REVERB_TYPES = {
    #            er_gain  er_span_ms  onset_ms  bright  size_mul
    "Room":      (0.9,    40.0,       3.0,      0.9,    0.5),
    "Plate":     (0.0,    0.0,        1.0,      1.15,   0.7),
    "Hall":      (0.6,    90.0,       18.0,     0.85,   1.2),
    "Chamber":   (0.7,    60.0,       8.0,      0.95,   0.9),
    "Cathedral": (0.5,    140.0,      35.0,     0.7,    1.8),
}


def make_ir(fs: int, p: dict) -> np.ndarray:
    """Синтез стерео-импульсной характеристики (2, n) единичной энергии."""
    rt60 = float(np.clip(p.get("decay", 1.8), 0.1, 12.0))
    pre = float(p.get("predelay", 20.0)) * 0.001
    damp = float(np.clip(p.get("damping", 50.0) / 100.0, 0.0, 1.0))
    size = float(np.clip(p.get("size", 50.0) / 100.0, 0.0, 1.0))
    width = float(np.clip(p.get("width", 100.0) / 100.0, 0.0, 1.5))
    lc = float(p.get("lowcut", 150.0))
    hc = float(p.get("highcut", 9000.0))
    er_gain, er_span, onset, bright, size_mul = REVERB_TYPES.get(p.get("type", "Plate"), REVERB_TYPES["Plate"])

    n = int((rt60 * 1.05) * fs) + 1
    t = np.arange(n) / fs
    rng = np.random.default_rng(1234)

    # три частотные полосы с разным временем затухания
    x_lo = butter(2, 300.0 / (0.5 * fs), btype="low", output="sos")
    x_mid = butter(2, [300.0 / (0.5 * fs), 3000.0 / (0.5 * fs)], btype="band", output="sos")
    x_hi = butter(2, 3000.0 / (0.5 * fs), btype="high", output="sos")
    rt_lo = rt60 * (1.0 + 0.25 * (1.0 - damp))
    rt_mid = rt60
    rt_hi = max(0.06, rt60 * (1.0 - 0.88 * damp) * bright)
    e_lo = 10.0 ** (-3.0 * t / rt_lo)
    e_mid = 10.0 ** (-3.0 * t / rt_mid)
    e_hi = 10.0 ** (-3.0 * t / rt_hi)

    on_tau = max(0.0005, (onset + 12.0 * size * size_mul) * 0.001 * 0.35)
    onset_env = 1.0 - np.exp(-t / on_tau)

    ir = np.zeros((2, n))
    for ch in range(2):
        noise = rng.standard_normal(n)
        ir[ch] = (sosfilt(x_lo, noise) * e_lo + sosfilt(x_mid, noise) * e_mid + sosfilt(x_hi, noise) * e_hi) * onset_env

    # ранние отражения
    if er_gain > 0:
        n_er = int(10 + 14 * size)
        span = er_span * (0.5 + size) * 0.001
        for ch in range(2):
            ts = np.sort(rng.uniform(0.002, max(0.004, span), n_er))
            amps = er_gain * (0.55 * np.exp(-ts / max(0.01, span * 0.6))) * rng.choice([-1.0, 1.0], n_er)
            for tt, a in zip(ts, amps):
                i = int(tt * fs)
                if i < n:
                    ir[ch, i] += a * 3.0

    # ширина стерео
    mid = 0.5 * (ir[0] + ir[1])
    side = 0.5 * (ir[0] - ir[1])
    ir[0] = mid + width * side
    ir[1] = mid - width * side

    # фильтры реверберации
    if lc > 25:
        ir = sosfilt(butter(2, lc / (0.5 * fs), btype="high", output="sos"), ir, axis=-1)
    if hc < 0.45 * fs:
        ir = sosfilt(butter(2, hc / (0.5 * fs), btype="low", output="sos"), ir, axis=-1)

    ir = np.concatenate([np.zeros((2, int(pre * fs))), ir], axis=1)
    energy = np.sqrt((ir ** 2).sum() / 2.0)
    return ir / max(energy, 1e-9)


def reverb_tail_samples(p: dict, fs: int) -> int:
    return int((float(p.get("decay", 1.8)) * 1.05 + float(p.get("predelay", 20.0)) * 0.001) * fs)


_ir_cache: dict = {}


def reverb_process(x: np.ndarray, fs: int, p: dict) -> np.ndarray:
    """x: mono (n,) -> wet (2, n)."""
    key = (fs, p.get("type"), round(float(p.get("decay", 1.8)), 3), round(float(p.get("predelay", 20.0)), 2),
           round(float(p.get("damping", 50)), 1), round(float(p.get("size", 50)), 1),
           round(float(p.get("width", 100)), 1), round(float(p.get("lowcut", 150)), 1),
           round(float(p.get("highcut", 9000)), 1))
    ir = _ir_cache.get(key)
    if ir is None:
        if len(_ir_cache) > 8:
            _ir_cache.clear()
        ir = make_ir(fs, p)
        _ir_cache[key] = ir
    n = len(x)
    out = np.empty((2, n))
    for ch in range(2):
        out[ch] = oaconvolve(x, ir[ch])[:n]
    return out
