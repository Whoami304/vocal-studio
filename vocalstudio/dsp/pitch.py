"""Детектор основного тона (F0).

Алгоритм: YIN (кумулятивная нормализованная разностная функция) в пакетном
FFT-варианте + параболическая интерполяция + Витерби по нескольким
кандидатам на кадр (идея pYIN). Витерби убирает октавные ошибки и
"дребезг" вокруг границ озвончённых участков.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly
from scipy.fft import rfft, irfft, next_fast_len


@dataclass
class PitchTrack:
    sr: int                 # частота дискретизации исходного сигнала
    hop: int                # шаг кадров в отсчётах исходного сигнала
    f0: np.ndarray          # Гц (0 там, где нет тона)
    voiced: np.ndarray      # bool
    conf: np.ndarray        # 0..1 уверенность
    rms_db: np.ndarray      # уровень кадра, дБFS

    @property
    def times(self) -> np.ndarray:
        """Время центров кадров, в отсчётах исходного сигнала."""
        return np.arange(len(self.f0)) * self.hop

    def midi(self, ref_a: float = 440.0) -> np.ndarray:
        m = np.full(len(self.f0), np.nan)
        v = self.f0 > 0
        m[v] = 69.0 + 12.0 * np.log2(self.f0[v] / ref_a)
        return m


# Диапазоны поиска F0 для типов голоса (Гц)
INPUT_RANGES = {
    "Auto": (60.0, 1200.0),
    "Soprano": (170.0, 1250.0),
    "Alto/Tenor": (85.0, 700.0),
    "Low Male": (55.0, 400.0),
    "Instrument": (50.0, 2000.0),
}


def _frames_diff(frames: np.ndarray, W: int, tau_max: int) -> np.ndarray:
    """Нормализованная разностная функция YIN для пачки кадров (F, W+tau_max)."""
    F = frames.shape[0]
    nfft = next_fast_len(W + tau_max)
    a = frames[:, :W]
    A = rfft(a, nfft, axis=1)
    B = rfft(frames, nfft, axis=1)
    r = irfft(np.conj(A) * B, nfft, axis=1)[:, : tau_max + 1]
    sq = frames.astype(np.float64) ** 2
    cs = np.concatenate([np.zeros((F, 1)), np.cumsum(sq, axis=1)], axis=1)
    e1 = cs[:, W][:, None]
    taus = np.arange(tau_max + 1)
    e2 = cs[:, W + taus] - cs[:, taus]
    d = e1 + e2 - 2.0 * r
    d[:, 0] = 0.0
    d = np.maximum(d, 0.0)
    csum = np.cumsum(d, axis=1)
    dn = np.ones_like(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        dn[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(csum[:, 1:], 1e-12)
    return dn


def detect_pitch(
    x: np.ndarray,
    sr: int,
    fmin: float = 60.0,
    fmax: float = 1200.0,
    tracking: float = 0.22,
    gate_db: float = -52.0,
    progress=None,
    cancel=None,
) -> PitchTrack:
    """Возвращает PitchTrack. tracking — порог озвончённости (больше = чувствительнее)."""
    x = np.asarray(x, dtype=np.float64)
    target_sr = 22050.0
    dec = max(1, int(round(sr / target_sr)))
    xd = resample_poly(x, 1, dec) if dec > 1 else x.copy()
    srd = sr / dec

    tau_min = max(2, int(np.floor(srd / fmax)))
    tau_max = int(np.ceil(srd / fmin))
    W = int(np.ceil(2.2 * tau_max / 64.0) * 64)
    W = max(W, 768)
    hop_d = 128
    hop = hop_d * dec

    # Центр кадра: сдвигаем так, чтобы кадр i соответствовал времени i*hop.
    pad = (W + int(srd / 200.0)) // 2
    xp = np.concatenate([np.zeros(pad), xd, np.zeros(W + tau_max + hop_d * 2)])
    n_frames = int(np.ceil(len(x) / hop)) + 1
    flen = W + tau_max

    win = np.lib.stride_tricks.sliding_window_view(xp, flen)[::hop_d]
    win = win[:n_frames]
    if win.shape[0] < n_frames:  # на всякий случай
        n_frames = win.shape[0]

    K = 8
    cand_tau = np.full((n_frames, K), np.nan)
    cand_cost = np.full((n_frames, K), np.nan)
    dmin = np.ones(n_frames)

    # Уровень кадров (по центральной части окна)
    rms_db = np.full(n_frames, -120.0)

    chunk = 1500
    for s in range(0, n_frames, chunk):
        if cancel is not None and cancel():
            raise InterruptedError
        e = min(n_frames, s + chunk)
        fr = np.array(win[s:e], dtype=np.float64)
        dn = _frames_diff(fr, W, tau_max)

        seg = fr[:, : W]
        rms = np.sqrt(np.mean(seg ** 2, axis=1) + 1e-20)
        rms_db[s:e] = 20.0 * np.log10(rms + 1e-12)

        # локальные минимумы
        mid = dn[:, 1:-1]
        ismin = (mid < dn[:, :-2]) & (mid <= dn[:, 2:])
        valid = np.zeros_like(dn, dtype=bool)
        valid[:, 1:-1] = ismin
        valid[:, :tau_min] = False
        valid[:, tau_max:] = False
        valid &= dn < 0.6
        # ключ отбора: d' + небольшой штраф за длинный период (против субгармоник)
        bias = 0.05 * np.log2(np.maximum(np.arange(tau_max + 1), 1) / float(tau_min))
        key = np.where(valid, dn + bias[None, :], np.inf)
        idx = np.argpartition(key, K - 1, axis=1)[:, :K]
        kv = np.take_along_axis(key, idx, axis=1)
        order = np.argsort(kv, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(np.where(valid, dn, np.inf), idx, axis=1)

        # параболическая интерполяция
        i0 = np.clip(idx - 1, 0, tau_max)
        i2 = np.clip(idx + 1, 0, tau_max)
        y0 = np.take_along_axis(dn, i0, axis=1)
        y1 = np.take_along_axis(dn, idx, axis=1)
        y2 = np.take_along_axis(dn, i2, axis=1)
        den = y0 - 2.0 * y1 + y2
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
        shift = np.clip(shift, -1.0, 1.0)
        tau_ref = idx + shift
        ok = np.isfinite(vals)
        tau_ref = np.where(ok, tau_ref, np.nan)
        vals = np.where(ok, vals, np.nan)
        cand_tau[s:e] = tau_ref
        cand_cost[s:e] = vals
        dmin[s:e] = np.where(ok.any(axis=1), np.nanmin(np.where(ok, vals, np.inf), axis=1), 1.0)
        if progress:
            progress(0.5 * e / n_frames)

    # Порог шумового гейта: абсолютный и относительный к пику
    peak_db = np.percentile(rms_db, 98) if n_frames else -120.0
    gate = max(gate_db, peak_db - 48.0)
    silent = rms_db < gate

    f0, voiced, conf = _viterbi(cand_tau, cand_cost, silent, srd, tracking, progress)

    # Вырезаем слишком короткие озвончённые фрагменты, заполняем мелкие дырки
    voiced = _clean_voicing(voiced, f0)
    f0 = np.where(voiced, f0, 0.0)
    f0 = _median_smooth_log(f0, voiced, 5)
    voiced = _reject_outliers(f0, voiced)
    f0 = np.where(voiced, f0, 0.0)
    f0, voiced = _extend_runs(f0, voiced, rms_db, gate, 2)
    return PitchTrack(sr=sr, hop=hop, f0=f0, voiced=voiced, conf=conf, rms_db=rms_db)


def _viterbi(cand_tau, cand_cost, silent, srd, tracking, progress=None):
    F, K = cand_tau.shape
    S = K + 1  # последнее состояние — безголосое
    INF = 1e9
    big = 1e3

    unv_cost = np.full(F, tracking)
    unv_cost[silent] = 0.0

    # стоимость кандидатов с небольшим смещением в пользу короткого периода
    tmin_frame = np.nanmin(np.where(np.isfinite(cand_tau), cand_tau, np.inf), axis=1)
    tmin_frame = np.where(np.isfinite(tmin_frame), tmin_frame, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cost_v = cand_cost + 0.05 * np.log2(np.maximum(cand_tau, 1e-6) / tmin_frame[:, None])
    cost_v = np.where(np.isfinite(cost_v), cost_v, big)
    cost_v[silent] = big

    semis = np.where(np.isfinite(cand_tau), 12.0 * np.log2(srd / np.maximum(cand_tau, 1e-6) / 100.0), 0.0)

    delta = np.full(S, INF)
    delta[:K] = cost_v[0]
    delta[K] = unv_cost[0]
    back = np.zeros((F, S), dtype=np.int8)

    vv_lambda, vv_cap, vu_cost = 0.03, 0.35, 0.10
    for i in range(1, F):
        d = np.abs(semis[i][None, :] - semis[i - 1][:, None])  # (prev, cur) для воиc-воиc
        trans = np.empty((S, S))
        trans[:K, :K] = np.minimum(vv_lambda * d, vv_cap)
        trans[:K, K] = vu_cost
        trans[K, :K] = vu_cost
        trans[K, K] = 0.0
        tot = delta[:, None] + trans
        bk = np.argmin(tot, axis=0)
        newd = tot[bk, np.arange(S)]
        newd[:K] += cost_v[i]
        newd[K] += unv_cost[i]
        delta = newd
        back[i] = bk
    state = int(np.argmin(delta))
    path = np.empty(F, dtype=np.int64)
    for i in range(F - 1, -1, -1):
        path[i] = state
        state = back[i, state]

    f0 = np.zeros(F)
    voiced = path < K
    rows = np.arange(F)
    sel = np.where(voiced, path, 0)
    tau = cand_tau[rows, sel]
    voiced &= np.isfinite(tau)
    f0[voiced] = srd / tau[voiced]
    c = cand_cost[rows, sel]
    conf = np.where(voiced, np.clip(1.0 - c, 0.0, 1.0), 0.0)
    conf = np.nan_to_num(conf)
    return f0, voiced, conf


def _runs(mask: np.ndarray):
    """Список (start, end_exclusive) для True-серий."""
    if len(mask) == 0:
        return []
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    s = np.nonzero(d == 1)[0]
    e = np.nonzero(d == -1)[0]
    return list(zip(s, e))


def _clean_voicing(voiced: np.ndarray, f0: np.ndarray, min_len: int = 5, max_gap: int = 3) -> np.ndarray:
    v = voiced.copy()
    # заполняем короткие дырки, если тон по краям близок
    for s, e in _runs(~v):
        if s == 0 or e >= len(v):
            continue
        if e - s <= max_gap and f0[s - 1] > 0 and f0[e] > 0:
            if abs(12 * np.log2(f0[s - 1] / f0[e])) < 1.5:
                v[s:e] = True
                a, b = np.log(f0[s - 1]), np.log(f0[e])
                f0[s:e] = np.exp(np.linspace(a, b, e - s + 2)[1:-1])
    for s, e in _runs(v):
        if e - s < min_len:
            v[s:e] = False
    return v


def _median_smooth_log(f0: np.ndarray, voiced: np.ndarray, k: int) -> np.ndarray:
    out = f0.copy()
    half = k // 2
    for s, e in _runs(voiced):
        seg = np.log(f0[s:e])
        if e - s < k:
            continue
        pad = np.pad(seg, half, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, k)
        out[s:e] = np.exp(np.median(win, axis=1))
    return out


def _reject_outliers(f0: np.ndarray, voiced: np.ndarray, win: int = 21, thr_st: float = 3.5) -> np.ndarray:
    """Отбрасывает одиночные кадры, сильно отклоняющиеся от локальной медианы (ошибки трекинга)."""
    v = voiced.copy()
    half = win // 2
    for s, e in _runs(voiced):
        if e - s < 4:
            continue
        m = 12.0 * np.log2(f0[s:e])
        pad = np.pad(m, half, mode="edge")
        med = np.median(np.lib.stride_tricks.sliding_window_view(pad, win), axis=1)
        bad = np.abs(m - med) > thr_st
        v[s:e][bad] = False
    # короткие обрывки после вычистки
    for s, e in _runs(v):
        if e - s < 5:
            v[s:e] = False
    return v


def _extend_runs(f0: np.ndarray, voiced: np.ndarray, rms_db: np.ndarray, gate: float, n: int = 2):
    """Продлевает озвончённые серии на n кадров с краёв (если там есть сигнал): начало и конец нот
    часто теряются детектором, из-за чего их края оставались бы нескорректированными."""
    f0 = f0.copy()
    v = voiced.copy()
    F = len(f0)
    for s, e in _runs(voiced):
        for k in range(1, n + 1):
            i = s - k
            if i >= 0 and not voiced[i] and rms_db[i] > gate - 6.0:
                f0[i], v[i] = f0[s], True
        for k in range(0, n):
            i = e + k
            if i < F and not voiced[i] and rms_db[i] > gate - 6.0:
                f0[i], v[i] = f0[e - 1], True
    return f0, v
