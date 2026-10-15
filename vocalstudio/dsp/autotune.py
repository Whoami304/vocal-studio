"""Автотюн: расчёт целевой высоты (Retune Speed / Humanize / Flex-Tune / ...)
и ресинтез голоса методом TD-PSOLA с субсэмпловой точностью.

PSOLA сохраняет форманты (тембр) естественным образом: двигается только
расстояние между голосовыми импульсами, а форма самих импульсов остаётся.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt, resample as fft_resample
from scipy.signal.windows import hann

from .pitch import PitchTrack, _runs


@dataclass
class Correction:
    """Результат расчёта коррекции (покадрово, как у PitchTrack)."""
    f_out: np.ndarray        # итоговая частота, Гц (0 — без тона)
    target_midi: np.ndarray  # целевая нота, MIDI (nan — нет)
    in_midi: np.ndarray      # исходная высота, MIDI (nan — нет)
    out_midi: np.ndarray     # итоговая высота, MIDI (nan — нет)


def _nearest_allowed(m: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(allowed, m)
    lo = allowed[np.clip(idx - 1, 0, len(allowed) - 1)]
    hi = allowed[np.clip(idx, 0, len(allowed) - 1)]
    return np.where(np.abs(m - lo) <= np.abs(hi - m), lo, hi)


def compute_correction(track: PitchTrack, p: dict, overrides=None) -> Correction:
    """p — параметры модуля tune. overrides — [(t0_sec, t1_sec, midi|None)]."""
    sr, hop = track.sr, track.hop
    dt = hop / sr
    ref_a = float(p.get("ref_a", 440.0))
    F = len(track.f0)
    in_midi = track.midi(ref_a)
    out_midi = np.full(F, np.nan)
    target_midi = np.full(F, np.nan)

    mask = list(p.get("notes", [True] * 12))
    allowed = np.array([n for n in range(12, 128) if mask[n % 12]], dtype=np.float64)
    retune = float(p.get("retune", 20.0)) / 1000.0
    humanize = float(p.get("humanize", 0.0)) / 100.0
    flex = float(p.get("flex", 0.0)) * 0.5  # центы
    nv = float(p.get("vibrato", 100.0)) / 100.0
    transpose = float(p.get("transpose", 0.0))
    detune = float(p.get("detune", 0.0)) / 100.0
    amount = float(p.get("amount", 100.0)) / 100.0
    hyst = 0.12 + 0.25 * humanize

    ov = []
    for (t0, t1, note) in (overrides or []):
        ov.append((int(t0 * sr / hop), int(np.ceil(t1 * sr / hop)), note))

    vib_win = max(3, int(round(0.18 / dt)))
    for (s, e) in _runs(track.voiced):
        m = in_midi[s:e]
        n = e - s
        w = min(vib_win, n)
        centre = uniform_filter1d(m, size=w, mode="nearest")

        if len(allowed) > 0:
            near = _nearest_allowed(centre, allowed)
        else:
            near = centre.copy()

        tgt = np.empty(n)
        cur = near[0]
        for i in range(n):
            ni = near[i]
            if ni != cur and abs(centre[i] - ni) + hyst < abs(centre[i] - cur):
                cur = ni
            tgt[i] = cur

        # ручные цели из графика
        bypass = np.zeros(n, dtype=bool)
        for (a, b, note) in ov:
            lo, hi = max(a, s), min(b, e)
            if hi > lo:
                if note is None:
                    bypass[lo - s: hi - s] = True
                else:
                    tgt[lo - s: hi - s] = note
        if len(allowed) == 0:
            bypass[:] = True

        tgt_shifted = tgt + transpose + detune

        # Flex-Tune: естественные небольшие отклонения от ноты сохраняются
        e_c = (centre - tgt) * 100.0
        if flex > 0:
            resid = e_c / (1.0 + (np.abs(e_c) / flex) ** 2)
        else:
            resid = np.zeros_like(e_c)
        desired = tgt_shifted + resid / 100.0

        # масштабирование вибрато
        m2 = centre + nv * (m - centre)

        off = 0.0
        out = np.empty(n)
        t_on = 0.0
        last_t = tgt[0]
        for i in range(n):
            if tgt[i] != last_t:
                t_on = 0.0
                last_t = tgt[i]
            else:
                t_on += dt
            if bypass[i]:
                off_target = transpose + detune
            else:
                off_target = desired[i] - m2[i]
            off_target *= amount
            tau = retune + humanize * 0.45 * (1.0 - np.exp(-t_on / 0.6))
            a = 1.0 if tau <= 1e-4 else 1.0 - np.exp(-dt / tau)
            off += a * (off_target - off)
            out[i] = m2[i] + off
        out_midi[s:e] = out
        target_midi[s:e] = np.where(bypass, np.nan, tgt_shifted)

    f_out = np.zeros(F)
    v = np.isfinite(out_midi)
    f_out[v] = ref_a * 2.0 ** ((out_midi[v] - 69.0) / 12.0)
    return Correction(f_out=f_out, target_midi=target_midi, in_midi=in_midi, out_midi=out_midi)


# ---------------------------------------------------------------- PSOLA ----

def _find_marks(x: np.ndarray, sr: int, t0: int, t1: int, tin) -> np.ndarray:
    """Питч-метки на отрезке [t0, t1). tin(t) — период в отсчётах."""
    n = len(x)
    pad = int(0.02 * sr)
    a, b = max(0, t0 - pad), min(n, t1 + pad)
    seg = x[a:b]
    if len(seg) < 64:
        return np.array([])
    f_med = sr / np.median(tin(np.arange(t0, t1, 16)))
    fc = float(np.clip(2.2 * f_med, 150.0, 0.45 * sr))
    try:
        sos = butter(4, fc / (0.5 * sr), btype="low", output="sos")
        y = sosfiltfilt(sos, seg)
    except ValueError:
        y = seg.copy()
    if abs(y.min()) > y.max():
        y = -y
    ymax = max(np.max(np.abs(y)), 1e-9)
    y = y / ymax

    marks = []
    T0 = tin(t0)
    lo = t0 - a
    hi = min(len(y) - 2, int(lo + T0))
    if hi <= lo + 2:
        return np.array([])
    p = lo + int(np.argmax(y[lo:hi]))
    last = t1 - a
    while p < last:
        marks.append(p)
        T = tin(p + a)
        exp_p = p + T
        w = 0.28 * T
        i0 = int(np.ceil(exp_p - w))
        i1 = int(np.floor(exp_p + w))
        if i1 >= len(y) - 1:
            break
        i0 = max(i0, p + 1)
        if i1 <= i0:
            p = int(round(exp_p))
            continue
        seg_y = y[i0:i1 + 1]
        pos = np.arange(i0, i1 + 1)
        score = seg_y - 0.35 * np.abs(pos - exp_p) / w
        p = i0 + int(np.argmax(score))
    if len(marks) < 3:
        return np.array([])
    marks = np.array(marks, dtype=np.float64)
    # субсэмпловое уточнение
    mi = marks.astype(int)
    y0, y1, y2 = y[mi - 1], y[mi], y[mi + 1]
    den = y0 - 2 * y1 + y2
    d = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    marks = marks + np.clip(d, -0.5, 0.5)
    return marks + a


def _plan_run(A, tout_frames, hop, sr):
    """Планирует синтезируемые зёрна для одной озвончённой серии.

    A — питч-метки (дробные, отсчёты). tout_frames — желаемый период (отсчёты) на кадрах серии,
    кадр j находится в отсчёте ft0 + j*hop. Возвращает массивы (k, s_pos, Ts)."""
    nA = len(A)
    ks, ss, Tss = [], [], []
    s_pos = float(A[0])
    end_pos = float(A[-1])
    k = 0
    nf = len(tout_frames)
    tl = tout_frames.tolist()
    inv_hop = 1.0 / hop
    Al = A.tolist()
    while s_pos <= end_pos:
        while k + 1 < nA and abs(Al[k + 1] - s_pos) <= abs(Al[k] - s_pos):
            k += 1
        fpos = s_pos * inv_hop
        j = int(fpos)
        if j >= nf - 1:
            Ts = tl[-1]
        elif j < 0:
            Ts = tl[0]
        else:
            fr = fpos - j
            Ts = tl[j] * (1.0 - fr) + tl[j + 1] * fr
        ks.append(k)
        ss.append(s_pos)
        Tss.append(Ts)
        s_pos += Ts
    return np.array(ks), np.array(ss), np.array(Tss)


def psola_shift(
    x: np.ndarray,
    track: PitchTrack,
    f_out: np.ndarray,
    formant_st: float = 0.0,
    cancel=None,
    progress=None,
) -> np.ndarray:
    sr, hop = track.sr, track.hop
    n = len(x)
    PADW = 8192
    xp = np.concatenate([np.zeros(PADW), x, np.zeros(PADW)])
    wet = np.zeros(n + 2 * PADW)
    wsum = np.zeros_like(wet)
    cov = np.zeros_like(wet)
    phi = 2.0 ** (formant_st / 12.0)
    use_formant = abs(formant_st) > 1e-3

    runs = _runs(track.voiced & (f_out > 0))
    plan = []   # (c, delta, cs, shift, h, Ts) по зёрнам всех серий
    total_runs = max(1, len(runs))
    for ri, (s, e) in enumerate(runs):
        if cancel is not None and cancel():
            raise InterruptedError
        ft = np.arange(s, e) * hop
        fin = np.log(track.f0[s:e])
        t0 = max(0, s * hop - hop // 2)
        t1 = min(n, e * hop + hop // 2)
        if t1 - t0 < int(0.02 * sr):
            continue

        def tin(t, ft=ft, fin=fin):
            return sr / np.exp(np.interp(t, ft, fin))

        marks = _find_marks(x, sr, t0, t1, tin)
        if len(marks) < 3:
            continue
        A = marks
        Pk = np.empty_like(A)
        Pk[1:-1] = (A[2:] - A[:-2]) * 0.5
        Pk[0] = A[1] - A[0]
        Pk[-1] = A[-1] - A[-2]
        Pk = np.clip(Pk, 8.0, 0.5 * sr / 40.0)

        # желаемый период по кадрам серии, индексация относительно начала s*hop
        tout_frames = sr / f_out[s:e]
        ks, ss, Tss = _plan_run(A - s * hop, tout_frames, hop, sr)
        sp = ss + s * hop
        h = np.minimum(np.round(np.maximum(Pk[ks], 0.9 * Tss)).astype(np.int64), 4000)
        c = np.round(A[ks]).astype(np.int64)
        delta = A[ks] - c
        cs = np.round(sp).astype(np.int64)
        shift = (sp - cs) - delta
        plan.append((c, cs, shift, h, Tss))
        if progress:
            progress(0.5 * (ri + 1) / total_runs)

    if plan:
        c = np.concatenate([p_[0] for p_ in plan])
        cs = np.concatenate([p_[1] for p_ in plan])
        shift = np.concatenate([p_[2] for p_ in plan])
        h = np.concatenate([p_[3] for p_ in plan])
        Ts = np.concatenate([p_[4] for p_ in plan])
        order = np.argsort(cs, kind="stable")
        c, cs, shift, h, Ts = c[order], cs[order], shift[order], h[order], Ts[order]
        if use_formant:
            _render_grains_loop(xp, wet, wsum, cov, PADW, c, cs, shift, h, Ts, phi, cancel)
        else:
            _render_grains_vec(xp, wet, wsum, cov, PADW, c, cs, shift, h, Ts, cancel, progress)

    wet = wet[PADW:PADW + n]
    wsum = wsum[PADW:PADW + n]
    cov = cov[PADW:PADW + n]
    good = wsum > 1e-3
    wet_n = np.where(good, wet / np.maximum(wsum, 1e-3), 0.0)
    m = np.clip(cov, 0.0, 1.0)
    m = uniform_filter1d(m, size=max(3, int(0.001 * sr)), mode="nearest")
    # Выравниваем громкость: PSOLA слегка теряет энергию при сдвиге высоты
    win = max(3, int(0.03 * sr)) | 1
    e_in = uniform_filter1d(x * x, size=win, mode="nearest")
    e_out = uniform_filter1d(wet_n * wet_n, size=win, mode="nearest")
    g = np.sqrt((e_in + 1e-9) / (e_out + 1e-9))
    g = np.where(m > 0.5, np.clip(g, 0.6, 1.8), 1.0)
    g = uniform_filter1d(g, size=max(3, int(0.01 * sr)), mode="nearest")
    wet_n = wet_n * g
    return x * (1.0 - m) + wet_n * m


def _render_grains_vec(xp, wet, wsum, cov, PADW, c, cs, shift, h, Ts, cancel=None, progress=None, B=512):
    """Пакетная раскладка зёрен: окно, дробный сдвиг через FFT и overlap-add блоками."""
    G = len(c)
    for b0 in range(0, G, B):
        if cancel is not None and cancel():
            raise InterruptedError
        sl = slice(b0, min(G, b0 + B))
        cb, csb, shb, hb, Tb = c[sl], cs[sl], shift[sl], h[sl], Ts[sl]
        hmax = int(hb.max())
        N = next_fast_len(2 * hmax + 1 + 16)
        col = np.arange(N)[None, :]
        j = col - 8                                   # индекс внутри зерна
        L1 = (2 * hb + 1)[:, None]
        valid = (j >= 0) & (j < L1)
        hh = hb[:, None]
        src_idx = PADW + cb[:, None] + (j - hh)
        src_idx = np.clip(src_idx, 0, len(xp) - 1)
        w = np.where(valid, 0.5 - 0.5 * np.cos(2.0 * np.pi * (j + 1) / (2 * hh + 2)), 0.0)
        seg = xp[src_idx] * w
        Fq = rfft(seg, N, axis=1)
        k = np.arange(Fq.shape[1])[None, :]
        Fq *= np.exp(-2j * np.pi * k * (shb[:, None] / N))
        seg = irfft(Fq, N, axis=1)
        pos = (PADW + csb - hb - 8)[:, None] + col      # позиции в буфере
        lo = int(pos.min())
        span = int(pos.max()) - lo + 1
        flat = (pos - lo).ravel()
        wet[lo:lo + span] += np.bincount(flat, weights=seg.ravel(), minlength=span)
        wsum[lo:lo + span] += np.bincount(flat, weights=w.ravel(), minlength=span)
        cov[lo:lo + span] += np.bincount(flat, weights=(w * (Tb / hb)[:, None]).ravel(), minlength=span)
        if progress:
            progress(0.5 + 0.5 * min(1.0, (b0 + B) / G))


def _render_grains_loop(xp, wet, wsum, cov, PADW, c, cs, shift, h, Ts, phi, cancel=None):
    """Медленный путь (используется со сдвигом формант): зерно ресэмплируется."""
    win_cache: dict[int, np.ndarray] = {}

    def get_win(hh: int) -> np.ndarray:
        w = win_cache.get(hh)
        if w is None:
            w = hann(2 * hh + 3, sym=True)[1:-1]
            win_cache[hh] = w
        return w

    for i in range(len(c)):
        if cancel is not None and (i & 255) == 0 and cancel():
            raise InterruptedError
        hh = int(h[i])
        ci, csi, sh = int(c[i]), int(cs[i]), float(shift[i])
        hs = max(4, int(round(hh * phi)))
        seg = xp[PADW + ci - hs: PADW + ci + hs + 1] * get_win(hs)
        seg = fft_resample(seg, 2 * hh + 1)
        L = 2 * hh + 1
        if abs(sh) > 0.01:
            N = next_fast_len(L + 16)
            buf = np.zeros(N)
            buf[8:8 + L] = seg
            Fq = rfft(buf)
            Fq *= np.exp(-2j * np.pi * np.arange(len(Fq)) * sh / N)
            buf = irfft(Fq, N)
            st = PADW + csi - hh - 8
            wet[st: st + N] += buf
        else:
            st = PADW + csi - hh
            wet[st: st + L] += seg
        w = get_win(hh)
        st = PADW + csi - hh
        wsum[st: st + L] += w
        cov[st: st + L] += w * (float(Ts[i]) / hh)


def autotune(x: np.ndarray, track: PitchTrack, p: dict, overrides=None, cancel=None, progress=None):
    corr = compute_correction(track, p, overrides)
    y = psola_shift(
        x, track, corr.f_out,
        formant_st=float(p.get("formant", 0.0)),
        cancel=cancel, progress=progress,
    )
    return y, corr
