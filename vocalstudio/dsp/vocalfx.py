"""Вокальные эффекты в духе iZotope Nectar: Breath Control, Vocal Rider, Doubler, Harmony, Exciter, Unmasker."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt

from . import dynamics
from .autotune import compute_correction, psola_shift
from .eq import bandpass_sos
from .pitch import PitchTrack, _runs
from .scales import NOTE_NAMES

# интервалы гармонии: (подпись, шаги по ступеням лада, полутонов для «октавы»)
INTERVALS = {
    "Off": None,
    "3rd above": ("steps", 2), "5th above": ("steps", 4), "6th above": ("steps", 5), "Octave above": ("semi", 12),
    "3rd below": ("steps", -2), "5th below": ("steps", -4), "6th below": ("steps", -5), "Octave below": ("semi", -12),
}


def _control_apply(x: np.ndarray, gain_db_frames: np.ndarray, hop: int) -> np.ndarray:
    """Применяет покадровое усиление (дБ) с интерполяцией до частоты дискретизации."""
    n = x.shape[-1]
    centers = np.arange(len(gain_db_frames)) * hop
    g = np.interp(np.arange(n), centers, gain_db_frames)
    return x * (10.0 ** (g / 20.0))


# ------------------------------------------------------------------ Breath --

def breath_control(x: np.ndarray, sr: int, p: dict, track: PitchTrack) -> np.ndarray:
    """Приглушает вдохи: долгие безголосые участки умеренной громкости без «свистящего» спектра."""
    reduction = float(p.get("reduction", 12.0))
    sens = float(p.get("sensitivity", 50.0)) / 100.0
    if reduction <= 0.1 or len(track.f0) < 8:
        return x
    mono = x if x.ndim == 1 else x.mean(axis=0)
    hop = track.hop
    F = len(track.f0)
    rms = track.rms_db
    voiced = track.voiced
    if voiced.sum() < 10:
        return x
    lvl_v = float(np.percentile(rms[voiced], 60))
    floor = float(np.percentile(rms, 8))
    # спектральный признак: доля энергии выше 5 кГц в каждом кадре
    hp = sosfilt(butter(2, 5000.0 / (0.5 * sr), "high", output="sos"), mono)
    n = len(mono)
    pad = F * hop - n
    e_all = np.pad(mono * mono, (0, max(0, pad)))[: F * hop].reshape(F, hop).mean(axis=1) + 1e-12
    e_hf = np.pad(hp * hp, (0, max(0, pad)))[: F * hop].reshape(F, hop).mean(axis=1) + 1e-12
    hf_ratio = e_hf / e_all
    # кандидаты: не голос, тише голоса, заметно выше шума, спектр не «сибилянтный»
    hi = lvl_v - (4.0 + 6.0 * (1.0 - sens))
    lo = floor + 5.0 - 3.0 * sens
    cand = (~voiced) & (rms < hi) & (rms > lo) & (hf_ratio < 0.5 + 0.2 * sens)
    min_len = int((0.20 - 0.10 * sens) / (hop / sr))
    gain = np.zeros(F)
    for s_, e_ in _runs(cand):
        if e_ - s_ >= min_len:
            gain[s_:e_] = -reduction
    # плавные края (10 мс вниз, 30 мс вверх)
    sm = dynamics._smooth(gain, dynamics._coef(10.0, sr / hop), dynamics._coef(30.0, sr / hop))
    # опережение, чтобы не задевать начало слога: сдвиг на 1 кадр назад
    sm = np.concatenate([sm[1:], sm[-1:]])
    return _control_apply(x, sm, hop)


# ------------------------------------------------------------------- Rider --

def vocal_rider(x: np.ndarray, sr: int, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Автоматическая «ручка громкости»: подтягивает тихие фразы и убавляет громкие к целевому уровню."""
    target = float(p.get("target", -20.0))
    rng = float(p.get("range", 8.0))
    speed = float(p.get("speed", 250.0))
    gate = float(p.get("gate", -55.0))
    det = dynamics._detector(x, "rms")
    L, B = dynamics._level_db(det, sr, "rms")
    want = np.clip(target - L, -rng, rng)
    want = np.where(L > gate, want, 0.0)          # шум и паузы не усиливаем
    fsc = sr / B
    sm = dynamics._smooth(want, dynamics._coef(speed, fsc), dynamics._coef(speed * 1.6, fsc))
    g_db = dynamics._to_audio_rate(sm, B, x.shape[-1])
    return x * (10.0 ** (g_db / 20.0)), g_db


# ----------------------------------------------------------------- Exciter --

def exciter(x: np.ndarray, sr: int, p: dict) -> np.ndarray:
    f = float(p.get("freq", 4500.0))
    amount = float(p.get("amount", 25.0)) / 100.0
    if amount <= 0.001:
        return x
    sos = butter(2, min(f, 0.45 * sr) / (0.5 * sr), "high", output="sos")
    hp = sosfilt(sos, x, axis=-1)
    drive = 4.0
    exc = np.tanh(drive * hp + 0.25) - np.tanh(0.25)      # несимметрично: чётные + нечётные гармоники
    exc = sosfilt(sos, exc / drive, axis=-1)               # оставляем только верх
    return x + amount * 1.5 * exc


# ----------------------------------------------------------------- Doubler --

def _wander(n_frames: int, rng: np.random.Generator, amp: float, smooth: int = 40) -> np.ndarray:
    w = rng.standard_normal(n_frames + smooth * 4)
    w = uniform_filter1d(w, smooth)
    w = w[smooth * 2: smooth * 2 + n_frames]
    return w / (np.std(w) + 1e-9) * amp


def _frac_delay(x: np.ndarray, delay: np.ndarray) -> np.ndarray:
    """Дробная задержка с переменным значением (отсчёты)."""
    n = len(x)
    idx = np.arange(n) - delay
    return np.interp(idx, np.arange(n), x, left=0.0, right=0.0)


def doubler(x: np.ndarray, sr: int, p: dict, track: PitchTrack) -> np.ndarray:
    """Имитация «двойного дубля»: копии с лёгкой расстройкой, таймингом и панорамой -> (2, n)."""
    voices = int(round(float(p.get("voices", 2))))
    detune = float(p.get("detune", 9.0))
    delay_ms = float(p.get("delay", 18.0))
    width = float(p.get("width", 80.0)) / 100.0
    mix = float(p.get("mix", 35.0)) / 100.0
    mono = x if x.ndim == 1 else x.mean(axis=0)
    n = len(mono)
    out = np.vstack([mono, mono]).astype(np.float64)
    if voices < 1 or mix <= 0.001:
        return out
    rng = np.random.default_rng(7)
    F = len(track.f0)
    acc = np.zeros((2, n))
    for v in range(voices):
        side = -1.0 if v % 2 == 0 else 1.0
        cents = side * detune * (0.75 + 0.5 * rng.random()) + _wander(F, rng, 2.5)
        f_out = np.where(track.voiced, track.f0 * 2.0 ** (cents / 1200.0), 0.0)
        y = psola_shift(mono, track, f_out)
        d = (delay_ms * (1.0 + 0.45 * (v // 2)) * 0.001 * sr) * np.ones(n)
        d = d + np.interp(np.arange(n), np.arange(F) * track.hop, _wander(F, rng, 0.0025 * sr, 60))
        y = _frac_delay(y, np.maximum(d, 0.0))
        pan = side * width
        gl, gr = np.sqrt(0.5 * (1.0 - pan)), np.sqrt(0.5 * (1.0 + pan))
        acc[0] += y * gl
        acc[1] += y * gr
    return out + acc * mix * (1.0 / np.sqrt(max(1, voices))) * 1.2


# ----------------------------------------------------------------- Harmony --

def harmony(x: np.ndarray, sr: int, p: dict, track: PitchTrack) -> np.ndarray:
    """Создаёт гармонические голоса по ступеням лада (3-я, 5-я, 6-я, октава) -> (2, n)."""
    mono = x if x.ndim == 1 else x.mean(axis=0)
    n = len(mono)
    dry = np.vstack([mono, mono]).astype(np.float64)
    mix = float(p.get("mix", 55.0)) / 100.0
    mask = list(p.get("notes", [True] * 12))
    allowed = np.array([m for m in range(12, 128) if mask[m % 12]], dtype=float)
    if mix <= 0.001 or len(allowed) < 3 or track.voiced.sum() < 10:
        return dry
    ref_a = float(p.get("ref_a", 440.0))
    snap = {"retune": 0.0, "humanize": 0.0, "flex": 0.0, "vibrato": 100.0, "transpose": 0.0, "detune": 0.0,
            "amount": 100.0, "ref_a": ref_a, "notes": mask}
    corr = compute_correction(track, snap)
    in_midi = corr.in_midi
    lead_note = corr.target_midi
    ok = np.isfinite(lead_note) & np.isfinite(in_midi)
    centre = np.where(ok, uniform_filter1d(np.where(ok, in_midi, 0.0), max(3, int(0.18 / (track.hop / sr))), mode="nearest"), np.nan)
    vib = np.where(ok, in_midi - centre, 0.0)
    rng = np.random.default_rng(11)
    F = len(track.f0)
    formant = float(p.get("formant", 0.0))
    timing = float(p.get("timing", 12.0)) * 0.001 * sr
    drift = float(p.get("drift", 6.0))
    acc = np.zeros((2, n))
    nv = 0
    for k in range(1, 5):
        name = p.get(f"v{k}_int", "Off")
        spec = INTERVALS.get(name)
        if spec is None:
            continue
        nv += 1
        kind, amount = spec
        idx = np.abs(allowed[None, :] - np.nan_to_num(lead_note, nan=60.0)[:, None]).argmin(axis=1)
        if kind == "steps":
            h_note = allowed[np.clip(idx + int(amount), 0, len(allowed) - 1)]
        else:
            h_note = np.nan_to_num(lead_note, nan=60.0) + amount
        h_midi = h_note + vib + _wander(F, rng, drift / 100.0, 50)
        f_out = np.where(ok, ref_a * 2.0 ** ((h_midi - 69.0) / 12.0), 0.0)
        y = psola_shift(mono, track, f_out, formant_st=formant)
        d = timing * (0.5 + rng.random()) * np.ones(n)
        d = d + np.interp(np.arange(n), np.arange(F) * track.hop, _wander(F, rng, 0.0015 * sr, 60))
        y = _frac_delay(y, np.maximum(d, 0.0))
        lvl = 10.0 ** (float(p.get(f"v{k}_lvl", -4.0)) / 20.0)
        pan = float(np.clip(p.get(f"v{k}_pan", 0.0) / 100.0, -1.0, 1.0))
        acc[0] += y * lvl * np.sqrt(0.5 * (1.0 - pan))
        acc[1] += y * lvl * np.sqrt(0.5 * (1.0 + pan))
    return dry + acc * mix


# ----------------------------------------------------------------- Unmasker --

def unmasker(x: np.ndarray, sidechain: np.ndarray | None, sr: int, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Освобождает место голосу: когда поёт вокал, в музыке слегка приглушается полоса, где они конфликтуют."""
    f = float(p.get("freq", 2500.0))
    q = float(p.get("q", 1.2))
    red = float(p.get("reduction", 3.0))
    thr = float(p.get("threshold", -38.0))
    rel = float(p.get("release", 140.0))
    n = x.shape[-1]
    if sidechain is None or red <= 0.05 or len(sidechain) < 16:
        return x, np.zeros(max(1, n // 64))
    sc = np.zeros(n)
    m = min(n, len(sidechain))
    sc[:m] = sidechain[:m]
    sos = bandpass_sos(sr, f, 1.0)
    det = sosfilt(sos, sc)
    L, B = dynamics._level_db(det, sr, "peak")
    gr = -np.clip((L - thr) * 0.8, 0.0, red)
    fsc = sr / B
    sm = dynamics._smooth(gr, dynamics._coef(8.0, fsc), dynamics._coef(rel, fsc))
    g_db = dynamics._to_audio_rate(sm, B, n)
    g = 10.0 ** (g_db / 20.0)
    band = sosfilt(bandpass_sos(sr, f, q), x, axis=-1)
    return x - (1.0 - g) * band, g_db[::64]
