"""Анализ голоса и подложки: тональность, строй, темп, тембр, динамика, сибилянты.
На его основе Auto-Mix подбирает цепочку плагинов."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly, sosfilt, stft, welch

from .dsp import dynamics
from .dsp.eq import bandpass_sos
from .dsp.pitch import INPUT_RANGES, PitchTrack, detect_pitch
from .dsp.scales import NOTE_NAMES, SCALES, scale_mask

# профили Краскла-Шмуклера
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class KeyResult:
    key: int
    mode: str                       # "major" | "minor"
    corr: float
    margin: float
    tuning_cents: float
    hist: np.ndarray
    alternatives: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.key]} {'мажор' if self.mode == 'major' else 'минор'}"

    @property
    def scale(self) -> str:
        return "Major" if self.mode == "major" else "Minor"

    @property
    def confident(self) -> bool:
        return self.corr >= 0.62 and self.margin >= 0.04

    @property
    def confidence_label(self) -> str:
        if self.corr >= 0.75 and self.margin >= 0.08:
            return "высокая"
        if self.confident:
            return "средняя"
        return "низкая"


def _tuning_offset(midi: np.ndarray, w: np.ndarray) -> float:
    """Отклонение строя от равномерно-темперированного (центы), круговое среднее."""
    ang = 2.0 * np.pi * (midi - np.round(midi))
    v = np.sum(w * np.exp(1j * ang))
    if abs(v) < 1e-9:
        return 0.0
    return float(np.angle(v) / (2.0 * np.pi) * 100.0)


def _key_from_hist(hist: np.ndarray, tuning: float) -> KeyResult | None:
    if hist.sum() <= 0:
        return None
    h = hist / hist.sum()
    scores = []
    for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
        for k in range(12):
            r = np.corrcoef(h, np.roll(prof, k))[0, 1]
            scores.append((float(r), k, mode))
    scores.sort(reverse=True)
    best = scores[0]
    # «второй» — лучший из других тональностей, кроме той же ноты-тоники в другом ладу
    second = next(s for s in scores[1:] if not (s[1] == best[1]))
    return KeyResult(key=best[1], mode=best[2], corr=best[0], margin=best[0] - second[0], tuning_cents=tuning,
                     hist=hist, alternatives=[(f"{NOTE_NAMES[s[1]]} {s[2]}", s[0]) for s in scores[1:4]])


def pitch_histogram(track: PitchTrack):
    """Гистограмма высотных классов по устойчивым кадрам пения + оценка строя."""
    m = track.midi(440.0)
    conf = track.conf
    rms = 10 ** (track.rms_db / 20.0)
    all_m, all_w = [], []
    from .dsp.pitch import _runs
    for s, e in _runs(track.voiced):
        seg = m[s:e]
        if e - s < 6:
            continue
        sm = uniform_filter1d(seg, size=min(9, e - s), mode="nearest")
        stable = np.abs(seg - sm) < 0.45
        w = conf[s:e] * rms[s:e] * stable
        all_m.append(sm)
        all_w.append(w)
    if not all_m:
        return np.zeros(12), 0.0
    mm = np.concatenate(all_m)
    ww = np.concatenate(all_w)
    ww = ww / (ww.max() + 1e-12)
    tuning = _tuning_offset(mm, ww)
    pcs = np.round(mm - tuning / 100.0).astype(int) % 12
    hist = np.bincount(pcs, weights=ww, minlength=12)
    return hist, tuning


def detect_key_from_pitch(track: PitchTrack) -> KeyResult | None:
    hist, tuning = pitch_histogram(track)
    return _key_from_hist(hist, tuning)


def _mono_ds(x: np.ndarray, sr: int, target: int = 11025, max_seconds: float = 100.0):
    if x.ndim == 2:
        x = x.mean(axis=0)
    n = len(x)
    if n > max_seconds * sr:
        a = (n - int(max_seconds * sr)) // 2
        x = x[a:a + int(max_seconds * sr)]
    dec = max(1, int(round(sr / target)))
    if dec > 1:
        x = resample_poly(x, 1, dec)
    return x.astype(np.float64), sr / dec


def detect_key_from_audio(x: np.ndarray, sr: int) -> KeyResult | None:
    """Тональность и строй полифонической музыки (подложки) по спектральным пикам."""
    xd, fs = _mono_ds(x, sr)
    if len(xd) < 8192 or np.abs(xd).max() < 1e-5:
        return None
    nfft = 4096
    f, t, Z = stft(xd, fs, nperseg=nfft, noverlap=nfft // 2, boundary=None)
    mag = np.abs(Z)                      # (bins, frames)
    lo = int(np.searchsorted(f, 65.0))
    hi = int(np.searchsorted(f, 2000.0))
    mg = mag[lo:hi]                      # (b, F)
    thr = 0.12 * mg.max(axis=0, keepdims=True)
    left = np.vstack([mg[:1] * 0, mg[:-1]])
    right = np.vstack([mg[1:], mg[-1:] * 0])
    peak = (mg > left) & (mg >= right) & (mg > thr)
    bi, fi = np.nonzero(peak)
    if len(bi) < 20:
        return None
    mp = mg[bi, fi]
    # параболическая интерполяция
    a = mg[np.clip(bi - 1, 0, mg.shape[0] - 1), fi]
    c = mg[np.clip(bi + 1, 0, mg.shape[0] - 1), fi]
    den = a - 2 * mp + c
    delta = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    freq = (bi + lo + np.clip(delta, -0.5, 0.5)) * (fs / nfft)
    midi = 69.0 + 12.0 * np.log2(freq / 440.0)
    w = mp / mp.max()
    tuning = _tuning_offset(midi, w)
    pcs = np.round(midi - tuning / 100.0).astype(int) % 12
    hist = np.bincount(pcs, weights=w, minlength=12)
    return _key_from_hist(hist, tuning)


def detect_tempo(x: np.ndarray, sr: int):
    """Грубая оценка темпа (BPM) по автокорреляции онсет-огибающей. -> (bpm, confidence) или None."""
    xd, fs = _mono_ds(x, sr, target=11025, max_seconds=90.0)
    if len(xd) < fs * 4:
        return None
    nfft, hop = 1024, 256
    f, t, Z = stft(xd, fs, nperseg=nfft, noverlap=nfft - hop, boundary=None)
    mag = np.abs(Z)[2:140]
    S = np.log1p(30.0 * mag / (mag.max() + 1e-9))
    flux = np.maximum(np.diff(S, axis=1), 0.0).sum(axis=0)
    flux = flux - uniform_filter1d(flux, 43)
    flux = np.maximum(flux, 0.0)
    if flux.std() < 1e-9:
        return None
    n = len(flux)
    ac = np.fft.irfft(np.abs(np.fft.rfft(flux, 2 * n)) ** 2)[:n]
    ac /= ac[0] + 1e-12
    hop_t = hop / fs
    lags = np.arange(n)
    bpm = np.zeros(n)
    bpm[1:] = 60.0 / (lags[1:] * hop_t)
    sel = (bpm >= 62) & (bpm <= 190)
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1) / 112.0) / 0.7) ** 2)
    score = np.where(sel, ac * prior, -1)
    i = int(np.argmax(score))
    if i <= 1 or i >= n - 1:
        return None
    # параболическое уточнение
    y0, y1, y2 = ac[i - 1], ac[i], ac[i + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    lag = i + float(np.clip(d, -0.5, 0.5))
    return float(60.0 / (lag * hop_t)), float(ac[i])


# ------------------------------------------------------------ анализ голоса --

BANDS = {"low": (60, 150), "mud": (200, 500), "mid": (500, 2000), "pres": (2000, 5000),
         "sib": (5000, 9000), "air": (9000, 16000)}


@dataclass
class VocalAnalysis:
    duration: float = 0.0
    active_ratio: float = 0.0
    peak_db: float = -120.0
    loudness_db: float = -60.0      # RMS по активным участкам, дБFS
    noise_floor_db: float = -90.0
    dyn_range_db: float = 0.0
    f0_min: float = 0.0
    f0_med: float = 0.0
    f0_max: float = 0.0
    voice_type: str = "Auto"
    key: KeyResult | None = None
    backing_key: KeyResult | None = None
    backing_tempo: tuple | None = None
    backing_tuning: float = 0.0
    pitch_dev_cents: float = 0.0
    comp_thr_db: float = -20.0
    sib_thr_db: float = -32.0
    sib_rel_db: float = -20.0
    eq_mud: float = 0.0             # дБ над линией тренда
    eq_pres: float = 0.0
    eq_slope: float = -4.5          # дБ/окт
    clip_ratio: float = 0.0
    breaths: int = 0                # число найденных вдохов
    has_voice: bool = True
    notes: list = field(default_factory=list)


def _octave_levels(x, sr):
    f, p = welch(x, fs=sr, nperseg=8192)
    cents = [125, 250, 500, 1000, 2000, 4000, 8000]
    out = []
    for c in cents:
        a, b = c / np.sqrt(2), c * np.sqrt(2)
        m = (f >= a) & (f < b)
        out.append(10 * np.log10(p[m].sum() + 1e-18))
    return np.array(cents), np.array(out)


def analyze_vocal(x: np.ndarray, sr: int, track: PitchTrack | None = None, backing: np.ndarray | None = None,
                  progress=None, cancel=None) -> VocalAnalysis:
    """x — моно-сигнал голоса."""
    va = VocalAnalysis()
    x = np.asarray(x, dtype=np.float64)
    va.duration = len(x) / sr
    if len(x) < sr // 2 or np.abs(x).max() < 1e-4:
        va.has_voice = False
        va.notes.append("Запись слишком короткая или тихая для анализа.")
        return va
    va.peak_db = float(20 * np.log10(np.abs(x).max() + 1e-12))
    va.clip_ratio = float(np.mean(np.abs(x) > 0.995))
    if track is None:
        track = detect_pitch(x, sr, 60, 1200, tracking=0.22, progress=progress, cancel=cancel)
    if progress:
        progress("analysis", 0.5)

    rms = track.rms_db
    va.noise_floor_db = float(np.percentile(rms, 8))
    peak_rms = float(np.percentile(rms, 98))
    thr_active = max(va.noise_floor_db + 9.0, peak_rms - 40.0)
    active = rms > thr_active
    if active.sum() < 20:          # ровный сигнал без пауз — нет «шумового пола»
        active = rms > peak_rms - 30.0
        va.noise_floor_db = float(min(va.noise_floor_db, peak_rms - 45.0))
    va.active_ratio = float(active.mean())
    if active.sum() < 20 or track.voiced.sum() < 20:
        va.has_voice = False
        va.notes.append("В записи не найдено пение/речь — анализировать нечего.")
        return va
    va.loudness_db = float(10 * np.log10(np.mean(10 ** (rms[active] / 10.0)) + 1e-12))
    va.dyn_range_db = float(np.percentile(rms[active], 92) - np.percentile(rms[active], 12))

    f0 = track.f0[track.voiced]
    va.f0_min, va.f0_med, va.f0_max = (float(np.percentile(f0, 3)), float(np.median(f0)), float(np.percentile(f0, 97)))
    va.voice_type = "Soprano" if va.f0_med >= 300 else ("Alto/Tenor" if va.f0_med >= 140 else "Low Male")

    # тональность голоса
    va.key = detect_key_from_pitch(track)
    if backing is not None:
        try:
            va.backing_key = detect_key_from_audio(backing, sr)
            va.backing_tempo = detect_tempo(backing, sr)
            if va.backing_key is not None:
                va.backing_tuning = va.backing_key.tuning_cents
        except Exception:
            pass

    # отклонение от нот выбранной тональности
    key = va.backing_key if (va.backing_key is not None and va.backing_key.confident) else va.key
    if key is not None:
        mask = scale_mask(key.key, key.scale)
        m = track.midi(440.0) - key.tuning_cents / 100.0
        from .dsp.pitch import _runs
        devs, ws = [], []
        allowed = np.array([n for n in range(24, 108) if mask[n % 12]], dtype=float)
        for s, e in _runs(track.voiced):
            if e - s < 8:
                continue
            c = uniform_filter1d(m[s:e], size=min(26, e - s), mode="nearest")
            nearest = allowed[np.abs(allowed[None, :] - c[:, None]).argmin(axis=1)]
            devs.append(np.abs(c - nearest) * 100.0)
        if devs:
            va.pitch_dev_cents = float(np.mean(np.concatenate(devs)))

    # спектр активных участков
    hop = track.hop
    idx = np.nonzero(active)[0]
    seg_list = [x[max(0, i * hop - hop // 2): i * hop + hop // 2] for i in idx[:: max(1, len(idx) // 2500)]]
    xa = np.concatenate(seg_list) if seg_list else x
    cents, lev = _octave_levels(xa, sr)
    # линия тренда по 250 Гц … 4 кГц (без краёв)
    sel = slice(1, 6)
    lx = np.log2(cents[sel])
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef = np.linalg.lstsq(A, lev[sel], rcond=None)[0]
    va.eq_slope = float(coef[0])
    fit = coef[0] * np.log2(cents) + coef[1]
    va.eq_mud = float(lev[1] - fit[1])      # октава ~250 Гц
    va.eq_pres = float(lev[5] - fit[5])     # октава ~4 кГц
    try:
        va.breaths = estimate_breaths(track, x, sr)
    except Exception:
        va.breaths = 0

    # динамика: порог компрессора и сибилянты — теми же измерителями, что у плагинов
    L, B = dynamics._level_db(dynamics._detector(xa[None, :], "rms"), sr, "rms")
    va.comp_thr_db = float(np.percentile(L[L > L.max() - 45], 55)) if len(L) else -20.0
    sos = bandpass_sos(sr, 6500.0, 1.6)
    band = sosfilt(sos, xa)
    Lb, _ = dynamics._level_db(band, sr, "peak")
    Lb_act = Lb[Lb > Lb.max() - 40]
    va.sib_thr_db = float(np.percentile(Lb_act, 72)) if len(Lb_act) else -32.0
    va.sib_rel_db = float(np.percentile(Lb_act, 95) - va.loudness_db) if len(Lb_act) else -20.0
    return va


def estimate_breaths(track: PitchTrack, x: np.ndarray, sr: int) -> int:
    """Грубый подсчёт вдохов: долгие безголосые участки умеренной громкости без «свистящего» спектра."""
    from .dsp.pitch import _runs
    from scipy.signal import butter
    voiced, rms = track.voiced, track.rms_db
    if voiced.sum() < 10:
        return 0
    hop, F = track.hop, len(track.f0)
    lvl_v = float(np.percentile(rms[voiced], 60))
    floor = float(np.percentile(rms, 8))
    hp = sosfilt(butter(2, 5000.0 / (0.5 * sr), "high", output="sos"), x)
    n = len(x)
    pad = max(0, F * hop - n)
    e_all = np.pad(x * x, (0, pad))[: F * hop].reshape(F, hop).mean(axis=1) + 1e-12
    e_hf = np.pad(hp * hp, (0, pad))[: F * hop].reshape(F, hop).mean(axis=1) + 1e-12
    cand = (~voiced) & (rms < lvl_v - 7.0) & (rms > floor + 4.0) & (e_hf / e_all < 0.6)
    min_len = int(0.16 / (hop / sr))
    return sum(1 for s_, e_ in _runs(cand) if e_ - s_ >= min_len)


# --------------------------------------------------------------- Auto-Mix ---

STYLES = {
    "Natural": dict(
        rider=dict(range=4), breath=8, exciter=None, doubler=None,
        label="Естественный", desc="Прозрачная обработка, голос звучит как есть, только чище и ровнее.",
        retune=60, humanize=50, flex=30, vibrato=100, ratio_bias=-0.5, comp_attack=14, comp_release=180,
        sat=None, delay=None, reverb=dict(type="Room", decay=0.9, mix=9, predelay=15),
        presence=0.0, air=0.0, vs_backing=-2.0),
    "Pop": dict(
        rider=dict(range=5), breath=10, exciter=dict(amount=18, freq=6000), doubler=dict(mix=20, voices=2, detune=8),
        label="Поп", desc="Современный блестящий вокал: лёгкий автотюн, воздух, плотная динамика, эхо и реверберация.",
        retune=30, humanize=35, flex=15, vibrato=100, ratio_bias=0.5, comp_attack=10, comp_release=140,
        sat=None, delay=dict(division="1/8", feedback=24, mix=9, pingpong=50),
        reverb=dict(type="Plate", decay=1.5, mix=15, predelay=25), presence=1.0, air=2.0, vs_backing=-1.5),
    "Hard Tune": dict(
        rider=None, breath=10, exciter=dict(amount=22, freq=5000), doubler=None,
        label="Жёсткий автотюн / Trap", desc="Эффект «робот» (T-Pain): мгновенная коррекция, яркий плотный вокал.",
        retune=0, humanize=0, flex=0, vibrato=0, ratio_bias=1.0, comp_attack=5, comp_release=90,
        sat=dict(drive=6, mix=20), delay=dict(division="1/8", feedback=30, mix=14, pingpong=90),
        reverb=dict(type="Plate", decay=1.2, mix=12, predelay=20), presence=1.5, air=2.5, vs_backing=-1.0),
    "Rap": dict(
        rider=dict(range=6), breath=12, exciter=dict(amount=20, freq=5500), doubler=dict(mix=16, voices=1, detune=7, delay=22),
        label="Рэп", desc="Плотный и яркий вокал «в лицо»: сильная компрессия, насыщение, короткое пространство.",
        retune=15, humanize=10, flex=0, vibrato=40, ratio_bias=1.5, comp_attack=5, comp_release=80, comp_detect="Peak",
        sat=dict(drive=8, mix=25), delay=dict(division="1/8", feedback=20, mix=10, pingpong=40),
        reverb=dict(type="Room", decay=0.9, mix=8, predelay=10), presence=1.5, air=2.0, vs_backing=0.0),
    "Ballad": dict(
        rider=dict(range=5), breath=6, exciter=dict(amount=12, freq=6500), doubler=None,
        label="Баллада", desc="Мягкий вокал в большом зале: нежная коррекция, длинная реверберация.",
        retune=55, humanize=55, flex=30, vibrato=105, ratio_bias=-0.3, comp_attack=15, comp_release=200,
        sat=None, delay=dict(division="1/4", feedback=28, mix=10, pingpong=40),
        reverb=dict(type="Hall", decay=3.0, mix=23, predelay=40, damping=60), presence=0.5, air=1.5, vs_backing=-2.5),
    "Dance": dict(
        rider=None, breath=10, exciter=dict(amount=24, freq=5500), doubler=dict(mix=26, voices=2, detune=11, width=100),
        label="Танцевальный / EDM", desc="Яркий, плотный вокал с ритмичным пинг-понг эхо.",
        retune=20, humanize=20, flex=5, vibrato=80, ratio_bias=1.0, comp_attack=6, comp_release=100,
        sat=None, delay=dict(division="1/4", feedback=35, mix=18, pingpong=100),
        reverb=dict(type="Plate", decay=2.0, mix=19, predelay=30), presence=1.0, air=3.0, vs_backing=-1.0),
    "Clean": dict(
        rider=dict(range=4), breath=8, exciter=None, doubler=None,
        label="Только чистка", desc="Без автотюна и эффектов: убрать шум и гул, выровнять громкость.",
        retune=None, humanize=40, flex=0, vibrato=100, ratio_bias=-0.5, comp_attack=14, comp_release=180,
        sat=None, delay=None, reverb=None, presence=0.0, air=0.0, vs_backing=-1.5),
}
STYLE_ORDER = list(STYLES.keys())


def build_vocal_chain(va: VocalAnalysis, style: str, opts: dict, bpm_default: float = 120.0):
    """-> (plugins: list[dict], settings: dict, report: list[str]). opts: tune, eq, dynamics, deess, space, gate."""
    from .plugins import new_plugin
    st = STYLES[style]
    plugins, report = [], []
    opts = {"tune": True, "eq": True, "dynamics": True, "deess": True, "space": True, "gate": True, **(opts or {})}

    # --- gate
    if opts["gate"] and va.noise_floor_db > -62.0:
        thr = float(np.clip(va.noise_floor_db + 9.0, -70, -35))
        plugins.append(new_plugin("gate", {"threshold": round(thr, 1), "range": 22.0, "release": 150.0}))
        report.append(f"Шум в паузах заметен ({va.noise_floor_db:.0f} дБ) — добавлен Noise Gate (порог {thr:.0f} дБ).")
    elif opts["gate"]:
        report.append(f"Запись чистая (шум {va.noise_floor_db:.0f} дБ) — Noise Gate не нужен.")

    # --- breath
    if opts["gate"] and st.get("breath") and va.breaths >= 2:
        red = float(np.clip(st["breath"] + (2.0 if va.breaths >= 5 else 0.0), 4.0, 20.0))
        plugins.append(new_plugin("breath", {"reduction": red}))
        report.append(f"Найдено вдохов: {va.breaths} — Breath Control приглушит их на {red:.0f} дБ.")

    # --- tune
    if opts["tune"] and st["retune"] is not None:
        key = None
        src = ""
        if va.backing_key is not None and va.backing_key.confident:
            key, src = va.backing_key, "подложке"
        elif va.key is not None and va.key.confident:
            key, src = va.key, "вашему пению"
        elif va.key is not None:
            key, src = va.key, "вашему пению (неуверенно)"
        retune = float(st["retune"])
        if st["retune"] > 0:
            if va.pitch_dev_cents < 14:
                retune += 25
            elif va.pitch_dev_cents > 40:
                retune = max(8.0, retune - 10)
        ov = {"retune": retune, "humanize": st["humanize"], "flex": st["flex"], "vibrato": st["vibrato"],
              "input_type": va.voice_type}
        if va.backing_key is not None and abs(va.backing_tuning) > 8 and va.backing_key.confident:
            ov["ref_a"] = round(440.0 * 2 ** (va.backing_tuning / 1200.0), 1)
            report.append(f"Подложка строит на {ov['ref_a']} Гц (отклонение {va.backing_tuning:+.0f} цент) — автотюн настроен под неё.")
        if key is not None and (key.confident or src.startswith("подложке")):
            ov["key"], ov["scale"] = NOTE_NAMES[key.key], key.scale
            report.append(f"Тональность: {key.name} (по {src}, уверенность {key.confidence_label}). Автотюн тянет к нотам этого лада.")
        else:
            ov["scale"] = "Chromatic"
            report.append("Тональность определить не удалось уверенно — автотюн в режиме Chromatic (подтягивает к ближайшей ноте).")
        plugins.append(new_plugin("tune", ov))
        report.append(f"Фальшь: в среднем {va.pitch_dev_cents:.0f} цент от нот лада → Retune Speed {retune:.0f} мс.")

    # --- eq
    if opts["eq"]:
        hpf = float(np.clip(0.72 * va.f0_min, 70, 150))
        eq = {"hpf_f": round(hpf), "hpf_on": True}
        notes = [f"HPF {hpf:.0f} Гц (убирает гул ниже голоса)"]
        if va.eq_mud > 2.0:
            g = -float(min(4.0, (va.eq_mud - 1.0) * 0.8))
            eq.update(b1_f=290.0, b1_g=round(g, 1), b1_q=1.2)
            notes.append(f"{g:.1f} дБ на 290 Гц (убрать «бубнёж»)")
        pres = va.eq_pres + 0.0
        if pres < -2.0 or st["presence"] > 0:
            g = float(min(3.5, max(0.0, (-pres - 1.0) * 0.7) + st["presence"]))
            if g > 0.4:
                eq.update(b2_f=3000.0, b2_g=round(g, 1), b2_q=1.0)
                notes.append(f"+{g:.1f} дБ на 3 кГц (разборчивость)")
        air = 0.0
        if va.eq_slope < -7.5:
            air += min(3.0, (-7.5 - va.eq_slope) * 0.6)
        air += st["air"]
        air = float(min(5.0, air))
        if air > 0.4:
            eq.update(hs_f=10000.0, hs_g=round(air, 1))
            notes.append(f"+{air:.1f} дБ выше 10 кГц («воздух»)")
        plugins.append(new_plugin("eq", eq))
        report.append("EQ: " + "; ".join(notes) + ".")

    # --- rider (перед компрессором: сначала ровняем фразы, потом сжимаем пики)
    if opts["dynamics"] and st.get("rider") and va.dyn_range_db > 12.0:
        rg = float(np.clip(st["rider"]["range"] + (va.dyn_range_db - 14.0) * 0.3, 3.0, 10.0))
        plugins.append(new_plugin("rider", {"range": round(rg, 1), "target": round(float(np.clip(va.loudness_db, -32.0, -12.0)), 1)}))
        report.append(f"Vocal Rider: фразы разной громкости ({va.dyn_range_db:.0f} дБ) — буду подравнивать в пределах ±{rg:.0f} дБ.")

    # --- compressor
    if opts["dynamics"]:
        dr = va.dyn_range_db
        ratio = (4.2 if dr >= 20 else 3.2 if dr >= 14 else 2.3) + st["ratio_bias"]
        ratio = float(np.clip(ratio, 1.8, 8.0))
        thr = float(np.clip(va.comp_thr_db, -45, -8))
        cp = {"threshold": round(thr, 1), "ratio": round(ratio, 1), "attack": st["comp_attack"],
              "release": st["comp_release"], "auto_makeup": True, "makeup": 0.0, "knee": 8.0,
              "detect": st.get("comp_detect", "RMS")}
        plugins.append(new_plugin("comp", cp))
        report.append(f"Компрессор: динамика голоса {dr:.0f} дБ → порог {thr:.0f} дБ, сжатие {ratio:.1f}:1.")

    # --- de-esser
    if opts["deess"]:
        rng = float(np.clip(6.0 + (va.sib_rel_db + 22.0) * 0.5, 4.0, 12.0))
        thr = float(np.clip(va.sib_thr_db, -55, -15))
        plugins.append(new_plugin("deess", {"threshold": round(thr, 1), "range": round(rng, 1),
                                           "freq": 7500.0 if va.f0_med >= 260 else 6500.0}))
        report.append(f"De-esser: резкость «с» {'высокая' if va.sib_rel_db > -17 else 'умеренная'} → подавление до {rng:.0f} дБ.")

    # --- насыщение / пространство
    if opts["space"]:
        if st.get("exciter"):
            plugins.append(new_plugin("exciter", st["exciter"]))
            report.append("Exciter: добавлен блеск и «воздух» в верхах.")
        if st["sat"]:
            plugins.append(new_plugin("sat", st["sat"]))
        if st.get("doubler"):
            plugins.append(new_plugin("doubler", st["doubler"]))
            report.append("Doubler: голос расширен лёгким «двойным дублем».")
        if st["delay"]:
            bpm = bpm_default
            if va.backing_tempo is not None and va.backing_tempo[1] > 0.1:
                bpm = round(va.backing_tempo[0])
            d = dict(st["delay"], bpm=float(bpm), mode="Sync")
            plugins.append(new_plugin("delay", d))
            report.append(f"Delay синхронизирован с темпом {bpm:.0f} BPM." if va.backing_tempo else
                          f"Delay: темп по умолчанию {bpm:.0f} BPM (загрузите подложку — определю темп сам).")
        if st["reverb"]:
            plugins.append(new_plugin("reverb", st["reverb"]))
            report.append(f"Reverb: {st['reverb']['type']}, {st['reverb']['decay']} с.")
    settings = {"vs_backing": st["vs_backing"]}
    return plugins, settings, report


def active_loudness_db(x: np.ndarray, sr: int) -> float:
    """Громкость активных участков (дБFS, по RMS в окнах 50 мс, гейт −35 дБ от максимума)."""
    if x.ndim == 2:
        x = np.sqrt(np.mean(x * x, axis=0))
        x2 = x * x
    else:
        x2 = x * x
    w = max(1, int(0.05 * sr))
    n = (len(x2) // w) * w
    if n == 0:
        return -90.0
    fr = x2[:n].reshape(-1, w).mean(axis=1)
    db = 10 * np.log10(fr + 1e-12)
    keep = db > db.max() - 35.0
    return float(10 * np.log10(np.mean(fr[keep]) + 1e-12))
