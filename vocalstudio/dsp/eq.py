"""Параметрический эквалайзер на бикавадах (RBJ Audio EQ Cookbook)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz, tf2sos


def _norm(b, a):
    b = np.asarray(b, dtype=np.float64) / a[0]
    a = np.asarray(a, dtype=np.float64) / a[0]
    return np.concatenate([b, a])[None, :]


def peak_sos(fs, f0, gain_db, q):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * min(f0, 0.49 * fs) / fs
    al = np.sin(w0) / (2.0 * max(q, 0.05))
    c = np.cos(w0)
    b = [1 + al * A, -2 * c, 1 - al * A]
    a = [1 + al / A, -2 * c, 1 - al / A]
    return _norm(b, a)


def lowshelf_sos(fs, f0, gain_db, q=0.707):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * min(f0, 0.49 * fs) / fs
    al = np.sin(w0) / (2.0 * max(q, 0.05))
    c = np.cos(w0)
    sa = 2 * np.sqrt(A) * al
    b = [A * ((A + 1) - (A - 1) * c + sa), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sa)]
    a = [(A + 1) + (A - 1) * c + sa, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sa]
    return _norm(b, a)


def highshelf_sos(fs, f0, gain_db, q=0.707):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * min(f0, 0.49 * fs) / fs
    al = np.sin(w0) / (2.0 * max(q, 0.05))
    c = np.cos(w0)
    sa = 2 * np.sqrt(A) * al
    b = [A * ((A + 1) + (A - 1) * c + sa), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sa)]
    a = [(A + 1) - (A - 1) * c + sa, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sa]
    return _norm(b, a)


def bandpass_sos(fs, f0, q):
    w0 = 2 * np.pi * min(f0, 0.49 * fs) / fs
    al = np.sin(w0) / (2.0 * max(q, 0.05))
    c = np.cos(w0)
    b = [al, 0.0, -al]
    a = [1 + al, -2 * c, 1 - al]
    return _norm(b, a)


def hp_sos(fs, f0, order=2):
    return butter(order, min(f0, 0.49 * fs) / (0.5 * fs), btype="high", output="sos")


def lp_sos(fs, f0, order=2):
    return butter(order, min(f0, 0.49 * fs) / (0.5 * fs), btype="low", output="sos")


SLOPES = {"12 dB/oct": 2, "24 dB/oct": 4, "48 dB/oct": 8}


def eq_sos(p: dict, fs: int) -> np.ndarray | None:
    secs = []
    if p.get("hpf_on", True):
        secs.append(hp_sos(fs, p.get("hpf_f", 80.0), SLOPES.get(p.get("hpf_slope", "12 dB/oct"), 2)))
    if abs(p.get("ls_g", 0.0)) > 0.01:
        secs.append(lowshelf_sos(fs, p.get("ls_f", 120.0), p.get("ls_g", 0.0)))
    for b in ("b1", "b2", "b3"):
        if abs(p.get(b + "_g", 0.0)) > 0.01:
            secs.append(peak_sos(fs, p.get(b + "_f", 1000.0), p.get(b + "_g", 0.0), p.get(b + "_q", 1.0)))
    if abs(p.get("hs_g", 0.0)) > 0.01:
        secs.append(highshelf_sos(fs, p.get("hs_f", 10000.0), p.get("hs_g", 0.0)))
    if p.get("lp_on", False):
        secs.append(lp_sos(fs, p.get("lp_f", 18000.0), 2))
    if not secs:
        return None
    return np.vstack(secs)


def eq_process(x: np.ndarray, p: dict, fs: int) -> np.ndarray:
    sos = eq_sos(p, fs)
    if sos is None:
        return x
    return sosfilt(sos, x, axis=-1)


def eq_response(p: dict, fs: int, freqs: np.ndarray) -> np.ndarray:
    """АЧХ в дБ на заданных частотах (для отрисовки в интерфейсе)."""
    sos = eq_sos(p, fs)
    if sos is None:
        return np.zeros_like(freqs)
    w = 2 * np.pi * freqs / fs
    _, h = sosfreqz(sos, worN=w)
    return 20 * np.log10(np.maximum(np.abs(h), 1e-6))
