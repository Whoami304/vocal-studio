"""Правка аудио на дорожке: чистые функции над массивами (каналы, n). Индексы — в отсчётах буфера."""
from __future__ import annotations

import numpy as np


def clamp_range(n: int, i0: int, i1: int) -> tuple[int, int]:
    i0, i1 = sorted((int(i0), int(i1)))
    return max(0, min(n, i0)), max(0, min(n, i1))


def copy_range(a: np.ndarray, i0: int, i1: int) -> np.ndarray:
    i0, i1 = clamp_range(a.shape[1], i0, i1)
    return a[:, i0:i1].copy()


def silence_range(a: np.ndarray, i0: int, i1: int, fade: int = 0) -> np.ndarray:
    """Заглушает участок, длительность не меняется. fade — короткое затухание на краях (без щелчков)."""
    i0, i1 = clamp_range(a.shape[1], i0, i1)
    b = a.copy()
    if i1 <= i0:
        return b
    b[:, i0:i1] = 0.0
    f = min(fade, (i1 - i0) // 2)
    if f > 1:
        ramp = np.linspace(1.0, 0.0, f, dtype=np.float32)
        # затухание перед участком и нарастание после него: прячем щелчок на стыке
        lo = max(0, i0 - f)
        k = i0 - lo
        if k > 1:
            b[:, lo:i0] = a[:, lo:i0] * np.linspace(1.0, 0.0, k, dtype=np.float32)
        hi = min(a.shape[1], i1 + f)
        k2 = hi - i1
        if k2 > 1:
            b[:, i1:hi] = a[:, i1:hi] * np.linspace(0.0, 1.0, k2, dtype=np.float32)
    return b


def delete_range(a: np.ndarray, i0: int, i1: int, xfade: int = 96) -> np.ndarray:
    """Вырезает участок и смыкает края (ripple). Длина результата ровно n - (i1 - i0).
    На стыке — короткий кроссфейд из хвоста слева в звук прямо перед правой границей: щелчка нет."""
    i0, i1 = clamp_range(a.shape[1], i0, i1)
    if i1 <= i0:
        return a.copy()
    if i0 == 0:
        return a[:, i1:].copy()
    if i1 >= a.shape[1]:
        return a[:, :i0].copy()
    x = min(xfade, i0, i1, a.shape[1] - i1)
    if x > 1:
        w = np.linspace(0.0, 1.0, x, dtype=np.float32)
        mix = a[:, i0 - x:i0] * (1.0 - w) + a[:, i1 - x:i1] * w
        return np.concatenate([a[:, :i0 - x], mix, a[:, i1:]], axis=1)
    return np.concatenate([a[:, :i0], a[:, i1:]], axis=1)


def crop(a: np.ndarray, i0: int, i1: int) -> tuple[np.ndarray, int]:
    """Оставляет только участок. Возвращает (аудио, сдвиг начала относительно старого буфера)."""
    i0, i1 = clamp_range(a.shape[1], i0, i1)
    return a[:, i0:i1].copy(), i0


def fade(a: np.ndarray, i0: int, i1: int, kind: str = "in", curve: str = "cos") -> np.ndarray:
    i0, i1 = clamp_range(a.shape[1], i0, i1)
    b = a.copy()
    n = i1 - i0
    if n < 2:
        return b
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    g = (0.5 - 0.5 * np.cos(np.pi * t)) if curve == "cos" else t
    if kind == "out":
        g = g[::-1]
    b[:, i0:i1] = b[:, i0:i1] * g
    return b


def gain_db(a: np.ndarray, i0: int, i1: int, db: float) -> np.ndarray:
    i0, i1 = clamp_range(a.shape[1], i0, i1)
    b = a.copy()
    b[:, i0:i1] *= 10.0 ** (db / 20.0)
    return b


def normalize(a: np.ndarray, i0: int, i1: int, peak_db: float = -1.0) -> tuple[np.ndarray, float]:
    """Нормализует участок до заданного пика. Возвращает (аудио, применённое усиление, дБ)."""
    i0, i1 = clamp_range(a.shape[1], i0, i1)
    seg = a[:, i0:i1]
    pk = float(np.abs(seg).max()) if seg.size else 0.0
    if pk < 1e-6:
        return a.copy(), 0.0
    g = 10.0 ** (peak_db / 20.0) / pk
    return gain_db(a, i0, i1, 20.0 * np.log10(g)), 20.0 * np.log10(g)


def reverse(a: np.ndarray, i0: int, i1: int) -> np.ndarray:
    i0, i1 = clamp_range(a.shape[1], i0, i1)
    b = a.copy()
    b[:, i0:i1] = b[:, i0:i1][:, ::-1]
    return b


def trim_silence(a: np.ndarray, thr_db: float = -55.0, keep_ms: float = 150.0, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Убирает тишину в начале и конце. Возвращает (аудио, сдвиг начала)."""
    lvl = np.abs(a).max(axis=0)
    idx = np.nonzero(lvl > 10.0 ** (thr_db / 20.0))[0]
    if len(idx) == 0:
        return a.copy(), 0
    keep = int(keep_ms * 0.001 * sr)
    i0 = max(0, int(idx[0]) - keep)
    i1 = min(a.shape[1], int(idx[-1]) + keep)
    return a[:, i0:i1].copy(), i0


def match_channels(clip: np.ndarray, channels: int) -> np.ndarray:
    if clip.shape[0] == channels:
        return clip
    if channels == 1:
        return clip.mean(axis=0, keepdims=True)
    return np.vstack([clip[0], clip[0]])


def mix_into(a: np.ndarray | None, pos: int, clip: np.ndarray, start: int = 0) -> tuple[np.ndarray, int]:
    """Накладывает clip на таймлайн в позиции pos (отсчёты) поверх буфера a, начинающегося с `start`.
    Возвращает (новый буфер, новый start)."""
    if a is None or a.shape[1] == 0:
        return clip.copy(), int(pos)
    clip = match_channels(clip, a.shape[0])
    new_start = min(start, pos)
    new_end = max(start + a.shape[1], pos + clip.shape[1])
    out = np.zeros((a.shape[0], new_end - new_start), dtype=np.float32)
    out[:, start - new_start: start - new_start + a.shape[1]] = a
    out[:, pos - new_start: pos - new_start + clip.shape[1]] += clip
    return out, new_start


def overwrite_into(a: np.ndarray | None, pos: int, clip: np.ndarray, start: int = 0) -> tuple[np.ndarray, int]:
    """Записывает clip поверх буфера (punch-in): прежний звук на этом участке заменяется."""
    if a is None or a.shape[1] == 0:
        return clip.copy(), int(pos)
    clip = match_channels(clip, a.shape[0])
    new_start = min(start, pos)
    new_end = max(start + a.shape[1], pos + clip.shape[1])
    out = np.zeros((a.shape[0], new_end - new_start), dtype=np.float32)
    out[:, start - new_start: start - new_start + a.shape[1]] = a
    out[:, pos - new_start: pos - new_start + clip.shape[1]] = clip
    return out, new_start
