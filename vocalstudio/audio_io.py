"""Ввод/вывод звука: запись с микрофона, воспроизведение, импорт/экспорт файлов."""
from __future__ import annotations

import os
import threading
import time
from math import gcd

import numpy as np
from scipy.signal import resample_poly

try:  # sounddevice тянет PortAudio; без него приложение остаётся рабочим для файлов
    import sounddevice as sd
    SD_ERROR = None
except Exception as e:  # pragma: no cover
    sd = None
    SD_ERROR = str(e)

try:
    import soundfile as sf
except Exception as e:  # pragma: no cover
    sf = None

SR = 44100  # рабочая частота проекта


# ----------------------------------------------------------------- файлы ---

def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.shape[-1] == 0:
        return x
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g, axis=-1)


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Читает файл → (каналы, n) float32, приведено к sr."""
    if sf is None:
        raise RuntimeError("soundfile недоступен")
    data, fs = sf.read(path, dtype="float32", always_2d=True)
    data = data.T  # (ch, n)
    if fs != sr:
        data = resample(data, fs, sr).astype(np.float32)
    return np.ascontiguousarray(data)


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.vstack([x, x])
    if x.shape[0] == 1:
        return np.vstack([x[0], x[0]])
    return x[:2]


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(axis=0)


EXPORT_FORMATS = {
    "WAV 24-bit": ("WAV", "PCM_24", ".wav"),
    "WAV 16-bit": ("WAV", "PCM_16", ".wav"),
    "WAV 32-bit float": ("WAV", "FLOAT", ".wav"),
    "FLAC 24-bit": ("FLAC", "PCM_24", ".flac"),
    "OGG Vorbis": ("OGG", "VORBIS", ".ogg"),
    "MP3": ("MP3", "MPEG_LAYER_III", ".mp3"),
}


def save_audio(path: str, data: np.ndarray, sr: int, fmt_name: str = "WAV 24-bit", dither: bool = True):
    """data: (ch, n) float. Пишет файл выбранного формата."""
    if sf is None:
        raise RuntimeError("soundfile недоступен")
    fmt, sub, _ = EXPORT_FORMATS[fmt_name]
    x = np.asarray(data, dtype=np.float64).T
    if sub == "PCM_16" and dither:
        rng = np.random.default_rng(0)
        x = x + (rng.random(x.shape) - rng.random(x.shape)) / 32768.0
    x = np.clip(x, -1.0, 1.0)
    sf.write(path, x, sr, format=fmt, subtype=sub)


def takes_dir() -> str:
    base = os.path.join(os.path.expanduser("~"), "Documents", "VocalStudio", "Takes")
    try:
        os.makedirs(base, exist_ok=True)
    except OSError:
        base = os.path.join(os.path.expanduser("~"), "VocalStudio", "Takes")
        os.makedirs(base, exist_ok=True)
    return base


# ------------------------------------------------------------- устройства ---

def list_devices():
    """→ (inputs, outputs, default_in, default_out); элементы — (index, name)."""
    if sd is None:
        return [], [], None, None
    try:
        devs = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception:
        return [], [], None, None
    ins, outs = [], []
    try:
        d_in, d_out = sd.default.device
    except Exception:
        d_in = d_out = None
    for i, d in enumerate(devs):
        try:
            api = hostapis[d["hostapi"]]["name"]
        except Exception:
            api = ""
        # На Windows оставляем MME/DirectSound/WASAPI, но прячем WDM-KS (нестабилен)
        if "WDM-KS" in api:
            continue
        label = f"{d['name']}  [{api}]" if api else d["name"]
        if d["max_input_channels"] > 0:
            ins.append((i, label))
        if d["max_output_channels"] > 0:
            outs.append((i, label))
    if d_in is not None and d_in < 0:
        d_in = None
    if d_out is not None and d_out < 0:
        d_out = None
    return ins, outs, d_in, d_out


# ---------------------------------------------------------------- движок ---

class AudioEngine:
    """Воспроизведение + запись. Для записи с подложкой используется один
    дуплексный поток (sd.Stream) — вход и выход на одних часах, без рассинхрона."""

    def __init__(self, sr: int = SR):
        self.sr = sr
        self.in_dev = None
        self.out_dev = None
        self.stream = None
        self.mode = "idle"            # idle | play | record
        self.buffer = np.zeros((2, 0), dtype=np.float32)   # что играем
        self.pos = 0                  # текущая позиция воспроизведения, отсчёты
        self.loop = None              # (a, b) отсчёты или None
        self.monitor = False
        self.monitor_gain = 1.0
        self._rec_chunks: list[np.ndarray] = []
        self._rec_lock = threading.Lock()
        self.in_peak = 0.0
        self.out_peak = 0.0
        self.error = None
        self.rec_sr = sr
        self.latency_s = 0.0
        self.fs = sr                  # фактическая частота текущего потока
        self.finished = False
        self.warning = ""
        self.recording = False
        self._rec_count = 0

    # --- публичный API ------------------------------------------------
    @property
    def available(self) -> bool:
        return sd is not None

    @property
    def pos_seconds(self) -> float:
        return self.pos / float(self.fs)

    def set_buffer(self, buf: np.ndarray):
        self.buffer = buf   # атомарная подмена ссылки

    def stop(self):
        st = self.stream
        self.stream = None
        if st is not None:
            try:
                st.stop()
                st.close()
            except Exception:
                pass
        self.mode = "idle"
        self.in_peak = 0.0
        self.out_peak = 0.0

    def start_play(self, pos: int = 0):
        self.stop()
        if sd is None:
            raise RuntimeError("Аудио-подсистема недоступна: " + str(SD_ERROR))
        self.pos = int(pos)
        self.fs = self.sr
        self.finished = False
        self.mode = "play"
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sr, channels=2, dtype="float32",
                device=self.out_dev, callback=self._cb_play, blocksize=0, latency="low",
            )
            self.stream.start()
        except Exception:
            self.mode = "idle"
            self.stream = None
            raise

    def start_record(self, pos: int = 0, backing: np.ndarray | None = None, record: bool = True):
        """Запись с входа; backing — (2, n) подложка для проигрывания (опционально).
        record=False — только мониторинг/индикатор уровня (режим ARM)."""
        self.stop()
        if sd is None:
            raise RuntimeError("Аудио-подсистема недоступна: " + str(SD_ERROR))
        self.pos = int(pos)
        self.fs = self.sr
        self.warning = ""
        self.recording = bool(record)
        self._rec_count = 0
        self._backing = backing if backing is not None else np.zeros((2, 0), np.float32)
        with self._rec_lock:
            self._rec_chunks = []
        self.finished = False
        in_info = sd.query_devices(self.in_dev if self.in_dev is not None else sd.default.device[0], "input")
        self.rec_sr = self.sr
        opened = False
        try:
            self._open_duplex(self.sr)
            opened = True
        except Exception:
            pass
        if not opened:
            # устройство не принимает 44.1 кГц — открываем на родной частоте и ресэмплируем потом
            native = int(in_info["default_samplerate"])
            self.rec_sr = native
            self.fs = native
            if native != self.sr:
                self._backing = resample(self._backing, self.sr, native).astype(np.float32)
                self.pos = int(pos * native / self.sr)
            try:
                self._open_duplex(native)
                opened = True
            except Exception:
                pass
        if not opened:
            # дуплекс невозможен (вход и выход на разных API) — пишем без подложки
            self.rec_sr = int(in_info["default_samplerate"])
            self.fs = self.rec_sr
            self.warning = "Подложка не воспроизводится: вход и выход используют разные аудио-API."
            self._backing = np.zeros((2, 0), np.float32)
            st = sd.InputStream(samplerate=self.rec_sr, channels=1, dtype="float32",
                                device=self.in_dev, callback=self._cb_input_only)
            st.start()
            self.stream = st
            lat = st.latency
            self.latency_s = float(lat[0] if isinstance(lat, (tuple, list)) else lat)
        self.mode = "record" if record else "monitor"

    def stop_record(self) -> tuple[np.ndarray, float]:
        """Останавливает запись, возвращает (моно float32 @ self.sr, задержка петли, с)."""
        lat = self.latency_s
        self.stop()
        with self._rec_lock:
            chunks, self._rec_chunks = self._rec_chunks, []
        if not chunks:
            return np.zeros(0, np.float32), lat
        rec = np.concatenate(chunks).astype(np.float32)
        if self.rec_sr != self.sr:
            rec = resample(rec, self.rec_sr, self.sr).astype(np.float32)
        return rec, lat

    def live_tail(self, max_seconds: float = 12.0):
        """Последние max_seconds записи → (массив, смещение начала в отсчётах rec_sr)."""
        with self._rec_lock:
            chunks = list(self._rec_chunks)
            total = self._rec_count
        need = int(max_seconds * self.rec_sr)
        out, got = [], 0
        for c in reversed(chunks):
            out.append(c)
            got += len(c)
            if got >= need:
                break
        if not out:
            return np.zeros(0, np.float32), 0
        arr = np.concatenate(out[::-1])
        return arr, total - len(arr)

    # --- внутреннее ----------------------------------------------------
    def _open_duplex(self, fs: int, native: bool = False):
        in_dev = self.in_dev
        out_dev = self.out_dev
        st = None
        try:
            st = sd.Stream(
                samplerate=fs, channels=(1, 2), dtype="float32",
                device=(in_dev, out_dev), callback=self._cb_duplex, latency="low",
            )
            st.start()
            lat = st.latency
            self.latency_s = float(sum(lat) if isinstance(lat, (tuple, list)) else lat)
        except Exception:
            if st is not None:
                try:
                    st.close()
                except Exception:
                    pass
            raise
        self.stream = st

    def _cb_input_only(self, indata, frames, time_info, status):
        self.in_peak = max(self.in_peak * 0.9, float(np.abs(indata).max()) if frames else 0.0)
        if self.recording:
            with self._rec_lock:
                self._rec_chunks.append(indata[:, 0].copy())
                self._rec_count += frames
        self.pos += frames

    def _cb_play(self, outdata, frames, time_info, status):
        buf = self.buffer
        n = buf.shape[1]
        pos = self.pos
        out = outdata
        out[:] = 0
        i = 0
        while i < frames:
            if self.loop is not None and pos >= self.loop[1]:
                pos = self.loop[0]
            lim = n if self.loop is None else min(n, self.loop[1])
            take = min(frames - i, max(0, lim - pos))
            if take <= 0:
                self.finished = True
                break
            out[i:i + take, 0] = buf[0, pos:pos + take]
            out[i:i + take, 1] = buf[1, pos:pos + take]
            pos += take
            i += take
        self.pos = pos
        self.out_peak = max(self.out_peak * 0.9, float(np.abs(out).max()) if frames else 0.0)

    def _cb_duplex(self, indata, outdata, frames, time_info, status):
        self.in_peak = max(self.in_peak * 0.9, float(np.abs(indata).max()) if frames else 0.0)
        if self.recording:
            with self._rec_lock:
                self._rec_chunks.append(indata[:, 0].copy())
                self._rec_count += frames
        outdata[:] = 0
        bk = self._backing
        pos = self.pos
        if bk.shape[1] > pos:
            take = min(frames, bk.shape[1] - pos)
            outdata[:take, 0] = bk[0, pos:pos + take]
            outdata[:take, 1] = bk[1, pos:pos + take]
        if self.monitor:
            outdata[:, 0] += indata[:, 0] * self.monitor_gain
            outdata[:, 1] += indata[:, 0] * self.monitor_gain
        self.pos = pos + frames
        self.out_peak = max(self.out_peak * 0.9, float(np.abs(outdata).max()) if frames else 0.0)
