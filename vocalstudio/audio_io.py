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
        raise RuntimeError("Библиотека чтения аудио (soundfile) недоступна")
    try:
        data, fs = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        ext = os.path.splitext(path)[1].lower()
        hint = ""
        if ext in (".m4a", ".aac", ".wma", ".mp4", ".opus", ".amr"):
            hint = f"\nФормат {ext} не поддерживается. Сконвертируйте файл в WAV, FLAC или MP3 и откройте снова."
        raise RuntimeError(f"Не удалось прочитать файл «{os.path.basename(path)}»: {e}{hint}")
    if data.shape[0] == 0:
        raise RuntimeError(f"Файл «{os.path.basename(path)}» пустой")
    data = np.nan_to_num(data.T)  # (ch, n)
    if fs != sr:
        data = resample(data, fs, sr).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


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

class AudioError(Exception):
    """Ошибка аудио-устройства с понятным пользователю текстом."""


def friendly_error(e: Exception) -> str:
    t = str(e)
    low = t.lower()
    tips = []
    if "sample rate" in low:
        tips.append("Устройство не поддерживает частоту 44100 Гц. Попробуйте другое устройство в «Микрофон / звук…» "
                    "или в настройках Windows поставьте формат 44100 или 48000 Гц.")
    if "channels" in low:
        tips.append("Устройство не поддерживает нужное число каналов. Выберите другое устройство.")
    if "unavailable" in low or "device" in low and "invalid" in low:
        tips.append("Устройство недоступно или занято другой программой. Закройте программы, использующие звук, "
                    "или выберите другое устройство.")
    if "exclusive" in low or "-9985" in low:
        tips.append("Устройство занято в монопольном режиме другой программой.")
    if "unanticipated host error" in low or "-9999" in low:
        tips.append("Ошибка драйвера Windows. Проверьте, что устройство включено, и выберите его вручную в «Микрофон / звук…».")
    if "permission" in low or "access" in low:
        tips.append("Windows может запрещать доступ к микрофону: Параметры → Конфиденциальность → Микрофон → разрешить "
                    "доступ классическим приложениям.")
    return t + ("\n\n" + "\n".join(tips) if tips else "")


def _default_dev(kind: str):
    """Индекс устройства по умолчанию ('input'|'output') или None."""
    if sd is None:
        return None
    try:
        d = sd.default.device
        v = d[0] if kind == "input" else d[1]
        if v is not None and v >= 0:
            return int(v)
    except Exception:
        pass
    try:
        return int(sd.query_devices(kind=kind)["index"])
    except Exception:
        return None


class AudioEngine:
    """Воспроизведение + запись. Для записи с подложкой используется один дуплексный поток
    (sd.Stream) — вход и выход на одних часах. Все режимы открытия имеют запасные варианты."""

    def __init__(self, sr: int = SR):
        self.sr = sr
        self.in_dev = None
        self.out_dev = None
        self.stream = None
        self.mode = "idle"            # idle | play | record | monitor
        self.buffer = np.zeros((2, 0), dtype=np.float32)   # что играем (на частоте проекта)
        self._pbuf = self.buffer                           # то же на частоте потока
        self.pos = 0                  # позиция в отсчётах ПОТОКА (self.fs)
        self.loop = None              # (a, b) в отсчётах ПОТОКА
        self.monitor = False
        self.monitor_gain = 1.0
        self._rec_chunks: list[np.ndarray] = []
        self._rec_lock = threading.Lock()
        self.in_peak = 0.0
        self.out_peak = 0.0
        self.error = None
        self.rec_sr = sr
        self.latency_s = 0.0
        self.fs = sr                  # частота текущего потока
        self.finished = False
        self.warning = ""
        self.recording = False
        self._rec_count = 0
        self._backing = np.zeros((2, 0), np.float32)
        self._out_ch = 2
        self.last_error = ""
        # метроном / отсчёт (генерируются прямо в аудио-колбэке)
        self.click_on = False
        self.click_bpm = 120.0
        self._count_in = 0            # длина отсчёта в отсчётах потока (запись начинается после него)
        self._rec_origin = 0          # позиция начала записи на таймлайне, отсчёты потока (для фазы метронома)
        t = np.arange(int(0.03 * SR)) / SR
        self._click_hi = (np.sin(2 * np.pi * 1600 * t) * np.exp(-t / 0.008) * 0.45).astype(np.float32)
        self._click_lo = (np.sin(2 * np.pi * 1000 * t) * np.exp(-t / 0.008) * 0.35).astype(np.float32)

    # --- публичный API ------------------------------------------------
    @property
    def available(self) -> bool:
        return sd is not None

    @property
    def pos_seconds(self) -> float:
        return self.pos / float(self.fs)

    def seek_seconds(self, t: float):
        self.pos = max(0, int(t * self.fs))

    def set_loop_seconds(self, a: float | None, b: float | None = None):
        if a is None or b is None or b <= a:
            self.loop = None
        else:
            self.loop = (int(a * self.fs), int(b * self.fs))

    def set_buffer(self, buf: np.ndarray):
        """Подменяет проигрываемый буфер (во время воспроизведения — без остановки)."""
        self.buffer = buf
        if self.fs == self.sr:
            self._pbuf = buf
        else:
            self._pbuf = resample(buf, self.sr, self.fs).astype(np.float32)

    def stop(self):
        st = self.stream
        self.stream = None
        if st is not None:
            try:
                st.stop()
            except Exception:
                pass
            try:
                st.close()
            except Exception:
                pass
        self.mode = "idle"
        self.in_peak = 0.0
        self.out_peak = 0.0

    # --- воспроизведение ----------------------------------------------
    def _native_rate(self, kind: str) -> int:
        try:
            dev = self.out_dev if kind == "output" else self.in_dev
            if dev is None:
                dev = _default_dev(kind)
            return int(sd.query_devices(dev)["default_samplerate"])
        except Exception:
            return self.sr

    def _max_channels(self, kind: str) -> int:
        try:
            dev = self.out_dev if kind == "output" else self.in_dev
            if dev is None:
                dev = _default_dev(kind)
            d = sd.query_devices(dev)
            return int(d["max_output_channels" if kind == "output" else "max_input_channels"])
        except Exception:
            return 2

    def start_play(self, pos_samples: int = 0):
        """pos_samples — позиция в отсчётах проекта (self.sr)."""
        self.stop()
        if sd is None:
            raise AudioError("Аудио-подсистема недоступна: " + str(SD_ERROR))
        self.finished = False
        self.last_error = ""
        native = self._native_rate("output")
        out_ch = 1 if self._max_channels("output") == 1 else 2
        rates = [self.sr] + ([native] if native != self.sr else [])
        configs = []
        for fs in rates:
            configs += [(fs, "low", 0), (fs, "high", 0), (fs, "high", 2048)]
        errors = []
        for fs, lat, bs in configs:
            try:
                self.fs = fs
                self._out_ch = out_ch
                self._pbuf = self.buffer if fs == self.sr else resample(self.buffer, self.sr, fs).astype(np.float32)
                self.pos = int(pos_samples * fs / self.sr)
                self.mode = "play"
                st = sd.OutputStream(samplerate=fs, channels=out_ch, dtype="float32", device=self.out_dev,
                                     callback=self._cb_play, blocksize=bs, latency=lat)
                st.start()
                self.stream = st
                return
            except Exception as e:  # пробуем следующий вариант
                errors.append(f"{fs} Гц, latency={lat}, block={bs}: {e}")
                self.mode = "idle"
                try:
                    st.close()  # type: ignore[name-defined]
                except Exception:
                    pass
        self.fs = self.sr
        self.mode = "idle"
        raise AudioError("Не удалось открыть устройство вывода.\n" + friendly_error(Exception(errors[-1] if errors else "")) +
                         "\n\nПопытки:\n" + "\n".join(errors))

    # --- запись -------------------------------------------------------
    def start_record(self, pos_samples: int = 0, backing: np.ndarray | None = None, record: bool = True, count_in: int = 0):
        """Запись с входа; backing — (2, n) подложка для проигрывания (на частоте проекта).
        record=False — только мониторинг/индикатор уровня."""
        self.stop()
        if sd is None:
            raise AudioError("Аудио-подсистема недоступна: " + str(SD_ERROR))
        self.warning = ""
        self.last_error = ""
        self.recording = bool(record)
        self._rec_count = 0
        base_backing = backing if backing is not None else np.zeros((2, 0), np.float32)
        with self._rec_lock:
            self._rec_chunks = []
        self.finished = False
        if _default_dev("input") is None and self.in_dev is None:
            raise AudioError("Микрофон не найден. Подключите микрофон и выберите его в «Микрофон / звук…».")
        native_in = self._native_rate("input")
        native_out = self._native_rate("output")
        out_ch = 1 if self._max_channels("output") == 1 else 2
        rates = [self.sr]
        for r in (native_in, native_out):
            if r not in rates:
                rates.append(r)
        errors = []
        for fs in rates:
            for lat in ("low", "high"):
                st = None
                try:
                    self.fs = fs
                    self.rec_sr = fs
                    self._out_ch = out_ch
                    self._backing = base_backing if fs == self.sr else resample(base_backing, self.sr, fs).astype(np.float32)
                    self._count_in = int(count_in * fs / self.sr)
                    self._rec_origin = int(pos_samples * fs / self.sr)
                    # подложка передаётся начиная с позиции записи; отсчёт играет до неё
                    self.pos = 0
                    self.mode = "record" if record else "monitor"
                    st = sd.Stream(samplerate=fs, channels=(1, out_ch), dtype="float32", device=(self.in_dev, self.out_dev),
                                   callback=self._cb_duplex, latency=lat)
                    st.start()
                    self.stream = st
                    lt = st.latency
                    self.latency_s = float(sum(lt) if isinstance(lt, (tuple, list)) else lt)
                    return
                except Exception as e:
                    errors.append(f"{fs} Гц, latency={lat}: {e}")
                    self.mode = "idle"
                    if st is not None:
                        try:
                            st.close()
                        except Exception:
                            pass
        # дуплекс невозможен — пишем без подложки
        try:
            fs = native_in
            self.fs = self.rec_sr = fs
            self._backing = np.zeros((2, 0), np.float32)
            self._count_in = 0
            self.pos = 0
            self.mode = "record" if record else "monitor"
            st = sd.InputStream(samplerate=fs, channels=1, dtype="float32", device=self.in_dev, callback=self._cb_input_only)
            st.start()
            self.stream = st
            lt = st.latency
            self.latency_s = float(lt[0] if isinstance(lt, (tuple, list)) else lt)
            self.warning = ("Подложка не воспроизводится: вход и выход используют разные аудио-API или устройство "
                            "не поддерживает одновременную работу. Запись идёт без бита.")
            return
        except Exception as e:
            errors.append(f"только вход: {e}")
            self.mode = "idle"
            self.fs = self.sr
            raise AudioError("Не удалось открыть микрофон.\n" + friendly_error(Exception(errors[-1])) +
                             "\n\nПопытки:\n" + "\n".join(errors))

    def stop_record(self) -> tuple[np.ndarray, float]:
        """Останавливает запись, возвращает (моно float32 @ self.sr, задержка петли, с)."""
        lat = self.latency_s
        self.stop()
        with self._rec_lock:
            chunks, self._rec_chunks = self._rec_chunks, []
        if not chunks:
            return np.zeros(0, np.float32), lat
        rec = np.concatenate(chunks).astype(np.float32)
        ci = self._count_in
        if ci > 0:
            rec = rec[ci:]
        self._count_in = 0
        if self.rec_sr != self.sr:
            rec = resample(rec, self.rec_sr, self.sr).astype(np.float32)
        self.fs = self.sr
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

    def count_in_samples(self, bars: int = 1, beats_per_bar: int = 4) -> int:
        """Длина отсчёта в отсчётах частоты проекта."""
        return int(round(bars * beats_per_bar * 60.0 / max(30.0, self.click_bpm) * self.sr))

    def _add_click(self, out, pos_rel: int, frames: int):
        """Подмешивает щелчки метронома в блок out (frames × ch). pos_rel — позиция относительно доли 0."""
        beat = self.fs * 60.0 / max(30.0, self.click_bpm)
        L = len(self._click_hi)
        k0 = int(np.floor((pos_rel - L) / beat))
        k1 = int(np.ceil((pos_rel + frames) / beat))
        for k in range(k0, k1 + 1):
            start = int(round(k * beat))
            a = max(pos_rel, start)
            b = min(pos_rel + frames, start + L)
            if b <= a:
                continue
            wave = self._click_hi if (k % 4 == 0) else self._click_lo
            seg = wave[a - start: b - start]
            out[a - pos_rel: b - pos_rel, :] += seg[:, None]

    # --- колбэки (поток PortAudio): никогда не бросают исключений --------
    def _cb_input_only(self, indata, frames, time_info, status):
        try:
            self.in_peak = max(self.in_peak * 0.9, float(np.abs(indata).max()) if frames else 0.0)
            if self.recording:
                with self._rec_lock:
                    self._rec_chunks.append(indata[:, 0].copy())
                    self._rec_count += frames
            self.pos += frames
        except Exception as e:  # pragma: no cover
            self.last_error = repr(e)

    def _cb_play(self, outdata, frames, time_info, status):
        try:
            buf = self._pbuf
            n = buf.shape[1]
            pos = self.pos
            out = outdata
            out[:] = 0
            i = 0
            ch = out.shape[1]
            while i < frames:
                if self.loop is not None and pos >= self.loop[1]:
                    pos = self.loop[0]
                lim = n if self.loop is None else min(n, self.loop[1])
                take = min(frames - i, max(0, lim - pos))
                if take <= 0:
                    self.finished = True
                    break
                if ch == 1:
                    out[i:i + take, 0] = buf[0, pos:pos + take] * 0.5 + buf[1, pos:pos + take] * 0.5
                else:
                    out[i:i + take, 0] = buf[0, pos:pos + take]
                    out[i:i + take, 1] = buf[1, pos:pos + take]
                pos += take
                i += take
            if self.click_on:
                self._add_click(out, self.pos, frames)
            self.pos = pos
            self.out_peak = max(self.out_peak * 0.9, float(np.abs(out).max()) if frames else 0.0)
        except Exception as e:  # pragma: no cover
            self.last_error = repr(e)
            outdata[:] = 0

    def _cb_duplex(self, indata, outdata, frames, time_info, status):
        try:
            self.in_peak = max(self.in_peak * 0.9, float(np.abs(indata).max()) if frames else 0.0)
            if self.recording:
                with self._rec_lock:
                    self._rec_chunks.append(indata[:, 0].copy())
                    self._rec_count += frames
            outdata[:] = 0
            bk = self._backing
            pos = self.pos
            ch = outdata.shape[1]
            bpos = pos - self._count_in                     # позиция в подложке (до начала записи — тишина)
            if bpos < 0:
                skip = min(frames, -bpos)
                bpos2, off = 0, skip
            else:
                bpos2, off = bpos, 0
            if bk.shape[1] > bpos2 and off < frames:
                take = min(frames - off, bk.shape[1] - bpos2)
                if ch == 1:
                    outdata[off:off + take, 0] = 0.5 * (bk[0, bpos2:bpos2 + take] + bk[1, bpos2:bpos2 + take])
                else:
                    outdata[off:off + take, 0] = bk[0, bpos2:bpos2 + take]
                    outdata[off:off + take, 1] = bk[1, bpos2:bpos2 + take]
            if self.click_on or pos < self._count_in:
                self._add_click(outdata, pos - self._count_in + self._rec_origin, frames)
            if self.monitor:
                for c in range(ch):
                    outdata[:, c] += indata[:, 0] * self.monitor_gain
            self.pos = pos + frames
            self.out_peak = max(self.out_peak * 0.9, float(np.abs(outdata).max()) if frames else 0.0)
        except Exception as e:  # pragma: no cover
            self.last_error = repr(e)
            outdata[:] = 0


def make_click(bpm: float, n_samples: int, sr: int = SR, beats_per_bar: int = 4, offset_beats: float = 0.0) -> np.ndarray:
    """Дорожка метронома (2, n): акцент на первой доле такта. offset_beats — смещение начала в долях."""
    out = np.zeros(n_samples, dtype=np.float32)
    beat = 60.0 / max(30.0, bpm)
    L = int(0.03 * sr)
    t = np.arange(L) / sr
    hi = (np.sin(2 * np.pi * 1600 * t) * np.exp(-t / 0.008)).astype(np.float32)
    lo = (np.sin(2 * np.pi * 1000 * t) * np.exp(-t / 0.008)).astype(np.float32)
    k = 0
    while True:
        p = int((k - offset_beats) * beat * sr)
        if p >= n_samples:
            break
        if p >= 0:
            c = hi if (k % beats_per_bar == 0) else lo
            m = min(L, n_samples - p)
            out[p:p + m] += c[:m] * 0.5
        k += 1
    return np.vstack([out, out])


def diagnostics_text() -> str:
    """Сведения для диагностики проблем со звуком."""
    import platform
    import sys
    lines = [f"Python {sys.version.split()[0]} · {platform.platform()}", f"numpy {np.__version__}"]
    try:
        import scipy
        lines.append(f"scipy {scipy.__version__}")
    except Exception:
        pass
    lines.append(f"soundfile: {'ok' if sf is not None else 'недоступна'}")
    if sd is None:
        lines.append(f"sounddevice: НЕДОСТУПЕН ({SD_ERROR})")
        return "\n".join(lines)
    try:
        lines.append(f"sounddevice {sd.__version__} · {sd.get_portaudio_version()[1]}")
        apis = sd.query_hostapis()
        devs = sd.query_devices()
        d_in, d_out = sd.default.device
        lines.append(f"Устройство по умолчанию: вход={d_in}, выход={d_out}")
        for i, d in enumerate(devs):
            api = apis[d["hostapi"]]["name"]
            lines.append(f"  [{i}] {d['name']} · {api} · вх {d['max_input_channels']} / вых {d['max_output_channels']} · "
                         f"{int(d['default_samplerate'])} Гц")
    except Exception as e:
        lines.append(f"Ошибка опроса устройств: {e}")
    return "\n".join(lines)
