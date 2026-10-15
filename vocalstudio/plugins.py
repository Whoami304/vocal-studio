"""Реестр плагинов: параметры, обработка, пресеты.

Плагин — это словарь {"id", "type", "on", "params": {...}}; тип определяет набор параметров
(PLUGINS[type].specs) и функцию обработки. Плагины стоят в цепочках дорожек и мастера.
"""
from __future__ import annotations

import copy
import itertools
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .dsp import dynamics, eq as eqm, timefx
from .dsp.autotune import autotune
from .dsp.scales import NOTE_NAMES, SCALES, scale_mask
from .params import B, C, F, INPUT_TYPES, Spec


# ---------------------------------------------------------------- обработка --

class Env:
    """Окружение, которое микшер передаёт плагину при обработке."""

    def __init__(self, sr, cancel=None, progress=None, get_pitch=None):
        self.sr = sr
        self.cancel = cancel
        self.progress = progress
        self.get_pitch = get_pitch
        self.results: dict = {}


def _stereo(x):
    return x if x.shape[0] == 2 else np.vstack([x[0], x[0]])


def _mono(x):
    return x[0] if x.shape[0] == 1 else x.mean(axis=0)


def proc_tune(x, sr, p, env):
    mono = _mono(x)
    track = env.get_pitch(mono, p)
    prog = (lambda f: env.progress("tune", f)) if env.progress else None
    y, corr = autotune(mono, track, p, overrides=p.get("overrides") or None, cancel=env.cancel, progress=prog)
    env.results["tune"] = (track, corr)
    return y[None, :]


def proc_gate(x, sr, p, env):
    return dynamics.gate(x, sr, p)


def proc_eq(x, sr, p, env):
    return eqm.eq_process(x, p, sr)


def proc_comp(x, sr, p, env):
    y, gr = dynamics.compressor(x, sr, p)
    env.results["gr"] = gr[::64]
    return y


def proc_deess(x, sr, p, env):
    y, gr = dynamics.deesser(x, sr, p)
    env.results["gr"] = gr[::64]
    return y


def proc_sat(x, sr, p, env):
    return dynamics.saturate(x, sr, p)


def proc_delay(x, sr, p, env):
    wet = timefx.delay_process(_mono(x), sr, p)
    return _stereo(x) + wet * (float(p.get("mix", 18.0)) / 100.0)


def proc_reverb(x, sr, p, env):
    wet = timefx.reverb_process(_mono(x), sr, p)
    return _stereo(x) + wet * (float(p.get("mix", 16.0)) / 100.0)


def proc_limiter(x, sr, p, env):
    return dynamics.limiter(x, sr, float(p.get("ceiling", -1.0)), release_ms=float(p.get("release", 60.0)))


def proc_gain(x, sr, p, env):
    y = x * (10.0 ** (float(p.get("gain", 0.0)) / 20.0))
    if p.get("invert", False):
        y = -y
    if p.get("mono", False) and y.shape[0] == 2:
        m = y.mean(axis=0)
        y = np.vstack([m, m])
    return y


def tail_none(p, sr):
    return 0


def tail_delay(p, sr):
    return timefx.delay_tail_samples(p, sr) if p.get("mix", 0) > 0 else 0


def tail_reverb(p, sr):
    return timefx.reverb_tail_samples(p, sr) if p.get("mix", 0) > 0 else 0


# --------------------------------------------------------------- описание --

@dataclass
class PluginDef:
    type: str
    title: str
    category: str
    tagline: str                 # понятное описание в одну строку (для новичка)
    help: str                    # подробнее
    specs: list
    process: Callable
    tail: Callable = tail_none
    extra: dict = field(default_factory=dict)
    presets: dict = field(default_factory=dict)
    icon: str = "•"


def _eq_specs():
    return [
        B("hpf_on", "HPF", True),
        F("hpf_f", "Freq", 20, 400, 80.0, "Hz", 0, log=True, tip="Частота среза фильтра низов. Убирает гул и «бубнёж» ниже голоса"),
        C("hpf_slope", "HPF Slope", ["12 dB/oct", "24 dB/oct", "48 dB/oct"], "12 dB/oct"),
        F("ls_f", "Freq", 40, 600, 120.0, "Hz", 0, log=True),
        F("ls_g", "Gain", -15, 15, 0.0, "dB", 1, tip="Низкие частоты: + даёт плотность, − убирает гул"),
        F("b1_f", "Freq", 80, 1200, 300.0, "Hz", 0, log=True),
        F("b1_g", "Gain", -15, 15, 0.0, "dB", 1, tip="Нижняя середина (200–500 Гц): минус убирает «коробочный» звук"),
        F("b1_q", "Q", 0.3, 8, 1.0, "", 2, log=True, tip="Ширина полосы: больше Q — уже"),
        F("b2_f", "Freq", 400, 6000, 2500.0, "Hz", 0, log=True),
        F("b2_g", "Gain", -15, 15, 0.0, "dB", 1, tip="Середина (2–4 кГц): плюс добавляет разборчивость и «присутствие»"),
        F("b2_q", "Q", 0.3, 8, 1.0, "", 2, log=True),
        F("b3_f", "Freq", 1500, 12000, 5000.0, "Hz", 0, log=True),
        F("b3_g", "Gain", -15, 15, 0.0, "dB", 1, tip="Верхняя середина: плюс — чёткость, минус — мягче «с»"),
        F("b3_q", "Q", 0.3, 8, 1.2, "", 2, log=True),
        F("hs_f", "Freq", 2000, 16000, 10000.0, "Hz", 0, log=True),
        F("hs_g", "Gain", -15, 15, 0.0, "dB", 1, tip="Верха: плюс добавляет «воздух» и блеск"),
        B("lp_on", "LPF", False),
        F("lp_f", "Freq", 4000, 20000, 18000.0, "Hz", 0, log=True, tip="Срез верхов — для эффекта «телефон»/радио"),
    ]


PLUGINS: dict[str, PluginDef] = {}


def _reg(d: PluginDef):
    PLUGINS[d.type] = d


_reg(PluginDef(
    "tune", "Auto-Tune", "Pitch", "Исправляет фальшивые ноты",
    "Подтягивает голос к нотам выбранной тональности. Retune Speed 0 — жёсткий «робот»-эффект, "
    "15–40 мс — современный поп, 80+ — почти незаметная подстройка. Кнопка «Определить тональность» "
    "найдёт ключ по вашему пению.",
    [
        C("key", "Key", NOTE_NAMES, "C", "Тональность песни"),
        C("scale", "Scale", list(SCALES.keys()) + ["Custom"], "Chromatic", "Лад: мажор, минор и др. Chromatic — все ноты разрешены"),
        F("retune", "Retune Speed", 0, 400, 20.0, "ms", 0, 2.0,
          tip="Скорость подтягивания к ноте. 0 = мгновенно (эффект «робот»), 400 = очень мягко"),
        F("humanize", "Humanize", 0, 100, 25.0, "%", 0, tip="Замедляет коррекцию на долгих нотах — сохраняет живое «плавание» голоса"),
        F("flex", "Flex-Tune", 0, 100, 0.0, "%", 0, tip="Не трогает небольшие естественные отклонения, исправляет только явные ошибки"),
        F("vibrato", "Vibrato", 0, 200, 100.0, "%", 0, tip="Глубина вибрато (100% = как спето)"),
        F("transpose", "Transpose", -12, 12, 0.0, "st", 0, tip="Сдвиг целевых нот в полутонах"),
        F("detune", "Detune", -100, 100, 0.0, "ct", 0, tip="Тонкая расстройка в центах"),
        F("formant", "Formant", -6, 6, 0.0, "st", 1, tip="Сдвиг тембра (формант) независимо от высоты: голос «крупнее» или «меньше»"),
        F("amount", "Amount", 0, 100, 100.0, "%", 0, tip="Общая сила эффекта автотюна"),
        F("ref_a", "Ref. A", 415, 466, 440.0, "Hz", 1, tip="Строй: частота ноты ля (440 Гц — стандарт)"),
        C("input_type", "Input Type", INPUT_TYPES, "Auto", "Тип голоса — диапазон поиска высоты"),
        F("tracking", "Tracking", 0, 100, 40.0, "%", 0, tip="Чувствительность определения высоты. Выше = ловит тихие/шумные участки"),
    ],
    proc_tune, extra={"notes": scale_mask(0, "Chromatic"), "overrides": []},
    presets={
        "Natural": {"retune": 35, "humanize": 40, "flex": 20},
        "Transparent (gentle fix)": {"retune": 90, "humanize": 60, "flex": 40, "vibrato": 110},
        "Modern Pop": {"retune": 20, "humanize": 25, "flex": 0},
        "Rap / Fast": {"retune": 10, "humanize": 10, "vibrato": 40},
        "Hard Tune (T-Pain)": {"retune": 0, "humanize": 0, "flex": 0, "vibrato": 0, "amount": 100},
    },
    icon="♪",
))

_reg(PluginDef(
    "gate", "Noise Gate", "Dynamics", "Убирает шум в паузах",
    "Приглушает запись, когда вы не поёте: гудение, дыхание, шум комнаты. Порог (Threshold) должен быть "
    "выше шума, но ниже вашего голоса.",
    [
        F("threshold", "Threshold", -80, -10, -55.0, "dB", 0, tip="Громкость, ниже которой звук считается шумом"),
        F("range", "Range", 0, 60, 30.0, "dB", 0, tip="На сколько децибел приглушать шум"),
        F("attack", "Attack", 0.1, 50, 2.0, "ms", 1, 2.0),
        F("release", "Release", 10, 500, 120.0, "ms", 0, 2.0),
    ],
    proc_gate, presets={"Light": {"threshold": -58, "range": 18}, "Tight": {"threshold": -45, "range": 40, "release": 80}},
    icon="▭",
))

_reg(PluginDef(
    "eq", "Equalizer", "EQ", "Тембр: больше тепла, чёткости, воздуха",
    "Регулирует звучание по частотам. Перетаскивайте цветные узлы на графике. Срез HPF убирает гул, "
    "Bell 1 вырезает «бубнёж», Bell 2 добавляет разборчивость, High Shelf — «воздух».",
    _eq_specs(), proc_eq,
    presets={
        "Clean (HPF only)": {"hpf_f": 90},
        "Bright Pop": {"hpf_f": 90, "b1_g": -2.5, "b2_g": 2.0, "hs_g": 3.0},
        "Warm / Intimate": {"hpf_f": 80, "ls_g": 2.0, "b1_g": -1.5, "hs_g": 1.5},
        "De-mud": {"hpf_f": 100, "b1_f": 280, "b1_g": -4, "b1_q": 1.2},
        "Telephone / Radio": {"hpf_f": 350, "hpf_slope": "24 dB/oct", "lp_on": True, "lp_f": 6500, "b2_f": 1800, "b2_g": 4},
    },
    icon="∿",
))

_reg(PluginDef(
    "comp", "Compressor", "Dynamics", "Выравнивает громкость голоса",
    "Делает громкие места тише, чтобы голос звучал ровно и «стоял» над музыкой. Threshold — порог, "
    "Ratio — сила сжатия, Makeup — компенсация громкости.",
    [
        F("threshold", "Threshold", -60, 0, -18.0, "dB", 1, tip="Всё, что громче порога, будет сжато"),
        F("ratio", "Ratio", 1, 20, 3.0, ":1", 1, 2.0, tip="Сила сжатия: 2:1 мягко, 4:1 заметно, 10:1+ жёстко"),
        F("attack", "Attack", 0.1, 100, 8.0, "ms", 1, 2.0, tip="Как быстро компрессор реагирует"),
        F("release", "Release", 10, 1000, 120.0, "ms", 0, 2.0, tip="Как быстро отпускает"),
        F("knee", "Knee", 0, 24, 6.0, "dB", 1, tip="Мягкость перехода к сжатию"),
        F("makeup", "Makeup", 0, 24, 3.0, "dB", 1, tip="Добавить громкость после сжатия"),
        B("auto_makeup", "Auto Makeup", False, "Автоматически компенсировать громкость"),
        C("detect", "Detection", ["RMS", "Peak"], "RMS", "RMS — плавно, Peak — ловит быстрые пики"),
        F("mix", "Mix", 0, 100, 100.0, "%", 0, tip="Параллельное сжатие: смешать с несжатым звуком"),
    ],
    proc_comp,
    presets={
        "Gentle Leveler": {"threshold": -20, "ratio": 2.5, "attack": 14, "release": 180, "makeup": 3},
        "Pop Vocal": {"threshold": -20, "ratio": 3.5, "attack": 10, "release": 140, "makeup": 4},
        "Rap Punch": {"threshold": -26, "ratio": 5, "attack": 5, "release": 80, "makeup": 6, "detect": "Peak"},
        "Squash (heavy)": {"threshold": -30, "ratio": 8, "attack": 3, "release": 70, "makeup": 8, "detect": "Peak"},
    },
    icon="⇊",
))

_reg(PluginDef(
    "deess", "De-esser", "Dynamics", "Убирает резкие свистящие «с» и «ш»",
    "Приглушает свистящие согласные, когда они слишком резкие. Frequency — где живут ваши «с» "
    "(обычно 5–9 кГц), Threshold — когда включаться.",
    [
        F("freq", "Freq", 3000, 12000, 6500.0, "Hz", 0, log=True, tip="Частота свистящих согласных"),
        F("threshold", "Threshold", -60, 0, -30.0, "dB", 1, tip="Порог срабатывания"),
        F("range", "Reduce", 0, 24, 8.0, "dB", 1, tip="Максимальное подавление"),
        B("wide", "Wideband", False, "Подавлять весь голос, а не только полосу «с»"),
    ],
    proc_deess,
    presets={"Light": {"range": 5, "threshold": -28}, "Medium": {"range": 8, "threshold": -30}, "Strong": {"range": 12, "threshold": -34}},
    icon="s",
))

_reg(PluginDef(
    "sat", "Saturation", "Colour", "Тепло и плотность («ламповый» звук)",
    "Добавляет приятные гармоники — голос звучит плотнее и ближе. Drive — сила, Mix — сколько эффекта подмешать.",
    [
        F("drive", "Drive", 0, 30, 6.0, "dB", 1),
        F("warmth", "Warmth", 0, 3, 1.0, "", 1, tip="Больше — теплее (чётные гармоники)"),
        F("mix", "Mix", 0, 100, 30.0, "%", 0),
    ],
    proc_sat, presets={"Warm": {"drive": 5, "mix": 25}, "Gritty": {"drive": 16, "mix": 55}},
    icon="≋",
))

_reg(PluginDef(
    "delay", "Delay", "Space", "Эхо-повторы голоса",
    "Повторяет голос с задержкой, синхронно с темпом (BPM) или в миллисекундах. Feedback — сколько повторов, "
    "Ping-Pong — прыжки между левым и правым ухом.",
    [
        C("mode", "Time Mode", ["Sync", "Free"], "Sync", "Sync — по темпу песни, Free — в миллисекундах"),
        F("bpm", "BPM", 40, 220, 120.0, "bpm", 0, tip="Темп песни"),
        C("division", "Note", ["1/2", "1/4", "1/8 dotted", "1/8", "1/8 triplet", "1/16"], "1/8", "Длительность между повторами"),
        F("time_ms", "Time", 20, 1500, 350.0, "ms", 0, 1.5),
        F("feedback", "Feedback", 0, 90, 30.0, "%", 0, tip="Сколько повторов"),
        F("lowcut", "Low Cut", 20, 1000, 200.0, "Hz", 0, log=True),
        F("highcut", "High Cut", 1000, 20000, 6000.0, "Hz", 0, log=True, tip="Чем ниже — тем «темнее» повторы"),
        F("pingpong", "Ping-Pong", 0, 100, 50.0, "%", 0),
        F("mix", "Mix", 0, 100, 18.0, "%", 0, tip="Громкость эха"),
    ],
    proc_delay, tail=tail_delay,
    presets={
        "Subtle 1/8": {"division": "1/8", "feedback": 25, "mix": 12},
        "Slapback": {"mode": "Free", "time_ms": 110, "feedback": 8, "mix": 20, "pingpong": 0},
        "Ping-Pong 1/4": {"division": "1/4", "feedback": 40, "pingpong": 100, "mix": 18},
    },
    icon="⟲",
))

_reg(PluginDef(
    "reverb", "Reverb", "Space", "Эхо помещения: голос «в зале»",
    "Добавляет пространство. Type — тип помещения, Decay — длина «хвоста», Pre-Delay — пауза перед отражениями "
    "(чем больше, тем разборчивее голос), Mix — сколько эффекта.",
    [
        C("type", "Type", ["Room", "Plate", "Hall", "Chamber", "Cathedral"], "Plate", "Тип помещения"),
        F("decay", "Decay", 0.2, 10.0, 1.6, "s", 2, 2.0, tip="Длина «хвоста» в секундах"),
        F("predelay", "Pre-Delay", 0, 200, 25.0, "ms", 0, tip="Пауза перед отражениями — повышает разборчивость"),
        F("size", "Size", 0, 100, 50.0, "%", 0),
        F("damping", "Damping", 0, 100, 50.0, "%", 0, tip="Поглощение верхов: больше — «мягче» хвост"),
        F("width", "Width", 0, 150, 100.0, "%", 0, tip="Ширина стерео"),
        F("lowcut", "Low Cut", 20, 800, 150.0, "Hz", 0, log=True),
        F("highcut", "High Cut", 1500, 20000, 9000.0, "Hz", 0, log=True),
        F("mix", "Mix", 0, 100, 16.0, "%", 0, tip="Громкость реверберации"),
    ],
    proc_reverb, tail=tail_reverb,
    presets={
        "Small Room": {"type": "Room", "decay": 0.8, "mix": 10, "predelay": 10},
        "Vocal Plate": {"type": "Plate", "decay": 1.5, "mix": 15, "predelay": 25},
        "Big Hall": {"type": "Hall", "decay": 3.0, "mix": 22, "predelay": 40, "damping": 60},
        "Cathedral": {"type": "Cathedral", "decay": 5.5, "mix": 20, "predelay": 55, "damping": 65},
    },
    icon="◎",
))

_reg(PluginDef(
    "limiter", "Limiter", "Dynamics", "Защита от перегруза (клиппинга)",
    "Не даёт звуку превысить заданный уровень (Ceiling). Ставится в конце цепочки, обычно на Master.",
    [
        F("ceiling", "Ceiling", -12, 0, -1.0, "dB", 1, tip="Максимальный уровень"),
        F("release", "Release", 10, 500, 60.0, "ms", 0, 2.0),
    ],
    proc_limiter, presets={"Streaming (-1 dB)": {"ceiling": -1.0}, "Loud (-0.3 dB)": {"ceiling": -0.3, "release": 40}},
    icon="⊓",
))

_reg(PluginDef(
    "gain", "Utility", "Utility", "Громкость, фаза, моно",
    "Простая регулировка уровня, инверсия фазы и сведение в моно.",
    [
        F("gain", "Gain", -24, 24, 0.0, "dB", 1),
        B("invert", "Invert phase", False),
        B("mono", "Mono", False),
    ],
    proc_gain, icon="▲",
))

CATEGORY_ORDER = ["Pitch", "Dynamics", "EQ", "Colour", "Space", "Utility"]
CATEGORY_TITLES = {
    "Pitch": "Высота тона", "Dynamics": "Динамика", "EQ": "Тембр (EQ)",
    "Colour": "Окраска", "Space": "Пространство", "Utility": "Утилиты",
}


# ----------------------------------------------------------- экземпляры -----

def new_plugin(ptype: str, overrides: dict | None = None, on: bool = True) -> dict:
    d = PLUGINS[ptype]
    params = {s.key: s.default for s in d.specs}
    for k, v in d.extra.items():
        params[k] = copy.deepcopy(v)
    if overrides:
        params.update(copy.deepcopy(overrides))
        if ptype == "tune" and ("key" in overrides or "scale" in overrides) and "notes" not in overrides:
            if params["scale"] in SCALES and params["key"] in NOTE_NAMES:
                params["notes"] = scale_mask(NOTE_NAMES.index(params["key"]), params["scale"])
    return {"id": uuid.uuid4().hex[:8], "type": ptype, "on": bool(on), "params": params}


def spec_map(ptype: str) -> dict:
    return {s.key: s for s in PLUGINS[ptype].specs}


def plugin_tail(plugin: dict, sr: int) -> int:
    if not plugin["on"]:
        return 0
    return int(PLUGINS[plugin["type"]].tail(plugin["params"], sr))


# ------------------------------------------------------ пресеты цепочек -----

def _chain(*items) -> list:
    """items: (type, overrides_or_None[, on])."""
    out = []
    for it in items:
        t, o = it[0], it[1]
        on = it[2] if len(it) > 2 else True
        out.append(new_plugin(t, o, on))
    return out


CHAIN_PRESETS: dict[str, Callable[[], list]] = {
    "Natural Pop": lambda: _chain(
        ("tune", {"retune": 35, "humanize": 40, "flex": 20}),
        ("comp", {"threshold": -20, "ratio": 3.5, "attack": 10, "release": 140, "makeup": 4}),
        ("deess", None),
        ("eq", {"hpf_f": 90, "b1_g": -2.5, "b2_g": 2.0, "hs_g": 3.0}),
        ("delay", {"mix": 10, "feedback": 28}),
        ("reverb", {"type": "Plate", "decay": 1.5, "mix": 14}),
    ),
    "Hard Tune (T-Pain)": lambda: _chain(
        ("tune", {"retune": 0, "humanize": 0, "flex": 0, "vibrato": 0}),
        ("comp", {"threshold": -24, "ratio": 4, "attack": 4, "release": 90, "makeup": 5}),
        ("deess", {"threshold": -32, "range": 10}),
        ("eq", {"hpf_f": 100, "b2_g": 2.5, "hs_g": 3.5}),
        ("delay", {"mix": 14, "pingpong": 80}),
        ("reverb", {"type": "Plate", "decay": 1.2, "mix": 12}),
    ),
    "Modern Rap": lambda: _chain(
        ("tune", {"retune": 15, "humanize": 10, "vibrato": 40}),
        ("comp", {"threshold": -26, "ratio": 5, "attack": 5, "release": 80, "makeup": 6, "detect": "Peak"}),
        ("deess", {"threshold": -30, "range": 10}),
        ("eq", {"hpf_f": 110, "ls_g": -2, "b1_g": -3, "b2_g": 2.5, "hs_f": 11000, "hs_g": 4}),
        ("sat", {"drive": 8, "mix": 25}),
        ("delay", {"mix": 14, "feedback": 20}),
        ("reverb", {"type": "Room", "decay": 0.9, "mix": 10}),
    ),
    "Airy Female Pop": lambda: _chain(
        ("tune", {"retune": 40, "humanize": 45, "flex": 25, "input_type": "Soprano"}),
        ("comp", {"threshold": -20, "ratio": 3, "attack": 12, "release": 150, "makeup": 3}),
        ("deess", {"threshold": -32, "range": 9, "freq": 7500}),
        ("eq", {"hpf_f": 120, "b1_f": 350, "b1_g": -2, "b2_g": 1.5, "hs_f": 12000, "hs_g": 4.5}),
        ("delay", {"mix": 12}),
        ("reverb", {"type": "Hall", "decay": 2.2, "mix": 20, "predelay": 35}),
    ),
    "Intimate / Close": lambda: _chain(
        ("tune", {"retune": 80, "humanize": 60, "flex": 40, "vibrato": 110}),
        ("comp", {"threshold": -22, "ratio": 2.5, "attack": 14, "release": 180, "makeup": 3}),
        ("eq", {"hpf_f": 80, "ls_g": 1.5, "b1_g": -1.5, "hs_g": 2.0}),
        ("reverb", {"type": "Room", "decay": 0.8, "mix": 9}),
    ),
    "Big Hall Ballad": lambda: _chain(
        ("tune", {"retune": 55, "humanize": 55, "flex": 30}),
        ("comp", {"threshold": -20, "ratio": 3, "attack": 15, "release": 200, "makeup": 3}),
        ("deess", None),
        ("eq", {"hpf_f": 90, "b3_g": 1.5, "hs_g": 2.5}),
        ("delay", {"mix": 12, "division": "1/4", "feedback": 32}),
        ("reverb", {"type": "Hall", "decay": 3.4, "mix": 26, "predelay": 45, "damping": 60}),
    ),
    "Radio / Lo-Fi": lambda: _chain(
        ("tune", {"retune": 25}),
        ("comp", {"threshold": -26, "ratio": 6, "attack": 3, "release": 70, "makeup": 7}),
        ("eq", {"hpf_f": 350, "hpf_slope": "24 dB/oct", "lp_on": True, "lp_f": 6500, "b2_f": 1800, "b2_g": 4}),
        ("sat", {"drive": 14, "mix": 55}),
    ),
    "Clean (EQ + Compressor)": lambda: _chain(
        ("eq", {"hpf_f": 90}),
        ("comp", {"threshold": -20, "ratio": 3, "makeup": 3}),
        ("deess", None),
    ),
    "Empty (no plugins)": lambda: [],
}

CHAIN_PRESET_ORDER = list(CHAIN_PRESETS.keys())


# ------------------------------------------------------- пользовательские ---

def apply_plugin_preset(plugin: dict, preset: dict):
    plugin["params"].update(copy.deepcopy(preset))
    if plugin["type"] == "tune" and ("key" in preset or "scale" in preset) and "notes" not in preset:
        p = plugin["params"]
        if p["scale"] in SCALES and p["key"] in NOTE_NAMES:
            p["notes"] = scale_mask(NOTE_NAMES.index(p["key"]), p["scale"])
