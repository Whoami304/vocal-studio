"""Описание параметров всех модулей (для интерфейса и для движка) + пресеты."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

from .dsp.scales import NOTE_NAMES, SCALES, scale_mask


@dataclass
class Spec:
    key: str
    label: str
    kind: str = "float"            # float | choice | bool
    lo: float = 0.0
    hi: float = 1.0
    default: object = 0.0
    unit: str = ""
    decimals: int = 1
    curve: float = 1.0             # >1: больше разрешения у нижней границы
    log: bool = False              # логарифмическая шкала (для частот)
    choices: list = field(default_factory=list)
    tip: str = ""


def F(key, label, lo, hi, default, unit="", decimals=1, curve=1.0, log=False, tip=""):
    return Spec(key, label, "float", lo, hi, default, unit, decimals, curve, log, [], tip)


def C(key, label, choices, default, tip=""):
    return Spec(key, label, "choice", 0, 0, default, "", 0, 1.0, False, list(choices), tip)


def B(key, label, default, tip=""):
    return Spec(key, label, "bool", 0, 1, default, "", 0, 1.0, False, [], tip)


INPUT_TYPES = ["Auto", "Soprano", "Alto/Tenor", "Low Male", "Instrument"]

MODULES: dict[str, dict] = {
    "input": {
        "title": "INPUT",
        "has_bypass": False,
        "specs": [
            F("gain", "Input Gain", -24, 24, 0.0, "dB", 1, tip="Усиление записи перед обработкой"),
        ],
    },
    "tune": {
        "title": "AUTO-TUNE",
        "has_bypass": True,
        "specs": [
            C("key", "Key", NOTE_NAMES, "C", "Тональность (заполняет маску нот)"),
            C("scale", "Scale", list(SCALES.keys()) + ["Custom"], "Chromatic", "Лад (заполняет маску нот)"),
            F("retune", "Retune Speed", 0, 400, 20.0, "ms", 0, 2.0,
              tip="Скорость подтягивания к ноте. 0 = мгновенно (эффект «робот»), 400 = очень мягко"),
            F("humanize", "Humanize", 0, 100, 25.0, "%", 0,
              tip="Замедляет коррекцию на долгих нотах — сохраняет живое движение голоса"),
            F("flex", "Flex-Tune", 0, 100, 0.0, "%", 0,
              tip="Сохраняет небольшие естественные отклонения от ноты, исправляя только явные ошибки"),
            F("vibrato", "Vibrato", 0, 200, 100.0, "%", 0,
              tip="Глубина естественного вибрато (100% = как спето)"),
            F("transpose", "Transpose", -12, 12, 0.0, "st", 0, tip="Сдвиг целевых нот в полутонах"),
            F("detune", "Detune", -100, 100, 0.0, "ct", 0, tip="Тонкая расстройка в центах"),
            F("formant", "Formant", -6, 6, 0.0, "st", 1,
              tip="Сдвиг формант (тембра) независимо от высоты"),
            F("amount", "Amount", 0, 100, 100.0, "%", 0, tip="Сила эффекта автотюна"),
            F("ref_a", "Ref. A", 415, 466, 440.0, "Hz", 1, tip="Строй: частота ноты ля"),
            C("input_type", "Input Type", INPUT_TYPES, "Auto", "Тип голоса — диапазон поиска высоты"),
            F("tracking", "Tracking", 0, 100, 40.0, "%", 0,
              tip="Чувствительность определения высоты. Выше = ловит тихие/шумные участки"),
        ],
        "extra": {"notes": scale_mask(0, "Chromatic")},
    },
    "gate": {
        "title": "GATE",
        "has_bypass": True,
        "default_on": False,
        "specs": [
            F("threshold", "Threshold", -80, -10, -55.0, "dB", 0),
            F("range", "Range", 0, 60, 30.0, "dB", 0),
            F("attack", "Attack", 0.1, 50, 2.0, "ms", 1, 2.0),
            F("release", "Release", 10, 500, 120.0, "ms", 0, 2.0),
        ],
    },
    "eq": {
        "title": "EQ",
        "has_bypass": True,
        "specs": [
            C("pos", "Position", ["Pre-Comp", "Post-Comp"], "Post-Comp"),
            B("hpf_on", "HPF", True),
            F("hpf_f", "Freq", 20, 400, 80.0, "Hz", 0, log=True),
            C("hpf_slope", "HPF Slope", ["12 dB/oct", "24 dB/oct", "48 dB/oct"], "12 dB/oct"),
            F("ls_f", "Freq", 40, 600, 120.0, "Hz", 0, log=True),
            F("ls_g", "Gain", -15, 15, 0.0, "dB", 1),
            F("b1_f", "Freq", 80, 1200, 300.0, "Hz", 0, log=True),
            F("b1_g", "Gain", -15, 15, 0.0, "dB", 1),
            F("b1_q", "Q", 0.3, 8, 1.0, "", 2, log=True),
            F("b2_f", "Freq", 400, 6000, 2500.0, "Hz", 0, log=True),
            F("b2_g", "Gain", -15, 15, 0.0, "dB", 1),
            F("b2_q", "Q", 0.3, 8, 1.0, "", 2, log=True),
            F("b3_f", "Freq", 1500, 12000, 5000.0, "Hz", 0, log=True),
            F("b3_g", "Gain", -15, 15, 0.0, "dB", 1),
            F("b3_q", "Q", 0.3, 8, 1.2, "", 2, log=True),
            F("hs_f", "Freq", 2000, 16000, 10000.0, "Hz", 0, log=True),
            F("hs_g", "Gain", -15, 15, 0.0, "dB", 1),
            B("lp_on", "LPF", False),
            F("lp_f", "Freq", 4000, 20000, 18000.0, "Hz", 0, log=True),
        ],
    },
    "comp": {
        "title": "COMPRESSOR",
        "has_bypass": True,
        "specs": [
            F("threshold", "Threshold", -60, 0, -18.0, "dB", 1),
            F("ratio", "Ratio", 1, 20, 3.0, ":1", 1, 2.0),
            F("attack", "Attack", 0.1, 100, 8.0, "ms", 1, 2.0),
            F("release", "Release", 10, 1000, 120.0, "ms", 0, 2.0),
            F("knee", "Knee", 0, 24, 6.0, "dB", 1),
            F("makeup", "Makeup", 0, 24, 3.0, "dB", 1),
            B("auto_makeup", "Auto Makeup", False),
            C("detect", "Detection", ["RMS", "Peak"], "RMS"),
            F("mix", "Mix", 0, 100, 100.0, "%", 0),
        ],
    },
    "deess": {
        "title": "DE-ESSER",
        "has_bypass": True,
        "specs": [
            F("freq", "Freq", 3000, 12000, 6500.0, "Hz", 0, log=True),
            F("threshold", "Threshold", -60, 0, -30.0, "dB", 1),
            F("range", "Reduce", 0, 24, 8.0, "dB", 1),
            B("wide", "Wideband", False),
        ],
    },
    "sat": {
        "title": "SATURATION",
        "has_bypass": True,
        "default_on": False,
        "specs": [
            F("drive", "Drive", 0, 30, 6.0, "dB", 1),
            F("warmth", "Warmth", 0, 3, 1.0, "", 1),
            F("mix", "Mix", 0, 100, 30.0, "%", 0),
        ],
    },
    "delay": {
        "title": "DELAY",
        "has_bypass": True,
        "default_on": False,
        "specs": [
            C("mode", "Time Mode", ["Sync", "Free"], "Sync"),
            F("bpm", "BPM", 40, 220, 120.0, "bpm", 0),
            C("division", "Note", ["1/2", "1/4", "1/8 dotted", "1/8", "1/8 triplet", "1/16"], "1/8"),
            F("time_ms", "Time", 20, 1500, 350.0, "ms", 0, 1.5),
            F("feedback", "Feedback", 0, 90, 30.0, "%", 0),
            F("lowcut", "Low Cut", 20, 1000, 200.0, "Hz", 0, log=True),
            F("highcut", "High Cut", 1000, 20000, 6000.0, "Hz", 0, log=True),
            F("pingpong", "Ping-Pong", 0, 100, 50.0, "%", 0),
            F("mix", "Mix", 0, 100, 18.0, "%", 0),
        ],
    },
    "reverb": {
        "title": "REVERB",
        "has_bypass": True,
        "specs": [
            C("type", "Type", ["Room", "Plate", "Hall", "Chamber", "Cathedral"], "Plate"),
            F("decay", "Decay", 0.2, 10.0, 1.6, "s", 2, 2.0),
            F("predelay", "Pre-Delay", 0, 200, 25.0, "ms", 0),
            F("size", "Size", 0, 100, 50.0, "%", 0),
            F("damping", "Damping", 0, 100, 50.0, "%", 0),
            F("width", "Width", 0, 150, 100.0, "%", 0),
            F("lowcut", "Low Cut", 20, 800, 150.0, "Hz", 0, log=True),
            F("highcut", "High Cut", 1500, 20000, 9000.0, "Hz", 0, log=True),
            F("mix", "Mix", 0, 100, 16.0, "%", 0),
        ],
    },
    "master": {
        "title": "OUTPUT",
        "has_bypass": False,
        "specs": [
            F("vocal_db", "Vocal", -30, 12, 0.0, "dB", 1),
            F("backing_db", "Backing", -30, 12, -3.0, "dB", 1),
            B("limiter_on", "Limiter", True),
            F("ceiling", "Ceiling", -12, 0, -1.0, "dB", 1),
        ],
    },
}

# Порядок обработки (EQ вставляется динамически: до или после компрессора)
MODULE_ORDER = ["input", "tune", "gate", "eq", "comp", "deess", "sat", "delay", "reverb", "master"]


def default_params() -> dict:
    """Полный набор значений: {module: {"on": bool, key: value, ...}}."""
    out: dict = {}
    for name, m in MODULES.items():
        d = {"on": bool(m.get("default_on", True))}
        for s in m["specs"]:
            d[s.key] = s.default
        for k, v in m.get("extra", {}).items():
            d[k] = copy.deepcopy(v)
        out[name] = d
    return out


def merge_params(base: dict, over: dict) -> dict:
    res = copy.deepcopy(base)
    for mod, vals in over.items():
        if mod in res and isinstance(vals, dict):
            res[mod].update(copy.deepcopy(vals))
    t = over.get("tune", {})
    if ("key" in t or "scale" in t) and "notes" not in t:
        tn = res["tune"]
        if tn.get("scale") in SCALES and tn.get("key") in NOTE_NAMES:
            tn["notes"] = scale_mask(NOTE_NAMES.index(tn["key"]), tn["scale"])
    return res


def params_key(d: dict) -> str:
    return json.dumps(d, sort_keys=True, default=str)


def _eq(**kw):
    return kw


PRESETS: dict[str, dict] = {
    "Default (Natural)": {},
    "Natural Pop": {
        "tune": {"retune": 35, "humanize": 40, "flex": 20, "scale": "Chromatic"},
        "comp": {"threshold": -20, "ratio": 3.5, "attack": 10, "release": 140, "makeup": 4},
        "eq": {"hpf_f": 90, "b1_g": -2.5, "b2_g": 2.0, "hs_g": 3.0},
        "reverb": {"type": "Plate", "decay": 1.5, "mix": 14},
        "delay": {"on": True, "mix": 10, "feedback": 28},
    },
    "Hard Tune (T-Pain)": {
        "tune": {"retune": 0, "humanize": 0, "flex": 0, "vibrato": 0, "amount": 100},
        "comp": {"threshold": -24, "ratio": 4, "attack": 4, "release": 90, "makeup": 5},
        "eq": {"hpf_f": 100, "b2_g": 2.5, "hs_g": 3.5},
        "deess": {"threshold": -32, "range": 10},
        "reverb": {"type": "Plate", "decay": 1.2, "mix": 12},
        "delay": {"on": True, "mix": 14, "pingpong": 80},
    },
    "Modern Rap Vocal": {
        "tune": {"retune": 15, "humanize": 10, "flex": 0, "vibrato": 40},
        "comp": {"threshold": -26, "ratio": 5, "attack": 5, "release": 80, "makeup": 6, "detect": "Peak"},
        "eq": {"hpf_f": 110, "ls_g": -2, "b1_g": -3, "b2_g": 2.5, "hs_f": 11000, "hs_g": 4},
        "deess": {"threshold": -30, "range": 10},
        "sat": {"on": True, "drive": 8, "mix": 25},
        "reverb": {"type": "Room", "decay": 0.9, "mix": 10},
        "delay": {"on": True, "mix": 14, "feedback": 20, "division": "1/8"},
    },
    "Airy Female Pop": {
        "tune": {"retune": 40, "humanize": 45, "flex": 25, "input_type": "Soprano"},
        "comp": {"threshold": -20, "ratio": 3, "attack": 12, "release": 150, "makeup": 3},
        "eq": {"hpf_f": 120, "b1_f": 350, "b1_g": -2, "b2_g": 1.5, "hs_f": 12000, "hs_g": 4.5},
        "deess": {"threshold": -32, "range": 9, "freq": 7500},
        "reverb": {"type": "Hall", "decay": 2.2, "mix": 20, "predelay": 35},
        "delay": {"on": True, "mix": 12},
    },
    "Intimate / Close": {
        "tune": {"retune": 80, "humanize": 60, "flex": 40, "vibrato": 110},
        "comp": {"threshold": -22, "ratio": 2.5, "attack": 14, "release": 180, "makeup": 3},
        "eq": {"hpf_f": 80, "ls_g": 1.5, "b1_g": -1.5, "hs_g": 2.0},
        "reverb": {"type": "Room", "decay": 0.8, "mix": 9},
        "delay": {"on": False},
    },
    "Big Hall Ballad": {
        "tune": {"retune": 55, "humanize": 55, "flex": 30},
        "comp": {"threshold": -20, "ratio": 3, "attack": 15, "release": 200, "makeup": 3},
        "eq": {"hpf_f": 90, "b3_g": 1.5, "hs_g": 2.5},
        "reverb": {"type": "Hall", "decay": 3.4, "mix": 26, "predelay": 45, "damping": 60},
        "delay": {"on": True, "mix": 12, "division": "1/4", "feedback": 32},
    },
    "Radio / Lo-Fi": {
        "tune": {"retune": 25},
        "comp": {"threshold": -26, "ratio": 6, "attack": 3, "release": 70, "makeup": 7},
        "eq": {"hpf_f": 350, "hpf_slope": "24 dB/oct", "lp_on": True, "lp_f": 6500, "b2_f": 1800, "b2_g": 4},
        "sat": {"on": True, "drive": 14, "mix": 55},
        "reverb": {"on": False},
    },
}

FACTORY_ORDER = list(PRESETS.keys())


def preset_params(name: str) -> dict:
    return merge_params(default_params(), PRESETS.get(name, {}))
