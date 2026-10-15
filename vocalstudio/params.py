"""Описание параметров: Spec и конструкторы F/C/B (используются в plugins.py и в интерфейсе)."""
from __future__ import annotations

from dataclasses import dataclass, field


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
