"""Модель проекта: дорожки, плагины, мастер. Сохранение/загрузка проекта (.vsp)."""
from __future__ import annotations

import copy
import io
import json
import time
import uuid
import zipfile
from dataclasses import dataclass, field

import numpy as np

from .plugins import PLUGINS, new_plugin

SR = 44100
TRACK_COLORS = ["#2fd0c0", "#ff9f43", "#6aa8ff", "#c58bff", "#ff6b81", "#6be38a", "#ffd36b", "#8fb3ff"]
FORMAT_VERSION = 1


@dataclass
class Track:
    id: str
    name: str
    kind: str = "vocal"                 # vocal | audio (бит / инструментал)
    color: str = TRACK_COLORS[0]
    audio: np.ndarray | None = None     # (каналы, n) float32
    audio_id: str = ""
    vol_db: float = 0.0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    arm: bool = False
    plugins: list = field(default_factory=list)
    source_name: str = ""
    start: int = 0                      # положение записи на таймлайне, отсчёты

    @property
    def length(self) -> int:
        return 0 if self.audio is None else self.audio.shape[1]

    @property
    def end(self) -> int:
        """Конец записи на таймлайне (отсчёты)."""
        return 0 if self.audio is None else self.start + self.audio.shape[1]

    def mono(self) -> np.ndarray | None:
        if self.audio is None:
            return None
        return self.audio[0] if self.audio.shape[0] == 1 else self.audio.mean(axis=0)

    def find_plugin(self, pid: str):
        for p in self.plugins:
            if p["id"] == pid:
                return p
        return None


def new_audio_id() -> str:
    return uuid.uuid4().hex[:10]


class Project:
    def __init__(self):
        self.sr = SR
        self.name = "Untitled"
        self.path: str | None = None
        self.tracks: list[Track] = []
        self.master = {"vol_db": 0.0, "plugins": [new_plugin("limiter")]}
        self.bpm = 120.0
        self.dirty = False

    # ---- дорожки ----------------------------------------------------
    def get(self, tid: str) -> Track | None:
        for t in self.tracks:
            if t.id == tid:
                return t
        return None

    def add_track(self, name: str | None = None, kind: str = "vocal", audio=None, source: str = "") -> Track:
        n = len(self.tracks)
        if name is None:
            base = "Vocal" if kind == "vocal" else "Backing"
            k = 1
            names = {t.name for t in self.tracks}
            while f"{base} {k}" in names:
                k += 1
            name = f"{base} {k}"
        t = Track(id=uuid.uuid4().hex[:8], name=name, kind=kind, color=TRACK_COLORS[n % len(TRACK_COLORS)])
        if audio is not None:
            t.audio = np.ascontiguousarray(audio, dtype=np.float32)
            t.audio_id = new_audio_id()
        t.source_name = source
        self.tracks.append(t)
        self.dirty = True
        return t

    def remove_track(self, tid: str):
        self.tracks = [t for t in self.tracks if t.id != tid]
        self.dirty = True

    def total_samples(self) -> int:
        return max([t.end for t in self.tracks] + [0])

    # ---- снимок для рендера ------------------------------------------
    def snapshot(self, bypass_all: bool = False) -> dict:
        return {
            "sr": self.sr,
            "bypass_all": bypass_all,
            "bpm": self.bpm,
            "master": copy.deepcopy(self.master),
            "tracks": [
                {
                    "id": t.id, "audio": t.audio, "audio_id": t.audio_id, "start": t.start, "vol_db": t.vol_db, "pan": t.pan,
                    "mute": t.mute, "solo": t.solo, "kind": t.kind,
                    "plugins": copy.deepcopy(t.plugins),
                }
                for t in self.tracks
            ],
        }

    # ---- сериализация -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": FORMAT_VERSION, "name": self.name, "sr": self.sr, "bpm": self.bpm,
            "master": self.master,
            "tracks": [
                {"id": t.id, "name": t.name, "kind": t.kind, "color": t.color, "audio_id": t.audio_id,
                 "vol_db": t.vol_db, "pan": t.pan, "mute": t.mute, "solo": t.solo, "arm": t.arm,
                 "plugins": t.plugins, "source_name": t.source_name, "has_audio": t.audio is not None,
                 "start": t.start}
                for t in self.tracks
            ],
        }

    def save(self, path: str):
        import soundfile as sf
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("project.json", json.dumps(self.to_dict(), ensure_ascii=False, indent=1))
            for t in self.tracks:
                if t.audio is not None:
                    buf = io.BytesIO()
                    sf.write(buf, t.audio.T, self.sr, format="FLAC", subtype="PCM_24")
                    z.writestr(f"audio/{t.id}.flac", buf.getvalue())
        self.path = path
        self.dirty = False

    @classmethod
    def load(cls, path: str) -> "Project":
        import soundfile as sf
        pr = cls()
        with zipfile.ZipFile(path, "r") as z:
            d = json.loads(z.read("project.json").decode("utf-8"))
            pr.name = d.get("name", "Untitled")
            pr.sr = int(d.get("sr", SR))
            pr.bpm = float(d.get("bpm", 120.0))
            pr.master = d.get("master", pr.master)
            for td in d.get("tracks", []):
                t = Track(id=td["id"], name=td["name"], kind=td.get("kind", "vocal"), color=td.get("color", TRACK_COLORS[0]),
                          audio_id=td.get("audio_id", ""), vol_db=td.get("vol_db", 0.0), pan=td.get("pan", 0.0),
                          mute=td.get("mute", False), solo=td.get("solo", False), arm=False,
                          plugins=[p for p in td.get("plugins", []) if p.get("type") in PLUGINS],
                          source_name=td.get("source_name", ""), start=int(td.get("start", 0)))
                if td.get("has_audio"):
                    with z.open(f"audio/{t.id}.flac") as f:
                        data, fs = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=True)
                    t.audio = np.ascontiguousarray(data.T)
                    if not t.audio_id:
                        t.audio_id = new_audio_id()
                pr.tracks.append(t)
        pr.master["plugins"] = [p for p in pr.master.get("plugins", []) if p.get("type") in PLUGINS]
        pr.path = path
        pr.dirty = False
        return pr
