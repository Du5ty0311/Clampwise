"""Rig profiles: Clampwise works on any rig it is on.

A rig is learned from what its files say about it, never written in for one
lab: the digitizer and the amplifier the header names (their type codes as the
file stores them), the main recorded channel's name and units, and the
protocols used on it. The user names a rig once, from a few of its files; its
preset (if one is chosen) and the inputs that carry light on it (its wiring,
which a line's name cannot tell) then apply to every file that matches it, in
any project. A file matches a rig only on the same hardware; a channel or protocol
not seen on it before is said as a doubt, and two rigs matching equally is a
doubt too, never a silent choice.

Profiles are the user's, kept beside Clampwise's other per-user files
(``%APPDATA%\\Clampwise\\rigs.json``; ``CLAMPWISE_RIGS`` names another file).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Fingerprint:
    format: str
    digitizer: int          # the header's digitizer type code (-1: not stored)
    amplifier: int          # the main channel's telegraph instrument code (0: none)
    channel: str            # the main recorded channel's name
    units: str
    protocol: str = ""
    creator: str = ""

    def describe(self) -> str:
        hw = f"digitizer type {self.digitizer}" if self.digitizer >= 0 else "digitizer not stored"
        amp = f"amplifier code {self.amplifier}" if self.amplifier else "no amplifier telegraph"
        return f"{self.format}, {hw}, {amp}, channel {self.channel} ({self.units})"


def fingerprint(path: str | Path) -> Fingerprint:
    """What a file says about the rig it was recorded on (its header only)."""
    path = Path(path)
    if path.suffix.lower() == ".abf":
        import pyabf

        abf = pyabf.ABF(str(path), loadData=False)
        proto = getattr(abf, "_protocolSection", None)
        adc = getattr(abf, "_adcSection", None)
        v1 = getattr(abf, "_headerV1", None)
        digitizer = int(getattr(proto, "nDigitizerType", getattr(v1, "nDigitizerType", -1)) or 0) \
            if (proto is not None or v1 is not None) else -1
        first = abf.channelList[0] if getattr(abf, "channelList", None) else 0
        tele = getattr(adc, "nTelegraphInstrument", None) or getattr(v1, "nTelegraphInstrument", None) or []
        amplifier = int(tele[first]) if len(tele) > first and tele[first] is not None else 0
        return Fingerprint("ABF", digitizer, amplifier, str(abf.adcNames[0]), str(abf.adcUnits[0]),
                           Path(abf.protocolPath or "").stem, str(getattr(abf, "creator", "") or ""))
    from ..io import load

    rec = load(path)
    sig = rec.signal_channel
    return Fingerprint(rec.header.get("format", path.suffix.lstrip(".").upper()), -1, 0,
                       sig.name if sig else "", rec.signal_units, rec.protocol or "")


@dataclass
class Rig:
    name: str
    format: str
    digitizer: int
    amplifier: int
    channels: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    protocols: list[str] = field(default_factory=list)
    preset: str = ""
    files: int = 0
    # inputs that carry light on this rig whatever their names say elsewhere, each with why (the user's wiring)
    light_inputs: dict[str, str] = field(default_factory=dict)

    def hardware(self, fp: Fingerprint) -> bool:
        return (self.format, self.digitizer, self.amplifier) == (fp.format, fp.digitizer, fp.amplifier)

    def score(self, fp: Fingerprint) -> int:
        if not self.hardware(fp):
            return -1
        return 2 * (fp.channel in self.channels) + (fp.units in self.units) + (fp.protocol in self.protocols)


@dataclass
class Match:
    rig: Rig | None
    why: str
    doubts: list[str] = field(default_factory=list)


def store_path() -> Path:
    given = os.environ.get("CLAMPWISE_RIGS")
    if given:
        return Path(given)
    return Path(os.environ.get("APPDATA") or Path.home()) / "Clampwise" / "rigs.json"


class Rigs:
    """The user's rig profiles."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else store_path()
        self.rigs: dict[str, Rig] = {}
        if self.path.is_file():
            for d in json.loads(self.path.read_text(encoding="utf-8")).get("rigs", []):
                self.rigs[d["name"]] = Rig(**d)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"format": "clampwise-rigs", "rigs": [asdict(r) for r in self.rigs.values()]},
                                        indent=1), encoding="utf-8")

    def learn(self, name: str, prints: list[Fingerprint]) -> Rig:
        """Name the rig these files were recorded on (adds to a rig of that name)."""
        if not prints:
            raise ValueError("a rig is learned from at least one of its files")
        first = prints[0]
        rig = self.rigs.get(name) or Rig(name, first.format, first.digitizer, first.amplifier)
        for fp in prints:
            if not rig.hardware(fp):
                raise ValueError(f"{fp.describe()} is other hardware than {name} ({first.describe()}): "
                                 "name the files of one rig at a time")
            for attr, v in (("channels", fp.channel), ("units", fp.units), ("protocols", fp.protocol)):
                if v and v not in getattr(rig, attr):
                    getattr(rig, attr).append(v)
            rig.files += 1
        self.rigs[name] = rig
        self.save()
        return rig

    def set_preset(self, name: str, preset: str) -> None:
        self.rigs[name].preset = preset
        self.save()

    def set_light_inputs(self, name: str, inputs: dict[str, str]) -> None:
        self.rigs[name].light_inputs = dict(inputs)
        self.save()

    def forget(self, name: str) -> None:
        self.rigs.pop(name, None)
        self.save()

    def match(self, fp: Fingerprint) -> Match:
        scored = sorted(((r.score(fp), r) for r in self.rigs.values() if r.score(fp) >= 0), key=lambda x: -x[0])
        if not scored:
            return Match(None, f"no named rig has this hardware ({fp.describe()})")
        best, rig = scored[0]
        m = Match(rig, f"{rig.name}: the same hardware ({fp.describe()})")
        if fp.channel not in rig.channels:
            m.doubts.append(f"channel {fp.channel} has not been seen on {rig.name} before")
        if len(scored) > 1 and scored[1][0] == best:
            m.doubts.append(f"{scored[1][1].name} matches as well as {rig.name}")
            m.rig = None
            m.why = "two rigs match equally: name which one these files are from"
        return m


_cache: dict[str, Fingerprint] = {}


def match_file(path: str | Path, rigs: Rigs | None = None) -> Match:
    key = str(path)
    if key not in _cache:
        _cache[key] = fingerprint(path)
    return (rigs or Rigs()).match(_cache[key])
