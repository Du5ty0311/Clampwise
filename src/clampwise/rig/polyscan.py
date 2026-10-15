"""The Polygon's sessions, read from PolyScan's own folders (read only).

PolyScan (Mightex's program for the Polygon pattern illuminator) keeps each session in a folder named for when it
started, ``Documents\\Mightex\\PolyScan3\\User Data\\<account>\\<experiment>\\<template>\\<yyyymmdd_hhmmss>``:

- ``EventLog.txt``: one timestamped line per event (the session starting and stopping, each hardware trigger with its
  count, each output to the Polygon or an LED with its step, errors);
- ``SessionSummary.dat``: the experiment, the template, the session and PolyScan's version;
- ``Hardware.config``: the devices and how they are wired (``External Trigger #1`` into the Polygon's ``Trig In``,
  an LED driver into an LED, the light guide into the Polygon);
- ``Polygon #1.settings`` / ``.guiconfig``: the pattern sequence loaded, the objective, the calibration map, the
  output trigger's pulse width;
- the sequence file itself (``*.seq``): its regions of interest, one per spot;
- ``External Trigger #1.settings``: the edge the trigger answers to;
- ``Workflow.config``: the steps of an automated workflow (nearly empty when the Polygon is driven by hand or by
  its trigger alone).

The binary files are read for the text they hold. Personal details they also hold (the account's e-mail, the
computer's and drives' identifiers) are never kept. A recording is matched to the session that was running when it
was made (both programs run on this PC's clock).
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

from .files import modified, read_shared

ROOT = Path.home() / "Documents" / "Mightex"
SESSION_NAME = re.compile(r"^\d{8}_\d{6}$")
_LINE = re.compile(r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) (?P<type>\w+) (?P<text>[^\"]*)\"?(?P<detail>.*)$")
_DEVICE = re.compile(r"^(Polygon|PolyEcho|External Trigger|Software Trigger|LED Driver|LED|Dichroic|Light Guide|"
                     r"Camera|Beam Combiner|Event Response)( #\d+)?$")
_OBJECTIVE = re.compile(r"^\d{1,3}(\.\d+)?[xX]$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
BRANDS = ("Zeiss", "Olympus", "Nikon", "Leica")


def strings(data: bytes, least: int = 3) -> list[str]:
    """The runs of printable text in a binary file."""
    return [w.decode("latin-1") for w in re.findall(rb"[\x20-\x7e]{%d,}" % least, data)]


@dataclass
class Event:
    time: dt.datetime
    kind: str                         # Msg, HardwareTrigger, DigitalOutput, AnalogOutput, Error...
    text: str
    detail: dict[str, str] = field(default_factory=dict)

    @property
    def count(self) -> int | None:
        value = self.detail.get("Trigger Count")
        return int(value) if value and value.isdigit() else None


@dataclass
class Session:
    folder: Path
    name: str                                   # yyyymmdd_hhmmss
    started: dt.datetime | None = None
    stopped: dt.datetime | None = None
    ends: dt.datetime | None = None             # with no stop logged (a crash): when the next session started
    experiment: str = ""
    template: str = ""
    version: str = ""
    devices: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    externally_triggered: bool = False          # the Polygon's Trig In is wired to an external trigger
    trigger_edge: str = ""                      # RisingEdge / FallingEdge
    sequence: str = ""                          # the pattern sequence loaded (*.seq)
    regions: int = 0                            # its regions of interest (spots)
    objective: str = ""
    calibration: str = ""
    output_pulse_ms: float | None = None
    workflow: list[str] = field(default_factory=list)       # the automated workflow's steps; empty: by hand or trigger
    events: list[Event] = field(default_factory=list)

    @property
    def triggers(self) -> list[Event]:
        return [e for e in self.events if e.kind == "HardwareTrigger"]

    @property
    def errors(self) -> list[Event]:
        return [e for e in self.events if e.kind == "Error"]

    def running_at(self, when: dt.datetime) -> bool:
        start, end = self.started or _from_name(self.name), self.stopped or self.ends
        return start is not None and start <= when and (end is None or when <= end)

    def describe(self, named: bool = True) -> str:
        """``named=False``: the program's name left out (the interface names the rig's roles, Help the programs)."""
        parts = [f"{'PolyScan session' if named else 'Session'} {self.name}" +
                 (f" ({self.experiment})" if self.experiment else "")]
        if self.sequence:
            parts.append(f"pattern sequence {self.sequence}" + (f", {self.regions} region(s)" if self.regions else ""))
        if self.objective:
            parts.append(self.objective)
        if self.externally_triggered:
            parts.append(("the Polygon on its external trigger" if named else "on its external trigger") +
                         (f" ({_edge(self.trigger_edge)})" if self.trigger_edge else ""))
        leds = [d for d in self.devices if d.startswith("LED #")]
        if leds:
            parts.append(", ".join(leds))
        return "; ".join(parts)


def _edge(edge: str) -> str:
    return {"RisingEdge": "rising edge", "FallingEdge": "falling edge"}.get(edge, edge)


def _from_name(name: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(name, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _read(folder: Path, name: str) -> bytes:
    p = folder / name
    try:
        return read_shared(p) if p.is_file() else b""
    except OSError:
        return b""


def parse_events(text: str) -> list[Event]:
    out = []
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        detail = {}
        for part in m["detail"].split(","):
            key, sep, value = part.partition("=")
            if sep:
                detail[key.strip()] = value.strip()
        out.append(Event(dt.datetime.strptime(m["time"], "%Y-%m-%d %H:%M:%S.%f"), m["type"], m["text"].strip(), detail))
    return out


def read_session(folder: str | Path) -> Session:
    folder = Path(folder)
    s = Session(folder, folder.name)
    s.events = parse_events(_read(folder, "EventLog.txt").decode("utf-8", errors="replace"))
    for e in s.events:
        if e.kind == "Msg" and e.text.startswith(f"Session ({s.name})"):
            if e.text.endswith("started."):
                s.started = e.time
            elif e.text.endswith("stopped."):
                s.stopped = e.time
    s.started = s.started or _from_name(s.name)
    summary = strings(_read(folder, "SessionSummary.dat"))
    if summary:
        s.experiment = summary[0]
        s.template = summary[1] if len(summary) > 1 and not SESSION_NAME.match(summary[1]) else ""
        s.version = next((w for w in reversed(summary) if _VERSION.match(w)), "")
    hardware = strings(_read(folder, "Hardware.config"))
    s.devices = list(dict.fromkeys(w for w in hardware if _DEVICE.match(w)))
    s.models = list(dict.fromkeys(w for w in hardware if re.match(r"^(DSI|BLS)-", w)))
    for k in range(len(hardware) - 2):           # the wiring: an external trigger into the Polygon's Trig In
        if hardware[k].startswith("External Trigger") and hardware[k + 1].startswith("Polygon") \
                and hardware[k + 2] == "Trig In":
            s.externally_triggered = True
    trigger = strings(_read(folder, "External Trigger #1.settings"))
    s.trigger_edge = next((w for w in trigger if w.endswith("Edge")), "")
    polygon = strings(_read(folder, "Polygon #1.settings")) + strings(_read(folder, "Polygon #1.guiconfig"))
    seqs = [w for w in polygon if w.lower().endswith(".seq") and "\\" not in w and "/" not in w]
    s.sequence = seqs[0] if seqs else ""
    maps = [w.rsplit("\\", 1)[-1] for w in polygon if w.lower().endswith(".map")]
    s.calibration = maps[0] if maps else ""
    for k in range(len(polygon) - 1):
        if polygon[k] in BRANDS and _OBJECTIVE.match(polygon[k + 1]):
            s.objective = f"{polygon[k]} {polygon[k + 1].upper()}"
            break
    if "OutputTrigger_PulseWidth" in polygon:
        k = polygon.index("OutputTrigger_PulseWidth")
        if k + 1 < len(polygon) and polygon[k + 1].isdigit():
            s.output_pulse_ms = float(polygon[k + 1])
    if s.sequence:
        s.regions = sum(1 for w in strings(_read(folder, s.sequence)) if re.match(r"^ROI\d+$", w))
    workflow = strings(_read(folder, "Workflow.config"))
    s.workflow = [w for w in workflow if _DEVICE.match(w) or w.lower().endswith(".seq")]
    return s


_folders: dict[Path, tuple[float, list[Path]]] = {}
_cache: dict[Path, tuple[float, Session]] = {}
LIST_FOR_S = 10.0                               # the folder list is looked at again after this


def session_folders(root: str | Path | None = None) -> list[Path]:
    """Every session folder under PolyScan's user data, in the order they started (their names are the times)."""
    import time

    root = Path(root or ROOT)
    held = _folders.get(root)
    if held is None or time.monotonic() - held[0] > LIST_FOR_S:
        found = [p for p in root.glob("PolyScan*/User Data/*/*/*/*") if SESSION_NAME.match(p.name) and p.is_dir()]
        held = (time.monotonic(), sorted(found, key=lambda p: p.name))
        _folders[root] = held
    return held[1]


def session(folder: Path, following: Path | None = None) -> Session:
    """One session, read again only when its event log changes; ``following``: the next session's folder."""
    stamp = modified(folder / "EventLog.txt") or modified(folder)
    held = _cache.get(folder)
    if held is None or held[0] != stamp:
        held = (stamp, read_session(folder))
        _cache[folder] = held
    s = held[1]
    if s.stopped:
        s.ends = None
    else:
        # no stop logged: PolyScan was closed without one, or crashed. The session ends at the last thing it wrote
        # (or when the next began, if sooner); only the newest, with PolyScan running now, is still going. A light
        # taken for running when it was not would make lines read as light: it is better to miss one
        last = _last_activity(folder)
        nxt = _from_name(following.name) if following is not None else None
        if following is None and program_running():
            s.ends = None
        else:
            s.ends = min(x for x in (last, nxt) if x is not None) if (last or nxt) else s.started
    return s


def _last_activity(folder: Path) -> dt.datetime | None:
    stamps = [modified(p) for p in folder.iterdir() if p.is_file()] if folder.is_dir() else []
    stamps = [x for x in stamps if x]
    return dt.datetime.fromtimestamp(max(stamps)) if stamps else None


def program_running() -> bool:
    """Whether PolyScan is running now (its windows are looked for)."""
    from .windows import rig_windows

    try:
        return bool(rig_windows().get("light"))
    except OSError:
        return False


def running_at(when: dt.datetime, root: str | Path | None = None) -> Session | None:
    """The session running at ``when``. PolyScan runs one at a time, so only the last to start before it can be
    (read alone: there are hundreds of sessions); one with no stop logged ends at its last activity."""
    import bisect

    folders = session_folders(root)
    names = [p.name for p in folders]
    k = bisect.bisect_right(names, when.strftime("%Y%m%d_%H%M%S")) - 1
    if k < 0:
        return None
    s = session(folders[k], folders[k + 1] if k + 1 < len(folders) else None)
    return s if s.running_at(when) else None


def latest(root: str | Path | None = None) -> Session | None:
    """The newest session (the one open now, when PolyScan is running)."""
    folders = session_folders(root)
    return session(folders[-1]) if folders else None
