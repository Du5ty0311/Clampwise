"""A cell's files grouped into episodes (auto session grouping): the protocol sequences a session is made of.

- a **step episode** (DSI/DSE): the recording before the step, the step file,
  and the recording that follows it;
- an **induction episode** (LTP/LTD): the series before, the induction, the
  series after;
- a **condition change** (a drug's wash-in): the last file in one condition
  and the first in the next.

From the cell's Notes.txt when it names the files (``from_notes``); otherwise
from what the files are and when they were recorded (``infer``): a file is a
step when its protocol says so or it is one short sweep stepping away from the
holding level; an induction when its protocol names one (TBS, HFS, pairing...).
Whatever is not certain is said in the episode's doubts, never guessed
quietly; a file that belongs to no episode is left out of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .notes import Notes, step_duration

STEP_WORDS = re.compile(r"step|depol|dsi_?step|dse", re.I)
INDUCTION_WORDS = re.compile(r"tbs|theta|hfs|tetan|induct|pairing|lfs|ltp_?ind|ltd_?ind", re.I)
FOLLOWS_S = 30.0              # a recording that starts within this after a step or induction ends follows it
LEADS_S = 600.0               # and one that ends within this before it leads into it


@dataclass
class FileInfo:
    path: Path
    start: float              # s on the session's clock (the file's start time)
    end: float
    protocol: str
    kind: str                 # "step", "induction", "recording"
    step_s: float | None = None
    condition: str = ""


@dataclass
class Episode:
    index: int
    kind: str                 # "step", "induction", "condition"
    label: str                # "2.5 s step", "TBS", "control -> carbachol"
    members: list[tuple[Path, str]] = field(default_factory=list)    # (file, "before" | "step" | "induction" | "after")
    doubts: list[str] = field(default_factory=list)
    source: str = ""          # "Notes.txt" or "the files"

    def role_of(self, path: Path) -> str:
        return next((r for p, r in self.members if Path(p) == Path(path)), "")


def file_info(path: Path, rec, condition: str = "") -> FileInfo:
    """What a file is, from its header (protocol, start, epochs)."""
    start = rec.start_time.timestamp() if rec.start_time is not None else 0.0
    length = rec.duration_s if rec.is_gap_free else float(rec.sweep_start_times_s[-1]) + rec.sweep_duration_s
    proto = rec.protocol or ""
    step = step_duration(rec)
    if INDUCTION_WORDS.search(proto):
        kind = "induction"
    elif STEP_WORDS.search(proto) or (rec.n_sweeps == 1 and step is not None and step >= 0.1 and length < 15):
        kind = "step"
    else:
        kind = "recording"
    return FileInfo(Path(path), start, start + length, proto, kind, step, condition)


def from_notes(notes: Notes, infos: dict[Path, FileInfo] | None = None) -> list[Episode]:
    """One episode per step the notes name: the file before it (the note's baseline, or
    the previous step's after file), the step, and the after file of that duration."""
    out: list[Episode] = []
    before: tuple[Path, str] | None = None
    entries = [e for e in notes.entries if e.path is not None or e.role == "step"]
    for k, e in enumerate(entries):
        if e.role == "baseline" and e.path is not None:
            before = (e.path, e.describe(), e.doubt)
        if e.role != "step":
            continue
        ep = Episode(len(out) + 1, "step", f"{e.duration_s:g} s step (note)" if e.duration_s else "step",
                     source=NOTES_SOURCE)
        if before is not None:
            ep.members.append((before[0], "before"))
            if before[2]:
                ep.doubts.append(f"{before[1]}: {before[2]}")
        else:
            ep.doubts.append(f"the notes name no recording before {e.describe()}")
        if e.path is not None:
            ep.members.append((e.path, "step"))
        else:
            ep.doubts.append(f"{e.describe()}: {e.doubt}")
        after = next((a for a in entries[k + 1:] if a.role == "after"), None)
        nxt = next((a for a in entries[k + 1:] if a.role == "step"), None)
        if after is not None and (nxt is None or after.line < nxt.line):
            if after.duration_s and e.duration_s and abs(after.duration_s - e.duration_s) > 1e-9:
                ep.doubts.append(f"{after.describe()} is listed after {e.describe()} but names another duration")
            if after.path is not None:
                ep.members.append((after.path, "after"))
                before = (after.path, after.describe(), "")
            if after.doubt:
                ep.doubts.append(f"{after.describe()}: {after.doubt}")
        else:
            ep.doubts.append(f"the notes name no recording after {e.describe()}")
        for p, role in ep.members:
            if p is not None and Path(p) in (infos or {}):
                info = infos[Path(p)]
                if role == "step" and info.kind != "step":
                    ep.doubts.append(f"{Path(p).name} is named a step in the notes but holds no step")
                if role == "step" and info.step_s and e.duration_s and abs(info.step_s - e.duration_s) > \
                        max(0.05, 0.1 * e.duration_s):
                    ep.label = f"{info.step_s:g} s step (the note says {e.duration_s:g} s)"
        _timing(ep, infos)
        out.append(ep)
    return out


NOTES_SOURCE = "Notes.txt"


def infer(infos: list[FileInfo]) -> list[Episode]:
    """Episodes from the files alone, in the order they were recorded."""
    files = sorted(infos, key=lambda f: f.start)
    out: list[Episode] = []
    for k, f in enumerate(files):
        if f.kind not in ("step", "induction"):
            continue
        label = f"{f.step_s:g} s step" if f.kind == "step" and f.step_s else (f.protocol or f.kind)
        ep = Episode(len(out) + 1, f.kind, label, source="the files")
        prev = next((g for g in reversed(files[:k]) if g.kind == "recording"), None)
        if prev is not None and f.start - prev.end <= LEADS_S:
            ep.members.append((prev.path, "before"))
        else:
            ep.doubts.append(f"no recording ends within {LEADS_S / 60:g} min before {f.path.name}")
        ep.members.append((f.path, f.kind))
        nxt = files[k + 1] if k + 1 < len(files) else None
        if nxt is not None and nxt.kind == "recording" and nxt.start - f.end <= FOLLOWS_S:
            ep.members.append((nxt.path, "after"))
        elif nxt is not None and nxt.kind == "recording":
            ep.members.append((nxt.path, "after"))
            ep.doubts.append(f"{nxt.path.name} starts {nxt.start - f.end:.0f} s after {f.path.name} ends (more than "
                             f"{FOLLOWS_S:g} s): it may not be this {f.kind}'s recording")
        else:
            ep.doubts.append(f"no recording follows {f.path.name}")
        out.append(ep)
    for a, b in zip(files, files[1:]):                 # a condition changes between two files: a wash-in
        if a.condition and b.condition and a.condition != b.condition:
            out.append(Episode(len(out) + 1, "condition", f"{a.condition} -> {b.condition}",
                               [(a.path, "before"), (b.path, "after")], source="the files"))
    return out


def _timing(ep: Episode, infos: dict[Path, FileInfo] | None) -> None:
    if not infos:
        return
    step = next((p for p, r in ep.members if r == "step"), None)
    after = next((p for p, r in ep.members if r == "after"), None)
    if step in infos and after in infos and infos[after].start - infos[step].end > FOLLOWS_S:
        ep.doubts.append(f"{Path(after).name} starts {infos[after].start - infos[step].end:.0f} s after the step ends")


def group(paths: list[Path], load, notes: Notes | None = None, conditions: dict | None = None) -> list[Episode]:
    """A cell's episodes: from its notes when they name files, else from the files."""
    infos = {}
    for p in paths:
        try:
            infos[Path(p)] = file_info(Path(p), load(p), (conditions or {}).get(Path(p), ""))
        except Exception:
            continue
    if notes is not None and any(e.role == "step" for e in notes.entries):
        return from_notes(notes, infos)
    return infer(list(infos.values()))
