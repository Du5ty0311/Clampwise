"""What Clampex did, from its Lab Book (read only).

Clampex logs its events to ``Clampex LabBook.rtf`` (``C:\\ProgramData\\Molecular Devices\\pCLAMP``): plain text in
RTF, one event per paragraph, in order, with no dates (a file's start and end carry Clampex's stopwatch time).
Read here: the folder data goes to, each protocol opened, each file started and how it ended (completed or
aborted), each Membrane Test and the values it logged (Rt, Ra, Rm, Cm, Tau, holding current), holding and
digital-output changes, and Clampex opening and closing.

A recording is matched to its record by its name and its deepest folders: files are often moved after recording
(``E:\\Data\\Cell1\\...`` to ``E:\\Data\\archive\\Cell1\\...``) and names repeat across cells. The
Membrane Test logged before a file is its start, the first one after it its end (Rs then, as typed from
Clampex's Membrane Test before, but read).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .files import modified, read_shared

DEFAULT = Path(r"C:\ProgramData\Molecular Devices\pCLAMP\Clampex LabBook.rtf")
NAN = float("nan")

# Membrane Test values as Clampex logs them, to the lab's units
TEST_VALUES = {"Rt": "rt_mohm", "Ra": "ra_mohm", "Rm": "rm_mohm", "Cm": "cm_pf", "Tau": "tau_us", "Hold": "hold_pa"}
UNITS = {"gohms": 1e3, "gohm": 1e3, "mohms": 1.0, "mohm": 1.0, "kohms": 1e-3, "kohm": 1e-3, "pf": 1.0, "nf": 1e3,
         "us": 1.0, "\u00b5s": 1.0, "\u03bcs": 1.0, "ms": 1e3, "pa": 1.0, "na": 1e3, "ua": 1e6}

_PATTERNS = (
    ("data_folder", re.compile(r"^Acquired data will be written to '(?P<path>.*)'\.$")),
    ("clampex_open", re.compile(r"^Opening Clampex")),
    ("clampex_close", re.compile(r"^Closing Clampex")),
    ("protocol_opened", re.compile(r"^Protocol '(?P<path>.*)' opened\.$")),
    ("protocol_saved", re.compile(r"^Protocol '(?P<path>.*)' saved\.$")),
    ("protocol_new", re.compile(r"^New protocol started")),
    ("test_start", re.compile(r"^Membrane test initiated")),
    ("test_end", re.compile(r"^Membrane test terminated")),
    ("test_value", re.compile(r"^[A-Za-z ]+\((?P<key>Rt|Ra|Rm|Cm|Tau|Hold)\)\s*=\s*(?P<value>[-+]?[\d.]+)\s*"
                              r"(?P<unit>\S+)")),
    ("file", re.compile(r"^(?P<path>.+\.abf) (?P<how>started|completed|aborted) at (?P<h>\d+):(?P<m>\d+):(?P<s>\d+)"
                        r" stopwatch time")),
    ("trial_saved", re.compile(r"^Last Trial was saved to file (?P<path>.+\.abf)")),
    ("holding", re.compile(r"^Holding on '(?P<path>[^']*)' => (?P<value>[-+]?[\d.]+) ?(?P<unit>\S+)")),
    ("digital", re.compile(r"^Digital Outputs")),
)


@dataclass
class Entry:
    line: int
    kind: str                       # one of _PATTERNS' names, or "other"
    text: str
    path: str = ""
    value: float = NAN
    unit: str = ""
    key: str = ""                   # for a Membrane Test value: rt_mohm, ra_mohm...
    stopwatch_s: float = NAN


@dataclass
class MembraneTest:
    line: int
    values: dict[str, float] = field(default_factory=dict)     # rt_mohm, ra_mohm, rm_mohm, cm_pf, tau_us, hold_pa

    def describe(self) -> str:
        words = {"ra_mohm": ("Ra", "MOhm"), "rm_mohm": ("Rm", "MOhm"), "cm_pf": ("Cm", "pF"), "rt_mohm": ("Rt", "MOhm"),
                 "tau_us": ("Tau", "us"), "hold_pa": ("holding", "pA")}
        return ", ".join(f"{words[k][0]} {v:.4g} {words[k][1]}" for k, v in self.values.items() if k in words)


@dataclass
class FileRecord:
    path: str                       # as Clampex logged it
    line: int
    outcome: str = "started"        # completed, aborted, or started (Clampex closed first)
    protocol: str = ""              # the protocol open when it started
    folder: str = ""                # where Clampex was writing
    started_s: float = NAN          # Clampex's stopwatch
    ended_s: float = NAN
    before: MembraneTest | None = None
    after: MembraneTest | None = None

    @property
    def name(self) -> str:
        return _parts(self.path)[-1] if self.path else ""


def rtf_lines(raw: str) -> list[str]:
    """The paragraphs of a simple RTF (Clampex's: one font, ``\\par`` between events) as plain text."""
    raw = re.sub(r"\{\\fonttbl.*?\}\}", "", raw, flags=re.S)
    lines, buf, i, n = [], [], 0, len(raw)
    word = re.compile(r"\\([a-zA-Z]+)(-?\d+)? ?")
    while i < n:
        ch = raw[i]
        if ch == "\\":
            nxt = raw[i + 1] if i + 1 < n else ""
            if nxt in "\\{}":
                buf.append(nxt)
                i += 2
            elif nxt == "'":
                try:
                    buf.append(bytes([int(raw[i + 2:i + 4], 16)]).decode("cp1252"))
                except (ValueError, UnicodeDecodeError):
                    pass
                i += 4
            else:
                m = word.match(raw, i)
                if m:
                    if m.group(1) in ("par", "line"):
                        lines.append("".join(buf).strip())
                        buf = []
                    elif m.group(1) == "tab":
                        buf.append("\t")
                    i = m.end()
                else:
                    i += 1
        elif ch in "{}\r\n":
            i += 1
        else:
            buf.append(ch)
            i += 1
    if "".join(buf).strip():
        lines.append("".join(buf).strip())
    return lines


def parse(lines: list[str]) -> list[Entry]:
    out = []
    for k, text in enumerate(lines):
        if not text:
            continue
        entry = Entry(k, "other", text)
        for kind, pattern in _PATTERNS:
            m = pattern.match(text)
            if not m:
                continue
            entry.kind = kind
            g = m.groupdict()
            entry.path = g.get("path") or ""
            try:
                if kind == "test_value":
                    scale = UNITS.get(g["unit"].lower().rstrip("."), math.nan)
                    entry.key, entry.value, entry.unit = TEST_VALUES[g["key"]], float(g["value"]) * scale, g["unit"]
                elif kind == "holding":
                    entry.value, entry.unit = float(g["value"]), g["unit"]
                elif kind == "file":
                    entry.kind = f"file_{g['how']}"
                    entry.stopwatch_s = int(g["h"]) * 3600 + int(g["m"]) * 60 + int(g["s"])
            except ValueError:                    # a value Clampex never writes (".", "1.2.3"): kept as text
                entry.kind, entry.key = "other", ""
            break
        out.append(entry)
    return out


def _parts(path: str) -> list[str]:
    return [p.lower() for p in re.split(r"[\\/]+", str(path)) if p]


class LabBook:
    """Clampex's Lab Book, parsed: its entries, and a record per file with the Membrane Tests around it."""

    def __init__(self, entries: list[Entry], source: Path | None = None) -> None:
        self.source = source
        self.entries = entries
        self.files = self._files(entries)

    @classmethod
    def read(cls, path: str | Path | None = None) -> "LabBook":
        path = path or DEFAULT
        raw = read_shared(path).decode("cp1252", errors="replace")
        return cls(parse(rtf_lines(raw)), Path(path))

    @classmethod
    def from_text(cls, text: str) -> "LabBook":
        return cls(parse(text.splitlines()))

    @staticmethod
    def _files(entries: list[Entry]) -> list[FileRecord]:
        files: list[FileRecord] = []
        protocol = folder = ""
        test: MembraneTest | None = None          # the test being logged
        last: MembraneTest | None = None          # the last test that logged values
        open_record: FileRecord | None = None     # a file whose end awaits its next test
        for e in entries:
            if e.kind == "protocol_opened":
                protocol = e.path
            elif e.kind == "data_folder":
                if e.path != folder:              # a new cell: no test crosses into it or out of it
                    last = None
                    if open_record is not None and e.path != open_record.folder:
                        open_record = None
                folder = e.path
            elif e.kind == "test_start":
                test = MembraneTest(e.line)
            elif e.kind == "test_value":
                if test is None:
                    test = MembraneTest(e.line)
                test.values[e.key] = e.value
                last = test
                if open_record is not None and open_record.after is None and open_record.outcome != "started":
                    open_record.after = test
            elif e.kind == "file_started":
                before = last
                if before is None and files and files[-1].path == e.path and files[-1].outcome == "aborted":
                    before = files[-1].before         # started again after an abort, with no new test between
                rec = FileRecord(e.path, e.line, protocol=protocol, folder=folder, started_s=e.stopwatch_s,
                                 before=before)
                files.append(rec)
                open_record, last = rec, None
            elif e.kind in ("file_completed", "file_aborted"):
                for rec in reversed(files):
                    if rec.path == e.path and rec.outcome == "started":
                        rec.outcome, rec.ended_s = e.kind.split("_", 1)[1], e.stopwatch_s
                        break
            elif e.kind == "clampex_close":
                test = None
        return files

    def record(self, path: str | Path, least: int = 0) -> FileRecord | None:
        """The record of the recording at ``path``: same name, the most folders in common (at least ``least`` of
        them, from the file's own up); the later of equals."""
        parts = _parts(str(path))
        if not parts:
            return None
        best, best_score = None, -1
        for rec in self.files:
            rp = _parts(rec.path)
            if not rp or rp[-1] != parts[-1]:
                continue
            score = 0
            for a, b in zip(reversed(rp[:-1]), reversed(parts[:-1])):
                if a != b:
                    break
                score += 1
            if score >= least and score >= best_score:
                best, best_score = rec, score
        return best


_cache: dict[Path, tuple[float, LabBook]] = {}


def current(path: str | Path | None = None) -> LabBook | None:
    """Clampex's Lab Book as it is now (read again only when it has changed); None when it is not there."""
    path = Path(path or DEFAULT)
    stamp = modified(path)
    if not stamp:
        return None
    held = _cache.get(path)
    if held is None or held[0] != stamp:
        try:
            held = (stamp, LabBook.read(path))
        except OSError:
            return held[1] if held else None
        _cache[path] = held
    return held[1]
