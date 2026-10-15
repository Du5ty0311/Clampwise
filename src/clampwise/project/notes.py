"""A cell folder's Notes.txt, read (never written): who the cell is and what each
file holds, in a lab's own format, for example:

    5-6 wk adolescent male rat
    Recording CA1 pyramidal neuron
    Drugs:
    DAP5 50 uM
    File (baseline): 24125268
    Depol (5s): 24125269
    5s: 24125270
    2nd 5s: 24125280

The top lines give the age, the sex, the region, the cell type and the drugs in
the bath; each "label: number" line names a file. A number that matches no file
is matched to the one file it is a digit away from, with the doubt said; a file
that disagrees with its note (a "5 s" step that lasts 2.5 s) is flagged.
Notes are evidence: they label, they never change a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

NAME = "Notes.txt"
REGIONS = ("CA1", "CA2", "CA3", "dentate gyrus", "DG", "mPFC", "PFC", "prefrontal cortex", "striatum",
           "nucleus accumbens", "thalamus", "PVT", "amygdala", "BLA", "hippocampus", "cortex", "cerebellum", "VTA")
CELL_TYPES = ("pyramidal neuron", "pyramidal cell", "interneuron", "granule cell", "medium spiny neuron",
              "Purkinje cell", "stellate cell", "dopamine neuron", "relay neuron")


@dataclass
class Entry:
    line: int                   # 1-based line in Notes.txt
    label: str                  # as written: "Depol (5s)"
    role: str                   # "baseline", "step", "after", "other"
    duration_s: float | None    # the step's duration the label gives
    number: str                 # the file number as written
    path: Path | None = None    # the file it names
    doubt: str = ""             # why the match is uncertain, or that there is none

    def describe(self) -> str:
        return f"{self.label} (Notes.txt, line {self.line})"


@dataclass
class Notes:
    path: Path
    age: str = ""
    sex: str = ""
    species: str = ""
    region: str = ""
    cell_type: str = ""
    drugs: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)       # top lines not read as any of the above
    entries: list[Entry] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)       # where a file disagrees with its note

    def entry_for(self, path: Path) -> list[Entry]:
        return [e for e in self.entries if e.path is not None and Path(e.path) == Path(path)]


def duration(text: str) -> float | None:
    """'5s', '(500ms)', '2nd 500 ms', '1 sec' -> seconds."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(ms|msec|s|sec|seconds?)\b", text, re.I)
    if not m:
        return None
    v = float(m.group(1))
    return v / 1000.0 if m.group(2).lower().startswith("ms") else v


def _role(label: str) -> str:
    low = label.lower()
    if "baseline" in low:
        return "baseline"
    if re.search(r"depol|step|stim", low):
        return "step"
    if duration(label) is not None:
        return "after"
    return "other"


def _near(number: str, stems: dict[str, Path]) -> tuple[Path | None, str]:
    """A file whose number is one digit away (missing, extra or changed)."""
    def one_apart(a: str, b: str) -> bool:
        if a == b:
            return False
        if len(a) == len(b):
            return sum(x != y for x, y in zip(a, b)) == 1
        short, long = (a, b) if len(a) < len(b) else (b, a)
        return len(long) - len(short) == 1 and any(long[:k] + long[k + 1:] == short for k in range(len(long)))

    near = [(s, p) for s, p in stems.items() if one_apart(number, s)]
    if len(near) == 1:
        s, p = near[0]
        return p, f"the note says {number}; no file has that number, and {s} is one digit away"
    if near:
        return None, f"the note says {number}; no file has that number, and {len(near)} are one digit away"
    return None, f"the note says {number}; no file in the folder has that number"


def read(folder: str | Path, files: list[Path] | None = None) -> Notes | None:
    """The folder's Notes.txt, its files matched to ``files`` (default: the folder's
    recordings); None when there is no Notes.txt."""
    folder = Path(folder)
    path = folder / NAME
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    files = files if files is not None else [p for p in folder.iterdir() if p.suffix.lower() in (".abf", ".atf")]
    stems = {p.stem: p for p in files}
    out = Notes(path)
    in_drugs = False
    for k, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            in_drugs = False
            continue
        m = re.match(r"^(?P<label>[^:]+?)\s*:\s*(?P<rest>.*)$", line)
        if m and m.group("label").lower().startswith("drug"):
            in_drugs = True
            if m.group("rest").strip():
                out.drugs.append(m.group("rest").strip())
            continue
        numbers = re.findall(r"\b\d{3,}\b", m.group("rest")) if m else []
        if m and (numbers or _role(m.group("label")) != "other" and not m.group("rest").strip()):
            label = m.group("label").strip()
            for number in numbers:              # a line with no number names a file not recorded
                e = Entry(k, label, _role(label), duration(label), number)
                if number in stems:
                    e.path = stems[number]
                else:
                    e.path, e.doubt = _near(number, stems)
                out.entries.append(e)
            continue
        if in_drugs:
            out.drugs.append(line)
            continue
        _header_line(out, line)
    return out


def _header_line(out: Notes, line: str) -> None:
    low = line.lower()
    found = False
    age = re.search(r"\b(P\d+(?:\s*-\s*P?\d+)?|\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*(?:wk|wks|weeks?|days?|d|"
                    r"months?|mo))\b", line, re.I)
    if age and not out.age:
        out.age, found = age.group(1), True
    sex = re.search(r"\b(female|male)\b", low)
    if sex and not out.sex:
        out.sex, found = sex.group(1), True
    species = re.search(r"\b(rat|mouse|mice|human|macaque)\b", low)
    if species and not out.species:
        out.species, found = species.group(1), True
    for r in REGIONS:
        if re.search(rf"\b{re.escape(r)}\b", line, re.I) and not out.region:
            out.region, found = r, True
    for c in CELL_TYPES:
        if c.lower() in low and not out.cell_type:
            out.cell_type, found = c, True
    if not found:
        out.other.append(line)


def check(notes: Notes, load) -> list[str]:
    """Where a file disagrees with its note: a step whose recorded duration is not the
    note's (``load`` reads a recording's header and epochs). Also kept on ``notes.flags``."""
    flags = []
    for e in notes.entries:
        if e.path is None:
            flags.append(f"{e.describe()}: {e.doubt}")
            continue
        if e.doubt:
            flags.append(f"{e.describe()}: {e.doubt}")
        if e.role == "step" and e.duration_s:
            try:
                rec = load(e.path)
            except Exception as exc:
                flags.append(f"{e.describe()}: {Path(e.path).name} could not be read ({exc})")
                continue
            got = step_duration(rec)
            if got is not None and abs(got - e.duration_s) > max(0.05, 0.1 * e.duration_s):
                flags.append(f"{e.describe()} names {Path(e.path).name}: the note says a {e.duration_s:g} s step; the "
                             f"file's step lasts {got:g} s (its protocol is {rec.protocol or 'unnamed'})")
    notes.flags = flags
    return flags


def step_duration(rec) -> float | None:
    """The longest epoch away from the holding level in the first sweep, in s."""
    epochs = rec.epochs[0] if rec.epochs else []
    hold = rec.holding if rec.holding is not None and rec.holding == rec.holding else None
    away = [e for e in epochs if hold is not None and abs(e.level - hold) > 1e-6 and e.end_s > e.start_s]
    if not away:
        return None
    return round(max(e.end_s - e.start_s for e in away), 4)
