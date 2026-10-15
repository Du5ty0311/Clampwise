"""WinLTP sweep files (``.P0``, ``.FP0`` and the rest of that family).

WinLTP (W.W. Anderson) saves every sweep as its own text file: a header of
quoted keys with their values (acquisition and stimulation settings), then the
samples, one row per sample, one column per AD channel. The layout is the
one documented in the WinLTP 3.01 manual, chapter 17: the ``"LTP_ASF"`` header
of WinLTP 0.90-2.32 and the ``"WinLTP"`` header of 3.00-3.01 are both read.

What the extension says (manual 4.6): the letter is what was done to the
sweep, the digit which pathway of an alternating pair it is.

    P0 P1    raw pulse sweeps        T0 T1   raw train sweeps
    BP0 BP1  artifact-blanked        BT0 BT1
    FP0 FP1  low-pass filtered       FT0 FT1
    AP0 AP1  averaged; LP = averaged and blanked; IP = averaged and filtered

One file is one sweep, so a recording here is a **series**: every file in a
folder with the same name prefix and extension, in file-number order. WinLTP
numbers files with a running number shared by the alternating sweeps
(89240217.P0, 89240218.P1, 89240219.P0...), so ``.P0`` and ``.P1``, the S0 and
S1 pathways, are two recordings.

Read from each file (B18, item 11): the sample interval, the channels and
their units (V and nA converted to mV and pA), the stimulation settings of
every output (what WinLTP delivered in that sweep, as trigger channels), the
start time (seconds after midnight, with the date), the Rs/Rm test settings
and intracellular stimulation (which say whether the recording is a cell, not
a field), and the blanking and filtering applied.
"""

from __future__ import annotations

import datetime as dt
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.recording import (ArraySource, ClampMode, Channel, ExperimentMetadata, Note, Recording, Role,
                              Stimulus)

PROCESSING = ("", "B", "F", "A", "L", "I")
PROCESSING_LABEL = {"": "raw", "B": "blanked", "F": "filtered", "A": "averaged",
                    "L": "averaged, blanked", "I": "averaged, filtered"}
SUFFIXES = {f".{p}{kind}{n}".lower() for p in PROCESSING for kind in ("P", "T") for n in (0, 1)
            if not (kind == "T" and p in ("A", "L", "I"))}
# When several processed copies of the same sweeps sit together, the one
# analyzed by default: filtered first, because that is what WinLTP measures
# when its filter is on and what the lab's own sheets were made from; then
# blanked, then raw; averaged copies last (fewer, smoothed sweeps of the same data).
PREFERENCE = {"F": 0, "B": 1, "": 2, "A": 3, "L": 4, "I": 5}
UNIT_SCALE = {"v": ("mV", 1e3), "mv": ("mV", 1.0), "uv": ("mV", 1e-3), "µv": ("mV", 1e-3),
              "pa": ("pA", 1.0), "na": ("pA", 1e3), "ua": ("pA", 1e6)}
TOKEN = re.compile(r'"[^"]*"|\S+')
STEM = re.compile(r"^(?P<prefix>.*?)(?P<number>\d{1,4})$")


# ------------------------------------------------------------------ naming


def is_winltp(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUFFIXES


@dataclass(frozen=True)
class Name:
    prefix: str
    processing: str
    number: int
    kind: str           # "P" or "T"
    pathway: str        # "0" or "1"

    @classmethod
    def of(cls, path: Path) -> "Name":
        ext = path.suffix[1:].upper()
        kind, pathway, processing = ext[-2], ext[-1], ext[:-2]
        m = STEM.match(path.stem)
        prefix, number = (m["prefix"], int(m["number"])) if m else (path.stem, 0)
        return cls(prefix, processing, number, kind, pathway)

    @property
    def series(self) -> str:
        return f"{self.prefix}.{self.processing}{self.kind}{self.pathway}"


def series_path(path: str | Path) -> Path:
    """The name a whole series is known by (8924.P0 for 89240217.P0): stable
    as sweeps are added, so a growing series keeps one result."""
    p = Path(path)
    n = Name.of(p)
    return p.with_name(n.series if n.prefix else f"{p.stem}.{n.processing}{n.kind}{n.pathway}")


def series_members(path: str | Path) -> list[Path]:
    """Every file of the series ``path`` belongs to, in file-number order."""
    p = Path(path)
    target = series_path(p).name.lower()
    members = [q for q in p.parent.iterdir() if q.is_file() and is_winltp(q) and series_path(q).name.lower() == target]
    return sorted(members, key=lambda q: Name.of(q).number)


def choose_series(paths) -> tuple[list[Path], list[str]]:
    """One file per series to analyze, and notes on what was left out: processed
    copies of the same sweeps (filtered, blanked, averaged) are one recording,
    and a lone sweep numbered inside another series (an induction train) is
    part of that experiment, not a recording of its own."""
    groups: dict[tuple, dict[str, Path]] = {}
    for raw in paths:
        p = Path(raw)
        n = Name.of(p)
        groups.setdefault((str(p.parent).lower(), n.prefix.lower(), n.kind, n.pathway), {}).setdefault(n.processing, p)
    spans = {}
    for key, versions in groups.items():
        members = series_members(next(iter(versions.values())))
        numbers = [Name.of(m).number for m in members]
        spans[key] = ((min(numbers), max(numbers)) if numbers else (0, -1), len(members))
    chosen, notes = [], []
    for key in list(groups):
        (lo, hi), count = spans[key]
        if count != 1:
            continue
        for other, ((o_lo, o_hi), o_count) in spans.items():
            if other != key and other[:2] == key[:2] and o_count > 1 and o_lo < lo < o_hi:
                p = next(iter(groups.pop(key).values()))
                notes.append(f"{p.name} is a single sweep recorded between the sweeps of another series "
                             "(an induction train, or a test); it is not analyzed on its own. Open it "
                             "directly to analyze it.")
                break
    for (_, prefix, kind, pathway), versions in sorted(groups.items()):
        best = min(versions, key=lambda v: PREFERENCE.get(v, 9))
        chosen.append(versions[best])
        others = sorted((v for v in versions if v != best), key=lambda v: PREFERENCE.get(v, 9))
        if others:
            skipped = ", ".join(f".{v}{kind}{pathway} ({PROCESSING_LABEL.get(v, v)})" for v in others)
            notes.append(f"{series_path(versions[best]).name}: analyzed the {PROCESSING_LABEL.get(best, best)} "
                         f"sweeps; skipped {skipped}, copies of the same sweeps. Open one of those files "
                         "directly to analyze it instead.")
    return chosen, notes


# ------------------------------------------------------------------ one sweep file


@dataclass
class SweepFile:
    path: Path
    version: str
    header: dict[str, list[str]]
    channels: list[str]
    units: list[str]
    data: np.ndarray                  # (n_samples, n_channels), the time column removed
    interval_s: float
    start_s: float                    # seconds after midnight
    date: str                         # YYYYMMDD
    pulses: dict[int, list[tuple[float, float]]] = field(default_factory=dict)   # output -> (on, off) s
    notes: list[str] = field(default_factory=list)
    incomplete: bool = False


def _tokens(line: str) -> list[str]:
    return [t[1:-1] if t.startswith('"') and t.endswith('"') else t for t in TOKEN.findall(line) if t != "///"]


def _num(text, default=float("nan")) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _all_numbers(tokens: list[str]) -> bool:
    return bool(tokens) and all(np.isfinite(_num(t)) for t in tokens)


def parse(path: str | Path) -> SweepFile:
    path = Path(path)
    lines = path.read_text(encoding="latin-1").splitlines()
    header: dict[str, list[str]] = {}
    data_start = None
    for i, line in enumerate(lines):
        tokens = _tokens(line)
        if _all_numbers(tokens):
            data_start = i
            break
        if tokens:
            header.setdefault(tokens[0], tokens[1:])
    if data_start is None or data_start < 2:
        raise ValueError(f"{path.name}: no sample data found; not a WinLTP sweep file?")
    first = lines[0].strip()
    if not (first.startswith('"LTP_ASF"') or first.startswith('"WinLTP"')):
        raise ValueError(f"{path.name}: the header does not start with \"WinLTP\" or \"LTP_ASF\"")
    version = " ".join(_tokens(first))
    head = [_tokens(l) for l in lines[:data_start] if _tokens(l)]
    names, units = head[-2], head[-1]
    has_time = (bool(names) and names[0] == "") or (bool(units) and units[0].lower() in ("ms", "s"))
    data = np.loadtxt(io.StringIO("\n".join(lines[data_start:])), ndmin=2)
    if has_time:
        data, names, units = data[:, 1:], names[1:], units[1:]
    interval_ms = _num((header.get("SampleInterval_ms") or ["nan"])[0])
    if not np.isfinite(interval_ms) or interval_ms <= 0:
        raise ValueError(f"{path.name}: no SampleInterval_ms in the header")
    sweep = SweepFile(path, version, header, names, units, data, interval_ms / 1e3,
                      _num((header.get("TimeStartedMidnite_s") or ["nan"])[0]),
                      (header.get("DateFileSaved") or [""])[0])
    expected = _num((header.get("NumSamples") or ["nan"])[0])
    if np.isfinite(expected) and int(expected) != data.shape[0]:
        sweep.notes.append(f"{path.name}: the header says {int(expected)} samples, the file holds {data.shape[0]}")
        sweep.incomplete = data.shape[0] < int(expected)
    sweep.pulses = _stimulation(header, version)
    return sweep


def _values(header: dict, key: str, n: int) -> list[float]:
    vals = [_num(v) for v in header.get(key, [])]
    return (vals + [float("nan")] * n)[:n]


def _stimulation(header: dict, version: str) -> dict[int, list[tuple[float, float]]]:
    """Pulse (on, off) times in seconds for each stimulation output S0-S7."""
    out = {}
    for s in range(8):
        stim = header.get(f"S{s}_Stim")
        if not stim:
            continue
        pulses = _segments(header, s, stim) if version.startswith("WinLTP") else _legacy(header, s, stim)
        if pulses:
            out[s] = pulses
    return out


def _legacy(header: dict, s: int, stim: list[str]) -> list[tuple[float, float]]:
    """WinLTP 0.90-2.32: a pre-pulse delay, then N pulses at an interval."""
    if (stim[0] if stim else "").lower() in ("", "off", "notused"):
        return []
    delay = _num((header.get(f"S{s}_PrePulseDur_ms") or ["0"])[0], 0.0)
    n = _num((header.get(f"S{s}_NumPulses") or ["1"])[0], 1.0)
    interval = _num((header.get(f"S{s}_PulseIntvl_ms") or ["0"])[0], 0.0)
    width = _num((header.get(f"S{s}_PulseDur_ms") or ["0.1"])[0], 0.1)
    width = width if width > 0 else 0.1
    if n < 1:
        return []
    return [((delay + k * interval) / 1e3, (delay + k * interval + width) / 1e3) for k in range(int(n))]


def _segments(header: dict, s: int, stim: list[str]) -> list[tuple[float, float]]:
    """WinLTP 3.x: a row of segments (Delay0 Pulses1 Off2 ...). A Delay waits
    its interval; Pulses gives NumPulses at PulseIntvl; Trains gives NumTrains
    trains every TrainPer, each of NumPulses; the first Off ends it."""
    n = len(stim)
    width = _values(header, f"S{s}_PulseDur_ms", n)
    interval = _values(header, f"S{s}_PulseIntvl_ms", n)
    count = _values(header, f"S{s}_NumPulses", n)
    period = _values(header, f"S{s}_TrainPer_ms", n)
    trains = _values(header, f"S{s}_NumTrains", n)

    def fin(x: float, default: float) -> float:
        return x if np.isfinite(x) else default

    t = 0.0
    pulses: list[tuple[float, float]] = []
    for i, name in enumerate(stim):
        kind = re.sub(r"\d+$", "", name).lower()
        if kind in ("off", "notused", ""):
            break
        w = fin(width[i], 0.1)
        w = w if w > 0 else 0.1
        if kind.startswith("delay"):
            t += fin(interval[i], 0.0)
        elif kind.startswith("pulse"):
            k_n, step = int(fin(count[i], 1)), fin(interval[i], 0.0)
            pulses += [(t + k * step, t + k * step + w) for k in range(k_n)]
            t += k_n * step
        elif kind.startswith("train"):
            k_n, step = int(fin(count[i], 1)), fin(interval[i], 0.0)
            j_n, per = int(fin(trains[i], 1)), fin(period[i], k_n * step)
            pulses += [(t + j * per + k * step, t + j * per + k * step + w) for j in range(j_n) for k in range(k_n)]
            t += j_n * per
        else:
            break           # an unknown segment: stop rather than guess
    return [(a / 1e3, b / 1e3) for a, b in pulses]


_CACHE: dict[tuple[str, int, int], SweepFile] = {}


def _parse_cached(path: Path) -> SweepFile:
    """A growing series is not re-read in full for every new sweep."""
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _CACHE:
        if len(_CACHE) > 20000:
            _CACHE.clear()
        _CACHE[key] = parse(path)
    return _CACHE[key]


# ------------------------------------------------------------------ the series as a recording


def _scale(values: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
    target, factor = UNIT_SCALE.get(unit.strip().lower(), (unit, 1.0))
    return values * factor, target


def _recording_kind(header: dict, unit: str) -> tuple[str, str]:
    """("field" | "current clamp" | "voltage clamp", why). WinLTP does not say
    what was recorded: a voltage with no current injected into a cell and no
    Rs/Rm test is an extracellular field; the identity panel can override."""
    if unit == "pA":
        return "voltage clamp", "the signal is a current"
    ic_on = any(v and v[0].lower() not in ("off", "", "notused") for k, v in header.items()
                if re.fullmatch(r"IC\d_Stim", k))
    rsrm = [_num(x) for k in ("RsRmDur_ms", "RmRsDur_ms") for x in header.get(k, [])]
    if ic_on:
        return "current clamp", "the header shows current injected into the cell (IC stimulation on)"
    if any(np.isfinite(x) and x > 0 for x in rsrm):
        return "current clamp", "the header shows an Rs/Rm test pulse"
    return "field", "a voltage with no current injected into a cell and no Rs/Rm test"


def _start_time(sweep: SweepFile) -> dt.datetime | None:
    if len(sweep.date) == 8 and sweep.date.isdigit() and np.isfinite(sweep.start_s):
        day = dt.datetime(int(sweep.date[:4]), int(sweep.date[4:6]), int(sweep.date[6:8]))
        return day + dt.timedelta(seconds=float(sweep.start_s))
    return None


def load(path: str | Path) -> Recording:
    """The whole series ``path`` belongs to, as one recording of many sweeps."""
    path = Path(path)
    members = series_members(path) if path.exists() else []
    if not members and path.parent.is_dir():
        first = next(iter(path.parent.glob(f"{path.stem}*{path.suffix}")), None)
        members = series_members(first) if first else []
    if not members:
        raise FileNotFoundError(path)
    sweeps, notes = [], []
    for p in members:
        try:
            sweeps.append(_parse_cached(p))
        except (ValueError, OSError) as exc:
            notes.append(f"skipped {p.name}: {exc}")
    if not sweeps:
        raise ValueError(f"no readable WinLTP sweeps in the series of {path.name}: " + "; ".join(notes))
    complete = [s for s in sweeps if not s.incomplete]
    for s in sweeps:
        if s.incomplete:
            notes.append(f"left out {s.path.name}: {s.data.shape[0]} of {s.header.get('NumSamples', ['?'])[0]} "
                         "samples (cut short, or still being written)")
    sweeps = complete or sweeps
    first = sweeps[0]
    same = [s for s in sweeps if np.isclose(s.interval_s, first.interval_s) and s.channels == first.channels]
    if len(same) < len(sweeps):
        notes.append(f"{len(sweeps) - len(same)} sweep file(s) left out: a different sample interval or "
                     "channels from the first sweep of the series")
    sweeps = same
    n = min(s.data.shape[0] for s in sweeps)
    if any(s.data.shape[0] != n for s in sweeps):
        notes.append(f"the sweeps differ in length; all were cut to the shortest, {n} samples")
    for s in sweeps:
        notes.extend(s.notes)

    fs = 1.0 / first.interval_s
    arrays, channels = [], []
    for c, (name, unit) in enumerate(zip(first.channels, first.units)):
        values, u = _scale(np.stack([s.data[:n, c] for s in sweeps]), unit)
        arrays.append(values)
        mode = ClampMode.from_units(u)
        role = Role.SIGNAL if c == 0 else (Role.OTHER if mode is ClampMode.UNKNOWN else Role.MONITOR)
        why = "AD0, the analyzed signal in WinLTP" if c == 0 else f"another AD channel in {u or 'no units'}"
        channels.append(Channel(c, name or f"AD{c}", u, role, True, mode, role_reason=why))
    kind, why = _recording_kind(first.header, channels[0].units)
    notes.append(f"recording kind: {kind} ({why})")

    # Stimulation outputs as trigger channels, this pathway's output first.
    name = Name.of(path)
    pathway = int(name.pathway)
    outputs = sorted({o for s in sweeps for o in s.pulses}, key=lambda o: (o != pathway, o))
    stimulus = Stimulus(source="WinLTP stimulation settings", kind="electrical")
    for out in outputs:
        trace = np.zeros((len(sweeps), n))
        for i, s in enumerate(sweeps):
            for a, b in s.pulses.get(out, []):
                lo = int(round(a * fs))
                hi = max(int(round(b * fs)), lo + 1)
                trace[i, lo:min(hi, n)] = 1.0
        if float(np.ptp(trace)) > 0:
            arrays.append(trace)
            channels.append(Channel(len(channels), f"S{out}", "TTL", Role.TRIGGER, False,
                                    role_reason="WinLTP's stimulation settings for this output"))
            if out == pathway or not stimulus.times_s:
                stimulus.times_s = [np.array([a for a, _ in s.pulses.get(out, [])]) for s in sweeps]
                stimulus.widths_s = [np.array([b - a for a, b in s.pulses.get(out, [])]) for s in sweeps]
                stimulus.channel = f"S{out}"
                stimulus.reason = f"output S{out} is this pathway's" if out == pathway else "the first output used"

    starts = np.array([s.start_s for s in sweeps], dtype=float)
    if np.all(np.isfinite(starts)):
        starts = np.concatenate([[starts[0]], starts[0] + np.cumsum(np.mod(np.diff(starts), 86400))])
        starts = starts - starts[0]
    else:
        starts = np.arange(len(sweeps)) * (n / fs)
        notes.append("sweep start times are missing from the headers; sweeps are placed end to end")
    metadata = ExperimentMetadata()
    blank = [_num(x) for k in first.header if k.startswith("AD_S") and k.endswith("_StimArtBlank_ms")
             for x in first.header[k][:1]]
    filt = [_num(x) for x in first.header.get("AD_DigFilter_Hz", [])[:1]]
    if any(np.isfinite(b) and b > 0 for b in blank):
        metadata.notes.append(Note(0.0, f"WinLTP blanked the stimulus artifact ({max(b for b in blank if np.isfinite(b)):g} ms)",
                                   "WinLTP header"))
    acquisition = filt[0] if filt and np.isfinite(filt[0]) and filt[0] > 0 else float("nan")
    return Recording(
        source=series_path(path), sample_rate_hz=fs, channels=channels, samples=ArraySource(arrays),
        sweep_start_times_s=starts, protocol=f"WinLTP {name.kind}{name.pathway} "
                                             f"{PROCESSING_LABEL.get(name.processing, name.processing)} sweeps",
        mode="episodic", start_time=_start_time(first), acquisition_filter_hz=acquisition,
        stimulus=stimulus, metadata=metadata, notes=notes,
        header={"format": "WinLTP", "winltp_version": first.version, "files": [str(s.path) for s in sweeps],
                "first_file": sweeps[0].path.name, "last_file": sweeps[-1].path.name, "date_saved": first.date,
                "recording_kind": kind, "recording_kind_reason": why,
                "slice_from_date": f"{first.date[4:6]}{first.date[6:8]}{first.date[2:4]}_S1" if len(first.date) == 8 else ""},
    )
