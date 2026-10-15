"""Axon Text Files (``.atf``): the text format Clampfit exports and NeuroExpress
reads.

The layout (Axon's ATF 1.0, as Clampfit writes it): ``ATF<tab>1.0``; the
number of header records and of data columns; that many quoted ``key=value``
records (``AcquisitionMode``, ``SweepStartTimesMS``, ``SignalsExported``, a
``Signals=`` record naming the signal of every data column); the column titles
(``"Time (s)"``, ``"Trace #1 (pA)"``, ...); then the numbers, tab-separated.

Every column of one signal is a sweep of that signal, in trace order. The
first signal in pA or mV is the recorded cell; another in the other unit is
its amplifier's monitor; one in V the trigger. A file of only a time and a
"Template" column is an event template (Clampfit's), not a recording.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..core.recording import ArraySource, Channel, ClampMode, Recording, Role

SUFFIXES = {".atf"}
TITLE = re.compile(r"^(?P<name>.*?)\s*(?:#\s*(?P<n>\d+))?\s*\((?P<units>[^)]*)\)\s*$")


class NotARecording(ValueError):
    """The file is valid ATF but holds something else (an event template)."""


def _unquote(x: str) -> str:
    x = x.strip()
    return x[1:-1] if len(x) >= 2 and x[0] == x[-1] == '"' else x


def read_header(path: str | Path) -> tuple[dict, list[str], list[str], int]:
    """(header records, the Signals= names per data column, column titles, lines before the data)."""
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        first = f.readline()
        if not first.startswith("ATF"):
            raise ValueError(f"{Path(path).name}: not an Axon Text File (no 'ATF' on its first line)")
        n_header = int(f.readline().split()[0])
        records, signals = {}, []
        for _ in range(n_header):
            line = f.readline().rstrip("\r\n")
            parts = line.split("\t")
            key, _, value = _unquote(parts[0]).partition("=")
            if key == "Signals":
                signals = [_unquote(x) for x in parts[1:] if _unquote(x)]
            else:
                records[key] = value
        titles = [_unquote(x) for x in f.readline().rstrip("\r\n").split("\t") if x.strip()]
    return records, signals, titles, n_header + 3


def load(path: str | Path) -> Recording:
    import pandas as pd

    p = Path(path)
    records, signals, titles, skip = read_header(p)
    if len(titles) == 2 and titles[1].lower().startswith("template"):
        raise NotARecording(f"{p.name}: an event template, not a recording; use it as a template file")
    data = pd.read_csv(p, sep="\t", skiprows=skip, header=None, engine="c", dtype=float,
                       encoding="latin-1").to_numpy()
    data = data[:, :len(titles)]
    has_time = titles[0].lower().startswith("time")
    if not has_time:
        raise ValueError(f"{p.name}: no time column")
    m = TITLE.match(titles[0])
    t = data[:, 0] * (1e-3 if m and m.group("units").strip().lower() == "ms" else 1.0)
    dt = np.median(np.diff(t)) if t.size > 1 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"{p.name}: the time column does not advance")
    fs = 1.0 / dt
    # columns grouped by signal, each column a sweep
    groups: dict[str, list[tuple[int, int, str]]] = {}
    order: list[str] = []
    for j, title in enumerate(titles[1:], start=1):
        tm = TITLE.match(title)
        units = tm.group("units").strip() if tm else ""
        trace = int(tm.group("n")) if tm and tm.group("n") else j
        name = signals[j - 1] if j - 1 < len(signals) else (tm.group("name").strip() if tm else title)
        key = f"{name} ({units})"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((trace, j, units))
    n_sweeps = max(len(v) for v in groups.values())
    arrays, channels = [], []
    first_signal = None
    for key in order:
        cols = sorted(groups[key])
        units = cols[0][2]
        name = key.rsplit(" (", 1)[0]
        block = np.full((n_sweeps, data.shape[0]), np.nan)
        for s_i, (_, j, _) in enumerate(cols):
            block[s_i] = data[:, j]
        u = units.lower()
        if u in ("pa", "na", "mv") and first_signal is None:
            first_signal = u
            mode = ClampMode.VOLTAGE_CLAMP if u in ("pa", "na") else ClampMode.CURRENT_CLAMP
            ch = Channel(len(arrays), name, units, Role.SIGNAL, False, mode,
                         role_reason="the first signal in pA or mV: the recorded cell")
        elif u in ("pa", "na", "mv") and (u == "mv") != (first_signal == "mv"):
            ch = Channel(len(arrays), name, units, Role.MONITOR, False, ClampMode.UNKNOWN,
                         role_reason="the same amplifier's other quantity: its monitor")
        elif u in ("v", "ttl"):
            ch = Channel(len(arrays), name, units, Role.TRIGGER, False, ClampMode.UNKNOWN,
                         role_reason="a signal in volts: a trigger or stimulus line")
        else:
            ch = Channel(len(arrays), name, units, Role.OTHER, False, ClampMode.UNKNOWN,
                         role_reason="another recorded signal")
        arrays.append(block)
        channels.append(ch)
    if first_signal is None:
        what = ", ".join(c.units for c in channels if c.units) or "columns without units"
        raise NotARecording(f"{p.name}: no signal in pA or mV ({what}): a stimulus waveform for Clampex, not a "
                            "recording")
    # no command in an ATF: the amplifier's monitor of the other quantity is what was
    # applied (the current injected in current clamp, the voltage in voltage clamp); of
    # several, the one that moves (an idle headstage's reads flat: CRCNS hc-10's IN 2)
    monitors = [c for c in channels if c.role is Role.MONITOR]
    mon = max(monitors, key=lambda c: float(np.subtract(*np.nanpercentile(arrays[c.index], [99.5, 0.5]))),
              default=None)
    if mon is not None:
        channels[mon.index] = Channel(mon.index, mon.name, mon.units, Role.COMMAND, False, ClampMode.UNKNOWN,
                                      role_reason="the amplifier's monitor of what was applied, standing in for the "
                                      "command (an ATF holds no command)" + (
                                          f"; of the {len(monitors)} in {mon.units}, the one that moves"
                                          if len(monitors) > 1 else ""))
    starts = records.get("SweepStartTimesMS", "")
    try:
        sweep_t = np.array([float(x) for x in starts.split(",") if x.strip()], float)[:n_sweeps] / 1e3
    except ValueError:
        sweep_t = np.array([])
    if sweep_t.size != n_sweeps:
        sweep_t = np.arange(n_sweeps) * data.shape[0] / fs
    acq = records.get("AcquisitionMode", "")
    mode = "gap-free" if "gap" in acq.lower() or n_sweeps == 1 and "episodic" not in acq.lower() else "episodic"
    notes = [f"an Axon Text File ({acq or 'acquisition mode not given'}): {n_sweeps} sweep(s) of "
             f"{', '.join(c.name + ' (' + c.units + ')' for c in channels)}; no command or epoch table is in the file"]
    return Recording(source=p, sample_rate_hz=fs, channels=channels, samples=ArraySource(arrays),
                     sweep_start_times_s=sweep_t, protocol=records.get("Comment", ""), mode=mode,
                     header={"format": "ATF", **{k: v for k, v in records.items() if k != "SweepStartTimesMS"}},
                     notes=notes)
