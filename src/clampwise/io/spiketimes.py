"""Files of spike times (``.asc``, ``.spk``): one time per line in seconds, or
one column per unit (NeuroExpress's spike-train input).

A spike-time file has no trace. It is read as a recording whose one channel is
the spike train drawn as unit pulses on a 1 kHz clock (so the trace can be
shown and the file sits in a project like any other), with the times
themselves, exactly as written, in ``header["spike_times_s"]`` (one array per
unit). The spike-train analysis reads the times, never the pulses.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..core.recording import ArraySource, Channel, ClampMode, Recording, Role

SUFFIXES = {".asc", ".spk"}
CLOCK_HZ = 1000.0
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def read_times(path: str | Path) -> list[np.ndarray]:
    """Each unit's spike times (s), sorted; the columns of the file, separated by
    tabs, commas, semicolons or spaces; lines without a number are skipped."""
    cols: list[list[float]] = []
    for line in Path(path).read_text(encoding="latin-1", errors="replace").splitlines():
        vals = [float(x) for x in NUMBER.findall(line)]
        if not vals:
            continue
        while len(cols) < len(vals):
            cols.append([])
        for j, x in enumerate(vals):
            cols[j].append(x)
    units = [np.sort(np.asarray(c, float)) for c in cols if c]
    if not units or not any(u.size for u in units):
        raise ValueError(f"{Path(path).name}: no spike times in the file")
    if any(np.any(u < 0) for u in units):
        raise ValueError(f"{Path(path).name}: negative times: not a file of spike times")
    return units


def load(path: str | Path) -> Recording:
    p = Path(path)
    units = read_times(p)
    end = max(float(u.max()) for u in units if u.size) + 1.0
    n = int(np.ceil(end * CLOCK_HZ)) + 1
    trains = np.zeros((len(units), n), np.float32)
    for j, u in enumerate(units):
        np.add.at(trains[j], np.clip(np.round(u * CLOCK_HZ).astype(int), 0, n - 1), 1.0)
    channels = [Channel(j, f"unit {j + 1}" if len(units) > 1 else "spikes", "count", Role.SIGNAL if j == 0
                        else Role.OTHER, False, ClampMode.UNKNOWN,
                        role_reason="spike times from the file, drawn as unit pulses on a 1 kHz clock")
                for j in range(len(units))]
    return Recording(source=p, sample_rate_hz=CLOCK_HZ, channels=channels,
                     samples=ArraySource([trains[j][None, :] for j in range(len(units))]),
                     sweep_start_times_s=np.zeros(1), mode="gap-free",
                     header={"format": "spike times", "recording_kind": "spike_times", "spike_times_s": units},
                     notes=[f"a file of spike times: {len(units)} unit(s), {sum(u.size for u in units)} spikes over "
                            f"{end - 1:.1f} s"])
