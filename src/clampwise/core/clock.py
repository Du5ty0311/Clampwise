"""A cell's files on one clock (B1, B18.11).

Each file carries its own start time (Clampex writes it in the header;
WinLTP in each sweep file). Put together, a baseline file, a step file and
the file after it line up as one trial. Without a start time a file cannot
be placed, and the clock says so rather than guessing.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from .recording import Recording


@dataclass
class Placed:
    recording: Recording
    offset_s: float                     # the file's first sample on the cell's clock
    sweep_times_s: np.ndarray           # each sweep's start on the cell's clock


@dataclass
class CellClock:
    """Files of one cell ordered by their start time; time 0 is the first file's start."""

    placed: list[Placed] = field(default_factory=list)
    unplaced: list[tuple[Recording, str]] = field(default_factory=list)
    origin: dt.datetime | None = None

    @classmethod
    def build(cls, recordings: list[Recording]) -> "CellClock":
        timed = [r for r in recordings if r.start_time is not None]
        out = cls()
        for r in recordings:
            if r.start_time is None:
                out.unplaced.append((r, "the file has no start time; it cannot be placed on the cell's clock"))
        if not timed:
            return out
        timed.sort(key=lambda r: r.start_time)
        out.origin = timed[0].start_time
        for r in timed:
            offset = (r.start_time - out.origin).total_seconds()
            out.placed.append(Placed(r, offset, offset + np.asarray(r.sweep_start_times_s, dtype=float)))
        return out

    def offset_of(self, recording: Recording) -> float:
        for p in self.placed:
            if p.recording is recording:
                return p.offset_s
        return float("nan")

    def describe(self) -> list[str]:
        lines = [f"{p.recording.source.name}: starts at {p.offset_s:.1f} s on the cell's clock" for p in self.placed]
        lines += [f"{r.source.name}: {why}" for r, why in self.unplaced]
        return lines
