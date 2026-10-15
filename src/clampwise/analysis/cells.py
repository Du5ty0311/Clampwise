"""A cell's recordings analyzed together (B18 levels: recording, cell).

Some measures exist only across files: a DSI experiment is a
baseline file, a file holding the step and a file recorded after it, and only
together are they a trial (Analysis Decisions, 2026-09-29). A cell is the
recordings sharing an identity (animal, slice, cell); they are put on one
clock by their start times (``core.clock``), and the cell's analyses read the
recordings' own results from there. Nothing here re-analyzes a recording.

A recording with no cell in its identity is a cell of its own.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import __version__
from ..core.clock import CellClock
from . import registry
from .result import Flag, Provenance, Result
from .tools import ledger


def cell_key(identity, path) -> str:
    """The cell a recording belongs to: 'A12 / Slice2 / cell1' when its identity
    names the animal and the cell; else its folder when the identity names a
    cell (the lab keeps a cell's files in one folder, and 'Slice1 / cell1'
    recurs every day); else the file itself, a cell of its own."""
    animal, slc, cell = (getattr(identity, k, "") for k in ("animal", "slice", "cell"))
    if cell and animal:
        return " / ".join(p for p in (animal, slc, cell) if p)
    if cell:
        return str(Path(path).parent)
    return str(path)


@dataclass
class Member:
    name: str
    recording: Any                      # Recording (samples on demand)
    results: dict[str, Result]
    recording_type: str = ""

    def result(self, key: str) -> Result | None:
        r = self.results.get(key)
        return r if r is not None and not r.error else None


@dataclass
class Cell:
    key: str
    members: list[Member]
    clock: CellClock = field(default_factory=CellClock)

    @classmethod
    def build(cls, key: str, members: list[Member]) -> "Cell":
        return cls(key, members, CellClock.build([m.recording for m in members]))

    def placed(self) -> list[tuple[Member, float, np.ndarray]]:
        """(member, its offset on the cell's clock, each sweep's start on it), in time order."""
        out = []
        for p in self.clock.placed:
            m = next((m for m in self.members if m.recording is p.recording), None)
            if m is not None:
                out.append((m, p.offset_s, p.sweep_times_s))
        if not self.clock.placed and len(self.members) == 1:      # a cell of one file needs no start time
            m = self.members[0]
            out.append((m, 0.0, np.asarray(m.recording.sweep_start_times_s, dtype=float)))
        return out


@dataclass
class CellResult:
    key: str
    files: list[str]
    results: dict[str, Result] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    provenance: Provenance | None = None
    made_at: str = ""

    @property
    def flags(self) -> list[Flag]:
        return [f for r in self.results.values() for f in r.qc_flags]


def analyze_cell(cell: Cell, by_analysis: dict[str, dict] | None = None, disabled=()) -> CellResult:
    """Every cell-level analysis on ``cell``; one that fails is a flag, not an exception."""
    import traceback

    by_analysis = by_analysis or {}
    out = CellResult(cell.key, [m.name for m in cell.members], made_at=dt.datetime.now().isoformat(timespec="seconds"))
    out.notes += [line for line in cell.clock.describe() if "cannot be placed" in line]
    versions, used = {}, {}
    for a in registry.cell_analyses(disabled=disabled):
        try:
            resolved, changed = a.resolve(by_analysis.get(a.key, {}))
        except ValueError as exc:
            out.notes.append(f"{a.label} not run: {exc}")
            continue
        with ledger.recording() as methods:
            try:
                result = a.run(cell, resolved)
            except Exception as exc:
                result = Result(a.key, a.version, error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
                result.flags.append(Flag("software error", f"{a.label} could not be computed ({type(exc).__name__}: "
                                         f"{exc}). A problem in Clampwise, not in the recordings.", level="cell"))
        if result is None:
            continue
        result.key, result.version, result.methods = a.key, a.version, list(methods)
        result.settings, result.changed_settings = resolved, changed
        out.results[a.key] = result
        versions[a.key], used[a.key] = a.version, resolved
    out.provenance = Provenance.now(cell.key, "", __version__, versions, used)
    return out
