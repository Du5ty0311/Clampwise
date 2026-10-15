"""A Clampex file while it is being written (B19: "gap-free files while they are written"; Phase 3 of the round).

Clampex writes an ABF2 file's header sections first (protocol, inputs, outputs, epochs, strings: blocks 1-12 on
the rig's files), then the samples as they are acquired, and at the end the counts and what follows the samples
(the synch array, tags). So, while it writes, a sweep is complete once the file holds all its bytes. Each look
reads only the new complete sweeps, with read, write and delete sharing, the file opened and closed at once and
never mapped (``files.read_shared``); a partial last sweep waits for the next look. The header and the new sweeps
are written into a private copy, its counts set to those sweeps, and read by the same loader as any file, so each
sweep is measured by the analysis's own code: the membrane test's step (Rs, Rm, Cm, holding) in voltage clamp,
the resting level and noise otherwise. Nothing is written next to the data.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .files import read_shared

log = logging.getLogger("clampwise")
NAN = float("nan")
SECTIONS = 76                     # the section map: 18 entries of (block index, bytes, entries) after the file header
BLOCK = 512
PROTOCOL, ADC, EPOCH_PER_DAC = 0, 1, 5
DATA, TAG, DELTA, VOICE, SYNCH, ANNOTATION, STATS = 10, 11, 13, 14, 15, 16, 17
AFTER_DATA = (TAG, DELTA, VOICE, SYNCH, ANNOTATION, STATS)      # written at the end: not in a copy of new sweeps
EPISODIC, GAP_FREE = 5, 3
CHUNK_S = 5.0                     # a gap-free file is measured in stretches this long
MAX_PER_LOOK = 20                 # sweeps read in one look, so a look stays light during a recording


def scratch() -> Path:
    """Clampwise's own folder for the copies (``%LOCALAPPDATA%\\Clampwise\\live``), never the data's."""
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(base) / "Clampwise" / "live"


def _section(head: bytes, i: int) -> tuple[int, int, int]:
    return struct.unpack_from("<IIq", head, SECTIONS + 16 * i)


@dataclass
class SweepTest:
    """One sweep's (or one gap-free stretch's) numbers, read from the recorded trace."""

    file: str
    sweep: int
    time_s: float                 # since the file began
    clamp: str                    # "voltage clamp", "current clamp" or ""
    rs_mohm: float = NAN
    rm_mohm: float = NAN
    cm_pf: float = NAN
    holding: float = NAN          # holding current (pA) in voltage clamp, the resting level (mV) in current clamp
    noise: float = NAN            # the SD of the stretch before any step, same unit as holding
    start: object = None          # the file's start (datetime), when the header has it
    rt_mohm: float = NAN          # the test step over its steady-state current
    no_cell: bool = False         # Rt below the quality's input-resistance floor: an open pipette, no cell to measure
    seal: bool = False            # Rt above the quality's input-resistance ceiling, or no current to measure: a seal


class Growing:
    """One file as Clampex writes it. ``look()`` gives the numbers of the sweeps completed since the last look."""

    def __init__(self, path: str | Path, folder: Path | None = None) -> None:
        self.path = Path(path)
        self.folder = folder or scratch()
        self.done = 0                     # sweeps (or gap-free stretches) already measured
        self.mode: int | None = None
        self.why = ""                     # why nothing can be read from it, when so
        self._head: bytes = b""
        self._layout: tuple[int, int, int, int] | None = None      # data start, point bytes, channels, samples/sweep

    def look(self) -> list[SweepTest]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if self._layout is None and not self._read_layout(size):
            return []
        start, point_bytes, channels, per_sweep = self._layout
        sweep_bytes = per_sweep * channels * point_bytes
        complete = max(0, (size - start) // sweep_bytes)
        head = read_shared(self.path, SECTIONS + 16 * 18)
        entries = _section(head, DATA)[2] if head[:4] == b"ABF2" else 0
        if entries > 0:                   # finished: what follows the samples is not a sweep
            complete = min(complete, entries // (per_sweep * channels))
        if complete <= self.done:
            return []
        first, n = self.done, min(complete - self.done, MAX_PER_LOOK)     # a long file: the rest next look
        data = read_shared(self.path, n * sweep_bytes, start + first * sweep_bytes)
        n = len(data) // sweep_bytes
        if n == 0:
            return []
        self.done = first + n
        return self._measure(data[:n * sweep_bytes], first, n)

    def _read_layout(self, size: int) -> bool:
        head = read_shared(self.path, BLOCK)
        if len(head) < SECTIONS + 16 * 18 or head[:4] != b"ABF2":
            self.why = "no ABF2 header yet" if size < BLOCK else "not an ABF2 file"
            return False
        block, point_bytes, _ = _section(head, DATA)
        protocol = _section(head, PROTOCOL)[0] * BLOCK
        channels = _section(head, ADC)[2]                # one entry an input
        if not block or not point_bytes or not channels or not protocol:
            self.why = "its header does not say where the samples are yet"
            return False
        start = block * BLOCK
        if size < start or protocol + 26 > start:
            return False
        self._head = read_shared(self.path, start)
        # the protocol section: the operation mode, the sample interval (us, a channel), samples an episode (all
        # channels together: 40000 for 2 x 20000 on the rig's files)
        self.mode, interval_us = struct.unpack_from("<hf", self._head, protocol)
        per_episode = struct.unpack_from("<i", self._head, protocol + 22)[0]
        per_sweep = per_episode // channels if self.mode == EPISODIC else \
            int(round(CHUNK_S * 1e6 / interval_us)) if interval_us > 0 else 0
        if self.mode not in (EPISODIC, GAP_FREE) or per_sweep <= 0:
            self.why = "only episodic and gap-free files are read while they are written"
            return False
        self._layout = (start, point_bytes, channels, per_sweep)
        self.why = ""
        return True

    def _copy(self, data: bytes, n: int, first: int = 0) -> Path | None:
        """The header with its counts set to ``n`` sweeps and nothing after them, then ``data``, as a private file.
        The copy's sweeps begin at the file's sweep ``first``: each epoch's level and duration start where that
        sweep's are, so the command rebuilt from the epoch table is the sweep's own (an IV's -20 mV step, read
        alone, is not the first sweep's -100 mV; the model cell test, 2026-10-15)."""
        head = bytearray(self._head)
        _, point_bytes, _ = _section(head, DATA)
        if first and self.mode == EPISODIC:
            block, size, entries = _section(head, EPOCH_PER_DAC)
            for e in range(entries if block and size >= 22 else 0):
                at = block * BLOCK + e * size + 6       # fEpochInitLevel, fEpochLevelInc, lEpochInitDuration, ...Inc
                level, level_inc, duration, duration_inc = struct.unpack_from("<ffii", head, at)
                struct.pack_into("<ffii", head, at, level + first * level_inc, level_inc,
                                 duration + first * duration_inc, duration_inc)
        struct.pack_into("<I", head, 12, n)                                       # the episodes acquired
        struct.pack_into("<q", head, SECTIONS + 16 * DATA + 8, len(data) // max(point_bytes, 1))
        for i in AFTER_DATA:
            struct.pack_into("<IIq", head, SECTIONS + 16 * i, 0, 0, 0)
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(suffix=".abf", prefix=self.path.stem + "_", dir=self.folder)
            with os.fdopen(fd, "wb") as fh:
                fh.write(bytes(head) + data)
            return Path(name)
        except OSError:
            log.exception("a copy of %s could not be written", self.path)
            return None

    def _measure(self, data: bytes, first: int, n: int) -> list[SweepTest]:
        from ..analysis import passive, registry
        from ..io import load

        copy = self._copy(data, n if self.mode == EPISODIC else 1, first)
        if copy is None:
            return []
        out = []
        try:
            rec = load(copy)
            clamp = "voltage clamp" if rec.is_voltage_clamp else "current clamp" if rec.signal_units == "mV" else ""
            interval = rec.header.get("sweep_interval_s") or rec.duration_s
            s = registry.load()["membrane_test"].defaults()
            quality = registry.load()["quality"].defaults()
            floor, ceiling = quality["rin_low_mohm"], quality["rin_high_mohm"]
            fs = rec.sample_rate_hz
            episodic = self.mode == EPISODIC
            per = self._layout[3]
            whole = None if episodic else rec.sweep(0)              # gap-free: one sweep, cut into its stretches
            cmd = rec.command if episodic else None
            for k in range(n):
                y = rec.sweep(k) if episodic else whole[k * per:(k + 1) * per]
                index = first + k
                t = index * (interval if episodic else CHUNK_S)
                test = SweepTest(self.path.name, index, t, clamp, start=rec.start_time)
                step = passive.test_steps(rec, k, s["min_step_mv"], s["max_step_mv"], s["min_step_ms"] / 1e3) \
                    if cmd is not None and rec.is_voltage_clamp else None
                # the noise and the level before the command first changes (an IV's steps are not noise)
                edges = np.flatnonzero(np.abs(np.diff(cmd[k])) > 0.25) if cmd is not None else []
                quiet = y[: edges[0] + 1] if len(edges) and edges[0] + 1 > int(0.005 * fs) else y
                if cmd is not None and clamp == "current clamp":
                    _current_step(test, rec, k, y, quiet, s["min_step_ms"] / 1e3)
                if step is not None:
                    i0, i1, dv = step
                    m = passive.measure_step(rec.time_s, y, i0, i1, dv, fs)
                    test.rs_mohm, test.rm_mohm, test.cm_pf = m["rs_mohm"], m["rm_mohm"], m["cm_tau_pf"]
                    test.holding, test.rt_mohm = m["holding_pa"], m["rt_mohm"]
                    if test.rt_mohm < floor:
                        # a lab recording before break-in: Rt 4-5 MOhm, the pipette itself, and transient
                        # fits explaining little (R2 0.14-0.56) that gave Rs 0-1.6 MOhm: no Rs to show
                        test.no_cell = True
                        test.rs_mohm = test.rm_mohm = test.cm_pf = NAN
                    elif not test.rt_mohm <= ceiling:
                        # the model cell's PATCH (10 GOhm): 0-6 pA for 20-100 mV steps, at most one sample count
                        test.seal = True
                        test.rs_mohm = test.rm_mohm = test.cm_pf = NAN
                else:
                    test.holding = float(np.median(quiet))
                test.noise = float(np.std(quiet)) if quiet.size > 1 else NAN
                out.append(test)
            del rec
        except Exception:
            log.exception("sweeps %d-%d of %s could not be measured", first, first + n - 1, self.path)
        finally:
            gc.collect()                          # the loader maps the copy: let go of it before removing it
            _remove(copy)
        return out


def _current_step(test: SweepTest, rec, k: int, v: np.ndarray, quiet: np.ndarray, min_s: float) -> None:
    """Current clamp: the input resistance (the steady change over the current step, ``rm_mohm``), the membrane
    tau (the intrinsic analysis's own fit, 10-95% of the charging) and Cm = tau / Rin, from the sweep's first
    current step of 5-200 pA (a step of 0 pA, an IV's middle sweep, gives none)."""
    from ..analysis import passive, registry
    from ..analysis.intrinsic import tau_fit

    step = passive.test_steps(rec, k, 5.0, 200.0, min_s)
    if step is None or quiet.size < 2:
        return
    i0, i1, di = step
    base = float(np.median(quiet))
    steady = float(np.median(v[i1 - max(3, (i1 - i0) // 5):i1])) - base
    rin = steady / di * 1e3                                       # mV / pA = GOhm
    if not rin > 0:
        return
    test.rm_mohm = test.rt_mohm = rin
    seg = v[i0:i1] - base
    peak = i0 + int(np.argmax(seg * np.sign(di)))
    s = registry.load()["intrinsic"].defaults()
    tau, _, _ = tau_fit(rec.time_s, v, i0, peak, base, s["tau_start_fraction"], s["tau_end_fraction"], 0, 0,
                        rec.sample_rate_hz)
    if math.isfinite(tau):
        test.cm_pf = tau / rin * 1e3                              # ms / MOhm = nF


def warm() -> None:
    """Load what reading a sweep needs (the loader, the membrane test, the registry's settings) ahead of the first
    sweep: loaded on demand it held the first look 1.8 s (a sweep read 2.2 s after it was written, on the rig PC)."""
    import importlib

    for name in ("clampwise.io.abf", "clampwise.analysis.passive"):
        importlib.import_module(name)
    from ..analysis import registry

    registry.load()


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def newest(folder: str | Path, since: float = 0.0) -> Path | None:
    """The newest ABF file in ``folder`` changed after ``since`` (time.time()), none when there is none."""
    try:
        files = [(p.stat().st_mtime, p) for p in Path(folder).glob("*.abf")]
    except OSError:
        return None
    files = [f for f in files if f[0] >= since]
    return max(files)[1] if files else None


def rs_change(tests: list[SweepTest], reference: int = 3) -> float:
    """Rs's change from the cell's first ``reference`` measured sweeps to its last one (0.2: 20% higher)."""
    rs = [t.rs_mohm for t in tests if math.isfinite(t.rs_mohm) and t.rs_mohm > 0]
    if len(rs) <= reference:
        return NAN
    first = float(np.median(rs[:reference]))
    return rs[-1] / first - 1.0
