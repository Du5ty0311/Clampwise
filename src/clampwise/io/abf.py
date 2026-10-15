"""Axon Binary Format (Clampex ``.abf``, ABF1 and ABF2, episodic and gap-free).

The header is read with pyabf (the documented reader of the format); the
samples are **memory-mapped** from the file's data block and scaled on
access, so opening a file costs its header and a 60-minute recording is read
only where it is drawn or analyzed (B18, item 3). The scaling is pyabf's own
(instrument scale factor, signal gain, programmable and telegraphed gain,
ADC range and resolution, offsets), and a test checks the mapped samples
against pyabf's ``sweepY`` bit for bit.

Everything Clampex records is kept (B18, item 11): the protocol name and
path, the epoch table and holding level of the command, the digital outputs
per epoch, telegraphed gain, filter and mode, the signal low-pass filter, the
file's start time, tags and comments typed during the recording, and the
start-to-start interval of the sweeps.

Channel roles are decided from names and units, and the reason is kept on
each channel, so nothing is chosen silently:

- the recorded signal: the first input in mV or pA (the primary channel);
- a monitor: another input from the same amplifier in the command's units
  (the current monitor of a current-clamp cell, the voltage monitor of a
  voltage-clamp cell);
- a trigger: an input whose name says TTL (``IN 8``, ``AM_Stim``, ``BlueLED``,
  ``IN 7``) or that is in volts and toggles;
- light: an output or input named for an LED, laser, light or Polygon;
- the command: the output of the amplifier that drives the primary channel,
  rebuilt from the epoch table as Clampex played it;
- digital outputs that toggle: triggers, commanded.

A file still being written by Clampex lacks the sections stored after the
data; ``read_header`` says so plainly (``live.py`` reads those).
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.recording import ClampMode, Channel, Epoch, ExperimentMetadata, Note, Recording, Role

GAP_FREE = 3
EPISODIC = 5
MODES = {1: "variable-length", 2: "fixed-length", GAP_FREE: "gap-free", 4: "high-speed oscilloscope",
         EPISODIC: "episodic"}
DIGITAL_OUTPUTS = 8
HOLDING_AGREE_MV = 2.0     # a recorded monitor further than this from the output's holding is believed instead

TRIGGER_NAME = re.compile(r"\b(in\s*\d+|ttl|trig|stim|isolator|am[_ ]?stim|sync|frame)\b", re.IGNORECASE)
LIGHT_NAME = re.compile(r"(led|laser|light|polygon|dmd|opto|photo)", re.IGNORECASE)


@dataclass
class AbfHeader:
    """What the file says about itself, without its samples."""

    path: Path
    version: str
    operation_mode: int
    sample_rate_hz: float
    n_channels: int
    n_sweeps: int
    sweep_points: int                    # samples per sweep per channel
    data_byte_start: int
    point_bytes: int                     # 2 (int16) or 4 (float32)
    data_point_count: int
    gains: list[float]
    offsets: list[float]
    adc_names: list[str]
    adc_units: list[str]
    dac_names: list[str]
    dac_units: list[str]
    holding: list[float]
    protocol: str
    protocol_path: str
    start_time: dt.datetime | None
    sweep_interval_s: float
    tags: list[tuple[float, str]]
    telegraph_enabled: list[bool]
    telegraph_gain: list[float]
    telegraph_filter_hz: list[float]
    telegraph_mode: list[int]
    signal_lowpass_hz: list[float]
    creator: str = ""
    comment: str = ""
    uniform_sweeps: bool = True
    active_dac: int = 0
    # Event-driven files (operation modes 1 and 2): where each sweep starts (the
    # synch array), and the array's time unit in microseconds (0: in samples).
    synch_starts: list[int] = field(default_factory=list)
    synch_time_unit_us: float = 0.0

    @property
    def mode(self) -> str:
        return "gap-free" if self.operation_mode == GAP_FREE else "episodic"

    @property
    def sweep_duration_s(self) -> float:
        return self.sweep_points / self.sample_rate_hz

    @property
    def duration_s(self) -> float:
        return self.data_point_count / self.n_channels / self.sample_rate_hz


def _finite(values, n: int, default=float("nan")) -> list[float]:
    out = []
    for i in range(n):
        try:
            v = float(values[i])
        except (TypeError, ValueError, IndexError):
            v = default
        out.append(v)
    return out


def read_header(path: str | Path) -> AbfHeader:
    """The header of a finished ABF file. A damaged or unfinished file raises a
    plain ValueError that says so (pyabf's own errors say nothing about the file)."""
    import pyabf

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"{path.name} is empty (0 bytes); it may still be being written by Clampex")
    try:
        abf = pyabf.ABF(str(path), loadData=False)
    except Exception as exc:
        raise ValueError(f"{path.name} is not a readable ABF file: damaged, cut short, or still being "
                         f"written by Clampex ({type(exc).__name__}: {exc})") from exc
    return _header_from(abf, path)


def _header_from(abf, path: Path) -> AbfHeader:
    n = int(abf.channelCount)
    v1 = getattr(abf, "_headerV1", None)
    adc = getattr(abf, "_adcSection", None)
    if adc is not None:
        idx = list(abf.channelList) if hasattr(abf, "channelList") else list(range(n))
        tele_on = [bool((adc.nTelegraphEnable or [0] * n)[i]) for i in idx]
        tele_gain = _finite([adc.fTelegraphAdditGain[i] for i in idx], n)
        tele_filter = _finite([adc.fTelegraphFilter[i] for i in idx], n)
        tele_mode = [int((adc.nTelegraphMode or [0] * n)[i] or 0) for i in idx]
        lowpass = _finite([adc.fSignalLowpassFilter[i] for i in idx], n)
    elif v1 is not None:
        idx = list(abf.channelList)
        tele_on = [bool(v1.nTelegraphEnable[i]) for i in idx]
        tele_gain = _finite([v1.fTelegraphAdditGain[i] for i in idx], n)
        tele_filter = _finite([v1.fTelegraphFilter[i] for i in idx], n)
        tele_mode = [0] * n
        lowpass = _finite([getattr(v1, "fSignalLowpassFilter", [float("nan")] * 16)[i] for i in idx], n)
    else:
        tele_on, tele_gain, tele_filter, tele_mode, lowpass = ([False] * n, [float("nan")] * n,
                                                               [float("nan")] * n, [0] * n, [float("nan")] * n)
    tags = []
    for t, text in zip(getattr(abf, "tagTimesSec", []) or [], getattr(abf, "tagComments", []) or []):
        tags.append((float(t), str(text).strip()))
    start = getattr(abf, "abfDateTime", None)
    uniform = True
    synch = getattr(abf, "_synchArraySection", None)
    synch_starts: list[int] = []
    if synch is not None and getattr(synch, "lLength", None):
        uniform = len(set(synch.lLength)) <= 1
        synch_starts = [int(x) for x in (getattr(synch, "lStart", None) or [])]
    synch_unit = float(getattr(getattr(abf, "_protocolSection", None), "fSynchTimeUnit", 0.0) or 0.0)
    comment = ""
    try:
        strings = abf._stringsSection._indexedStrings
        comment = strings[abf._protocolSection.lFileCommentIndex] if abf._protocolSection.lFileCommentIndex else ""
    except Exception:
        comment = str(getattr(v1, "sFileCommentNew", "") or "") if v1 is not None else ""
    return AbfHeader(
        path=path, version=str(abf.abfVersionString), operation_mode=int(abf.nOperationMode),
        sample_rate_hz=float(abf.dataRate), n_channels=n, n_sweeps=int(abf.sweepCount),
        sweep_points=int(abf.sweepPointCount), data_byte_start=int(abf.dataByteStart),
        point_bytes=int(abf.dataPointByteSize), data_point_count=int(abf.dataPointCount),
        gains=[float(g) for g in abf._dataGain], offsets=[float(o) for o in abf._dataOffset],
        adc_names=[str(x) for x in abf.adcNames], adc_units=[str(x) for x in abf.adcUnits],
        dac_names=[str(x) for x in abf.dacNames], dac_units=[str(x) for x in abf.dacUnits],
        holding=_finite(list(abf.holdingCommand or []), len(abf.dacNames)),
        protocol=str(abf.protocol or ""), protocol_path=str(abf.protocolPath or ""),
        start_time=start if isinstance(start, dt.datetime) else None,
        sweep_interval_s=float(abf.sweepIntervalSec), tags=tags,
        telegraph_enabled=tele_on, telegraph_gain=tele_gain, telegraph_filter_hz=tele_filter,
        telegraph_mode=tele_mode, signal_lowpass_hz=lowpass,
        creator=str(getattr(abf, "creator", "") or ""), comment=comment.strip(), uniform_sweeps=uniform,
        active_dac=int(getattr(getattr(abf, "_protocolSection", None), "nActiveDACChannel", 0) or 0),
        synch_starts=synch_starts, synch_time_unit_us=synch_unit,
    )


EVENT_DRIVEN = (1, 2, 4)        # triggered sweeps, their starts in the synch array


def event_sweep_times(header: AbfHeader) -> tuple[np.ndarray, bool]:
    """(each sweep's start in seconds, whether those are real times) for an
    event-driven file, from its synch array. Clampex stores where each
    triggered sweep began; a file of events saved from Clampfit stores them
    back to back (each start right after the previous sweep ends), so the
    times between events are not in it."""
    starts = np.asarray(header.synch_starts[:header.n_sweeps], dtype=float)
    if starts.size != header.n_sweeps or starts.size == 0:
        return np.zeros(max(header.n_sweeps, 1)), False
    if header.synch_time_unit_us > 0:
        t = starts * header.synch_time_unit_us * 1e-6
    else:                                   # in samples, counted over every channel
        t = starts / (header.sample_rate_hz * max(header.n_channels, 1))
    t = t - t[0]
    gaps = np.diff(t)
    back_to_back = gaps.size > 0 and bool(np.all(gaps <= header.sweep_duration_s + 2.0 / header.sample_rate_hz))
    return t, not back_to_back


# ------------------------------------------------------------------ samples, on demand


class MappedSource:
    """The file's samples, memory-mapped and scaled when read.

    Clampex stores one block of samples with the channels interleaved
    (sample 0 of channel 0, sample 0 of channel 1, ...); reshaped to
    (n_points, n_channels) the map costs nothing until a column is read.
    Extra channels beyond the file's inputs (the command rebuilt from the
    epoch table, digital outputs, a second output) are made by callables,
    once, and kept.
    """

    def __init__(self, header: AbfHeader) -> None:
        self.header = header
        self.n_recorded = header.n_channels
        self.n_sweeps = max(header.n_sweeps, 1)
        self.n_samples = header.sweep_points
        self._extra: dict[int, object] = {}
        self._cache: dict[int, np.ndarray] = {}
        self._map = None
        self._open()

    def _open(self) -> None:
        dtype = np.int16 if self.header.point_bytes == 2 else np.float32
        count = self.header.data_point_count
        self._map = np.memmap(str(self.header.path), dtype=dtype, mode="r", offset=self.header.data_byte_start,
                              shape=(count // self.header.n_channels, self.header.n_channels))

    # A mapped source crosses to a worker process as its header, never its
    # samples (a batch would otherwise pickle whole files); the map is reopened
    # on the other side and the virtual channels rebuilt on first read.
    def __getstate__(self) -> dict:
        return {"header": self.header, "n_recorded": self.n_recorded, "n_sweeps": self.n_sweeps,
                "n_samples": self.n_samples, "_extra": self._extra, "_cache": {}}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._map = None
        self._open()

    @property
    def n_channels(self) -> int:
        return self.n_recorded + len(self._extra)

    def add_virtual(self, make) -> int:
        """Register a channel computed on first read: ``make() -> (n_sweeps, n_samples)``."""
        index = self.n_channels
        self._extra[index] = make
        return index

    def _rows(self, sweeps) -> list[int]:
        if sweeps is None:
            return list(range(self.n_sweeps))
        if isinstance(sweeps, slice):
            return list(range(*sweeps.indices(self.n_sweeps)))
        return [int(s) for s in sweeps]

    def read(self, channel: int, sweeps=None) -> np.ndarray:
        if channel >= self.n_recorded:
            if channel not in self._cache:
                self._cache[channel] = np.atleast_2d(np.asarray(self._extra[channel](), dtype=float))
            full = self._cache[channel]
            return full if sweeps is None else full[self._rows(sweeps)]
        rows = self._rows(sweeps)
        spc = self.n_samples
        gain, offset = self.header.gains[channel], self.header.offsets[channel]
        # pyabf's own arithmetic (float32, multiply, then add), so the samples
        # match its sweepY bit for bit; widened to float64 afterwards, exactly.
        if sweeps is None and len(rows) == self.n_sweeps:
            out = np.asarray(self._map[: self.n_sweeps * spc, channel], dtype=np.float32).reshape(self.n_sweeps, spc)
        else:
            out = np.empty((len(rows), spc), dtype=np.float32)
            for k, s in enumerate(rows):
                out[k] = self._map[s * spc:(s + 1) * spc, channel]
        if self.header.point_bytes == 2:
            out = np.multiply(out, gain)
            out = np.add(out, offset)
        return out.astype(np.float64)


# ------------------------------------------------------------------ roles, epochs, outputs


def _amplifier(name: str) -> str:
    """The amplifier part of a channel name: "Right V" -> "right", "I_MTest 2" -> "i_mtest"."""
    return (name or "").strip().split(" ")[0].lower()


def _roles(header: AbfHeader) -> list[Channel]:
    """Each input's role from its name and units, with the reason."""
    channels: list[Channel] = []
    primary = None
    for i, (name, units) in enumerate(zip(header.adc_names, header.adc_units)):
        if primary is None and ClampMode.from_units(units) is not ClampMode.UNKNOWN:
            primary = i
    for i, (name, units) in enumerate(zip(header.adc_names, header.adc_units)):
        mode = ClampMode.from_units(units)
        gain = header.telegraph_gain[i] if header.telegraph_enabled[i] else float("nan")
        if i == primary:
            role, why = Role.SIGNAL, f"the first input in {units}: the recorded cell or field"
        elif primary is not None and mode is not ClampMode.UNKNOWN and \
                _amplifier(name) == _amplifier(header.adc_names[primary]) and \
                mode is not ClampMode.from_units(header.adc_units[primary]):
            role, why = Role.MONITOR, f"the same amplifier's other quantity ({units}): its monitor, not a signal"
        elif LIGHT_NAME.search(name):
            role, why = Role.LIGHT, "named for a light source"
        elif TRIGGER_NAME.search(name) or units.strip().lower() in ("v", ""):
            role, why = Role.TRIGGER, ("named like a TTL line" if TRIGGER_NAME.search(name)
                                       else f"an input in {units or 'no units'}: a TTL or an auxiliary line")
        elif mode is not ClampMode.UNKNOWN:
            role, why = Role.MONITOR, f"a second input in {units} from another amplifier"
        else:
            role, why = Role.OTHER, f"an input in {units}"
        channels.append(Channel(i, name or f"ADC{i}", units, role, True, mode, gain,
                                bool(header.telegraph_enabled[i]), why))
    return channels


def _epoch_tables(abf) -> dict[int, object]:
    """pyabf's epoch table per DAC, built once per file (pyabf rebuilds it on
    every sweep otherwise, which made a file of 810 sweeps take 25 s to open)."""
    import warnings

    import pyabf.waveform

    tables = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # pyabf warns on epoch types it does not draw (no epochs at all)
        for dac in range(len(abf.dacNames)):
            try:
                tables[dac] = pyabf.waveform.EpochTable(abf, dac)
            except Exception:
                continue
    return tables


def _epochs(table, n_sweeps: int, fs: float) -> list[list[Epoch]]:
    out: list[list[Epoch]] = []
    try:
        for i in range(n_sweeps):
            w = table.epochWaveformsBySweep[i]
            sweep = []
            for k, (p1, p2) in enumerate(zip(w.p1s, w.p2s)):
                digital = tuple(int(b) for b in (w.digitalStates[k] if k < len(w.digitalStates) else []))
                sweep.append(Epoch(str(w.types[k]).lower() if k < len(w.types) else "", p1 / fs, p2 / fs,
                                   float(w.levels[k]) if k < len(w.levels) else float("nan"), digital))
            out.append(sweep)
    except Exception:
        return []
    return out


def _waveforms(table, n_sweeps: int, n_samples: int) -> np.ndarray:
    out = np.full((n_sweeps, n_samples), np.nan)
    for i in range(n_sweeps):
        try:
            w = table.epochWaveformsBySweep[i].getWaveform()
        except Exception:        # an epoch table pyabf cannot draw (a file with no real epochs): the holding
            continue
        n = min(w.size, n_samples)
        out[i, :n] = w[:n]
    # Clampex holds the first and last 1/64 of a sweep at the holding level;
    # pyabf leaves them NaN, which is the holding, not a missing value.
    return out


def _digital(table, bit: int, n_sweeps: int, n_samples: int) -> np.ndarray:
    out = np.zeros((n_sweeps, n_samples))
    for i in range(n_sweeps):
        d = table.epochWaveformsBySweep[i].getDigitalWaveform(bit)
        n = min(d.size, n_samples)
        out[i, :n] = d[:n]
    return out


def _digital_toggles(table, bit: int, n_sweeps: int, n_samples: int) -> bool:
    """Whether ``_digital`` of this bit holds more than one value, worked out from
    the epochs as pyabf paints them (each epoch's state over its span, later over
    earlier, 5 where none reaches) rather than drawn: a long sweep is not built
    eight times to find the bits that do nothing. Raises where drawing would."""
    values: set[int] = set()
    for i in range(n_sweeps):
        w = table.epochWaveformsBySweep[i]
        size = w.p2s[-1]
        if size < 0:
            raise ValueError("negative dimensions are not allowed")
        n = min(size, n_samples)
        if n < n_samples:
            values.add(0)                       # the rest of the row, left at zero
        spans = []
        for k in range(len(w.levels)):
            a, b, _ = slice(w.p1s[k], w.p2s[k]).indices(size)
            spans.append((min(a, n), min(b, n), int(w.digitalStates[k][bit])))
        cuts = sorted({0, n} | {x for a, b, _ in spans for x in (a, b)})
        for a, b in zip(cuts, cuts[1:]):
            if b > a:
                values.add(next((s for p, q, s in reversed(spans) if p <= a and b <= q), 5))
    return len(values) > 1                  # every sweep looked at: one that cannot be drawn raises, as drawing does


def _varies(table, n_sweeps: int, n_samples: int) -> bool:
    """Whether a DAC's waveform plays anything other than a flat line: its finite
    values, laid out as ``_waveforms`` does, are not all one level (read sweep by
    sweep, so a long sweep is not copied whole to answer it)."""
    lo, hi = np.inf, -np.inf
    try:
        for i in range(n_sweeps):
            try:
                w = table.epochWaveformsBySweep[i].getWaveform()
            except Exception:    # an epoch table pyabf cannot draw: that sweep stays at the holding
                continue
            v = np.asarray(w[: min(w.size, n_samples)], dtype=float)
            if v.size == 0:
                continue
            a, b = float(np.fmin.reduce(v)), float(np.fmax.reduce(v))     # NaN left out
            if not (np.isfinite(a) and np.isfinite(b)):                     # only NaN, or an infinity
                finite = v[np.isfinite(v)]
                if finite.size == 0:
                    continue
                a, b = float(finite.min()), float(finite.max())
            lo, hi = min(lo, a), max(hi, b)
            if hi > lo:
                return True
    except Exception:
        return False
    return False


def load(path: str | Path, *, primary_channel: int | None = None) -> Recording:
    """The recording in an ABF file, its samples mapped, everything else read."""
    import pyabf

    header = read_header(path)
    path = header.path
    if header.sweep_points == 0:
        raise ValueError(f"{path.name}: the file holds no samples (the recording stopped before its first sweep)")
    if not header.uniform_sweeps:
        raise ValueError(f"{path.name}: its sweeps are not all the same length (a variable-length file); "
                         "Clampwise reads episodic and gap-free files with equal sweeps")
    abf = pyabf.ABF(str(path), loadData=False)
    source = MappedSource(header)
    channels = _roles(header)
    notes: list[str] = []
    if primary_channel is not None:
        channels = [Channel(c.index, c.name, c.units, Role.SIGNAL if c.index == primary_channel else
                            (Role.OTHER if c.role is Role.SIGNAL else c.role), c.recorded, c.clamp_mode,
                            c.gain, c.telegraphed, "chosen as the recorded signal" if c.index == primary_channel
                            else c.role_reason) for c in channels]
    signal = next((c for c in channels if c.role is Role.SIGNAL), None)
    if signal is None:
        notes.append("no input is in mV or pA; the first input is taken as the recorded signal")
        c0 = channels[0]
        channels[0] = Channel(c0.index, c0.name, c0.units, Role.SIGNAL, True, c0.clamp_mode, c0.gain,
                              c0.telegraphed, "the first input (none is in mV or pA)")
        signal = channels[0]

    n_sweeps, n_samples, fs = source.n_sweeps, source.n_samples, header.sample_rate_hz
    tables = _epoch_tables(abf) if header.operation_mode == EPISODIC else {}
    epochs: list[list[Epoch]] = []
    holding = float("nan")
    # The command: the output paired with the primary input (the same index) when
    # it plays something; else an output of the same amplifier in the expected
    # units (a second amplifier recorded on input 0 is driven from the output
    # paired with its other input).
    expected = "pA" if signal.clamp_mode is ClampMode.CURRENT_CLAMP else "mV"
    plays: dict[int, bool] = {}

    def varies(dac: int) -> bool:               # each output judged once, though two loops ask
        if dac not in plays:
            plays[dac] = _varies(tables[dac], n_sweeps, n_samples)
        return plays[dac]

    command_dac = None
    candidates = [signal.index] + [d for d in range(len(header.dac_names)) if d != signal.index]
    for dac in candidates:
        table = tables.get(dac)
        if table is None:
            continue
        units = header.dac_units[dac] if dac < len(header.dac_units) else ""
        same_amp = _amplifier(header.dac_names[dac]) == _amplifier(signal.name) if dac != signal.index else True
        if varies(dac) and (dac == signal.index or
                                                    (same_amp and units.strip().lower() == expected.lower())):
            command_dac = dac
            break
    if command_dac is None and signal.index in tables:
        command_dac = signal.index          # a flat command still says the holding
    if command_dac is not None:
        table = tables[command_dac]
        epochs = _epochs(table, n_sweeps, fs)
        holding = header.holding[command_dac] if command_dac < len(header.holding) else float("nan")
        units = header.dac_units[command_dac] if command_dac < len(header.dac_units) else expected
        index = source.add_virtual(lambda t=table: np.nan_to_num(_waveforms(t, n_sweeps, n_samples), nan=holding))
        channels.append(Channel(index, header.dac_names[command_dac] or f"DAC{command_dac}", units, Role.COMMAND,
                                False, ClampMode.UNKNOWN, role_reason=
                                f"the amplifier's command, rebuilt from the epoch table of output {command_dac}"))
        if command_dac != signal.index:
            notes.append(f"the command comes from output {command_dac} ({header.dac_names[command_dac]}), "
                         "the one that drives this amplifier")
    elif header.operation_mode != EPISODIC and signal.index < len(header.holding) \
            and np.isfinite(header.holding[signal.index]):
        # a gap-free or event-driven file has no epoch table, but Clampex still
        # records the holding level of the output paired with the input
        holding = header.holding[signal.index]
        notes.append(f"holding {holding:g} {header.dac_units[signal.index] if signal.index < len(header.dac_units) else ''}"
                     f" as Clampex's output {signal.index} was set; a holding set on the amplifier itself is not in "
                     "the file")
    # In voltage clamp the amplifier's own membrane-potential monitor, when it was
    # recorded, says where the cell was held: a holding set on the amplifier's
    # front panel never reaches Clampex's output (NeuroExpress's mEPSC test file:
    # output 0 mV, the monitor -70.9 mV, and the inward events read as outward).
    if signal.clamp_mode is ClampMode.VOLTAGE_CLAMP:
        monitor = next((c for c in channels if c.role is Role.MONITOR and c.units.strip().lower() == "mv"
                        and _amplifier(c.name) == _amplifier(signal.name)), None)
        if monitor is not None:
            try:
                first = np.asarray(source.read(monitor.index, [0])[0], dtype=float)
                measured = float(np.nanmedian(first[: max(1, min(first.size, int(fs)))]))
            except Exception:
                measured = float("nan")
            if np.isfinite(measured) and (not np.isfinite(holding) or abs(measured - holding) > HOLDING_AGREE_MV):
                notes.append(f"holding {measured:.1f} mV as the amplifier's monitor ({monitor.name}) reads it"
                             + (f"; Clampex's output was at {holding:g} mV (a holding set on the amplifier is not "
                                "in the output)" if np.isfinite(holding) else ""))
                holding = measured
    # Other outputs that play something: a light driver, a second amplifier.
    for dac, table in tables.items():
        if dac == command_dac or not varies(dac):
            continue
        name = header.dac_names[dac] or f"DAC{dac}"
        role = Role.LIGHT if LIGHT_NAME.search(name) else Role.OTHER
        index = source.add_virtual(lambda t=table: np.nan_to_num(_waveforms(t, n_sweeps, n_samples), nan=0.0))
        channels.append(Channel(index, name, header.dac_units[dac] if dac < len(header.dac_units) else "",
                                role, False, role_reason=f"output {dac} plays a waveform" +
                                (" and is named for a light source" if role is Role.LIGHT else "")))
    # Digital outputs that toggle, from the active DAC's epoch table.
    active = tables.get(header.active_dac) or (tables.get(command_dac) if command_dac is not None else None)
    if active is not None:
        for bit in range(DIGITAL_OUTPUTS):
            try:
                toggles = _digital_toggles(active, bit, n_sweeps, n_samples)
            except Exception:
                continue
            if toggles:
                index = source.add_virtual(lambda t=active, b=bit: _digital(t, b, n_sweeps, n_samples))
                channels.append(Channel(index, f"Digital out {bit}", "TTL", Role.TRIGGER, False,
                                        role_reason="a digital output that toggles during the sweep"))

    metadata = ExperimentMetadata(notes=[Note(t, text, "Clampex tag") for t, text in header.tags if text])
    if header.comment:
        metadata.notes.append(Note(0.0, header.comment, "Clampex comment"))
    acquisition = float("nan")
    i = signal.index
    if header.telegraph_enabled[i] and np.isfinite(header.telegraph_filter_hz[i]) and header.telegraph_filter_hz[i] > 0:
        acquisition = header.telegraph_filter_hz[i]
    elif np.isfinite(header.signal_lowpass_hz[i]) and header.signal_lowpass_hz[i] > 0:
        acquisition = header.signal_lowpass_hz[i]
    event_driven = header.operation_mode in EVENT_DRIVEN
    times_recorded = None
    if header.operation_mode == EPISODIC:
        sweep_times = np.arange(n_sweeps) * header.sweep_interval_s
        real, _ = event_sweep_times(header)
        if n_sweeps > 1 and real.size == n_sweeps and np.max(np.abs(real - sweep_times)) > max(2.0 / fs, 1e-3):
            # a run paused and resumed (Clampwise's recording, 2026-10-15): its synch array keeps the real starts
            sweep_times = real
            notes.append("the sweeps' start times are the file's synch array's (not evenly spaced: a pause)")
    elif event_driven:
        sweep_times, times_recorded = event_sweep_times(header)
        notes.append(f"event-driven file ({MODES[header.operation_mode]}): {n_sweeps} triggered sweeps of "
                     f"{header.sweep_duration_s * 1e3:g} ms"
                     + ("; each sweep's start time is from the file's synch array" if times_recorded else
                        "; the sweeps are stored back to back (as saved by event-detection exports), so the "
                        "times between events are not in the file"))
    else:
        sweep_times = np.zeros(1)
    return Recording(
        source=path, sample_rate_hz=fs, channels=channels, samples=source, sweep_start_times_s=sweep_times,
        protocol=header.protocol, mode=header.mode, start_time=header.start_time, epochs=epochs,
        holding=holding, acquisition_filter_hz=acquisition, metadata=metadata, notes=notes,
        header={
            "format": "ABF", "abf_version": header.version, "operation_mode": MODES.get(header.operation_mode, "?"),
            "protocol_path": header.protocol_path, "creator": header.creator,
            "sweep_interval_s": header.sweep_interval_s, "channel_count": header.n_channels,
            "sweep_count": header.n_sweeps, "point_bytes": header.point_bytes,
            "telegraph_gain": header.telegraph_gain, "telegraph_filter_hz": header.telegraph_filter_hz,
            "telegraph_mode": header.telegraph_mode, "signal_lowpass_hz": header.signal_lowpass_hz,
            "holding_levels": header.holding, "dac_names": header.dac_names, "command_dac": command_dac,
            "tags": header.tags, "event_driven": event_driven, "event_times_recorded": times_recorded,
        },
    )
