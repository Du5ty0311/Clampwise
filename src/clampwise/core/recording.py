"""One recording: sweeps of channels, with everything the file says about them.

This is the one data model (B18, item 1). The readers in ``clampwise.io``
fill it; every analysis reads it and never a file. Samples are read through a
``SampleSource`` only when asked for, so a 60-minute file costs its header
until a trace is drawn or analyzed (the ABF reader memory-maps the data
block; a simulation keeps arrays in memory).

Units and conventions, used everywhere:
- time in seconds, from the start of the sweep; ``sweep_start_times_s`` from
  the start of the file; ``start_time`` is the file's own clock;
- voltages in mV, currents in pA, as recorded (the readers convert V and nA);
- a sweep is one row of a channel's samples; gap-free files have one sweep.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

NAN = float("nan")


class ClampMode(str, Enum):
    VOLTAGE_CLAMP = "VC"
    CURRENT_CLAMP = "CC"
    UNKNOWN = "unknown"

    @classmethod
    def from_units(cls, units: str) -> "ClampMode":
        """What is recorded says what was clamped: a current means voltage clamp."""
        u = (units or "").strip().lower()
        if u in ("pa", "na", "a", "ua"):
            return cls.VOLTAGE_CLAMP
        if u in ("mv", "v", "uv"):
            return cls.CURRENT_CLAMP
        return cls.UNKNOWN


class Role(str, Enum):
    """What a channel is for; decided by the reader from names and units, and
    overridable from the window (nothing is chosen silently: the reason is
    kept on the channel)."""

    SIGNAL = "signal"          # the recorded cell or field
    COMMAND = "command"        # what the amplifier was told to do
    MONITOR = "monitor"        # the amplifier's own readback (a current or voltage monitor)
    TRIGGER = "trigger"        # a TTL: stimulus isolator, LED, frame sync
    LIGHT = "light"            # the light driver's analog command or photodiode
    OTHER = "other"


@dataclass(frozen=True)
class Channel:
    """One recorded or commanded waveform, by name, with its units."""

    index: int
    name: str
    units: str
    role: Role = Role.OTHER
    recorded: bool = True              # an input (ADC); False for a commanded output (DAC)
    clamp_mode: ClampMode = ClampMode.UNKNOWN
    gain: float = NAN                  # the telegraphed gain, when the file carries one
    telegraphed: bool = False
    role_reason: str = ""              # why the reader gave it this role

    def describe(self) -> str:
        kind = "input" if self.recorded else "output"
        return f"{self.name} ({self.units}, {kind}, {self.role.value})"


@dataclass(frozen=True)
class Epoch:
    """One step of the protocol's epoch table, on one sweep."""

    kind: str                           # "step", "ramp", "pulse", "train", "hold"...
    start_s: float
    end_s: float
    level: float                        # the command level over the epoch (units of the command channel)
    digital: tuple[int, ...] = ()       # the digital-output bits during the epoch


@dataclass
class Stimulus:
    """When the stimulus came, in each sweep, and how that is known."""

    times_s: list[np.ndarray] = field(default_factory=list)      # one array of onsets per sweep
    widths_s: list[np.ndarray] = field(default_factory=list)     # matching pulse widths
    source: str = ""                    # "trigger channel IN 8", "epoch table", "command", "typed", "file record"
    kind: str = "unknown"               # "light", "electrical", "unknown"
    channel: str = ""
    reason: str = ""                    # why this source was chosen over the others

    def __len__(self) -> int:
        return int(sum(len(t) for t in self.times_s))

    def pulses_in(self, sweep: int) -> int:
        return int(len(self.times_s[sweep])) if sweep < len(self.times_s) else 0

    def fixed_times(self, tolerance_s: float = 1e-4) -> np.ndarray:
        """The onsets that sit at the same time in every sweep (an empty array
        when the pulses move between sweeps, as in an interval series)."""
        if not self.times_s:
            return np.array([])
        first = np.asarray(self.times_s[0], dtype=float)
        if not all(len(t) == first.size for t in self.times_s):
            return np.array([])
        stack = np.vstack([np.asarray(t, dtype=float) for t in self.times_s]) if first.size else np.empty((0, 0))
        if stack.size and np.all(np.ptp(stack, axis=0) <= tolerance_s):
            return stack.mean(axis=0)
        return np.array([])


@dataclass
class Identity:
    """Who a recording is of: animal, slice, cell and condition, typed or
    parsed from the path (``source`` says which, per field)."""

    animal: str = ""
    slice: str = ""
    cell: str = ""
    condition: str = ""
    label: str = ""
    sources: dict[str, str] = field(default_factory=dict)       # field -> "typed" | "path rule <name>" | "file name"

    def cell_key(self) -> tuple[str, str, str]:
        return (self.animal, self.slice, self.cell)

    def describe(self) -> str:
        parts = [p for p in (self.animal, self.slice, self.cell) if p]
        text = " > ".join(parts) if parts else "(no identity)"
        return f"{text} [{self.condition}]" if self.condition else text


@dataclass
class DrugEvent:
    name: str
    on_s: float
    off_s: float = NAN                  # NaN: still on at the end
    source: str = ""                    # "typed", "Clampex tag", "Clampex comment"


@dataclass
class Note:
    time_s: float
    text: str
    source: str = ""                    # "typed", "Clampex tag", "Clampex comment"


@dataclass
class ExperimentMetadata:
    """What the experiment was: typed in the project, read from Clampex's
    tags and comments, or left blank (never invented)."""

    age: str = ""
    sex: str = ""
    genotype: str = ""
    region: str = ""
    cell_type: str = ""
    internal_solution: dict[str, float] = field(default_factory=dict)      # ion -> mM
    external_solution: dict[str, float] = field(default_factory=dict)
    temperature_c: float = NAN
    junction_potential_mv: float = NAN
    rs_typed_mohm: tuple[float, float] = (NAN, NAN)     # Clampex's Membrane Test at start and end, when typed
    rs_source: str = ""                                 # "": typed; else where Rs was read (the acquisition's Lab Book)
    drugs: list[DrugEvent] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)

    def rs_words(self, typed: str) -> str:
        """Where the Rs at start and end came from, as the analyses that use it say it."""
        return f"Clampex's Membrane Test, read from {self.rs_source}" if self.rs_source else typed

    def drug_on(self, name: str) -> DrugEvent | None:
        for d in self.drugs:
            if d.name.lower() == name.lower():
                return d
        return None


class SampleSource(Protocol):
    """Where a recording's samples come from, read on demand."""

    n_channels: int
    n_sweeps: int
    n_samples: int

    def read(self, channel: int, sweeps: slice | Sequence[int] | None = None) -> np.ndarray:
        """Samples of ``channel`` as a float array (n_sweeps_selected, n_samples)."""
        ...


class ArraySource:
    """Samples already in memory: simulations, tests, small files."""

    def __init__(self, arrays: Sequence[np.ndarray]) -> None:
        self._arrays = [np.atleast_2d(np.asarray(a)) for a in arrays]
        shapes = {a.shape for a in self._arrays}
        if len(shapes) > 1:
            raise ValueError(f"channels differ in shape: {sorted(shapes)}")
        self.n_channels = len(self._arrays)
        self.n_sweeps, self.n_samples = self._arrays[0].shape if self._arrays else (0, 0)

    def read(self, channel: int, sweeps=None) -> np.ndarray:
        a = self._arrays[channel]
        if sweeps is None:
            return a
        return a[sweeps]


class SubsetSource:
    """Some sweeps of another source, renumbered from 0 (a chosen selection)."""

    def __init__(self, source, sweeps: Sequence[int]) -> None:
        self._source, self._sweeps = source, np.asarray(list(sweeps), dtype=int)
        self.n_channels, self.n_samples = source.n_channels, source.n_samples
        self.n_sweeps = int(self._sweeps.size)

    def read(self, channel: int, sweeps=None) -> np.ndarray:
        chosen = self._sweeps if sweeps is None else self._sweeps[sweeps]
        return self._source.read(channel, list(chosen))


def parse_sweeps(text: str, n_sweeps: int) -> list[int]:
    """'135-200, 455' (sweeps numbered from 1, as Clampex numbers them) -> the
    0-based indices that exist, in order; blank -> every sweep."""
    text = (text or "").strip()
    if not text:
        return list(range(n_sweeps))
    out: set[int] = set()
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        a, sep, b = part.partition("-")
        try:
            lo = int(a)
            hi = int(b) if sep and b.strip() else (n_sweeps if sep else lo)
        except ValueError:
            raise ValueError(f"'{part}' is not a sweep or a range of sweeps (e.g. 135-200, 455)") from None
        out.update(k - 1 for k in range(lo, hi + 1) if 1 <= k <= n_sweeps)
    if not out:
        raise ValueError(f"'{text}' names no sweep of the {n_sweeps} in the file")
    return sorted(out)


@dataclass
class Recording:
    """A file's worth of sweeps and everything the file says about them."""

    source: Path
    sample_rate_hz: float
    channels: list[Channel]
    samples: SampleSource
    sweep_start_times_s: np.ndarray                 # from the start of the file
    protocol: str = ""
    mode: str = "episodic"                          # "episodic" or "gap-free"
    start_time: dt.datetime | None = None
    epochs: list[list[Epoch]] = field(default_factory=list)        # per sweep, for the command channel
    holding: float = NAN                            # the command's holding level
    acquisition_filter_hz: float = NAN              # the amplifier's low-pass, when recorded
    stimulus: Stimulus = field(default_factory=Stimulus)
    identity: Identity = field(default_factory=Identity)
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    header: dict = field(default_factory=dict)      # whatever else the file carries, by name
    notes: list[str] = field(default_factory=list)  # what the reader decided and why
    _hash: str = field(default="", repr=False)

    # ---- shape
    @property
    def n_sweeps(self) -> int:
        return self.samples.n_sweeps

    @property
    def n_samples(self) -> int:
        return self.samples.n_samples

    @property
    def sweep_duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz

    @property
    def duration_s(self) -> float:
        """From the first sample to the last, on the file's clock."""
        if self.sweep_start_times_s.size == 0:
            return 0.0
        return float(self.sweep_start_times_s[-1]) + self.sweep_duration_s

    @property
    def time_s(self) -> np.ndarray:
        return np.arange(self.n_samples) / self.sample_rate_hz

    @property
    def is_gap_free(self) -> bool:
        return self.mode == "gap-free"

    # ---- channels
    def channel(self, role: Role) -> Channel | None:
        for c in self.channels:
            if c.role is role:
                return c
        return None

    def channels_with(self, role: Role) -> list[Channel]:
        return [c for c in self.channels if c.role is role]

    @property
    def signal_channel(self) -> Channel:
        c = self.channel(Role.SIGNAL)
        if c is None:
            raise ValueError(f"{self.source.name}: no channel is the recorded signal")
        return c

    @property
    def clamp_mode(self) -> ClampMode:
        return self.signal_channel.clamp_mode

    @property
    def is_voltage_clamp(self) -> bool:
        return self.clamp_mode is ClampMode.VOLTAGE_CLAMP

    @property
    def signal_units(self) -> str:
        return self.signal_channel.units

    # ---- samples, on demand
    def read(self, channel: Channel | int, sweeps=None) -> np.ndarray:
        index = channel.index if isinstance(channel, Channel) else int(channel)
        return self.samples.read(index, sweeps)

    @property
    def signal(self) -> np.ndarray:
        """Every sweep of the recorded signal, (n_sweeps, n_samples)."""
        return self.read(self.signal_channel)

    def sweep(self, i: int, channel: Channel | int | None = None) -> np.ndarray:
        return self.read(self.signal_channel if channel is None else channel, [i])[0]

    @property
    def command(self) -> np.ndarray | None:
        """The command waveform, (n_sweeps, n_samples), or None without one."""
        c = self.channel(Role.COMMAND)
        return None if c is None else self.read(c)

    def select_sweeps(self, sweeps: Sequence[int]) -> "Recording":
        """The same recording with only ``sweeps`` (0-based), renumbered from 0;
        everything per sweep (start times, epochs, the stimulus) follows them."""
        import copy

        idx = [int(k) for k in sweeps]
        out = copy.copy(self)
        out.samples = SubsetSource(self.samples, idx)
        starts = np.asarray(self.sweep_start_times_s, dtype=float)
        out.sweep_start_times_s = starts[idx] if starts.size == self.n_sweeps else starts
        out.epochs = [self.epochs[k] for k in idx] if len(self.epochs) == self.n_sweeps else list(self.epochs)
        st = self.stimulus
        if st.times_s and len(st.times_s) == self.n_sweeps:
            out.stimulus = Stimulus([st.times_s[k] for k in idx], [st.widths_s[k] for k in idx], st.source, st.kind,
                                    st.channel, st.reason)
        out.notes = list(self.notes) + [f"sweeps chosen: {len(idx)} of {self.n_sweeps} "
                                        f"({', '.join(str(k + 1) for k in idx[:8])}{' ...' if len(idx) > 8 else ''})"]
        out._hash = self._hash
        return out

    def index_at(self, t_s: float) -> int:
        return int(np.clip(round(t_s * self.sample_rate_hz), 0, max(self.n_samples - 1, 0)))

    # ---- identity on the file
    @property
    def content_hash(self) -> str:
        """SHA-256 of the file (provenance, the result cache); computed once."""
        if not self._hash:
            self._hash = file_hash(self.source)
        return self._hash

    def describe(self) -> str:
        sig = self.signal_channel
        kind = "gap-free" if self.is_gap_free else f"{self.n_sweeps} sweep(s) of {self.sweep_duration_s:g} s"
        return (f"{self.source.name}: {kind} at {self.sample_rate_hz:g} Hz, {sig.name} in {sig.units} "
                f"({self.clamp_mode.value}), protocol '{self.protocol}'")


def file_hash(path: Path | str) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
