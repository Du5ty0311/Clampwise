"""A simple stimulus editor's core: what it saves is always a protocol Clampex opens.

A protocol is edited where Clampex keeps it, in its own header: each output's epochs (the epochs-per-DAC section:
type, first level and its change a sweep, first duration and its change a sweep, a pulse train's period and width),
each epoch's digital outputs (the epoch section), each output's holding level and whether it plays its epochs (the
DAC section), and the run (sweeps, their start-to-start interval, their length). Every edit is made in place, in the
sections the protocol already has (no section moved, nothing added to the header that Clampex would not write), and
an edited protocol is saved under a new name and read back before it is used.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from .acquire import DAC_SECTION, N_AO, Protocol, read_protocol
from .growing import BLOCK, EPISODIC, PROTOCOL, SECTIONS, _section

EPOCH, EPOCH_PER_DAC = 3, 5
TYPES = {1: "step", 2: "ramp", 3: "pulse train"}      # Clampex's epoch types the editor offers (0: off)
ENTRY = "<hhhffiiii"                                   # nEpochNum, nDACNum, nEpochType, fEpochInitLevel, fEpochLevelInc,
#                                                        lEpochInitDuration, lEpochDurationInc, lEpochPulsePeriod,
#                                                        lEpochPulseWidth (48 bytes an entry)


@dataclass
class Epoch:
    kind: int = 1                 # 1 step, 2 ramp, 3 pulse train
    level: float = 0.0            # in the output's units (the amplifier's, for the command)
    level_inc: float = 0.0        # its change from one sweep to the next
    samples: int = 0              # its duration, in samples (a channel's)
    samples_inc: int = 0
    period: int = 0               # a pulse train's period and pulse width, in samples
    width: int = 0
    digital: int = 0              # the digital outputs during it (bits 0-7)


@dataclass
class Edit:
    """A protocol as the editor shows it: each output's epochs and holding, and the run."""

    protocol: Protocol
    epochs: dict[int, list[Epoch]] = field(default_factory=dict)   # output -> its epochs (A, B, C...)
    holding: list[float] = field(default_factory=list)
    enabled: list[bool] = field(default_factory=list)
    sweeps: int = 1
    start_to_start_s: float = 0.0
    sweep_samples: int = 0        # a channel's samples a sweep

    @property
    def rate_hz(self) -> float:
        return self.protocol.rate_hz


def load(path: str | Path) -> Edit:
    pr = read_protocol(path)
    head = pr.head
    e = Edit(pr, holding=list(pr.holding), sweeps=pr.sweeps, start_to_start_s=pr.start_to_start_s,
             sweep_samples=pr.sweep_frames)
    block, size, n = _section(head, EPOCH_PER_DAC)
    digital = _digital(head)
    for i in range(n):
        num, dac, kind, level, inc, dur, dinc, period, width = struct.unpack_from(ENTRY, head, block * BLOCK + size * i)
        e.epochs.setdefault(dac, []).append(Epoch(kind, level, inc, dur, dinc, period, width, digital.get(num, 0)))
    block, size, n_dac = _section(head, DAC_SECTION)
    e.enabled = [False] * N_AO
    for i in range(n_dac):
        num, = struct.unpack_from("<h", head, block * BLOCK + size * i)
        if 0 <= num < N_AO:
            e.enabled[num] = bool(struct.unpack_from("<h", head, block * BLOCK + size * i + 40)[0])
    return e


def _digital(head: bytes) -> dict[int, int]:
    block, size, n = _section(head, EPOCH)
    out = {}
    for i in range(n):
        num, value = struct.unpack_from("<hh", head, block * BLOCK + size * i)
        out[num] = value
    return out


def room(head: bytes, section: int) -> int:
    """Bytes a section has before the next one begins (the room it can grow into in place)."""
    block = _section(head, section)[0]
    later = sorted(_section(head, i)[0] for i in range(18) if _section(head, i)[0] > block)
    return ((later[0] if later else len(head) // BLOCK) - block) * BLOCK


def to_bytes(e: Edit) -> bytes:
    """The protocol's header with the edit made in place."""
    head = bytearray(e.protocol.head)
    if e.protocol.mode != EPISODIC:
        raise ValueError("only episodic protocols are edited here (a gap-free one has no epochs to play)")
    if not 1 <= e.sweeps <= 10_000 or e.sweep_samples < 64:
        raise ValueError("a run needs 1 to 10000 sweeps of at least 64 samples")
    n_ai = len(e.protocol.ai)
    if e.start_to_start_s and e.start_to_start_s * e.rate_hz < e.sweep_samples:
        raise ValueError("the start-to-start interval is shorter than a sweep")
    entries = [(dac, k, ep) for dac in sorted(e.epochs) for k, ep in enumerate(e.epochs[dac])]
    for dac, k, ep in entries:
        if ep.kind not in TYPES:
            raise ValueError(f"output {dac}, epoch {chr(65 + k)}: only steps, ramps and pulse trains")
        last = ep.samples + (e.sweeps - 1) * ep.samples_inc
        if ep.samples < 0 or last < 0:
            raise ValueError(f"output {dac}, epoch {chr(65 + k)}: a duration below 0 by the last sweep")
        if ep.kind == 3 and not 0 < ep.width <= ep.period:
            raise ValueError(f"output {dac}, epoch {chr(65 + k)}: a pulse train needs 0 < width <= period")
    for dac in e.epochs:                                  # Clampex holds the first 1/64 of a sweep
        longest = sum(ep.samples + max(0, (e.sweeps - 1) * ep.samples_inc) for ep in e.epochs[dac])
        if longest > e.sweep_samples - e.sweep_samples // 64:
            raise ValueError(f"output {dac}: its epochs run past the sweep's end")
    block, size, _ = _section(head, EPOCH_PER_DAC)
    if len(entries) * size > room(head, EPOCH_PER_DAC):
        raise ValueError(f"the protocol has room for {room(head, EPOCH_PER_DAC) // size} epochs in all")
    head[block * BLOCK:block * BLOCK + room(head, EPOCH_PER_DAC)] = bytes(room(head, EPOCH_PER_DAC))
    for i, (dac, k, ep) in enumerate(entries):
        struct.pack_into(ENTRY, head, block * BLOCK + size * i, k, dac, ep.kind, float(ep.level), float(ep.level_inc),
                         int(ep.samples), int(ep.samples_inc), int(ep.period), int(ep.width))
    struct.pack_into("<IIq", head, SECTIONS + 16 * EPOCH_PER_DAC, block, size, len(entries))
    n_epochs = max((len(v) for v in e.epochs.values()), default=0)    # the digital outputs, an entry an epoch
    dblock, dsize, _ = _section(head, EPOCH)
    if dblock:
        if n_epochs * dsize > room(head, EPOCH):
            raise ValueError("the protocol has no room for the digital outputs of that many epochs")
        src = e.protocol.head
        _, _, n_old = _section(src, EPOCH)
        old = [src[dblock * BLOCK + dsize * k:dblock * BLOCK + dsize * (k + 1)] for k in range(n_old)]
        head[dblock * BLOCK:dblock * BLOCK + room(head, EPOCH)] = bytes(room(head, EPOCH))
        active = e.epochs.get(e.protocol.active_dac, [])
        for k in range(n_epochs):              # each entry kept as it was (trains, alternates), its value set
            at = dblock * BLOCK + dsize * k
            if k < len(old):
                head[at:at + dsize] = old[k]
            struct.pack_into("<hh", head, at, k, active[k].digital if k < len(active) else
                             struct.unpack_from("<h", head, at + 2)[0])
        struct.pack_into("<IIq", head, SECTIONS + 16 * EPOCH, dblock, dsize, n_epochs)
    block, size, n_dac = _section(head, DAC_SECTION)
    for i in range(n_dac):
        at = block * BLOCK + size * i
        num, = struct.unpack_from("<h", head, at)
        if 0 <= num < N_AO:
            struct.pack_into("<f", head, at + 12, float(e.holding[num]))
            struct.pack_into("<h", head, at + 40, 1 if e.enabled[num] and e.epochs.get(num) else 0)
            if e.enabled[num] and e.epochs.get(num):
                struct.pack_into("<h", head, at + 42, 1)                   # its waveform from its epochs
    p = _section(head, PROTOCOL)[0] * BLOCK
    struct.pack_into("<i", head, p + 22, e.sweep_samples * n_ai)
    struct.pack_into("<i", head, p + 30, e.sweeps)
    struct.pack_into("<f", head, p + 62, float(e.start_to_start_s))
    return bytes(head)


def save(e: Edit, path: str | Path) -> Protocol:
    """Saved under ``path`` (a new name: an existing file is never replaced) and read back."""
    path = Path(path)
    if path.suffix.lower() != ".pro":
        path = path.with_suffix(".pro")
    if path.exists():
        raise FileExistsError(f"{path.name} exists: save the edit under a new name")
    blob = to_bytes(e)
    path.write_bytes(blob)
    try:
        return read_protocol(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
