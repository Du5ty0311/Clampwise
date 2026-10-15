"""Acquisition in Clampwise, without Clampex when the user chooses: a Clampex protocol played on the Digidata and recorded into an ABF2 file that Clampfit, pyabf and
Clampwise read as one of Clampex's own.

A Clampex protocol (``.pro``) is an ABF2 header without samples: the same protocol, inputs, outputs, epoch and
strings sections a data file opens with (6656 bytes on the rig's). So a protocol is read by giving its header
zeroed samples and a synch array and loading it (pyabf builds each output's epochs, sweep by sweep), and a
recording is written as Clampex writes one (``rig/growing.py``): that header, the samples, the counts and start time
filled in, and the synch array (each sweep's start in the protocol's synch unit, 6.25 us, and its length in samples).

The outputs are played as frames of the Digidata's four analog outputs and its digital word: each sweep at its
start-to-start time, the holding levels between, the epochs' levels in the protocol's units over its output scale
factor (20 mV/V, 400 pA/V on the rig) in volts, and volts to the converter's codes as pCLAMP's library converts them
(``DD1400_VoltsToDAC``: volts x 3276.8 x the output's gain, less its offset, clipped to 16 bits, rounded).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .growing import BLOCK, DATA, EPISODIC, GAP_FREE, PROTOCOL, SECTIONS, SYNCH, _section, scratch

ADC_SECTION, DAC_SECTION, STRINGS = 1, 2, 9
N_AO = 4
N_OUT = N_AO + 1                     # a frame of output: the four analog outputs, then the digital word
COUNTS_PER_VOLT = 3276.8


@dataclass
class Protocol:
    """One Clampex protocol, as its header says."""

    path: Path
    head: bytes
    mode: int                        # 5 episodic, 3 gap-free
    interval_us: float               # between samples of one channel
    per_episode: int                 # samples a sweep, all channels together
    sweeps: int
    start_to_start_s: float
    synch_unit_us: float
    seconds_per_run: float
    ai: list[int] = field(default_factory=list)          # the inputs sampled, in their order (physical numbers)
    dac_scale: list[float] = field(default_factory=list)  # each output's units a volt
    holding: list[float] = field(default_factory=list)    # each output's holding level, in its units
    dac_units: list[str] = field(default_factory=list)
    active_dac: int = 0
    digital_enable: bool = False      # the protocol's digital outputs are on (off: they stay at 0)
    digital_holding: int = 0

    @property
    def rate_hz(self) -> float:
        return 1e6 / self.interval_us

    @property
    def sweep_frames(self) -> int:
        return self.per_episode // max(len(self.ai), 1)

    @property
    def interval_frames(self) -> int:
        return int(round(self.start_to_start_s * self.rate_hz)) if self.mode == EPISODIC else self.sweep_frames

    def run_frames(self, seconds: float | None = None, first: int = 0) -> int:
        """Frames recorded for one run (from sweep ``first``): every sweep at its start, or a gap-free run of
        ``seconds``."""
        if self.mode == GAP_FREE:
            return int(round((seconds or self.seconds_per_run) * self.rate_hz))
        return (self.sweeps - first - 1) * max(self.interval_frames, self.sweep_frames) + self.sweep_frames


def protocols_folder() -> str:
    """Where the user's protocols are: the folder Clampex itself last opened one from (its registry), else pCLAMP's
    own Params folder, else the home folder. Any folder can be chosen after (Protocols...)."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Molecular Devices\Clampex\Protocol") as key:
            folder = str(winreg.QueryValueEx(key, "CurrentDirectory")[0] or "")
        if folder and Path(folder).is_dir():
            return folder
    except (OSError, ImportError):
        pass
    params = Path.home() / "Documents" / "Molecular Devices" / "pCLAMP" / "Params"
    return str(params if params.is_dir() else Path.home())


def read_protocol(path: str | Path) -> Protocol:
    path = Path(path)
    head = path.read_bytes()
    if head[:4] != b"ABF2":
        raise ValueError(f"{path.name} is not a Clampex 10 protocol (no ABF2 header)")
    p = _section(head, PROTOCOL)[0] * BLOCK
    mode, interval = struct.unpack_from("<hf", head, p)
    synch_unit = struct.unpack_from("<f", head, p + 14)[0]
    seconds_per_run = struct.unpack_from("<f", head, p + 18)[0]
    per_episode, _, sweeps = struct.unpack_from("<iii", head, p + 22)
    start_to_start = struct.unpack_from("<f", head, p + 62)[0]
    block, size, n_ai = _section(head, ADC_SECTION)
    ai = [struct.unpack_from("<h", head, block * BLOCK + size * i)[0] for i in range(n_ai)]
    block, size, n_dac = _section(head, DAC_SECTION)
    # nDACNum, nTelegraphDACScaleFactorEnable, fInstrumentHoldingLevel, fDACScaleFactor, fDACHoldingLevel
    dac = [struct.unpack_from("<hhfff", head, block * BLOCK + size * i) for i in range(n_dac)]
    scale = [1.0] * N_AO
    hold = [0.0] * N_AO
    for num, _, _, s, h in dac:
        if 0 <= num < N_AO:
            scale[num], hold[num] = (s or 1.0), h
    pr = Protocol(path, head, mode, interval, per_episode, sweeps if mode == EPISODIC else 1, start_to_start,
                  synch_unit, seconds_per_run, ai, scale, hold)
    import pyabf

    f = readable(pr)
    try:                                 # the active output (its epochs drive the digital outputs) and the digital
        abf = pyabf.ABF(str(f), loadData=False)                           # outputs between sweeps, as pyabf reads
        sec = abf._protocolSection
        pr.dac_units = [str(u) for u in abf.dacUnits[:N_AO]]
        pr.active_dac = int(sec.nActiveDACChannel)
        pr.digital_enable = bool(sec.nDigitalEnable)
        pr.digital_holding = int(sec.nDigitalHolding) if sec.nDigitalEnable else 0
    finally:
        f.unlink(missing_ok=True)
    return pr


def abf_bytes(protocol: Protocol, samples: np.ndarray, start: dt.datetime | None = None,
              sweeps: int | None = None) -> bytes:
    """The ABF2 file Clampex would write for ``samples`` (int16, frames x inputs, as recorded continuously): the
    protocol's header with its counts and start, each sweep cut at its start, then the synch array."""
    pr = protocol
    samples = np.asarray(samples, dtype="<i2").reshape(-1, len(pr.ai))
    if pr.mode == EPISODIC:
        step = max(pr.interval_frames, pr.sweep_frames)
        done = sweeps if sweeps is not None else max(0, (len(samples) - pr.sweep_frames) // step + 1)
        done = min(done, pr.sweeps) if sweeps is None else done
        cut = [samples[k * step:k * step + pr.sweep_frames] for k in range(done)]
        return episodic_bytes(pr, cut, [k * step / pr.rate_hz for k in range(done)], start)
    return _file(pr, samples, 0, samples.size, [], start)  # gap-free: Clampex leaves the episode count at 0


def episodic_bytes(protocol: Protocol, sweeps: list, starts_s: list[float], start: dt.datetime | None = None) -> bytes:
    """The episodic file for ``sweeps`` (each frames x inputs) begun ``starts_s`` after the file's start: their real
    times in the synch array (a run paused and resumed keeps its gap, as Clampex's synch array would)."""
    pr = protocol
    data = np.concatenate(sweeps) if sweeps else np.zeros((0, len(pr.ai)), "<i2")
    if starts_s and starts_s[0]:          # the file begins with its first sweep, as Clampex's (its synch array from 0)
        start = (start or dt.datetime.now()) + dt.timedelta(seconds=starts_s[0])
        starts_s = [s - starts_s[0] for s in starts_s]
    return _file(pr, data, len(sweeps), len(sweeps) * pr.per_episode, starts_s, start)


def _file(pr: Protocol, data: np.ndarray, episodes: int, entries: int, starts_s: list[float],
          start: dt.datetime | None) -> bytes:
    head = bytearray(pr.head)
    head += bytes(-len(head) % BLOCK)
    start = start or dt.datetime.now()
    struct.pack_into("<I", head, 12, episodes)
    struct.pack_into("<II", head, 16, int(start.strftime("%Y%m%d")),
                     (start.hour * 3600 + start.minute * 60 + start.second) * 1000 + start.microsecond // 1000)
    data_block = len(head) // BLOCK
    struct.pack_into("<IIq", head, SECTIONS + 16 * DATA, data_block, 2, entries)
    body = data.astype("<i2").tobytes()
    body += bytes(-len(body) % BLOCK)
    synch = b""
    if pr.mode == EPISODIC and episodes:
        unit = pr.synch_unit_us or pr.interval_us / max(len(pr.ai), 1)
        synch = b"".join(struct.pack("<ii", int(round(s * 1e6 / unit)), pr.per_episode) for s in starts_s)
        struct.pack_into("<IIq", head, SECTIONS + 16 * SYNCH, data_block + len(body) // BLOCK, 8, episodes)
    else:
        struct.pack_into("<IIq", head, SECTIONS + 16 * SYNCH, 0, 0, 0)
    return bytes(head) + body + synch


def readable(protocol: Protocol, folder: Path | None = None) -> Path:
    """The protocol with one run of zeroed samples, as a file the loaders read (its epochs, sweep by sweep)."""
    pr = protocol
    frames = pr.sweeps * max(pr.interval_frames, pr.sweep_frames) if pr.mode == EPISODIC else pr.sweep_frames
    blob = abf_bytes(pr, np.zeros((frames, len(pr.ai)), "<i2"), sweeps=pr.sweeps if pr.mode == EPISODIC else None)
    folder = folder or scratch()
    folder.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".abf", prefix=pr.path.stem + "_", dir=folder)
    with open(fd, "wb") as fh:
        fh.write(blob)
    return Path(name)


def dac_codes(volts, gain: float, offset: int) -> np.ndarray:
    """``DD1400_VoltsToDAC``: volts x 3276.8 x gain - offset, clipped to 16 bits, rounded."""
    x = np.asarray(volts, dtype=float) * COUNTS_PER_VOLT * gain - offset
    return np.floor(np.clip(x, -32768, 32767) + 0.5).astype("<i2")


def waveforms(protocol: Protocol) -> tuple[np.ndarray, np.ndarray]:
    """Each sweep's outputs in the protocol's units, (sweeps, N_AO, sweep frames), and its digital word,
    (sweeps, sweep frames), from the epochs (pyabf's epoch tables)."""
    import pyabf
    import pyabf.waveform

    pr = protocol
    f = readable(pr)
    try:
        abf = pyabf.ABF(str(f))
        n = abf.sweepPointCount
        out = np.empty((pr.sweeps, N_AO, n))
        for d in range(N_AO):
            out[:, d] = pr.holding[d]
        dig = np.zeros((pr.sweeps, n), dtype=np.int64)
        for d in range(min(N_AO, len(abf._dacSection.nWaveformEnable))):     # the outputs the header has
            st = pyabf.stimulus.Stimulus(abf, d)
            for k in range(pr.sweeps):
                out[k, d] = st.stimulusWaveform(k)
        if pr.digital_enable and 0 <= pr.active_dac < len(abf.holdingCommand):
            tables = pyabf.waveform.EpochTable(abf, pr.active_dac)
            for k in range(pr.sweeps):
                w = tables.epochWaveformsBySweep[k]
                for bit in range(8):
                    d = np.asarray(w.getDigitalWaveform(bit), dtype=np.int64)
                    if d.size == n:
                        dig[k] |= (d > 0).astype(np.int64) << bit
        del abf
        return out, dig
    finally:
        f.unlink(missing_ok=True)


RIG_INPUTS = {0: "primary", 1: "secondary"}   # the rig's wiring: channel 1's primary to AI 0, its secondary to AI 1


def with_telegraph(protocol: Protocol, tg, inputs: dict[int, str] | None = None) -> tuple[Protocol, str]:
    """The protocol as the amplifier now has it, as Clampex plays one: the command's scale and unit from the
    telegraph (20 mV/V in voltage clamp, 400 pA/V in current clamp), the protocol's numbers read in that unit
    (a Simple_IV protocol saved in pA is played +-100 mV in voltage clamp and +-100 pA in current clamp), and each
    telegraphed input's scale, gain, filter and unit written into the header. Never the scale the protocol was
    saved with (F-I_Curve and Sag_Ih keep 20 with steps in pA: +-10 V, +-4 nA, if played so). Also a note when
    the protocol was saved in the other mode."""
    if tg is None:
        raise ValueError("the amplifier's telegraph did not answer (is Commander open?): not played")
    if tg.mode != "I=0" and not tg.command_per_volt > 0:
        raise ValueError(f"the amplifier reports no command sensitivity in {tg.mode}: not played")
    head = bytearray(protocol.head)
    strings_block, strings_size, _ = _section(head, STRINGS)

    def unit_index(unit: str) -> int:
        nonlocal strings_size
        blob = bytes(head[strings_block * BLOCK:strings_block * BLOCK + strings_size])
        names = blob[blob.rfind(b"\x00\x00"):].split(b"\x00")[1:]
        if unit.encode() in names:
            return names.index(unit.encode())
        add = b"\x00" + unit.encode()                     # appended where the block has room
        if strings_size + len(add) > BLOCK * max(1, _blocks_after(head, strings_block) or 1):
            raise ValueError("the protocol's strings have no room for a unit")
        head[strings_block * BLOCK + strings_size:strings_block * BLOCK + strings_size + len(add)] = add
        strings_size += len(add)
        struct.pack_into("<I", head, SECTIONS + 16 * STRINGS + 4, strings_size)
        return len(names)

    block, size, n_ai = _section(head, ADC_SECTION)
    for i in range(n_ai):
        at = block * BLOCK + size * i
        which = (inputs or RIG_INPUTS).get(struct.unpack_from("<h", head, at)[0])
        if which is None:
            continue
        unit = tg.primary_unit if which == "primary" else tg.secondary_unit
        scale = tg.primary_v_per_unit / tg.primary_gain if which == "primary" else \
            tg.secondary_v_per_unit / tg.secondary_gain
        gain = tg.primary_gain if which == "primary" else tg.secondary_gain
        lpf = tg.primary_lpf_hz if which == "primary" else float("nan")
        struct.pack_into("<h", head, at + 2, 1)                           # telegraphed
        struct.pack_into("<f", head, at + 6, gain)
        if lpf == lpf:
            struct.pack_into("<f", head, at + 10, lpf)
        struct.pack_into("<h", head, at + 18, {"VC": 0, "IC": 1, "I=0": 2}[tg.mode])
        struct.pack_into("<f", head, at + 40, scale)
        struct.pack_into("<f", head, at + 48, 1.0)
        struct.pack_into("<i", head, at + 78, unit_index(unit))
    block, size, n_dac = _section(head, DAC_SECTION)
    note = ""
    for i in range(n_dac):
        at = block * BLOCK + size * i
        if struct.unpack_from("<h", head, at)[0] == 0:
            struct.pack_into("<f", head, at + 8, tg.command_per_volt or 1.0)
            struct.pack_into("<i", head, at + 28, unit_index(tg.command_unit))
    saved = (protocol.dac_units or [""])[0]
    if saved and saved != tg.command_unit:
        note = (f"{protocol.path.name} was saved in {saved}; played as {tg.command_unit} "
                f"({tg.mode}), as Clampex plays it")
    pr = dataclasses.replace(protocol, head=bytes(head),
                             dac_scale=[tg.command_per_volt or 1.0] + protocol.dac_scale[1:],
                             dac_units=[tg.command_unit] + protocol.dac_units[1:])
    return pr, note


def inputs(protocol: Protocol) -> list[tuple[str, str, float]]:
    """Each input's (name, unit, volts at the digitizer a unit) as the header says, in the order recorded (what
    pyabf divides by: the instrument scale, the signal gain, the telegraphed gain when telegraphed)."""
    head = protocol.head
    sb, ss, _ = _section(head, STRINGS)
    blob = head[sb * BLOCK:sb * BLOCK + ss]
    names = [s.decode("ascii", "replace") for s in blob[blob.rfind(b"\x00\x00"):].split(b"\x00")[1:]]
    block, size, n = _section(head, ADC_SECTION)
    out = []
    for i in range(n):
        at = block * BLOCK + size * i
        tele, = struct.unpack_from("<h", head, at + 2)
        tgain, = struct.unpack_from("<f", head, at + 6)
        scale, = struct.unpack_from("<f", head, at + 40)
        sgain, = struct.unpack_from("<f", head, at + 48)
        name_i, unit_i = struct.unpack_from("<ii", head, at + 74)
        pick = (lambda k: names[k] if 0 <= k < len(names) else "")
        volts = (scale or 1.0) * (sgain or 1.0) * ((tgain or 1.0) if tele else 1.0)
        out.append((pick(name_i) or f"AI {protocol.ai[i]}", pick(unit_i), volts))
    return out


def _blocks_after(head: bytes, block: int) -> int:
    """Blocks from ``block`` to the next section's (the room a section has to grow in place)."""
    starts = sorted(_section(head, i)[0] for i in range(18) if _section(head, i)[0] > block)
    return (starts[0] - block) if starts else 1


def output_frames(protocol: Protocol, calibration: dict, seconds: float | None = None, first: int = 0) -> np.ndarray:
    """The run's output codes from sweep ``first`` on, (frames, N_OUT) int16: the sweeps at their starts, the
    holding between. Played only from ``with_telegraph``'s protocol, whose command scale is the amplifier's."""
    pr = protocol
    gains, offsets = calibration["dac_gains"], calibration["dac_offsets"]
    total = pr.run_frames(seconds, first)
    volts = np.empty((total, N_AO))
    for d in range(N_AO):
        volts[:, d] = pr.holding[d] / pr.dac_scale[d]
    digital = np.full(total, pr.digital_holding, dtype=np.int64)
    if pr.mode == EPISODIC:
        units, dig = waveforms(pr)
        step = max(pr.interval_frames, pr.sweep_frames)
        for k in range(first, pr.sweeps):
            a = (k - first) * step
            for d in range(N_AO):
                volts[a:a + pr.sweep_frames, d] = units[k, d] / pr.dac_scale[d]
            digital[a:a + pr.sweep_frames] = dig[k]
    frames = np.empty((total, N_OUT), dtype="<i2")
    for d in range(N_AO):
        frames[:, d] = dac_codes(volts[:, d], gains[d], offsets[d])
    frames[:, N_AO] = (digital & 0xFFFF).astype(np.uint16).view("<i2")
    return frames


def holding_codes(protocol: Protocol | None, calibration: dict) -> list[int]:
    """The codes that hold every output at the protocol's holding levels (none: 0 V), and the digital holding."""
    gains, offsets = calibration["dac_gains"], calibration["dac_offsets"]
    if protocol is None:
        return [int(dac_codes(0.0, gains[d], offsets[d])) for d in range(N_AO)] + [0]
    return [int(dac_codes(protocol.holding[d] / protocol.dac_scale[d], gains[d], offsets[d])) for d in range(N_AO)] \
        + [protocol.digital_holding]


def seconds_of(frames: int, protocol: Protocol) -> float:
    return frames / protocol.rate_hz if protocol.rate_hz else math.nan
