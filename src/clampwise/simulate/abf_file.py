"""Simulated recordings written as ABF2 files, laid out as Clampex lays them out (the header sections, then the
samples, int16, channels interleaved), so the reader, the replay and the self-test meet a real file. The field
offsets are those pyabf reads (its abf2 section readers); only what a reader needs is filled.
"""

from __future__ import annotations

import datetime as dt
import struct
from pathlib import Path

import numpy as np

BLOCK = 512
RESOLUTION = 32768
RANGE_V = 10.0


def write_abf(path: str | Path, signal: np.ndarray, *, fs: float, units=("pA", "mV"), names=("IN 0", "IN 1"),
              epochs=(), holding: float = 0.0, command_units: str = "mV", interval_s: float | None = None,
              start: dt.datetime | None = None, protocol: str = "sim_protocol", gap_free: bool = False) -> Path:
    """Write ``signal`` (channels, sweeps, samples) as an ABF2 file. ``epochs``: the command's steps on output 0
    as (level, samples) or (level, samples, the level's change from sweep to sweep: an IV), after Clampex's held
    first 1/64 of each sweep."""
    path = Path(path)
    signal = np.asarray(signal, dtype=float)
    n_ch, n_sweeps, n = signal.shape
    strings = ["Clampwise", protocol, *names, *units, "Cmd 0", command_units]
    index = {s: i + 1 for i, s in reversed(list(enumerate(strings)))}
    blob = b"\x00\x00" + b"\x00".join(s.encode("ascii") for s in strings)
    # each input's instrument scale (V a unit): its largest value at 80% of the range, inside 16 bits
    scale = [0.8 * RANGE_V / (float(np.max(np.abs(signal[c]))) or 1.0) for c in range(n_ch)]
    blocks = {"protocol": 1, "adc": 2, "dac": 3, "epoch_dac": 4, "epoch": 5, "strings": 6}
    data_block = 6 + (len(blob) + BLOCK - 1) // BLOCK
    head = bytearray(data_block * BLOCK)
    start = start or dt.datetime(2026, 1, 1, 12, 0, 0)
    struct.pack_into("<4s4sII", head, 0, b"ABF2", bytes([0, 0, 0, 2]), BLOCK, 1 if gap_free else n_sweeps)
    struct.pack_into("<II", head, 16, int(start.strftime("%Y%m%d")),
                     (start.hour * 3600 + start.minute * 60 + start.second) * 1000 + start.microsecond // 1000)
    struct.pack_into("<4sI", head, 56, bytes([0, 7, 10, 10]), index["Clampwise"])
    struct.pack_into("<I", head, 72, index[protocol])

    def section(i, block, size, entries):
        struct.pack_into("<IIq", head, 76 + 16 * i, block, size, entries)

    section(0, blocks["protocol"], BLOCK, 1)
    section(1, blocks["adc"], 128, n_ch)
    section(2, blocks["dac"], 256, n_ch)
    section(3, blocks["epoch"], 32, len(epochs))
    section(5, blocks["epoch_dac"], 48, len(epochs))
    section(9, blocks["strings"], len(blob), 1)
    section(10, data_block, 2, n_ch * n * (1 if gap_free else n_sweeps))
    p = blocks["protocol"] * BLOCK
    struct.pack_into("<hf", head, p, 3 if gap_free else 5, 1e6 / fs)
    struct.pack_into("<ii", head, p + 22, n_ch * n, 0)
    struct.pack_into("<i", head, p + 30, n_sweeps)
    struct.pack_into("<f", head, p + 62, interval_s if interval_s else n / fs)
    struct.pack_into("<ffii", head, p + 110, RANGE_V, RANGE_V, RESOLUTION, RESOLUTION)
    for c in range(n_ch):
        a = blocks["adc"] * BLOCK + 128 * c
        struct.pack_into("<h", head, a, c)
        struct.pack_into("<h", head, a + 26, c)
        struct.pack_into("<f", head, a + 28, 1.0)                      # programmable gain
        struct.pack_into("<f", head, a + 40, scale[c])                 # instrument scale
        struct.pack_into("<f", head, a + 48, 1.0)                      # signal gain
        struct.pack_into("<f", head, a + 56, 10_000.0)                 # low-pass
        struct.pack_into("<ii", head, a + 74, index[names[c]], index[units[c]])
        d = blocks["dac"] * BLOCK + 256 * c
        struct.pack_into("<h", head, d, c)
        struct.pack_into("<ff", head, d + 8, 20.0, holding if c == 0 else 0.0)
        struct.pack_into("<ii", head, d + 24, index["Cmd 0"], index[command_units])
        struct.pack_into("<hh", head, d + 40, 1 if c == 0 and epochs else 0, 1)
    for k, (level, samples, *inc) in enumerate(epochs):           # (level, samples[, level's change a sweep])
        e = blocks["epoch_dac"] * BLOCK + 48 * k
        struct.pack_into("<hhhffiiii", head, e, k, 0, 1, float(level), float(inc[0]) if inc else 0.0, int(samples),
                         0, 0, 0)
        struct.pack_into("<hh", head, blocks["epoch"] * BLOCK + 32 * k, k, 0)
    head[blocks["strings"] * BLOCK: blocks["strings"] * BLOCK + len(blob)] = blob
    # samples: int16, the channels interleaved, sweep after sweep (what pyabf multiplies back by its gain)
    gains = [RANGE_V / (np.float32(scale[c]) * RESOLUTION) for c in range(n_ch)]
    rows = np.stack([np.round(signal[c] / gains[c]) for c in range(n_ch)], axis=-1)       # (sweeps, n, channels)
    path.write_bytes(bytes(head) + np.clip(rows, -32768, 32767).astype("<i2").tobytes())
    return path


def membrane_test_file(path: str | Path, *, n_sweeps: int = 12, fs: float = 20_000.0, sweep_s: float = 0.2,
                       step_mv: float = -5.0, rs_mohm=12.0, rm_mohm: float = 200.0, cm_pf: float = 100.0,
                       holding_mv: float = -70.0, holding_pa: float = -40.0, noise: float = 1.0, seed: int = 0,
                       start: dt.datetime | None = None) -> Path:
    """A voltage-clamp file whose every sweep opens with a test step on an Rs-(Rm || Cm) cell (the exact
    solution, as ``simulate.membrane_test``); ``rs_mohm`` may be one value a sweep (an Rs jump)."""
    rng = np.random.default_rng(seed)
    n = int(round(sweep_s * fs))
    pre = n // 64                                  # Clampex holds the first 1/64 of each sweep
    a, b = pre + int(0.02 * fs), pre + int(0.06 * fs)
    rs_all = np.broadcast_to(np.asarray(rs_mohm, dtype=float), (n_sweeps,))
    t = np.arange(n) / fs
    current = np.full((n_sweeps, n), holding_pa)
    for s in range(n_sweeps):
        rs, rm, cm = rs_all[s] * 1e6, rm_mohm * 1e6, cm_pf * 1e-12
        tau = rs * rm * cm / (rs + rm)
        v = step_mv * 1e-3
        i_inf, i_peak = v / (rs + rm), v / rs
        current[s, a:b] += (i_inf + (i_peak - i_inf) * np.exp(-(t[a:b] - t[a]) / tau)) * 1e12
        current[s, b:] += -(i_peak - i_inf) * np.exp(-(t[b:] - t[b]) / tau) * 1e12
        current[s] += rng.normal(0, noise, n)
    potential = np.full((n_sweeps, n), holding_mv)
    potential[:, a:b] += step_mv
    epochs = [(holding_mv, a - pre), (holding_mv + step_mv, b - a), (holding_mv, n - b - pre)]
    return write_abf(path, np.stack([current, potential]), fs=fs, epochs=epochs, holding=holding_mv,
                     interval_s=sweep_s, start=start, protocol="sim_membrane_test")
