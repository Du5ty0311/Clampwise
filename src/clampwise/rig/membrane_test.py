"""A membrane test in Clampwise, beside the acquisition, as Clampex's Membrane Test: a square wave on the command, centred on the holding (its
height peak to peak), 500 samples a period (250 a step; the sample rate follows the frequency), the current recorded,
and every pulse measured as pCLAMP measures it (the pCLAMP 10 User Guide, "Membrane Test", pp. 191-193):

- I1, I2: the mean current over the last 20% of each half of the pulse; dI = I1 - I2; Rt = dV / dI;
  the holding current (I1 + I2) / 2;
- each edge's transient fitted with a single exponential between 80% and 20% of its peak (a straight line through
  its logarithm), tau;
- the charge: the transient above the new steady state, summed over the step from its first sample of its own
  sign (the amplifier's compensation spike at the edge left out, as Clampex's numbers show it is), plus dI tau (the
  membrane still charging while the step settles): Q;
- Ra (Rs) = tau dV / Q; Rm = Rt - Ra; Cm = Q Rt / (dV Rm);
- both edges, the falling one inverted, averaged.

In current clamp (Clampex's test assumes voltage clamp) the input resistance, tau and Cm come from the charging
fitted on the rising half. What it is reading is said: the pipette alone (Rt under the input-resistance floor), a seal
(over the ceiling), or a cell. Nothing is saved: a test, not a recording.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import acquire, digidata, telegraph
from .growing import scratch

NAN = float("nan")
FLOOR_MOHM, CEILING_MOHM = 20.0, 1000.0          # the quality's input-resistance floor and ceiling
RING_PULSES = 200                                # pulses the raw file holds before it is written round
SAMPLES = 500                                    # samples a period, as Clampex's (250 a step)
MAX_RATE_HZ = 250_000.0                          # the Digidata 1440A, one input
READ_EVERY_S = 0.05                              # how often the newest pulses are measured (the screen draws as often)


@dataclass
class Settings:
    amplitude: float = 10.0       # the pulse peak to peak, in the amplifier's unit (mV in VC, pA in IC)
    hz: float = 50.0              # pulses a second (Clampex's Cell stage: 10 ms steps)
    average: int = 5              # pulses averaged into each reading
    hours: float = 4.0            # the longest a test runs before it ends by itself

    @property
    def rate_hz(self) -> float:
        return min(SAMPLES * self.hz, MAX_RATE_HZ)

    @property
    def period(self) -> int:
        return int(round(self.rate_hz / self.hz))


@dataclass
class Reading:
    at: float = 0.0               # seconds since the test began
    mode: str = ""
    what: str = ""                # "cell", "seal", "open pipette", "no step"
    rt_mohm: float = NAN
    rs_mohm: float = NAN
    rm_mohm: float = NAN
    cm_pf: float = NAN
    tau_ms: float = NAN
    holding: float = NAN          # pA in voltage clamp, mV in current clamp
    unit: str = "pA"
    resolution: float = NAN       # one count of the digitizer, in ``unit`` (6.1 pA at 0.05 V/nA): Rt's grain
    pulses: int = 0               # pulses recorded so far
    t_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    trace: np.ndarray = field(default_factory=lambda: np.zeros(0))       # the averaged period, rising edge at 1/4
    fits: list = field(default_factory=list)      # (t_ms, y) of each edge's fitted exponential, drawn over the trace


def _edge(d: np.ndarray, fs: float) -> tuple[float, float, tuple[np.ndarray, np.ndarray] | None]:
    """One edge's transient ``d`` (the current above the new steady state, made positive; the edge at index 0):
    tau (s) fitted from 80% to 20% of its peak, the charge it carries (pA s), and the fitted stretch."""
    search = max(3, d.size // 2)
    p = int(np.argmax(d[:search]))
    top = d[p]
    if not top > 0:
        return NAN, NAN, None
    after = np.arange(p, d.size)
    a = after[d[p:] <= 0.8 * top]
    if not a.size:
        return NAN, NAN, None
    a = int(a[0])
    b = np.flatnonzero(d[a:] <= 0.2 * top)
    b = a + (int(b[0]) if b.size else 0)
    k = np.arange(a, b + 1)
    k = k[d[k] > 0]
    if k.size < 2:
        return NAN, NAN, None
    slope, icept = np.polyfit(k / fs, np.log(d[k]), 1)
    if not slope < 0:
        return NAN, NAN, None
    tau = -1.0 / slope
    # the charge from the transient's first sample of its own sign: the samples of the other sign before it are the
    # amplifier's fast capacitance compensation (on the rig -319, -1164, -116 pA at each edge of a 10 mV step), not
    # charge into the cell; left out, Ra and Cm read as Clampex's on the same model cell (12.3 and 29.1, Clampex
    # 11.7-12.3 and 29.0-30.6; counted in, 13.9 and 25.8)
    lead = np.flatnonzero(d[:p + 1] >= 0)
    s0 = int(lead[0]) if lead.size else p
    return tau, float(np.sum(d[s0:])) / fs, (k, np.exp(icept + slope * k / fs))


def measure(y: np.ndarray, fs: float, amplitude: float, mode: str) -> Reading:
    """One averaged period measured (``y`` in pA in voltage clamp, mV in current clamp; low for its first quarter,
    high for the middle half, low again; ``amplitude`` peak to peak, mV or pA)."""
    n = y.size
    up, down = n // 4, 3 * n // 4
    half = down - up
    t = np.arange(n) / fs
    r = Reading(mode=mode, unit="pA" if mode == "VC" else "mV", t_ms=t * 1e3, trace=y)
    w = max(1, int(round(0.2 * half)))
    i1 = float(np.mean(y[down - w:down]))                  # the last 20% of the high half
    i2 = float(np.mean(y[up - w:up]))                      # and of the low half (the period is periodic)
    di = i1 - i2
    r.holding = (i1 + i2) / 2
    if mode != "VC":
        return _current_clamp(r, y, fs, up, down, i2, amplitude)
    if di * amplitude > 0:
        r.rt_mohm = amplitude / di * 1e3                    # mV / pA = GOhm -> MOhm
    if not r.rt_mohm >= 0:
        noise = float(np.std(y[up - w:up]))
        r.what = "seal" if abs(di) < 3 * noise / np.sqrt(w) + 1e-12 else "no step"
        return r
    if r.rt_mohm < FLOOR_MOHM:
        r.what = "open pipette"
        return r
    if r.rt_mohm > CEILING_MOHM:
        r.what = "seal"
        return r
    r.what = "cell"
    rolled = np.roll(y, -up)                               # the rising edge at 0, the falling at ``half``
    sign = np.sign(amplitude)
    taus, charges = [], []
    for start, base, s in ((0, i1, sign), (half, i2, -sign)):
        d = s * (rolled[start:start + half] - base)
        tau, q, fitted = _edge(d, fs)
        if not (tau > 2.0 / fs):                           # faster than the samples resolve: no Ra from it
            continue
        taus.append(tau)
        charges.append(q + abs(di) * tau)                  # Q1 + dI tau
        k, fy = fitted
        at = (up + start + k) % n
        r.fits.append((at / fs * 1e3, base + s * fy))
    if not taus:
        return r
    tau, q = float(np.mean(taus)), float(np.mean(charges))
    dv = abs(amplitude)
    ra = tau * dv / q * 1e3                                # s mV / (pA s) = GOhm -> MOhm
    rm = r.rt_mohm - ra
    r.tau_ms = tau * 1e3
    if ra > 0 and rm > 0:
        r.rs_mohm, r.rm_mohm = ra, rm
        r.cm_pf = q * r.rt_mohm / (dv * rm) * 1e3          # pA s / mV = nF -> pF
    return r


def _current_clamp(r: Reading, y: np.ndarray, fs: float, up: int, down: int, base: float,
                   amplitude: float) -> Reading:
    """The input resistance, tau and Cm from the charging on the rising half (a pulse may end before the cell settles:
    tau = Rm Cm, 16.5 ms on the model cell): its plateau is the step's own; the jump at the onset (the pipette, less
    any bridge balance) is left out."""
    from ..analysis.tools import fits

    t = np.arange(y.size) / fs
    w = max(1, int(round(0.2 * (down - up))))
    steady = float(np.mean(y[down - w:down])) - base
    if steady * amplitude <= 0:
        r.what = "no step"
        return r
    skip = max(2, int(0.0005 * fs))
    fit = fits.exp_rise(t[up + skip:down], y[up + skip:down] - base, purpose="membrane test, current clamp: charging")
    if fit.ok and fit["tau"] > 0:
        plateau = fit["amp"] + fit["offset"]
        r.tau_ms = fit["tau"] * 1e3
        r.rt_mohm = r.rm_mohm = plateau / amplitude * 1e3      # mV / pA = GOhm
        r.cm_pf = r.tau_ms / (fit["amp"] / amplitude * 1e3) * 1e3 if fit["amp"] * amplitude > 0 else NAN
        tt = t[up + skip:down]
        r.fits.append((tt * 1e3, base + fit["offset"] + fit["amp"] * (1 - np.exp(-(tt - tt[0]) / fit["tau"]))))
    else:
        r.rt_mohm = r.rm_mohm = steady / amplitude * 1e3
    r.what = "cell" if FLOOR_MOHM <= r.rm_mohm <= CEILING_MOHM else "seal" if r.rm_mohm > CEILING_MOHM else \
        "open pipette"
    return r


class MembraneTest:
    """Runs until stopped, in the background; ``latest`` is the newest reading, ``history`` the readings so far."""

    def __init__(self, serial: str | None = None, channel: int = 1, work: Path | None = None,
                 read_telegraph=telegraph.read, make_helper=digidata.lend) -> None:
        self.serial, self.channel = serial, channel
        self.work = work or scratch() / "test"
        self.read_telegraph, self.make_helper = read_telegraph, make_helper
        self.state, self.message = "idle", ""
        self.latest: Reading | None = None
        self.history: deque = deque(maxlen=20_000)
        self.settings = Settings()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._frames = 0

    @property
    def running(self) -> bool:
        return self.state in ("starting", "running")

    def start(self, settings: Settings | None = None) -> None:
        if self.running:
            return
        why = digidata.available()
        if why and self.make_helper is digidata.lend:
            raise RuntimeError(why)
        self.settings = settings or self.settings
        self.state, self.message, self.latest = "starting", "reading the amplifier's telegraph", None
        self.history.clear()
        self._stop.clear()
        self._frames = 0
        self._thread = threading.Thread(target=self._run, name="Clampwise membrane test", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        (self.work / "test.raw.stop").touch() if self.work.exists() else None

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        s = self.settings
        helper = None
        try:
            tg = self.read_telegraph(self.serial, self.channel)
            if tg is None:
                raise RuntimeError("the amplifier's telegraph did not answer (is Commander open?)")
            if tg.mode == "I=0" or not tg.command_per_volt > 0:
                raise RuntimeError(f"no command in {tg.mode}: switch the amplifier to voltage or current clamp")
            helper = self.make_helper()
            cal = helper.calibration
            period = s.period
            up, down = period // 4, 3 * period // 4               # high for the middle half, centred on the holding
            volts = np.zeros((period, acquire.N_AO))
            volts[:, 0] = -0.5 * s.amplitude / tg.command_per_volt
            volts[up:down, 0] = 0.5 * s.amplitude / tg.command_per_volt
            wave = np.empty((period, acquire.N_OUT), "<i2")
            for d in range(acquire.N_AO):
                wave[:, d] = acquire.dac_codes(volts[:, d], cal["dac_gains"][d], cal["dac_offsets"][d])
            wave[:, acquire.N_AO] = 0
            self.work.mkdir(parents=True, exist_ok=True)
            wfile, raw = self.work / "test.wave", self.work / "test.raw"
            for f in (wfile, raw, Path(str(raw) + ".stop")):
                f.unlink(missing_ok=True)
            wave.tofile(wfile)
            scale = tg.primary_v_per_unit                     # volts at the digitizer a pA (VC) or a mV (IC)
            reader = threading.Thread(target=self._read, args=(raw, period, scale, tg), daemon=True)
            self.state, self.message = "running", (f"{s.amplitude:g} {tg.command_unit} peak to peak at {s.hz:g} Hz "
                                                   f"({s.rate_hz / 1e3:g} kHz), {tg.mode}")
            reader.start()
            line = helper.record({"ai": [0], "period_us": 1e6 / s.rate_hz, "frames": int(s.hours * 3600 * s.rate_hz),
                                  "hold": acquire.holding_codes(None, cal), "waveform": str(wfile), "repeat": True,
                                  "out": str(raw), "ring_frames": period * RING_PULSES, "say_s": READ_EVERY_S},
                                 self._progress)
            self._stop.set()
            reader.join()
            if "error" in line:
                raise RuntimeError(line["error"])
            self.state, self.message = "idle", "stopped"
        except Exception as exc:
            self._stop.set()
            self.state, self.message = "failed", str(exc)
        finally:
            if helper is not None:
                helper.close()
            for f in self.work.glob("test.*") if self.work.exists() else []:
                f.unlink(missing_ok=True)

    def _progress(self, frames: int) -> None:
        self._frames = frames
        if self._stop.is_set():
            (self.work / "test.raw.stop").touch()

    def _read(self, raw: Path, period: int, scale: float, tg) -> None:
        s = self.settings
        t0 = time.monotonic()
        measured = 0
        while not self._stop.is_set():
            time.sleep(READ_EVERY_S)
            whole = self._frames // period                   # pulses recorded so far (the file holds the last ones)
            if whole < 1 or whole == measured:
                continue
            n = min(s.average, whole, RING_PULSES - 1)
            try:
                with open(raw, "rb") as fh:
                    parts = []
                    for k in range(whole - n, whole):
                        fh.seek(2 * (k % RING_PULSES) * period)
                        parts.append(fh.read(2 * period))
            except OSError:
                continue
            counts = np.frombuffer(b"".join(parts), "<i2")
            if counts.size < n * period:
                continue
            y = counts[: n * period].reshape(n, period).astype(float).mean(axis=0) * (10.0 / 32768) / scale
            r = measure(y, s.rate_hz, s.amplitude, tg.mode)
            r.at = time.monotonic() - t0
            r.resolution = (10.0 / 32768) / scale
            r.pulses = whole
            measured = whole
            self.latest = r
            self.history.append(r)
