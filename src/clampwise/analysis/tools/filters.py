"""Filters, in one place (B21, item 1).

Every analysis that filters a trace does it through here, so the same filter
gives the same answer on every screen and every result can say exactly what
was applied: ``FilterSpec.describe()`` reads "Butterworth 4-pole band-pass
1-200 Hz, zero-phase". The raw data are never changed: ``apply`` returns a
new array.

Two implementations are kept on purpose, because they are what the validated
analyses already use and they are not bit-identical to each other:

- Butterworth filters run as second-order sections (``sosfiltfilt``), the
  numerically safe form for higher orders;
- the Bessel low-pass runs in transfer-function form (``filtfilt``), as the
  Allen Institute's IPFX does before differentiating a trace for spike
  detection, and the spike counts are checked against IPFX.

Warnings (``check``) follow the Axon Guide's rule of thumb: a cutoff must sit
below the Nyquist frequency (half the sampling rate), and filtering above the
acquisition filter's cutoff removes nothing that is still there.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.signal import bessel, butter, filtfilt, iirnotch, lfilter, sosfilt, sosfiltfilt

from . import ledger

Kind = Literal["butterworth", "bessel", "gaussian"]
Band = Literal["low", "high", "band"]


@dataclass(frozen=True)
class FilterSpec:
    """What was applied: the words a methods section needs."""

    kind: Kind = "butterworth"
    band: Band = "low"
    order: int = 4
    low_hz: float = float("nan")      # the lower edge (high-pass and band-pass)
    high_hz: float = float("nan")     # the upper edge (low-pass and band-pass)
    zero_phase: bool = True           # forward and backward: no phase shift
    # 60 Hz (or 50 Hz) notch with its harmonics, applied before the filter.
    notch_hz: float = 0.0
    notch_harmonics: int = 1
    notch_q: float = 30.0
    # A Gaussian is described by its SD in seconds, not by an order.
    sigma_s: float = float("nan")

    def describe(self) -> str:
        if self.kind == "gaussian":
            text = f"Gaussian smoothing, SD {self.sigma_s * 1e3:g} ms"
        else:
            name = {"butterworth": "Butterworth", "bessel": "Bessel"}[self.kind]
            edges = {"low": f"low-pass {self.high_hz:g} Hz",
                     "high": f"high-pass {self.low_hz:g} Hz",
                     "band": f"band-pass {self.low_hz:g}-{self.high_hz:g} Hz"}[self.band]
            text = f"{name} {self.order}-pole {edges}, {'zero-phase' if self.zero_phase else 'causal'}"
        if self.notch_hz:
            harmonics = f" and {self.notch_harmonics - 1} harmonic(s)" if self.notch_harmonics > 1 else ""
            text = f"{self.notch_hz:g} Hz notch{harmonics}, then {text}"
        return text

    def check(self, fs: float, acquisition_hz: float | None = None) -> list[str]:
        """Warnings about this filter at sampling rate ``fs``: an edge at or
        above Nyquist, or above the acquisition filter (nothing left to remove)."""
        out = []
        nyquist = fs / 2
        for name, edge in (("lower", self.low_hz), ("upper", self.high_hz)):
            if np.isfinite(edge) and edge >= nyquist:
                out.append(f"the filter's {name} edge ({edge:g} Hz) is at or above the Nyquist frequency "
                           f"({nyquist:g} Hz at {fs:g} Hz sampling); it cannot be applied as set")
        if acquisition_hz and np.isfinite(self.high_hz) and self.high_hz >= acquisition_hz:
            out.append(f"the low-pass at {self.high_hz:g} Hz is above the acquisition filter "
                       f"({acquisition_hz:g} Hz): the recording already lacks what it would remove")
        return out


# ------------------------------------------------------------------ the filters


def _edges(spec: FilterSpec, fs: float) -> tuple[str, float | list[float]]:
    nyquist = fs / 2
    if spec.band == "low":
        return "low", min(spec.high_hz, nyquist * 0.95)
    if spec.band == "high":
        return "high", spec.low_hz
    return "band", [spec.low_hz, min(spec.high_hz, nyquist * 0.95)]


def apply(y: np.ndarray, fs: float, spec: FilterSpec, purpose: str = "") -> np.ndarray:
    """``y`` filtered as ``spec`` says, along its last axis. Returns a new array,
    and records what was applied in the methods ledger."""
    ledger.note(spec.describe(), purpose)
    y = np.asarray(y, dtype=float)
    if spec.notch_hz:
        y = notch(y, fs, spec.notch_hz, spec.notch_harmonics, spec.notch_q)
    if spec.kind == "gaussian":
        return gaussian(y, fs, spec.sigma_s)
    btype, wn = _edges(spec, fs)
    if spec.kind == "butterworth":
        sos = butter(spec.order, wn, btype=btype, fs=fs, output="sos")
        return sosfiltfilt(sos, y) if spec.zero_phase else sosfilt(sos, y)
    if spec.kind == "bessel":
        # transfer-function form, as IPFX (normalized frequency, no fs argument); not clamped
        raw = spec.high_hz if spec.band == "low" else (spec.low_hz if spec.band == "high" else [spec.low_hz, spec.high_hz])
        coefficient = (np.asarray(raw, float) / (fs / 2)).tolist()
        b, a = bessel(spec.order, coefficient, btype)
        return filtfilt(b, a, y) if spec.zero_phase else lfilter(b, a, y)
    raise ValueError(f"unknown filter kind {spec.kind!r}")


def notch(y: np.ndarray, fs: float, f0: float = 60.0, harmonics: int = 1, q: float = 30.0,
          max_hz: float = 300.0, purpose: str = "") -> np.ndarray:
    """``f0`` and its first ``harmonics - 1`` multiples removed with a narrow
    zero-phase notch (quality ``q``), each below ``max_hz`` and Nyquist."""
    out = np.asarray(y, dtype=float)
    if purpose:
        ledger.note(f"{f0:g} Hz notch with {harmonics - 1} harmonic(s), zero-phase (Q {q:g})", purpose)
    for k in range(1, harmonics + 1):
        f = f0 * k
        if f >= min(max_hz, fs / 2 * 0.95):
            break
        b, a = iirnotch(f, q, fs)
        out = filtfilt(b, a, out)
    return out


def bessel_low_pass_normalized(y: np.ndarray, order: int, coefficient: float, purpose: str = "") -> np.ndarray:
    """A zero-phase Bessel low-pass given its cutoff as a fraction of Nyquist, in
    transfer-function form: the call IPFX makes before differentiating a trace
    for spike detection, kept in that exact form so spike counts can be checked
    against IPFX."""
    ledger.note(f"Bessel {order}-pole low-pass at {coefficient:.3g} x Nyquist, zero-phase", purpose)
    b, a = bessel(order, coefficient, "low")
    return filtfilt(b, a, np.asarray(y, dtype=float))


def gaussian(y: np.ndarray, fs: float, sigma_s: float) -> np.ndarray:
    """Gaussian smoothing (SD ``sigma_s``), zero-phase by construction, with the
    ends held at their own level (``nearest``) so a trace is not pulled to zero."""
    from scipy.ndimage import gaussian_filter1d

    sigma = sigma_s * fs
    if sigma <= 0:
        return np.array(y, dtype=float)
    return gaussian_filter1d(np.asarray(y, dtype=float), sigma, axis=-1, mode="nearest")


def moving_average(y: np.ndarray, n: int, mode: str = "same") -> np.ndarray:
    """A boxcar of ``n`` samples (``np.convolve`` with ones / n), the smoothing
    most analyses use to find a crossing without noise triggering it early.
    ``mode="edge"``: the ends extended with their own values (``same`` length),
    so the first and last samples are not bent towards zero."""
    y = np.asarray(y, dtype=float)
    n = max(int(n), 1)
    if n == 1 or y.size < n:
        return y.copy()
    if mode == "edge":
        return np.convolve(np.pad(y, (n // 2, n - 1 - n // 2), mode="edge"), np.ones(n) / n, mode="valid")
    return np.convolve(y, np.ones(n) / n, mode=mode)


def savitzky_golay(y: np.ndarray, fs: float, window_s: float, order: int = 3, purpose: str = "") -> np.ndarray:
    """Savitzky-Golay smoothing (a local polynomial of ``order`` over ``window_s``;
    Savitzky & Golay 1964): smooths noise while keeping a fast rise's shape better
    than a boxcar of the same width. NeuroExpress offers it for noisy mini traces."""
    from scipy.signal import savgol_filter

    y = np.asarray(y, dtype=float)
    n = max(order + 2, int(round(window_s * fs)) | 1)        # odd, longer than the order
    if y.size <= n:
        return y.copy()
    ledger.note(f"Savitzky-Golay smoothing, order {order} over {n / fs * 1e3:.2g} ms", purpose)
    return savgol_filter(y, n, order)


# ------------------------------------------------------------------ the specs the analyses use

def band_pass(low_hz: float, high_hz: float, order: int = 4) -> FilterSpec:
    return FilterSpec("butterworth", "band", order, low_hz, high_hz)


def low_pass(high_hz: float, order: int = 4, kind: Kind = "butterworth") -> FilterSpec:
    return FilterSpec(kind, "low", order, float("nan"), high_hz)


def high_pass(low_hz: float, order: int = 2) -> FilterSpec:
    return FilterSpec("butterworth", "high", order, low_hz, float("nan"))


def bessel_low_pass(high_hz: float, order: int = 4) -> FilterSpec:
    return FilterSpec("bessel", "low", order, float("nan"), high_hz)


def with_notch(spec: FilterSpec, f0: float = 60.0, harmonics: int = 5) -> FilterSpec:
    return replace(spec, notch_hz=f0, notch_harmonics=harmonics)
