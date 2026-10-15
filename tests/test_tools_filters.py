"""The shared filters (B21): analytic answers, the ledger, and bit-identity with
the calls the analyses made before they were moved here."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import bessel, butter, filtfilt, iirnotch, sosfiltfilt

from clampwise.analysis.tools import filters, ledger

FS = 20_000.0


def _tone(f, fs=FS, n=40_000, amp=1.0):
    t = np.arange(n) / fs
    return t, amp * np.sin(2 * np.pi * f * t)


def test_low_pass_keeps_slow_and_removes_fast():
    t, slow = _tone(10.0)
    _, fast = _tone(3000.0)
    out = filters.apply(slow + fast, FS, filters.low_pass(200.0))
    mid = slice(5000, 35000)
    assert np.abs(out[mid] - slow[mid]).max() < 0.02        # the 10 Hz tone survives
    assert np.std(out[mid] - slow[mid]) < 0.01                # the 3 kHz tone is gone


def test_band_pass_passes_the_band_only():
    t, inband = _tone(50.0)
    _, below = _tone(0.2)
    _, above = _tone(2000.0)
    out = filters.apply(inband + below + above, FS, filters.band_pass(5.0, 300.0))
    mid = slice(10000, 30000)
    assert np.abs(out[mid] - inband[mid]).max() < 0.05


def test_notch_removes_60_hz_and_its_harmonics_but_not_40():
    t, line = _tone(60.0)
    _, h2 = _tone(120.0)
    _, keep = _tone(40.0)
    spec = filters.with_notch(filters.low_pass(2000.0), 60.0, harmonics=3)
    out = filters.apply(line + h2 + keep, FS, spec)
    mid = slice(10000, 30000)
    assert np.abs(out[mid] - keep[mid]).max() < 0.05
    assert "60 Hz notch and 2 harmonic(s)" in spec.describe()


def test_gaussian_is_zero_phase_and_unit_gain():
    t, slow = _tone(5.0)
    out = filters.apply(slow, FS, filters.FilterSpec("gaussian", sigma_s=0.0005))
    mid = slice(5000, 35000)
    assert np.abs(out[mid] - slow[mid]).max() < 1e-3
    assert np.allclose(filters.gaussian(np.ones(1000), FS, 0.001), 1.0)


def test_check_warns_at_nyquist_and_above_the_acquisition_filter():
    spec = filters.low_pass(12_000.0)
    notes = spec.check(FS, acquisition_hz=10_000.0)
    assert any("Nyquist" in n for n in notes) and any("acquisition filter" in n for n in notes)
    assert filters.low_pass(2000.0).check(FS, acquisition_hz=10_000.0) == []


def test_describe_reads_as_a_methods_sentence():
    assert filters.band_pass(1.0, 200.0).describe() == "Butterworth 4-pole band-pass 1-200 Hz, zero-phase"
    assert filters.bessel_low_pass(10_000.0).describe() == "Bessel 4-pole low-pass 10000 Hz, zero-phase"
    assert filters.high_pass(1.0).describe() == "Butterworth 2-pole high-pass 1 Hz, zero-phase"


def test_the_raw_array_is_never_changed():
    rng = np.random.default_rng(0)
    y = rng.normal(size=5000)
    before = y.copy()
    filters.apply(y, FS, filters.band_pass(1.0, 200.0))
    filters.apply(y, FS, filters.bessel_low_pass(5000.0))
    filters.apply(y, FS, filters.with_notch(filters.low_pass(2000.0)))
    assert np.array_equal(y, before)


# ---- bit-identity with the calls the analyses made before (the fingerprint's guarantee)

@pytest.fixture
def noisy():
    rng = np.random.default_rng(1)
    return rng.normal(size=20_000) + np.sin(2 * np.pi * 60.0 * np.arange(20_000) / FS)


def test_butterworth_band_pass_matches_the_old_call(noisy):
    hi = min(200.0, FS / 2 * 0.9)
    sos = butter(4, [1.0, hi], btype="band", fs=FS, output="sos")
    expected = sosfiltfilt(sos, noisy)
    assert np.array_equal(filters.apply(noisy, FS, filters.band_pass(1.0, hi)), expected)


def test_butterworth_high_pass_matches_the_old_call(noisy):
    sos = butter(2, 1.0, btype="high", fs=FS, output="sos")
    assert np.array_equal(filters.apply(noisy, FS, filters.high_pass(1.0)), sosfiltfilt(sos, noisy))


def test_bessel_low_pass_matches_ipfx_style_call(noisy):
    nyquist_khz = FS / 1e3 / 2
    b, a = bessel(4, 5.0 / nyquist_khz, "low")
    expected = filtfilt(b, a, noisy)
    assert np.array_equal(filters.apply(noisy, FS, filters.bessel_low_pass(5000.0)), expected)
    assert np.array_equal(filters.bessel_low_pass_normalized(noisy, 4, 5.0 / nyquist_khz), expected)


def test_notch_matches_the_old_call(noisy):
    out = noisy
    for f in (60.0, 120.0, 180.0, 240.0):
        b, a = iirnotch(f, 30.0, FS)
        out = filtfilt(b, a, out)
    assert np.array_equal(filters.notch(noisy, FS, 60.0, harmonics=5), out)


def test_ledger_collects_what_ran_and_only_inside_a_recording():
    filters_used = filters.band_pass(1.0, 200.0)
    with ledger.recording() as lines:
        ledger.note(filters_used.describe(), "event detection")
        ledger.note(filters_used.describe(), "event detection")      # once
        ledger.note("single exponential from 80% of the peak", "decay tau")
    assert lines == ["Butterworth 4-pole band-pass 1-200 Hz, zero-phase (event detection)",
                     "single exponential from 80% of the peak (decay tau)"]
    ledger.note("outside", "nothing")          # no ledger open: ignored
    assert ledger.current() is None
