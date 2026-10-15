"""The shared fits (B21): known answers at the noise of real recordings,
confidence intervals that cover the truth, flags on bounds and failures,
model choice shown, and the old fitting module still importable."""

from __future__ import annotations

import numpy as np
import pytest

from clampwise.analysis.tools import fits, ledger

RNG = np.random.default_rng(7)


def _decay(tau=0.02, amp=-50.0, offset=-3.0, noise=1.5, n=2000, dt=5e-5):
    t = np.arange(n) * dt
    return t, amp * np.exp(-t / tau) + offset + RNG.normal(0, noise, n)


def test_single_exponential_recovers_tau_with_an_interval_that_covers_it():
    t, y = _decay()
    fit = fits.exp_decay(t, y)
    assert fit.ok and fit.model == "exp1"
    assert abs(fit["tau"] - 0.02) / 0.02 < 0.03
    lo, hi = fit.ci["tau"]
    assert lo < 0.02 < hi and hi - lo < 0.004
    assert fit.residuals.size == 2000 and abs(fit.residuals.mean()) < 0.5
    assert fit.flags == []
    assert np.allclose(fit.curve(t), fit["amp"] * np.exp(-t / fit["tau"]) + fit["offset"])
    assert "tau" in fit.describe() and "r2" in fit.describe()


def test_profiled_fit_agrees_with_the_gradient_fit():
    t, y = _decay(noise=3.0)
    a, b = fits.exp_decay(t, y), fits.exp_decay_profiled(t, y)
    assert abs(a["tau"] - b["tau"]) / a["tau"] < 1e-3
    assert b.ci == {} and b.predict is not None


def test_two_exponentials_recovered_and_preferred_only_when_earned():
    t = np.arange(4000) * 5e-5
    y = -30 * np.exp(-t / 0.005) - 20 * np.exp(-t / 0.05) + RNG.normal(0, 0.8, t.size)
    mono, bi, chosen, reason = fits.choose_exponentials(t, y)
    assert chosen == "exp2" and reason.startswith("F test")
    assert abs(bi["tau1"] - 0.005) / 0.005 < 0.15 and abs(bi["tau2"] - 0.05) / 0.05 < 0.1
    mono2, bi2, chosen2, reason2 = fits.choose_exponentials(t, y, criterion="AICc")
    assert chosen2 == "exp2" and "AICc" in reason2
    # a true single exponential is not called double
    t1, y1 = _decay(noise=1.0)
    _, _, chosen3, _ = fits.choose_exponentials(t1, y1)
    assert chosen3 == "exp1"


def test_bound_and_failure_are_flagged_never_raised():
    t = np.arange(50) * 1e-3
    bad = fits.nonlinear(lambda x, a: a * x, [], [], (1.0,), ("a",), label="line")
    assert not bad.ok and bad.flags and bad.flags[0].startswith("fit failed")
    # a tau pinned at its lower limit says so
    fast = -50 * np.exp(-t / 1e-7) - 3 + RNG.normal(0, 0.1, 50)
    pinned = fits.nonlinear(lambda tt, a, tau, o: a * np.exp(-tt / tau) + o, t, fast, (-50, 1e-6, -3),
                            ("amp", "tau", "offset"), label="exp1",
                            bounds=([-np.inf, 1e-6, -np.inf], [np.inf, 1.0, np.inf]))
    assert pinned.ok and any("tau at the fit's limit" in f for f in pinned.flags)


def test_boltzmann_hill_gaussian_line_and_synaptic_waveform():
    v = np.arange(-80.0, 41.0, 10.0)
    g = 10.0 / (1 + np.exp((-20.0 - v) / 6.0)) + RNG.normal(0, 0.1, v.size)
    b = fits.boltzmann(v, g)
    assert b.ok and abs(b["v_half"] + 20) < 1.5 and abs(b["slope"] - 6) < 1.0 and abs(b["top"] - 10) < 0.5
    assert b.ci["v_half"][0] < -20 < b.ci["v_half"][1]

    x = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    y = 100 / (1 + (2.0 / x) ** 2) + RNG.normal(0, 2.0, x.size)
    h = fits.hill(x, y)
    assert h.ok and abs(h["ec50"] - 2.0) < 0.4

    centres = np.arange(-20.0, 20.5, 0.5)
    counts = 500 * np.exp(-0.5 * ((centres - 1.0) / 3.0) ** 2) + RNG.normal(0, 5, centres.size)
    gauss = fits.gaussian(centres, counts, (0.0, 2.0, 400.0))
    assert gauss.ok and abs(gauss["mu"] - 1.0) < 0.2 and abs(abs(gauss["sd"]) - 3.0) < 0.2

    line = fits.linear([0, 1, 2, 3, 4], [1.0, 3.1, 4.9, 7.2, 8.9])
    assert line.ok and abs(line["slope"] - 2.0) < 0.1 and line.ci["slope"][0] < 2.0 < line.ci["slope"][1]

    t = np.arange(3000) * 2e-5
    syn = -40 * (1 - np.exp(-t / 0.0005)) * np.exp(-t / 0.008) + 2.0 + RNG.normal(0, 0.7, t.size)
    s = fits.synaptic_waveform(t, syn)
    assert s.ok and abs(s["tau_rise"] - 0.0005) / 0.0005 < 0.25 and abs(s["tau_decay"] - 0.008) / 0.008 < 0.1


def test_every_fit_is_recorded_in_the_ledger():
    t, y = _decay()
    with ledger.recording() as lines:
        fits.exp_decay(t, y)
        fits.boltzmann(np.arange(-80.0, 41.0, 10.0), 10 / (1 + np.exp((-20.0 - np.arange(-80.0, 41.0, 10.0)) / 6)))
    assert lines == ["single-exponential fit", "Boltzmann fit"]


def test_purposes_reach_the_ledger():
    t, y = _decay()
    with ledger.recording() as lines:
        fits.exp_decay(t, y, purpose="evoked decay tau")
        fits.linear([0, 1, 2], [0, 1, 2], purpose="baseline drift")
    assert lines == ["single-exponential fit (evoked decay tau)", "straight-line fit (baseline drift)"]


@pytest.mark.parametrize("size", [0, 2, 4])
def test_too_few_points_give_a_not_ok_fit(size):
    t = np.arange(size) * 1e-3
    assert not fits.exp_decay(t, t).ok
    assert not fits.exp_rise(t, t).ok
    assert not fits.biexp_decay(t, t).ok


def test_no_analysis_fits_or_filters_on_its_own():
    """B21: every fit and filter goes through analysis/tools, so the methods
    ledger lists it and one change reaches every analysis."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "clampwise"
    banned = re.compile(r"\b(curve_fit|least_squares|butter|bessel|filtfilt|sosfiltfilt|lfilter|iirnotch)\s*\(")
    offenders = []
    for path in root.rglob("*.py"):
        if "tools" in path.relative_to(root).parts:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if banned.search(line.split("#")[0]):
                offenders.append(f"{path.relative_to(root)}:{n}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
