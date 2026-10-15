"""Field potentials and LTP (B7): the slope where WinLTP takes it, the volley
read at the average's latency, paired pulses, and the time course as % of the
minutes before the induction (decision D4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clampwise.analysis import pipeline
from clampwise.core.recording import ArraySource, Channel, Recording, Role, Stimulus


def field_series(n: int = 60, every_s: float = 20.0, after: float = 1.5, induction_min: float = 10.0,
                 isi_s: float = 0.05, fs: float = 20_000.0, seed: int = 0) -> Recording:
    """fEPSPs (a fast artifact, a volley at 2 ms, a negative fEPSP peaking at 6
    ms) at 10 ms and 10 ms + ``isi_s``; after the induction every fEPSP is
    ``after`` times as large, the volley unchanged; pulse 2 is 1.4 times pulse 1."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.15 * fs)) / fs
    sig = np.zeros((n, t.size))
    for k in range(n):
        scale = after if k * every_s / 60 >= induction_min else 1.0
        for j, t0 in enumerate((0.01, 0.01 + isi_s)):
            x = t - t0
            on = x >= 0
            sig[k, on] += 3.0 * np.exp(-x[on] / 0.0002)                              # the artifact
            sig[k] += -0.15 * np.exp(-((x - 0.002) / 0.0003) ** 2)                    # the volley
            g = np.where(x > 0.0025, (x - 0.0025), 0.0)
            sig[k] += -scale * (1.4 if j else 1.0) * 1.0 * (g / 0.0035) ** 2 * np.exp(2 * (1 - g / 0.0035)) \
                * (g > 0)
        sig[k] += rng.normal(0, 0.01, t.size)
    rec = Recording(Path("sim_field.FP0"), fs, [Channel(0, "AD0", "mV", Role.SIGNAL)], ArraySource([sig]),
                    sweep_start_times_s=np.arange(n) * every_s, protocol="field LTP")
    rec.header["recording_kind"] = "field"
    rec.stimulus = Stimulus([np.array([0.01, 0.01 + isi_s])] * n, [np.array([1e-4, 1e-4])] * n, "typed",
                            "electrical")
    return rec


def test_the_ltp_is_the_late_slope_over_the_minutes_before_the_induction():
    rec = field_series()
    r = pipeline.analyze(rec, pipeline.Settings(by_analysis={"field": {"induction_min": 10.0}}))
    assert r.recording_type == "field_evoked"
    v = r.results["field"].values
    assert v["fepsp_slope"] < 0 and v["fiber_volley_ms"] == pytest.approx(2.0, abs=0.2)
    assert v["late_pct"] == pytest.approx(150.0, abs=3.0)
    assert (v["baseline_from_min"], v["baseline_to_min"]) == (0.0, 10.0)
    assert v["ppf_slope"] == pytest.approx(1.4, abs=0.05)
    assert abs(v["baseline_drift"]) < 5 and np.isnan(v["slope_per_volley"])
    rows = r.results["field"].tables["sweeps"]
    assert sum(row["in_baseline"] == "yes" for row in rows) == 30
    # with no induction entered the first 10 min are the baseline, and it says so
    r2 = pipeline.analyze(rec).results["field"]
    assert r2.values["late_pct"] == pytest.approx(150.0, abs=3.0) and any("no induction" in n for n in r2.notes)


def test_twenty_eighty_is_the_alternative_and_says_so():
    rec = field_series(n=10, after=1.0)
    a = pipeline.analyze(rec).results["field"]
    b = pipeline.analyze(rec, pipeline.Settings(by_analysis={"field": {"slope_method": "20-80%"}})).results["field"]
    assert "Maximum Slope" in a.choices["slope"] and "20%" in b.choices["slope"]
    # the steepest line is steeper than the 20-80% one
    assert abs(a.values["fepsp_slope"]) >= abs(b.values["fepsp_slope"]) > 0
