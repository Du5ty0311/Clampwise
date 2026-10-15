"""Heat maps (Phase 2): known answers on simulated data. A missing value stays
missing; sequential limits are percentiles (over the values above the floor
when most of the map is empty); diverging limits are symmetric about their
centre; a known spectrogram peak, a known % of baseline matrix, a known
pulse-ratio matrix, known event rates and known correlations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clampwise.analysis import heatmaps as hm
from clampwise.analysis.result import Result
from clampwise.core.recording import ArraySource, Channel, Recording, Role, Stimulus


def _run(rec=None, kind="", **results):
    return SimpleNamespace(recording=rec, recording_type=kind, results=results)


def _gap_free(y, fs):
    return Recording(Path("sim.abf"), fs, [Channel(0, "IN 0", "pA", Role.SIGNAL)], ArraySource([y[None, :]]), np.zeros(1),
                     mode="gap-free")


def test_levels_keep_missing_values_and_follow_the_scale_rules():
    m = np.arange(100, dtype=float).reshape(10, 10)
    m[0, 0] = np.nan
    seq = hm.Heatmap(m, np.arange(10.0), np.arange(10.0), "", "", "", "", "", "", "")
    a, b = hm.levels(seq)
    assert a == pytest.approx(np.nanpercentile(m, 1)) and b == pytest.approx(np.nanpercentile(m, 99))
    assert hm.levels(seq, "full") == (1.0, 99.0) and hm.levels(seq, "fixed", (5, 6)) == (5.0, 6.0)
    sparse = np.zeros((10, 10))
    sparse[2, 3], sparse[5, 5] = 4.0, 8.0              # a PSTH: mostly empty
    lo, hi = hm.levels(hm.Heatmap(sparse, np.arange(10.0), np.arange(10.0), "", "", "", "", "", "", ""))
    assert lo == 0.0 and hi > 7.5                      # not 0 to ~0: the busy bins do not saturate
    pct = hm.Heatmap(np.array([[60.0, 100.0, 130.0]]), np.arange(3.0), np.arange(1.0), "", "", "", "", "", "", "",
                     "diverging", 100.0)
    lo, hi = hm.levels(pct, "full")
    assert (lo, hi) == (60.0, 140.0)                   # symmetric about 100
    assert hm.shared_levels([seq, pct], "full") == (1.0, 130.0)


def test_a_known_spectrogram_peak_hann_and_morlet():
    fs, a, f0 = 2000.0, 10.0, 40.0
    t = np.arange(int(60 * fs)) / fs
    y = np.random.default_rng(0).normal(0, 0.5, t.size)
    on = (t >= 20) & (t < 40)
    y[on] += a * np.sin(2 * np.pi * f0 * t[on])
    run = _run(_gap_free(y, fs), "vc_spontaneous")
    sp = hm.spectrogram(run, "hann")
    assert sp.matrix.shape == (sp.y.size, sp.x.size) and sp.value_unit == "pA²/Hz"
    mid = (sp.x > 22) & (sp.x < 38)
    peak_row = int(np.argmax(np.nanmean(sp.matrix[:, mid], axis=1)))
    assert sp.y[peak_row] == pytest.approx(f0, abs=1.0)
    df = sp.y[1] - sp.y[0]
    assert np.nanmean(sp.matrix[:, mid].sum(axis=0) * df) == pytest.approx(a ** 2 / 2, rel=0.1)   # Parseval
    quiet = sp.x < 18
    assert np.nanmean(sp.matrix[peak_row, quiet]) < 0.01 * np.nanmean(sp.matrix[peak_row, mid])
    mw = hm.spectrogram(run, "morlet")
    row = int(np.argmin(np.abs(mw.y - f0)))
    assert int(np.argmax(np.nanmean(mw.matrix[:, (mw.x > 25) & (mw.x < 35)], axis=1))) == row
    assert np.nanmean(mw.matrix[row, (mw.x > 25) & (mw.x < 35)]) == pytest.approx(a ** 2, rel=0.15)
    bp = hm.band_power(run)
    gamma = bp.row_labels.index("gamma 30-80 Hz")
    assert bp.scale == "diverging" and np.nanmean(bp.matrix[gamma, (bp.x > 22) & (bp.x < 38)]) > 20.0
    db = hm.in_db(sp)
    assert db.value_unit == "dB re 1 pA/Hz" and np.nanmax(db.matrix) == pytest.approx(
        10 * np.log10(np.nanmax(sp.matrix)))


def test_a_known_percent_of_baseline_matrix():
    rows = [{"episode": e, "duration_s": d, "response": k, "from_step_s": 4.0 * k, "pct": 100.0 - 10 * e * (k > 0),
             "power_pct": np.nan} for e, d in ((1, 1.0), (2, 5.0)) for k in (-2, -1, 1, 2, 3)]
    m = hm.dsi_course(_run(dsi=Result("dsi", 1, tables={"course": rows})), "dsi", "pct")
    assert list(m.x) == [-2, -1, 0, 1, 2, 3] and m.center == 100.0 and m.scale == "diverging"
    expect = np.array([[100, 100, np.nan, 90, 90, 90], [100, 100, np.nan, 80, 80, 80]], float)
    np.testing.assert_array_equal(m.matrix, expect)            # the step's column stays blank
    assert m.row_labels == ["1 (1 s)", "2 (5 s)"]


def test_pulse_ratios_and_event_rates_and_sweeps():
    rows = [{"sweep": s, "pulse": p, "amplitude": -100.0 * (1 + 0.5 * (p - 1)) * (1 + s), "onset_s": 0.1 + 0.05 * p}
            for s in range(3) for p in (1, 2, 3)]
    m = hm.pulses(_run(evoked=Result("evoked", 1, tables={"sweeps": rows})))
    np.testing.assert_allclose(m.matrix, [[1, 1.5, 2]] * 3)
    assert m.center == 1.0 and list(m.y) == [1, 2, 3]
    # gap-free events: 2 per second in minute 1, none in minute 2, minute 3 not analyzed
    fs = 1000.0
    rec = _gap_free(np.zeros(int(180 * fs)), fs)
    ev = [{"sweep": 0, "onset_s": t, "accepted": True} for t in np.arange(0.25, 60, 0.5)]
    res = Result("events", 1, tables={"events": ev, "analyzed": [{"sweep": 0, "start_s": 0.0, "end_s": 120.0}]})
    r = hm.event_rate(_run(rec, "vc_spontaneous", events=res))
    assert r.matrix.shape == (3, 60)
    np.testing.assert_allclose(r.matrix[0], 2.0)
    np.testing.assert_allclose(r.matrix[1], 0.0)
    assert np.isnan(r.matrix[2]).all()                          # not analyzed: missing, never zero
    # every sweep vs. time: a 1-sample spike survives the reduction to columns
    sweeps = np.zeros((4, 30_000))
    sweeps[2, 12_345] = 50.0
    rec = Recording(Path("s.abf"), 10_000.0, [Channel(0, "IN 0", "mV", Role.SIGNAL)], ArraySource([sweeps]),
                    sweep_start_times_s=np.arange(4) * 5.0)
    rec.stimulus = Stimulus([np.array([0.5])] * 4, [np.array([1e-4])] * 4, "typed")
    s = hm.sweeps(_run(rec))
    assert s.matrix.shape[0] == 4 and s.matrix.shape[1] <= hm.MAX_COLUMNS and np.nanmax(s.matrix[2]) == 50.0
    assert [mk.label for mk in s.markers] == ["stimulus"]


def test_known_correlations():
    x = np.arange(12.0)
    c = hm.correlation({"a": x, "b": x ** 2, "c": -x, "flat": np.ones(12)})
    assert c.row_labels is not None and "flat" not in c.row_labels and c.fixed == (-1.0, 1.0)
    i, j, k = (c.row_labels.index(n) for n in "abc")
    assert c.matrix[i, j] == pytest.approx(1.0) and c.matrix[i, k] == pytest.approx(-1.0)
    assert c.mask[i, j] and c.mask[i, k]


def test_the_catalogue_offers_only_what_a_run_can_make():
    rows = [{"episode": 1, "duration_s": 1.0, "response": k, "from_step_s": 4.0 * k, "pct": 50.0, "power_pct": 50.0}
            for k in (-1, 1)]
    keys = [k for k, _, _ in hm.catalogue(_run(None, "dsi", dsi=Result("dsi", 1, tables={"course": rows})))]
    assert keys == ["heatmap.dsi.pct", "heatmap.dsi.power_pct"]
    assert hm.catalogue(_run(None, "cc_steps")) == []
