"""The manual workbench: regions, Clampfit's statistics, fits, transforms, clicks, exclusions, templates and
provenance, against simulated recordings with known answers; the screen built offscreen and driven."""

from __future__ import annotations

import numpy as np
import pytest

from clampwise import simulate
from clampwise.analysis import workbench as wb
from clampwise.analysis.stimulus import find_stimulus


def _light_recording(seed=2, **kw):
    sim = simulate.photostim_steps(seed=seed, levels_pa=[0] * 8, onsets_s=(0.3356, 0.4406), **kw)
    rec = sim.recording
    rec.stimulus = find_stimulus(rec)
    return sim, rec


def _abf(write_abf, path, rec):
    """The simulated light recording as an ABF file: the cell on IN 0, the light's command on AM_Stim."""
    return write_abf(path, np.stack([rec.read(0), rec.read(2)]), fs=rec.sample_rate_hz, units=("mV", "V"),
                     names=("IN 0", "AM_Stim"), command_units="pA", protocol="sim_photostim_steps")


def _measurement(rec, **kw):
    m = wb.HandMeasurement("light", str(rec.source), wb.Selection(list(range(rec.n_sweeps)), "", "each"),
                           regions=[wb.Region("base", -0.020, -0.001, "baseline", per_pulse=True),
                                    wb.Region("resp", 0.0015, 0.050, "measure", per_pulse=True),
                                    wb.Region("decay", 0.0, 0.080, "fit", per_pulse=True, pulse=1)],
                           measures=["peak", "time_of_peak", "area", "rise_time", "half_width", "onset_latency",
                                     "decay_time", "max_rise_slope", "spike_count", "slope", "mean"])
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def test_regions_per_pulse_measure_each_response_against_its_own_baseline():
    sim, rec = _light_recording()
    res = wb.measure(rec, _measurement(rec))
    rows = [r for r in res["rows"] if r["region"] == "resp"]
    assert len(rows) == 16                                             # 8 sweeps x 2 pulses
    p1 = [r for r in rows if r["pulse"] == 1]
    assert np.mean([r["peak"] for r in p1]) == pytest.approx(sim.truth["amplitudes"][0], abs=0.3)
    assert np.mean([r["onset_latency"] for r in p1]) == pytest.approx(3.0, abs=0.4)      # the 2 x noise onset
    assert all(abs(r["baseline"] - (-60.0)) < 0.5 for r in p1)
    p2 = [r for r in rows if r["pulse"] == 2]
    assert np.mean([r["peak"] for r in p2]) == pytest.approx(sim.truth["amplitudes"][0] * sim.truth["ppr"], abs=0.3)
    assert res["summary"]["resp"]["peak"]["n"] == 16 and np.isfinite(res["summary"]["resp"]["peak"]["sd"])


def test_the_statistics_list_on_a_known_waveform():
    fs = 20_000.0
    t = np.arange(int(0.2 * fs)) / fs
    y = np.zeros(t.size)
    y[2000:] = 5.0 * (1 - np.exp(-(t[2000:] - 0.1) / 0.002)) * np.exp(-(t[2000:] - 0.1) / 0.02)
    y += 2.0                                                              # a baseline of 2
    s = wb.statistics(t, y, 0.1, 0.2, fs, baseline=2.0, noise_sd=0.01)
    pk = y.max() - 2.0
    assert s["peak"] == pytest.approx(pk, abs=0.02) and s["antipeak"] == pytest.approx(0.0, abs=0.02)
    assert s["time_of_peak"] == pytest.approx((np.argmax(y) - 2000) / fs * 1e3, abs=0.1)
    assert s["max"] == pytest.approx(pk, abs=0.02) and s["min"] == pytest.approx(0.0, abs=0.02)
    assert s["peak_to_peak"] == pytest.approx(pk, abs=0.02)
    assert 0 < s["rise_time"] < 5.0 and s["half_width"] > 5.0
    assert s["area"] == pytest.approx(np.trapezoid(y[2000:] - 2.0, t[2000:]) * 1e3, rel=0.01)
    assert s["max_rise_slope"] > 0 > s["max_decay_slope"]
    assert s["onset_latency"] == pytest.approx(0.0, abs=0.3)
    assert s["spike_count"] == 0
    # the decay's time constant, sweep by sweep: 20 ms here (the rise still adds a little just after the peak)
    assert s["decay_tau"] == pytest.approx(20.0, rel=0.08)
    pure = np.where(t >= 0.1, 2.0 - 5.0 * np.exp(-(t - 0.1) / 0.012), 2.0)        # a downward response, tau 12 ms
    down = wb.statistics(t, pure, 0.1, 0.2, fs, baseline=2.0)
    assert down["decay_tau"] == pytest.approx(12.0, rel=0.01)
    # a downward response is measured like an upward one: its peak negative, as recorded, its kinetics found
    # (they were left blank, and the peak came out positive)
    assert down["peak"] == pytest.approx(-5.0, abs=0.01) and down["antipeak"] == pytest.approx(0.0, abs=0.01)
    assert down["decay_time"] == pytest.approx(12.0 * np.log(9.0), rel=0.02)          # 90 to 10% of an exponential
    assert np.isnan(wb.statistics(t, pure, 0.1, 0.2, fs, baseline=2.0, want={"peak"})["decay_tau"])   # not asked for
    # a region of flat baseline: nothing
    s0 = wb.statistics(t, y, 0.0, 0.09, fs, baseline=2.0)
    assert abs(s0["mean"]) < 1e-9 and s0["sd"] == 0
    # the foot and curvature onsets land near the true onset too
    for method, within in (("foot", 1.0), ("curvature", 5.0)):
        s2 = wb.statistics(t, y, 0.1, 0.2, fs, baseline=2.0, options=wb.Options(onset_method=method))
        assert s2["onset_latency"] == pytest.approx(0.0, abs=within)
    # spikes in a region
    z = np.zeros(t.size)
    for at in (0.05, 0.07, 0.09):
        i = int(at * fs)
        z[i - 20:i + 80] += 90.0 * np.exp(-((t[i - 20:i + 80] - t[i]) / 0.0004) ** 2)
    s3 = wb.statistics(t, z, 0.0, 0.2, fs)
    assert s3["spike_count"] == 3 and s3["instantaneous_frequency"] == pytest.approx(50.0, rel=0.05)


def test_fits_on_a_region_recover_the_truth_and_say_their_words():
    fs = 20_000.0
    t = np.arange(int(0.3 * fs)) / fs
    y = 8.0 * np.exp(-t / 0.025) + 1.0
    f = wb.fit_region(t, y, 0.0, 0.3, "exp1")
    assert f["ok"] and f["params"]["tau"] == pytest.approx(25.0, rel=0.01) and "tau" in f["words"]
    assert f["curve_t"].size == t.size and np.abs(f["residuals"]).max() < 1e-6
    f2 = wb.fit_region(t, y, 0.0, 0.3, "linear")
    assert f2["ok"] and f2["params"]["slope"] < 0
    f3 = wb.fit_region(t, y, 0.0, 0.3, "custom", custom="a*exp(-t/b)+c")
    assert f3["ok"] and f3["params"]["b"] == pytest.approx(25.0, rel=0.02)
    f4 = wb.fit_region(t, y, 0.0, 0.3, "poly2")
    assert f4["ok"] and f4["r2"] > 0.7
    bad = wb.fit_region(t, y, 0.0, 0.0001, "exp1")
    assert not bad["ok"] and "too few" in bad["words"]
    assert not wb.fit_region(t, y, 0.0, 0.3, "custom", custom="")["ok"]


def test_transforms_never_touch_the_raw_data_and_subtract_between_sweeps():
    sim, rec = _light_recording()
    before = rec.sweep(0).copy()
    m = _measurement(rec, transforms=[wb.Transform("lowpass", {"hz": 1000.0}), wb.Transform("offset", {"value": 3.0}),
                                      wb.Transform("scale", {"factor": 2.0})])
    res = wb.measure(rec, m)
    assert np.array_equal(rec.sweep(0), before)
    rows = [r for r in res["rows"] if r["region"] == "resp" and r["pulse"] == 1]
    assert np.mean([r["peak"] for r in rows]) == pytest.approx(2 * sim.truth["amplitudes"][0], abs=0.6)
    # subtracting the average of the sweeps leaves the response's difference from it: near nothing
    avg = np.mean([rec.sweep(k) for k in range(rec.n_sweeps)], axis=0)
    m2 = _measurement(rec, transforms=[wb.Transform("subtract", {"label": "avg", "scale": 1.0})])
    res2 = wb.measure(rec, m2, aux={"avg": avg})
    rows2 = [r for r in res2["rows"] if r["region"] == "resp"]
    assert abs(np.mean([r["peak"] for r in rows2])) < 0.5
    # a baseline by region and by a sloping line
    m3 = _measurement(rec, transforms=[wb.Transform("baseline_region", {"region": "base"})])
    y3 = wb.transform(rec, 0, wb.raw(rec, 0), m3.transforms, m3.regions)
    on = float(rec.stimulus.times_s[0][0])
    i = int(on * rec.sample_rate_hz)
    assert abs(np.mean(y3[i - 400:i - 20])) < 0.2
    c = wb.combine(np.ones(10), np.arange(10.0), "-")
    assert c[3] == -2.0
    c2 = wb.combine(np.ones(10), np.ones(5), "+", fs_a=10.0, fs_b=5.0)
    assert c2.size == 10 and np.allclose(c2, 2.0)


def test_averaging_modes_exclusions_and_clicks():
    sim, rec = _light_recording()
    m = _measurement(rec)
    m.selection.mode = "average"
    res = wb.measure(rec, m)
    assert [r["trace"] for r in res["rows"]][0].startswith("average of 8")
    assert len(res["rows"]) == 3                                          # 2 pulses of resp + the fit region
    m.selection.mode = "typical"
    assert wb.measure(rec, m)["rows"][0]["trace"].startswith("typical")
    m.selection.mode = "each"
    m.exclusions = [wb.Exclusion(0, "noisy"), wb.Exclusion(3, "spike")]
    res = wb.measure(rec, m)
    rows = [r for r in res["rows"] if r["region"] == "resp" and r["pulse"] == 1]
    assert len(rows) == 6 and res["summary"]["resp"]["peak"]["n"] == 12
    on = float(rec.stimulus.times_s[1][0])
    m.overrides = [wb.Override(1, "onset", on + 0.010, "resp"), wb.Override(1, "peak", on + 0.020, "resp")]
    res = wb.measure(rec, m)
    row = next(r for r in res["rows"] if r["sweep"] == 1 and r["region"] == "resp" and r["pulse"] == 1)
    assert row["onset_latency"] == pytest.approx(10.0, abs=0.06)
    assert row["time_of_peak"] == pytest.approx(20.0, abs=0.06)
    assert "set by hand" in m.describe() and "left out" in m.describe()


def test_the_table_plots_a_measure_against_sweep_time_command_or_another_measure():
    sim = simulate.photostim_steps(seed=2)                                # a step family: the command differs
    rec = sim.recording
    rec.stimulus = find_stimulus(rec)
    m = _measurement(rec)
    res = wb.measure(rec, m)
    x, y, _ = wb.plot_series(res["rows"], "peak", "command", "resp")
    assert np.unique(x).size == 10 and np.corrcoef(x, y)[0, 1] < -0.9       # the PSP shrinks as Vm rises
    x2, y2, _ = wb.plot_series(res["rows"], "peak", "sweep", "resp")
    assert x2.min() == 1 and x2.max() == 10
    x3, y3, _ = wb.plot_series(res["rows"], "area", "peak", "resp")
    assert np.corrcoef(x3, y3)[0, 1] > 0.9


def test_a_measurement_is_kept_and_made_again_from_its_description(tmp_path):
    from clampwise.project import Project
    from clampwise.simulate.abf_file import write_abf

    sim, rec = _light_recording()
    path = _abf(write_abf, tmp_path / "light.abf", rec)
    m = _measurement(rec, fit_model="exp1", fit_region="decay")
    wb.stamp(m, "test")
    d = m.to_dict()
    again = wb.HandMeasurement.from_dict(d)
    assert again.to_dict() == d and again.made_at == m.made_at
    res = wb.measure(rec, again)
    assert res["fit"]["ok"]
    rows = wb.summary_rows(again, res)
    assert any(r["measure"].startswith("fit") for r in rows) and all(r["measured"].startswith("by hand") for r in rows)
    if path is not None:
        p = Project.open(tmp_path)
        p.add_recordings([path])
        hid = p.add_hand(path, again, rows)
        kept = p.hands(path)
        assert len(kept) == 1 and kept[0]["id"] == hid and kept[0]["summary"][0]["measure"] == "resp: Peak"
        assert wb.HandMeasurement.from_dict(kept[0]["measurement"]).describe() == again.describe()
        p.remove_hand(hid)
        assert not p.hands(path)
        tpl = wb.Template.from_measurement("light template", "sim_photostim*", again)
        p.save_template(tpl)
        back = p.templates()
        assert len(back) == 1 and back[0].matches(rec.protocol) and not back[0].matches("Simple_IV")
        m2 = back[0].measurement_for(rec)
        assert m2.selection.sweeps == list(range(rec.n_sweeps)) and m2.from_template == "light template"
        p.delete_template("light template")
        assert not p.templates()
        p.close()


def test_sweep_words():
    assert wb.parse_sweep_words("", 6) == [0, 1, 2, 3, 4, 5]
    assert wb.parse_sweep_words("every 2nd", 6) == [0, 2, 4]
    assert wb.parse_sweep_words("every 3rd from 2", 9) == [1, 4, 7]
    assert wb.parse_sweep_words("odd", 5) == [0, 2, 4] and wb.parse_sweep_words("even", 5) == [1, 3]
    assert wb.parse_sweep_words("last 2", 6) == [4, 5] and wb.parse_sweep_words("first 2", 6) == [0, 1]
    assert wb.parse_sweep_words("2-3, 6", 6) == [1, 2, 5]


def test_the_workbench_screen_measures_keeps_and_undoes(tmp_path, qapp=None):
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtCore, QtWidgets

    from clampwise.app.window import MainWindow
    from clampwise.simulate.abf_file import write_abf

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sim, rec = _light_recording()
    path = _abf(write_abf, tmp_path / "light.abf", rec)
    w = MainWindow(persist=False)
    w.open_project(tmp_path)
    w.select(path)
    w.go("workbench")
    s = w.screens["workbench"]
    app.processEvents()
    assert s._recording is not None and s._result is not None
    assert s.table.model().rowCount() >= 16
    # a region added, dragged and undone
    n = len(s._m.regions)
    s._add_region()
    assert len(s._m.regions) == n + 1
    w.undo.undo()
    assert len(s._m.regions) == n
    w.undo.redo()
    assert len(s._m.regions) == n + 1
    s.region_list.setCurrentRow(1)
    s.r_start.setValue(2.0)
    s.r_end.setValue(30.0)
    s._region_edited()
    assert s._m.regions[1].start_s == pytest.approx(0.002) and s._m.regions[1].end_s == pytest.approx(0.030)
    s.sweeps_box.setText("every 2nd")
    s._sweeps_typed()
    assert s._m.selection.sweeps == [0, 2, 4, 6]
    s.fit_model.setCurrentIndex(s.fit_model.findData("exp1"))
    assert s._m.fit_model == "exp1"
    s.name_box.setText("my light response")
    s.keep()
    assert [h["name"] for h in w.project.hands(path)] == ["my light response"]
    w.project.analyze(path)                                           # Results lists the analyses, then the hand tab
    w.go("results")
    app.processEvents()
    r = w.screens["results"]
    titles = [r.tabs.tabText(i) for i in range(r.tabs.count())]
    assert any(t.startswith("Measured by hand") for t in titles)
    w.close()
