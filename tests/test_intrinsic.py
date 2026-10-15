"""B6: intrinsic properties and the spike detector, beyond the known answers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from clampwise import simulate
from clampwise.analysis import intrinsic, pipeline
from clampwise.analysis.tools import spikes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))
import known_answers  # noqa: E402

FS = 50_000.0


def trace(seconds: float = 0.5, v: float = -65.0) -> np.ndarray:
    return np.full(int(seconds * FS), v)


def add_spike(y: np.ndarray, at_s: float, height: float = 100.0, width_ms: float = 0.5) -> None:
    """A Gaussian spike with an AHP after it, peaking at ``at_s``."""
    i = int(at_s * FS)
    t = (np.arange(-int(0.002 * FS), int(0.006 * FS)) / FS) * 1e3
    y[i + int(t[0] * FS / 1e3):i + int(t[0] * FS / 1e3) + t.size] += (height * np.exp(-(t / width_ms) ** 2)
                                                                      - 8.0 * np.exp(-((t - 2.0) / 1.5) ** 2))


def add_slow(y: np.ndarray, at_s: float, height: float, rise_ms: float, decay_ms: float) -> None:
    """A straight rise of ``height`` over ``rise_ms``, then an exponential fall."""
    i, n = int(at_s * FS), max(2, int(rise_ms * FS / 1e3))
    y[i:i + n] += np.linspace(0, height, n)
    tail = np.arange(y.size - i - n) / FS * 1e3
    y[i + n:] += height * np.exp(-tail / decay_ms)


def test_the_known_answers_pass():
    bad = [r for r in known_answers.run(["intrinsic"]) if not r.ok]
    assert not bad, "\n".join(f"{r.case}: {r.measure} true {r.truth:.4g} measured {r.measured:.4g}" for r in bad)


def test_spikes_are_found_with_both_thresholds_and_ipfx_features():
    y = trace()
    for x in (0.1, 0.2, 0.3):
        add_spike(y, x)
    sp = spikes.find(y, FS)
    assert [round(x.peak_index / FS, 3) for x in sp] == [0.1, 0.2, 0.3]
    for x in sp:
        assert x.full and not x.clipped
        assert x.ipfx_threshold_index <= x.threshold_index < x.peak_index     # 5% of the upstroke comes first
        assert x.ipfx_threshold_mv <= x.threshold_mv
        assert 0.3 < x.half_width_ms < 1.5 and np.isfinite(x.fast_trough_mv)
        assert x.upstroke > 100 and x.downstroke < -50


def test_a_window_takes_onsets_inside_it_and_peaks_before_its_end():
    y = trace()
    for x in (0.1, 0.2, 0.3):
        add_spike(y, x)
    assert len(spikes.find(y, FS, window=(int(0.15 * FS), int(0.35 * FS)))) == 2
    # an onset before the window's start: not that spike
    assert len(spikes.find(y, FS, window=(int(0.0999 * FS), int(0.35 * FS)))) == 2
    # the window's end falls between a spike's onset and its peak: IPFX counts it, clipped
    sp = spikes.find(y, FS, window=(int(0.15 * FS), int(0.2999 * FS)))
    assert len(sp) == 2 and sp[-1].clipped and not sp[0].clipped


def test_a_peak_that_never_repolarizes_before_the_next_onset_is_one_spike():
    # a spike with a second rise on its shoulder: dV/dt crosses 20 mV/ms twice, never below zero between
    y = trace()
    add_spike(y, 0.1)
    i = int(0.1 * FS)
    ramp = np.linspace(0, 30, int(0.0004 * FS))
    y[i + 5:i + 5 + ramp.size] += ramp
    y[i + 5 + ramp.size:i + int(0.006 * FS)] += 30
    assert len(spikes.find(y, FS)) == 1


def test_the_last_spike_cut_by_the_window_is_clipped():
    y = trace()
    add_spike(y, 0.1)
    i = int(0.2 * FS)
    y[i:] += np.minimum(np.arange(y.size - i) / FS * 1e5, 90.0)      # a rise that never comes back down
    sp = spikes.find(y, FS, window=(0, int(0.25 * FS)))
    assert sp[-1].clipped and not np.isfinite(sp[-1].half_width_ms)
    assert not sp[0].clipped


def test_the_labs_width_rule_drops_a_slow_bump_but_keeps_a_broad_full_spike():
    y = trace(v=-30.0)
    add_slow(y, 0.1, 4.0, 0.1, 10.0)          # a 4 mV bump, 40 mV/ms up, 7 ms at half height: no spike
    add_slow(y, 0.3, 60.0, 1.0, 10.0)         # full-sized and broadened to 7.4 ms: kept
    sp = spikes.find(y, FS)
    assert [round(x.peak_index / FS, 2) for x in sp] == [0.3]
    saved = spikes.MAX_HALF_WIDTH_S, spikes.MAX_HALF_WIDTH_SMALL_S
    try:
        spikes.MAX_HALF_WIDTH_S = spikes.MAX_HALF_WIDTH_SMALL_S = 1e9
        assert len(spikes.find(y, FS)) == 2                # IPFX alone counts the bump
    finally:
        spikes.MAX_HALF_WIDTH_S, spikes.MAX_HALF_WIDTH_SMALL_S = saved


def test_a_long_trace_in_pieces_gives_what_one_pass_gives(monkeypatch):
    y = trace(2.0) + np.random.default_rng(1).normal(0, 0.2, int(2.0 * FS))
    for x in np.arange(0.05, 1.95, 0.0731):
        add_spike(y, x)
    whole = spikes.find(y, FS)
    monkeypatch.setattr(spikes, "CHUNK", 20_000)
    pieces = spikes.find(y, FS)
    assert [x.peak_index for x in pieces] == [x.peak_index for x in whole]
    assert [x.threshold_index for x in pieces] == [x.threshold_index for x in whole]


def test_spike_counts_match_ipfx_itself():
    fx = pytest.importorskip("ipfx.feature_extractor")
    for pattern in ("tonic", "bursting", "adapting"):
        sim = simulate.firing_steps(pattern=pattern, noise=0.5, seed=3)
        rec, fs = sim.recording, sim.recording.sample_rate_hz
        a, b = 0.2, 1.0
        t = np.arange(rec.n_samples) / fs
        for k in range(rec.n_sweeps):
            v = rec.sweep(k).astype(float)
            df = fx.SpikeFeatureExtractor(start=a, end=b, filter=None).process(t, v, None)
            ours = spikes.find(v, fs, window=(int(round(a * fs)), int(round(b * fs))))
            assert len(ours) == len(df), (pattern, k)
            if len(df):
                assert np.allclose([x.ipfx_threshold_mv for x in ours], df["threshold_v"].to_numpy())
                assert np.allclose([x.peak_mv for x in ours], df["peak_v"].to_numpy())


def test_intrinsic_reports_its_choices_and_names_no_pattern():
    # firing NeuroExpress's way: the measures and the step they are read on, no label
    r = pipeline.analyze(simulate.firing_steps(pattern="adapting").recording).results["intrinsic"]
    assert "firing_pattern" not in r.values and r.values["isi_ratio"] > 2.0
    assert r.values["isi_step_pa"] in [x["current_pa"] for x in r.tables["steps"]]
    for key in ("input resistance", "membrane tau", "first spike", "f-I gain", "interval measures"):
        assert key in r.choices
    assert r.values["ap_threshold"] <= r.values["ap_threshold_20"]      # IPFX's comes no later on the upstroke
    rows = r.tables["spikes"]
    assert rows and {"threshold_mv", "threshold_20_mv", "half_width_ms", "half_width_threshold_ms", "clipped"} <= set(rows[0])


def test_the_input_resistance_choice_says_its_numbers():
    r = pipeline.analyze(simulate.steps().recording).results["intrinsic"]
    text = r.choices["input resistance"]
    assert "read at 0 pA" in text                                      # the resistance at rest was taken
    assert "{" not in text and f"{r.values['r_input_slope']:.1f} MOhm" in text


def test_rheobase_needs_a_full_sized_spike():
    sim = simulate.steps(n_sweeps=11, noise=0.2)
    rec = sim.recording
    # an attenuated spike (35 mV, its peak under -20 mV) on the step before rheobase: counted, never the rheobase
    v = rec.read(0)                     # a simulation's samples, held in memory: edited in place
    k = sim.truth["levels_pa"].index(40.0)
    i = int(0.4 * rec.sample_rate_hz)
    tt = (np.arange(-40, 160) / rec.sample_rate_hz) * 1e3
    v[k, i - 40:i + 160] += 35.0 * np.exp(-(tt / 0.4) ** 2)
    r = pipeline.analyze(rec).results["intrinsic"]
    assert r.values["rheobase"] == 60.0
    assert r.values["attenuated_spikes"] >= 1


def test_no_step_in_the_command_is_said_so():
    sim = simulate.steps(n_sweeps=3)
    rec = sim.recording
    rec.read(1)[:] = 0.0                # the command, flat
    assert intrinsic.find_step(rec, 0.1, 0.1) is None


def test_a_staircase_command_is_not_read_as_a_step_family():
    sim = simulate.steps(n_sweeps=3)
    rec = sim.recording
    c = rec.read(1)                     # the command, held in memory: edited in place
    fs = rec.sample_rate_hz
    c[:] = 0.0
    for j, level in enumerate((-50, 0, 50, 100, 150, 200, 250)):
        c[:, int((0.1 + 0.1 * j) * fs):int((0.2 + 0.1 * j) * fs)] = level
    why: list[str] = []
    assert intrinsic.find_step(rec, 0.1, 0.1, why) is None
    assert "staircase" in why[0]
    r = pipeline.analyze(rec, pipeline.Settings(type_override="cc_steps")).results["intrinsic"]
    assert "staircase" in r.notes[0] and not np.isfinite(r.values.get("rheobase", np.nan))


# ---- NeuroExpress's measures (iv_curves)

def test_resistance_at_rest_and_its_slope_on_a_rectifying_membrane():
    sim = simulate.steps(rm_mohm=150.0, r_slope_mohm_per_na=300.0, noise=0.2, n_sweeps=21, first_pa=-200.0,
                         rheobase_pa=1e9)
    r = pipeline.analyze(sim.recording).results["intrinsic"]
    assert abs(r.values["r_input"] - 150.0) < 3.0                    # the headline is the resistance at rest
    assert abs(r.values["r_steady_slope0"] - 300.0) < 30.0
    assert r.values["r_input_slope"] < 150.0                          # the slope over -100 to 0 pA reads lower here
    assert "input resistance" in r.choices and "Input resistance at rest" in r.choices["input resistance"]
    said = [e["statement"] for e in r.tables["evaluation"]]
    assert any(x.startswith("Inward rectification") for x in said)
    row = r.tables["steps"][0]
    assert {"r_peak", "r_steady", "sag_mv", "rebound_mv", "tau_on_ms", "tau_off_ms"} <= set(row)


def test_rectification_is_not_read_from_too_few_steps_beyond_the_midpoint():
    sim = simulate.steps(rm_mohm=150.0, r_slope_mohm_per_na=300.0, noise=0.2, n_sweeps=11, first_pa=-100.0,
                         rheobase_pa=1e9)
    r = pipeline.analyze(sim.recording).results["intrinsic"]
    assert not any(e["statement"].startswith("Inward") for e in r.tables["evaluation"])


def test_a_rebound_spike_is_not_a_rebound():
    from clampwise.analysis import iv_curves

    y = trace(0.3)
    add_spike(y, 0.15)
    clean = iv_curves.without_spikes(y, FS, 0, y.size - 1)
    assert clean.max() < -60.0 and abs(clean[: int(0.1 * FS)] - y[: int(0.1 * FS)]).max() == 0


def test_the_resistance_disagreement_rule_is_neuroexpresss():
    from clampwise.analysis import iv_curves

    out = iv_curves.Curves(values={"r_peak0": 140.0, "r_vi_peak0": 100.0, "r_steady0": 100.0})
    s = {s.key: s.default for s in intrinsic.SETTINGS}
    cur = np.array([-100.0, -50.0, 50.0])
    z = np.zeros(3)
    iv_curves._evaluate(out, cur, z.astype(int), np.full(3, 100.0), np.full(3, 1.0), z, np.full(3, -70.0), [{}] * 3,
                        [[]] * 3, 0, 1, FS, s)
    assert any(a.startswith("The resistance measures differ") and "1.40 times" in b for a, b, _ in out.statements)
    out.values["r_peak0"], out.statements = 125.0, []
    iv_curves._evaluate(out, cur, z.astype(int), np.full(3, 100.0), np.full(3, 1.0), z, np.full(3, -70.0), [{}] * 3,
                        [[]] * 3, 0, 1, FS, s)
    assert not any(a.startswith("The resistance measures differ") for a, _, _ in out.statements)


def test_the_holding_current_is_read_from_the_recorded_current_when_there_is_one():
    from labdata import DEMOS

    p = DEMOS / "Test_IV_1.abf"
    if not p.is_file():
        pytest.skip("NeuroExpress's test files are not on this machine")
    from clampwise.io import load

    r = pipeline.analyze(load(p), pipeline.Settings(only={"intrinsic"})).results["intrinsic"]
    assert abs(r.values["holding_current"] - 5.70) < 0.05            # NeuroExpress: Irest 5.70 pA
    assert abs(r.values["r_steady0"] - 117.36) < 2.0                  # NeuroExpress: Rend0 117.36 MOhm
    assert abs(r.values["first_threshold_5_mv"] + 37.03) < 1.0        # NeuroExpress: SpTh -37.03 mV
