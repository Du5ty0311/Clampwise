"""Connected or not (the lab's connectivity experiments, 2026-11-10): response windows against matched sham
windows, the light's own mark on the trace recognised and blanked, a direct photocurrent told from a synapse, a
mapping protocol with no record of its regions not averaged, and the cell's health sweep by sweep."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from clampwise import simulate
from clampwise.analysis import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))


def _run(sim, **shared):
    r = pipeline.analyze(sim.recording, pipeline.Settings(shared=shared))
    for key in ("evoked", "connectivity"):
        assert not r.results[key].error, r.results[key].error
    return r


def test_the_lab_s_photostim_pulses_on_am_stim_are_light():
    """AM_Stim is Clampex's name for a stimulator input; the lab's Polygon rig records the LED's command on it,
    and the protocol's name (Photostim, ChroME) says so."""
    r = _run(simulate.photostim_steps(seed=1, protocol="CC_-200mV_40pA_5ms_100%_Photostim2x_100ms between"))
    assert "light" in r.stimulus_summary and "electrical" not in r.stimulus_summary
    r = _run(simulate.photostim_steps(seed=1, protocol="ChroME_threshold_intensity_CC"))
    assert "light" in r.stimulus_summary


def test_a_connected_cell_and_an_unconnected_one_with_the_false_positive_rate_stated():
    r = _run(simulate.photostim_steps(seed=2))
    c = r.results["connectivity"].values
    assert c["connected"] == "yes" and c["p"] < 0.01
    assert c["success_rate"] == 1.0 and c["n_windows"] == 20
    assert c["false_positive_rate"] == 0.05 and c["chance_successes"] == pytest.approx(1.0)
    assert c["latency_median"] == pytest.approx(3.0, abs=0.4) and c["jitter"] < 0.2
    assert c["efficacy"] == pytest.approx(c["potency"])
    assert c["direct_photocurrent"] == "no"
    r = _run(simulate.photostim_steps(seed=2, success=0.0))
    c = r.results["connectivity"].values
    assert c["connected"] == "no" and c["p"] > 0.05
    assert c["successes"] <= 2 and np.isnan(c["potency"])


def test_failures_lower_the_efficacy_but_not_the_potency():
    sim = simulate.photostim_steps(seed=2, success=0.6, jitter_s=0.0005, levels_pa=[0] * 20, onsets_s=(0.3356,))
    r = _run(sim)
    c = r.results["connectivity"].values
    true_rate = np.mean([h[0] for h in sim.truth["hits"]])
    assert c["connected"] == "yes"
    assert c["success_rate"] == pytest.approx(true_rate, abs=0.06)
    amp = sim.truth["amplitudes"][0]
    assert c["potency"] == pytest.approx(amp, abs=0.4)
    assert c["efficacy"] == pytest.approx(amp * true_rate, abs=0.5)
    assert c["jitter"] == pytest.approx(0.5, abs=0.25)


def test_a_direct_photocurrent_is_told_from_a_synapse_and_is_no_connection():
    r = _run(simulate.photostim_steps(seed=2, direct=True))
    e, c = r.results["evoked"], r.results["connectivity"].values
    assert e.values["direct_photocurrent"] == "yes"
    f = next(x for x in e.flags if x.code == "direct photocurrent")
    assert "answers the light" in f.text and "reversal" in f.text          # its size follows Vm
    assert np.isnan(e.values["p1_latency_median"])                         # the 1.5 ms rule is kept
    assert c["connected"] == "direct photocurrent" and c["n_early"] == 20
    assert e.values["reversal"] == pytest.approx(-10.0, abs=1.5)
    r = _run(simulate.photostim_steps(seed=2))
    assert r.results["evoked"].values["direct_photocurrent"] == "no"


def test_the_light_s_own_mark_is_recognised_blanked_and_measured_after():
    sim = simulate.photostim_steps(seed=2, artifact=1.5, latency_s=0.008)
    r = _run(sim)
    e, c = r.results["evoked"], r.results["connectivity"].values
    assert e.values["light_artifact_blank_ms"] == pytest.approx(5.5)
    assert any(f.code == "light artifact" for f in e.flags)
    assert e.values["direct_photocurrent"] == "no"
    zero = sim.truth["levels_pa"].index(0)
    assert e.values["p1_amplitude"] == pytest.approx(sim.truth["amplitudes"][zero], abs=0.4)
    assert e.values["p1_latency_median"] == pytest.approx(8.0, abs=0.5)
    assert c["connected"] == "yes" and c["latency_median"] == pytest.approx(8.0, abs=0.5)
    # a response that begins inside the light: its onset is said to be hidden, not read at the blank's end
    r = _run(simulate.photostim_steps(seed=2, artifact=1.5, latency_s=0.003))
    e, c = r.results["evoked"], r.results["connectivity"].values
    assert any("inside the blank" in f.text for f in e.flags)
    assert np.isnan(c["latency_median"]) and c["n_in_blank"] > 0


def test_a_mapping_protocol_with_no_record_of_its_regions_is_not_averaged():
    sim = simulate.photostim_steps(seed=3, levels_pa=[0] * 11, protocol="ChroME_grid_map_CC", name="26n01008.abf")
    r = _run(sim)
    e, c = r.results["evoked"], r.results["connectivity"].values
    assert any(f.code == "region unknown" for f in e.flags)
    assert np.isnan(e.values["p1_amplitude"]) and "unknown" in e.values["headline_is"]
    assert e.values["region_known"] == "no"
    assert len(e.tables["sweeps"]) == 22                                   # each sweep's own response stays
    assert c["connected"] == "region unknown"
    # the same sweeps under a protocol that lights one region every sweep are averaged as before
    r = _run(simulate.photostim_steps(seed=3, levels_pa=[0] * 11, protocol="Photostim_one_region"))
    assert r.results["evoked"].values["p1_amplitude"] == pytest.approx(4.0, abs=0.4)


def test_the_pattern_comes_from_the_file_s_name_when_no_log_says():
    from clampwise.analysis.connectivity import pattern_from_name

    assert pattern_from_name("Cell_Only_PPulse_Good_Response") == "Cell only"
    assert pattern_from_name("Full_Field_cell_response") == "Full field"
    assert pattern_from_name("ROI2_cell_response") == "ROI2"
    assert pattern_from_name("26n01008") == ""
    r = _run(simulate.photostim_steps(seed=2, name="ROI1_cell_response.abf"))
    assert r.results["connectivity"].values["pattern"] == "ROI1"


def test_the_pattern_comes_from_the_typed_condition_before_the_file_s_name():
    from clampwise.core.recording import Identity

    sim = simulate.photostim_steps(seed=2, name="ROI1_cell_response.abf")
    sim.recording.identity = Identity(animal="A1", cell="cell1", condition="ROI2 dorsal")
    c = _run(sim).results["connectivity"].values
    assert c["pattern"] == "ROI2 dorsal" and "typed" in c["pattern_source"]


def test_a_cell_s_files_are_put_side_by_side():
    from clampwise.analysis.cells import Cell, Member, analyze_cell

    files = [("Cell_Only.abf", dict(direct=True)), ("ROI1.abf", dict(success=0.0)), ("ROI2.abf", dict()),
             ("Full_Field.abf", dict(success=0.8))]
    members = []
    for name, kw in files:
        sim = simulate.photostim_steps(seed=4, name=name, **kw)
        r = pipeline.analyze(sim.recording)
        members.append(Member(name, r.recording, r.results, r.recording_type))
    cell = analyze_cell(Cell.build("a cell", members))
    cc = cell.results["connectivity_cell"]
    rows = {r["file"]: r for r in cc.tables["files"]}
    assert rows["Cell_Only.abf"]["connected"] == "direct photocurrent"
    assert rows["ROI1.abf"]["connected"] == "no" and rows["ROI2.abf"]["connected"] == "yes"
    assert rows["Full_Field.abf"]["connected"] == "yes"
    assert cc.values["n_connected"] == 2 and "ROI2" in cc.values["patterns_connected"]
    assert any(f.code == "direct photocurrent" for f in cc.flags)
    # the patterns against each other (not two p values side by side): Fisher's exact on successes over windows
    from scipy import stats as sps

    pairs = {(r["pattern_a"], r["pattern_b"]): r for r in cc.tables["comparisons"]}
    assert set(pairs) == {("ROI1", "ROI2"), ("ROI1", "Full field"), ("ROI2", "Full field")}      # the direct one left out
    one = pairs[("ROI1", "ROI2")]
    want = sps.fisher_exact([[one["successes_a"], one["windows_a"] - one["successes_a"]],
                             [one["successes_b"], one["windows_b"] - one["successes_b"]]])[1]
    assert one["p"] == pytest.approx(want) and one["p_holm"] >= one["p"] and one["different"]
    assert all(r["p_holm"] >= r["p"] and r["different"] == (r["p_holm"] < 0.05) for r in pairs.values())
    assert "ROI1 against ROI2" in cc.values["comparison"]
    lo, hi = rows["ROI2.abf"]["success_rate_low"], rows["ROI2.abf"]["success_rate_high"]
    assert 0 <= lo < rows["ROI2.abf"]["success_rate"] <= hi <= 1 and rows["ROI2.abf"]["smallest_detectable"] > 0


def test_holm_s_correction():
    from clampwise.analysis.connectivity import holm

    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5] and holm([]) == []


def test_health_sweep_by_sweep_calls_a_depolarized_cell_unhealthy_in_words():
    sim = simulate.photostim_steps(seed=2, v_rest=-26.0, e_rev=-5.0)
    r = _run(sim)
    h = r.results["sweep_health"]
    assert h.values["healthy"] == "no" and h.values["unhealthy_sweeps"] == 10
    assert "depolarized" in h.values["words"] and any(f.code == "unhealthy" for f in h.flags)
    assert all("depolarized" in row["words"] for row in h.tables["sweeps"])
    assert h.values["rin_median"] == pytest.approx(100.0, abs=3.0)      # from each sweep's own step
    sim = simulate.photostim_steps(seed=2)
    h = _run(sim).results["sweep_health"]
    assert h.values["healthy"] == "yes" and h.values["unhealthy_sweeps"] == 0
    assert all(row["healthy"] for row in h.tables["sweeps"])


def test_the_input_resistance_per_sweep_comes_from_the_test_pulse_when_the_protocol_has_one():
    from clampwise.core.recording import ArraySource, Epoch

    sim = simulate.photostim_steps(seed=2, levels_pa=[0] * 8, onsets_s=(0.3106,))
    rec = sim.recording
    fs = rec.sample_rate_hz
    sig, cmd, light = rec.read(0).copy(), rec.read(1).copy(), rec.read(2)
    t = rec.time_s
    a, b = int(0.766 * fs), int(0.866 * fs)
    rm = 100.0
    for k in range(rec.n_sweeps):
        cmd[k, a:b] = -20.0
        dv = -20.0 * rm * 1e-3
        sig[k, a:b] += dv * (1 - np.exp(-(t[a:b] - t[a]) / 0.01))
        sig[k, b:] += dv * np.exp(-(t[b:] - t[b]) / 0.01)
    rec.samples = ArraySource([sig, cmd, light])
    rec.epochs = [[Epoch("step", 0.0, 0.766, 0.0), Epoch("step", 0.766, 0.866, -20.0), Epoch("step", 0.866, 1.0, 0.0)]
                  for _ in range(rec.n_sweeps)]
    h = pipeline.analyze(rec).results["sweep_health"]
    assert "test pulse" in h.values["rin_from"]
    assert h.values["rin_median"] == pytest.approx(rm, abs=3.0)
    assert all(abs(row["rin_mohm"] - rm) < 5.0 for row in h.tables["sweeps"])


def test_light_inside_a_step_family_stays_out_of_the_intrinsic_measures():
    """The lab's photostim protocols light the cell inside every step: a spike the light evokes is not the step's
    (no rheobase from it), and a light response is not the step's peak or steady state."""
    from clampwise.core.recording import ArraySource

    sim = simulate.photostim_steps(seed=2, g_mv_per_mv=0.0)
    rec = sim.recording
    fs, t = rec.sample_rate_hz, rec.time_s
    sig = rec.read(0).copy()
    for k, lv in enumerate(sim.truth["levels_pa"]):
        if lv < 40:
            continue                                         # a light-evoked spike on the depolarizing steps only
        peak = int((0.3356 + 0.006) * fs)
        st = (t[peak - int(0.001 * fs):peak + int(0.004 * fs)] - t[peak]) * 1e3
        sig[k, peak - int(0.001 * fs):peak + int(0.004 * fs)] += 95.0 * np.exp(-(st / 0.4) ** 2)
    rec.samples = ArraySource([sig, rec.read(1), rec.read(2)])
    r = pipeline.analyze(rec)
    it = r.results["intrinsic"]
    assert r.recording_type == "cc_evoked"
    assert it.values["stimulus_evoked_spikes"] == 4
    assert np.isnan(it.values["rheobase"]) and all(row["spikes"] == 0 for row in it.tables["steps"])
    assert it.values["r_input"] == pytest.approx(100.0, abs=3.0)
    assert any("inside the step" in n for n in it.notes)
    # a large light response on the hyperpolarizing steps must not be read as the step's peak (the sag)
    sim = simulate.photostim_steps(seed=2, g_mv_per_mv=0.3, e_rev=-120.0)   # a 12-18 mV hyperpolarizing response
    it = pipeline.analyze(sim.recording).results["intrinsic"]
    assert it.values["sag_ratio_ipfx"] == pytest.approx(0.0, abs=0.03)
    assert it.values["sag_ratio"] == pytest.approx(1.0, abs=0.03)        # Ahmed 2021's: last 100 ms over first 100 ms


def test_the_known_answers_pass():
    import known_answers

    rows = known_answers.run(["connectivity"])
    bad = [r for r in rows if not r.ok]
    assert not bad, "\n".join(f"{r.case} {r.measure}: true {r.truth} measured {r.measured}" for r in bad)
