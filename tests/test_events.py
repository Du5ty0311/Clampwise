"""B4: spontaneous and miniature events, beyond the known answers
(tools/audit/known_answers.py, section "events"): what is left out, sweeps,
DSI steps, event-driven files, the review, the experiment template, the
stretch, the slow-event flag, the timing test and the spike finder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from clampwise import simulate
from clampwise.analysis import events as E, pipeline
from clampwise.analysis.tools import spikes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))
import known_answers  # noqa: E402

TEN = {"events": {"template_decay_ms": 10.0}}


def run(sim, by_analysis=None, **shared):
    return pipeline.analyze(sim.recording, pipeline.Settings(by_analysis=by_analysis or dict(TEN), shared=shared))


def accepted(result):
    return [r for r in result.results["events"].tables["events"] if r["accepted"]]


def test_the_known_answers_pass():
    rows = known_answers.run(["events"])
    bad = [r for r in rows if not r.ok]
    assert not bad, "\n".join(f"{r.case}: {r.measure} true {r.truth:.4g} measured {r.measured:.4g}" for r in bad)


def test_every_number_says_how_it_was_measured():
    r = run(simulate.spontaneous(duration_s=20.0))
    e = r.results["events"]
    assert r.recording_type == "vc_spontaneous"
    assert e.choices["direction"].startswith("down") and "template" in e.choices
    assert any("deconvolution" in m for m in e.methods) and any("averaged event" in m for m in e.methods)
    assert e.values["template"].startswith("deconvolution, analytic, rise 0.5 / decay 10 ms")
    assert any("not independent" in n for n in e.notes)
    assert {"events", "timecourse", "excluded"} <= set(e.tables)


def test_current_clamp_leaves_action_potentials_out():
    sim = simulate.spontaneous(duration_s=20.0, voltage_clamp=False, baseline=-68.0, amp=1.5, noise=0.15,
                               rise_s=0.0015, decay_s=0.015, rate_hz=3.0, spikes_at=(3.0, 7.5, 12.0))
    r = run(sim, {"events": {"template_rise_ms": 1.5, "template_decay_ms": 15.0}})
    e = r.results["events"]
    assert r.recording_type == "cc_spontaneous"
    left = [x for x in e.tables["excluded"] if x["why"] == "action potential"]
    assert len(left) == 3
    for x in accepted(r):
        assert not any(a <= x["onset_s"] <= b for a, b in [(z["start_s"], z["end_s"]) for z in left])
    assert all(np.isfinite(x["amplitude"]) and x["amplitude"] < 10 for x in accepted(r))   # no spike read as an event
    assert e.values["analyzed_s"] == pytest.approx(20.0 - sum(z["end_s"] - z["start_s"] for z in left), abs=0.01)
    assert not any(f.code == "mostly firing" for f in e.flags)


def test_a_cell_firing_throughout_is_flagged():
    sim = simulate.spontaneous(duration_s=10.0, voltage_clamp=False, baseline=-60.0, amp=1.0, noise=0.15,
                               spikes_at=tuple(np.arange(0.1, 9.9, 0.03)))
    e = run(sim).results["events"]
    assert any(f.code == "mostly firing" for f in e.flags)


def test_intervals_never_cross_sweeps_or_left_out_stretches():
    sim = simulate.spontaneous(n_sweeps=4, sweep_s=10.0, rate_hz=4.0)
    r = run(sim)
    ev = accepted(r)
    firsts = {}
    for x in ev:
        firsts.setdefault(x["sweep"], x)
    assert all(not np.isfinite(x["iei_ms"]) for x in firsts.values())
    assert len(firsts) == 4
    assert r.results["events"].values["analyzed_s"] == pytest.approx(40.0, abs=0.01)


def test_dsi_steps_and_their_settling_are_left_out():
    sim = simulate.spontaneous(n_sweeps=3, sweep_s=20.0, rate_hz=6.0, step=(8.0, 2.0, 0.0), holding_mv=-70.0)
    r = run(sim)
    e = r.results["events"]
    assert r.recording_type == "dsi"
    steps = [x for x in e.tables["excluded"] if x["why"].startswith("step to 0 mV")]
    assert len(steps) == 3
    for x in steps:
        assert x["start_s"] == pytest.approx(7.995) and x["end_s"] >= 10.2
    recall, false = known_answers.match_events([x["onset_s"] + 100 * x["sweep"] for x in accepted(r)],
                                               sim.truth["times_s"] + 100 * sim.truth["sweeps"])
    assert recall > 0.9 and false <= 3


def test_an_evoked_recording_gets_events_only_with_long_free_sweeps():
    short = simulate.evoked(n_sweeps=5)
    assert "events" not in pipeline.analyze(short.recording).results
    long = simulate.evoked(n_sweeps=3, sweep_s=5.0, onsets_s=(0.5, 0.55))
    r = pipeline.analyze(long.recording)
    assert "events" in r.results
    for x in r.results["events"].tables["events"]:
        assert not (0.498 <= x["onset_s"] <= 0.652)


def test_outward_events_when_held_positive():
    r = run(simulate.spontaneous(duration_s=20.0, amp=25.0, holding_mv=10.0, baseline=300.0))
    e = r.results["events"]
    assert e.values["direction"] == "up" and e.values["amplitude_mean"] > 20
    assert "held at 10 mV" in e.choices["direction"]


def test_event_driven_files():
    sim = simulate.event_driven(times_recorded=True)
    r = pipeline.analyze(sim.recording)
    e = r.results["events"]
    assert r.recording_type == "vc_spontaneous" and "event-driven" in r.type_reason
    assert e.values["n_events"] == pytest.approx(120, abs=2)
    assert e.values["amplitude_mean"] == pytest.approx(float(np.mean(sim.truth["amplitudes"])), rel=0.05)
    assert e.values["decay_tau"] == pytest.approx(4.0, rel=0.1)
    assert np.isfinite(e.values["frequency"])
    back = pipeline.analyze(simulate.event_driven(times_recorded=False).recording).results["events"]
    assert not np.isfinite(back.values["frequency"]) and "back to back" in " ".join(back.notes)
    assert back.values["timing"].startswith("not in the file")


def test_the_review_is_kept_in_the_project_and_reapplied(tmp_path):
    from clampwise.io import abf  # noqa: F401  (the project needs a file on disk)
    from clampwise.project import Project

    sim = simulate.spontaneous(duration_s=15.0)
    r = run(sim)
    first = accepted(r)[0]
    edits = E.format_edits([(first["sweep"], first["onset_s"] + 0.0003)], [(0, 7.12345)])
    again = run(sim, {"events": {"template_decay_ms": 10.0, "edits": edits}})
    e = again.results["events"]
    assert e.values["n_events"] == r.results["events"].values["n_events"]       # one out, one in
    assert e.values["n_rejected"] == 1 and e.values["n_added"] == 1
    rows = {round(x["onset_s"], 4): x for x in e.tables["events"]}
    assert not rows[round(first["onset_s"], 4)]["accepted"]
    assert rows[7.1235]["manual"] and rows[7.1235]["accepted"]
    # the store keeps the review as decisions and hands it to the analysis
    project = Project.open(tmp_path)
    rec = tmp_path / "cell.abf"
    rec.write_bytes(b"0")
    project.add_recordings([rec], hash_files=False)
    project.review_event(rec, 0, 2.5, "reject")
    project.review_event(rec, 0, 3.25, "add")
    s = project.pipeline_settings(rec)
    rej, add = E.parse_edits(s.by_analysis["events"]["edits"])
    assert rej == [(0, 2.5)] and add == [(0, 3.25)]
    project.review_event(rec, 0, 2.5004, "restore")
    assert E.parse_edits(project.pipeline_settings(rec).by_analysis["events"]["edits"])[0] == []
    assert "edits" not in project.pipeline_settings().by_analysis.get("events", {})


def test_one_template_for_the_experiment_and_the_mismatch_warning():
    results = [run(simulate.spontaneous(duration_s=30.0, seed=k, decay_s=0.012)).results["events"] for k in range(3)]
    tpl = E.experiment_template(results)
    assert tpl.ok and tpl.n_recordings == 3 and tpl.n_events > 200
    assert tpl.decay_tau_ms == pytest.approx(12.0, rel=0.08)
    s = tpl.settings()
    assert s["template_decay_ms"] == pytest.approx(tpl.decay_tau_ms, abs=0.01) and "averaged event" in s["template_origin"]
    assert E.template_mismatch([(f"r{k}", r) for k, r in enumerate(results)]) == ""
    other = run(simulate.spontaneous(duration_s=10.0), {"events": {}}).results["events"]
    warning = E.template_mismatch([("a", results[0]), ("b", other)])
    assert "different templates" in warning and "confound" in warning
    assert not E.experiment_template([]).ok


def test_a_chosen_stretch():
    sim = simulate.spontaneous(duration_s=40.0)
    r = run(sim, stretch="10-30")
    e = r.results["events"]
    assert e.values["analyzed_s"] == pytest.approx(20.0, abs=0.01)
    assert all(10.0 <= x["onset_s"] <= 30.0 for x in e.tables["events"])
    assert "stretch" in e.choices
    with pytest.raises(ValueError):
        E.parse_stretch("30-10")


def test_slow_events_are_flagged_and_the_dsi_preset_template_fits_them():
    from clampwise.project.presets import BUILTIN

    sim = simulate.spontaneous(duration_s=30.0, decay_s=0.025, rate_hz=3.0)
    flagged = run(sim, {"events": {}}).results["events"]
    assert any(f.code == "slow events" for f in flagged.flags)
    sim10 = simulate.spontaneous(duration_s=30.0, decay_s=0.010, rate_hz=3.0)
    preset = BUILTIN["CA3 sIPSC DSI (carbachol)"].settings
    assert preset["events"]["template_decay_ms"] == 10.0
    clean = run(sim10, {"events": dict(preset["events"])}).results["events"]
    assert not any(f.code == "slow events" for f in clean.flags)
    assert "averaged sIPSCs (CA3 sIPSC DSI preset)" in clean.values["template"]


def test_timing_separates_a_rate_change_from_bursts():
    steps = run(simulate.spontaneous(duration_s=60.0, seed=4, rate_steps=[(0, 2.0), (20, 12.0), (40, 3.0)]))
    v = steps.results["events"].values
    assert v["iei_cv"] > 1.15                       # the raw CV alone would say bursty
    assert v["timing"].startswith("random")
    bursty = run(simulate.spontaneous(duration_s=60.0, seed=4, process="bursty")).results["events"].values
    assert bursty["timing"] == "bursty" and bursty["iei_lv"] > 1.2
    few = run(simulate.spontaneous(duration_s=5.0, rate_hz=2.0)).results["events"].values
    assert few["timing"].startswith("too few")


def test_events_over_time_and_charge():
    sim = simulate.spontaneous(duration_s=40.0, rate_steps=[(0, 2.0), (20, 8.0)])
    e = run(sim).results["events"]
    tc = e.tables["timecourse"]
    assert len(tc) == 4
    assert tc[0]["frequency_hz"] == pytest.approx(2.0, abs=1.0) and tc[3]["frequency_hz"] == pytest.approx(8.0, abs=2.0)
    # charge per second: the events' total charge over the time, within 10%
    kt = np.arange(0, 0.08, 1 / 20_000)
    shape = (1 - np.exp(-kt / 0.0005)) * np.exp(-kt / 0.010)
    q_unit = float(np.trapezoid(shape / shape.max(), kt))
    truth = float(np.sum(sim.truth["amplitudes"]) * q_unit / 40.0)
    assert e.values["charge_rate"] == pytest.approx(truth, rel=0.1)
    assert tc[3]["charge_rate"] < tc[0]["charge_rate"] < 0


def test_the_spike_finder_counts_what_the_simulator_put_in():
    sim = simulate.steps(n_sweeps=11, rheobase_pa=60.0)
    counts = [len(spikes.find(sim.recording.sweep(k), sim.recording.sample_rate_hz)) for k in range(11)]
    assert counts == sim.truth["spike_counts"]
    assert spikes.find(np.full(1000, -70.0), 20_000.0) == []
