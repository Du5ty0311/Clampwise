"""B5: DSI from evoked responses (per recording) and from spontaneous events
(per cell, over its files on one clock), beyond the known answers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from clampwise import simulate
from clampwise.analysis import dsi, dsi_events, pipeline, stimulus
from clampwise.analysis.cells import Cell, Member, analyze_cell, cell_key
from clampwise.core import Identity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))
import known_answers  # noqa: E402
from labdata import find  # noqa: E402

TEN = pipeline.Settings(by_analysis={"events": {"template_decay_ms": 10.0}})


def cell_of(recs) -> Cell:
    return Cell.build("sim", [Member(r.source.name, r, pipeline.analyze(r, TEN).results) for r in recs])


def test_the_known_answers_pass():
    bad = [r for r in known_answers.run(["dsi"]) if not r.ok]
    assert not bad, "\n".join(f"{r.case}: {r.measure} true {r.truth:.4g} measured {r.measured:.4g}" for r in bad)


def test_the_threshold_rules():
    trials = [dsi.Trial(0.5, 5.0, np.array([0.95, 0.97, 0.94]), np.array([1.0, 0.98, 1.03, 0.99, 1.0])),
              dsi.Trial(2.0, 40.0, np.array([0.6, 0.62, 0.58]), np.array([1.0, 0.98, 1.03, 0.99, 1.0]))]
    tests = dsi.group_tests(trials, rule="significance", alpha=0.05, min_suppression_pct=20.0)
    assert [t.significant for t in tests] == [False, True]                 # 5%: significant but under the floor
    assert tests[0].p < 0.05
    assert list(dsi.rising([10, 30, 20, 50])) == [10, 25, 25, 50]
    interp, longest = dsi.curve_threshold([dsi.GroupTest(1.0, 10, 0.1, 1, 5, 3, False),
                                           dsi.GroupTest(4.0, 30, 0.01, 1, 5, 3, True)], 20.0)
    assert interp == pytest.approx(2.0) and longest == 30            # halfway on log duration
    assert dsi.recovery_time(np.array([1.0, 3, 5, 7, 9]), np.array([0.5, 0.7, 0.8, 0.95, 1.0]), 0.9, 1) == \
        pytest.approx(5 + 2 * (0.1 / 0.15))


def test_evoked_dsi_needs_a_step_and_a_stimulus():
    assert "dsi" not in pipeline.analyze(simulate.evoked(n_sweeps=3).recording).results
    one = pipeline.analyze(simulate.dsi_evoked(durations=(2.0,), trials=2).recording).results["dsi"]
    assert np.isnan(one.values["threshold"]) and any("no threshold" in n for n in one.notes)
    assert one.values["suppression_pct"] > 20 and one.choices["response direction"].startswith("inward")


def test_a_step_file_alone_gives_no_events():
    rec = simulate.steps_file(2.0)
    r = pipeline.analyze(rec, TEN)
    assert r.recording_type == "dsi"
    e = r.results["events"]
    assert e.values["n_events"] == 0 and e.values["analyzed_s"] == 0
    assert any("shorter than 1 s at the sweep" in x["why"] for x in e.tables["excluded"])


def test_trials_are_fixed_to_the_step_and_say_what_was_recorded():
    recs, truth = simulate.dsi_cell(durations=(2.0,), trials=1, gap_s=30.0)       # the file after starts 30 s late
    c = analyze_cell(cell_of(recs)).results["dsi_events"]
    tr = c.tables["trials"][0]
    assert np.isnan(tr["suppression_pct"]) and tr["post_analyzed_s"] == 0
    assert tr["first_data_after_step_s"] == pytest.approx(30.3, abs=0.05)
    assert "30.3 s after it" in tr["notes"] and any(f.code == "thin window" for f in c.flags)


def test_a_cell_needs_its_files_on_one_clock():
    recs, _ = simulate.dsi_cell(durations=(1.0,), trials=1)
    recs[2].start_time = None                                                     # the file after has no start time
    cell = cell_of(recs)
    assert len(cell.placed()) == 2 and any("cannot be placed" in line for line in cell.clock.describe())
    result = analyze_cell(cell)
    assert any("cannot be placed" in n for n in result.notes)


def test_cell_keys():
    p = Path("D:/data/260929/Slice2/cell1/24125219.abf")
    assert cell_key(Identity(animal="A12", slice="S2", cell="c1"), p) == "A12 / S2 / c1"
    assert cell_key(Identity(slice="Slice2", cell="cell1"), p) == str(p.parent)     # 'Slice1 / cell1' recurs daily
    assert cell_key(Identity(), p) == str(p)


def test_cells_combined_by_duration():
    results = []
    for seed in (1, 2, 3):
        recs, _ = simulate.dsi_cell(durations=(2.0,), trials=2, seed=seed)
        results.append(analyze_cell(cell_of(recs)).results["dsi_events"])
    rows = dsi_events.combine(results)
    assert rows and {r["n_cells"] for r in rows} == {3}
    after = [r for r in rows if 1.0 <= r["from_step_end_s"] < 3.0]
    assert after and all(r["pct_of_control_mean"] < 100 for r in after)


def test_a_trigger_timed_to_a_holding_epoch_is_still_a_stimulus():
    from clampwise.core.recording import Epoch

    sim = simulate.evoked(n_sweeps=3, onsets_s=(0.2,), amplitudes=(-50.0,), light=False)
    rec = sim.recording
    # Clampex lists epochs at the holding level; the trigger fires at one of their boundaries
    rec.epochs = [[Epoch("step", 0.0, 0.2, -70.0), Epoch("step", 0.2, 1.0, -70.0)] for _ in range(3)]
    assert len(stimulus.find_stimulus(rec)) == 3
    rec.epochs = [[Epoch("step", 0.0, 0.2, -70.0), Epoch("step", 0.2, 1.0, -60.0)] for _ in range(3)]
    s = stimulus.find_stimulus(rec)
    assert len(s) == 0 and "line up with the command step" in s.reason


def test_every_episode_is_timed_suppressed_too_little_or_unmeasured_and_the_counts_sum():
    from clampwise.core.recording import ArraySource

    sim = simulate.dsi_evoked(durations=(0.5, 5.0), trials=2, cv=0.0, noise=1.0)     # 8% and 73% suppression
    rec = sim.recording
    signal, command, trigger = (rec.read(c).copy() for c in sorted(rec.channels, key=lambda c: c.index))
    trigger[3, int(25.0 * rec.sample_rate_hz):] = 0.0             # the last trial: no stimulus after its step
    rec.samples = ArraySource([signal, command, trigger])
    r = pipeline.analyze(rec).results["dsi"]
    episodes = r.tables["episodes"]
    timed = sum(bool(e["timed"]) for e in episodes)
    unmeasured = sum(not np.isfinite(e["suppression_pct"]) for e in episodes)
    weak = len(episodes) - timed - unmeasured
    assert (len(episodes), timed, weak, unmeasured) == (4, 1, 2, 1)
    assert any(n.startswith(f"{weak} episode(s) suppressed by less than") for n in r.notes), r.notes
    assert any(n.startswith(f"{unmeasured} episode(s) have no response") for n in r.notes), r.notes


def test_each_step_as_percent_of_its_baseline_and_the_power_beside_it():
    # a typical session: a response every 4 s, the last 4 before the step the baseline, 40 s after it
    from clampwise.project.presets import BUILTIN

    sim = simulate.dsi_evoked(durations=(5.0,), trials=3, stim_every_s=4.0, sweep_s=80.0, cv=0.0, noise=1.0)
    d = pipeline.analyze(sim.recording, pipeline.Settings(by_analysis=BUILTIN["Evoked IPSC DSI"].settings)) \
        .results["dsi"]
    course = d.tables["course_mean"]
    assert [r["response"] for r in course] == [-4, -3, -2, -1] + list(range(1, 11))
    assert all(r["n"] == 3 and r["pct"] == pytest.approx(100.0, abs=2.0) for r in course[:4])
    factor = [1 - sim.truth["s_max"] * 5 ** 2 / (5 ** 2 + 1.5 ** 2) * np.exp(-(r["from_step_s"]) / 15.0)
              for r in course[4:]]
    assert [r["pct"] for r in course[4:]] == pytest.approx([100 * f for f in factor], abs=3.0)
    # the response's Fourier power goes as the square of its size
    assert [r["power_pct"] for r in course[4:]] == pytest.approx([100 * f ** 2 for f in factor], abs=4.0)
    assert d.values["power_suppression_pct"] > d.values["suppression_pct"]
    assert {"psd_baseline", "psd_suppressed", "psd_recovered"} <= set(d.tables["spectra"][0])


def test_the_fourier_transform_of_a_known_sine():
    # 20 pA at 100 Hz for the whole 100 ms window: amplitude 20 at 100 Hz, phase -90 (a sine), power 20^2 / 2
    class Rec:
        sample_rate_hz = 10_000.0

        def sweep(self, k):
            t = np.arange(2000) / 10_000.0
            return np.where(t >= 0.05, 20.0 * np.sin(2 * np.pi * 100.0 * (t - 0.05)), 0.0)

    s = {"power_window": "response", "power_ms": 100.0, "spectrum_to_hz": 5000.0, "power_low_hz": 90.0,
         "power_high_hz": 110.0}
    s_ev = {"blank_ms": 0.0, "baseline_start_ms": -20.0, "baseline_end_ms": -1.0, "response_end_ms": 50.0}
    ft = dsi.fourier([(Rec(), (0, 0, 0.05, 1.0, 0.05, np.inf), [])], s, s_ev)
    k = int(np.argmin(np.abs(ft.freqs - 100.0)))
    assert ft.n == 1000 and ft.freqs[1] == pytest.approx(10.0)
    assert ft.amplitude[0, k] == pytest.approx(20.0, rel=1e-6) and ft.phase_deg[0, k] == pytest.approx(-90.0, abs=0.01)
    assert ft.band_power[0] == pytest.approx(200.0, rel=1e-6)
    # Parseval: the density over every frequency is the window's mean square
    assert ft.density[0].sum() * 10.0 == pytest.approx(200.0, rel=1e-6)


def test_a_tall_brief_artifact_does_not_turn_the_responses_round():
    # inward IPSCs of -200 pA behind a +1000 pA, 2 ms stimulus artifact: the charge says inward
    import copy

    from clampwise.core.recording import ArraySource

    sim = simulate.dsi_evoked(durations=(5.0,), trials=2, stim_every_s=4.0, sweep_s=80.0, noise=1.0)
    rec = copy.copy(sim.recording)
    y = rec.signal.copy()
    trig = rec.read(rec.channels[2])
    fs = rec.sample_rate_hz
    for k in range(y.shape[0]):
        for i in np.flatnonzero(np.diff((trig[k] > 2.5).astype(int)) == 1):
            y[k, i:i + int(0.002 * fs)] += 1000.0
    rec.samples = ArraySource([y] + [rec.read(c)[...] for c in rec.channels[1:]])
    rec._hash = "artifact"
    d = pipeline.analyze(rec).results["dsi"]
    assert d.choices["response direction"].startswith("inward")
    assert all(e["baseline"] < 0 for e in d.tables["episodes"])
    assert d.values["suppression_pct"] == pytest.approx(np.mean([t["suppression_pct"] for t in sim.truth["trials"]]),
                                                         abs=8.0)


CELL_261006 = find("261006/Slice1/cell2") or Path("missing")


@pytest.mark.skipif(not CELL_261006.exists(), reason="the lab's 261006 DSI cell is not among CLAMPWISE_LAB_DATA's")
def test_a_lab_dsi_session_step_and_responses_in_separate_files():
    from clampwise.io import load
    from clampwise.project.presets import BUILTIN

    members = []
    for p in sorted(CELL_261006.glob("*.abf")):
        rec = load(p)
        members.append(Member(p.name, rec, pipeline.analyze(rec).results))
    res = analyze_cell(Cell.build("cell2", members), BUILTIN["Evoked IPSC DSI"].settings).results["dsi_cell"]
    assert res.choices["response direction"].startswith("inward")       # behind a taller outward artifact
    assert res.values["n_episodes"] == 10
    assert [r["response"] for r in res.tables["course_mean"]] == [-4, -3, -2, -1] + list(range(1, 11))
    assert 15 < res.tables["course_mean"][4]["pct"] < 45                  # the first response after the step
    assert {f.code for f in res.flags} >= {"artifact length varies"}
    # the file named DSI_Step_5sec holds a 2.5 s step: what the file says, not its name
    assert sorted({round(e["step_duration_s"], 2) for e in res.tables["episodes"]}) == [0.5, 1.0, 2.0, 2.5, 3.0]
