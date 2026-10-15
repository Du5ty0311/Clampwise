"""Figure presets: each fills every panel from its kind of recording (or a
table of cells), draws, says how it chose, and reopens unchanged."""

from __future__ import annotations

from pathlib import Path

import pytest

from clampwise import simulate
from clampwise.analysis import pipeline
from clampwise.figure import presets, render
from clampwise.figure.recipe import Spec
from clampwise.project.presets import BUILTIN

from test_figure import nested

from labdata import DEMOS, find  # noqa: E402

CELL_261006 = find("261006/Slice1/cell2") or Path("missing")


def selection(rec, preset: str | None = None) -> presets.Selection:
    s = pipeline.Settings(by_analysis=BUILTIN[preset].settings) if preset else pipeline.Settings()
    return presets.Selection(results=pipeline.analyze(rec, s).results, load=lambda _f: rec, name=Path(str(rec.source)).name)


def filled(key: str, sel: presets.Selection) -> Spec:
    """The preset filled with no empty panel, drawn, and reopened from its JSON unchanged."""
    p = presets.PRESETS[key]
    assert presets.needs(p, sel) == ""
    spec = presets.fill(p, sel)
    assert len(spec.panels) == len(p.panels)
    for r, want in zip(spec.panels, p.panels):
        assert r.source.get("kind") != "preset", f"{want.title}: {r.source.get('needs')}"
        assert r.frame() is not None and len(r.frame()), want.title
    _, drawn = render.build(spec)
    assert all(d.prepared is not None for d in drawn)
    text = spec.to_json()
    assert Spec.from_json(text).to_json() == text
    return spec


def test_current_clamp_step_family_from_simulated_steps():
    spec = filled("cc_steps", selection(simulate.firing_steps().recording))
    a = spec.panels[0]
    names = list(dict.fromkeys(a.frame()["Trace"]))
    assert len(names) == 3 and "(rheobase)" in names[1] and a.source["kind"] == "traces"
    assert any("rheobase (the smallest step with a full spike)" in n for n in a.notes)
    assert set(spec.panels[2].frame()["Measure"]) == {"peak", "steady state"}
    assert (spec.panels[3].frame()["Current (pA)"] < 0).all()


@pytest.mark.skipif(not (DEMOS / "Test_IV_1.abf").is_file(), reason="the demo files are not on this machine")
def test_current_clamp_step_family_from_the_demo():
    from clampwise.io import load

    filled("cc_steps", selection(load(DEMOS / "Test_IV_1.abf")))


def test_evoked_psc_with_its_rise_and_decay_drawn():
    spec = filled("evoked", selection(simulate.evoked().recording))
    a = spec.panels[0].frame()
    assert {"averaged response", "decay fit", "10-90% rise"} <= set(a.Trace)
    assert any("decay τ" in n for n in spec.panels[0].notes)
    assert spec.panels[0].source["overlay"] == "kinetics"
    assert spec.panels[1].ref_lines[-1]["label"] == "mean" and "Time (min)" == spec.panels[1].x
    assert any(n.startswith("Jitter") for n in spec.panels[2].notes)


def test_paired_pulses_one_trace_and_one_ratio_per_interval():
    sim = simulate.interval_series(ppr=(0.6, 0.8, 1.0, 1.2))
    spec = filled("paired_pulses", selection(sim.recording))
    assert list(dict.fromkeys(spec.panels[0].frame()["Trace"])) == ["12 ms", "25 ms", "50 ms", "100 ms"]
    ppr = spec.panels[1].frame()
    assert list(ppr["Interval (ms)"]) == [12, 25, 50, 100]
    assert ppr["PPR (P2/P1)"].to_numpy() == pytest.approx([0.6, 0.8, 1.0, 1.2], abs=0.1)


def test_spontaneous_events_from_a_simulated_recording():
    spec = filled("events", selection(simulate.spontaneous().recording))
    a = spec.panels[0]
    assert "detected events" in set(a.frame().Trace) and a.series["detected events"]["line"] == ""
    assert any("closest to the mean" in n for n in a.notes)
    assert "decay fit" in set(spec.panels[1].frame().Trace)


@pytest.mark.skipif(not (DEMOS / "Test_mEPSC_1.abf").is_file(), reason="the demo files are not on this machine")
def test_spontaneous_events_from_the_demo():
    from clampwise.io import load

    filled("events", selection(load(DEMOS / "Test_mEPSC_1.abf")))


def test_dsi_from_a_simulated_recording_before_and_after_in_black_and_grey():
    sim = simulate.dsi_evoked(durations=(1.0, 5.0), trials=2)
    spec = filled("dsi", selection(sim.recording, "Evoked IPSC DSI"))
    a = spec.panels[0]
    assert [s["name"] for s in a.source["sets"]] == ["before the step", "just after the step"]
    assert all(s["mode"] == "average" for s in a.source["sets"])
    assert a.series["before the step"]["colour"] == "#000000" and a.series["just after the step"]["colour"] == "#999999"


@pytest.mark.skipif(not CELL_261006.exists(), reason="the lab's 261006 DSI cell is not among CLAMPWISE_LAB_DATA's")
def test_dsi_from_a_cell_whose_step_and_responses_are_in_separate_files():
    from clampwise.analysis.cells import Cell, Member, analyze_cell
    from clampwise.io import load

    recs = {p.name: load(p) for p in sorted(CELL_261006.glob("*.abf"))}
    members = [Member(n, r, pipeline.analyze(r).results) for n, r in recs.items()]
    res = analyze_cell(Cell.build("cell2", members), BUILTIN["Evoked IPSC DSI"].settings).results["dsi_cell"]
    first = next(iter(recs.values()))
    sel = presets.Selection(results={"dsi_cell": res}, load=lambda f: recs.get(Path(str(f)).name, first) if f else
                            first, name="cell2")
    spec = filled("dsi", sel)
    files = {Path(i[0]).name for s in spec.panels[0].source["sets"] for i in s["items"]}
    assert len(files) > 1                       # responses picked by their role around the steps, across files


def test_group_comparison_from_a_table_of_cells():
    sel = presets.Selection(table=nested(effect=3.0), source={"kind": "file", "path": "cells.csv"})
    spec = filled("groups", sel)
    a, b = spec.panels
    assert (a.y, a.group, a.cell, a.animal) == ("Amplitude (pA)", "Condition", "Cell", "Animal")
    assert b.plot == "superplot" and b.stats                 # animals named, several rows per cell
    assert a.series["control"]["colour"] == "#000000"


def test_a_preset_that_does_not_apply_says_what_it_needs_and_still_opens():
    sel = selection(simulate.evoked().recording)
    assert presets.needs(presets.PRESETS["cc_steps"], sel).startswith("Needs a current-step recording")
    spec = presets.fill(presets.PRESETS["cc_steps"], sel)
    assert all(r.source.get("needs") for r in spec.panels)
    _, drawn = render.build(spec)
    assert len(drawn) == 4
