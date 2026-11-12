"""The Figures screen: what each recording type opens on, the graphs of the lab's light experiments, the Show
menu by analysis, and every graph of every type opening in the figure builder (Edit) as a figure that draws.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from pyqtgraph.Qt import QtWidgets  # noqa: E402

import controls_harness as ch  # noqa: E402
from labdata import lab_file, lab_files  # noqa: E402

# one real cell's light files (the whole field and two regions) and its step family, named on this machine only
ROLES = ("light, whole field", "light, region 1", "light, region 2", "step family")
needs_lab = pytest.mark.skipif(not lab_files(*ROLES),
                               reason="the lab's own recordings are not named on this machine (tests/labdata.py)")


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def opened(tmp_path, qapp):
    """open(recordings) -> (window, paths), in a world whose dialogs are answered."""
    made = []
    with ch.Patched(ch.Answers(tmp_path / "out")) as patched:
        def open_(recordings: dict, folder: str = "cell1"):
            w, paths = ch.open_window(tmp_path / folder, recordings)
            made.append(w)
            return w, paths

        yield patched, open_
        for w in made:
            w.close()
            w.deleteLater()
        ch.flush(qapp)


def light_cell(folder: Path) -> dict:
    """A cell's light files as the lab names them, written as real ABF files in ``folder`` (a cell's files are
    read together from the disk): the whole field answers, one region does not."""
    from clampwise import simulate
    from clampwise.simulate.abf_file import write_abf

    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, seed, success in (("Full_Field.abf", 4, 0.8), ("ROI1.abf", 5, 0.0)):
        rec = simulate.photostim_steps(seed=seed, success=success, levels_pa=[0] * 10).recording
        out[name] = write_abf(folder / name, np.stack([rec.read(0), rec.read(2)]), fs=rec.sample_rate_hz,
                              units=("mV", "V"), names=("IN 0", "LED"), command_units="pA",
                              protocol="sim_photostim_steps")
    return out


def shown(w, path) -> list:
    w.select(path)
    w.go("figures")
    ch.flush(QtWidgets.QApplication.instance())
    return [p for p in w.screens["figures"].panels if p._key]


def draws(spec) -> list:
    """The figure built; every panel holds data and draws something."""
    from clampwise.figure import render
    from clampwise.figure.recipe import Spec

    again = Spec.from_json(spec.to_json())               # as it is saved and opened again
    fig, drawn = render.build(again)
    assert drawn and all(d.prepared is not None for d in drawn), [d.note for d in drawn if d.prepared is None]
    for ax in fig.axes:
        assert ax.lines or ax.collections or ax.patches or ax.images, "a panel with nothing drawn"
    return drawn


@pytest.mark.parametrize("kind", list(ch.simulated()))
def test_every_graph_of_every_type_opens_in_the_builder_and_draws(kind, opened):
    """Each graph the Show menu lists is shown, opened as a figure (Edit), saved, opened again and drawn."""
    _patched, open_ = opened
    w, paths = open_({f"{kind}.abf": ch.simulated()[kind]()})
    fig = w.screens["figures"]
    panels = shown(w, paths[f"{kind}.abf"])
    assert panels, "nothing shown"
    p = panels[0]
    assert len({v.key for v in p._views}) == len(p._views), "two graphs with one key"
    assert all(v.group for v in p._views), [v.key for v in p._views if not v.group]
    opened_ = 0
    for v in p._views:
        p.choose(v.key)
        assert p._key == v.key and [q._key for q in fig.panels if q._key].count(v.key) == 1, v.key
        if not p.edit_button.isEnabled():                # nothing drawn: the button says why
            assert "Nothing is drawn" in p.edit_button.toolTip(), v.key
            continue
        spec = p.as_figure(f"{kind}.abf")
        assert spec is not None, v.key
        drawn = draws(spec)
        assert drawn[0].recipe.source.get("kind") in ("graph", "traces"), v.key
        opened_ += 1
    assert opened_ >= 1


def test_a_light_recording_opens_on_its_sweeps_the_test_the_patterns_and_the_cells_health(opened, tmp_path):
    from clampwise.figure import render

    _patched, open_ = opened
    where = "R001/Slice1/cell1"                                # the lab's folders: animal, slice, cell
    w, paths = open_(light_cell(tmp_path / where), where)
    w.project.analyze_cells()
    fig = w.screens["figures"]
    panels = shown(w, paths["Full_Field.abf"])
    assert [p._key for p in panels] == ["light.example", "connectivity.sham", "connectivity_cell.patterns",
                                        "sweep_health.v_before"]
    run = w.project.last_run(paths["Full_Field.abf"], w.recording(paths["Full_Field.abf"]))
    c = run.results["connectivity"].values
    example, sham, patterns, health = panels
    # the sweeps around the light: every sweep and their average, the light shaded, the verdict in words
    import pyqtgraph as pg

    item = example.plot.getPlotItem()
    curves = [it for it in item.items if isinstance(it, pg.PlotDataItem)]
    bands = [it for it in item.items if isinstance(it, pg.LinearRegionItem)]
    assert len(curves) == run.recording.n_sweeps + 1 and len(bands) == 2
    assert bands[0].getRegion()[0] == pytest.approx(0.0) and bands[0].getRegion()[1] == pytest.approx(5.0, abs=0.2)
    assert "Connected:" in example.readout.toolTip() and "2 pulses of 5 ms" in example.readout.toolTip()
    spec = example.as_figure("Full_Field.abf")
    r = spec.panels[0]
    assert (r.question, r.plot, r.source["kind"]) == ("example", "trace", "traces")
    assert [b["start"] for b in r.stim_bars] == pytest.approx([0.0, 105.0], abs=0.6) and r.scale_bars
    assert all(b.get("nm") for b in r.stim_bars) and r.vm_labels            # the light's bars, the potential held
    fig_, _ = render.build(spec)
    assert {t.kind for t, _a in render.things(fig_)} >= {"stimbar", "scalebar", "vmlabel", "series"}
    # the test itself: sham windows as bars, the response windows over them, the threshold, the verdict
    text = sham.readout.toolTip()
    assert f"{int(c['successes'])} of {int(c['n_windows'])} response windows beat the sham threshold" in text
    assert f"chance alone gives {c['chance_successes']:.1f}" in text and "false-positive rate 5%" in text
    item = sham.plot.getPlotItem()
    line = next(it for it in item.items if isinstance(it, pg.InfiniteLine))
    assert 10 ** line.value() == pytest.approx(c["threshold_z"])
    marks = [it for it in item.items if isinstance(it, pg.PlotDataItem)]
    assert sum(len(it.getData()[0]) for it in marks) == c["n_windows"]
    d = draws(sham.as_figure("Full_Field.abf"))[0]
    assert d.recipe.plot == "cumulative" and set(d.prepared.rows.group) == {"sham windows", "response windows"}
    assert d.recipe.ref_lines[0]["at"] == pytest.approx(c["threshold_z"])
    # the cell's patterns side by side: opened in the builder as proportions with Fisher's exact test
    assert "Full field: yes" in patterns.readout.toolTip() and "ROI1: no" in patterns.readout.toolTip()
    d = draws(patterns.as_figure("Full_Field.abf"))[0]
    assert (d.recipe.question, d.recipe.plot, d.recipe.event) == ("counts", "proportion", "success")
    cell = w.project.cell_result(paths["Full_Field.abf"]).results["connectivity_cell"]
    files = {f["pattern"]: f for f in cell.tables["files"]}
    comp = d.comparisons[0]
    assert comp.test == "Fisher's exact test" and comp.counts == (
        files["Full field"]["successes"], files["Full field"]["n_windows"], files["ROI1"]["successes"],
        files["ROI1"]["n_windows"]) and comp.p < 0.001
    # the cell's health, sweep by sweep
    assert "Healthy in all" in health.readout.toolTip() or "Unhealthy in" in health.readout.toolTip()
    # the other graphs of the experiment are one click away, listed by analysis
    menus = {a.text().rsplit(" (", 1)[0]: a.menu() for a in example.menu.actions() if a.menu() is not None}
    assert {"Example traces", "Connected or not", "Connected or not, over a cell's files"} <= set(menus)
    names = [a.text() for a in menus["Connected or not"].actions()]
    assert names[:3] == ["Response windows against sham windows [% vs. × noise]",
                         "Success rate of each pulse [% vs. pulse]", "Latency of each success, by pulse [ms vs. pulse]"]
    # choosing a graph another panel shows changes the two places
    before = [p._key for p in fig.panels]
    example.choose("connectivity.sham")
    assert [p._key for p in fig.panels][:2] == [before[1], before[0]]
    # each pulse's success rate, with what its readout says of latency and jitter
    example.choose("connectivity.success")
    said = example.readout.toolTip()
    pulses = run.results["connectivity"].tables["pulses"]
    assert f"pulse 1: {pulses[0]['successes']} of {pulses[0]['n_windows']} windows" in said and "latency" in said \
        and "jitter" in said and "Wilson" in said
    d = draws(example.as_figure("Full_Field.abf"))[0]
    assert d.recipe.plot == "proportion" and list(dict.fromkeys(d.prepared.points.group)) == ["pulse 1", "pulse 2"]


def test_a_recording_of_unknown_kind_shows_its_sweeps_and_says_why(opened):
    from clampwise import simulate

    _patched, open_ = opened
    rec = simulate.firing_steps().recording
    w, paths = open_({"odd.abf": rec})
    path = paths["odd.abf"]
    run = w.project.last_run(path, w.recording(path))
    fig = w.screens["figures"]
    # as if its protocol had not been recognised: only the baseline measured
    from types import SimpleNamespace

    odd = SimpleNamespace(results={"baseline": run.results["baseline"]}, recording=run.recording,
                          recording_type="unknown", type_reason="unknown (detected): one sweep with one command step")
    w.project.last_run = lambda *a, **k: odd
    panels = shown(w, path)
    assert [p._key for p in panels] == ["recording.sweeps"] and sum(p.isVisible() for p in fig.panels) == 1
    note = fig.note.text()
    assert fig.note.isVisible() and "could not tell what kind of recording this is (one sweep with one command step)" \
        in note and "unknown (detected)" not in note and "set its type in Check" in note
    assert "as recorded" in panels[0].readout.text()
    d = draws(panels[0].as_figure("odd.abf"))[0]
    assert d.recipe.plot == "trace" and d.recipe.source["kind"] == "traces"
    assert panels[0].menu_button.isEnabled() and len(panels[0]._views) >= 2       # Show lists what there is


@needs_lab
def test_a_real_cell_on_the_figures_screen(opened, tmp_path):
    """Copies of one real cell's light files and step family (the originals are only read): the whole-field file
    opens on its sweeps with the light, the test, the patterns side by side and the cell's health, each readout
    saying what the analyses found; every graph shown opens in the builder and draws."""
    from scipy import stats as sps

    _patched, open_ = opened
    folder = tmp_path / "A1" / "Slice1" / "cell1"
    folder.mkdir(parents=True)
    names = []
    for role in ROLES:
        shutil.copy2(lab_file(role), folder / lab_file(role).name)
        names.append(lab_file(role).name)
    w, paths = ch.open_window(folder, {n: folder / n for n in names})
    try:
        w.project.analyze_cells()
        whole = paths[names[0]]
        panels = shown(w, whole)
        assert [p._key for p in panels] == ["light.example", "connectivity.sham", "connectivity_cell.patterns",
                                            "sweep_health.v_before"]
        run = w.project.last_run(whole, w.recording(whole))
        c = run.results["connectivity"].values
        assert (f"{int(c['successes'])} of {int(c['n_windows'])} response windows beat the sham threshold (chance "
                f"alone gives {c['chance_successes']:.1f})") in panels[1].readout.toolTip()
        cell = w.project.cell_result(whole).results["connectivity_cell"]
        files = cell.tables["files"]
        said = panels[2].readout.toolTip()
        assert len(files) == 3 and all(f"{f['pattern']}: {f['connected']}" in said for f in files)
        h = run.results["sweep_health"].values
        bad, n = int(h["unhealthy_sweeps"]), int(h["n_sweeps"])
        assert (f"Unhealthy in {bad} of {n} sweeps" if bad else f"Healthy in all {n} sweeps") in \
            panels[3].readout.toolTip()
        for p in panels:
            draws(p.as_figure(names[0]))
        d = draws(panels[2].as_figure(names[0]))[0]
        want = [(int(a["successes"]), int(a["n_windows"]), int(b["successes"]), int(b["n_windows"]))
                for i, a in enumerate(files) for b in files[i + 1:]]
        assert [c_.counts for c_ in d.comparisons] == want
        assert [c_.p_raw for c_ in d.comparisons] == pytest.approx(
            [float(sps.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]) for k1, n1, k2, n2 in want])
        panels = shown(w, paths[names[3]])                     # the step family opens on its own graphs
        assert panels[0]._key.startswith("intrinsic.steps.v_") and len(panels) == 4
        assert all((folder / n).read_bytes() == lab_file(r).read_bytes() for n, r in zip(names, ROLES))
    finally:
        w.close()
        w.deleteLater()
        ch.flush(QtWidgets.QApplication.instance())
