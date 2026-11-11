"""Every button, checkbox, pick list and menu action of the Workbench, the Figures screen and the figure builder
is pressed, on a simulated recording of every recording type and, when ``E:\\Daniel`` is there, on copies of his
files (nothing is ever analyzed in place). A control that raises is broken; so is one that changes nothing, an
edit that Undo does not put back, and a disabled control that does not say why. Then, control by control, that
each does what its label says.

``docs/development/CONTROLS.md`` lists every control with the effect checked; it is written from these runs
(``tools/audit/controls_doc.py``), and a control missing from it fails here.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path(r"C:\Windows\Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")       # offscreen screenshots with their text

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: E402

import controls_harness as ch  # noqa: E402

DANIEL = Path(r"E:\Daniel")
HIS = {"full field": DANIEL / "lab-project/A8/Slice3/cell1/Full_Field_cell_response.abf",
       "step family": DANIEL / "lab-project/A8/Slice3/cell1/IV_response.abf",
       "unknown": DANIEL / "lab-project-2/A4/Slice1/cell2/26n01002.abf"}
needs_daniel = pytest.mark.skipif(not all(p.is_file() for p in HIS.values()),
                                  reason="Daniel's recordings (E:\\Daniel) are not on this machine")
DOC = Path(__file__).resolve().parents[1] / "docs" / "development" / "CONTROLS.md"
TYPES = list(ch.simulated())


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def press(tmp_path, qapp):
    """A patched world to press in: (patched, open) where ``open(recordings)`` gives a window on them."""
    made = []
    with ch.Patched(ch.Answers(tmp_path / "out")) as patched:
        def open_(recordings: dict, analyze: bool = True):
            w, paths = ch.open_window(tmp_path / "cell1", recordings, analyze)
            made.append(w)
            return w, paths

        yield patched, open_
        for w in made:
            w.close()
            w.deleteLater()
        ch.flush(qapp)


def his_copies(folder: Path, *keys: str) -> dict:
    """Copies of Daniel's files in ``folder`` (his own are only ever read)."""
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for k in keys:
        dst = folder / HIS[k].name
        shutil.copy2(HIS[k], dst)
        out[dst.name] = dst
    return out


def fine(outs: list[ch.Outcome], allowed: dict | None = None) -> None:
    out = os.environ.get("CLAMPWISE_CONTROLS_OUT")       # set by tools/audit/controls_doc.py, which writes the list
    if out:
        ch.record(outs, out)
    bad = ch.problems(outs, allowed)
    assert not bad, "\n".join(bad)
    if not out:
        documented(outs)


def documented(outs: list[ch.Outcome]) -> None:
    """Every control pressed is in CONTROLS.md (written by tools/audit/controls_doc.py from these same runs)."""
    if not DOC.is_file():
        pytest.fail(f"{DOC} is missing: run tools/audit/controls_doc.py")
    listed = set()
    for line in DOC.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("| ") and len(cells) >= 5:
            listed.add(" | ".join(cells[:3]))
    missing = sorted({ch.doc_name(o.control) for o in outs} - listed)
    assert not missing, "not in CONTROLS.md (run tools/audit/controls_doc.py): " + "; ".join(missing)


def table(folder: Path) -> Path:
    """Cells in animals in two conditions over time: a table every question of the builder can draw."""
    rng = np.random.default_rng(0)
    rows = [{"Condition": g, "Animal": f"{g}-A{a}", "Cell": f"{g}-A{a}-C{c}", "Time (min)": float(t),
             "Amplitude (pA)": -20 + shift + rng.normal(0, 2), "Rise (ms)": 1 + rng.random()}
            for g, shift in (("control", 0.0), ("drug", 3.0)) for a in range(3) for c in range(3) for t in range(8)]
    path = folder / "cells.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- every control, every recording type

@pytest.mark.parametrize("kind", TYPES)
def test_every_workbench_control_on_a_recording_of_each_type(kind, press):
    patched, open_ = press
    w, paths = open_({f"{kind}.abf": ch.simulated()[kind]()})
    path = paths[f"{kind}.abf"]
    fine(ch.WorkbenchSession(patched, w, path).run(f"{kind}, as opened"))
    if kind in ("cc_evoked", "cc_steps"):            # and with something for every button to act on
        fine(ch.WorkbenchSession(patched, w, path, ch.busy_workbench).run(f"{kind}, in use"))


@pytest.mark.parametrize("kind", TYPES)
def test_every_figures_control_on_a_recording_of_each_type(kind, press):
    patched, open_ = press
    w, paths = open_({f"{kind}.abf": ch.simulated()[kind]()})
    s = ch.FiguresSession(patched, w, paths[f"{kind}.abf"])
    fine(s.run(kind, only=ch.first_menu_whole))
    s.reset()
    assert len({p._key for p in s.s.panels if p._key}) == len([p for p in s.s.panels if p._key]), "a graph twice"


def test_every_control_of_the_comparison_across_recordings(press):
    patched, open_ = press
    from clampwise import simulate

    recs = {f"cell{k}.abf": simulate.firing_steps(rm_mohm=110.0 + 25 * k, v_rest=-70.0 + 2 * k, seed=k,
                                                  rheobase_pa=40.0 + 20 * (k % 3)).recording for k in range(6)}
    w, paths = open_(recs)
    s = ch.FiguresSession(patched, w, paths["cell0.abf"], across=True)
    fine(s.run("across recordings"))
    s.reset()
    s.s.across.correlations.setChecked(True)            # the cells differ: the map is drawn, and can be edited
    assert s.s.across._hm is not None and s.s.across.edit_button.isEnabled()
    s.s.across.edit_button.click()
    assert s.s.builder is not None and s.s.builder.recipe.plot == "heatmap"
    s.s.builder.close()


BUILDER_STATES = ("empty", "compare", "time", "pulses", "relationship", "distribution", "counts", "example",
                  "a preset", "a graph of Figures")


def builder_in(state: str, w, paths, folder: Path, answers: ch.Answers):
    """The builder in ``state``: empty, a table drawn for one of the questions, a recording's typical response,
    a preset's panels, or a graph opened from the Figures screen."""
    from clampwise.figure.recipe import QUESTIONS

    fig = w.screens["figures"]
    path = next(iter(paths.values()))
    w.select(path)
    if state == "a graph of Figures":
        w.go("figures")
        return fig.edit_graph(fig.panels[0])
    b = fig.open_builder()
    if state == "a preset":
        b.apply_preset("evoked")
    elif state != "empty":
        b.open_file(str(table(folder)))
        i = [q[0] for q in QUESTIONS].index(state)
        b.question_box.setCurrentIndex(i)
        b._question_chosen(i)
        if state == "example":
            b.use_example()
    b.redraw()
    saved = folder / "saved.cwfig"
    b.save_figure(str(saved))
    answers.files.update({"*.csv": table(folder), "*.cwfig": saved, "*.cwstyles": answers.out / "figure styles.cwstyles"})
    return b


def test_every_builder_control_in_every_state(press, tmp_path):
    patched, open_ = press
    w, paths = open_({"evoked.abf": ch.simulated()["vc_evoked"]()})
    seen: set[str] = set()                     # a control pressed with an effect in an earlier state is not again
    for state in BUILDER_STATES:
        b = builder_in(state, w, paths, tmp_path, patched.answers)
        s = ch.BuilderSession(patched, b)
        outs = s.run(state, only=lambda c: ch.doc_name(c) + c.choice not in seen)
        fine(outs)
        seen |= {ch.doc_name(o.control) + o.control.choice for o in outs if o.effects}
        b.close()
    assert len(seen) > 150


def test_every_control_of_the_trace_picker(press, tmp_path):
    from clampwise.app.tracepicker import TracePicker

    patched, open_ = press
    w, paths = open_({"evoked.abf": ch.simulated()["vc_evoked"]()})
    w.select(paths["evoked.abf"])
    b = w.screens["figures"].open_builder()
    fine(ch.DialogSession(patched, lambda: TracePicker(b, w, b._load), ch.picker_ready).run("trace picker"))
    b.close()


def test_every_control_of_the_digitizer(press, tmp_path):
    from matplotlib.figure import Figure

    from clampwise.app.builder import DigitizeDialog

    patched, open_ = press
    w, paths = open_({"evoked.abf": ch.simulated()["vc_evoked"]()}, analyze=False)
    f = Figure(figsize=(4, 3), dpi=100)
    f.add_subplot().plot([0, 1], [0, 1], color="#d62728")
    f.savefig(tmp_path / "plot.png", dpi=100)
    b = w.screens["figures"].open_builder()
    fine(ch.DialogSession(patched, lambda: DigitizeDialog(b, tmp_path / "plot.png"), ch.digitizer_ready)
         .run("digitizer"))
    b.close()


# ---------------------------------------------------------------- on Daniel's files (copies)

@needs_daniel
@pytest.mark.parametrize("which", list(HIS))
def test_every_control_on_copies_of_daniels_files(which, press, tmp_path):
    patched, open_ = press
    w, paths = open_(his_copies(tmp_path / "cell1", which))
    path = next(iter(paths.values()))
    fine(ch.WorkbenchSession(patched, w, path).run(f"his {which} file, as opened"))
    fine(ch.WorkbenchSession(patched, w, path, ch.busy_workbench).run(f"his {which} file, in use"))
    fine(ch.FiguresSession(patched, w, path).run(f"his {which} file", only=ch.first_menu_whole))
    w.select(path)
    b = w.screens["figures"].open_builder()
    fine(ch.BuilderSession(patched, b).run(f"his {which} file", only=lambda c: c.kind in ("button", "menu action")))
    b.close()
    assert not any(p.read_bytes() != HIS[k].read_bytes() for k, p in zip([which], paths.values())), "a copy changed"


# ---------------------------------------------------------------- what was broken stays fixed

def test_the_workbench_table_survives_a_region_that_measures_nothing(press):
    """Ticking "Repeat on each stimulus pulse" on a recording with no stimulus, or removing the only measure
    region, leaves a table with no rows and no columns; the header then asked the old columns' names of it
    (IndexError in RowsModel.headerData, on every redraw)."""
    patched, open_ = press
    w, paths = open_({"steps.abf": ch.simulated()["cc_steps"]()})
    s = ch.WorkbenchSession(patched, w, paths["steps.abf"]).s
    n_cols = s.table.model().columnCount()
    assert n_cols > 5 and s.table.model().rowCount() > 0
    s.region_list.setCurrentRow(1)                       # the measure region
    s._m.regions[1].per_pulse = True                     # as the box does when it can be ticked
    s._apply_state(s._m.to_dict())
    ch.flush(QtWidgets.QApplication.instance())
    assert s.table.model().rowCount() == 0 and s.table.model().columnCount() == 0
    s.remove_region_button.click()
    w.undo.undo()
    ch.flush(QtWidgets.QApplication.instance())
    assert not patched.log.errors, patched.log.errors[-1]
    m = s.table.model()
    assert m.headerData(99, QtCore.Qt.Horizontal) is None and m.data(m.index(0, 0)) is None     # asked past its end
    assert s.table.model() is s.rows_model               # one model for the screen's life, its rows replaced


def test_the_workbench_table_numbers_sweeps_from_one_with_units(press):
    patched, open_ = press
    w, paths = open_({"light.abf": ch.simulated()["cc_evoked"]()})
    s = ch.WorkbenchSession(patched, w, paths["light.abf"]).s
    m = s.table.model()
    heads = [m.headerData(c, QtCore.Qt.Horizontal) for c in range(m.columnCount())]
    assert {"Sweep", "Peak (mV)", "Time of peak (ms)", "Area (mV·ms)", "Command (pA)", "Baseline (mV)"} <= set(heads)
    sweeps = [m.data(m.index(r, heads.index("Sweep"))) for r in range(m.rowCount())]
    assert sweeps[0] == "1" and sweeps[-1] == str(s._recording.n_sweeps)        # they read 2 to n + 1
    s.tabs.setCurrentIndex(3)
    assert s.summary.text().count("(n = ") >= len(s._m.measures)                # every line says its n


def test_the_trace_picker_opens_on_a_recording_with_several_sweeps(press):
    """It raised (the truth value of an array...) on any recording with more than one sweep, so "Choose
    traces..." and "Raw traces from the recordings" did nothing at all."""
    from clampwise.app.tracepicker import TracePicker

    patched, open_ = press
    w, paths = open_({"evoked.abf": ch.simulated()["vc_evoked"]()})
    w.select(paths["evoked.abf"])
    b = w.screens["figures"].open_builder()
    dlg = TracePicker(b, w, b._load)
    n = w.recording(paths["evoked.abf"]).n_sweeps
    assert dlg.table.rowCount() == n > 1 and dlg.table.item(1, 1).text() != ""
    dlg.t_from.setValue(0.0)
    dlg.t_to.setValue(1e6)
    dlg._select_time()
    assert len(dlg._selected_items()) == n
    dlg._add()
    assert len(dlg.sets) == 1 and dlg.ok.isEnabled() and not patched.log.errors
    dlg.close()
    b.choose_traces()                                    # the menu's entry: the dialog is made and asked
    assert patched.log.dialogs[-1].startswith("dialog:") and not patched.log.errors
    b.close()


def test_no_graph_is_shown_twice_and_a_file_of_unknown_type_says_why(press):
    from clampwise.app import panels

    patched, open_ = press
    w, paths = open_({"light.abf": ch.simulated()["cc_evoked"](), "steps.abf": ch.simulated()["vc_steps"]()})
    fig = w.screens["figures"]
    w.go("figures")
    for name in paths:
        w.select(paths[name])
        ch.flush(QtWidgets.QApplication.instance())
        shown = [p for p in fig.panels if p._key]
        assert len({p._key for p in shown}) == len(shown) == 4
        titles = [p.title.toolTip() for p in shown]
        assert len(set(titles)) == 4, titles             # nor two graphs of one name
        assert all(sum(a.isChecked() for a in p.menu.actions()) == 1 for p in shown)      # the one shown is ticked
    # one graph to give: one panel shows it, the others give it their room, and a line says why
    one = panels.View("only.one", "The only graph", "[mV vs. s]", lambda item, pal: (item.plot([0, 1], [0, 1]), "x")[1])
    taken: set = set()
    for p in fig.panels:
        taken.add(p.set_views([one], taken))
    assert [p._key for p in fig.panels] == ["only.one", "", "", ""]
    assert "Nothing more to show" in fig.panels[1].title.text() and not fig.panels[1].edit_button.isEnabled()
    assert "nothing to edit" in fig.panels[1].edit_button.toolTip()
    run = type("Run", (), {"type_reason": "one sweep with one step"})()
    assert "could not tell what kind of recording" in fig._why_few(run, "unknown", 1)
    assert fig._why_few(run, "cc_steps", 4) == ""


def test_a_graph_with_nothing_drawn_says_why_it_cannot_be_edited(press):
    from clampwise.app import panels, theme

    patched, open_ = press
    p = panels.Panel(theme.LIGHT)
    p.set_views([panels.View("health.rs", "Cell health: Rs", "[MOhm vs. min]", lambda item, pal: "No Rs: these "
                             "files have no test step")])
    assert not p.edit_button.isEnabled() and "No Rs" in p.edit_button.toolTip()
    p.set_views([panels.View("a.b", "A graph", "[mV]", lambda item, pal: (item.plot([0, 1], [0, 1]), "")[1])])
    assert p.edit_button.isEnabled() and p.as_figure("x") is not None


# ---------------------------------------------------------------- each control does what its label says

def light_files(folder: Path, *names: str) -> dict:
    """Simulated light recordings written as real ABF files (a template run on a protocol reads the files)."""
    from clampwise import simulate
    from clampwise.analysis.stimulus import find_stimulus
    from clampwise.simulate.abf_file import write_abf

    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, name in enumerate(names):
        rec = simulate.photostim_steps(seed=3 + k, levels_pa=[0] * 8).recording
        rec.stimulus = find_stimulus(rec)
        out[name] = write_abf(folder / name, np.stack([rec.read(0), rec.read(2)]), fs=rec.sample_rate_hz,
                              units=("mV", "V"), names=("IN 0", "AM_Stim"), command_units="pA",
                              protocol="sim_photostim_steps")
    return out


def test_the_workbench_does_what_its_labels_say(press, tmp_path):
    from clampwise.analysis import workbench as wb

    patched, open_ = press
    w, paths = open_(light_files(tmp_path / "cell1", "light.abf", "light2.abf"))
    path = paths["light.abf"]
    s = ch.WorkbenchSession(patched, w, path).s
    app = QtWidgets.QApplication.instance()
    rec = s._recording
    n = rec.n_sweeps

    def rows():
        return s._result["rows"]

    # Add region / Remove region, and Undo
    s.add_region_button.click()
    assert [r.name for r in s._m.regions] == ["baseline", "response", "region 3"] and s.region_list.count() == 3
    s.region_list.setCurrentRow(2)
    s.remove_region_button.click()
    assert len(s._m.regions) == 2
    w.undo.undo()
    assert len(s._m.regions) == 3
    w.undo.undo()
    assert len(s._m.regions) == 2
    # Repeat on each stimulus pulse: two pulses a sweep, one window without it
    s.region_list.setCurrentRow(1)
    assert s.r_per_pulse.isChecked() and len(rows()) == 2 * n
    s.r_pulse.setValue(1)
    s._region_edited()
    assert len(rows()) == n and {r["pulse"] for r in rows()} == {1}
    s.r_pulse.setValue(0)
    s._region_edited()
    # a measure ticked is a column of the table
    item = next(s.measure_list.item(k) for k in range(s.measure_list.count())
                if s.measure_list.item(k).data(QtCore.Qt.UserRole) == "half_width")
    item.setCheckState(QtCore.Qt.Checked)
    m = s.table.model()
    assert "Half-width (ms)" in [m.headerData(c, QtCore.Qt.Horizontal) for c in range(m.columnCount())]
    # the average of the sweeps: one trace measured
    s.mode.setCurrentIndex(s.mode.findData("average"))
    assert len(rows()) == 2 and rows()[0]["trace"] == f"average of {n}"
    s.mode.setCurrentIndex(s.mode.findData("each"))
    # the channel measured
    s.channel.setCurrentIndex(s.channel.findData("AM_Stim"))
    assert s._m.selection.channel == "AM_Stim" and "(V)" in m.headerData(
        [m.headerData(c, QtCore.Qt.Horizontal) for c in range(m.columnCount())].index("Baseline (V)"),
        QtCore.Qt.Horizontal)
    s.channel.setCurrentIndex(0)
    # Stimulus: a dashed line at each pulse; Command: the step drawn under the trace
    import pyqtgraph as pg

    def lines():
        return [i for i in s.plot.getPlotItem().items if isinstance(i, pg.InfiniteLine)
                and not isinstance(i.parentItem(), pg.LinearRegionItem)]

    def curves():
        return [i for i in s.plot.getPlotItem().items if isinstance(i, pg.PlotDataItem)]

    assert len(lines()) == 2
    s.show_stimulus.click()
    assert len(lines()) == 0
    s.show_stimulus.click()
    before = len(curves())
    s.show_command.click()
    assert len(curves()) == before + 1
    s.show_command.click()
    assert len(curves()) == before
    # Add transform > Low-pass; Remove; Clear all
    patched.answers.pick_action = None
    s._add_transform("lowpass")
    s._add_transform("offset")
    assert [x.kind for x in s._m.transforms] == ["lowpass", "offset"] and s.transform_list.count() == 2
    assert "low-pass 2000 Hz" in s.transform_list.item(0).text()
    s.transform_list.setCurrentRow(1)
    s.remove_transform_button.click()
    assert [x.kind for x in s._m.transforms] == ["lowpass"]
    s.clear_transforms_button.click()
    assert s._m.transforms == [] and not s.clear_transforms_button.isEnabled()
    assert "no transform" in s.clear_transforms_button.toolTip()
    # Set onset, then a click on the trace; Leave out a sweep, with its reason; Clear clicks and exclusions
    on = float(rec.stimulus.times_s[0][0])

    def click_at(t_s: float, sweep: int) -> None:
        y = float(wb.raw(rec, sweep)[int(t_s * rec.sample_rate_hz)])
        vb = s.plot.getPlotItem().getViewBox()
        event = type("Click", (), {"scenePos": lambda self: vb.mapViewToScene(QtCore.QPointF(t_s, y))})()
        s._clicked(event)

    next(s.measure_list.item(k) for k in range(s.measure_list.count())
         if s.measure_list.item(k).data(QtCore.Qt.UserRole) == "onset_latency").setCheckState(QtCore.Qt.Checked)
    s.click_buttons["onset"][0].click()
    assert s._click_mode == "onset" and "onset" in s.click_label.text() and s.click_buttons["onset"][0].isChecked()
    click_at(on + 0.006, 0)
    assert [(o.sweep, o.what, o.region) for o in s._m.overrides] == [(0, "onset", "response")]
    assert next(r for r in rows() if r["sweep"] == 0 and r["pulse"] == 1)["onset_latency"] == pytest.approx(6.0, abs=0.1)
    s.click_buttons["exclude"][0].click()
    patched.answers.text = "the seal went"
    click_at(on + 0.006, 2)
    patched.answers.text = None
    assert [(e.sweep, e.reason) for e in s._m.exclusions] == [(2, "the seal went")]
    assert all(r["sweep"] != 2 for r in rows()) and "sweep 3 left out: the seal went" in s.set_words.text()
    s.click_buttons["exclude"][0].click()                # the same button again: the mode ends
    assert s._click_mode == ""
    s.clear_set_button.click()
    assert s._m.overrides == [] and s._m.exclusions == [] and len(rows()) == 2 * n
    w.undo.undo()
    assert len(s._m.exclusions) == 1
    w.undo.redo()
    # a fit: its curve on the trace, its words and residuals on the Fit tab
    s.fit_region.setCurrentIndex(s.fit_region.findData("response"))
    s.fit_model.setCurrentIndex(s.fit_model.findData("exp1"))
    assert s._result["fit"] is not None and "tau" in s.fit_words.text()
    assert any(isinstance(i, pg.PlotDataItem) for i in s.residuals.getPlotItem().items)
    # the Plot tab: a measure against sweep, time, command, or another measure
    s.plot_x.setCurrentIndex(s.plot_x.findData("sweep"))
    x, y = s.series.getPlotItem().items[0].getData()
    mine = [r for r in rows() if r["region"] == "response"]
    assert sorted(set(x)) == sorted({r["sweep"] + 1 for r in mine}) and len(y) == len(mine)
    assert s.series.getPlotItem().getAxis("bottom").labelText == "sweep"
    # Keep (and Undo), Load, Delete (and Undo)
    s.name_box.setText("light response")
    s.keep_button.click()
    assert [h["name"] for h in w.project.hands(path)] == ["light response"] and s.kept.count() == 1
    w.undo.undo()
    assert w.project.hands(path) == [] and s.kept.count() == 0 and not s.load_kept_button.isEnabled()
    w.undo.redo()
    assert [h["name"] for h in w.project.hands(path)] == ["light response"]
    s.fit_model.setCurrentIndex(0)
    assert s._m.fit_model == ""
    s.load_kept_button.click()
    assert s._m.fit_model == "exp1"                      # as it was when kept
    s.delete_kept_button.click()
    assert w.project.hands(path) == []
    w.undo.undo()
    assert len(w.project.hands(path)) == 1
    # Save as template; Apply here; Run on its protocol's files; Delete
    patched.answers.text = None
    s.template_button.click()
    assert [t.name for t in w.project.templates()] == ["light response"] and s.template_list.count() == 1
    s.fit_model.setCurrentIndex(0)
    s.apply_template_button.click()
    assert s._m.fit_model == "exp1" and s._m.from_template == "light response"
    s.run_template_button.click()
    app.processEvents()
    assert all(any(h["name"] == "light response" for h in w.project.hands(p)) for p in paths.values())
    w.undo.undo()                                        # the batch taken back
    assert w.project.hands(paths["light2.abf"]) == []
    s.delete_template_button.click()
    assert w.project.templates() == [] and not s.apply_template_button.isEnabled()
    assert "No template" in s.apply_template_button.toolTip()
    w.undo.undo()
    assert len(w.project.templates()) == 1
    assert not patched.log.errors, patched.log.errors[-1]


def test_the_figures_screen_does_what_its_labels_say(press):
    patched, open_ = press
    w, paths = open_({"steps.abf": ch.simulated()["cc_steps"](), "steps2.abf": ch.simulated()["cc_steps"]()})
    s = ch.FiguresSession(patched, w, paths["steps.abf"])
    s.reset()
    fig, app = s.s, QtWidgets.QApplication.instance()
    p = fig.panels[1]
    # Show > a graph: that graph, its name in the title, ticked in the menu
    other = next(v for v in p._views if v.key not in {q._key for q in fig.panels})
    p._actions[other.key].trigger()
    assert p._key == other.key and other.title in p.title.text() and p._actions[other.key].isChecked()
    # Edit: the builder, on this graph
    p.edit_button.click()
    b = fig.builder
    assert b is not None and b.isVisible() and b.recipe.source["kind"] == "graph"
    assert b.recipe.source["table"] == other.title and b.recipe.frame() is not None
    b.close()
    # the maximize button fills the area and gives it back; the own-window button opens a window
    p.max_button.click()
    assert fig.maximized is p and [q.isVisibleTo(fig) for q in fig.panels] == [False, True, False, False]
    p.max_button.click()
    assert fig.maximized is None and all(q.isVisibleTo(fig) for q in fig.panels)
    p.detach_requested.emit(p)
    assert len(fig.windows) == 1 and other.title in fig.windows[0].windowTitle()
    fig.windows[0].close()
    # Across recordings: every analyzed recording, one point each; back again
    fig.across_button.click()
    assert fig.stack.currentWidget() is fig.across and "2 recording(s)" in fig.across.readout.text()
    fig.across.y.setCurrentIndex(2)
    assert fig.across.plot.getPlotItem().getAxis("left").labelText == fig.across.y.currentText()
    fig.across.edit_button.click()
    assert fig.builder is not None and " vs. " in fig.builder.recipe.source["table"]
    fig.builder.close()
    fig.across_button.click()
    assert fig.stack.currentIndex() == 1
    # Make a figure: the builder, empty
    fig.builder = None
    w.findChild(QtWidgets.QPushButton, "")              # (the action buttons have no object names)
    next(b_ for b_ in fig.findChildren(QtWidgets.QPushButton) if b_.text() == "Make a figure").click()
    assert fig.builder is not None and fig.builder.isVisible() and fig.builder.recipe.frame() is None
    fig.builder.close()
    app.processEvents()
    assert not patched.log.errors, patched.log.errors[-1]
