"""Every button, checkbox, pick list and menu action of the Workbench, the Figures screen and the figure builder
is pressed, on a simulated recording of every recording type and, when the lab's own recordings are named on this
machine (``tests/labdata.py``), on copies of them (nothing is ever analyzed in place). In the builder the figure itself is a control too: each kind of thing
drawn on it is clicked, dragged, double-clicked, moved with an arrow key, deleted and right-clicked. A control
that raises is broken; so is one that changes nothing, an edit that Undo does not put back, and a disabled control
that does not say why. Then, control by control, that each does what its label says.

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

from labdata import lab_file, lab_files  # noqa: E402

# the lab's own recordings, named on this machine only (tests/labdata.py): never in the repository
HIS = {"full field": lab_file("light, whole field"), "step family": lab_file("step family"),
       "unknown": lab_file("unknown kind")}
needs_daniel = pytest.mark.skipif(not lab_files("light, whole field", "step family", "unknown kind"),
                                  reason="the lab's own recordings are not named on this machine (tests/labdata.py)")
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


def table(folder: Path, third: bool = False) -> Path:
    """Cells in animals in two conditions (three with ``third``) over time: a table every question of the builder
    can draw."""
    rng = np.random.default_rng(0)
    groups = (("control", 0.0), ("drug", 3.0)) + ((("vehicle", 0.5),) if third else ())
    rows = [{"Condition": g, "Animal": f"{g}-A{a}", "Cell": f"{g}-A{a}-C{c}", "Time (min)": float(t),
             "Amplitude (pA)": -20 + shift + rng.normal(0, 2), "Rise (ms)": 1 + rng.random()}
            for g, shift in groups for a in range(3) for c in range(3) for t in range(8)]
    path = folder / ("cells3.csv" if third else "cells.csv")
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
    if fig.builder is not None:                # each state in a builder of its own (Figures keeps one open)
        fig.builder.close()
        fig.builder.deleteLater()
        fig.builder = None
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


def whole_figure(b, paths, folder: Path) -> None:
    """A figure with one of everything a person can select on it: traces of a light recording (the light's bars,
    scale bars, the command under them, the potential at each trace's start), a comparison with its bracket and a
    reference line, a time course with its legend, a mark along x and a text; a title, letters, a text on the
    page."""
    from clampwise.figure import traces

    b.open_file(str(table(folder)))
    r = b.recipe
    r.stats, r.title = True, "Amplitude"
    r.brackets = [{"a": "control", "b": "drug", "y": -14.6}]
    r.excluded = [1] + list(range(4, 40))                # (rows of every cell stay: 8 a cell)
    r.letter_offset = [1.0, 0.5]
    r.ref_lines = [{"axis": "y", "at": -21.0, "label": "baseline"}]
    b.current = b._target(None)
    b._load_controls()
    b.open_file(str(table(folder, third=True)))
    b.set_plot("time_course")
    r = b.recipe
    r.markers = [{"at": 2.0, "end": 4.0, "label": "drug"}]
    r.annotations = [{"kind": "text", "x": 5.0, "y": -19.0, "text": "washout"}]
    r.series = {"drug": {"colour": "#AA3377"}, "vehicle": {"hidden": True}}
    b.current = b._target(None)
    b._load_controls()
    b.set_traces([traces.TraceSet("0 pA", "average", [[str(paths["light.abf"]), 5, 1]]),
                  traces.TraceSet("-80 pA", "average", [[str(paths["light.abf"]), 3, 1]])],
                 traces.Window(20, 160, True, 0.0), "light.abf")
    b.recipe.command = {**b.recipe.command, "show": True}
    b.recipe.scale_bar = {"at": [0.9, 0.12]}
    b.spec.texts = [{"x": 60.0, "y": 1.5, "text": "n = 18 cells"}]
    b.spec.columns, b.spec.size = 3, "double"
    b._fill_sizes()
    b._changed()
    b.redraw_now()


EVERY_KIND = {"panel", "letter", "title", "xlabel", "ylabel", "xaxis", "yaxis", "legend", "scalebar", "stimbar",
              "command", "annotation", "bracket", "series", "refline", "marker", "vmlabel", "figtext"}


def busy_builder(b, group: int = 0) -> None:
    """Something for every control to act on: the middle panel the one being changed (it can move either way), a
    change to undo and one to redo, a row selected in each list (``group``: which group in the groups' list: the
    first can go down and be hidden, the last, hidden in this figure, can go up and be shown)."""
    b.current = 1
    b._load_controls()
    for title in ("one", "two"):
        b.recipe.title = title
        b._snapshot()
    b.undo()
    for name in ("series_table", "ref_lines", "marks", "notes", "filters"):
        tbl = getattr(getattr(b, name, None), "table", None)
        if tbl is not None and tbl.rowCount():
            tbl.selectRow(min(group, tbl.rowCount() - 1) if name == "series_table" else tbl.rowCount() - 1)


def test_every_builder_control_in_every_state(press, tmp_path):
    from clampwise.figure.elements import Tag

    patched, open_ = press
    sims = ch.simulated()
    w, paths = open_({"evoked.abf": sims["vc_evoked"](), "light.abf": sims["cc_evoked"](),
                      "steps.abf": sims["cc_steps"](), "minis.abf": sims["vc_spontaneous"](), "dsi.abf": sims["dsi"]()})
    seen: set[str] = set()                     # a control pressed with an effect in an earlier state is not again

    def new(c) -> bool:
        return ch.doc_name(c) + c.choice not in seen

    def note(outs) -> None:
        fine(outs)
        seen.update(ch.doc_name(o.control) + o.control.choice for o in outs if o.effects)

    for state in BUILDER_STATES:
        b = builder_in(state, w, paths, tmp_path, patched.answers)
        note(ch.BuilderSession(patched, b, side=2).run(state, only=new))
        b.close()
    # the starters of the other kinds of recording, each with a recording of its kind selected
    for name in ("steps.abf", "minis.abf", "dsi.abf"):
        b = builder_in("empty", w, {name: paths[name]}, tmp_path, patched.answers)
        note(ch.BuilderSession(patched, b).run(f"{name[:-4]} selected",
                                               only=lambda c: c.label.startswith("Starters > ") and new(c)))
        b.close()
    # a figure with one of everything: each thing selected (its own settings beside the figure), then the
    # figure itself worked on with the mouse and the keys
    b = builder_in("empty", w, paths, tmp_path, patched.answers)
    whole_figure(b, paths, tmp_path)
    s = ch.BuilderSession(patched, b)
    things = s.things()
    assert set(things) >= EVERY_KIND, f"not on the figure: {sorted(EVERY_KIND - set(things))}"
    note(ch.BuilderSession(patched, b, prepare=busy_builder, side=2).run(
        "a whole figure, in use", only=lambda c: new(c) or c.kind == "right-click"))
    # the panels' tabs, right-clicked on the second: it can move earlier
    busy = ch.BuilderSession(patched, b, prepare=busy_builder, side=2)
    busy.reset()
    patched.answers.pick_action = None
    b.tabs.customContextMenuRequested.emit(b.tabs.tabRect(1).center())
    entry = next(i for i, e in enumerate(patched.log.menus[-1]) if e[0] == "Move it earlier")

    def earlier() -> None:
        patched.answers.pick_action = entry
        b.tabs.customContextMenuRequested.emit(b.tabs.tabRect(1).center())
        patched.answers.pick_action = None

    where = "Figure builder / All settings"
    note([busy.act(ch.Control(f"{where}|right-click action|tabs earlier", "right-click action",
                              "The panel the controls below change > Move it earlier", where),
                   "a whole figure, in use", earlier, ignore=("menus",))])
    note(ch.BuilderSession(patched, b, prepare=lambda b: busy_builder(b, 2), side=2).run(
        "a whole figure, its hidden group selected", only=lambda c: c.where.endswith(" / Groups") and new(c)))
    for kind, tag in things.items():
        note(ch.BuilderSession(patched, b, select=tag).run(
            f"selected: the {kind}", only=lambda c: c.where.endswith(" / Selected") and new(c)))

    def rows_picked(b) -> None:                # a row drawn and a row left out, both selected
        m = b.dataview.table.selectionModel()
        for row in (0, 1):
            m.select(b.dataview.model.index(row, 0), QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    note(ch.BuilderSession(patched, b, select=Tag("panel", 0), side=1, prepare=rows_picked).run(
        "the data behind a panel", only=lambda c: c.where.endswith(" / Data and statistics")))
    done: set[str] = set()
    outs = s.gestures("a whole figure", done)
    note(outs)
    assert done >= EVERY_KIND
    by = {(o.control.kind, o.control.label): o for o in outs}
    for kind in ("letter", "legend", "scalebar", "stimbar", "annotation", "bracket", "figtext"):
        words = ch.KIND_WORDS[kind]
        assert "what is edited changed" in by[("drag", words)].effects, f"dragging the {kind} moved nothing"
        assert "what is edited changed" in by[("arrow key", words)].effects, f"an arrow key did not move the {kind}"
    for kind in ("letter", "title", "xlabel", "ylabel", "annotation", "bracket", "stimbar", "vmlabel", "figtext",
                 "series", "refline", "marker"):
        o = by[("double-click", ch.KIND_WORDS[kind])]
        assert "what is edited changed" in o.effects and o.undo == "restored", f"the {kind} was not rewritten in place"
    assert "asked in a dialog" in by[("double-click", "Panel")].effects          # how it is drawn
    # the panels placed freely: dragged, moved by a key and resized by each handle; an inset the same
    b.all_free()
    for r, (dx, dy, dw, dh) in zip(b.spec.panels, ((0, 0, 0, -6), (4, 4, -7, -12), (-3, 9, -2, -16))):
        x, y, wide, high = r.rect                  # no two alike, so aligning and spacing have something to do
        r.rect = [x + dx, y + dy, wide + dw, high + dh]
    b._changed()
    b.redraw_now()
    s = ch.BuilderSession(patched, b)
    outs = s.gestures("panels placed freely", EVERY_KIND - {"panel"})
    note(outs)
    by = {(o.control.kind, o.control.label): o for o in outs}
    assert "what is edited changed" in by[("drag", "Panel")].effects
    assert "what is edited changed" in by[("arrow key", "Panel")].effects
    outs = s.resize_handles("panels placed freely", things["panel"])
    note(outs)
    assert all("what is edited changed" in o.effects and o.undo == "restored" for o in outs)
    note(s.run("panels placed freely", only=lambda c: c.label.startswith("Arrange > ") and new(c)))
    b.all_grid()
    b.add_inset(0)
    inset = Tag("panel", len(b.spec.panels) - 1)
    s = ch.BuilderSession(patched, b)
    outs = s.resize_handles("an inset", inset)
    note(outs)
    assert all("what is edited changed" in o.effects for o in outs)
    note(ch.BuilderSession(patched, b, select=inset).run(
        "selected: an inset", only=lambda c: c.where.endswith(" / Selected") and new(c)))
    b.close()
    assert len(seen) > 300


def test_every_control_of_the_builders_choosers(press, tmp_path):
    """Add panel's two questions (what data, drawn how) and the export's dialog."""
    from clampwise.app.addpanel import DataChooser, PlotChooser
    from clampwise.app.builder import ExportDialog
    from clampwise.figure import render

    patched, open_ = press
    w, paths = open_({"evoked.abf": ch.simulated()["vc_evoked"]()})
    w.select(paths["evoked.abf"])
    b = w.screens["figures"].open_builder()
    b.open_file(str(table(tmp_path)))
    b.redraw_now()

    def file_picked(d) -> None:
        d.list.setCurrentRow(next(i for i in range(d.list.count())
                                  if d.list.item(i).data(QtCore.Qt.UserRole) == "file"))

    fine(ch.DialogSession(patched, lambda: DataChooser(b, *b._sources()), file_picked).run("data chooser"))
    fine(ch.DialogSession(patched, lambda: PlotChooser(b, "points")).run("plot chooser"))
    fine(ch.DialogSession(patched, lambda: ExportDialog(b, b.spec, render.check(b.spec))).run("export dialog"))
    b.close()


def test_every_control_of_the_trace_picker(press, tmp_path):
    from clampwise.app.tracepicker import TracePicker

    patched, open_ = press
    w, paths = open_({"evoked.abf": ch.simulated()["vc_evoked"]()})
    w.select(paths["evoked.abf"])
    b = w.screens["figures"].open_builder()
    fine(ch.DialogSession(patched, lambda: TracePicker(b, w, b._load), ch.picker_ready).run("trace picker"))
    fine(ch.DialogSession(patched, lambda: TracePicker(b, w, b._load), ch.picker_first_trace).run(
        "trace picker, its first trace selected", only=lambda c: c.label in ("Down", "Remove selected sweeps from it")))
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


# ---------------------------------------------------------------- on the lab's own files (copies)

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
        assert all(sum(a.isChecked() for acts in p._actions.values() for a in acts) == 1
                   for p in shown)                       # the one shown is ticked
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
    # Sweep > and < step through the sweeps: the one stepped to is drawn in bold, named, its row selected
    assert not s.alone.isEnabled() and "Step to a sweep first" in s.alone.toolTip()
    s.next_button.click()
    assert s._browse == 0 and s.browse_label.text() == f"sweep 1 (1 of {n})"
    bold = [c for c in curves() if c.opts["pen"].widthF() == 2 and c.name()]      # (the fit's curve has no name)
    assert len(bold) == 1 and np.allclose(bold[0].getData()[1], s._result["traces"][0][2], equal_nan=True)
    assert {i.row() for i in s.table.selectedIndexes()} == {0}
    s.next_button.click()
    s.prev_button.click()
    s.prev_button.click()                                # round to the last
    assert s._browse == n - 1
    drawn = len(curves())
    s.alone.click()                                      # the others are not drawn
    assert len(curves()) == drawn - (n - 1)
    s.table.clicked.emit(s.rows_model.index(4, 0))       # a row clicked steps to its sweep
    assert s._browse == rows()[4]["sweep"]
    s.all_button.click()
    assert s._browse is None and s.browse_label.text() == "" and not s.alone.isChecked() and len(curves()) == drawn
    s.keys["PgDown"].activated.emit()                    # Page Down does what Sweep > does
    assert s._browse == 0
    # Cursors: two lines over the trace; the readout gives each one's time and value, their difference, the slope
    assert not s.cursors_to_region.isEnabled() and "Tick Cursors" in s.cursors_to_region.toolTip()
    s.cursors_box.click()
    assert len(s._cursor_items) == 2 and "Δt" in s.cursor_label.text()
    s.set_cursor(0, on - 0.010)
    s.set_cursor(1, on + 0.012)
    c = s.cursor_values()
    y0 = wb.raw(rec, 0)
    fs = rec.sample_rate_hz
    assert c["dt_ms"] == pytest.approx(22.0, abs=0.1) and c["trace"] == s._result["traces"][0][0]
    assert c["v1"] == pytest.approx(float(y0[int(round((on - 0.010) * fs))]), abs=1e-6)
    assert c["dv"] == pytest.approx(float(y0[int(round((on + 0.012) * fs))] - y0[int(round((on - 0.010) * fs))]), abs=1e-6)
    assert c["slope"] == pytest.approx(c["dv"] / c["dt_ms"]) and f"{c['dt_ms']:.2f} ms" in s.cursor_label.text()
    s._cursor_items[1].setValue(on + 0.030)              # dragged: the readout follows
    assert s.cursor_values()["dt_ms"] == pytest.approx(40.0, abs=0.1)
    # Region to the cursors: the selected region spans them (kept from the pulse when it repeats on each pulse)
    s.region_list.setCurrentRow(1)
    was = (s._m.regions[1].start_s, s._m.regions[1].end_s)
    s.cursors_to_region.click()
    assert (s._m.regions[1].start_s, s._m.regions[1].end_s) == pytest.approx((-0.010, 0.030), abs=1e-4)
    w.undo.undo()
    assert (s._m.regions[1].start_s, s._m.regions[1].end_s) == pytest.approx(was)
    # the arrow keys move the selected region 0.1 ms (Shift: 1 ms; Alt: its end alone); Undo takes each step back
    s.keys["Right"].activated.emit()
    assert (s._m.regions[1].start_s, s._m.regions[1].end_s) == pytest.approx((was[0] + 1e-4, was[1] + 1e-4))
    s.keys["Shift+Left"].activated.emit()
    assert s._m.regions[1].start_s == pytest.approx(was[0] - 9e-4)
    s.keys["Alt+Right"].activated.emit()
    assert s._m.regions[1].start_s == pytest.approx(was[0] - 9e-4) and s._m.regions[1].end_s == pytest.approx(was[1] - 8e-4)
    for _ in range(3):
        w.undo.undo()
    assert (s._m.regions[1].start_s, s._m.regions[1].end_s) == pytest.approx(was)
    # Send to a figure: the traces in view as example traces with the light's bars; the table as a graph
    s.all_button.click()
    s.plot.setXRange(on - 0.020, on + 0.150, padding=0)
    app.processEvents()
    s.send_traces_action.trigger()
    b = w.screens["figures"].builder
    assert b is not None and b.isVisible()
    r = b.spec.panels[0]
    assert (r.question, r.plot, r.source["kind"]) == ("example", "trace", "traces") and r.scale_bars
    f = r.frame()
    assert f["Sweep"].nunique() == n and f[r.x].min() == pytest.approx(0.0, abs=0.2) and \
        f[r.x].max() == pytest.approx(170.0, abs=1.0)
    assert [bar["start"] for bar in r.stim_bars] == pytest.approx([20.0, 125.0], abs=0.6)     # where the light came
    assert "sent to the figure as panel" in b.statusBar().currentMessage() and b.canvas.selection.kind == "panel"
    s.send_table_action.trigger()
    assert len(b.spec.panels) == 2 and b.current == 1
    r = b.spec.panels[1]
    assert r.source["kind"] == "workbench" and r.x == "Sweep" and len(r.frame()) == len(rows())
    assert r.y.startswith("Peak") and "Measured by hand in the Workbench on light.abf" in " ".join(r.notes)
    b.close()
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
    p._actions[other.key][0].trigger()
    assert p._key == other.key and other.title in p.title.text() and p._actions[other.key][0].isChecked()
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


@pytest.mark.skipif(not lab_files("light, region 1", "light, region 2"),
                    reason="the lab's own recordings are not named on this machine (tests/labdata.py)")
def test_a_real_light_file_measured_by_hand_as_in_clampfit(press, tmp_path):
    """A real light file (a copy) worked through as Daniel would in Clampfit, by the screen's own controls: the
    amplitude, latency and decay of the response to every pulse, a sweep left out with its reason, the decay
    fitted, the amplitude plotted against the sweep, the measurement kept, saved as a template and run on the
    cell's other files of the protocol; every step taken back by Undo."""
    from clampwise.analysis import workbench as wb

    patched, open_ = press
    folder = tmp_path / "A1" / "Slice1" / "cell1"
    folder.mkdir(parents=True)
    originals = (lab_file("light, region 1"), lab_file("light, region 2"))
    names = tuple(p.name for p in originals)
    for p in originals:
        shutil.copy2(p, folder / p.name)
    w, paths = ch.open_window(folder, {name: folder / name for name in names}, analyze=False)
    try:
        path = paths[names[0]]
        s = ch.WorkbenchSession(patched, w, path).s
        rec = s._recording
        n = rec.n_sweeps
        assert n >= 4 and [r.name for r in s._m.regions] == ["baseline", "response"]      # set from the light

        def tick(key: str) -> None:
            item = next(s.measure_list.item(k) for k in range(s.measure_list.count())
                        if s.measure_list.item(k).data(QtCore.Qt.UserRole) == key)
            item.setCheckState(QtCore.Qt.Checked)

        for key in ("peak", "onset_latency", "decay_time"):
            tick(key)
        rows = s._result["rows"]
        assert len(rows) == 2 * n and {r["pulse"] for r in rows} == {1, 2}                # every pulse of every sweep
        assert all(np.isfinite(r["peak"]) for r in rows)
        m = s.table.model()
        heads = [m.headerData(c, QtCore.Qt.Horizontal) for c in range(m.columnCount())]
        assert {"Peak (mV)", "Onset latency (ms)", "Decay time (ms)"} <= set(heads)
        # a sweep left out, with why
        on = float(rec.stimulus.times_s[3][0])
        s.click_buttons["exclude"][0].click()
        patched.answers.text = "spontaneous event on the pulse"
        y = float(wb.raw(rec, 3)[int((on + 0.01) * rec.sample_rate_hz)])
        vb = s.plot.getPlotItem().getViewBox()
        s._clicked(type("Click", (), {"scenePos": lambda self: vb.mapViewToScene(QtCore.QPointF(on + 0.01, y))})())
        patched.answers.text = None
        s.click_buttons["exclude"][0].click()
        assert [(e.sweep, e.reason) for e in s._m.exclusions] == [(3, "spontaneous event on the pulse")]
        assert len(s._result["rows"]) == 2 * (n - 1)
        # the decay fitted on the average: the mode, a fit region added over the response, the model chosen
        s.mode.setCurrentIndex(s.mode.findData("average"))
        s.region_list.setCurrentRow(1)
        s.fit_region.setCurrentIndex(s.fit_region.findData("response"))
        s.fit_model.setCurrentIndex(s.fit_model.findData("exp1"))
        assert s._result["fit"] is not None and s.fit_words.text() and "No fit yet" not in s.fit_words.text()
        s.mode.setCurrentIndex(s.mode.findData("each"))
        # the amplitude against the sweep
        s.tabs.setCurrentIndex(1)
        s.plot_y.setCurrentIndex(s.plot_y.findData("peak"))
        s.plot_x.setCurrentIndex(s.plot_x.findData("sweep"))
        x, yv = s.series.getPlotItem().items[0].getData()
        assert len(x) == 2 * (n - 1) and 4 not in set(x)                  # sweep 4 (from 1) is the one left out
        # kept, saved as a template, run on the protocol's other file
        s.name_box.setText("ROI responses")
        s.keep_button.click()
        assert [h["name"] for h in w.project.hands(path)] == ["ROI responses"]
        s.template_button.click()
        assert [t_.name for t_ in w.project.templates()] == ["ROI responses"]
        s.run_template_button.click()
        QtWidgets.QApplication.instance().processEvents()
        other = w.project.hands(paths[names[1]])
        assert [h["name"] for h in other] == ["ROI responses"] and other[0]["summary"]
        # every step comes back off, last first
        steps = 0
        while w.undo.canUndo() and steps < 60:
            w.undo.undo()
            steps += 1
        assert steps >= 8 and w.project.hands(path) == [] and w.project.hands(paths[names[1]]) == []
        assert w.project.templates() == [] and s._m.exclusions == [] and s._m.fit_model == ""
        assert all((folder / p.name).read_bytes() == p.read_bytes() for p in originals)
    finally:
        w.close()
        w.deleteLater()
        ch.flush(QtWidgets.QApplication.instance())
