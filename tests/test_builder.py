"""The figure builder's window (B12): question first, a CSV opened, statistics
said in the caption, a bar starting at zero, undo and redo, two panels, an
export that reopens editable, and the Methods text."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from pyqtgraph.Qt import QtWidgets  # noqa: E402

from test_figure import nested  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def builder(qapp):
    from clampwise.app import theme
    from clampwise.app.window import MainWindow

    theme.apply(qapp, theme.LIGHT)
    w = MainWindow(theme.LIGHT, persist=False)
    b = w.screens["figures"].open_builder()
    yield b
    b.close()
    w.close()


def _choose(b, key):
    from clampwise.figure.recipe import QUESTIONS

    i = [q[0] for q in QUESTIONS].index(key)
    b.question_box.setCurrentIndex(i)
    b._question_chosen(i)
    b.redraw()


def test_question_first_a_csv_and_the_caption(builder, tmp_path):
    b = builder
    nested(effect=3.0, animals=5, cells=3, seed=4).to_csv(tmp_path / "cells.csv", index=False)
    b.open_file(str(tmp_path / "cells.csv"))
    r = b.recipe
    assert (r.cell, r.animal, r.group, r.y) == ("Cell", "Animal", "Condition", "Amplitude (pA)")   # guessed
    _choose(b, "compare")
    assert not b.fields["paired"].isHidden() and b.fields["bin"].isHidden()          # only what it needs
    assert "n = 15 cells from 5 animals" in b.caption_text()
    b.stats.setChecked(True)
    b.redraw()
    assert "Welch's t test" in b.caption_text() and "95% CI" in b.caption_text()
    b.plot.setCurrentIndex(b.plot.findData("bar"))
    assert b.recipe.plot == "bar" and b.recipe.y_from_zero is True
    b.undo()
    assert b.recipe.plot == "points"
    b.redo()
    assert b.recipe.plot == "bar"
    b.level.setCurrentIndex(b.level.findData("row"))
    b.redraw()
    assert any("pseudoreplication" in w.text() for w in b.warnings.findChildren(QtWidgets.QLabel))


def test_two_panels_export_reopen_and_methods(builder, tmp_path):
    b = builder
    nested().to_csv(tmp_path / "cells.csv", index=False)
    b.open_file(str(tmp_path / "cells.csv"))
    _choose(b, "compare")
    b.set_panels(2)
    b.tabs.setCurrentIndex(1)
    _choose(b, "distribution")
    assert b.spec.panels[1].question == "distribution" and b.spec.panels[0].question == "compare"
    out = b.export(str(tmp_path / "fig.pdf"))
    before = b.spec.to_json()
    b.set_panels(1)
    b.open_figure(str(out))
    assert b.spec.to_json() == before and len(b.spec.panels) == 2
    b.redraw()
    text = b.methods_text()
    assert "(A)" in text and "(B)" in text and "cells.csv" in text


def test_the_digitizer_dialog_calibrates_and_traces(builder, tmp_path):
    import numpy as np
    from matplotlib.figure import Figure

    from clampwise.app.builder import DigitizeDialog

    fig = Figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot()
    x = np.linspace(0, 10, 200)
    ax.plot(x, 2 * x + 1, color="#d62728")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 25)
    fig.savefig(tmp_path / "p.png", dpi=100)
    h = 300
    (a0, b0), (a1, b1) = ax.transData.transform([(0, 0), (10, 25)])
    dlg = DigitizeDialog(builder, tmp_path / "p.png")
    for col, row, v in ((a0, h - b0, 0.0), (a1, h - b0, 10.0), (a0, h - b0, 0.0), (a0, h - b1, 25.0)):
        dlg.clicked(col, row, v)
    dlg.mode = "trace"
    c, r = ax.transData.transform([(5, 11)])[0]
    dlg.clicked(c, h - r)
    t = dlg.table()
    mid = (t.x > 1) & (t.x < 9)
    assert mid.sum() > 100 and np.abs(t.y[mid] - (2 * t.x[mid] + 1)).max() < 0.5


def test_the_groups_look_a_theme_and_any_plot_type(builder, tmp_path):
    b = builder
    nested().to_csv(tmp_path / "cells.csv", index=False)
    b.open_file(str(tmp_path / "cells.csv"))
    _choose(b, "compare")
    assert [b.series_table.table.item(i, 0).text() for i in range(b.series_table.table.rowCount())] == \
        ["control", "drug"]
    b.series_table.table.cellWidget(0, 2).set_colour("#AA0000")      # control red, hollow squares
    b.series_table.table.cellWidget(0, 3).setCurrentIndex(b.series_table.table.cellWidget(0, 3).findData("s"))
    b.series_table.table.cellWidget(0, 6).setChecked(False)
    b.series_table.table.selectRow(1)
    b.series_table._move(-1)                                          # drug first
    r = b.recipe
    assert r.series["control"]["colour"] == "#AA0000" and r.series["control"]["marker"] == "s"
    assert r.series["control"]["filled"] is False and r.order == ["drug", "control"]
    b.theme_box.setCurrentText("Bold presentation")
    b._theme_chosen()
    assert b.spec.style.font_size == 14.0 and b.spec.style.theme == "Bold presentation"
    assert r.series["control"]["colour"] == "#AA0000"                 # what was set stays over a new theme
    b.plot.setCurrentIndex(b.plot.findData("density"))                # a plot of another question: it follows
    assert b.recipe.question == "distribution" and b.recipe.plot == "density"
    b.undo()
    assert b.recipe.question == "compare"
    b.redraw()
    out = b.export(str(tmp_path / "fig.svg"))
    from clampwise.figure import render

    back = render.read_spec(out)
    assert back.style.theme == "Bold presentation" and back.panels[0].series["control"]["marker"] == "s"


def test_a_figures_graph_opens_as_an_editable_figure(builder):
    """Any graph of the Figures screen opens in the builder: its series, their looks, traces drawn one by one,
    error bars on their series, bars, reference lines and axis titles."""
    import numpy as np
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from clampwise.app import editable, theme
    from clampwise.app.panels import Panel, View
    from clampwise.figure import render

    x = np.arange(5.0)

    def draw(item, pal) -> str:
        for k in range(3):                                         # unnamed traces drawn alike: one group
            item.plot(x, x * k, pen=pg.mkPen("#888888", width=0.8))
        item.addItem(pg.ErrorBarItem(x=x, y=x + 1, height=np.full(5, 0.4), pen=pg.mkPen("#0072B2")))
        item.plot(x, x + 1, pen=pg.mkPen("#0072B2", width=2), symbol="o", symbolBrush="#0072B2", name="mean")
        item.plot(x, 2 * x, pen=pg.mkPen("#D55E00", style=QtCore.Qt.DashLine), name="fit")
        item.addItem(pg.BarGraphItem(x0=x, x1=x + 1, height=x, brush="#009E73"))
        item.addItem(pg.InfiniteLine(0, angle=90))
        item.setLabel("bottom", "Time (ms)")
        item.setLabel("left", "pA")
        return ""

    p = Panel(theme.LIGHT)
    p.set_views([View("demo.course", "Amplitude vs. time", "[pA vs. ms]", draw)])
    spec = p.as_figure("cell1.abf")
    r = spec.panels[0]
    f = r.frame()
    assert (r.x, r.y, r.group, r.cell, r.err, r.plot) == ("Time (ms)", "Amplitude (pA)", "Series", "Trace",
                                                          "Error", "line")
    assert set(f.Series) == {"Series 1", "Series 2", "mean", "fit"} and f[f.Series == "Series 1"].Trace.nunique() == 3
    assert np.allclose(f[f.Series == "mean"].Error, 0.2)          # the error bars joined the mean they sit on
    assert r.series["mean"]["marker"] == "o" and r.series["fit"]["line"] == "--"
    assert r.series["Series 1"]["legend"] is False and r.series["Series 2"]["bars"] is True
    assert r.ref_lines == [{"axis": "x", "at": 0.0, "label": ""}]
    render.build(spec)
    b = builder.window_.screens["figures"].edit_graph(p)
    assert b.recipe.source["kind"] == "graph" and "Amplitude vs. time" in b.source.text()
    b.redraw()
    names = [b.series_table.table.item(i, 0).text() for i in range(b.series_table.table.rowCount())]
    assert {"mean", "fit", "Series 1", "Series 2"} <= set(names)
    b.series_table.table.cellWidget(names.index("mean"), 2).set_colour("#AA0000")    # a look changed by hand
    b.series_table.table.cellWidget(names.index("mean"), 5).setValue(2.5)
    assert b.recipe.series["mean"]["colour"] == "#AA0000" and b.recipe.series["Series 2"]["bars"] is True
    b.close()
    assert editable.from_plot(pg.PlotItem(), "nothing") is None


def test_the_builder_layout_sections_zoom_and_panels(builder, tmp_path):
    b = builder
    nested().to_csv(tmp_path / "cells.csv", index=False)
    b.open_file(str(tmp_path / "cells.csv"))
    _choose(b, "compare")
    heads = {h.text(): h for h, _ in b._sections}
    assert not heads["COLUMNS"].isHidden() and heads["AVERAGING"].isVisibleTo(b) is not None
    assert b.fields["err"].isHidden() and b.tabs.count() == 1 and b.tabs.tabText(0) == "Panel A"
    b.add_panel.click()
    assert len(b.spec.panels) == 2 and b.tabs.count() == 2
    b.zoom.setCurrentIndex(b.zoom.findData(1.0))                  # printed size: 85 mm at the screen's dpi
    w_in = b.fig.get_size_inches()[0]
    assert abs(b.canvas.width() - w_in * b.logicalDpiX()) <= 2
    assert "85 × " in b.size_note.text() and "2 panels" in b.size_note.text()


def test_raw_traces_in_a_panel_and_applying_changes(builder):
    """Trace sets drawn in a panel (kept in its source so the picker reopens on them); a typed title applies a
    moment after typing stops, and Apply applies at once."""
    from clampwise import simulate
    from clampwise.figure import traces

    sim = simulate.dsi_evoked(durations=(1.0,), trials=2)
    b = builder
    b._load = lambda _f: sim.recording
    sweeps = [k for k in range(sim.recording.n_sweeps) if traces.stimulus_s(sim.recording, k, 1) == \
              traces.stimulus_s(sim.recording, k, 1)]
    sets = [traces.TraceSet("Early", "average", [["", k, 1] for k in sweeps[:3]]),
            traces.TraceSet("Late", "each", [["", k, 1] for k in sweeps[-3:]])]
    b.set_traces(sets, traces.Window(5, 30, True, 1.0), "sim")
    r = b.recipe
    assert (r.question, r.plot, r.group, r.cell, r.source["kind"]) == ("example", "trace", "Trace", "Sweep", "traces")
    n = len(sets[1].items)
    assert not b.traces_box.isHidden() and "Early" in b.traces_said.text() and f"{n} sweeps" in b.traces_said.text()
    b.redraw()
    assert b._drawn[0].prepared is not None and f"Late: {n} sweeps, each drawn" in b.caption_text()
    b.sections.setCurrentIndex(3)                                   # Axes
    b.title_edit.setText("Before and after")
    b.title_edit.textEdited.emit("Before and after")               # as typed: applied when typing stops
    assert b.typing.isActive() and b.recipe.title == ""
    b.apply()
    assert b.recipe.title == "Before and after" and not b.typing.isActive()


def test_everything_added_can_be_removed(builder, tmp_path, monkeypatch):
    """Panels, reference lines, marks, text, filters, groups, saved styles and a panel's data: each has a button
    (and a right-click menu) that removes what was added."""
    from pyqtgraph.Qt import QtCore, QtWidgets

    b = builder
    nested().to_csv(tmp_path / "cells.csv", index=False)
    b.open_file(str(tmp_path / "cells.csv"))
    _choose(b, "compare")
    # panels: add, duplicate, move, remove (button, tab's x), undo brings one back
    b.add_panel.click()
    b.duplicate_panel(0)
    assert len(b.spec.panels) == 3 and b.tabs.tabsClosable() and b.remove_panel_button.isEnabled()
    b.spec.panels[2].title = "third"
    b.move_panel(2, -1)
    assert b.spec.panels[1].title == "third" and b.current == 1
    b.remove_panel_button.click()
    assert len(b.spec.panels) == 2 and all(p.title != "third" for p in b.spec.panels)
    b.tabs.tabCloseRequested.emit(1)
    assert len(b.spec.panels) == 1 and not b.remove_panel_button.isEnabled() and not b.tabs.tabsClosable()
    b.undo()
    assert len(b.spec.panels) == 2
    b.set_panels(1)
    m = QtWidgets.QMenu()
    b._panel_actions(m, 0)
    assert any(a.text() == "Remove panel A" and not a.isEnabled() for a in m.actions())
    b.redraw()
    m = QtWidgets.QMenu()
    b._canvas_menu(m, QtCore.QPoint(b.canvas.width() // 2, b.canvas.height() // 2))
    assert {"Copy the figure as a picture", "Export..."} <= {a.text() for a in m.actions()}
    # reference lines, marks, text: + Add, Remove selected, Remove every
    for table, field in ((b.ref_lines, "ref_lines"), (b.marks, "markers"), (b.notes, "annotations")):
        table.add()
        table.add()
        assert len(getattr(b.recipe, field)) == 2
        table.table.selectRow(0)
        table.remove_selected()
        assert len(getattr(b.recipe, field)) == 1
        table.clear()
        assert getattr(b.recipe, field) == [] and not table.remove_button.isEnabled()
    b.ref_lines.add()
    assert b.recipe.ref_lines == [{"axis": "y", "at": 0.0, "label": ""}]
    b.redraw()
    # filters: add, and remove every condition
    b.filters.add("Condition", "is", "control")
    assert len(b.recipe.filters) == 1
    b.filters.clear()
    assert b.recipe.filters == []
    # a group hidden from the figure and shown again
    b.series_table.table.selectRow(0)
    b.series_table._show(False)
    hidden = [g for g, d in b.recipe.series.items() if d.get("hidden")]
    assert len(hidden) == 1
    b.series_table._show(True)
    assert not any(d.get("hidden") for d in b.recipe.series.values())
    # a saved style deleted
    import json

    store = {"mine": {}}

    class Settings:
        def setValue(self, _key, text):
            store.clear()
            store.update(json.loads(text))

    monkeypatch.setattr(b, "_styles", lambda: dict(store))
    monkeypatch.setattr(b.window_, "_settings", lambda: Settings())
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.Yes)
    b.delete_style("mine")
    assert "mine" not in store
    # the panel's data removed, its look kept
    b.recipe.series = {"control": {"colour": "#AA0000"}}
    b.clear_data()
    assert b.recipe.frame() is None and b.recipe.series == {"control": {"colour": "#AA0000"}}
    b.undo()
    assert b.recipe.frame() is not None


def test_the_trace_pickers_sweeps_added_to_and_removed_from_a_trace(builder):
    from clampwise.app.tracepicker import TracePicker
    from clampwise.figure import traces

    dlg = TracePicker(builder, builder.window_, lambda _f: None,
                      [traces.TraceSet("Before", "average", [["a.abf", 0, 1], ["a.abf", 1, 1]])])
    dlg.list.selectRow(0)
    dlg._selected_items = lambda: [["a.abf", 2, 1], ["a.abf", 0, 1]]
    dlg._change_items(True)
    assert [i[1] for i in dlg.sets[0].items] == [0, 1, 2]              # the one already in it is not added twice
    dlg._selected_items = lambda: [["a.abf", 0, 1], ["a.abf", 1, 1]]
    dlg._change_items(False)
    assert [i[1] for i in dlg.sets[0].items] == [2]
    dlg._selected_items = lambda: [["a.abf", 2, 1]]
    dlg._change_items(False)                                              # its last sweep gone: the trace goes
    assert dlg.sets == [] and not dlg.ok.isEnabled()
    dlg.close()


def test_the_digitizers_clicks_can_be_taken_back(builder, tmp_path):
    from matplotlib.figure import Figure

    from clampwise.app.builder import DigitizeDialog

    fig = Figure(figsize=(4, 3), dpi=100)
    fig.add_subplot().plot([0, 1], [0, 1])
    fig.savefig(tmp_path / "p.png", dpi=100)
    dlg = DigitizeDialog(builder, tmp_path / "p.png")
    for col, row, v in ((50, 250, 0.0), (350, 250, 1.0), (50, 250, 0.0), (50, 30, 1.0)):
        dlg.clicked(col, row, v)
    dlg.undo_last()                                                      # the last calibration click taken back
    assert len(dlg.clicks) == 3 and dlg.mode == "calibrate"
    dlg.clicked(50, 30, 1.0)
    dlg._mode("points")
    for c, r in ((100, 200), (200, 150), (300, 100)):
        dlg.clicked(c, r)
    dlg.remove_nearest(198, 151)
    assert dlg.points == [(100, 200), (300, 100)] and len(dlg._point_marks) == 2
    dlg.undo_last()
    assert dlg.points == [(100, 200)]
    dlg.clear_points()
    assert dlg.points == [] and not dlg.done.isEnabled() and len(dlg.clicks) == 4
    dlg.restart()
    assert dlg.clicks == [] and dlg.mode == "calibrate"
    dlg.close()


def test_a_preset_fills_the_builder_and_every_panel_stays_editable(builder, tmp_path):
    """Presets list what applies (greyed with what it needs otherwise); one fills every panel; a panel edits,
    undoes, keeps its rise and decay when its traces are picked again, and the figure saves and reopens."""
    from clampwise import simulate
    from clampwise.analysis import pipeline
    from clampwise.figure import presets, traces

    rec = simulate.evoked().recording
    sel = presets.Selection(results=pipeline.analyze(rec).results, load=lambda _f: rec, name="sim.abf")
    b = builder
    b.selection, b._load = (lambda: sel), (lambda _f: rec)
    menu = QtWidgets.QMenu()
    b._layouts_menu(menu)
    acts = {a.text().split(":")[0]: a for a in menu.actions() if not a.isSeparator()}
    assert acts["Evoked PSC"].isEnabled() and not acts["Current-clamp step family"].isEnabled()
    assert "Needs" in acts["Current-clamp step family"].text()
    b.apply_preset("evoked")
    assert len(b.spec.panels) == 3 and b.tabs.count() == 3
    b.tabs.setCurrentIndex(1)
    b.plot.setCurrentIndex(b.plot.findData("line"))
    assert b.spec.panels[1].plot == "line"
    b.undo()
    assert b.spec.panels[1].plot == "scatter"
    b.tabs.setCurrentIndex(0)
    sets, win = traces.from_source(b.recipe.source)
    b.set_traces(sets[:1], win, b.recipe.source["table"])           # picked again in Choose traces
    assert "decay fit" in set(b.recipe.frame().Trace)
    out = b.export(str(tmp_path / "preset.pdf"))
    before = b.spec.to_json()
    b.set_panels(1)
    b.open_figure(str(out))
    assert b.spec.to_json() == before
    for _ in range(20):
        if len(b.spec.panels) != 3:
            break
        b.undo()
    assert len(b.spec.panels) == 1                                  # undo goes back to before the preset
