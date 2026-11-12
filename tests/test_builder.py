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

    from pyqtgraph.Qt import QtCore

    if not qapp.styleSheet():
        theme.apply(qapp, theme.LIGHT)
    w = MainWindow(theme.LIGHT, persist=False)
    b = w.screens["figures"].open_builder()
    yield b
    b.close()
    w.close()
    b.deleteLater()               # a closed window is still styled, each time the theme is set: gone, not kept
    w.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    qapp.processEvents()


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


# ---------------------------------------------------------------- the figure worked on directly

def _centre(canvas, tag):
    from pyqtgraph.Qt import QtCore

    o = canvas.outline(tag)
    assert o is not None, tag
    p = canvas.to_widget((o[0] + o[2]) / 2, (o[1] + o[3]) / 2)
    return QtCore.QPoint(int(p.x()), int(p.y()))


def _on(canvas, tag, part=""):
    """A point on one of the thing's own artists (an L of scale bars has an empty middle)."""
    from pyqtgraph.Qt import QtCore

    from clampwise.figure import render

    a = next(a for t, a in render.things(canvas.figure) if (t.kind, t.panel, t.key) == (tag.kind, tag.panel, tag.key)
             and (not part or t.part == part))
    bb = a.get_window_extent(render._renderer(canvas.figure))
    p = canvas.to_widget((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2)
    return QtCore.QPoint(int(p.x()), int(p.y()))


def _light(b):
    """The builder's first panel showing two traces of a simulated light recording."""
    from clampwise import simulate
    from clampwise.analysis.stimulus import find_stimulus
    from clampwise.figure import traces

    rec = simulate.photostim_steps(seed=3).recording
    rec.stimulus = find_stimulus(rec)
    b._load = lambda _f: rec
    b.set_traces([traces.TraceSet("0 pA", "average", [["", 5, 1]]), traces.TraceSet("-80 pA", "average", [["", 3, 1]])],
                 traces.Window(20, 160, True, 0.0), "light.abf")
    b.redraw_now()
    return rec


def test_a_click_selects_a_drag_moves_and_a_double_click_rewrites(builder, tmp_path):
    import controls_harness as ch
    from pyqtgraph.Qt import QtCore

    from clampwise.figure import render
    from clampwise.figure.elements import Tag

    b = builder
    b.resize(1400, 900)
    nested(effect=3.0).to_csv(tmp_path / "cells.csv", index=False)
    b.open_file(str(tmp_path / "cells.csv"))
    b.recipe.stats = True
    b.touch()
    b.redraw_now()
    c = b.canvas
    # a click on the y axis title selects it; the inspector shows it; the line above the figure says what can be done
    ch.click(c, _centre(c, Tag("ylabel", 0)))
    assert c.selection == Tag("ylabel", 0) and b.inspector.heading.text() == "Y axis title"
    assert "double-click rewrites it" in b.how.text().lower() and b.side.currentIndex() == 0
    # a double-click opens its words in place; Enter keeps them; Undo takes them back
    ch.double_click(c, _centre(c, Tag("ylabel", 0)))
    assert c._editor is not None and c._editor.text() == "Amplitude (pA)"
    c._editor.setText("EPSC amplitude (pA)")
    c._editor.returnPressed.emit()
    assert b.recipe.y_label == "EPSC amplitude (pA)" and c._editor is None
    b.undo()
    assert b.recipe.y_label == ""
    # Esc in the box leaves the words as they were
    c.edit_text(Tag("xlabel", 0))
    c._editor.setText("something else")
    ch.QtWidgets.QApplication.sendEvent(c._editor, ch.QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape,
                                                                      QtCore.Qt.NoModifier))
    assert b.recipe.x_label == "" and c._editor is None
    # the bracket dragged up stays there, and its text can be the person's own result
    auto = next(a for t, a in render.things(b.fig) if (t.kind, t.part) == ("bracket", "line")).get_ydata()[1]
    ch.drag(c, _on(c, Tag("bracket", 0, "control|drug"), "text"), 0, -30)
    b.redraw_now()
    assert b.recipe.brackets[0]["y"] > auto and b.recipe.brackets[0]["a"] == "control"
    b.set_words(Tag("bracket", 0, "control|drug"), "t(22) = 5.1, p < 0.001")
    b.redraw_now()
    assert b.words_of(Tag("bracket", 0, "control|drug")) == "t(22) = 5.1, p < 0.001"
    # Delete hides the selected bracket; a title is cleared; an axis cannot be removed and says so
    c.select(Tag("bracket", 0, "control|drug"))
    ch.QtWidgets.QApplication.sendEvent(c, ch.QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Delete,
                                                              QtCore.Qt.NoModifier))
    b.redraw_now()
    assert b.recipe.brackets[0].get("hidden") and c.selection is None
    b.remove_thing(Tag("yaxis", 0))
    assert "cannot be removed" in b.statusBar().currentMessage()
    # the points of a group: a click selects the group, whose colour then holds in every panel
    dot = next(a for t, a in render.things(b.fig) if t == Tag("series", 0, "drug") and len(a.get_offsets()))
    px = dot.get_offset_transform().transform(dot.get_offsets())[0]
    p = c.to_widget(px[0], px[1])
    ch.click(c, QtCore.QPoint(int(p.x()), int(p.y())))
    assert c.selection == Tag("series", 0, "drug") and "drug" in b.inspector.hint.text()
    b.set_group_colour("drug", "#AA3377")
    b.touch()
    b.redraw_now()
    assert b.look_of(0, "drug").colour == "#AA3377" and b.spec.style.group_colours == {"drug": "#AA3377"}


def test_traces_bars_and_text_are_dragged_where_they_go(builder):
    import controls_harness as ch

    from clampwise.figure.elements import Tag

    b = builder
    b.resize(1400, 900)
    _light(b)
    c = b.canvas
    r = b.recipe
    assert [bar["start"] for bar in r.stim_bars] == [0.0, 105.0] and r.vm_labels == {"0 pA": "-60 mV", "-80 pA": "-68 mV"}
    assert r.scale_bars and r.y2 == "Command (pA)" and "Raw traces: 0 pA (1 sweep)" in b.source_words(0)
    # the scale bars: dragged (by the corner they enclose too), then given a length and a label
    ch.drag(c, _centre(c, Tag("scalebar", 0, "")), -50, -30)
    b.redraw_now()
    at = r.scale_bar["at"]
    assert at[0] < 0.995 and at[1] > 0.02
    c.select(Tag("scalebar", 0, ""))
    assert b.inspector.heading.text() == "Scale bars"
    edit = next(w for w in b.inspector.findChildren(ch.QtWidgets.QLineEdit) if w.accessibleName() == "Time bar (ms)")
    edit.setText("20")
    edit.editingFinished.emit()
    b.redraw_now()
    assert r.scale_bar["x"] == 20.0
    # a stimulus bar: dragged up (never sideways: its time is the recording's), relabelled in place, removed
    y0 = r.stim_bars[0].get("y", 1.02)
    ch.drag(c, _centre(c, Tag("stimbar", 0, "0")), 40, -15)
    b.redraw_now()
    assert r.stim_bars[0]["y"] > y0 and r.stim_bars[0]["start"] == 0.0
    b.set_words(Tag("stimbar", 0, "0"), "470 nm, 5 ms")
    b.set_stim_light(r.stim_bars[1], "630")
    b.match_stim_bars(0, 1)
    assert all(bar["nm"] == 630.0 for bar in r.stim_bars) and r.stim_bars[0]["label"] == "470 nm, 5 ms"
    b.add_stim_bar(0)
    assert len(r.stim_bars) == 3 and c.selection == Tag("stimbar", 0, "2")
    b.remove_thing(c.selection)
    assert len(r.stim_bars) == 2
    # text added in the middle, dragged, rewritten; an arrow's words are what moves
    b.add_annotation(0, "text")
    where = dict(r.annotations[0])
    ch.drag(c, _centre(c, c.selection), 60, -25)
    b.redraw_now()
    assert r.annotations[0]["x"] > where["x"] and r.annotations[0]["y"] > where["y"]
    b.set_words(Tag("annotation", 0, "0"), "EPSP")
    b.add_annotation(0, "arrow")
    tip = (r.annotations[1]["x"], r.annotations[1]["y"])
    ch.drag(c, _centre(c, c.selection), -30, 10)
    b.redraw_now()
    assert (r.annotations[1]["x"], r.annotations[1]["y"]) == tip and r.annotations[0]["text"] == "EPSP"
    # an arrow key nudges the selection half a millimetre; the letter is dragged from its corner and put back
    c.select(Tag("annotation", 0, "0"))
    x = r.annotations[0]["x"]
    b._nudged(c.selection, 0.5, 0.0)
    assert r.annotations[0]["x"] > x
    b.set_panels(2)
    b.redraw_now()
    ch.drag(c, _centre(c, Tag("letter", 0)), 20, 10)
    b.redraw_now()
    assert r.letter_offset[0] > 0 and r.letter_offset[1] > 0
    # the potential at a trace's start rewritten; the labels hidden
    b.set_words(Tag("vmlabel", 0, "0 pA"), "rest")
    assert r.vm_labels["0 pA"] == "rest"
    b.remove_thing(Tag("vmlabel", 0, "0 pA"))
    assert r.show_vm is False


def test_panels_are_arranged_placed_freely_resized_and_inset(builder, tmp_path):
    import controls_harness as ch

    from clampwise.figure import page
    from clampwise.figure.elements import Tag

    b = builder
    b.resize(1400, 900)
    nested().to_csv(tmp_path / "cells.csv", index=False)
    b.open_file(str(tmp_path / "cells.csv"))
    b.use_template("wide_top_3")                                       # three places more: empty panels to fill
    b.redraw_now()
    assert len(b.spec.panels) == 4 and b.spec.columns == 3 and b.spec.size == "double"
    assert [r.span for r in b.spec.panels] == [[3, 1], [1, 1], [1, 1], [1, 1]]
    assert sum(r.frame() is None for r in b.spec.panels) == 3 and "two columns wide" in b.statusBar().currentMessage()
    c = b.canvas
    assert not b._can_resize(Tag("panel", 1))                          # in the grid: its cell sets its size
    # a grid panel dropped on another changes places with it (what each shows, not the cells)
    b.spec.panels[1].title = "second"
    ch.drag(c, _centre(c, Tag("panel", 1)), int(c.width() * 0.33), 0)
    b.redraw_now()
    assert b.spec.panels[2].title == "second" and b.spec.panels[1].title == ""
    assert [r.span for r in b.spec.panels] == [[3, 1], [1, 1], [1, 1], [1, 1]]
    # placed freely: every panel keeps where it was, the page its height; then one is dragged and resized
    w, h = b.spec.size_mm()
    b.place_freely(2, True)
    b.redraw_now()
    assert all(len(r.rect) == 4 for r in b.spec.panels) and b.spec.size_mm() == (w, pytest.approx(h, abs=0.06))
    before = list(b.spec.panels[2].rect)
    c.select(Tag("panel", 2))
    assert b._can_resize(Tag("panel", 2)) and len(c._handles()) == 8
    ch.drag(c, _centre(c, Tag("panel", 2)), 0, -40)
    b.redraw_now()
    moved = list(b.spec.panels[2].rect)
    assert moved[1] < before[1] and moved[0] == before[0] and moved[2:] == before[2:]
    handle = c._handles()["se"].center().toPoint()
    ch.drag(c, handle, -30, -20)
    b.redraw_now()
    assert b.spec.panels[2].rect[2] < moved[2] and b.spec.panels[2].rect[3] < moved[3]
    b.spec.snap_mm = 5.0                                               # then a drag lands on the 5 mm grid
    ch.drag(c, _centre(c, Tag("panel", 2)), 13, 7)
    b.redraw_now()
    assert all(abs(v / 5.0 - round(v / 5.0)) < 1e-6 for v in b.spec.panels[2].rect[:2])
    b.canvas.select(Tag("panel", 3))
    b.arrange("top")                                                   # the selected panel leads
    assert b.spec.panels[1].rect[1] == b.spec.panels[3].rect[1] == b.spec.panels[2].rect[1]
    b.all_grid()
    assert all(r.rect == [] for r in b.spec.panels) and b.spec.height_mm == 0.0
    # an inset in the first panel: empty, selected, resizable inside its panel; removing the panel frees it
    b.add_inset(0)
    assert len(b.spec.panels) == 5 and page.is_inset(b.spec.panels, 4) and c.selection == Tag("panel", 4)
    assert b._can_resize(Tag("panel", 4)) and b.tabs.tabText(4).endswith("(inset)")
    rect = list(b.spec.panels[4].inset_rect)
    ch.drag(c, _centre(c, Tag("panel", 4)), -40, 0)
    b.redraw_now()
    assert b.spec.panels[4].inset_rect[0] < rect[0]
    b.remove_panel(0)
    assert len(b.spec.panels) == 4 and not any(page.is_inset(b.spec.panels, k) for k in range(4))
    for _ in range(40):
        if not b.undo_button.isEnabled():
            break
        b.undo()
    assert len(b.spec.panels) == 1 and b.spec.panels[0].span == []     # every step taken back


def test_add_panel_asks_for_the_data_then_the_plot(builder, tmp_path, monkeypatch):
    import controls_harness as ch
    from pyqtgraph.Qt import QtWidgets as W

    from clampwise.app import addpanel

    b = builder
    nested().to_csv(tmp_path / "cells.csv", index=False)
    asked = []

    def answer(dlg):
        asked.append(type(dlg).__name__)
        if isinstance(dlg, addpanel.DataChooser):
            row = next(i for i in range(dlg.list.count()) if dlg.list.item(i).data(ch.QtCore.Qt.UserRole) == "file")
            dlg.list.setCurrentRow(row)
            assert dlg.ok.isEnabled()
            off = next(dlg.list.item(i) for i in range(dlg.list.count())
                       if dlg.list.item(i).data(ch.QtCore.Qt.UserRole) == "project")
            assert not off.flags() & ch.QtCore.Qt.ItemIsEnabled and "open a project" in off.toolTip()
            dlg.accept()
            return W.QDialog.Accepted
        if isinstance(dlg, addpanel.PlotChooser):
            assert len(dlg.buttons) >= 30 and not dlg.buttons["box"].icon().isNull()
            dlg.buttons["box"].click()
            assert "Box plot" in dlg.says.text() and dlg.ok.isEnabled()
            dlg.accept()
            return W.QDialog.Accepted
        return W.QDialog.Rejected

    monkeypatch.setattr(W.QDialog, "exec_", lambda dlg, *a: answer(dlg))
    monkeypatch.setattr(W.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(tmp_path / "cells.csv"), "")))
    monkeypatch.setattr(W.QInputDialog, "getItem", staticmethod(lambda *a, **k: ("", False)))
    assert b.spec.panels[0].frame() is None and "The page is empty" in b.add_button.toolTip()
    b.add_button.click()                                               # the empty page's one panel is filled
    assert asked == ["DataChooser", "PlotChooser"] and len(b.spec.panels) == 1
    r = b.spec.panels[0]
    assert (r.plot, r.y, r.group, r.cell) == ("box", "Amplitude (pA)", "Condition", "Cell")
    b.add_button.click()                                               # no empty panel left: a new one, beside it
    assert len(b.spec.panels) == 2 and b.spec.columns == 2 and b.current == 1 and b.spec.size == "double"
    assert b.canvas.selection.kind == "panel" and b.canvas.selection.panel == 1
    # a double-click on a panel with data asks how it is drawn; the plot's question follows
    asked.clear()
    b._activated(b.canvas.selection)
    assert asked == ["PlotChooser"]
    b.set_plot("cumulative")
    assert b.recipe.plot == "cumulative" and b.recipe.question == "distribution"
    b.set_plot("time_course")                                          # a plot that needs an x takes the time column
    assert b.recipe.question == "time" and b.recipe.x == ""            # (none in this table: left to be named)


def test_the_data_behind_a_panel_its_rows_and_the_test(builder, tmp_path):
    from pyqtgraph.Qt import QtCore
    from pyqtgraph.Qt import QtWidgets as W

    from clampwise.figure.elements import Tag

    b = builder
    nested(effect=3.0).to_csv(tmp_path / "cells.csv", index=False)
    b.open_file(str(tmp_path / "cells.csv"))
    b.redraw_now()
    b.side.setCurrentIndex(1)
    v = b.dataview
    m = v.model
    heads = [m.headerData(c, QtCore.Qt.Horizontal) for c in range(m.columnCount())]
    assert heads == ["Condition\nGroups", "Animal\nAnimal", "Cell\nCell", "Amplitude (pA)\nY"]
    assert m.rowCount() == 480 and "cells.csv: 480 rows" in v.where.text() and "480 rows, 4 columns" in v.count.text()
    assert "Y</b>: Amplitude (pA)" in v.roles.text() and "each point is a cell" in v.roles.text()
    assert not v.out_button.isEnabled() and "Select rows" in v.out_button.toolTip()
    # rows left out by hand: greyed in the table, out of the figure and its n, said in the legend; and put back
    n = b._drawn[0].prepared.counts["control"][2]
    v.table.selectRow(0)
    v.table.selectionModel().select(m.index(1, 0), QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    assert v.out_button.isEnabled()
    v.out_button.click()
    b.redraw_now()
    assert b.recipe.excluded == [0, 1] and b._drawn[0].prepared.counts["control"][2] == n - 2
    assert "2 left out by hand" in v.count.text() and "2 rows of the table left out by hand" in b.caption_text()
    assert m.data(m.index(0, 0), QtCore.Qt.FontRole).strikeOut() and not m.data(m.index(5, 0), QtCore.Qt.FontRole).strikeOut()
    v.table.selectRow(0)
    assert v.in_button.isEnabled()
    v.in_button.click()
    assert b.recipe.excluded == [1]
    b.undo()
    assert b.recipe.excluded == [0, 1]
    # the test is a choice; drawing it on the graph puts the bracket there with that test's p
    assert "Welch's t test" in v.result.toPlainText()
    v.test.setCurrentIndex(v.test.findData("mann_whitney"))
    v.on_graph.setChecked(True)
    b.redraw_now()
    assert b.recipe.test == "mann_whitney" and b.recipe.stats
    assert "Mann-Whitney U p" in v.result.toPlainText().split("Welch")[0]
    assert b._drawn[0].comparisons[0].test == "Mann-Whitney U"
    assert b.canvas.outline(Tag("bracket", 0, "control|drug")) is not None
    v.test.setCurrentIndex(v.test.findData("wilcoxon"))                # a paired test pairs the cells
    assert b.recipe.paired and v.says.currentData() == "p"
    v.copy()
    assert W.QApplication.clipboard().text().startswith("Condition\tAnimal\tCell\tAmplitude (pA)")


def test_a_journal_is_a_choice_and_the_export_is_checked_first(builder, tmp_path, monkeypatch):
    from pyqtgraph.Qt import QtWidgets as W

    from clampwise.app.builder import ExportDialog

    b = builder
    nested().to_csv(tmp_path / "cells.csv", index=False)
    b.open_file(str(tmp_path / "cells.csv"))
    assert b.spec.journal == "" and b.spec.style.font_size == 7.0 and b.spec.size_mm()[0] == 85.0
    b.set_journal("nature")
    b.touch()
    b.redraw_now()
    assert b.spec.size_mm()[0] == 89.0 and b.spec.style.label_style == "a" and "Nature" in b.size_note.text()
    assert [b.size.itemText(i) for i in range(b.size.count())] == ["Single column (89 mm)",
                                                                  "One and a half column (120 mm)",
                                                                  "Double column (183 mm)"]
    b.spec.style.tick_size = 3.0                                       # too small to print
    b.redraw_now()
    assert any("under 5 pt" in w.text() for w in b.warnings.findChildren(W.QLabel))
    seen = {}

    def accept(dlg, *a):
        if isinstance(dlg, ExportDialog):
            seen["found"] = [dlg.findings.item(i).text() for i in range(dlg.findings.count())]
            seen["format"] = dlg.format.currentData()
            assert not dlg.dpi.isEnabled() and "no resolution" in dlg.dpi.toolTip()
            dlg.format.setCurrentIndex(dlg.format.findData("png"))
            assert dlg.dpi.isEnabled() and dlg.dpi.value() == 600 and not dlg.outline.isEnabled()
            dlg.format.setCurrentIndex(dlg.format.findData("pdf"))
            return W.QDialog.Accepted
        return W.QDialog.Rejected

    monkeypatch.setattr(W.QDialog, "exec_", accept)
    monkeypatch.setattr(W.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "out"), "")))
    out = b.export()
    assert out == tmp_path / "out.pdf" and out.stat().st_size > 2000 and seen["format"] == "pdf"
    assert any(t.startswith("Fix:") and "under 5 pt" in t for t in seen["found"])
    assert "89.0 ×" in b.statusBar().currentMessage()
    b.set_journal("")
    b.set_width("custom")
    b.set_page(120.0, 50.0)
    assert b.spec.size == "120x50" and b.spec.size_mm() == (120.0, 50.0)
