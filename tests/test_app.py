"""The window: it opens with an empty state, opens a folder as a project,
moves between destinations with the keyboard, goes back, undoes an identity
edit, finds things with Ctrl+K, and analyzes in the background."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _settle(app, s=0.3):
    end = time.time() + s
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


class FakeWatch:
    """The rig as the Live screen sees it, made up: an amplifier in voltage clamp, the acquisition program open
    with a file logged, the patterned light open on a pattern, the camera closed (no rig is read in tests)."""

    def __init__(self) -> None:
        from clampwise.rig import labbook, multiclamp, polyscan, state, windows

        self.state, self.n = state, 0
        amp = multiclamp.AmplifierState(device=multiclamp.Device(1, "00835188", 0, 0, 1), connected=True, mode=0,
                                        values={"mode": "voltage clamp", "holding": -70.0, "holding_on": True,
                                                "rs": 12.5, "cm": 25.0, "whole_cell_on": True, "rs_comp_on": True,
                                                "rs_correction": 70.0, "primary_signal": "membrane current",
                                                "primary_gain": 2.0, "primary_lpf": 10000.0, "test_signal_on": False},
                                        units={"holding": "mV", "rs": "MOhm", "cm": "pF", "rs_correction": "%",
                                               "primary_lpf": "Hz"}, meters={0: -0.065, 1: -4.0e-11})
        light = polyscan.Session(Path("20261001_151431"), "20261001_151431", experiment="BLA Connectivity",
                                 sequence="Grid1.seq", regions=101, objective="Zeiss 40X", externally_triggered=True,
                                 trigger_edge="RisingEdge", devices=["Polygon #1", "LED #1"])
        rec = labbook.FileRecord(r"E:\lab_data\R007\Slice1\cell2\26o01005.abf", 1, outcome="completed",
                                 protocol=r"E:\Protocols\Lab\Simple_IV.pro", folder=r"E:\lab_data\R007\Slice1\cell2")
        self.snap = dict(amplifier=amp, clampex_open=True, data_folder=rec.folder, last_file=rec,
                         last_test=labbook.MembraneTest(1, {"ra_mohm": 9.1, "cm_pf": 52.0}), light=light,
                         light_open=True, windows={"acquisition": [windows.AppWindow(11, "acq", "clampex.exe", 1, True,
                                                                                    800, 600)]})

    def look(self):
        self.n += 1
        snap = self.state.Snapshot(**self.snap)
        snap.amplifier.values["rs"] = 12.5 if self.n < 3 else 16.0       # Rs creeping up: a 28% change by look 3
        return snap

    def close(self) -> None:
        pass


@pytest.fixture
def window(qapp, monkeypatch):
    from clampwise.app import theme
    from clampwise.app.screens.live import LiveScreen
    from clampwise.app.window import MainWindow

    monkeypatch.setattr(LiveScreen, "make_watch", FakeWatch)
    theme.apply(qapp, theme.LIGHT)
    w = MainWindow(theme.LIGHT, persist=False)
    w.resize(1366, 768)
    w.show()
    yield w
    w.close()
    w.deleteLater()
    _settle(qapp, 0.05)


@pytest.fixture
def folder(tmp_path):
    import pyabf.abfWriter

    rng = np.random.default_rng(0)
    root = tmp_path / "A9"
    for c in ("cell1", "cell2"):
        d = root / "Slice1" / c
        d.mkdir(parents=True)
        pyabf.abfWriter.writeABF1(rng.normal(-65, 0.5, (3, 2000)).astype(np.float32), str(d / f"{c}.abf"),
                                  10_000, units="mV")
    return root


def test_a_background_task_is_held_until_its_thread_has_ended(qapp, monkeypatch):
    """The result is sent from inside the task's run(): the task is let go only
    once its thread has ended (freed while running, Qt aborts the program)."""
    from clampwise.app import worker

    sends = worker.Task.run

    def send_then_linger(self):
        sends(self)
        time.sleep(0.05)                       # what a thread does after: unwinding, freeing what it made

    monkeypatch.setattr(worker.Task, "run", send_then_linger)
    held, tasks = [], []
    for i in range(5):
        tasks.append(worker.run(lambda: sum(range(10_000)),
                                lambda result, i=i: held.append(tasks[i] in worker._running or tasks[i].isFinished())))
    end = time.time() + 10
    while (len(held) < 5 or worker._running) and time.time() < end:
        qapp.processEvents()
        time.sleep(0.001)
    assert held == [True] * 5 and not worker._running             # held while running, let go once ended


def test_opens_on_files_with_an_empty_state_that_teaches(window):
    files = window.screens["files"]
    assert window.frame.stack.currentWidget() is files
    assert files.stack.currentWidget() is files.empty
    assert not files.analyze_button.isEnabled()
    assert window.frame.sidebar.buttons["files"].isChecked()


def _primaries(screen):
    return [b.text() for b in screen.findChildren(QtWidgets.QPushButton)
            if b.objectName() == "primary" and b.isVisibleTo(screen)]


def test_every_screen_has_one_primary_action_at_most(qapp, window, folder):
    for key, screen in window.screens.items():                   # nothing open
        assert len(_primaries(screen)) <= 1, (key, _primaries(screen))
    window.open_project(folder)
    window.screens["files"].table.selectRow(0)
    _settle(qapp)
    for key, screen in window.screens.items():                   # a project and a selection
        assert len(_primaries(screen)) <= 1, (key, _primaries(screen))


def test_a_large_folder_is_looked_through_while_the_window_goes_on(qapp, window, folder, monkeypatch):
    """5,000 recordings take 2 s to find (a deep tree on a slow disk longer): the window answers meanwhile and says
    what it is doing, and the project is open when open_project returns."""
    import clampwise.io as cio

    real = cio.find_recordings

    def slow(f):
        time.sleep(1.2)
        return real(f)

    monkeypatch.setattr(cio, "find_recordings", slow)
    beats, said = [], []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: beats.append(time.monotonic()))
    timer.start(20)
    monkeypatch.setattr(window, "say", lambda text: said.append(text))
    window.open_project(folder)
    timer.stop()
    assert window.project is not None and len(window.project.recordings()) == 2
    assert np.diff(beats).max() < 0.3 and len(beats) > 30
    assert any(t.startswith("Opening A9: looking for its recordings (") for t in said)


def test_the_workbook_is_written_off_the_window_s_thread(qapp, window, folder, monkeypatch):
    from clampwise import export

    window.open_project(folder)
    window.project.set_meta("note_for_export", "kept")                     # what the window wrote, seen by it
    seen = {}

    def write(project):
        seen["meta"] = project.meta("note_for_export")
        seen["same"] = project is window.project
        return export.write_workbook(project)

    monkeypatch.setattr("clampwise.app.screens.export.open_path", lambda p: None, raising=False)
    out = window.screens["export"]._write("workbook", write)
    assert out is not None and out.exists() and seen == {"meta": "kept", "same": False}


def test_open_a_folder_select_navigate_back_and_undo(qapp, window, folder):
    window.open_project(folder)
    files = window.screens["files"]
    assert files.table.rowCount() == 2 and files.analyze_button.text() == "Analyze 2 recordings"
    files.table.selectRow(0)
    _settle(qapp)
    assert window.selected is not None
    assert "Slice1" in window.frame.breadcrumb.text()
    window.go("check")
    assert window.frame.stack.currentWidget() is window.screens["check"]
    window.back()
    assert window.frame.stack.currentWidget() is files
    path = window.selected
    from clampwise.core import Identity

    window.set_identity([path], Identity(condition="carbachol"))
    assert window.project.identity(path).condition == "carbachol"
    window.undo.undo()
    assert window.project.identity(path).condition == ""
    window.undo.redo()
    assert window.project.identity(path).condition == "carbachol"


def test_find_lists_destinations_actions_and_recordings(window, folder):
    window.open_project(folder)
    labels = [label for label, _, _ in window.findable()]
    assert "Check" in labels and "Open folder" in labels and "cell1.abf" in labels


def test_analyze_runs_in_the_background_and_fills_the_type(qapp, window, folder):
    window.open_project(folder)
    window.analyze_all()
    t0 = time.time()
    while window._job is not None and window._job.isRunning() and time.time() - t0 < 60:
        _settle(qapp, 0.1)
    _settle(qapp, 0.3)
    rows = window.project.summaries()
    assert all(r["analyzed_at"] for r in rows)
    files = window.screens["files"]
    assert files.table.item(0, 7).text() != "not yet"


def test_theme_switch_restyles_without_restarting(qapp, window):
    from clampwise.app import theme

    window.set_theme(theme.DARK)
    assert theme.DARK.window in qapp.styleSheet()
    window.set_theme(theme.LIGHT)
    assert theme.LIGHT.window in qapp.styleSheet()


def test_results_and_analyze_after_a_run(qapp, window, folder):
    window.open_project(folder)
    window.analyze_all()
    t0 = time.time()
    while window._job is not None and window._job.isRunning() and time.time() - t0 < 60:
        _settle(qapp, 0.1)
    _settle(qapp, 0.3)
    window.screens["files"].table.selectRow(0)
    _settle(qapp)
    results = window.screens["results"]
    window.go("results")
    assert results.stack.currentIndex() == 2                          # the overview, not an empty state
    assert results.tabs.count() >= 3
    assert [results.tabs.tabText(i) for i in range(results.tabs.count())][-3:] == \
        ["Every number", "Flags and notes", "How it was measured"]
    analyze = window.screens["analyze"]
    window.go("analyze")
    assert analyze.stack.currentIndex() == 1 and analyze.run_button.text() == "Analyze 2 recordings"
    from clampwise.app.screens.analyze import AnalysisSection

    section = next(s for s in analyze.findChildren(AnalysisSection) if s.analysis.key == "baseline")
    control = next(c for c in section.controls if c.setting.key == "piece_s")
    control.control.setValue(20.0)
    control._write()
    assert window.project.settings()["baseline"]["piece_s"] == 20.0
    assert section.advanced.header.text() == "Advanced (1 changed)"
    control._reset()
    assert "piece_s" not in window.project.settings().get("baseline", {})
    assert section.advanced.header.text() == "Advanced"


def test_event_review_on_check_and_the_experiment_template(qapp, window, tmp_path):
    """B4: events on the trace, reject and undo, apply, and one template for the experiment."""
    import pyabf.abfWriter

    from clampwise import simulate

    root = tmp_path / "sIPSC"
    root.mkdir()
    for k in range(2):
        sim = simulate.spontaneous(duration_s=12.0, fs=10_000.0, seed=k, decay_s=0.012)
        pyabf.abfWriter.writeABF1(sim.recording.sweep(0)[None, :].astype(np.float32), str(root / f"cell{k}.abf"),
                                  10_000, units="pA")
    window.open_project(root)
    window.analyze_all()
    t0 = time.time()
    while window._job is not None and window._job.isRunning() and time.time() - t0 < 120:
        _settle(qapp, 0.1)
    _settle(qapp, 0.3)
    path = window.project.recordings()[0].path
    window.select(path)
    window.go("check")
    _settle(qapp, 0.2)
    check = window.screens["check"]
    assert check._review_on and len(check._events) > 20
    check.step_event(1)
    first = check._events[check._event_i]
    check.toggle_event()
    assert "1 rejected" in check.review_count.text() and check.ev_apply.isEnabled()
    window.undo.undo()
    assert "0 rejected" in check.review_count.text() and not check.ev_apply.isEnabled()
    window.undo.redo()
    check.apply_review()
    t0 = time.time()
    while window._job is not None and window._job.isRunning() and time.time() - t0 < 120:
        _settle(qapp, 0.1)
    _settle(qapp, 0.3)
    run = window.project.last_run(path)
    rows = {round(r["onset_s"], 4): r for r in run.results["events"].tables["events"]}
    assert not rows[round(first["onset_s"], 4)]["accepted"] and run.results["events"].values["n_rejected"] == 1
    window.go("analyze")
    _settle(qapp, 0.2)
    window.screens["analyze"].make_template()
    s = window.project.settings()["events"]
    assert s["template_decay_ms"] == pytest.approx(12.0, rel=0.15) and "averaged event" in s["template_origin"]


def test_every_graph_the_results_offer_draws(qapp):
    # B11, NeuroExpress's panels: the views come from the result tables and drawables
    import pyqtgraph as pg

    from clampwise import simulate
    from clampwise.analysis import pipeline
    from clampwise.app import panels, theme

    steps = pipeline.analyze(simulate.firing_steps(pattern="adapting").recording)
    minis = pipeline.analyze(simulate.spontaneous(duration_s=20.0).recording)
    for run in (steps, minis):
        found = panels.views(run)
        assert len({v.key for v in found}) == len(found)
        for v in found:
            item = pg.PlotItem()
            text = panels.draw_heatmap(item, theme.LIGHT, v.built(), panels.heat_options(), v.key) if v.heatmap                 else v.draw(item, theme.LIGHT)
            assert isinstance(text, str) and not text.startswith("Nothing"), v.key
    keys = [v.key for v in panels.views(steps)]
    r = next(v for v in panels.views(steps) if v.key.startswith("intrinsic.steps.r_peak"))
    assert r.title == "R peak and R steady vs. current" and r.units == "[MOhm vs. pA]"
    assert "Resistance at rest" in r.draw(pg.PlotItem(), theme.DARK)          # the readout names R at 0 pA
    assert any(k.startswith("events.histogram_amplitudes") for k in [v.key for v in panels.views(minis)])
    assert not any(k.endswith(".peak_s") for k in keys)                       # positions are not graphs
    assert {"intrinsic.first spikes.v", "intrinsic.first spikes.phase"} <= set(keys)    # the spike panel
    assert "events.overlay" in [v.key for v in panels.views(minis)]                     # 20 events and their average
    assert "heatmap.sweeps" in keys and {"heatmap.events.rate", "heatmap.spectrogram.hann", "heatmap.spectrogram.morlet",
                                        "heatmap.band_power"} <= {v.key for v in panels.views(minis)}


def test_a_point_moved_by_hand_is_kept_reanalyzed_and_undone(qapp, window, tmp_path):
    # step 5: a step's measured point dragged on Check, kept as a decision, Ctrl+Z puts it back
    import shutil

    import pyqtgraph as pg

    from labdata import DEMOS

    src = DEMOS / "Test_IV_1.abf"
    if not src.is_file():
        pytest.skip("NeuroExpress's test files are not on this machine")
    shutil.copy2(src, tmp_path / src.name)
    path = tmp_path / src.name

    def wait():
        end = time.time() + 120
        while window._job is not None and window._job.isRunning() and time.time() < end:
            _settle(qapp, 0.05)
        _settle(qapp, 0.3)

    window.open_project(tmp_path)
    window.analyze_all()
    wait()
    window.select(path)
    window.go("check")
    check = window.screens["check"]
    check.mode.setCurrentIndex(1)
    _settle(qapp)
    targets = [i for i in check.plot.getPlotItem().items if isinstance(i, pg.TargetItem)]
    peak = next(i for i in targets if i.toolTip().startswith("peak"))
    peak.setPos(peak.pos().x() + 0.05, peak.pos().y())
    peak.sigPositionChangeFinished.emit(peak)
    wait()
    assert ":peak:" in window.project.moved_points(path)
    run = window.project.last_run(path)
    assert any(f.code == "moved by hand" for f in run.results["intrinsic"].flags)
    assert check.reset_points.isVisible()
    window.undo.undo()
    wait()
    assert window.project.moved_points(path) == ""


def test_help_has_an_entry_for_every_analysis_and_settings_switch_the_theme(qapp, window):
    # step 6: the manual is written from the registry; a "?" opens its analysis's entry
    from clampwise.analysis import registry
    from clampwise.app import widgets

    help_ = window.screens["help"]
    keys = [help_.list.item(i).data(QtCore.Qt.UserRole) for i in range(help_.list.count())]
    assert {"quick start", "screens", "glossary", "reading"} <= set(keys)
    assert set(registry.load()) <= set(keys)
    widgets.HelpButton.open_manual("intrinsic")
    assert window.frame.stack.currentWidget() is help_
    assert "Rheobase" in help_.page.toPlainText()
    help_.search.setText("short square")
    shown = [help_.list.item(i).text() for i in range(help_.list.count()) if not help_.list.item(i).isHidden()]
    assert "Short-square threshold" in shown and "Spike train" not in shown
    window.go("settings")
    window.screens["settings"].theme_buttons["dark"].setChecked(True)
    assert window.palette.name == "dark"


def test_check_steps_through_files_and_marks_them_with_keys_only(qapp, window, folder):
    """Phase 1: J/K between files, G/B marks with a reason, N to the next unreviewed; no Stimulus screen."""
    from pyqtgraph.Qt import QtTest

    from clampwise import export

    assert "stimulus" not in window.screens and "Stimulus" not in [t for t, _, _ in window.findable()]
    window.open_project(folder)
    paths = sorted(r.path for r in window.project.recordings())
    window.select(paths[0])
    window.go("check")
    _settle(qapp, 0.2)
    check = window.screens["check"]
    window.activateWindow()
    check.mode.setCurrentIndex(1)                      # one sweep at a time: kept from file to file

    def key(k):
        QtTest.QTest.keyClick(check.plot, k)
        _settle(qapp, 0.05)

    key(QtCore.Qt.Key_K)
    assert window.selected == paths[1] and check.mode.currentIndex() == 1
    assert check.picker.currentData() == str(paths[1])
    key(QtCore.Qt.Key_J)
    assert window.selected == paths[0]
    key(QtCore.Qt.Key_B)
    assert window.project.file_review(paths[0]) == ("bad", "")
    key(QtCore.Qt.Key_N)
    assert window.selected == paths[1]
    check.reason.setText("Rs above 25")
    key(QtCore.Qt.Key_G)
    assert window.project.file_review(paths[1]) == ("good", "Rs above 25")
    window.undo.undo()
    assert window.project.file_review(paths[1]) == ("", "")
    window.undo.redo()
    files = window.screens["files"]
    col = [files.table.horizontalHeaderItem(j).text() for j in range(files.table.columnCount())].index("Review")
    assert sorted(files.table.item(i, col).text() for i in range(2)) == ["bad", "good"]
    assert export.identity_columns()[-1].title == "Review"
    window.project.set_feature("file_review", False)
    window.screens["settings"].on_project()
    check.on_project()
    assert not check.mark_good.isVisibleTo(check) and "File review on Check: off" in export._features(window.project)


def test_a_figure_maximizes_restores_and_opens_in_its_own_window(qapp, window):
    from pyqtgraph.Qt import QtTest

    from clampwise.app.panels import View

    window.go("figures")
    fig = window.screens["figures"]
    fig.stack.setCurrentIndex(1)
    view = View("a.b", "A graph", "pA", lambda item, palette: "")
    for p in fig.panels:
        p.set_views([view])
    _settle(qapp, 0.1)
    fig.panels[2].max_button.click()
    assert [p.isVisibleTo(fig) for p in fig.panels] == [False, False, True, False]
    QtTest.QTest.keyClick(fig.panels[2].plot, QtCore.Qt.Key_Escape)
    _settle(qapp, 0.05)
    assert all(p.isVisibleTo(fig) for p in fig.panels) and fig.maximized is None
    fig.panels[0].detach_requested.emit(fig.panels[0])
    assert len(fig.windows) == 1 and fig.windows[0].graph.title.toolTip() == "A graph pA"
    fig.windows[0].close()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)       # closed: deleted
    assert not fig.windows


def _wait(qapp, until, seconds=8.0):
    end = time.time() + seconds
    while time.time() < end and not until():
        qapp.processEvents()
        time.sleep(0.02)
    return until()


def test_live_shows_the_rig_in_one_place_in_the_rig_s_words(qapp, window):
    window.go("live")
    live = window.screens["live"]
    assert _wait(qapp, lambda: live.last is not None)
    r = live.readouts
    assert r["mode"].value.text() == "voltage clamp" and r["holding"].value.text() == "-70.0 mV"
    assert r["rs"].value.text() == "12.5 MOhm" and r["cm"].value.text() == "25.0 pF"
    assert r["rs_comp"].value.text() == "70.0 % correction"
    assert live.views["amplifier"].badge.text() == "✓ reading" and live.views["light"].badge.text() == "✓ open"
    assert "Grid1.seq, 101 regions" in live.views["light"].detail.text()
    assert live.views["acquisition"].state.text().startswith("Minimized")
    assert "26o01005.abf (completed)" in live.acquisition.text() and "Ra 9.1 MOhm" in live.test.text()
    assert live.meters.text() == "Meters: Channel 1: -65.0 mV, -40.0 pA"
    off = live.last.amplifier
    off.values["whole_cell_on"] = False                                          # the dials, not an estimate
    live._amplifier(live.last)
    assert r["rs"].value.text() == "12.5 MOhm (off)" and r["cm"].value.text() == "25.0 pF (off)"
    assert "not an estimate" in r["rs"].value.toolTip()
    off.values["whole_cell_on"] = True
    assert _wait(qapp, lambda: "+28%" in live.rs_badge.text())               # Rs crept from 12.5 to 16 MOhm
    assert live.rs_badge.objectName() == "badge_warn" and live.plot.isVisible()
    # the interface names the rig's roles; Help names the programs
    words = " ".join(w.text() for w in live.findChildren(QtWidgets.QLabel)) + live.purpose
    assert not any(name in words for name in ("Clampex", "MultiClamp", "Commander", "PolyScan", "Infinity", "Polygon"))


def test_live_reads_each_sweep_s_test_pulse_and_flags_an_rs_jump(qapp, window):
    import datetime as dt

    from clampwise.rig import growing, state

    window.go("live")
    live = window.screens["live"]
    start = dt.datetime(2026, 10, 14, 11, 0)

    def sweeps(path, rs, first=0):
        return state.Snapshot(writing=path, sweeps=[growing.SweepTest(path.rsplit("\\", 1)[-1], first + k, 10.0 * k,
                                                                      "voltage clamp", r, 200.0, 100.0, -40.0, 2.0,
                                                                      start) for k, r in enumerate(rs)])

    live._trace(sweeps(r"E:\data\R9\Slice1\cell1\26o14000.abf", [12.0, 12.1, 11.9, 12.0]))
    assert live.trace.text() == ("From the recorded trace (26o14000.abf, sweep 4): Rs 12.0 MOhm, Rm 200 MOhm, "
                                 "Cm 100 pF, holding -40.0 pA, noise 2.0 pA SD.")
    assert live.rs_badge.text().endswith("Rs +0% since this cell's first sweeps") and live.plot.isVisible()
    live._trace(sweeps(r"E:\data\R9\Slice1\cell1\26o14001.abf", [15.0, 18.0]))
    assert "Rs +50% since this cell's first sweeps" in live.rs_badge.text()
    assert live.rs_badge.objectName() == "badge_warn"
    assert len(live.trace_curve.getData()[0]) == 6
    live._trace(sweeps(r"E:\data\R9\Slice1\cell2\26o14002.abf", [10.0]))       # a new cell's folder: begins again
    assert len(live._sweeps) == 1


def test_live_s_views_can_be_hidden_enlarged_and_show_part_of_a_window(qapp, window):
    # maximize or minimize the views, hide the unused ones, the camera and acquisition largest,
    # the camera showing only where its picture is
    from clampwise.rig import windows

    toggle = window.frame.details_toggle
    toggle.setChecked(True)
    window.go("live")
    assert not toggle.isChecked()                                         # the programs get the room
    window.go("files")
    assert toggle.isChecked()                                             # and Details is back elsewhere
    window.go("live")
    live = window.screens["live"]
    grid = live.grid
    assert grid.getItemPosition(grid.indexOf(live.views["acquisition"])) == (0, 0, 1, 3)    # the top, all across
    assert grid.getItemPosition(grid.indexOf(live.views["camera"]))[:2] == (1, 0)
    assert grid.columnStretch(0) > grid.columnStretch(1)                                    # the camera largest below
    shown = lambda: {r for r, v in live.views.items() if v.isVisibleTo(live)}            # noqa: E731
    live.set_view_shown("light", False)
    assert shown() == {"acquisition", "amplifier", "camera"}
    live.toggle_enlarged("camera")
    assert shown() == {"camera"} and live.views["camera"].enlarged
    live.toggle_enlarged("camera")
    assert shown() == {"acquisition", "amplifier", "camera"}
    live.set_view_shown("light", True)
    assert shown() == set(live.views)
    # the camera's usual part is its picture panel; a part dragged out replaces it; the whole window on request
    cam = live.views["camera"]
    cam.source = windows.AppWindow(5, "cam", "infinityanalyze.exe", 1, False, 1534, 1040)
    assert cam.shown_part() == windows.PARTS["infinityanalyze.exe"]
    live.set_part("camera", (100, 160, 460, 144))
    assert cam.shown_part() == (100, 160, 460, 144)
    live.set_part("camera", "whole")
    assert cam.shown_part() is None


def test_live_uses_the_camera_itself_on_request(qapp, window, tmp_path):
    # the camera used in Clampwise, without Infinity Analyze
    from test_rig_camera_direct import FakeLucam

    from clampwise.rig import lucam, picture, windows

    window.go("live")
    live = window.screens["live"]
    view, panel = live.views["camera"], live.camera
    fake = FakeLucam()
    panel.make = lambda: lucam.Camera(fake, owner_check=lambda: False)
    panel.folder = lambda: str(tmp_path)
    view.source = windows.AppWindow(5, "cam", "infinityanalyze.exe", 1, False, 1534, 1040)
    live.use_camera()
    assert not view.own and fake.calls == []                                    # its program holds it: not taken
    view.source = None
    live.use_camera()
    assert view.own and view.use_button.text() == "Let go" and panel.isVisibleTo(view)
    assert _wait(qapp, lambda: panel.last is not None)
    assert panel.image.image.shape == (6, 8, 3)
    panel._fps_at -= 1.0                                                         # a second's frames: the rate
    panel._rate()
    assert panel.fps.text().split()[1] == "fps" and float(panel.fps.text().split()[0]) > 0
    panel.dialog.show_adjust(picture.Adjust(gamma=0.6), apply=True)
    path = panel.snapshot()
    assert path is not None and path.parent == tmp_path and path.suffix == ".tif"
    assert path.with_name(path.stem + "_as_shown.png").is_file()                 # the original kept as taken
    panel.capture("background")
    assert _wait(qapp, lambda: panel.corrections.background is not None)
    assert panel.corrections.mode == "subtract" and panel.dialog.bg_state.text() == "kept"
    panel.capture("overlay")
    assert _wait(qapp, lambda: panel.corrections.overlay is not None)
    panel.auto_exposure()
    assert _wait(qapp, lambda: panel.status.text().startswith("Auto exposure:"))
    panel.toggle_measure()
    assert panel.ruler is not None and "px (not calibrated)" in panel.ruler_text.toPlainText()
    # the objective's slider and the calibration: one objective measured, the others by their magnification
    assert panel.magnification == 20.0 and not panel.bar_line.isVisible()
    panel.dialog.um.setValue(0.2)                                                 # 20x measured: 0.2 um a pixel
    assert panel.calibrated == {20.0: 0.2} and panel.bar_line.isVisible() and "um" in panel.ruler_text.toPlainText()
    panel.objective.setValue(panel.objectives.index(2.5))
    assert panel.scale() == (pytest.approx(1.6), "estimated from the 20x calibration")
    assert panel.objective_caption.text() == "Objective 2.5x  (1.6 um/px)" and panel.dialog.um.value() == 0.0
    panel.set_objectives("2.5, 10, 20, 63")
    assert panel.magnification == 2.5 and panel.objective.maximum() == 3
    # no calibration: the size of a pixel from the camera's own (4.5 um) and the adapter, shown on the panel
    panel.calibrated.clear()
    panel.dialog.pixel_um.setValue(4.5)
    panel.objective.setValue(panel.objectives.index(20.0))
    assert panel.scale()[0] == pytest.approx(0.225) and panel.scale()[1].startswith("nominal")
    assert panel.objective_caption.text() == "Objective 20x  (0.225 um/px)" and panel.bar_line.isVisible()
    panel.dialog.adapter.setValue(0.5)
    assert panel.scale()[0] == pytest.approx(0.45) and "(0.45 um/px)" in panel.objective_caption.text()
    # the slice's compass: rostral, caudal, medial, lateral over the picture, turned as the slice lies
    assert not panel.compass_lines["Rostral"].isVisible()
    panel.dialog.compass.setChecked(True)
    assert all(panel.compass_text[k].isVisible() for k in ("Rostral", "Caudal", "Medial", "Lateral"))
    (x0, x1), (y0, y1) = [list(v) for v in panel.compass_lines["Rostral"].getData()]
    assert x1 == pytest.approx(x0) and y1 < y0                                    # rostral up
    panel.dialog.rostral.setValue(90)                                             # rostral to the right
    (x0, x1), (y0, y1) = [list(v) for v in panel.compass_lines["Rostral"].getData()]
    assert x1 > x0 and y1 == pytest.approx(y0)
    (x0, x1), (y0, y1) = [list(v) for v in panel.compass_lines["Medial"].getData()]
    assert y1 > y0                                                                # medial its right: down
    panel.dialog.medial.setCurrentIndex(1)
    (x0, x1), (y0, y1) = [list(v) for v in panel.compass_lines["Medial"].getData()]
    assert y1 < y0
    assert panel.orientation() == (90, False)
    # a coronal slice: dorsal-ventral and medial-lateral drawn, rostral-caudal cut across; dorsal the one turned
    panel.dialog.plane.setCurrentIndex(panel.dialog.plane.findData("coronal"))
    assert set(panel.compass_text) == {"Dorsal", "Ventral", "Medial", "Lateral"}
    assert panel.compass_text["Dorsal"].toPlainText() == "Dorsal" and panel.dialog.up_label.text() == "Dorsal points"
    assert panel.dialog.out_of_plane.text() == "RC is cut across"
    (x0, x1), (y0, y1) = [list(v) for v in panel.compass_lines["Dorsal"].getData()]
    assert x1 > x0 and y1 == pytest.approx(y0)                                    # dorsal where rostral was
    panel.dialog.plane.setCurrentIndex(panel.dialog.plane.findData("sagittal"))
    panel.dialog.naming.setCurrentText("Anterior, posterior")
    assert set(panel.compass_text) == {"Dorsal", "Ventral", "Anterior", "Posterior"}
    assert panel.dialog.medial.currentText() == "Anterior to the left of dorsal"
    assert panel.orientation_text() == ("sagittal slice: dorsal 90 deg clockwise from up, anterior to its left; "
                                        "ML cut across")
    panel.dialog.naming.setCurrentText("Letters: A P D V M L")
    assert set(panel.compass_text) == {"D", "V", "A", "P"}
    # sized freely: the Adjust window shrinks below its content (which then scrolls); the controls' share is dragged
    assert panel.dialog.isSizeGripEnabled() and panel.dialog.minimumSize().height() <= 200
    panel.dialog.resize(300, 200)
    assert panel.dialog.height() <= 220
    assert panel.split.count() == 2 and panel.split.widget(1) is panel.controls_area
    # the adjustable histogram: its handles are the levels shown black and white; Reset all puts them back
    assert _wait(qapp, lambda: panel.levels.counts is not None)
    panel.levels.set_levels(40, 180, 1.0)
    assert (panel.adjust.black, panel.adjust.white) == (40.0, 180.0) and panel.levels.region.getRegion() == (40, 180)
    assert panel.levels.mid.value() == pytest.approx(110.0)                       # mid-grey half way at gamma 1
    panel.levels.mid.setValue(40 + 140 * 0.25)                                    # dragged left: dim parts lifted
    assert panel.adjust.gamma == pytest.approx(2.0, abs=0.01) and panel.levels.gamma.value() == pytest.approx(2.0)
    assert panel.controls_area.minimumHeight() >= 40                              # the buttons never squeezed away
    panel.dialog.show_adjust(picture.Adjust(), apply=True)
    assert (panel.adjust.black, panel.adjust.white) == (0.0, 255.0) and panel.levels.white.value() == 255
    # binning: off unless chosen; on, the picture is half as wide and each pixel twice the um across
    assert panel.binning.currentData() == 1
    um = panel.scale()[0]
    panel.binning.setCurrentIndex(panel.binning.findData(2))
    assert panel._thread.binning == 2 and panel.scale()[0] == pytest.approx(2 * um)
    assert _wait(qapp, lambda: panel.last is not None and panel.last.rgb.shape[:2] == (3, 4))
    panel.binning.setCurrentIndex(0)
    assert _wait(qapp, lambda: panel.last.rgb.shape[:2] == (6, 8))
    # the Polygon in the patterned light view: Use here (PolyScan closed) shows its controls there
    from test_polygon import FakeSDK

    from clampwise.rig.polygon import Polygon

    light, sdk = live.views["light"], FakeSDK()
    assert not light.own and not live.patterned.polygon.connected                    # nothing taken until asked
    live.patterned.polygon = Polygon(sdk, polyscan_running=lambda: False)
    light.source = None
    light.use_button.click()                                                     # found without the window waiting
    assert light.own and live.patterned.wait() and live.patterned.polygon.connected
    assert live.patterned.isVisibleTo(light)
    assert light.use_button.text() == "Let go"
    live.patterned.source.setCurrentText("Grid")
    live.patterned.make_steps()
    assert live.patterned.list.count() == 100 and live.patterned.list.item(0).text() == "1"
    live.patterned.project()
    assert live.patterned.wait() and live.patterned.polygon.shown == "1" and sdk.image is not None
    light.use_button.click()                                                     # Let go: PolyScan can have it
    assert not light.own and live.patterned.wait() and not live.patterned.polygon.connected
    assert ("uninit",) in sdk.calls
    panel.split.setSizes([500, 10])                                               # dragged right down:
    assert panel.split.sizes()[1] == panel.controls_area.minimumHeight()          # the buttons still in view
    panel.release()
    assert not view.own and fake.calls[-1] == "close" and view.use_button.text() == "Use here"


def test_live_uses_a_basler_camera_as_it_does_the_other(qapp, window, tmp_path):
    # a rig whose camera is Basler's: taken through pylon, its pixel from its model, a gain of 0 dB set as asked
    from test_rig_basler import FakePylon

    from clampwise.rig import basler

    window.go("live")
    live = window.screens["live"]
    view, panel = live.views["camera"], live.camera
    fake = FakePylon()
    fake.features["Gain"] = 6.0
    panel.make = lambda: basler.Camera(fake, owner_check=lambda: False)
    panel.folder = lambda: str(tmp_path)
    panel.dialog.pixel_um.setValue(0.0)
    view.source = None
    live.use_camera()
    assert _wait(qapp, lambda: panel.last is not None)
    assert panel.image.image.shape[:2] == (6, 8) and fake.features["PixelFormat"] == "Mono8"
    assert panel.dialog.pixel_um.value() == pytest.approx(5.86)                  # an acA1920-40um's pixel
    assert "acA1920-40um" in panel.status.text() and panel.gain.value() == 6.0
    panel.gain.setValue(0.0)
    panel._exposure()
    assert _wait(qapp, lambda: fake.features["Gain"] == 0.0)
    panel.release()
    assert not view.own and fake.calls[-1] == "terminate" and fake.features["PixelFormat"] == "Mono12"


class FakeRecorder:
    """rig.recorder.Recorder without the digitizer: two sweeps of two inputs appear in its raw file."""

    def __init__(self, work):
        from clampwise.rig import recorder

        self.work, self.status, self.started = work, recorder.Status(), None

    def start(self, protocol, folder, seconds=None, save=True):
        import threading

        import numpy as np

        self.started = (str(protocol), str(folder), seconds, save)
        st = self.status
        st.state, st.message, st.total, st.saving = ("recording" if save else "playing"), "VC, sim.pro", 8000, save
        st.raw, st.inputs = self.work / "inputs.raw", [("IN 0", "pA", 5e-5), ("I", "mV", 0.01)]
        st.rate_hz, st.sweep_frames, st.step_frames, st.episodic = 20_000.0, 4000, 4000, True
        st.segment += 1
        st.raw.write_bytes(np.zeros((8000, 2), "<i2").tobytes())
        self._stop = threading.Event()

    def pause(self):
        self.status.state = "paused"

    def resume(self):
        self.status.state = "recording"

    def stop(self):
        self._stop.set()

    def finish(self, folder):
        st = self.status
        st.frames, st.state, st.path = 8000, "done", folder / "26o15cw000.abf"
        st.files = [st.path] if st.saving else []
        st.message = "26o15cw000.abf: 0.4 s" if st.saving else "played: nothing kept"

    def wait(self, timeout=None):
        return self.status


def test_live_records_in_clampwise_on_request(qapp, window, tmp_path):
    # acquisition in Clampwise without Clampex, as the camera is used without its own program
    from clampwise.rig import windows
    from clampwise.simulate.abf_file import membrane_test_file
    from clampwise.rig import growing

    window.go("live")
    live = window.screens["live"]
    view, panel = live.views["acquisition"], live.recorder
    (tmp_path / "protocols").mkdir()
    head = membrane_test_file(tmp_path / "sim.abf", n_sweeps=2).read_bytes()
    pro = tmp_path / "protocols" / "sim.pro"
    pro.write_bytes(head[:growing._section(head, growing.DATA)[0] * growing.BLOCK])
    fakes = []
    panel.make = lambda: fakes.append(FakeRecorder(tmp_path)) or fakes[-1]
    view.source = windows.AppWindow(5, "Clampex", "clampex.exe", 1, False, 1600, 900)
    live.use_acquisition()
    assert not view.own                                                   # the acquisition program holds the digitizer
    view.source = None
    live.use_acquisition()
    assert view.own and view.use_button.text() == "Let go" and panel.isVisibleTo(view)
    panel._protocols_dir = str(tmp_path / "protocols")
    panel.fill_protocols()
    assert panel.protocol.currentText() == "sim" and not panel.seconds.isVisibleTo(panel)   # episodic: no length
    # never a blank scope: the protocol drawn first
    assert "sim" in panel.note.text() and "as it will play" in panel.note.text() and len(panel.plots) == 1
    assert len(panel.plots[0].listDataItems()) == 2                       # each sweep's command
    panel.set_folder(str(tmp_path / "data"))
    assert panel.play_button.isEnabled() and panel.record_button.isEnabled() and not panel.pause_button.isEnabled()
    panel.play()                                                          # Play: drawn and measured, nothing kept
    assert fakes[-1].started[3] is False and not panel.record_button.isEnabled() and panel.stop_button.isEnabled()
    assert "sweep 1" in panel.note.text() and len(panel.plots) == 2      # the inputs' axes at once, and what it does
    fakes[-1].finish(tmp_path / "data")
    assert _wait(qapp, lambda: panel.status.text().startswith("Played"))
    assert panel.note.text() == "Played: nothing kept."
    panel.record()
    rec = fakes[-1]
    assert rec.started == (str(pro), str(tmp_path / "data"), None, True) and not panel.play_button.isEnabled()
    assert live._watched() == str(tmp_path / "data")                   # Live reads the file being recorded
    assert _wait(qapp, lambda: len(panel.plots) == 2 and panel._drawn == 1)
    panel.pause()
    assert _wait(qapp, lambda: panel.pause_button.text() == "Resume")
    panel.pause()                                                         # Resume
    assert _wait(qapp, lambda: panel.pause_button.text() == "Pause")
    panel.release()
    assert view.own and panel.status.text() == "Stop the recording first."
    panel.stop()
    assert rec._stop.is_set()
    rec.finish(tmp_path / "data")
    assert _wait(qapp, lambda: panel.record_button.isEnabled() and panel.play_button.isEnabled())
    assert panel.status.text().startswith("Recorded: 26o15cw000.abf") and live._recording == ""
    assert panel.note.text() == "Recorded: 26o15cw000.abf."
    panel.release()
    assert not view.own and view.use_button.text() == "Use here"


def test_the_stimulus_editor_saves_a_new_protocol_as_edited(qapp, tmp_path):
    # the simple stimulus editor; the original is never changed
    from clampwise.app import theme
    from clampwise.app.screens.protocol_editor import ProtocolEditor
    from clampwise.rig import acquire, growing
    from clampwise.simulate.abf_file import membrane_test_file

    head = membrane_test_file(tmp_path / "sim.abf", n_sweeps=3).read_bytes()
    pro = tmp_path / "sim.pro"
    pro.write_bytes(head[:growing._section(head, growing.DATA)[0] * growing.BLOCK])
    original = pro.read_bytes()
    d = ProtocolEditor(None, str(pro), theme.LIGHT)
    t = d.tables[0]
    t.cellWidget(1, 1).setValue(-90.0)                 # epoch B's level
    t.cellWidget(1, 2).setValue(10.0)                  # +10 a sweep
    d.sweeps.setValue(5)
    assert d.status.text().startswith("5 sweeps")
    d.name.setText("sim")
    d.save()
    assert d.saved is None and "exists" in d.status.text() and pro.read_bytes() == original
    d.name.setText("sim_iv")
    d.save()
    pr = acquire.read_protocol(d.saved)
    units, _ = acquire.waveforms(pr)
    at = pr.sweep_frames // 64 + d.edit.epochs[0][0].samples + 10
    assert pr.sweeps == 5 and [round(float(units[k, 0, at])) for k in range(5)] == [-90, -80, -70, -60, -50]
    t2 = ProtocolEditor(None, str(pro), theme.LIGHT)
    t2.tables[0].cellWidget(1, 4).setValue(5000.0)     # longer each sweep, past the sweep's end
    assert "past the sweep" in t2.status.text()


def test_a_dragged_rectangle_is_the_window_s_part():
    from clampwise.rig import windows

    t = object.__new__(windows.Thumbnail)                 # the arithmetic only: no window is shown in a test
    t._handle = None
    t.shown = ((8, 52, 1170, 1004), (100, 50, 100 + 581, 50 + 476))      # the picture panel, drawn at half size
    assert t.to_source(100, 50) == (8, 52) and t.to_source(681, 526) == (1170, 1004)
    assert t.to_source(390, 288) == (588, 528)                              # 8 + 290 x 2, 52 + 238 x 2
    assert t.to_source(0, 0) == (8, 52)                                    # outside: held to the edge


def test_live_keeps_a_note_with_the_project_and_is_a_feature(qapp, window, folder):
    window.go("live")
    live = window.screens["live"]
    live.note.setText("drug on")
    live.add_note()
    assert live.note.text() == "drug on"                                       # no project: kept typed, not lost
    window.open_project(folder)
    live.add_note()
    assert any(n["text"].endswith("drug on") for n in window.project.notes())
    assert window.frame.sidebar.buttons["live"].isVisibleTo(window)
    window.project.set_feature("live_experiment", False)
    window.project_changed()
    assert not window.frame.sidebar.buttons["live"].isVisibleTo(window)


def test_live_shows_the_cell_s_health_from_its_best_source(qapp, window):
    # access resistance and the other general readouts of a cell's health; the made-up rig: the acquisition's last Membrane Test Ra 9.1 MOhm, Cm 52 pF
    window.go("live")
    live = window.screens["live"]
    assert _wait(qapp, lambda: live.last is not None and "rs" in live.health_rows)
    cap, value = live.health_rows["rs"]
    assert cap.text() == "Access (Rs)" and value.text() == "✓ 9.10 MOhm" and "Membrane Test" in value.toolTip()
    assert "at most 25 MOhm" in value.toolTip()
    assert {"rm", "rt", "cm", "rs_rm", "clamp_tau", "holding", "noise", "rs_change"} <= set(live.health_rows)


def test_a_view_pops_out_into_a_window_of_its_own_and_closing_it_puts_it_back(qapp, window):
    # the acquisition's and the camera's views as their own resizable windows, laid over each other
    window.go("live")
    live = window.screens["live"]
    view = live.views["acquisition"]
    live.toggle_popped("acquisition")
    assert view.popped and view.isWindow() and view.windowTitle() == "Acquisition (Clampwise)"
    assert not view.enlarge_button.isVisibleTo(view)                             # a window has its own buttons
    view.setWindowOpacity(0.6)
    view.close()                                                                  # closing puts it back
    assert not view.popped and not view.isWindow() and view.windowOpacity() == 1.0
    grid = live.grid
    assert grid.getItemPosition(grid.indexOf(view))[:2] == (0, 0)


def test_the_views_can_be_windows_inside_clampwise(qapp, window):
    # the views as their own windows inside the app too
    from pyqtgraph.Qt import QtWidgets as W

    window.go("live")
    live = window.screens["live"]
    live.set_free(True)
    assert live.stack.currentWidget() is live.mdi and set(live._subs) == set(live.views)
    view = live.views["acquisition"]
    assert isinstance(view.parent(), W.QMdiSubWindow) and not view.enlarge_button.isVisibleTo(view)
    live._subs["camera"].close()                                                  # closing one hides it
    assert "camera" in live._hidden and not live._subs["camera"].isVisible()
    live.set_view_shown("camera", True)
    assert live._subs["camera"].isVisible()
    live.toggle_popped("light")                                                   # out of Clampwise and back
    assert "light" not in live._subs and live.views["light"].isWindow()
    live.toggle_popped("light")
    assert "light" in live._subs
    live.set_free(False)
    assert live.stack.currentIndex() == 0 and not live._subs
    assert live.grid.getItemPosition(live.grid.indexOf(view)) == (0, 0, 1, 3)
