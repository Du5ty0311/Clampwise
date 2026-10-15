"""B1 and B2: opening files and identity (naming rules kept and re-applied, typed
experiment details, folder watching, long recordings drawn without stalling),
and detection (a brief command pulse as the stimulus, the user's overrides,
sweep selection)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from clampwise import simulate
from clampwise.analysis import pipeline, stimulus
from clampwise.core.identity import Rule
from clampwise.core.recording import Epoch, parse_sweeps


def _abf(path: Path, data: np.ndarray, fs: float = 10_000.0, units: str = "pA") -> Path:
    import pyabf.abfWriter

    pyabf.abfWriter.writeABF1(np.atleast_2d(data).astype(np.float32), str(path), fs, units=units)
    return path


# ---- B2: detection

def test_a_brief_command_pulse_is_the_stimulus_of_last_resort():
    sim = simulate.steps(n_sweeps=3)
    rec = sim.recording
    cmd = rec.read(1).copy()
    cmd[:] = 0.0
    cmd[:, 2000:2010] = 4000.0                   # 0.5 ms at 20 kHz: a current pulse to make the cell fire
    from clampwise.core.recording import ArraySource

    rec.samples = ArraySource([rec.read(0), cmd])
    rec.epochs = [[Epoch("hold", 0, 0.1, 0.0), Epoch("step", 0.1, 0.1005, 4000.0), Epoch("hold", 0.1005, 1.0, 0.0)]
                  for _ in range(3)]
    s = stimulus.find_stimulus(rec)
    assert s.source == "command Command" and s.kind == "command pulse"
    assert np.allclose(s.fixed_times(), [0.1]) and "last place a stimulus can be read from" in s.reason
    assert len(stimulus.find_stimulus(simulate.steps(n_sweeps=3).recording)) == 0      # a step family is not one
    assert len(stimulus.find_stimulus(simulate.membrane_test(n_sweeps=2).recording)) == 0   # nor a test pulse


def _with_trigger(name: str, trigger: np.ndarray, protocol: str, **kw):
    """A simulated evoked recording whose trigger line is ``trigger`` on an input named ``name``."""
    from dataclasses import replace

    from clampwise.core.recording import ArraySource

    rec = simulate.evoked(n_sweeps=3, sweep_s=3.0, onsets_s=(1.078,), amplitudes=(-50.0,), **kw).recording
    rec.channels[2] = replace(rec.channels[2], name=name)
    rec.samples = ArraySource([rec.read(0), rec.read(1), trigger])
    rec.protocol = protocol
    return rec


def test_the_labs_led_trigger_on_in_8_or_in_14_is_light_whatever_its_height_or_length():
    # a lab's files: the LED trigger recorded on IN 8 through a 10x attenuator (0.45 V) or on IN 14
    # at 4.5 V, a 1 s pulse (held on), 5 ms, or 10 pulses at 200 ms; none of them was taken before 2026-10-09
    n = 3 * 20_000
    one_second = np.zeros((3, n))
    one_second[:, 21560:41560] = 4.5
    s = stimulus.find_stimulus(_with_trigger("IN 14", one_second, "blue_LED_single_pulse_NA"))
    assert s.kind == "light" and s.channel == "IN 14" and np.allclose(s.fixed_times(), [1.078])
    assert "IN 14 is the input the lab's rig records the LED trigger on" in s.reason
    assert np.allclose(np.concatenate(s.widths_s), 1.0)
    brief = np.zeros((3, n))
    brief[:, 21560:21660] = 0.45
    s = stimulus.find_stimulus(_with_trigger("IN 8", brief, "blue_LED_shortsingle_pulse_NA"))
    assert s.kind == "light" and s.channel == "IN 8" and np.allclose(s.fixed_times(), [1.078])
    r = pipeline.analyze(_with_trigger("IN 8", brief, "blue_LED_shortsingle_pulse_NA"))
    assert r.recording_type == "vc_evoked" and "evoked" in r.results
    # a line that never rises is still nothing connected, and the file says so
    s = stimulus.find_stimulus(_with_trigger("IN 8", brief * 0.02, "blue_LED_single_pulse_NA"))
    assert len(s) == 0 and "never rises above 0.009 V" in s.reason and "nothing is connected" in s.reason
    # an input named for neither light nor the lab's rig still needs the protocol to say what it is
    s = stimulus.find_stimulus(_with_trigger("IN 3", brief, "blue_LED_single_pulse_NA"))
    assert s.kind == "light" and "protocol's name" in s.reason
    assert stimulus.find_stimulus(_with_trigger("IN 3", brief, "paired_pulse")).kind == "unknown"
    # a once-per-sweep pulse where Clampex's held first 1/64 ends marks the sweep's start, even on IN 14
    marker = np.zeros((3, n))
    marker[:, n // 64: n // 64 + 40] = 4.5
    s = stimulus.find_stimulus(_with_trigger("IN 14", marker, "Blue_LED_ImageSeq"))
    assert len(s) == 0 and "marks the sweep's start" in s.reason


def test_a_ramp_of_light_is_taken_from_its_foot():
    n = 3 * 20_000
    ramp = np.zeros((3, n))
    ramp[:, 22048:58048] = np.linspace(0.0, 0.45, 36000)          # 1.8 s from 1.1024 s, as blue_LED_ramp_NA plays it
    s = stimulus.find_stimulus(_with_trigger("IN 8", ramp, "blue_LED_ramp_NA"))
    assert s.kind == "light" and "a ramp of light rising over 1800 ms to 0.45 V" in s.reason
    assert abs(s.fixed_times()[0] - 1.1024) < 0.010 and 1.75 < float(s.widths_s[0][0]) < 1.85


def test_chosen_sweeps_are_all_the_analyses_see():
    assert parse_sweeps("2-3, 7", 10) == [1, 2, 6] and parse_sweeps("", 3) == [0, 1, 2]
    assert parse_sweeps("8-", 10) == [7, 8, 9]
    with pytest.raises(ValueError):
        parse_sweeps("40-50", 10)
    sim = simulate.evoked(n_sweeps=10)
    r = pipeline.analyze(sim.recording, pipeline.Settings(shared={"sweeps": "1-4"}))
    assert r.results["evoked"].values["n_sweeps"] == 4
    assert any("sweeps chosen" in n for n in r.notes)


def test_overrides_and_typed_details_change_the_cache_key(tmp_path):
    from clampwise.core import ExperimentMetadata
    from clampwise.project import Project

    path = _abf(tmp_path / "a.abf", np.random.default_rng(0).normal(-50, 2, (3, 20_000)))
    p = Project.open(tmp_path)
    p.add_recordings([path])
    first = p.pipeline_settings(path).shared
    p.set_overrides(path, type_override="vc_spontaneous", stimulus_override="200 @ 1")
    second = p.pipeline_settings(path).shared
    assert second["type_override"] == "vc_spontaneous" and second["stimulus_override"] == "200 @ 1"
    meta = ExperimentMetadata()
    meta.rs_typed_mohm = (18.0, 22.0)
    p.set_metadata(path, meta)
    third = p.pipeline_settings(path).shared
    assert "metadata" in third and first != second != third
    r = p.analyze(path)
    assert r.recording_type == "vc_spontaneous" and "set for this recording" in r.type_reason


# ---- B1: identity, watching, long recordings

def test_naming_rules_are_kept_and_reapplied_keeping_what_was_typed(tmp_path):
    from clampwise.core import Identity
    from clampwise.project import Project

    d = tmp_path / "exp" / "rat7" / "Slice2" / "cell1"
    d.mkdir(parents=True)
    a = _abf(d / "rec.abf", np.zeros((3, 2000)))
    p = Project.open(tmp_path / "exp")
    p.add_recordings([a])
    assert (p.identity(a).animal, p.identity(a).cell) == ("rat7", "cell1")
    p.set_identity(a, Identity(condition="carbachol"))
    changed = p.set_rules([Rule("animal folder", r"^(?P<animal>rat\d+)$"), Rule("slice", r"^(?P<slice>slice\s*\d+)$"),
                           Rule("cell", r"^(?P<cell>cell\s*\d+)$")])
    assert changed == 0 and p.identity(a).condition == "carbachol"          # typed stays
    p.set_rules([Rule("cell as slice", r"^(?P<slice>cell\d+)$")])
    assert p.identity(a).slice == "cell1" and p.identity(a).condition == "carbachol"
    p.close()
    again = Project.open(tmp_path / "exp")
    assert [r.name for r in again.rules.rules] == ["cell as slice"]          # kept in the project
    with pytest.raises(Exception):
        again.set_rules([Rule("bad", r"(?P<cell>")])


def test_the_watcher_takes_a_file_once_clampex_has_finished_it(tmp_path):
    from clampwise.io.live import FolderWatcher

    _abf(tmp_path / "old.abf", np.zeros((3, 2000)))
    w = FolderWatcher(tmp_path, settle_s=1.0)
    assert w.poll(now=0.0) == []                                              # what was there is not new
    growing = tmp_path / "new.abf"
    growing.write_bytes(b"\0" * 100)
    assert w.poll(now=1.0) == []                                              # first seen
    growing.write_bytes(b"\0" * 200)
    assert w.poll(now=2.5) == []                                              # still growing
    _abf(growing, np.zeros((3, 2000)))
    assert w.poll(now=4.0) == []
    assert w.poll(now=5.5) == [growing.resolve()]                             # stopped growing, and readable
    assert w.poll(now=7.0) == []


def test_a_long_sweep_is_drawn_from_its_envelope_without_stalling(tmp_path, qapp=None):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from clampwise.app import theme
    from clampwise.app.window import MainWindow

    path = _abf(tmp_path / "long.abf", np.random.default_rng(1).normal(-60, 3, (1, 3_000_000)))
    theme.apply(app, theme.LIGHT)
    w = MainWindow(theme.LIGHT, persist=False)
    w.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    w.show()
    w.open_project(tmp_path)
    w.go("check")
    t0 = time.perf_counter()
    w.select(path)
    assert time.perf_counter() - t0 < 0.5
    check = w.screens["check"]
    end = time.time() + 20
    while time.time() < end and not (check._long and check._long.is_ready):
        app.processEvents()
        time.sleep(0.01)
    assert check._long is not None and check._long.is_ready
    x, y = check._long.item.getData()
    assert 0 < x.size < 20_000                                    # the envelope, not 3 million points
    check.plot.setXRange(100.0, 100.01, padding=0)
    app.processEvents()
    x, _ = check._long.item.getData()
    assert x.size == pytest.approx(100, abs=5)                    # zoomed in: the samples themselves
    w.close()


def test_the_naming_rules_dialog_previews_and_refuses_a_bad_pattern(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from clampwise.app.dialogs import RulesDialog
    from clampwise.core.identity import Rules

    path = tmp_path / "rat3" / "Slice1" / "cell2" / "r.abf"
    dlg = RulesDialog(None, Rules(), [path], tmp_path)
    assert "cell <b>cell2</b>" in dlg.preview.text() and "animal <b>rat3</b>" in dlg.preview.text()
    dlg.table.item(0, 2).setText("(?P<slice>")
    app.processEvents()
    rules, problem = dlg.rules()
    assert "not valid" in problem and "not valid" in dlg.preview.text()
