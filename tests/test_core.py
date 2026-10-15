"""The data model: channels and roles, samples on demand, the stimulus's fixed
times, identity from paths by visible rules, and the cell clock."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from clampwise.core import (ArraySource, Channel, ClampMode, Identity, Recording, Role, Stimulus)
from clampwise.core.clock import CellClock
from clampwise.core.identity import Rule, Rules


def _recording(n_sweeps=3, n_samples=1000, fs=10_000.0, start=None, name="x.abf"):
    rng = np.random.default_rng(0)
    signal = rng.normal(-65.0, 0.5, (n_sweeps, n_samples))
    command = np.zeros((n_sweeps, n_samples))
    command[:, 200:600] = -100.0
    channels = [Channel(0, "Vm", "mV", Role.SIGNAL, True, ClampMode.CURRENT_CLAMP),
                Channel(1, "Cmd", "pA", Role.COMMAND, False)]
    return Recording(Path(name), fs, channels, ArraySource([signal, command]),
                     sweep_start_times_s=np.arange(n_sweeps) * 2.0, protocol="steps", start_time=start)


def test_channels_roles_and_units():
    r = _recording()
    assert r.clamp_mode is ClampMode.CURRENT_CLAMP and not r.is_voltage_clamp
    assert r.signal_units == "mV"
    assert r.signal.shape == (3, 1000) and r.command.shape == (3, 1000)
    assert r.sweep(1).shape == (1000,)
    assert r.n_sweeps == 3 and r.n_samples == 1000 and r.sweep_duration_s == 0.1
    assert r.duration_s == pytest.approx(4.1)
    assert r.index_at(0.05) == 500
    assert "3 sweep(s) of 0.1 s at 10000 Hz, Vm in mV (CC)" in r.describe()
    assert ClampMode.from_units("pA") is ClampMode.VOLTAGE_CLAMP
    assert ClampMode.from_units("") is ClampMode.UNKNOWN


def test_samples_are_read_on_demand_and_sliced():
    r = _recording()
    assert np.array_equal(r.read(0, [2]), r.signal[[2]])
    assert np.array_equal(r.read(r.channels[1], slice(0, 2)), r.command[:2])


def test_no_signal_channel_is_an_error_not_a_guess():
    r = _recording()
    r.channels = [Channel(0, "Vm", "mV", Role.OTHER)]
    with pytest.raises(ValueError, match="no channel is the recorded signal"):
        _ = r.signal


def test_stimulus_fixed_times_and_moving_pulses():
    fixed = Stimulus([np.array([0.1, 0.15])] * 4, [np.array([0.005, 0.005])] * 4, source="trigger channel IN 8")
    assert len(fixed) == 8 and fixed.pulses_in(2) == 2
    assert np.allclose(fixed.fixed_times(), [0.1, 0.15])
    moving = Stimulus([np.array([0.1, 0.12]), np.array([0.1, 0.2])], [np.array([0.005] * 2)] * 2)
    assert moving.fixed_times().size == 0
    assert Stimulus().fixed_times().size == 0


def test_identity_from_lab_folders_and_file_names():
    rules = Rules()
    who = rules.parse(Path(r"D:\data\A1_61826\Slice1\cell1\26618005.abf"))
    assert (who.animal, who.slice, who.cell) == ("A1_61826", "Slice1", "cell1")
    assert who.sources["cell"].startswith("folder 'cell1'")
    assert who.sources["animal"].startswith("folder 'A1_61826'")
    who = rules.parse(Path(r"D:\exp\a12_s2_c3_cc_0001.abf"))
    assert (who.animal, who.slice, who.cell, who.condition) == ("a12", "s2", "c3", "cc")
    assert who.sources["animal"].startswith("file name")
    nothing = rules.parse(Path(r"D:\exp\18815022.abf"))
    assert nothing.describe() == "(no identity)"


def test_typed_identity_wins_and_rules_are_editable():
    parsed = Rules().parse(Path(r"D:\data\A1\Slice1\cell1\x.abf"))
    merged = Rules.merge(parsed, Identity(cell="cell 7", condition="carbachol"))
    assert merged.cell == "cell 7" and merged.sources["cell"] == "typed"
    assert merged.animal == "A1" and merged.condition == "carbachol"
    custom = Rules([Rule("date animal", r"^(?P<animal>\d{6})$", "folders"), *Rules().rules])
    who = custom.parse(Path(r"D:\DSI_data\260929\Slice2\cell1\24125219.abf"))
    assert (who.animal, who.slice, who.cell) == ("260929", "Slice2", "cell1")


def test_cell_clock_orders_files_by_start_time_and_names_the_unplaced():
    t0 = dt.datetime(2026, 9, 29, 14, 0, 0)
    a = _recording(start=t0 + dt.timedelta(seconds=120), name="b.abf")
    b = _recording(start=t0, name="a.abf")
    c = _recording(start=None, name="c.abf")
    clock = CellClock.build([a, b, c])
    assert [p.recording.source.name for p in clock.placed] == ["a.abf", "b.abf"]
    assert clock.offset_of(a) == 120.0 and clock.offset_of(b) == 0.0
    assert np.allclose(clock.placed[1].sweep_times_s, [120.0, 122.0, 124.0])
    assert clock.unplaced[0][0] is c and "no start time" in clock.unplaced[0][1]
    assert np.isnan(clock.offset_of(c))
    assert any("c.abf" in line for line in clock.describe())
