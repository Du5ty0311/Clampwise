"""Reading Clampex's files while they are written (Phase 3; B19's "done when"): a simulated cell whose Rs jumps from
12 to 18 MOhm is replayed as Clampex writes a file (the header first, its counts unfilled, then sweep after sweep),
and each sweep's test pulse is read at the first look after it is complete; a partial sweep waits; the jump is
flagged; nothing is left open, so the file can be renamed at once; a gap-free file is read in stretches."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from clampwise.io import load
from clampwise.rig import growing, replay, state
from clampwise.simulate.abf_file import membrane_test_file, write_abf

RS = [12.0] * 6 + [18.0] * 6


def test_the_simulated_file_is_read_back_as_written(tmp_path):
    rec = load(membrane_test_file(tmp_path / "cell.abf", rs_mohm=RS))
    assert rec.n_sweeps == 12 and rec.is_voltage_clamp and rec.signal_units == "pA"
    assert [round(e.level) for e in rec.epochs[0][1:4]] == [-70, -75, -70]


def test_each_sweep_is_read_as_it_is_written_and_the_rs_jump_is_flagged(tmp_path):
    (tmp_path / "rec").mkdir()
    src = membrane_test_file(tmp_path / "rec" / "26o14000.abf", rs_mohm=RS)
    target = tmp_path / "watched" / src.name
    target.parent.mkdir()
    reader = growing.Growing(target, tmp_path / "scratch")
    seen: list[list] = []

    def look(_seconds):                       # the reader looks while the writer waits for the next sweep
        if target.exists():
            seen.append(reader.look())

    replay.write_as_recorded(src, target, speed=1e9, sleep=look)
    seen.append(reader.look())
    got = [t for batch in seen for t in batch]
    assert [t.sweep for t in got] == list(range(12))
    # every sweep at the first look after it was complete: the look before sweep k+1 is written reads sweep k
    assert [len(b) for b in seen] == [0] + [1] * 12
    assert [round(t.rs_mohm) for t in got] == [12] * 6 + [18] * 6
    assert all(abs(t.cm_pf - 100) < 3 and abs(t.holding + 40) < 1 and t.clamp == "voltage clamp" for t in got)
    assert got[1].time_s == pytest.approx(0.2) and got[0].start is not None
    assert growing.rs_change(got) == pytest.approx(0.5, abs=0.02)            # flagged: over 20%
    target.rename(target.with_suffix(".moved"))                               # nothing of Clampwise's left open
    assert not list((tmp_path / "scratch").glob("*"))                         # nor any copy left behind


def test_a_partial_last_sweep_waits_and_a_header_not_yet_written_is_said(tmp_path):
    data = membrane_test_file(tmp_path / "whole.abf", rs_mohm=RS).read_bytes()
    f = tmp_path / "part.abf"
    reader = growing.Growing(f, tmp_path / "scratch")
    f.write_bytes(data[:300])
    assert reader.look() == [] and reader.why == "no ABF2 header yet"
    start, sweep = 7 * 512, 2 * 4000 * 2
    f.write_bytes(data[:start + 2 * sweep + sweep // 2])
    assert [t.sweep for t in reader.look()] == [0, 1]
    f.write_bytes(data[:start + 3 * sweep])
    assert [t.sweep for t in reader.look()] == [2]


def test_an_open_pipette_is_no_cell_and_shows_no_rs(tmp_path):
    # Rt 5 MOhm, below the quality's 20 MOhm input-resistance floor: the pipette itself, before any cell
    f = membrane_test_file(tmp_path / "bath.abf", n_sweeps=3, rs_mohm=2.0, rm_mohm=3.0, cm_pf=1.0)
    got = growing.Growing(f, tmp_path / "scratch").look()
    assert all(t.no_cell and math.isnan(t.rs_mohm) and abs(t.rt_mohm - 5.0) < 0.2 for t in got)
    cell = growing.Growing(membrane_test_file(tmp_path / "cell.abf", n_sweeps=3), tmp_path / "scratch").look()
    assert not any(t.no_cell for t in cell) and all(abs(t.rt_mohm - 212.0) < 3 for t in cell)


def test_a_seal_is_said_and_shows_no_rs(tmp_path):
    # the model cell's PATCH (10 GOhm, 2026-10-15): a 5 mV step passes 0.5 pA, under the noise: no current to measure
    f = membrane_test_file(tmp_path / "seal.abf", n_sweeps=3, rs_mohm=10_000.0, rm_mohm=10.0, cm_pf=0.5, noise=0.01)
    got = growing.Growing(f, tmp_path / "scratch").look()
    assert all(t.seal and not t.no_cell and math.isnan(t.rs_mohm) and math.isnan(t.cm_pf) for t in got)
    cell = growing.Growing(membrane_test_file(tmp_path / "cell.abf", n_sweeps=3), tmp_path / "scratch").look()
    assert not any(t.seal for t in cell)


def _iv_file(path, clamp="voltage clamp", n_sweeps=11, fs=20_000.0):
    """An IV on the model cell (Rs 10, Rm 500 MOhm, Cm 33 pF): steps of -100 + 20 a sweep (mV, or pA in current
    clamp) from the level held, 250 ms, the step's level changing from sweep to sweep as Clampex's epochs do."""
    n = int(0.5 * fs)
    pre, a, b = n // 64, n // 64 + int(0.05 * fs), n // 64 + int(0.3 * fs)
    t = (np.arange(b - a)) / fs
    rs, rm, cm = 10e6, 500e6, 33e-12
    y, other = np.zeros((n_sweeps, n)), np.zeros((n_sweeps, n))
    for s in range(n_sweeps):
        step = -100.0 + 20.0 * s
        other[s, a:b] = step
        if clamp == "voltage clamp":                 # the current: Rs-(Rm || Cm) charging, then held
            v, tau = step * 1e-3, rs * rm * cm / (rs + rm)
            y[s, a:b] = (v / (rs + rm) + (v / rs - v / (rs + rm)) * np.exp(-t / tau)) * 1e12
        else:                                        # the potential: Rs I at once, Rm I charging with Rm Cm
            i = step * 1e-12
            y[s, a:b] = (rs * i + rm * i * (1 - np.exp(-t / (rm * cm)))) * 1e3 - 30.0
            y[s, :a] = y[s, b:] = -30.0
    y += np.random.default_rng(0).normal(0, 0.05, y.shape)
    units, cmd = (("pA", "mV"), "mV") if clamp == "voltage clamp" else (("mV", "pA"), "pA")
    epochs = [(0.0, a - pre), (-100.0, b - a, 20.0), (0.0, n - b - pre)]
    return write_abf(path, np.stack([y, other]), fs=fs, units=units, epochs=epochs, command_units=cmd,
                     interval_s=1.0)


def _read_as_written(src, tmp_path):
    target = tmp_path / "watched" / src.name
    target.parent.mkdir(exist_ok=True)
    reader, got = growing.Growing(target, tmp_path / "scratch"), []
    replay.write_as_recorded(src, target, speed=1e9, sleep=lambda _s: target.exists() and got.extend(reader.look()))
    return got + reader.look()


def test_an_iv_read_sweep_by_sweep_measures_each_sweep_by_its_own_step(tmp_path):
    # the model cell test (2026-10-15): read one at a time, each sweep's copy began at sweep 1's -100 mV step,
    # so the -20 and +20 mV test steps were never found; the noise was the whole sweep's, steps and all
    (tmp_path / "rec").mkdir()
    got = _read_as_written(_iv_file(tmp_path / "rec" / "iv.abf"), tmp_path)
    assert [t.sweep for t in got] == list(range(11))
    measured = [t.sweep for t in got if math.isfinite(t.rs_mohm)]
    assert measured == [4, 6]                                     # -20 and +20 mV: within the 20 mV test step
    assert all(abs(got[k].rt_mohm - 510) < 5 and abs(got[k].rs_mohm - 10) < 1 for k in measured)
    assert all(t.noise < 0.2 for t in got)                        # the stretch before the step, not the steps


def test_current_clamp_gives_rin_and_cm_from_the_current_step(tmp_path):
    (tmp_path / "rec").mkdir()
    got = _read_as_written(_iv_file(tmp_path / "rec" / "cc.abf", "current clamp"), tmp_path)
    assert all(t.clamp == "current clamp" and abs(t.holding + 30) < 0.1 for t in got)
    assert math.isnan(got[5].rm_mohm)                             # 0 pA: no step
    rin = [t.rm_mohm for k, t in enumerate(got) if k != 5]
    assert all(abs(r - 510) < 5 for r in rin)                     # Rs + Rm, no bridge balance
    assert all(abs(t.cm_pf - 33) < 2 for k, t in enumerate(got) if k != 5)


def test_a_gap_free_file_is_read_in_stretches(tmp_path):
    fs, n = 10_000.0, int(12 * 10_000)
    current = np.random.default_rng(1).normal(-35.0, 2.0, (1, n))
    f = write_abf(tmp_path / "gf.abf", np.stack([current, np.full((1, n), -70.0)]), fs=fs, gap_free=True)
    got = growing.Growing(f, tmp_path / "scratch").look()
    assert [t.sweep for t in got] == [0, 1]                                    # two whole 5 s stretches of 12 s
    assert all(abs(t.holding + 35) < 0.5 and abs(t.noise - 2) < 0.2 and math.isnan(t.rs_mohm) for t in got)


def test_the_rig_s_look_reads_the_newest_file_of_the_watched_folder(tmp_path):
    from test_rig_multiclamp import VC_STATE, FakeCommander
    from clampwise.rig import multiclamp

    watch = state.Watch(multiclamp.Commander(FakeCommander(VC_STATE), window_check=lambda: False),
                        find_windows=lambda: {})
    watch.folder = str(tmp_path)
    membrane_test_file(tmp_path / "26o14001.abf", rs_mohm=RS)
    snap = watch.look()
    assert snap.writing.endswith("26o14001.abf") and len(snap.sweeps) == 12
    assert watch.look().sweeps == []                                           # nothing new since


def test_a_file_s_last_sweeps_are_read_before_the_next_file_is_taken(tmp_path):
    from test_rig_multiclamp import VC_STATE, FakeCommander
    from clampwise.rig import multiclamp

    watch = state.Watch(multiclamp.Commander(FakeCommander(VC_STATE), window_check=lambda: False),
                        find_windows=lambda: {})
    watch.folder = str(tmp_path)
    membrane_test_file(tmp_path / "26o14001.abf", n_sweeps=30, rs_mohm=12.0)
    first = watch.look()
    assert len(first.sweeps) == growing.MAX_PER_LOOK
    membrane_test_file(tmp_path / "26o14002.abf", n_sweeps=3, rs_mohm=12.0)    # the next file, newer
    rest = watch.look()
    assert rest.writing.endswith("26o14001.abf") and len(rest.sweeps) == 30 - growing.MAX_PER_LOOK
    nxt = watch.look()
    assert nxt.writing.endswith("26o14002.abf") and len(nxt.sweeps) == 3


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing")
def test_the_replay_tool_keeps_the_files_order_and_gaps(tmp_path):
    import datetime as dt

    rec = tmp_path / "rec"
    rec.mkdir()
    t0 = dt.datetime(2026, 10, 14, 11, 0, 0)
    for k, gap in enumerate((0, 30, 90)):
        membrane_test_file(rec / f"26o14{k:03d}.abf", n_sweeps=2, start=t0 + dt.timedelta(seconds=gap))
    assert [(round(a), p.name) for a, p in replay.plan(rec)] == [(0, "26o14000.abf"), (30, "26o14001.abf"),
                                                                  (90, "26o14002.abf")]
    waits = []
    out = replay.replay(rec, tmp_path / "watched", speed=10, sleep=waits.append, say=lambda s: None)
    assert [p.read_bytes() for p in out] == [p.read_bytes() for _, p in replay.plan(rec)]
    # the files start 30 and 90 s after the first, at 10x: 3 and 9 s after the replay began (no time passes here)
    assert any(abs(w - 3.0) < 0.5 for w in waits) and any(abs(w - 9.0) < 0.5 for w in waits)
