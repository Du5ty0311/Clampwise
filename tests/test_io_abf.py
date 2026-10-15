"""The ABF reader: mapped samples identical to pyabf's, every protocol the lab
uses, roles with reasons, the epoch table, tags, timing; and the header-only
cost of opening a file."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from clampwise.core import ClampMode, Role
from clampwise.io import abf as abf_io
from clampwise.io import find_recordings, load
from clampwise.io.pyramid import Pyramid, minmax

from labdata import DEMOS, lab_roots
from labdata import needs as _needs


def _lab_files() -> list[Path]:
    """The lab's recordings (the first folder CLAMPWISE_LAB_DATA names)."""
    roots = lab_roots()
    return find_recordings(roots[0]) if roots else []


# ---- a file written here, so the reader is tested without the lab's data

@pytest.fixture(scope="module")
def written_abf(tmp_path_factory):
    import pyabf.abfWriter

    rng = np.random.default_rng(0)
    sweeps = rng.normal(-65.0, 1.0, (4, 5000)).astype(np.float32)
    sweeps[:, 1000:3000] -= 10.0
    path = tmp_path_factory.mktemp("abf") / "written.abf"
    pyabf.abfWriter.writeABF1(sweeps, str(path), 10_000, units="mV")
    return path, sweeps


def test_abf1_written_here_reads_back(written_abf):
    path, sweeps = written_abf
    r = load(path)
    assert r.n_sweeps == 4 and r.n_samples == 5000 and r.sample_rate_hz == 10_000
    assert r.clamp_mode is ClampMode.CURRENT_CLAMP and r.signal_units == "mV"
    # within the int16 quantization of the written file
    assert np.abs(r.signal - sweeps).max() < 0.01
    assert r.header["format"] == "ABF" and r.header["abf_version"].startswith("1")
    assert r.mode == "episodic" and r.sweep_start_times_s.shape == (4,)


# ---- the lab's files

@pytest.mark.parametrize("name", ["26618005.abf", "19n02012.abf", "18815022.abf", "24125208.abf"])
def test_mapped_samples_match_pyabf_bit_for_bit(name):
    import pyabf

    path = _needs(name)
    r = load(path)
    a = pyabf.ABF(str(path))
    for ch in range(a.channelCount):
        for s in (0, r.n_sweeps - 1):
            a.setSweep(s, channel=ch)
            assert np.array_equal(r.read(ch, [s])[0], a.sweepY), f"{name} ch{ch} sweep {s}"
    assert np.array_equal(r.read(0), np.vstack([a.setSweep(s, 0) or a.sweepY for s in range(a.sweepCount)]))


def test_current_steps_file_roles_epochs_and_command():
    r = load(_needs("26618005.abf"))
    assert r.clamp_mode is ClampMode.CURRENT_CLAMP
    assert r.protocol == "RYes_ap10"
    cmd = r.channel(Role.COMMAND)
    assert cmd is not None and not cmd.recorded and "epoch table" in cmd.role_reason
    command = r.command
    assert command.shape == (r.n_sweeps, r.n_samples)
    levels = np.unique(np.round(command, 1))
    assert len(levels) > 3                                 # a step family: many distinct levels
    assert r.epochs and any(e.kind == "step" for e in r.epochs[0])
    assert np.isfinite(r.holding)
    assert r.start_time is not None and r.start_time.year >= 2017
    monitor = r.channel(Role.MONITOR)
    assert monitor is not None and monitor.units == "pA"
    assert "recorded cell or field" in r.signal_channel.role_reason


def test_light_train_file_has_a_trigger_or_light_channel():
    r = load(_needs("19n02012.abf"))
    kinds = {c.role for c in r.channels}
    assert Role.SIGNAL in kinds
    assert Role.TRIGGER in kinds or Role.LIGHT in kinds, [c.describe() for c in r.channels]
    assert r.n_sweeps >= 1 and r.sweep_duration_s > 1.0


def test_gap_free_file_is_one_long_sweep_read_on_demand():
    path = _needs("24125208.abf")
    t0 = time.perf_counter()
    r = load(path)
    opened = time.perf_counter() - t0
    assert r.is_gap_free and r.n_sweeps == 1
    assert r.is_voltage_clamp
    assert opened < 1.0, f"opening a gap-free file took {opened:.2f} s: the samples were read, not mapped"
    assert r.sweep_start_times_s.shape == (1,)
    piece = r.read(0, [0])[0][:20_000]
    assert piece.shape == (20_000,) and np.isfinite(piece).all()


def test_headers_of_many_files_open_fast():
    files = [f for f in _lab_files() if f.suffix.lower() == ".abf"][:100]
    if len(files) < 20:
        pytest.skip("fewer than 20 lab files on this machine")
    t0 = time.perf_counter()
    headers = [abf_io.read_header(f) for f in files]
    elapsed = time.perf_counter() - t0
    assert all(h.sample_rate_hz > 0 for h in headers)
    assert elapsed < 5.0 * len(files) / 100 + 1.0, f"{len(files)} headers took {elapsed:.1f} s"


def test_every_lab_protocol_loads(tmp_path):
    files = [f for f in _lab_files() if f.suffix.lower() == ".abf"]
    if not files:
        pytest.skip("no lab files on this machine")
    seen, failures = {}, []
    for f in files:
        try:
            h = abf_io.read_header(f)
        except ValueError as exc:
            failures.append(f"{f.name}: {exc}")
            continue
        if h.protocol in seen:
            continue
        seen[h.protocol] = f
        r = load(f)
        assert r.signal_channel is not None
        s = r.sweep(0)
        assert s.shape == (r.n_samples,) and np.isfinite(s).all(), f.name
    assert not failures, failures
    assert len(seen) >= 5


def test_unreadable_files_say_so():
    with pytest.raises(FileNotFoundError):
        abf_io.read_header("no_such_file.abf")


def test_pyramid_envelopes_cover_the_samples():
    rng = np.random.default_rng(0)
    y = rng.normal(size=100_000)
    y[50_000] = 50.0
    mins, maxs = minmax(y, 1000)
    assert mins.shape == (100,) and maxs.max() == 50.0
    p = Pyramid.build(y, 10_000.0, smallest=512)
    assert 2 in p.levels and max(p.levels) >= 64
    t, lo, hi, factor = p.envelope(0.0, 10.0, 1000)
    assert factor > 1 and lo.size == hi.size == t.size and hi.max() == 50.0
    assert np.all(hi >= lo)
    _, _, _, raw = p.envelope(4.9, 5.1, 4000)
    assert raw == 1                                         # zoomed in: draw the samples


def test_event_driven_sweeps_take_their_times_from_the_synch_array():
    rec = load(_needs("17815003Events_Control_10mins.abf"))
    assert rec.header["event_driven"] and rec.header["operation_mode"] == "fixed-length"
    assert rec.sweep_start_times_s.size == rec.n_sweeps == 170
    assert rec.header["event_times_recorded"] is False             # Clampfit saved them back to back
    assert any("back to back" in n for n in rec.notes)


def test_a_gap_free_file_keeps_clampex_holding_level():
    rec = load(_needs("24125219.abf"))
    assert rec.is_gap_free and rec.holding == -70.0
    assert any("a holding set on the amplifier itself is not in the file" in n for n in rec.notes)


NEX = DEMOS


def test_voltage_clamp_holding_comes_from_the_recorded_monitor_when_the_output_disagrees():
    # NeuroExpress's mEPSC test file: Clampex's output at 0 mV, the amplifier's
    # potential monitor at -70.9 mV (the holding was set on the amplifier)
    p = NEX / "Test_mEPSC_1.abf"
    if not p.is_file():
        pytest.skip("NeuroExpress's test files are not on this machine")
    rec = load(p)
    assert abs(rec.holding + 70.9) < 0.2
    assert any("as the amplifier's monitor" in n and "Clampex's output was at 0 mV" in n for n in rec.notes)


def test_a_monitor_that_agrees_with_the_output_changes_nothing():
    rec = load(_needs("24125219.abf"))           # held at -70 mV by Clampex's output
    assert rec.holding == -70.0 and not any("as the amplifier's monitor" in n for n in rec.notes)


def test_outputs_that_play_and_bits_that_toggle_are_found_as_drawing_them_found_them():
    """Opening a file asks of each output whether it plays anything and of each
    digital bit whether it toggles, without drawing every sample of a long sweep;
    the answers are those of the drawn waveforms, on random epoch tables (spans
    that overlap, run backwards or past the sweep, every epoch type, state lists
    too short to read)."""
    import warnings
    from types import SimpleNamespace

    from pyabf.waveform import EpochSweepWaveform

    def drawn_varies(table, n_sweeps, n_samples):          # the answer from the drawn waveforms
        try:
            w = abf_io._waveforms(table, n_sweeps, n_samples)
        except Exception:
            return False
        finite = w[np.isfinite(w)]
        return finite.size > 0 and float(np.ptp(finite)) > 0

    def answer(fn, *args):
        try:
            return fn(*args)
        except Exception:
            return "raises"

    rng = np.random.default_rng(7)
    types = ["Step", "Ramp", "Pulse", "Tri", "Cos", "BiPhsc", "Unknown"]
    toggling = playing = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3000):
            n_sweeps = int(rng.integers(1, 4))
            sweeps = []
            for _ in range(n_sweeps):
                size = int(rng.integers(-2, 60)) if rng.random() < 0.05 else int(rng.integers(0, 60))
                ep = EpochSweepWaveform()
                n_epochs = int(rng.integers(1, 6))
                tiled = rng.random() < 0.5 and size >= 0              # end to end over the sweep, as Clampex lays them
                edges = np.sort(rng.integers(0, size + 1, n_epochs - 1)) if tiled else None
                for e in range(n_epochs):
                    p1, p2 = (int(x) for x in rng.integers(-10, 70, 2))
                    if tiled:
                        p1, p2 = ([0] + [int(x) for x in edges])[e], ([int(x) for x in edges] + [size])[e]
                    elif rng.random() < 0.5:
                        p1, p2 = sorted((p1, p2))
                    states = [int(rng.random() < 0.2) for _ in range(8 if rng.random() < 0.97 else 3)]
                    ep.addEpoch(p1, p2, float(rng.choice([-70.0, -70.0, 0.0, 10.5])), str(rng.choice(types)),
                                int(rng.integers(0, 20)), int(rng.integers(0, 20)), states)
                ep.p2s[-1] = size                                  # the sweep's length, as pyabf's last epoch ends it
                sweeps.append(ep)
            table = SimpleNamespace(epochWaveformsBySweep=sweeps)
            n_samples = int(rng.integers(1, 70))
            new = abf_io._varies(table, n_sweeps, n_samples)
            assert new == drawn_varies(table, n_sweeps, n_samples)
            playing += new
            for bit in range(8):
                drawn = answer(lambda: float(np.ptp(abf_io._digital(table, bit, n_sweeps, n_samples))) > 0)
                assert answer(abf_io._digital_toggles, table, bit, n_sweeps, n_samples) == drawn
                toggling += drawn is True
    assert playing > 100 and toggling > 100                        # both answers were exercised
