"""Acquisition in Clampwise: a Clampex protocol is read from its own header, its outputs built
as the Digidata's codes, and a recording written as Clampex writes one, read back by the same loader."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from clampwise.io import load
from clampwise.rig import acquire, growing, telegraph
from clampwise.simulate.abf_file import membrane_test_file

CAL = {"dac_gains": [0.998883, 0.998994, 0.999105, 0.998939], "dac_offsets": [-3, -1, -2, -2]}


def _protocol(tmp_path, n_sweeps=4):
    """A protocol file as Clampex saves one: a data file's header without its samples."""
    f = membrane_test_file(tmp_path / "sim.abf", n_sweeps=n_sweeps, holding_mv=-70.0, step_mv=-5.0)
    head = f.read_bytes()
    block = growing._section(head, growing.DATA)[0]
    pro = tmp_path / "sim.pro"
    pro.write_bytes(head[:block * growing.BLOCK])
    return acquire.read_protocol(pro)


def test_a_protocol_is_read_from_its_header(tmp_path):
    pr = _protocol(tmp_path)
    assert pr.mode == growing.EPISODIC and pr.sweeps == 4 and pr.rate_hz == pytest.approx(20_000)
    assert pr.ai == [0, 1] and pr.sweep_frames == 4000 and pr.dac_scale[0] == 20.0 and pr.holding[0] == -70.0


def test_the_outputs_are_the_epochs_in_the_converter_s_codes(tmp_path):
    pr = _protocol(tmp_path)
    frames = acquire.output_frames(pr, CAL)
    assert frames.shape == (pr.run_frames(), acquire.N_OUT)
    v = (frames[:, 0].astype(float) + CAL["dac_offsets"][0]) / (acquire.COUNTS_PER_VOLT * CAL["dac_gains"][0])
    mv = v * pr.dac_scale[0]
    assert mv[0] == pytest.approx(-70.0, abs=0.01)                       # holding, the first 1/64
    assert np.min(mv[:pr.sweep_frames]) == pytest.approx(-75.0, abs=0.01)  # the test step
    assert set(frames[:, 1].tolist()) == {int(acquire.dac_codes(0.0, CAL["dac_gains"][1], CAL["dac_offsets"][1]))}
    assert int(acquire.dac_codes(10.5, 1.0, 0)) == 32767 and int(acquire.dac_codes(-1.0, 1.0, -3)) == -3274


def _telegraph(mode="VC", command=20.0):
    vc = mode == "VC"
    return telegraph.Telegraph(mode, "pA" if vc else "mV", 5e-5 if vc else 0.01, 1.0, 10_000.0,
                               "mV" if vc else "pA", 0.01 if vc else 5e-4, 1.0, command, "00123456", 1, 0.0, 0.0)


def test_the_command_s_scale_and_the_inputs_come_from_the_amplifier_s_telegraph(tmp_path):
    pr = _protocol(tmp_path)
    vc, note = acquire.with_telegraph(pr, _telegraph("VC", 20.0))
    assert note == "" and vc.dac_scale[0] == 20.0
    ic, note = acquire.with_telegraph(pr, _telegraph("IC", 400.0))         # the same numbers, read as pA
    assert ic.dac_scale[0] == 400.0 and ic.dac_units[0] == "pA" and "saved in mV" in note
    a, b = (acquire.output_frames(p, CAL)[0, 0] + CAL["dac_offsets"][0] for p in (vc, ic))
    assert abs(int(b)) == pytest.approx(abs(int(a)) / 20, abs=1)          # -70 at 400/V is 1/20 of -70 at 20/V
    for tg in (None, _telegraph("VC", 0.0)):
        with pytest.raises(ValueError, match="not played"):
            acquire.with_telegraph(pr, tg)
    rng = np.random.default_rng(1)
    out = tmp_path / "ic.abf"
    out.write_bytes(acquire.abf_bytes(ic, rng.integers(-500, 500, size=(ic.run_frames(), 2)).astype("<i2")))
    rec = load(out)                                                       # the file says what was played
    assert rec.signal_units == "mV" and not rec.is_voltage_clamp


def test_a_recording_is_written_as_clampex_writes_one_and_read_back(tmp_path):
    pr = _protocol(tmp_path, n_sweeps=3)
    rng = np.random.default_rng(0)
    samples = rng.integers(-2000, 2000, size=(pr.run_frames(), 2)).astype("<i2")
    start = dt.datetime(2026, 10, 15, 11, 30, 0)
    out = tmp_path / "26o15100.abf"
    out.write_bytes(acquire.abf_bytes(pr, samples, start))
    rec = load(out)
    assert rec.n_sweeps == 3 and rec.n_samples == pr.sweep_frames and rec.start_time == start
    step = max(pr.interval_frames, pr.sweep_frames)
    raw = np.frombuffer(out.read_bytes(), "<i2", offset=growing._section(out.read_bytes(), growing.DATA)[0] * 512,
                        count=3 * pr.per_episode).reshape(3, pr.sweep_frames, 2)
    for k in range(3):                                                    # each sweep cut at its start
        assert np.array_equal(raw[k], samples[k * step:k * step + pr.sweep_frames])
    assert [round(e.level) for e in rec.epochs[1][1:4]] == [-70, -75, -70]
