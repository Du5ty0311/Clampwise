"""The WinLTP reader: a series as one recording, the pathway's stimulation as
the stimulus, real sweep times, processed copies not double-counted, a sweep
file written here read back exactly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clampwise.core import Role
from clampwise.io import group_recordings, load, recording_path

from labdata import find_all  # noqa: E402

TEST_SERIES = find_all("Test_WinLTP_Files*/Test_WinLTP_Files")


def _write_sweep(folder: Path, number: int, pathway: int = 0, processing: str = "", start_s: float = 44539.4,
                 n: int = 400, interval_ms: float = 0.05, pulse_ms: float = 5.0):
    path = folder / f"5417{number:04d}.{processing}P{pathway}"
    rows = "\n".join(f"{-0.5 + 0.1 * np.sin(i / 20):.4f}\t0.0" for i in range(n))
    text = (f'"WinLTP"\t\t"3.00A"\n"FileName"\t\t"{path.name}"\n"DateFileSaved"\t\t20250417\n'
            f'"TimeFileStarted"\t"12:22:19.4"\n"TimeStartedMidnite_s"\t{start_s}\n\n'
            f'"NumSamples"\t\t{n}\n"SampleInterval_ms"\t{interval_ms}\n'
            '"AD_Chs"\t\t"AD0"\t\t"AD1"\n"AD_DataType"\t\t"mV"\t\t""\n"AD_DigFilter_Hz"\t0\t0\n'
            '"RsRm_Stim"\t\t"IC0"\t\t"IC1"\n"RsRmDur_ms"\t\t0.0\t\t0.0\n'
            '"IC0_Stim"\t\t"Off"\n"IC1_Stim"\t\t"Off"\n'
            '"S0_Stim"\t\t"Delay0"\t"Pulses1"\t"Off2"\n'
            f'"S0_PulseDur_ms"\t0.0\t\t0.1\t\t0.0\n"S0_PulseIntvl_ms"\t{pulse_ms}\t\t0.0\t\t0.0\n'
            '"S0_NumPulses"\t\t0\t\t1\t\t0\n"S0_TrainPer_ms"\t\t0\t\t0\t\t0\n"S0_NumTrains"\t\t0\t\t0\t\t0\n'
            '"S1_Stim"\t\t"Off0"\n\n"\t\t"AD0"\t\t"AD1"\n"ms"\t\t"mV"\t\t""\n')
    body = "\n".join(f"{i * interval_ms:.3f}\t{line}" for i, line in enumerate(rows.splitlines()))
    path.write_text(text + body + "\n", encoding="latin-1")
    return path


def test_series_written_here_reads_back_with_times_and_stimulus(tmp_path):
    for k in range(3):
        _write_sweep(tmp_path, 100 + 2 * k, start_s=44539.4 + 30 * k)
    _write_sweep(tmp_path, 101, pathway=1)
    r = load(tmp_path / "54170100.P0")
    assert r.source.name == "5417.P0" and r.n_sweeps == 3 and r.n_samples == 400
    assert r.sample_rate_hz == pytest.approx(20_000.0)
    assert r.signal_units == "mV" and r.signal.shape == (3, 400)
    assert np.allclose(r.sweep_start_times_s, [0.0, 30.0, 60.0])
    assert r.start_time is not None and r.start_time.year == 2025 and r.start_time.hour == 12
    trig = r.channel(Role.TRIGGER)
    assert trig is not None and trig.name == "S0"
    assert len(r.stimulus) == 3 and np.allclose(r.stimulus.fixed_times(), [0.005])
    assert r.header["recording_kind"] == "field"
    assert recording_path(tmp_path / "54170102.P0").name == "5417.P0"


def test_processed_copies_and_pathways_are_grouped(tmp_path):
    files = [_write_sweep(tmp_path, 10 + 2 * k) for k in range(3)]
    files += [_write_sweep(tmp_path, 10 + 2 * k, processing="F") for k in range(3)]
    files += [_write_sweep(tmp_path, 11, pathway=1), _write_sweep(tmp_path, 13, pathway=1)]
    chosen, notes = group_recordings(files)
    names = sorted(p.name for p in chosen)
    assert names == ["54170010.FP0", "54170011.P1"]            # filtered preferred; P1 its own series
    assert any("copies of the same sweeps" in n for n in notes)


def test_a_lone_sweep_inside_a_series_is_an_induction_not_a_recording(tmp_path):
    files = [_write_sweep(tmp_path, 20 + 2 * k) for k in range(4)]
    train = tmp_path / "54170023.T0"
    train.write_text((tmp_path / "54170022.P0").read_text(encoding="latin-1"), encoding="latin-1")
    chosen, notes = group_recordings(files + [train])
    assert [p.name for p in chosen] == ["54170020.P0"]
    assert any("induction train" in n for n in notes)


def test_a_sweep_cut_short_is_left_out_and_said(tmp_path):
    for k in range(3):
        _write_sweep(tmp_path, 30 + 2 * k)
    short = tmp_path / "54170036.P0"
    text = (tmp_path / "54170034.P0").read_text(encoding="latin-1").splitlines()
    short.write_text("\n".join(text[:-100]) + "\n", encoding="latin-1")
    r = load(short)
    assert r.n_sweeps == 3
    assert any("cut short" in n for n in r.notes)


def test_units_are_converted_to_mv_and_pa(tmp_path):
    p = _write_sweep(tmp_path, 40)
    text = p.read_text(encoding="latin-1").replace('"AD_DataType"\t\t"mV"', '"AD_DataType"\t\t"V"').replace(
        '"ms"\t\t"mV"', '"ms"\t\t"V"')
    p.write_text(text, encoding="latin-1")
    r = load(p)
    assert r.signal_units == "mV" and abs(r.signal[0, 0] - (-500.0)) < 1e-6


@pytest.mark.skipif(not TEST_SERIES, reason="the WinLTP test series is not on this machine")
def test_the_lab_series_loads_with_real_minutes():
    folder = Path(TEST_SERIES[0])
    chosen, notes = group_recordings(list(folder.iterdir()))
    assert any(p.name.endswith(".FP0") for p in chosen)
    r = load(next(p for p in chosen if p.name.endswith(".FP0")))
    assert r.n_sweeps > 100 and r.signal_units == "mV"
    assert np.all(np.diff(r.sweep_start_times_s) > 0)
    assert r.sweep_start_times_s[-1] > 30 * 60                  # an hour-long experiment, in seconds
    assert len(r.stimulus) >= r.n_sweeps
    assert r.header["recording_kind"] == "field"
