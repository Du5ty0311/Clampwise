"""Axon Text Files: Clampfit's export, read as recordings."""

from __future__ import annotations

import numpy as np
import pytest

from clampwise.analysis import pipeline
from clampwise.core import Role
from clampwise.io import atf, is_supported, load


def write_atf(path, signals: list[tuple[str, str]], sweeps: list[list[np.ndarray]], fs: float = 10_000.0,
              mode: str = "Episodic Stimulation"):
    """sweeps[k][j]: sweep k of signal j."""
    n = sweeps[0][0].size
    t = np.arange(n) / fs
    names, titles, cols = [], ["Time (s)"], [t]
    for k, sweep in enumerate(sweeps):
        for (name, units), y in zip(signals, sweep):
            names.append(name)
            titles.append(f"Trace #{k + 1} ({units})")
            cols.append(y)
    header = [f'"AcquisitionMode={mode}"', '"Comment="',
              '"SweepStartTimesMS=' + ",".join(f"{k * 1000.0:.3f}" for k in range(len(sweeps))) + '"',
              '"Signals="\t' + "\t".join(f'"{x}"' for x in names)]
    with open(path, "w") as f:
        f.write("ATF\t1.0\n")
        f.write(f"{len(header)}\t{len(cols)}\n")
        f.write("\n".join(header) + "\n")
        f.write("\t".join(f'"{x}"' for x in titles) + "\n")
        for row in np.column_stack(cols):
            f.write("\t".join(f"{v:.6g}" for v in row) + "\n")


def test_a_current_step_family_in_text_reads_and_is_typed(tmp_path):
    fs, n = 10_000.0, 6000
    sweeps = []
    for level in (-100.0, -50.0, 0.0, 50.0, 100.0):
        i = np.full(n, 2.0)
        i[1000:4000] += level
        v = -70.0 + (i - 2.0) * 0.15                    # 150 MOhm, no capacitance
        sweeps.append([v, i])
    p = tmp_path / "steps.atf"
    write_atf(p, [("IN 0", "mV"), ("I", "pA")], sweeps, fs)
    assert is_supported(p)
    rec = load(p)
    assert rec.n_sweeps == 5 and abs(rec.sample_rate_hz - fs) < 1e-6
    assert rec.channels[0].role is Role.SIGNAL and rec.channels[1].role is Role.COMMAND
    assert "standing in for the command" in rec.channels[1].role_reason
    assert np.allclose(rec.sweep_start_times_s, np.arange(5) * 1.0)
    r = pipeline.analyze(rec)
    assert r.recording_type == "cc_steps"
    assert abs(r.results["intrinsic"].values["r_input_slope"] - 150.0) < 5.0
    # an idle headstage's current first (CRCNS hc-10's IN 2): the monitor that moves is the command
    p2 = tmp_path / "two_monitors.atf"
    write_atf(p2, [("IN 1", "mV"), ("IN 2", "pA"), ("IN 11", "pA")],
              [[v, np.full(n, 3.8) + 0.1 * np.sin(np.arange(n)), i] for v, i in sweeps], fs)
    rec2 = load(p2)
    assert rec2.channel(Role.COMMAND).name == "IN 11" and "the one that moves" in rec2.channel(Role.COMMAND).role_reason
    assert pipeline.analyze(rec2).recording_type == "cc_steps"


def test_typed_steps_for_a_file_of_the_membrane_potential_alone(tmp_path):
    # CRCNS hc-10: AxoClamp text of the voltage only, the steps in the dataset's description
    fs, n, tau = 10_000.0, 6000, 0.02
    t = np.arange(n) / fs
    sweeps = []
    for level in (-100.0, -50.0, 0.0, 50.0, 100.0):
        on = np.where(t >= 0.1, 1 - np.exp(-(t - 0.1) / tau), 0.0)
        off = np.where(t >= 0.4, 1 - np.exp(-(t - 0.4) / tau), 0.0)
        sweeps.append([-70.0 + level * 0.15 * (on - off)])         # 150 MOhm, tau 20 ms
    p = tmp_path / "vm_only.atf"
    write_atf(p, [("IN 0", "mV")], sweeps, fs)
    raw = p.read_bytes().replace(b'"Comment="', b'"Comment=\xf1"')   # a latin-1 byte, as AxoClamp wrote one
    p.write_bytes(raw)
    rec = load(p)
    assert pipeline.analyze(rec).recording_type == "unknown"
    for typed, edges in (("steps -100:50:100", True), ("steps -100:50 @ 100-400", False)):
        r = pipeline.analyze(rec, pipeline.Settings(stimulus_override=typed))
        assert r.recording_type == "cc_steps"
        assert any("read from the response" in x for x in r.recording.notes) == edges
        e = r.recording.epochs[0][1]
        assert abs(e.start_s - 0.1) < 0.002 and abs(e.end_s - 0.4) < 0.002 and e.level == -100.0
        assert abs(r.results["intrinsic"].values["r_input_slope"] - 150.0) < 5.0


def test_a_clampfit_template_and_a_stimulus_waveform_are_not_recordings(tmp_path):
    p = tmp_path / "template.atf"
    p.write_text('ATF\t1.0\n0\t2\n"Time"\t"Template"\n0\t-1\n0.1\t-2\n')
    with pytest.raises(atf.NotARecording, match="template"):
        load(p)
    q = tmp_path / "ramp.atf"
    q.write_text('ATF\t1.0\n2\t2\n"AcquisitionMode=Episodic Stimulation"\n"Signals="\t"Trace #1"\n'
                 '"Time (s)"\t"Trace #1"\n0\t0\n0.0001\t0.1\n0.0002\t0.2\n')
    with pytest.raises(atf.NotARecording, match="stimulus waveform"):
        load(q)


def test_a_lab_atf_reads_like_its_abf():
    from labdata import needs

    rec = load(needs("1781409-control-filtered.atf"))
    assert rec.n_sweeps == 36 and abs(rec.sample_rate_hz - 20_000.0) < 1e-3
    assert [c.role for c in rec.channels][:1] == [Role.SIGNAL]
    assert pipeline.analyze(rec).recording_type == "vc_evoked"
