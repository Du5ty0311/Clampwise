"""What the signal shows that no channel records: a
stimulator's artifact taken as the stimulus, an unrecorded test pulse kept out
of the time-locked test, and none of it read into action potentials or a step
family's own response."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from clampwise import simulate
from clampwise.analysis import artifacts, pipeline
from clampwise.core import Role
from clampwise.core.recording import ArraySource


def unrecorded(fs: float = 20_000.0, pulse: bool = True):
    """pvc-9 as it was recorded: an extracellular stimulator Clampex never saw (its
    capacitive spike at 200 ms), a response 3 ms later, a -20 pA test pulse from
    the amplifier at 500-700 ms with its transients, and no trigger channel."""
    sim = simulate.evoked(onsets_s=(0.2,), amplitudes=(-50.0,), n_sweeps=20, fs=fs, seed=4)
    rec = copy.copy(sim.recording)
    y = rec.signal.copy()
    i = int(0.2 * fs)
    y[:, i] += 300.0                                  # the stimulator's spike: up, then down past, then back
    y[:, i + 1] -= 120.0
    if pulse:
        t = np.arange(y.shape[1]) / fs
        a, b = int(0.5 * fs), int(0.7 * fs)
        y[:, a:b] -= 20.0
        tail = np.exp(-np.arange(int(0.002 * fs)) / (0.0003 * fs))
        y[:, a:a + tail.size] -= 400 * tail
        y[:, b:b + tail.size] += 400 * tail
        assert t.size == y.shape[1]
    rec.channels = [c for c in rec.channels if c.role is not Role.TRIGGER]
    rec.samples = ArraySource([y, rec.read(rec.channel(Role.COMMAND))])
    rec._hash = "unrecorded"
    return rec


def test_an_unrecorded_stimulus_is_found_from_its_artifact_and_the_test_pulse_is_kept_out_of_the_sham_test():
    rec = unrecorded()
    found = artifacts.find(rec)
    kinds = sorted(a.kind for a in found)
    assert kinds == ["stimulus artifact", "test pulse"], [a.describe() for a in found]
    stim = next(a for a in found if a.kind == "stimulus artifact")
    pulse = next(a for a in found if a.kind == "test pulse")
    assert abs(stim.start_s - 0.2) < 1e-4
    assert abs(pulse.start_s - 0.5) < 0.002 and abs(pulse.end_s - 0.7) < 0.002 and pulse.size == pytest.approx(-20, 1)
    r = pipeline.analyze(rec)
    assert r.recording_type == "vc_evoked" and r.stimulus_summary
    e = r.results["evoked"]
    assert e.values["p1_amplitude"] == pytest.approx(-50.0, rel=0.1)
    assert not any(f.code == "no response" for f in r.flags)
    assert any("sham times keep clear of test pulse" in n for n in e.notes)
    a = r.results["artifacts"]
    assert a.values["n_stimulus_artifacts"] == 1 and a.values["n_test_pulses"] == 1
    assert {f.code for f in a.flags} >= {"unrecorded test pulse", "stimulus from its artifact"}


def test_no_stimulus_is_read_into_action_potentials_or_a_step_family():
    for rec in (simulate.firing_steps(pattern="tonic").recording, simulate.cc_evoked_spikes().recording):
        found = artifacts.find(rec)
        assert not [a for a in found if a.kind in ("stimulus artifact", "test pulse")], [a.describe() for a in found]
    steps = artifacts.find(simulate.firing_steps().recording)
    assert [a.kind for a in steps] == ["command change", "command change"]
    one = copy.copy(unrecorded(pulse=False))
    one.samples = ArraySource([one.signal[:1], one.read(one.channel(Role.COMMAND))[:1]])
    assert [a.kind for a in artifacts.find(one)] == ["artifact"]      # one sweep cannot show it repeats
