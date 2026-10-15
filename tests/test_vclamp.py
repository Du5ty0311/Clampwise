"""Voltage-clamp analyses (B8): the I-V of a step family and of a staircase, the
leak taken off only when it can be told from zero, the Boltzmann fit with a
typed reversal, and the tonic current as the shift a drug makes."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from clampwise import simulate
from clampwise.analysis import pipeline
from clampwise.core.recording import ArraySource, DrugEvent


def test_a_step_family_gives_its_leak_and_conductance():
    sim = simulate.vc_family()
    r = pipeline.analyze(sim.recording, pipeline.Settings(by_analysis={"vc_iv": {"e_rev_mv": -90.0}}))
    v = r.results["vc_iv"].values
    # every level but the holding one (-70 mV) is a step
    assert r.recording_type == "vc_steps" and v["n_steps"] == sum(lv != -70 for lv in sim.truth["levels_mv"])
    assert v["leak_ns"] == pytest.approx(2.0, abs=0.05) and v["leak_removed"] == "yes"
    assert v["v_half_mv"] == pytest.approx(-20.0, abs=1.0) and v["g_max_ns"] == pytest.approx(10.0, abs=0.5)
    assert v["zero_current_mv"] == pytest.approx(-70.0, abs=1.0)
    no_e = pipeline.analyze(sim.recording).results["vc_iv"]
    assert np.isnan(no_e.values["v_half_mv"]) and any("no reversal potential" in n for n in no_e.notes)


def test_a_staircase_is_read_level_by_level():
    # ear-1's protocol: one sweep stepping through its levels with no return to a holding level
    sim = simulate.vc_family(levels=(-110, -90, -70, -50, -30, -10, 10), g_max_ns=0.0)
    rec = copy.copy(sim.recording)
    fs = rec.sample_rate_hz
    seg = int(0.2 * fs)
    levels = [-70.0, -110.0, -90.0, -70.0, -50.0, -30.0, -10.0, 10.0]
    cmd = np.concatenate([np.full(seg, lv) for lv in levels])
    sig = 2.0 * (cmd + 70.0) + np.random.default_rng(1).normal(0, 1.0, cmd.size)
    rec.samples = ArraySource([sig[None, :], cmd[None, :]])
    rec.epochs = []
    rec.sweep_start_times_s = np.zeros(1)
    rec._hash = "staircase"
    r = pipeline.analyze(rec)
    assert r.recording_type == "vc_steps" and "staircase" in r.type_reason
    rows = r.results["vc_iv"].tables["steps"]
    assert [row["v_mv"] for row in rows] == levels[1:]
    assert all(row["i_steady_pa"] == pytest.approx(2.0 * (row["v_mv"] + 70.0), abs=1.0) for row in rows)


def test_the_tonic_current_is_the_shift_a_drug_makes():
    sim = simulate.spontaneous(duration_s=120.0, seed=2)
    rec = copy.copy(sim.recording)
    y = rec.signal.copy()
    t = np.arange(y.shape[1]) / rec.sample_rate_hz
    y[:, t >= 60.0] += 25.0                       # a GABA-A antagonist takes off 25 pA of tonic inward current
    rec.samples = ArraySource([y] + [rec.read(c)[...] for c in rec.channels[1:]])
    rec.metadata = copy.copy(rec.metadata)
    rec.metadata.drugs = [DrugEvent("gabazine", 60.0)]
    rec._hash = "tonic"
    r = pipeline.analyze(rec)
    v = r.results["tonic"].values
    assert v["tonic_shift"] == pytest.approx(25.0, abs=2.0) and v["tonic_drug"] == "gabazine"
    assert v["tonic_level"] == pytest.approx(-150.0 + 12.5, abs=15.0)
