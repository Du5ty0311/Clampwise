"""The cell's health readings: each from the best source, said with it, checked against the
quality limits and marked, never changed."""

from __future__ import annotations

import math

import pytest

from clampwise.rig import growing, health, labbook, multiclamp


def _amp(mode=0, whole=False, meters=None):
    return multiclamp.AmplifierState(device=multiclamp.Device(1, "00123456", 0, 0, 1), connected=True, mode=mode,
                                     values={"rs": 13.9, "cm": 18.0, "whole_cell_on": whole, "bridge": 15.0,
                                             "bridge_on": True}, meters=meters or {0: -0.065, 1: -150e-12})


def _sweeps(rs=(12.0, 12.1, 11.9, 15.0)):
    return [growing.SweepTest("26o14000.abf", k, k * 10.0, "voltage clamp", r, 200.0, 100.0, -45.0, 3.0,
                              rt_mohm=r + 200.0) for k, r in enumerate(rs)]


def _by_key(rows):
    return {r.key: r for r in rows}


def test_the_recorded_test_pulse_comes_first_and_every_reading_says_its_source():
    rows = _by_key(health.readings(_amp(), _sweeps()))
    assert rows["rs"].value == 15.0 and "recorded test pulse (26o14000.abf, sweep 4)" in rows["rs"].source
    assert rows["rs_change"].value == pytest.approx(25.0) and rows["rs_change"].status == "warn"     # over 20%
    assert rows["rs"].status == "ok" and rows["rm"].status == "ok"
    assert rows["rs_rm"].value == pytest.approx(0.075) and rows["clamp_tau"].value == pytest.approx(1.5)
    assert rows["holding"].value == pytest.approx(-150.0) and "current meter" in rows["holding"].source
    assert rows["noise"].value == 3.0 and rows["noise"].status == "ok"
    assert rows["rs"].shown() == "15.0 MOhm" and rows["rs_change"].shown() == "+25 %"


def test_without_a_recorded_pulse_the_membrane_test_then_the_amplifier():
    test = labbook.MembraneTest(1, {"ra_mohm": 28.0, "rm_mohm": 150.0, "rt_mohm": 178.0, "cm_pf": 60.0})
    rows = _by_key(health.readings(_amp(), [], test))
    assert rows["rs"].value == 28.0 and rows["rs"].source == "the acquisition's last Membrane Test"
    assert rows["rs"].status == "warn"                                                             # over 25 MOhm
    rows = _by_key(health.readings(_amp(whole=True), [], None))
    assert rows["rs"].value == 13.9 and "compensation" in rows["rs"].source
    rows = _by_key(health.readings(_amp(whole=False), [], None))
    assert math.isnan(rows["rs"].value) and rows["rs"].status == "muted" and rows["rs"].shown() == "–"


def test_current_clamp_reads_the_membrane_potential_and_the_bridge():
    rows = _by_key(health.readings(_amp(mode=1, meters={0: -0.071, 1: 0.0}), []))
    assert rows["vm"].value == pytest.approx(-71.0) and rows["vm"].status == "ok"
    assert rows["bridge"].value == 15.0
    rows = _by_key(health.readings(_amp(mode=1, meters={0: -0.040, 1: 0.0}), []))
    assert rows["vm"].status == "warn"                                                             # depolarized


def test_current_clamp_takes_rin_and_cm_from_the_recorded_current_step():
    cc = [growing.SweepTest("26o15003.abf", k, k * 1.0, "current clamp", rm_mohm=r, cm_pf=36.0, holding=-28.5)
          for k, r in enumerate((427.0, math.nan))]                     # the IV's 0 pA sweep: no step
    rows = _by_key(health.readings(_amp(mode=1), cc))
    assert rows["rm"].value == 427.0 and "current step (26o15003.abf, sweep 1)" in rows["rm"].source
    assert rows["cm"].value == 36.0 and rows["cm"].source.startswith("tau / Rin")
