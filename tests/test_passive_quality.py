"""B20 and B10: the membrane test, capacitance two ways, the junction
potential, series-resistance error, bridge balance, and the quality checks,
against known answers and against the old app."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from clampwise import simulate
from clampwise.analysis import pipeline
from clampwise.core import ljp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))


@pytest.mark.parametrize("rs, rm, cm", [(12, 200, 100), (25, 400, 60), (8, 120, 150)])
def test_membrane_test_recovers_the_circuit(rs, rm, cm):
    r = pipeline.analyze(simulate.membrane_test(rs_mohm=rs, rm_mohm=rm, cm_pf=cm).recording)
    m = r.results["membrane_test"]
    assert r.recording_type == "membrane_test"
    assert m.values["rs"] == pytest.approx(rs, rel=0.01)
    assert m.values["rm"] == pytest.approx(rm, rel=0.01)
    assert m.values["cm_tau"] == pytest.approx(cm, rel=0.01)          # two ways, both within 1%
    assert m.values["cm_charge"] == pytest.approx(cm, rel=0.01)
    assert m.values["holding"] == pytest.approx(-40.0, abs=1.0)
    assert m.values["rs_raw"] >= m.values["rs"] * 0.99
    assert "test step" in m.choices


def test_an_open_pipette_s_step_gives_no_rs_rather_than_zero():
    # a transient faster than two samples is not one the recording resolves: a lab recording (a pipette
    # before break-in, Rt 5.5 MOhm, tau 0.04 ms at 20 kHz) read Rs 1e-12 MOhm in 392 sweeps before 2026-10-14: the
    # filter-delayed peak, extrapolated back at that tau, made the onset current enormous
    import math

    import numpy as np

    from clampwise.analysis import passive

    fs, i0, i1, dv = 20_000.0, 200, 1200, 5.0
    t = np.arange(2000) / fs
    current = np.full(t.size, -250.0) + np.random.default_rng(0).normal(0, 2.0, t.size)
    current[i0:i1] += dv / 5.5e-3                                      # 5 mV over 5.5 MOhm: 909 pA
    k = np.arange(i1 - i0)
    edge = np.where(k < 3, k / 3.0, np.exp(-(k - 3) / fs / 40e-6))     # the filter's rise, then tau 0.04 ms
    current[i0:i1] += 3000.0 * edge
    m = passive.measure_step(t, current, i0, i1, dv, fs)
    assert math.isnan(m["rs_mohm"]) and math.isnan(m["rm_mohm"])
    assert m["holding_pa"] == pytest.approx(-250.0, abs=2.0)              # the holding is still read


def test_a_sweep_with_no_cell_reports_no_rs_and_says_so():
    # Rs is reported once on the cell: a step whose total resistance is under the
    # input-resistance floor (20 MOhm) is the pipette alone (a lab recording before break-in: Rt 4-5 MOhm,
    # Rs 0-1.6 MOhm from fits explaining little); the model cell's BATH read Rt 10.2 MOhm
    import math

    r = pipeline.analyze(simulate.membrane_test(rs_mohm=2, rm_mohm=4, cm_pf=300).recording)
    m = r.results["membrane_test"]
    rows = m.tables["sweeps"]
    assert all(row["no_cell"] and abs(row["rt_mohm"] - 6.0) < 0.3 for row in rows)
    assert all(math.isnan(row[k]) for row in rows for k in ("rs_mohm", "rm_mohm", "cm_tau_pf", "cm_charge_pf"))
    assert math.isnan(m.values["rs"]) and math.isnan(m.values["rm"]) and math.isnan(m.values["cm_tau"])
    assert m.values["holding"] == pytest.approx(-40.0, abs=1.0)            # the holding is still read
    assert any("no cell" in n for n in m.notes)
    cell = pipeline.analyze(simulate.membrane_test(rs_mohm=12, rm_mohm=200, cm_pf=100).recording)
    assert not any(row["no_cell"] for row in cell.results["membrane_test"].tables["sweeps"])


def test_junction_potential_matches_published_values():
    # 150 mM KCl against 150 mM NaCl: +4.3 mV (Barry & Lynch 1991)
    assert ljp.henderson({"K": 150, "Cl": 150}, {"Na": 150, "Cl": 150}).mv == pytest.approx(4.3, abs=0.2)
    # a K-gluconate internal against ACSF: about +15 mV (the commonly quoted JPCalc value)
    assert ljp.henderson(ljp.K_GLUCONATE_INTERNAL, ljp.ACSF).mv == pytest.approx(15.0, abs=1.0)
    assert ljp.henderson({"Na": 150, "Cl": 150}, {"Na": 150, "Cl": 150}).mv == pytest.approx(0.0, abs=1e-9)
    assert "not yet checked against JPCalc" in ljp.henderson({"K": 150, "Cl": 150}, ljp.ACSF).describe()
    with pytest.raises(KeyError):
        ljp.ion("unobtainium")


def test_bridge_balance_is_flagged_only_when_unbalanced():
    clean = pipeline.analyze(simulate.steps().recording).results["bridge"]
    assert not clean.flags and clean.values["jump"] < 1.0
    off = pipeline.analyze(simulate.steps(bridge_mohm=30.0).recording).results["bridge"]
    assert off.flags and off.values["unbalanced"] == pytest.approx(30.0, rel=0.1)


def test_series_resistance_error_from_a_typed_rs():
    sim = simulate.evoked(amplitudes=(-400.0,), onsets_s=(0.2,), n_sweeps=10)
    sim.recording.metadata.rs_typed_mohm = (20.0, 20.0)
    e = pipeline.analyze(sim.recording).results["rs_error"]
    assert e.values["voltage_error"] == pytest.approx(400 * 20 * 1e-3, rel=0.05)        # 8 mV
    assert any(f.code == "voltage error" for f in e.flags)
    r = pipeline.analyze(sim.recording, pipeline.Settings(by_analysis={"rs_error": {"compensation_pct": 70.0}}))
    assert r.results["rs_error"].values["voltage_error"] == pytest.approx(8.0 * 0.3, rel=0.05)
    assert not r.results["rs_error"].flags


def test_quality_passes_flags_and_says_what_it_could_not_check():
    ok = pipeline.analyze(simulate.steps().recording)
    assert ok.quality() == "passed"
    bad = pipeline.analyze(simulate.steps(v_rest=-45.0).recording)
    assert bad.quality() == "flagged"
    assert any(f.code == "rest" and "-55" in f.limit for f in bad.flags)
    vc = pipeline.analyze(simulate.evoked().recording).results["quality"]
    assert any("Series resistance not assessed" in n for n in vc.notes)
    rows = {r["check"]: r for r in vc.tables["checks"]}
    assert rows["Holding current"]["result"] == "passed" and rows["Holding current"]["source"]


def test_a_check_with_nothing_to_judge_is_not_assessed():
    # a check with no data on it is not assessed (one sweep has no drift across sweeps)
    def rows(sim):
        return {r["check"]: r for r in pipeline.analyze(sim.recording).results["quality"].tables["checks"]}

    cc = rows(simulate.evoked(n_sweeps=1, voltage_clamp=False))
    assert cc["Drift across sweeps"]["result"] == "not assessed" and "nothing to compare" in cc["Drift across sweeps"]["note"]
    assert cc["Input resistance"]["result"] == "not assessed"
    vc = rows(simulate.evoked(n_sweeps=1))
    assert vc["Holding current change"]["result"] == "not assessed"
    short = rows(simulate.gap_free_events(duration_s=5.0))
    assert short["Holding current change"]["result"] == "not assessed" and "10 s" in short["Holding current change"]["note"]
    assert rows(simulate.steps())["Drift across sweeps"]["result"] == "passed"     # several sweeps: judged as before


def test_the_dsi_preset_raises_the_holding_limit_to_500_pa():
    sim = simulate.gap_free_events(duration_s=25.0, holding_pa=-350.0)
    default = pipeline.analyze(sim.recording)
    assert default.quality() == "flagged"
    from clampwise.project.presets import BUILTIN

    preset = BUILTIN["CA3 sIPSC DSI (carbachol)"]
    with_preset = pipeline.analyze(sim.recording, pipeline.Settings(by_analysis=dict(preset.settings)))
    assert with_preset.quality() == "passed"


def test_field_recordings_get_no_quality_report():
    from clampwise.analysis import registry

    assert "field_evoked" not in registry.load()["quality"].types
