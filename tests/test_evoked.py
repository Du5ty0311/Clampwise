"""B3, evoked responses, every pulse: known answers on simulated recordings,
and the brief's lab files (19n02012, an 8-pulse light train at 8 Hz in
current clamp; 18815022, a paired pulse with nothing time-locked to it) and
one interval series checked sweep by sweep on the raw trace."""

from __future__ import annotations


import numpy as np
import pytest

from clampwise import simulate
from clampwise.analysis import pipeline



def _evoked(sim):
    r = pipeline.analyze(sim.recording)
    e = r.results["evoked"]
    assert not e.error, e.error
    return r, e


def _true_means(sim):
    return np.array(sim.truth["per_sweep_amplitudes"]).mean(axis=0)


@pytest.mark.parametrize("isi", [0.05, 0.02])
def test_paired_pulses_ppr_is_the_ratio_of_means_with_the_tail_taken_off(isi):
    sim = simulate.evoked(onsets_s=(0.2, 0.2 + isi), amplitudes=(-50.0, -40.0), decay_s=0.008, n_sweeps=30, seed=3)
    r, e = _evoked(sim)
    truth = _true_means(sim)
    assert r.recording_type == "vc_evoked"
    assert e.values["p1_amplitude"] == pytest.approx(truth[0], rel=0.02)
    assert e.tables["pulses"][1]["amplitude"] == pytest.approx(truth[1], rel=0.02)
    assert e.values["ppr"] == pytest.approx(truth[1] / truth[0], abs=0.01)
    if isi == 0.02:                                   # without the tail model the raw ratio is far off
        assert e.values["ppr_raw"] < 0.6


def test_a_stimulus_in_few_sweeps_measures_those_sweeps():
    # most sweeps without a pulse (CRCNS hc-10 ngf_09o21004 read a noisy channel so): the pulsed ones are measured
    from clampwise.core import Role

    sim = simulate.evoked(onsets_s=(0.2,), amplitudes=(-50.0,), n_sweeps=30, seed=3)
    trigger = sim.recording.channel(Role.TRIGGER).index
    light = sim.recording.samples._arrays[trigger]
    light[[k for k in range(30) if k not in (3, 11, 20, 27)]] = 0.0      # the trigger fires in 4 sweeps of 30
    _, e = _evoked(sim)
    assert "26 sweep(s) with a different number of pulses left out" in " ".join(e.notes)
    assert e.values["p1_amplitude"] == pytest.approx(-50.0, rel=0.1)


def test_kinetics_latency_and_reliability_against_the_truth():
    sim = simulate.evoked(onsets_s=(0.2,), amplitudes=(-50.0,), decay_s=0.010, n_sweeps=30, seed=3)
    _, e = _evoked(sim)
    v = e.values
    assert 3.0 <= v["p1_latency_median"] <= 3.4            # true 3.0 ms; a threshold crossing lags a little
    assert v["p1_jitter"] == pytest.approx(0.2, abs=0.08)  # true 0.2 ms
    assert v["p1_decay_tau"] == pytest.approx(10.0, rel=0.05)
    assert v["p1_success_rate"] == 1.0 and np.isfinite(v["p1_inv_cv2"])
    assert v["time_locked_p"] < 0.05
    assert v["headline_is"].startswith("Pulse 1 amplitude (first stimulus of each sweep")
    sim = simulate.evoked(failures=0.3, n_sweeps=60, seed=8)
    _, e = _evoked(sim)
    assert e.values["p1_success_rate"] == pytest.approx(0.7, abs=0.1)


@pytest.mark.parametrize("hz", [8, 100])
def test_every_pulse_of_a_train_against_the_truth(hz):
    isi = 1.0 / hz
    sim = simulate.evoked(onsets_s=tuple(0.1 + isi * k for k in range(8)), amplitudes=tuple(-50 * 0.8 ** k for k in range(8)),
                          n_sweeps=15, sweep_s=1.4, decay_s=0.008, seed=5)
    _, e = _evoked(sim)
    truth = _true_means(sim)
    pn = np.array([p["pn_p1"] for p in e.tables["pulses"]])
    assert len(pn) == 8
    assert np.allclose(pn, truth / truth[0], atol=0.02)
    assert e.values["steady_state_ratio"] == pytest.approx(np.mean(truth[-5:]) / truth[0], abs=0.02)
    assert {"pulse", "amplitude", "pn_p1", "latency_median_ms", "rise_10_90_ms", "decay_tau_ms", "integral",
            "success_rate", "spike_probability", "amplitude_local"} <= set(e.tables["pulses"][0])


def test_interval_series_each_pulse_on_its_own_aligned_average():
    sim = simulate.interval_series(ppr=(0.6, 0.7, 0.8, 0.9))
    _, e = _evoked(sim)
    assert any("move between sweeps" in n for n in e.notes)
    assert e.values["ppr"] == pytest.approx(sim.truth["mean_ppr"], abs=0.02)


def test_current_clamp_spiking_sweeps_leave_the_psp_measures_and_count_as_fidelity():
    sim = simulate.cc_evoked_spikes()
    _, e = _evoked(sim)
    truth = _true_means(sim)
    assert e.values["n_sweeps"] == 17
    p1 = e.tables["pulses"][0]
    assert p1["spike_probability"] == pytest.approx(3 / 20)
    assert np.isfinite(p1["spike_latency_ms"])
    assert e.values["ppr"] == pytest.approx(truth[1] / truth[0], abs=0.03)
    assert any(f.code == "spikes" for f in e.flags)


def test_the_headline_is_chosen_and_says_what_it_is():
    sim = simulate.evoked(onsets_s=tuple(0.1 + 0.05 * k for k in range(6)), amplitudes=tuple(-50 * 0.8 ** k for k in range(6)),
                          n_sweeps=10, sweep_s=0.8, seed=2)
    for choice, starts in (("a chosen pulse", "Pulse 3 amplitude"), ("steady state", "Steady-state amplitude"),
                           ("train mean", "Train mean amplitude")):
        r = pipeline.analyze(sim.recording, pipeline.Settings(by_analysis={"evoked": {"headline": choice,
                                                                                        "headline_pulse": 3}}))
        e = r.results["evoked"]
        assert e.values["headline_is"].startswith(starts)
        assert "headline" in e.choices


def _lab(name):
    from clampwise.io import load
    from labdata import needs

    return load(needs(name))


def test_19n02012_every_pulse_of_the_8_hz_light_train_is_measured():
    r = pipeline.analyze(_lab("19n02012.abf"))
    e = r.results["evoked"]
    assert r.recording_type == "cc_evoked"
    assert e.values["n_pulses"] == 8 and len(e.tables["pulses"]) == 8
    assert all(np.isfinite(p["amplitude"]) for p in e.tables["pulses"])
    assert e.values["time_locked_p"] < 0.05


def test_18815022_paired_pulse_with_nothing_time_locked_is_flagged_not_blanked():
    r = pipeline.analyze(_lab("18815022.abf"))
    e = r.results["evoked"]
    assert e.values["n_pulses"] == 2
    assert any(f.code == "no response" for f in e.flags)
    assert np.isfinite(e.values["p1_amplitude"])                 # flags are evidence, never an edit


def test_an_interval_series_matches_the_raw_sweeps():
    rec = _lab("CM ChR2 13 slice 4 cell 1 doublets.abf")
    e = pipeline.analyze(rec).results["evoked"]
    # measured by hand on each raw sweep: P2 -1133 to -1328 pA, P2/P1 1.42 to 1.67
    assert -1330 <= e.tables["pulses"][1]["amplitude"] <= -1130
    assert 1.42 <= e.values["ppr"] <= 1.67
