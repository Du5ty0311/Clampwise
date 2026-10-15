"""Every analysis against simulations with a known answer.

    python tools/audit/known_answers.py             # every section
    python tools/audit/known_answers.py baseline    # one or more sections

Each section simulates recordings whose true values are exact
(clampwise.simulate), analyzes them the way the app does, and prints measured
against true with the error and whether it is within tolerance. The test
suite runs the same sections (tests/test_known_answers.py), so a method that
drifts fails a test, not only this script.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np


@dataclass
class Row:
    section: str
    case: str
    measure: str
    truth: float
    measured: float
    tolerance: float            # absolute, in the measure's units
    note: str = ""

    @property
    def error(self) -> float:
        return self.measured - self.truth

    @property
    def ok(self) -> bool:
        return np.isfinite(self.measured) and abs(self.error) <= self.tolerance


def baseline() -> list[Row]:
    from clampwise import simulate
    from clampwise.analysis import pipeline

    rows = []
    for v_rest, noise in ((-65.0, 0.3), (-55.0, 1.0), (-80.0, 0.1)):
        sim = simulate.steps(v_rest=v_rest, noise=noise, n_sweeps=9)
        r = pipeline.analyze(sim.recording).results["baseline"]
        case = f"steps, rest {v_rest:g} mV, noise {noise:g}"
        rows.append(Row("baseline", case, "level (mV)", v_rest, r.values["level"], 3 * noise / 10 + 0.05))
        rows.append(Row("baseline", case, "noise_sd (mV)", noise, r.values["noise_sd"], 0.15 * noise + 0.02))
        rows.append(Row("baseline", case, "drift_range (mV)", 0.0, r.values["drift_range"], 3 * noise / 5 + 0.05))
    sim = simulate.gap_free_events(duration_s=40.0, holding_pa=-120.0, noise=3.0, rate_hz=2.0)
    r = pipeline.analyze(sim.recording).results["baseline"]
    rows.append(Row("baseline", "gap-free, holding -120 pA", "level (pA)", -120.0, r.values["level"], 3.0,
                    "the median sits inside the events' tails a little"))
    return rows


def evoked() -> list[Row]:
    from clampwise import simulate
    from clampwise.analysis import pipeline

    rows = []
    for isi in (0.1, 0.05, 0.02, 0.01):
        sim = simulate.evoked(onsets_s=(0.2, 0.2 + isi), amplitudes=(-50.0, -40.0), decay_s=0.008, n_sweeps=30, seed=3)
        e = pipeline.analyze(sim.recording).results["evoked"]
        truth = np.array(sim.truth["per_sweep_amplitudes"]).mean(axis=0)
        case = f"paired pulses {isi * 1e3:.0f} ms, decay 8 ms"
        rows.append(Row("evoked", case, "P1 (pA)", truth[0], e.values["p1_amplitude"], 0.02 * abs(truth[0])))
        rows.append(Row("evoked", case, "P2 (pA)", truth[1], e.tables["pulses"][1]["amplitude"], 0.02 * abs(truth[1])))
        rows.append(Row("evoked", case, "PPR", truth[1] / truth[0], e.values["ppr"], 0.01))
    sim = simulate.evoked(onsets_s=(0.2,), amplitudes=(-50.0,), decay_s=0.010, n_sweeps=30, seed=3)
    e = pipeline.analyze(sim.recording).results["evoked"]
    rows.append(Row("evoked", "single pulse", "latency median (ms)", 3.0, e.values["p1_latency_median"], 0.4,
                    "a threshold crossing lags the true onset"))
    rows.append(Row("evoked", "single pulse", "jitter (ms)", 0.2, e.values["p1_jitter"], 0.08))
    rows.append(Row("evoked", "single pulse", "decay tau (ms)", 10.0, e.values["p1_decay_tau"], 0.5))
    for hz in (8, 20, 50, 100):
        isi = 1.0 / hz
        sim = simulate.evoked(onsets_s=tuple(0.1 + isi * k for k in range(8)),
                              amplitudes=tuple(-50 * 0.8 ** k for k in range(8)), n_sweeps=15, sweep_s=1.4,
                              decay_s=0.008, seed=5)
        e = pipeline.analyze(sim.recording).results["evoked"]
        truth = np.array(sim.truth["per_sweep_amplitudes"]).mean(axis=0)
        rows.append(Row("evoked", f"8 pulses at {hz} Hz", "P8/P1", truth[7] / truth[0], e.tables["pulses"][7]["pn_p1"], 0.02))
    sim = simulate.interval_series(ppr=(0.6, 0.7, 0.8, 0.9))
    e = pipeline.analyze(sim.recording).results["evoked"]
    rows.append(Row("evoked", "interval series 12-100 ms", "PPR (pooled)", sim.truth["mean_ppr"], e.values["ppr"], 0.02))
    sim = simulate.evoked(failures=0.3, n_sweeps=60, seed=8)
    e = pipeline.analyze(sim.recording).results["evoked"]
    rows.append(Row("evoked", "30% failures", "success rate", 0.7, e.values["p1_success_rate"], 0.1))
    return rows


def match_events(found_s, true_s, tol_s: float = 0.002) -> tuple[float, int]:
    """(share of true events found within ``tol_s``, found events matching none)."""
    found = np.sort(np.asarray(found_s, float))
    used = np.zeros(found.size, bool)
    hits = 0
    for t in np.sort(np.asarray(true_s, float)):
        j = int(np.searchsorted(found, t - tol_s))
        while j < found.size and found[j] <= t + tol_s:
            if not used[j]:
                used[j] = True
                hits += 1
                break
            j += 1
    return hits / max(len(true_s), 1), int((~used).sum())


def kernel_rise_10_90_ms(rise_s: float, decay_s: float) -> float:
    """The 10-90% rise of (1 - e^-t/rise) e^-t/decay, computed finely."""
    t = np.arange(0, 10 * decay_s, rise_s / 2000)
    y = (1 - np.exp(-t / rise_s)) * np.exp(-t / decay_s)
    p = int(np.argmax(y))
    y = y / y[p]
    return float((t[np.searchsorted(y[:p], 0.9)] - t[np.searchsorted(y[:p], 0.1)]) * 1e3)


EVENT_CASES = (
    # case, simulator arguments, events settings, (amplitude tolerance as a fraction)
    ("sEPSCs 5 Hz, 0.5/10 ms, noise 3 pA", dict(), {"template_decay_ms": 10.0}),
    ("default template on 10 ms decays", dict(), {}),
    ("slow sIPSCs 25 ms decay, 3 Hz", dict(rate_hz=3.0, decay_s=0.025, amp=-35.0),
     {"template_decay_ms": 25.0}),
    ("outward IPSCs held at 0 mV", dict(amp=25.0, holding_mv=0.0, baseline=200.0), {"template_decay_ms": 10.0}),
    ("fast mEPSCs 0.2/2 ms at 50 kHz", dict(fs=50_000.0, rise_s=0.0002, decay_s=0.002, amp=-15.0, noise=2.0,
                                          duration_s=30.0), {"template_rise_ms": 0.2, "template_decay_ms": 2.0}),
    ("sEPSPs in current clamp", dict(voltage_clamp=False, baseline=-68.0, amp=1.5, noise=0.15, rise_s=0.0015,
                                     decay_s=0.015, rate_hz=3.0), {"template_rise_ms": 1.5, "template_decay_ms": 15.0}),
    ("template matching, 2 Hz", dict(rate_hz=2.0), {"method": "template matching", "template_decay_ms": 10.0}),
)


def events() -> list[Row]:
    from clampwise import simulate
    from clampwise.analysis import pipeline

    rows = []
    for case, sim_kw, settings in EVENT_CASES:
        kw = dict(duration_s=60.0, seed=11)
        kw.update(sim_kw)
        sim = simulate.spontaneous(**kw)
        r = pipeline.analyze(sim.recording, pipeline.Settings(by_analysis={"events": settings}))
        e = r.results["events"]
        found = [x["onset_s"] for x in e.tables["events"] if x["accepted"]]
        recall, false = match_events(found, sim.truth["times_s"], 0.002 if kw.get("voltage_clamp", True) else 0.004)
        dur = sim.truth["duration_s"]
        amps = sim.truth["amplitudes"]
        rows.append(Row("events", case, "found (share of true)", 1.0, recall, 0.05))
        rows.append(Row("events", case, "false events per minute", 0.0, false / dur * 60, 3.0))
        rows.append(Row("events", case, "frequency (Hz)", len(amps) / dur, e.values["frequency"], 0.06 * len(amps) / dur))
        rows.append(Row("events", case, "amplitude mean", float(np.mean(amps)), e.values["amplitude_mean"],
                        0.06 * abs(float(np.mean(amps))), "the events' noise and overlap"))
        if "decay" in case or "template" in case or "sEPSCs 5" in case or "slow" in case or "fast" in case \
                or "current clamp" in case or "outward" in case:
            rise = kernel_rise_10_90_ms(sim.truth["rise_s"], sim.truth["decay_s"])
            rows.append(Row("events", case, "averaged rise 10-90 (ms)", rise, e.values["rise_10_90"], 0.2 * rise + 0.05,
                            "alignment jitter widens the averaged rise a little"))
            rows.append(Row("events", case, "averaged decay tau (ms)", sim.truth["decay_s"] * 1e3, e.values["decay_tau"],
                            0.08 * sim.truth["decay_s"] * 1e3))
    for process, kw, label in (("poisson", {}, "random (Poisson-like) at the rate it has"),
                               ("poisson", {"rate_steps": [(0, 2.0), (20, 10.0), (40, 4.0)]},
                                "random (Poisson-like) at the rate it has"),
                               ("bursty", {}, "bursty"), ("regular", {"rate_hz": 8.0}, "more regular than random")):
        sim = simulate.spontaneous(duration_s=60.0, seed=11, process=process, **kw)
        e = pipeline.analyze(sim.recording, pipeline.Settings(by_analysis={"events": {"template_decay_ms": 10.0}}))
        v = e.results["events"].values
        case = f"timing: {process}{' with the rate stepping 2-10-4 Hz' if 'rate_steps' in kw else ''}"
        rows.append(Row("events", case, f"reads '{label.split(' (')[0]}'", 1.0, float(v["timing"] == label), 0.0,
                        f"read: {v['timing']} (p {v['timing_p']:.3f}, CV {v['iei_cv']:.2f}, LV {v['iei_lv']:.2f})"))
    return rows



def events_neuroexpress() -> list[Row]:
    """NeuroExpress's mini additions against known answers: the whole-event fit,
    the stability statements, its acceptance criteria (reported, off)."""
    from clampwise import simulate
    from clampwise.analysis import pipeline

    rows = []
    st = pipeline.Settings(by_analysis={"events": {"event_fit": "whole event"}})
    sim = simulate.spontaneous(duration_s=120.0, rate_hz=4.0, rise_s=0.0005, decay_s=0.008, noise=2.0, seed=11)
    r = pipeline.analyze(sim.recording, st).results["events"]
    ok = [e for e in r.tables["events"] if e["accepted"] and np.isfinite(e["rise_tau_ms"])]
    rows.append(Row("events+", "whole-event fit, 0.5 / 8 ms", "rise tau (ms, median)", 0.5,
                    float(np.median([e["rise_tau_ms"] for e in ok])), 0.15, f"{len(ok)} events fitted"))
    rows.append(Row("events+", "whole-event fit, 0.5 / 8 ms", "decay tau (ms, median)", 8.0,
                    float(np.median([e["decay_tau_whole_ms"] for e in ok])), 1.0))
    said = {e["statement"] for e in r.tables["evaluation"]}
    rows.append(Row("events+", "a steady 4 Hz for 120 s", "counts said stable", 1.0,
                    float("Event counts are stable" in said), 0.0, "; ".join(sorted(said))))
    rows.append(Row("events+", "a steady 4 Hz for 120 s", "amplitudes said stable", 1.0,
                    float("Event amplitudes are stable" in said), 0.0))
    down = simulate.spontaneous(duration_s=120.0, rate_hz=4.0, rate_steps=((0.0, 6.0), (40.0, 3.0), (80.0, 1.5)),
                                decay_s=0.008, noise=2.0, seed=12)
    r = pipeline.analyze(down.recording).results["events"]
    said = {e["statement"] for e in r.tables["evaluation"]}
    rows.append(Row("events+", "a rate running down 6, 3, 1.5 Hz", "counts said to fluctuate or drop", 1.0,
                    float("Event counts fluctuate or drop" in said), 0.0, "; ".join(sorted(said))))
    sim = simulate.spontaneous(duration_s=60.0, rate_hz=4.0, amp=-30.0, decay_s=0.008, noise=3.0, seed=13)
    r = pipeline.analyze(sim.recording).results["events"]
    true_above = int(np.sum(np.abs(sim.truth["amplitudes"]) >= r.values["noise_pp"]))
    rows.append(Row("events+", "NeuroExpress's criteria, reported", "events kept by them", float(true_above),
                    float(r.values["n_events_nex_criteria"]), max(3.0, 0.1 * true_above),
                    f"truth: the simulated events at least the noise's peak-to-peak ({r.values['noise_pp']:.1f} pA)"))
    return rows



def spike_train() -> list[Row]:
    """Spike trains (B24) from files of spike times whose truth is exact."""
    import tempfile

    from clampwise import simulate
    from clampwise.analysis import pipeline
    from clampwise.io import load

    rows = []
    folder = Path(tempfile.mkdtemp(prefix="cw_trains_"))

    def train(name: str, times: np.ndarray):
        p = folder / f"{name}.asc"
        p.write_text("\n".join(f"{x:.6f}" for x in times))
        return pipeline.analyze(load(p)).results["spike_train"]

    # 50 bursts of 5 spikes 20 ms apart, every 3 s
    t = np.concatenate([1.0 + 3.0 * j + 0.02 * np.arange(5) for j in range(50)])
    r = train("bursts", t)
    for measure, truth, got, tol in (("spikes", 250, r.values["n_spikes"], 0), ("bursts", 50, r.values["n_bursts"], 0),
                                     ("spikes per burst", 5, r.values["spikes_per_burst"], 0.0),
                                     ("intraburst frequency (Hz)", 50.0, r.values["intraburst_hz"], 0.01),
                                     ("burst cycle period (s)", 3.0, r.values["burst_period_s"], 1e-6),
                                     ("share of spikes in bursts", 1.0, r.values["burst_fraction"], 0.0),
                                     ("rhythm (Hz)", 1 / 3.0, r.values["sdf_peak_hz"], 0.02)):
        rows.append(Row("spike train", "50 bursts of 5 at 20 ms, every 3 s", measure, truth, float(got), tol))
    said = {e["statement"] for e in r.tables["evaluation"]}
    rows.append(Row("spike train", "50 bursts of 5 at 20 ms, every 3 s", "burst cycle said very stable", 1.0,
                    float("The burst cycle period is very stable" in said), 0.0))
    rng = np.random.default_rng(21)
    t = simulate.event_times(rng, 600.0, 5.0)
    r = train("poisson", t)
    isi = np.diff(t)
    rows.append(Row("spike train", "Poisson 5 Hz, 600 s", "rate (Hz)", (t.size - 1) / (t[-1] - t[0]),
                    r.values["rate_hz"], 1e-4, "the file holds times to 1 us"))
    rows.append(Row("spike train", "Poisson 5 Hz, 600 s", "ISI CV", float(isi.std(ddof=1) / isi.mean()),
                    r.values["isi_cv"], 1e-4, "truth: the same formula on the simulated intervals"))
    rows.append(Row("spike train", "Poisson 5 Hz, 600 s", "LV (about 1 for a Poisson train)", 1.0,
                    r.values["isi_lv"], 0.1))
    t = simulate.event_times(rng, 600.0, 5.0, process="regular", regular_shape=16.0)
    r = train("regular", t)
    rows.append(Row("spike train", "gamma (shape 16) 5 Hz", "LV (3 / (2 x 16 + 1) for a gamma train)", 3 / 33,
                    r.values["isi_lv"], 0.03, "Shinomoto et al. 2003 (verify)"))
    said = {e["statement"] for e in r.tables["evaluation"]}
    rows.append(Row("spike train", "gamma (shape 16) 5 Hz", "said regular", 1.0,
                    float("Firing regularity: regular" in said), 0.0))
    return rows


def dsi() -> list[Row]:
    from clampwise import simulate
    from clampwise.analysis import pipeline
    from clampwise.analysis.cells import Cell, Member, analyze_cell

    rows = []
    sim = simulate.dsi_evoked(seed=2)
    d = pipeline.analyze(sim.recording).results["dsi"]
    for e, tr in zip(d.tables["episodes"], sim.truth["trials"]):
        rows.append(Row("dsi", f"evoked, {tr['duration_s']:g} s step", "suppression (%)", tr["realized_pct"],
                        e["suppression_pct"], 2.0, "truth: the amplitudes drawn (last 5 before, first 3 after)"))
    rows.append(Row("dsi", "evoked, durations 0.5-5 s", "a threshold found", 1.0, float(np.isfinite(d.values["threshold"])),
                    0.0))
    six = simulate.dsi_evoked(durations=(5.0,), trials=6, seed=2)
    d6 = pipeline.analyze(six.recording).results["dsi"]
    rows.append(Row("dsi", "evoked, six 5 s steps", "recovery tau (s)", six.truth["tau_s"], d6.values["recovery_tau_s"],
                    0.2 * six.truth["tau_s"], "responses every 2 s with 8% scatter"))
    # spontaneous events over a cell's files: baseline file, step file, file after
    recs, truth = simulate.dsi_cell(seed=4)
    st = pipeline.Settings(by_analysis={"events": {"template_decay_ms": 10.0}})
    members = []
    for r in recs:
        res = pipeline.analyze(r, st)
        members.append(Member(r.source.name, r, res.results, res.recording_type))
    c = analyze_cell(Cell.build("sim", members)).results["dsi_events"]
    T = truth["times_s"]
    for row, tr in zip(c.tables["trials"], truth["trials"]):
        a, b = tr["step_start_s"], tr["step_end_s"]
        lo = b + row["settle_s"]
        pre_n = ((T >= a - 10) & (T < a)).sum()
        post_n = ((T >= max(lo, tr["post_start_s"])) & (T < lo + 5)).sum()
        real = 100 * (1 - (post_n / row["post_analyzed_s"]) / (pre_n / row["pre_analyzed_s"]))
        rows.append(Row("dsi", f"events over files, {tr['duration_s']:g} s step", "suppression (%)", real,
                        row["suppression_pct"], 8.0, "truth: the simulated events in the windows; 2-3 events missed in "
                        "a 5 s window of about 30 move it 6-7 points"))
        recorded = (a - 0.1 - (a - 10.0)) + (lo + 5.0 - tr["post_start_s"])     # the step file's 0.1 s before is too short
        rows.append(Row("dsi", f"events over files, {tr['duration_s']:g} s step", "windows recorded (s)", recorded,
                        row["pre_analyzed_s"] + row["post_analyzed_s"], 0.02, "the files as placed on the clock"))
    # Heinbockel's pooled recovery: one duration, many trials, a high rate
    recs, truth = simulate.dsi_cell(durations=(2.0,), trials=24, rate_hz=5.0, s_max=0.7, d50_s=0.1, tau_s=8.0,
                                    seed=6)
    members = []
    for r in recs:
        res = pipeline.analyze(r, st)
        members.append(Member(r.source.name, r, res.results, res.recording_type))
    c = analyze_cell(Cell.build("sim", members)).results["dsi_events"]
    k = c.tables["durations"][0]
    # the truth: the same pooled fit on the simulated events themselves (what the recording holds)
    from clampwise.analysis.tools import fits

    T, trs = truth["times_s"], truth["trials"]
    pre = sum(((T >= t["step_start_s"] - 10) & (T < t["step_start_s"] - 0.1)).sum() for t in trs) / (9.9 * len(trs))
    edges = np.arange(0.6, 60.0, 1.0)
    rate = [sum(((T >= t["step_end_s"] + a) & (T < t["step_end_s"] + a + 1)).sum() for t in trs) / len(trs) for a in edges]
    true_tau = fits.suppression_recovery(edges + 0.5, np.array(rate) / pre)["tau"]
    rows.append(Row("dsi", "events, 24 trials of 2 s at 5 Hz, pooled", "recovery tau (s)", true_tau, k["recovery_tau_s"],
                    1.0, f"truth: the same fit on the simulated events (the process's tau is {truth['tau_s']:g} s)"))
    rows.append(Row("dsi", "events, 24 trials of 2 s at 5 Hz, pooled", "joint fit recovery tau (s)", true_tau,
                    k["joint_recovery_tau_s"], 1.5, "onset and recovery fitted together"))
    return rows


def intrinsic() -> list[Row]:
    from clampwise import simulate
    from clampwise.analysis import pipeline
    from clampwise.analysis.tools import fits

    rows = []
    for rm, tau, noise in ((150.0, 0.025, 0.3), (400.0, 0.040, 1.0), (80.0, 0.012, 0.3)):
        sim = simulate.steps(rm_mohm=rm, tau_s=tau, noise=noise, n_sweeps=11)
        r = pipeline.analyze(sim.recording).results["intrinsic"]
        case = f"steps, {rm:g} MOhm, {tau * 1e3:g} ms, noise {noise:g}"
        rows.append(Row("intrinsic", case, "v_rest (mV)", sim.truth["v_rest"], r.values["v_rest"], 0.2))
        rows.append(Row("intrinsic", case, "r_input (MOhm)", rm, r.values["r_input"], 0.02 * rm,
                        "NeuroExpress's resistance at rest"))
        rows.append(Row("intrinsic", case, "r_input_slope (MOhm)", rm, r.values["r_input_slope"], 0.02 * rm))
        rows.append(Row("intrinsic", case, "r_peak0 (MOhm)", rm, r.values["r_peak0"], 0.04 * rm + 2.0,
                        "a passive step's peak is its plateau; the largest deflection of a noisy plateau reads a few % "
                        "above it (NeuroExpress's raw extreme more: 135 against 128 on its test cell)"))
        rows.append(Row("intrinsic", case, "r_vi_steady0 (MOhm)", rm, r.values["r_vi_steady0"], 0.02 * rm))
        rows.append(Row("intrinsic", case, "holding_current (pA)", 0.0, r.values["holding_current"], 0.5))
        rows.append(Row("intrinsic", case, "tau_m (ms)", tau * 1e3, r.values["tau_m"], 0.05 * tau * 1e3))
        rows.append(Row("intrinsic", case, "capacitance (pF)", tau / rm * 1e6, r.values["capacitance"],
                        0.07 * tau / rm * 1e6))
        rows.append(Row("intrinsic", case, "rheobase (pA)", sim.truth["rheobase_pa"], r.values["rheobase"], 0.0))
        # IPFX's floor for a peak is -30 mV: a membrane held above it, with 1 mV of
        # unfiltered noise at 20 kHz, gives noise that passes as attenuated spikes
        # (IPFX's criteria, kept), so the count is
        # checked on the steps that stay clear of it
        below = [k for k, lv in enumerate(sim.truth["levels_pa"])
                 if sim.truth["v_rest"] + lv * rm * 1e-3 + 5 * noise < -30]
        counts = [r.tables["steps"][k]["spikes"] for k in below]
        rows.append(Row("intrinsic", case, "steps with the true count", len(below),
                        sum(a == sim.truth["spike_counts"][k] for a, k in zip(counts, below)), 0.0,
                        "the steps whose steady state stays 5 noise SDs below -30 mV"))
    # sag: the truth is the ratio on the same family without noise, read where the definition reads it
    for frac in (0.15, 0.35):
        sim = simulate.steps(sag_fraction=frac, noise=0.3, n_sweeps=11, first_pa=-200.0, rm_mohm=200.0)
        clean = simulate.steps(sag_fraction=frac, noise=0.0, n_sweeps=11, first_pa=-200.0, rm_mohm=200.0)
        r = pipeline.analyze(sim.recording).results["intrinsic"]
        k = sim.truth["levels_pa"].index(r.values["sag_step_pa"])
        v = clean.recording.sweep(k)
        fs = sim.recording.sample_rate_hz
        a, b = int(0.2 * fs), int(0.7 * fs)
        vp, vss, vb = float(v[a:b - int(0.1 * fs)].min()), float(v[b - int(0.1 * fs):b].mean()), float(v[:a].mean())
        rows.append(Row("intrinsic", f"steps, Ih sag {frac:g}", "sag_ratio", (vp - vss) / (vp - vb),
                        r.values["sag_ratio"], 0.01, f"on the {r.values['sag_step_pa']:g} pA step, its peak nearest "
                        "-100 mV; truth: the noiseless trace"))
    # a rectifying membrane: R(I) = Rm + slope * I, known exactly
    for slope in (300.0, -200.0):
        sim = simulate.steps(rm_mohm=150.0, r_slope_mohm_per_na=slope, noise=0.2, n_sweeps=21, first_pa=-200.0,
                             delta_pa=20.0, rheobase_pa=1e9)
        r = pipeline.analyze(sim.recording).results["intrinsic"]
        case = f"R(I) = 150 MOhm {slope:+g} MOhm/nA x I"
        rows.append(Row("intrinsic", case, "r_steady0 (MOhm)", 150.0, r.values["r_steady0"], 3.0))
        rows.append(Row("intrinsic", case, "r_steady_slope0 (MOhm/nA)", slope, r.values["r_steady_slope0"],
                        0.1 * abs(slope)))
        said = any(e["statement"].startswith("Inward rectification") for e in r.tables["evaluation"])
        rows.append(Row("intrinsic", case, "inward rectification said", float(slope > 0), float(said), 0.0,
                        "R at -200 pA is 40% below rest when the slope is +300 MOhm/nA"))
    # the f-I curve: rate against current, from the spikes the simulation placed
    sim = simulate.firing_steps(pattern="tonic")
    r = pipeline.analyze(sim.recording).results["intrinsic"]
    lv = np.array(sim.truth["levels_pa"])
    true_rate = np.array([len(x) for x in sim.truth["spike_times_s"]]) / 0.8
    use = np.flatnonzero(lv >= sim.truth["rheobase_pa"])[:6]
    gain = fits.linear(lv[use], true_rate[use])["slope"]
    rows.append(Row("intrinsic", "tonic family", "fi_gain (Hz/pA)", gain, r.values["fi_gain"], 0.01,
                    "truth: the same line through the simulated spikes"))
    rows.append(Row("intrinsic", "tonic family", "spikes (all steps)", float(sum(len(x) for x in sim.truth["spike_times_s"])),
                    float(sum(x["spikes"] for x in r.tables["steps"])), 0.0))
    # firing, NeuroExpress's way: no pattern named; the interval measures
    # on the step they are read on, against the simulated spikes of that step, one family each
    for pat in ("tonic", "adapting", "bursting", "delayed", "stuttering", "single"):
        sim = simulate.firing_steps(pattern=pat)
        r = pipeline.analyze(sim.recording).results["intrinsic"]
        if pat == "single":
            rows.append(Row("intrinsic", f"{pat} family", "most spikes in a step", 1.0,
                            float(max(x["spikes"] for x in r.tables["steps"])), 0.0))
            continue
        k = sim.truth["levels_pa"].index(r.values["isi_step_pa"])
        isi = np.diff(sim.truth["spike_times_s"][k])
        rows.append(Row("intrinsic", f"{pat} family", "ISI ratio (last / first)", float(isi[-1] / isi[0]),
                        r.values["isi_ratio"], 0.02, f"at the {r.values['isi_step_pa']:g} pA step"))
    return rows


def thresholds() -> list[Row]:
    """Short squares and ramps, against the simulated cell's own threshold."""
    from clampwise import simulate
    from clampwise.analysis import pipeline

    rows = []
    sim = simulate.short_squares(threshold_pa=190.0)
    r = pipeline.analyze(sim.recording)
    rows.append(Row("thresholds", "short squares, 100-300 by 20 pA", "routed as short squares", 1.0,
                    float(r.recording_type == "cc_short_square"), 0.0))
    ss = r.results["short_square"].values
    first = min(lv for lv in sim.truth["levels_pa"] if lv >= sim.truth["threshold_pa"])
    rows.append(Row("thresholds", "short squares, fires from 190 pA", "threshold (pA)", float(first),
                    ss["ss_threshold_pa"], 0.5, "the first level at or above 190 pA"))
    rows.append(Row("thresholds", "short squares, fires from 190 pA", "resolution (pA)", 20.0, ss["ss_resolution_pa"], 0.5))
    rows.append(Row("thresholds", "short squares, fires from 190 pA", "pulse width (ms)", 3.0, ss["ss_width_ms"], 0.05))
    sim = simulate.ramp(threshold_pa=(130.0, 128.0, 133.0))
    r = pipeline.analyze(sim.recording)
    rows.append(Row("thresholds", "ramp at 100 pA/s", "routed as a ramp", 1.0, float(r.recording_type == "cc_ramp"), 0.0))
    rp = r.results["ramp"]
    rows.append(Row("thresholds", "ramp at 100 pA/s", "rate (pA/s)", 100.0, rp.values["ramp_rate_pa_s"], 0.5))
    for row, truth in zip(rp.tables["ramps"], sim.truth["threshold_pa"]):
        rows.append(Row("thresholds", f"ramp sweep {row['sweep'] + 1}", "threshold (pA)", float(truth),
                        row["threshold_pa"], 1.0, "the current where the simulated spike begins"))
    return rows


def voltage_steps() -> list[Row]:
    """B8: a voltage-step family with a known leak and a Boltzmann conductance."""
    from clampwise import simulate
    from clampwise.analysis import pipeline

    rows = []
    sim = simulate.vc_family()
    truth = sim.truth
    r = pipeline.analyze(sim.recording, pipeline.Settings(by_analysis={"vc_iv": {"e_rev_mv": truth["e_rev"]}}))
    rows.append(Row("voltage steps", "leak 2 nS + Boltzmann", "routed as voltage steps", 1.0,
                    float(r.recording_type == "vc_steps"), 0.0))
    v = r.results["vc_iv"].values
    case = "leak 2 nS + Boltzmann (-20 mV, 8 mV, 10 nS)"
    rows.append(Row("voltage steps", case, "leak (nS)", truth["g_leak_ns"], v["leak_ns"], 0.05))
    rows.append(Row("voltage steps", case, "leak taken off", 1.0, float(v["leak_removed"] == "yes"), 0.0))
    rows.append(Row("voltage steps", case, "V1/2 (mV)", truth["v_half"], v["v_half_mv"], 1.0))
    rows.append(Row("voltage steps", case, "slope factor (mV)", truth["slope"], v["slope_mv"], 0.5))
    rows.append(Row("voltage steps", case, "maximal conductance (nS)", truth["g_max_ns"], v["g_max_ns"], 0.5))
    flat = simulate.vc_family(g_leak_ns=0.0, noise=4.0)
    v = pipeline.analyze(flat.recording).results["vc_iv"].values
    rows.append(Row("voltage steps", "no leak, noise 4 pA", "a leak that cannot be told from zero is left in", 1.0,
                    float(v["leak_removed"] == "no"), 0.0, f"slope {v['leak_ns']:.3g} nS"))
    return rows


SECTIONS = {"baseline": baseline, "evoked": evoked, "events": events, "events+": events_neuroexpress, "dsi": dsi,
            "intrinsic": intrinsic, "spike train": spike_train, "thresholds": thresholds, "voltage steps": voltage_steps}


def run(names: list[str]) -> list[Row]:
    rows: list[Row] = []
    for name in names or list(SECTIONS):
        rows += SECTIONS[name]()
    return rows


def main(argv: list[str]) -> int:
    rows = run(argv)
    bad = 0
    for r in rows:
        mark = "ok " if r.ok else "BAD"
        bad += not r.ok
        print(f"{mark} {r.section:12s} {r.case:40s} {r.measure:22s} true {r.truth:10.4g}  measured {r.measured:10.4g}"
              f"  error {r.error:+9.4g} (tol {r.tolerance:g}) {r.note}")
    print(f"{len(rows) - bad} of {len(rows)} within tolerance")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
