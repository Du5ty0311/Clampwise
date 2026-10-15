"""Simulated recordings with exact known answers, for the known-answer tests
and the demos. Every simulator returns the ``Recording`` and a ``truth``
dictionary of what it put in, so a test can say how far a measurement is
from the truth, not merely that it ran.

The simulations are built on a passive membrane (Rm, Cm, Rs where it
matters), synaptic currents as differences of exponentials, action potentials
as a stereotyped waveform, and Gaussian noise; each parameter is in the
units the app reports.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.recording import ArraySource, ClampMode, Channel, Epoch, Recording, Role


@dataclass
class Simulated:
    recording: Recording
    truth: dict = field(default_factory=dict)


def _channels(signal_units: str, command_units: str, extra: list[Channel] = ()) -> list[Channel]:
    mode = ClampMode.from_units(signal_units)
    out = [Channel(0, "Signal", signal_units, Role.SIGNAL, True, mode, role_reason="simulated: the recorded cell"),
           Channel(1, "Command", command_units, Role.COMMAND, False, role_reason="simulated: the command")]
    return out + list(extra)


def psc(t: np.ndarray, onset_s: float, amp: float, rise_s: float, decay_s: float) -> np.ndarray:
    """A synaptic current/potential: amp (1 - e^-t/rise) e^-t/decay from onset, scaled so its peak is amp."""
    tt = t - onset_s
    shape = np.where(tt >= 0, (1 - np.exp(-np.maximum(tt, 0) / rise_s)) * np.exp(-np.maximum(tt, 0) / decay_s), 0.0)
    peak = shape.max() if shape.max() > 0 else 1.0
    return amp * shape / peak


def evoked(*, n_sweeps: int = 20, fs: float = 20_000.0, sweep_s: float = 1.0, onsets_s=(0.2, 0.25),
           amplitudes=(-50.0, -40.0), rise_s: float = 0.0008, decay_s: float = 0.010, latency_s: float = 0.003,
           jitter_s: float = 0.0002, noise: float = 2.0, holding: float = -70.0, voltage_clamp: bool = True,
           baseline: float = -20.0, failures: float = 0.0, seed: int = 0, light: bool = True) -> Simulated:
    """Evoked responses to one or more pulses per sweep, with a TTL trigger channel."""
    rng = np.random.default_rng(seed)
    n = int(sweep_s * fs)
    t = np.arange(n) / fs
    signal = np.full((n_sweeps, n), baseline, dtype=float)
    trigger = np.zeros((n_sweeps, n))
    command = np.full((n_sweeps, n), holding)
    per_sweep_amps = []
    for s in range(n_sweeps):
        amps = []
        for onset, amp in zip(onsets_s, amplitudes):
            lo = int(onset * fs)
            trigger[s, lo:lo + int(0.002 * fs)] = 5.0
            a = 0.0 if rng.random() < failures else amp * (1 + 0.1 * rng.normal())
            amps.append(a)
            signal[s] += psc(t, onset + latency_s + jitter_s * rng.normal(), a, rise_s, decay_s)
        per_sweep_amps.append(amps)
        signal[s] += rng.normal(0, noise, n)
    units = "pA" if voltage_clamp else "mV"
    name = "BlueLED" if light else "AM_Stim"
    channels = _channels(units, "mV" if voltage_clamp else "pA",
                         [Channel(2, name, "V", Role.TRIGGER, True, role_reason="simulated TTL")])
    rec = Recording(Path(f"sim_evoked_{seed}.abf"), fs, channels, ArraySource([signal, command, trigger]),
                    sweep_start_times_s=np.arange(n_sweeps) * (sweep_s + 1.0), protocol="sim_evoked",
                    start_time=dt.datetime(2026, 1, 1, 12, 0, 0), holding=holding,
                    epochs=[[Epoch("hold", 0.0, sweep_s, holding)] for _ in range(n_sweeps)])
    truth = {"onsets_s": list(onsets_s), "amplitudes": list(amplitudes), "per_sweep_amplitudes": per_sweep_amps,
             "rise_s": rise_s, "decay_s": decay_s, "latency_s": latency_s, "jitter_s": jitter_s,
             "ppr": amplitudes[1] / amplitudes[0] if len(amplitudes) > 1 else float("nan"), "noise": noise,
             "failures": failures, "kind": "light" if light else "electrical"}
    return Simulated(rec, truth)


def interval_series(*, intervals_s=(0.012, 0.025, 0.05, 0.1), repeats: int = 8, fs: float = 20_000.0,
                    sweep_s: float = 0.6, first_s: float = 0.2, p1: float = -50.0, ppr=None, rise_s: float = 0.0008,
                    decay_s: float = 0.008, latency_s: float = 0.003, noise: float = 2.0, seed: int = 0) -> Simulated:
    """A paired-pulse interval series: pulse 2 at a different interval in each
    sweep (the case where the sweep average cannot be used). ``ppr`` gives the
    true P2/P1 at each interval (default 0.8 everywhere)."""
    rng = np.random.default_rng(seed)
    ppr = list(ppr) if ppr is not None else [0.8] * len(intervals_s)
    n = int(sweep_s * fs)
    t = np.arange(n) / fs
    order = [k for _ in range(repeats) for k in range(len(intervals_s))]
    signal = np.zeros((len(order), n))
    trigger = np.zeros_like(signal)
    command = np.full_like(signal, -70.0)
    for s, k in enumerate(order):
        for onset, amp in ((first_s, p1), (first_s + intervals_s[k], p1 * ppr[k])):
            lo = int(onset * fs)
            trigger[s, lo:lo + int(0.002 * fs)] = 5.0
            signal[s] += psc(t, onset + latency_s, amp, rise_s, decay_s)
        signal[s] += rng.normal(0, noise, n) - 20.0
    channels = _channels("pA", "mV", [Channel(2, "BlueLED", "V", Role.TRIGGER, True, role_reason="simulated TTL")])
    rec = Recording(Path(f"sim_interval_{seed}.abf"), fs, channels, ArraySource([signal, command, trigger]),
                    sweep_start_times_s=np.arange(len(order)) * (sweep_s + 1.0), protocol="sim_interval_series",
                    start_time=dt.datetime(2026, 1, 1, 12, 20, 0), holding=-70.0,
                    epochs=[[Epoch("hold", 0.0, sweep_s, -70.0)] for _ in order])
    return Simulated(rec, {"intervals_s": list(intervals_s), "ppr": ppr, "p1": p1, "decay_s": decay_s,
                           "latency_s": latency_s, "mean_ppr": float(np.mean(ppr))})


def cc_evoked_spikes(*, n_sweeps: int = 20, spiking_sweeps=(3, 7, 11), fs: float = 20_000.0, sweep_s: float = 1.0,
                     onsets_s=(0.2, 0.25), amplitudes=(5.0, 4.0), v_rest: float = -70.0, noise: float = 0.2,
                     seed: int = 0) -> Simulated:
    """Current-clamp EPSPs with an action potential on pulse 1 in some sweeps."""
    sim = evoked(n_sweeps=n_sweeps, fs=fs, sweep_s=sweep_s, onsets_s=onsets_s, amplitudes=amplitudes,
                 rise_s=0.002, decay_s=0.02, noise=noise, holding=0.0, voltage_clamp=False, baseline=v_rest,
                 seed=seed)
    rec = sim.recording
    data = rec.read(0).copy()
    t = rec.time_s
    for s in spiking_sweeps:
        peak = int((onsets_s[0] + 0.008) * fs)
        st = (t[peak - int(0.001 * fs):peak + int(0.004 * fs)] - t[peak]) * 1e3
        data[s, peak - int(0.001 * fs):peak + int(0.004 * fs)] += 95.0 * np.exp(-(st / 0.4) ** 2)
    rec.samples = ArraySource([data, rec.read(1), rec.read(2)])
    sim.truth["spiking_sweeps"] = list(spiking_sweeps)
    return sim


def steps(*, n_sweeps: int = 11, fs: float = 20_000.0, sweep_s: float = 1.0, step_start_s: float = 0.2,
          step_s: float = 0.5, first_pa: float = -100.0, delta_pa: float = 20.0, rm_mohm: float = 150.0,
          tau_s: float = 0.025, v_rest: float = -65.0, sag_fraction: float = 0.0, sag_tau_s: float = 0.1,
          rheobase_pa: float = 60.0, noise: float = 0.3, bridge_mohm: float = 0.0, r_slope_mohm_per_na: float = 0.0,
          seed: int = 0) -> Simulated:
    """A current-step family on a passive membrane (Rm, tau), optional Ih-like
    sag, and stereotyped spikes above rheobase (one per 25 pA above it). With
    ``r_slope_mohm_per_na`` the resistance changes with the current, R(I) = Rm +
    slope * I (positive: lower when hyperpolarized, inward rectification)."""
    rng = np.random.default_rng(seed)
    n = int(sweep_s * fs)
    t = np.arange(n) / fs
    a, b = int(step_start_s * fs), int((step_start_s + step_s) * fs)
    signal = np.full((n_sweeps, n), v_rest, dtype=float)
    command = np.zeros((n_sweeps, n))
    levels = [first_pa + k * delta_pa for k in range(n_sweeps)]
    spike_counts = []
    for s, i_pa in enumerate(levels):
        command[s, a:b] = i_pa
        r_here = rm_mohm + r_slope_mohm_per_na * i_pa / 1e3
        dv = i_pa * r_here * 1e-3                        # pA x MOhm = uV -> mV: pA*MOhm = 1e-12*1e6 V = 1e-6 V
        tt = t[a:b] - t[a]
        charge = dv * (1 - np.exp(-tt / tau_s))
        if sag_fraction and i_pa < 0:
            charge *= 1 - sag_fraction * (1 - np.exp(-np.maximum(tt - 3 * tau_s, 0) / sag_tau_s))
        signal[s, a:b] += charge + i_pa * bridge_mohm * 1e-3      # an unbalanced bridge: an instant I x R
        tail = t[b:] - t[b]
        signal[s, b:] += charge[-1] * np.exp(-tail / tau_s)
        count = 0
        if i_pa >= rheobase_pa:
            count = 1 + int((i_pa - rheobase_pa) // 25)
            isi = step_s / (count + 1)
            for k in range(count):
                peak = int((step_start_s + 0.02 + k * isi) * fs)
                spike_t = (t[peak - int(0.001 * fs):peak + int(0.004 * fs)] - t[peak]) * 1e3
                shape = 100.0 * np.exp(-(spike_t / 0.5) ** 2) - 8.0 * np.exp(-((spike_t - 2.0) / 1.5) ** 2)
                signal[s, peak - int(0.001 * fs):peak + int(0.004 * fs)] += shape
        spike_counts.append(count)
        signal[s] += rng.normal(0, noise, n)
    channels = _channels("mV", "pA")
    rec = Recording(Path(f"sim_steps_{seed}.abf"), fs, channels, ArraySource([signal, command]),
                    sweep_start_times_s=np.arange(n_sweeps) * (sweep_s + 1.0), protocol="sim_steps",
                    start_time=dt.datetime(2026, 1, 1, 12, 5, 0), holding=0.0,
                    epochs=[[Epoch("hold", 0.0, step_start_s, 0.0), Epoch("step", step_start_s, step_start_s + step_s, lv),
                             Epoch("hold", step_start_s + step_s, sweep_s, 0.0)] for lv in levels])
    truth = {"levels_pa": levels, "rm_mohm": rm_mohm, "tau_s": tau_s, "bridge_mohm": bridge_mohm, "v_rest": v_rest,
             "sag_fraction": sag_fraction, "r_slope_mohm_per_na": r_slope_mohm_per_na,
             "rheobase_pa": rheobase_pa, "spike_counts": spike_counts, "step_start_s": step_start_s, "step_s": step_s}
    return Simulated(rec, truth)


def gap_free_events(*, duration_s: float = 60.0, fs: float = 20_000.0, rate_hz: float = 5.0, amp: float = -30.0,
                    amp_cv: float = 0.3, rise_s: float = 0.0005, decay_s: float = 0.010, noise: float = 3.0,
                    holding_pa: float = -150.0, seed: int = 0) -> Simulated:
    """Spontaneous synaptic currents as a Poisson process in a gap-free voltage-clamp file."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    signal = np.full(n, holding_pa, dtype=float)
    times = np.sort(rng.uniform(0, duration_s - 0.1, rng.poisson(rate_hz * duration_s)))
    amps = amp * np.maximum(1 + amp_cv * rng.normal(size=times.size), 0.2)
    kernel_n = int(8 * decay_s * fs)
    kt = np.arange(kernel_n) / fs
    base = (1 - np.exp(-kt / rise_s)) * np.exp(-kt / decay_s)
    base /= base.max()
    for t0, a in zip(times, amps):
        i = int(t0 * fs)
        seg = base[: n - i]
        signal[i:i + seg.size] += a * seg
    signal += rng.normal(0, noise, n)
    command = np.full((1, n), -70.0)
    channels = _channels("pA", "mV")
    rec = Recording(Path(f"sim_gapfree_{seed}.abf"), fs, channels, ArraySource([signal[None, :], command]),
                    sweep_start_times_s=np.zeros(1), protocol="sim_gap_free", mode="gap-free",
                    start_time=dt.datetime(2026, 1, 1, 12, 10, 0), holding=-70.0)
    truth = {"times_s": times, "amplitudes": amps, "rate_hz": rate_hz, "rise_s": rise_s, "decay_s": decay_s,
             "noise": noise, "holding_pa": holding_pa}
    return Simulated(rec, truth)


def membrane_test(*, n_sweeps: int = 10, fs: float = 50_000.0, sweep_s: float = 0.1, step_mv: float = -10.0,
                  rs_mohm: float = 12.0, rm_mohm: float = 200.0, cm_pf: float = 100.0, holding_mv: float = -70.0,
                  holding_pa: float = -40.0, noise: float = 2.0, seed: int = 0) -> Simulated:
    """A test pulse on an Rs-(Rm || Cm) circuit, the exact solution: the
    current jumps to V/Rs and decays with tau = Rs Rm Cm / (Rs + Rm) to V/(Rs + Rm)."""
    rng = np.random.default_rng(seed)
    n = int(sweep_s * fs)
    t = np.arange(n) / fs
    a, b = int(0.02 * fs), int(0.06 * fs)
    rs, rm, cm = rs_mohm * 1e6, rm_mohm * 1e6, cm_pf * 1e-12
    tau = rs * rm * cm / (rs + rm)
    v = step_mv * 1e-3
    i_inf, i_peak = v / (rs + rm), v / rs
    signal = np.full((n_sweeps, n), holding_pa, dtype=float)
    command = np.full((n_sweeps, n), holding_mv)
    for s in range(n_sweeps):
        # on: the current jumps to V/Rs and relaxes to V/(Rs + Rm); off: the mirror image
        tt = t[a:b] - t[a]
        signal[s, a:b] += (i_inf + (i_peak - i_inf) * np.exp(-tt / tau)) * 1e12
        tt2 = t[b:] - t[b]
        signal[s, b:] += -(i_peak - i_inf) * np.exp(-tt2 / tau) * 1e12
        command[s, a:b] += step_mv
        signal[s] += rng.normal(0, noise, n)
    channels = _channels("pA", "mV")
    rec = Recording(Path(f"sim_mtest_{seed}.abf"), fs, channels, ArraySource([signal, command]),
                    sweep_start_times_s=np.arange(n_sweeps) * sweep_s, protocol="sim_membrane_test",
                    start_time=dt.datetime(2026, 1, 1, 12, 15, 0), holding=holding_mv,
                    epochs=[[Epoch("hold", 0.0, 0.02, holding_mv), Epoch("step", 0.02, 0.06, holding_mv + step_mv),
                             Epoch("hold", 0.06, sweep_s, holding_mv)] for _ in range(n_sweeps)])
    truth = {"rs_mohm": rs_mohm, "rm_mohm": rm_mohm, "cm_pf": cm_pf, "tau_s": tau, "holding_pa": holding_pa,
             "step_mv": step_mv}
    return Simulated(rec, truth)


def event_times(rng: np.random.Generator, duration_s: float, rate_hz: float, process: str = "poisson",
                rate_steps=(), burst_size: float = 4.0, burst_interval_s: float = 0.008,
                regular_shape: float = 4.0) -> np.ndarray:
    """Event times over ``duration_s``.

    ``poisson``: a Poisson process at ``rate_hz``, or piecewise at
    ``rate_steps`` [(from_s, rate_hz), ...] (an inhomogeneous Poisson process,
    by thinning). ``bursty``: bursts as a Poisson process, each of 1 +
    Poisson(burst_size - 1) events at exponential intervals of mean
    ``burst_interval_s``, the burst rate set so the mean rate is ``rate_hz``.
    ``regular``: gamma-distributed intervals of shape ``regular_shape``
    (CV 1/sqrt(shape)) at ``rate_hz``."""
    if process == "bursty":
        n_b = rng.poisson(rate_hz / burst_size * duration_s)
        starts = rng.uniform(0, duration_s, n_b)
        out = []
        for s in starts:
            k = 1 + rng.poisson(burst_size - 1)
            out.extend(s + np.concatenate([[0.0], np.cumsum(rng.exponential(burst_interval_s, k - 1))]))
        times = np.sort(np.asarray(out))
    elif process == "regular":
        iei = rng.gamma(regular_shape, 1.0 / (rate_hz * regular_shape), int(rate_hz * duration_s * 1.5) + 10)
        times = np.cumsum(iei) + rng.uniform(0, 1.0 / rate_hz)
    else:
        steps_ = sorted(rate_steps) if rate_steps else [(0.0, rate_hz)]
        top = max(r for _, r in steps_)
        cand = np.sort(rng.uniform(0, duration_s, rng.poisson(top * duration_s)))
        starts = np.array([a for a, _ in steps_])
        rates = np.array([r for _, r in steps_])
        local = rates[np.clip(np.searchsorted(starts, cand, side="right") - 1, 0, len(rates) - 1)]
        times = cand[rng.uniform(0, top, cand.size) < local]
    return times[(times >= 0) & (times < duration_s)]


def spontaneous(*, duration_s: float = 60.0, fs: float = 20_000.0, rate_hz: float = 5.0, amp: float = -30.0,
                amp_cv: float = 0.3, rise_s: float = 0.0005, decay_s: float = 0.010, noise: float = 3.0,
                voltage_clamp: bool = True, baseline: float = -150.0, holding_mv: float = -70.0,
                process: str = "poisson", rate_steps=(), n_sweeps: int = 0, sweep_s: float = 10.0,
                spikes_at=(), step=None, edge_s: float = 0.05, seed: int = 0, name: str = "",
                times_s=None, start_time: dt.datetime | None = None) -> Simulated:
    """Spontaneous synaptic events with exact known times and sizes.

    Voltage clamp (pA, events of the sign of ``amp``) or current clamp (mV).
    Gap-free over ``duration_s``, or ``n_sweeps`` sweeps of ``sweep_s`` (the
    process runs on through the second between sweeps). ``spikes_at`` adds
    action potentials (current clamp) at those times in every sweep. ``step``
    = (start_s, length_s, level_mv) puts a depolarizing step in each sweep
    (voltage clamp): an outward current during it, an inward tail after it,
    and no events during it (the shape of a DSI protocol)."""
    rng = np.random.default_rng(seed)
    episodic = n_sweeps > 0
    total = n_sweeps * (sweep_s + 1.0) if episodic else duration_s
    times = np.sort(np.asarray(times_s, float)) if times_s is not None else event_times(rng, total, rate_hz, process,
                                                                                        rate_steps)
    amps = amp * np.maximum(1 + amp_cv * rng.normal(size=times.size), 0.2)
    kernel_n = int(8 * decay_s * fs)
    kt = np.arange(kernel_n) / fs
    base = (1 - np.exp(-kt / rise_s)) * np.exp(-kt / decay_s)
    base /= base.max()
    spike_t = (np.arange(-int(0.001 * fs), int(0.006 * fs)) / fs) * 1e3
    spike = 95.0 * np.exp(-(spike_t / 0.45) ** 2) - 10.0 * np.exp(-((spike_t - 2.5) / 1.8) ** 2)
    sweeps = n_sweeps if episodic else 1
    n = int((sweep_s if episodic else duration_s) * fs)
    signal = np.full((sweeps, n), baseline, dtype=float)
    command = np.full((sweeps, n), holding_mv if voltage_clamp else 0.0)
    starts = np.arange(sweeps) * (sweep_s + 1.0) if episodic else np.zeros(1)
    kept_t, kept_a, kept_sweep = [], [], []
    for s in range(sweeps):
        lo = starts[s]
        quiet = None
        if step is not None:
            a0, length, level = step
            ia, ib = int(a0 * fs), int((a0 + length) * fs)
            command[s, ia:ib] = level
            signal[s, ia:ib] += 800.0
            signal[s, ib:] += -300.0 * np.exp(-(np.arange(n - ib) / fs) / 0.15)
            quiet = (a0 - 0.001, a0 + length)
        for t0, a in zip(times, amps):
            rel = t0 - lo
            if not (edge_s <= rel < n / fs - edge_s):
                continue
            if quiet is not None and quiet[0] <= rel < quiet[1]:
                continue
            i = int(round(rel * fs))
            seg = base[: n - i]
            signal[s, i:i + seg.size] += a * seg
            kept_t.append(rel)
            kept_a.append(a)
            kept_sweep.append(s)
        if not voltage_clamp:
            for ts in spikes_at:
                i = int(ts * fs)
                j0, j1 = i - int(0.001 * fs), i + int(0.006 * fs)
                if j0 >= 0 and j1 <= n:
                    signal[s, j0:j1] += spike
        signal[s] += rng.normal(0, noise, n)
    units, cmd_units = ("pA", "mV") if voltage_clamp else ("mV", "pA")
    epochs = []
    if episodic:
        hold = holding_mv if voltage_clamp else 0.0
        for _ in range(sweeps):
            if step is not None:
                a0, length, level = step
                epochs.append([Epoch("hold", 0.0, a0, hold), Epoch("step", a0, a0 + length, level),
                               Epoch("hold", a0 + length, sweep_s, hold)])
            else:
                epochs.append([Epoch("hold", 0.0, sweep_s, hold)])
    rec = Recording(Path(name or f"sim_spont_{seed}.abf"), fs, _channels(units, cmd_units),
                    ArraySource([signal, command]), sweep_start_times_s=starts, protocol="sim_spontaneous",
                    mode="episodic" if episodic else "gap-free",
                    start_time=start_time or dt.datetime(2026, 1, 1, 12, 30, 0),
                    holding=holding_mv if voltage_clamp else 0.0, epochs=epochs)
    truth = {"times_s": np.asarray(kept_t), "sweeps": np.asarray(kept_sweep, int), "amplitudes": np.asarray(kept_a),
             "rate_hz": rate_hz, "rise_s": rise_s, "decay_s": decay_s, "noise": noise, "process": process,
             "spikes_at": list(spikes_at), "duration_s": float(sweeps * n / fs)}
    return Simulated(rec, truth)


def event_driven(*, n_events: int = 120, fs: float = 20_000.0, sweep_ms: float = 15.0, pre_ms: float = 3.0,
                 amp: float = -20.0, amp_cv: float = 0.25, rise_s: float = 0.0005, decay_s: float = 0.004,
                 noise: float = 1.5, rate_hz: float = 4.0, times_recorded: bool = True, seed: int = 0) -> Simulated:
    """An event-driven (fixed-length) file: one triggered event per sweep,
    ``pre_ms`` into it; the sweep start times real (as Clampex records them)
    or back to back (as Clampfit saves detected events)."""
    rng = np.random.default_rng(seed)
    n = int(sweep_ms * 1e-3 * fs)
    t = np.arange(n) / fs
    amps = amp * np.maximum(1 + amp_cv * rng.normal(size=n_events), 0.2)
    signal = np.zeros((n_events, n))
    for s, a in enumerate(amps):
        signal[s] = psc(t, pre_ms * 1e-3, a, rise_s, decay_s) + rng.normal(0, noise, n) - 50.0
    if times_recorded:
        starts = np.cumsum(np.maximum(rng.exponential(1.0 / rate_hz, n_events), sweep_ms * 1e-3 * 1.5))
    else:
        starts = np.arange(n_events) * n / fs
    starts = starts - starts[0]
    channels = [Channel(0, "IN 0", "pA", Role.SIGNAL, True, ClampMode.VOLTAGE_CLAMP, role_reason="simulated")]
    rec = Recording(Path(f"sim_eventdriven_{seed}.abf"), fs, channels, ArraySource([signal]),
                    sweep_start_times_s=starts, protocol="sim_events", start_time=dt.datetime(2026, 1, 1, 12, 40, 0),
                    holding=-70.0, header={"event_driven": True, "event_times_recorded": times_recorded,
                                           "operation_mode": "fixed-length"})
    return Simulated(rec, {"amplitudes": amps, "starts_s": starts, "rise_s": rise_s, "decay_s": decay_s,
                           "rate_hz": rate_hz, "pre_ms": pre_ms})


def dsi_evoked(*, durations=(0.5, 1.0, 2.0, 5.0), trials: int = 2, sweep_s: float = 60.0, step_at_s: float = 20.0,
               stim_every_s: float = 2.0, amp: float = -200.0, cv: float = 0.08, s_max: float = 0.8,
               d50_s: float = 1.5, hill: float = 2.0, tau_s: float = 15.0, fs: float = 10_000.0, noise: float = 3.0,
               seed: int = 0) -> Simulated:
    """Evoked DSI: one sweep per trial with a response every ``stim_every_s``,
    a step to 0 mV at ``step_at_s`` for each duration (``trials`` times each,
    in turn), and the response after it suppressed by S(d) = s_max d^h / (d^h +
    d50^h), recovering with ``tau_s``: amplitude x (1 - S exp(-(t - end)/tau))."""
    rng = np.random.default_rng(seed)
    order = [d for _ in range(trials) for d in durations]
    n = int(sweep_s * fs)
    t = np.arange(n) / fs
    signal = np.zeros((len(order), n))
    command = np.full((len(order), n), -70.0)
    trigger = np.zeros((len(order), n))
    epochs, truth_rows = [], []
    for k, d in enumerate(order):
        a, b = step_at_s, step_at_s + d
        command[k, int(a * fs):int(b * fs)] = 0.0
        signal[k, int(a * fs):int(b * fs)] += 800.0
        signal[k, int(b * fs):] += -300.0 * np.exp(-(t[int(b * fs):] - b) / 0.15)
        sup = s_max * d ** hill / (d ** hill + d50_s ** hill)
        post, pre, post_real = [], [], []
        for t0 in np.arange(2.0, sweep_s - 0.2, stim_every_s):
            if a - 0.01 <= t0 <= b + 0.2:
                continue
            factor = 1.0 - sup * np.exp(-(t0 - b) / tau_s) if t0 > b else 1.0
            trigger[k, int(t0 * fs):int((t0 + 0.001) * fs)] = 5.0
            real = amp * factor * (1 + cv * rng.normal())
            signal[k] += psc(t, t0 + 0.002, real, 0.0008, 0.012)
            if t0 > b:
                post.append(factor)
                post_real.append(real)
            elif t0 < a:
                pre.append(real)
        signal[k] += rng.normal(0, noise, n) - 150.0
        epochs.append([Epoch("hold", 0.0, a, -70.0), Epoch("step", a, b, 0.0), Epoch("hold", b, sweep_s, -70.0)])
        truth_rows.append({"duration_s": d, "S": sup, "suppression_pct": 100 * (1 - float(np.mean(post[:3]))),
                           # what the recording holds: the amplitudes drawn, the last 5 before and first 3 after
                           "realized_pct": 100 * (1 - float(np.mean(post_real[:3])) / float(np.mean(pre[-5:])))})
    channels = _channels("pA", "mV", [Channel(2, "AM_Stim", "V", Role.TRIGGER, True, role_reason="simulated TTL")])
    rec = Recording(Path(f"sim_dsi_evoked_{seed}.abf"), fs, channels, ArraySource([signal, command, trigger]),
                    sweep_start_times_s=np.arange(len(order)) * (sweep_s + 30.0), protocol="sim_dsi",
                    start_time=dt.datetime(2026, 1, 1, 13, 0, 0), holding=-70.0, epochs=epochs)
    return Simulated(rec, {"trials": truth_rows, "tau_s": tau_s, "s_max": s_max, "d50_s": d50_s, "hill": hill})


def dsi_cell(*, durations=(0.5, 1.0, 2.0, 5.0), trials: int = 2, rate_hz: float = 10.0, s_max: float = 0.8,
             d50_s: float = 1.5, hill: float = 2.0, tau_s: float = 8.0, baseline_s: float = 30.0,
             post_s: float = 60.0, gap_s: float = 0.3, fs: float = 10_000.0, seed: int = 0) -> tuple[list, dict]:
    """A DSI experiment from spontaneous IPSCs, as it is recorded:
    for each trial a gap-free baseline file, a short file holding the step to
    0 mV, and a gap-free file from ``gap_s`` after it; the event rate after the
    step r(t) = rate (1 - S(d) exp(-(t - end)/tau)), S(d) as in ``dsi_evoked``.
    Returns the recordings (with start times, one cell) and the truth, with the
    events' times on the cell's clock."""
    rng = np.random.default_rng(seed)
    order = [d for _ in range(trials) for d in durations]
    t0 = dt.datetime(2026, 1, 1, 14, 0, 0)
    clock = 0.0
    recs, truth_trials, all_times = [], [], []
    for k, d in enumerate(order):
        sup = s_max * d ** hill / (d ** hill + d50_s ** hill)
        base_times = event_times(rng, baseline_s, rate_hz)
        sim = spontaneous(duration_s=baseline_s, fs=fs, times_s=base_times, edge_s=0.0, seed=seed * 100 + 3 * k,
                          name=f"t{k:02d}_a_baseline.abf", start_time=t0 + dt.timedelta(seconds=clock), decay_s=0.010)
        recs.append(sim.recording)
        all_times += list(clock + sim.truth["times_s"])
        step_start = clock + baseline_s + 0.0
        sweep = 0.1 + d + 0.3
        stepper = steps_file(d, fs=fs, name=f"t{k:02d}_b_step.abf", start=t0 + dt.timedelta(seconds=step_start))
        recs.append(stepper)
        a, b = step_start + 0.1, step_start + 0.1 + d
        post_start = step_start + sweep + gap_s
        cand = np.sort(rng.uniform(0, post_s, rng.poisson(rate_hz * post_s)))
        lam = 1.0 - sup * np.exp(-((post_start + cand) - b) / tau_s)
        kept = cand[rng.uniform(0, 1, cand.size) < lam]
        sim = spontaneous(duration_s=post_s, fs=fs, times_s=kept, edge_s=0.0, seed=seed * 100 + 3 * k + 1,
                          name=f"t{k:02d}_c_after.abf", start_time=t0 + dt.timedelta(seconds=post_start), decay_s=0.010)
        recs.append(sim.recording)
        all_times += list(post_start + sim.truth["times_s"])
        truth_trials.append({"duration_s": d, "S": sup, "step_start_s": a, "step_end_s": b, "post_start_s": post_start})
        clock = post_start + post_s + 20.0
    return recs, {"trials": truth_trials, "times_s": np.sort(np.array(all_times)), "rate_hz": rate_hz, "tau_s": tau_s}


def steps_file(d: float, *, fs: float = 10_000.0, name: str = "step.abf", start: dt.datetime | None = None,
               noise: float = 3.0) -> Recording:
    """A file holding only a step to 0 mV for ``d`` s (0.1 s at -70 mV before, 0.3 s after), as a DSI_Step file."""
    n = int((0.1 + d + 0.3) * fs)
    t = np.arange(n) / fs
    command = np.full((1, n), -70.0)
    command[0, int(0.1 * fs):int((0.1 + d) * fs)] = 0.0
    signal = np.full((1, n), -150.0) + np.random.default_rng(7).normal(0, noise, (1, n))
    signal[0, int(0.1 * fs):int((0.1 + d) * fs)] += 800.0
    signal[0, int((0.1 + d) * fs):] += -300.0 * np.exp(-(t[int((0.1 + d) * fs):] - 0.1 - d) / 0.15)
    return Recording(Path(name), fs, _channels("pA", "mV"), ArraySource([signal, command]),
                     sweep_start_times_s=np.zeros(1), protocol="sim_DSI_Step", start_time=start, holding=-70.0,
                     epochs=[[Epoch("hold", 0.0, 0.1, -70.0), Epoch("step", 0.1, 0.1 + d, 0.0),
                              Epoch("hold", 0.1 + d, 0.1 + d + 0.3, -70.0)]])


def firing_steps(*, pattern: str = "tonic", n_sweeps: int = 12, fs: float = 20_000.0, sweep_s: float = 1.2,
                 step_start_s: float = 0.2, step_s: float = 0.8, first_pa: float = -100.0, delta_pa: float = 20.0,
                 rm_mohm: float = 150.0, tau_s: float = 0.025, v_rest: float = -65.0, rheobase_pa: float = 60.0,
                 noise: float = 0.2, seed: int = 0) -> Simulated:
    """A current-step family whose spikes follow a chosen firing pattern, for
    the firing-pattern rule: "tonic" (regular intervals), "adapting" (each
    interval 1.25 times the last), "bursting" (bursts of 4 at 5 ms), "delayed"
    (tonic, the first spike 300 ms into the step), "stuttering" (clusters of 3 at
    20 ms with 150 ms pauses), "single" (one spike, whatever the current). The
    rate grows with the current above rheobase (1 Hz per 2 pA, from 10 Hz)."""
    rng = np.random.default_rng(seed)
    n = int(sweep_s * fs)
    t = np.arange(n) / fs
    a, b = int(step_start_s * fs), int((step_start_s + step_s) * fs)
    levels = [first_pa + k * delta_pa for k in range(n_sweeps)]
    signal = np.full((n_sweeps, n), v_rest)
    command = np.zeros((n_sweeps, n))
    spike_t = (np.arange(-int(0.001 * fs), int(0.004 * fs)) / fs) * 1e3
    shape = 100.0 * np.exp(-(spike_t / 0.5) ** 2) - 8.0 * np.exp(-((spike_t - 2.0) / 1.5) ** 2)
    times_by_sweep = []
    for k, i_pa in enumerate(levels):
        command[k, a:b] = i_pa
        dv = i_pa * rm_mohm * 1e-3
        tt = t[a:b] - t[a]
        charge = dv * (1 - np.exp(-tt / tau_s))
        signal[k, a:b] += charge
        signal[k, b:] += charge[-1] * np.exp(-(t[b:] - t[b]) / tau_s)
        times = []
        if i_pa >= rheobase_pa:
            rate = 10.0 + (i_pa - rheobase_pa) / 2.0
            isi = 1.0 / rate
            start = 0.3 if pattern == "delayed" else 0.02
            if pattern == "single":
                times = [0.02]
            elif pattern in ("tonic", "delayed"):
                times = list(np.arange(start, step_s - 0.01, isi))
            elif pattern == "adapting":
                x, d = start, isi * 0.6
                while x < step_s - 0.01:
                    times.append(x)
                    x += d
                    d *= 1.25
            elif pattern == "bursting":
                x = start
                while x < step_s - 0.03:
                    times += [x + 0.005 * j for j in range(4)]
                    x += max(isi * 3, 0.06)
            elif pattern == "stuttering":
                x = start
                while x < step_s - 0.05:
                    times += [x + 0.02 * j for j in range(3)]
                    x += 0.04 + 0.15
            times = [x for x in times if x < step_s - 0.006]
        for x in times:
            p = a + int(x * fs)
            j0, j1 = p - int(0.001 * fs), p + int(0.004 * fs)
            signal[k, j0:j1] += shape
        times_by_sweep.append([step_start_s + x for x in times])
        signal[k] += rng.normal(0, noise, n)
    rec = Recording(Path(f"sim_firing_{pattern}_{seed}.abf"), fs, _channels("mV", "pA"), ArraySource([signal, command]),
                    sweep_start_times_s=np.arange(n_sweeps) * (sweep_s + 1.0), protocol="sim_firing",
                    start_time=dt.datetime(2026, 1, 1, 12, 50, 0), holding=0.0,
                    epochs=[[Epoch("hold", 0.0, step_start_s, 0.0), Epoch("step", step_start_s, step_start_s + step_s, lv),
                             Epoch("hold", step_start_s + step_s, sweep_s, 0.0)] for lv in levels])
    return Simulated(rec, {"pattern": pattern, "levels_pa": levels, "spike_times_s": times_by_sweep,
                           "rheobase_pa": rheobase_pa, "rm_mohm": rm_mohm, "tau_s": tau_s})


def _spike_shape(fs: float) -> tuple[np.ndarray, int]:
    """firing_steps' action potential: (the waveform, its peak's sample)."""
    spike_t = (np.arange(-int(0.001 * fs), int(0.004 * fs)) / fs) * 1e3
    return 100.0 * np.exp(-(spike_t / 0.5) ** 2) - 8.0 * np.exp(-((spike_t - 2.0) / 1.5) ** 2), int(0.001 * fs)


def short_squares(*, levels=tuple(range(100, 320, 20)), width_s: float = 0.003, threshold_pa: float = 190.0,
                  fs: float = 50_000.0, sweep_s: float = 0.3, onset_s: float = 0.1, rm_mohm: float = 150.0,
                  tau_s: float = 0.02, v_rest: float = -70.0, noise: float = 0.1, seed: int = 0) -> Simulated:
    """The Allen Institute's short squares: one 3 ms current pulse per sweep, its
    size stepping across sweeps; a pulse at or above ``threshold_pa`` fires one
    action potential 1 ms after it ends."""
    rng = np.random.default_rng(seed)
    n = int(sweep_s * fs)
    t = np.arange(n) / fs
    a, b = int(onset_s * fs), int((onset_s + width_s) * fs)
    shape, peak = _spike_shape(fs)
    signal = np.full((len(levels), n), v_rest)
    command = np.zeros((len(levels), n))
    spikes_at = []
    for k, lv in enumerate(levels):
        command[k, a:b] = lv
        dv = lv * rm_mohm * 1e-3 * (1 - np.exp(-(t[a:b] - t[a]) / tau_s))
        signal[k, a:b] += dv
        signal[k, b:] += dv[-1] * np.exp(-(t[b:] - t[b]) / tau_s)
        if lv >= threshold_pa:
            p = b + int(0.001 * fs)
            signal[k, p - peak:p - peak + shape.size] += shape
            spikes_at.append(t[p])
        else:
            spikes_at.append(None)
        signal[k] += rng.normal(0, noise, n)
    rec = Recording(Path(f"sim_short_squares_{seed}.abf"), fs, _channels("mV", "pA"), ArraySource([signal, command]),
                    sweep_start_times_s=np.arange(len(levels)) * (sweep_s + 1.0), protocol="sim_short_square",
                    start_time=dt.datetime(2026, 1, 1, 13, 0, 0), holding=0.0,
                    epochs=[[Epoch("hold", 0.0, onset_s, 0.0), Epoch("step", onset_s, onset_s + width_s, float(lv)),
                             Epoch("hold", onset_s + width_s, sweep_s, 0.0)] for lv in levels])
    return Simulated(rec, {"levels_pa": list(levels), "threshold_pa": threshold_pa, "spike_times_s": spikes_at,
                           "width_s": width_s})


def ramp(*, n_sweeps: int = 3, rate_pa_s: float = 100.0, threshold_pa: float = (130.0, 128.0, 133.0),
         fs: float = 20_000.0, sweep_s: float = 3.0, onset_s: float = 0.2, rm_mohm: float = 150.0, tau_s: float = 0.02,
         v_rest: float = -70.0, noise: float = 0.1, seed: int = 0) -> Simulated:
    """The Allen Institute's ramp: the current rises at ``rate_pa_s`` from
    ``onset_s`` to the sweep's end; the cell fires when it reaches its threshold
    (one per sweep, ``threshold_pa``), then every 50 ms. The ramp is an epoch of
    kind "ramp", as Clampex records one."""
    rng = np.random.default_rng(seed)
    n = int(sweep_s * fs)
    t = np.arange(n) / fs
    a = int(onset_s * fs)
    shape, peak = _spike_shape(fs)
    thresholds = list(np.broadcast_to(np.atleast_1d(threshold_pa), (n_sweeps,)))
    signal = np.full((n_sweeps, n), v_rest)
    command = np.zeros((n_sweeps, n))
    first = []
    for k in range(n_sweeps):
        command[k, a:] = rate_pa_s * (t[a:] - t[a])
        # the membrane follows the current through its time constant (a ramp's response lags it by tau)
        target = command[k] * rm_mohm * 1e-3
        v = np.empty(n)
        v[0] = 0.0
        for i in range(1, n):
            v[i] = v[i - 1] + (target[i] - v[i - 1]) / (tau_s * fs)
        signal[k] += v
        at = onset_s + thresholds[k] / rate_pa_s
        first.append(at)
        for x in np.arange(at, sweep_s - 0.01, 0.05):
            p = int(x * fs)
            if p - peak + shape.size < n:
                signal[k, p - peak:p - peak + shape.size] += shape
        signal[k] += rng.normal(0, noise, n)
    rec = Recording(Path(f"sim_ramp_{seed}.abf"), fs, _channels("mV", "pA"), ArraySource([signal, command]),
                    sweep_start_times_s=np.arange(n_sweeps) * (sweep_s + 1.0), protocol="sim_ramp",
                    start_time=dt.datetime(2026, 1, 1, 13, 10, 0), holding=0.0,
                    epochs=[[Epoch("hold", 0.0, onset_s, 0.0),
                             Epoch("ramp", onset_s, sweep_s, rate_pa_s * (sweep_s - onset_s))] for _ in range(n_sweeps)])
    return Simulated(rec, {"rate_pa_s": rate_pa_s, "threshold_pa": thresholds, "first_spike_s": first})


def vc_family(*, levels=tuple(range(-110, 50, 10)), holding: float = -70.0, g_leak_ns: float = 2.0,
              e_leak: float = -70.0, g_max_ns: float = 10.0, v_half: float = -20.0, slope: float = 8.0,
              e_rev: float = -90.0, fs: float = 20_000.0, sweep_s: float = 0.4, step_start_s: float = 0.1,
              step_s: float = 0.2, tau_act_s: float = 0.003, noise: float = 2.0, seed: int = 0) -> Simulated:
    """A voltage-step family: a linear leak (``g_leak_ns`` reversing at ``e_leak``)
    and a non-inactivating voltage-gated conductance (Boltzmann: ``g_max_ns``,
    ``v_half``, ``slope``; reversing at ``e_rev``, activating with ``tau_act_s``),
    with a capacitive transient at each edge. Currents in pA (nS x mV)."""
    rng = np.random.default_rng(seed)
    n = int(sweep_s * fs)
    a, b = int(step_start_s * fs), int((step_start_s + step_s) * fs)
    signal = np.zeros((len(levels), n))
    command = np.full((len(levels), n), holding)
    for k, v in enumerate(levels):
        command[k, a:b] = v
        vm = command[k]
        g_inf = g_max_ns / (1 + np.exp((v_half - vm) / slope))
        g = np.empty(n)
        g[0] = g_inf[0]
        for i in range(1, n):
            g[i] = g[i - 1] + (g_inf[i] - g[i - 1]) / (tau_act_s * fs)
        signal[k] = g_leak_ns * (vm - e_leak) + g * (vm - e_rev)
        for edge, dv in ((a, v - holding), (b, holding - v)):
            tail = np.exp(-np.arange(int(0.003 * fs)) / (0.0002 * fs))
            signal[k, edge:edge + tail.size] += 40.0 * dv * tail
        signal[k] += rng.normal(0, noise, n)
    rec = Recording(Path(f"sim_vc_family_{seed}.abf"), fs, _channels("pA", "mV"), ArraySource([signal, command]),
                    sweep_start_times_s=np.arange(len(levels)) * (sweep_s + 1.0), protocol="sim_vc_steps",
                    start_time=dt.datetime(2026, 1, 1, 13, 20, 0), holding=holding,
                    epochs=[[Epoch("hold", 0.0, step_start_s, holding),
                             Epoch("step", step_start_s, step_start_s + step_s, float(v)),
                             Epoch("hold", step_start_s + step_s, sweep_s, holding)] for v in levels])
    return Simulated(rec, {"levels_mv": list(levels), "g_leak_ns": g_leak_ns, "e_leak": e_leak, "g_max_ns": g_max_ns,
                           "v_half": v_half, "slope": slope, "e_rev": e_rev, "holding": holding})
