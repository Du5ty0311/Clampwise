"""Spike trains: spontaneous firing and files of spike times (B24; NeuroExpress's
spike-train module).

**Where the spikes come from**: a file of spike times (``io/spiketimes``) as
written; or a current-clamp recording of spontaneous activity, its action
potentials found by IPFX's detection (``tools/spikes``) over every sweep, each
sweep on the file's clock. Fewer than 3 spikes: nothing to analyze (the
recording's events, if any, are the events analysis's).

**Intervals** (Dayan & Abbott 2001, ch. 1): rate, ISI mean / median / SD, CV;
CV2, the mean of 2|I(n+1) - I(n)| / (I(n+1) + I(n)) (Holt et al. 1996 *verify*);
LV, 3 / (n - 1) times the sum of ((I(n) - I(n+1)) / (I(n) + I(n+1)))^2
(Shinomoto et al. 2003 *verify*), read as regular below 0.5, random 0.5-1,
bursty above 1 (the same paper); the ISI histogram with a Gaussian kernel
estimate, the return map (each interval against the next), the autocorrelogram.

**The spike density function** (Dayan & Abbott 2001, ch. 1): the spikes
convolved with a Gaussian of the kernel width (NeuroExpress's default 1 s, as
its SD), sampled every 50 ms; its Fourier amplitude and the frequency at its
peak (a firing rhythm).

**Bursts**, three rules as NeuroExpress offers them, each said with its numbers:
- *intraburst limit* (its default): runs of intervals no longer than the
  maximum intraburst ISI (0.5 s) with at least the minimum spikes (2): 136
  bursts on NeuroExpress's test train, its own count exactly (requiring a 1 s
  pause before each gives 104, so its interburst minimum is not part of this
  rule);
- *max interval*: a burst starts at an interval no longer than half the
  intraburst limit, continues while intervals stay within the limit, and bursts
  closer than the minimum interburst interval are joined (the MaxInterval
  method, as NeuroExplorer describes it *verify*);
- *ISI ratio*: intervals shorter than the train's mean interval divided by the
  ratio (2.5) are inside bursts (Clampwise's reading of NeuroExpress's name;
  its own rule is not published).
Burst count and rate, cycle period (onset to onset) and its CV, length, spikes
per burst, intraburst frequency, the share of spikes in bursts.

**The evaluation**: the number of spikes and bursts; the firing rate's
stability (the CV of the rate in 10 s windows: very stable below 0.1, fairly
stable below 0.25); the burst cycle's (the CV of its periods, the same
limits); the firing's regularity from LV. NeuroExpress's statements, Clampwise's
rules, printed with each.
"""

from __future__ import annotations

import numpy as np

from ..core.recording import Recording
from .registry import Output, Setting, register
from .result import Result
from .tools import ledger, spikes

NAN = float("nan")
BURST_METHODS = ("intraburst limit", "max interval", "ISI ratio")


def spike_times(recording: Recording) -> tuple[list[np.ndarray], str]:
    """Each unit's spike times on the file's clock (s), and where they came from."""
    if recording.header.get("spike_times_s") is not None:
        return [np.asarray(u, float) for u in recording.header["spike_times_s"]], "the file's spike times"
    fs = recording.sample_rate_hz
    out = []
    for k in range(recording.n_sweeps):
        t0 = float(recording.sweep_start_times_s[k]) if k < len(recording.sweep_start_times_s) else 0.0
        found = spikes.find(np.asarray(recording.sweep(k), float), fs, purpose="the spike train")
        out.extend(t0 + x.peak_index / fs for x in found)
    return [np.sort(np.asarray(out, float))], "action potentials detected in the trace (their peaks)"


def isi_measures(t: np.ndarray) -> dict:
    isi = np.diff(t)
    out = {"n_spikes": int(t.size), "isi_mean_s": NAN, "isi_median_s": NAN, "isi_sd_s": NAN, "isi_cv": NAN,
           "isi_cv2": NAN, "isi_lv": NAN}
    if isi.size >= 1:
        out.update({"isi_mean_s": float(isi.mean()), "isi_median_s": float(np.median(isi))})
    if isi.size >= 2:
        out.update({"isi_sd_s": float(isi.std(ddof=1)),
                    "isi_cv": float(isi.std(ddof=1) / isi.mean()) if isi.mean() > 0 else NAN})
        a, b = isi[:-1], isi[1:]
        ok = (a + b) > 0
        out["isi_cv2"] = float(np.mean(2 * np.abs(b[ok] - a[ok]) / (a[ok] + b[ok]))) if ok.any() else NAN
        out["isi_lv"] = float(3.0 * np.mean(((a[ok] - b[ok]) / (a[ok] + b[ok])) ** 2)) if ok.any() else NAN
    return out


def bursts(t: np.ndarray, method: str, max_intra: float, min_inter: float, min_spikes: int,
           ratio: float) -> list[tuple[int, int]]:
    """(first spike, last spike) index pairs of each burst."""
    isi = np.diff(t)
    if isi.size == 0:
        return []
    if method == "ISI ratio":
        inside = isi < isi.mean() / ratio
    elif method == "max interval":
        inside = isi <= max_intra
    else:
        inside = isi <= max_intra
    runs, j = [], 0
    while j < isi.size:
        if inside[j] and (method != "max interval" or isi[j] <= max_intra / 2):
            k = j
            while k + 1 < isi.size and inside[k + 1]:
                k += 1
            runs.append([j, k + 1])                     # spikes j .. k+1
            j = k + 1
        else:
            j += 1
    if method == "max interval":                        # join bursts closer than the interburst minimum
        joined: list[list[int]] = []
        for r in runs:
            if joined and t[r[0]] - t[joined[-1][1]] < min_inter:
                joined[-1][1] = r[1]
            else:
                joined.append(r)
        runs = joined
    out = []
    for a, b in runs:
        if b - a + 1 < min_spikes:
            continue
        out.append((a, b))
    return out


def sdf(t: np.ndarray, end: float, sigma: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    """(times, rate in Hz): the spikes convolved with a Gaussian of SD ``sigma``."""
    grid = np.arange(0.0, end + step / 2, step)
    rate = np.zeros(grid.size)
    if t.size == 0 or sigma <= 0:
        return grid, rate
    norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
    half = int(np.ceil(5 * sigma / step))
    for x in t:
        c = int(round(x / step))
        lo, hi = max(0, c - half), min(grid.size, c + half + 1)
        rate[lo:hi] += norm * np.exp(-0.5 * ((grid[lo:hi] - x) / sigma) ** 2)
    return grid, rate


def autocorrelogram(t: np.ndarray, max_lag: float, width: float) -> list[dict]:
    if t.size < 2 or width <= 0:
        return []
    edges = np.arange(width, max_lag + width / 2, width)
    edges = np.concatenate([[0.0], edges])
    counts = np.zeros(edges.size - 1)
    j0 = 0
    for i in range(t.size):
        while j0 < t.size and t[j0] <= t[i]:
            j0 += 1
        j = j0
        lags = []
        while j < t.size and t[j] - t[i] <= max_lag:
            lags.append(t[j] - t[i])
            j += 1
        if lags:
            counts += np.histogram(lags, edges)[0]
    dur = t[-1] - t[0]
    rate = t.size / dur if dur > 0 else NAN
    # in spikes per second: counts / (spikes x bin width)
    return [{"lag_s": float((a + b) / 2), "count": int(c), "rate_hz": float(c / (t.size * width)),
             "chance_hz": float(rate)} for a, b, c in zip(edges[:-1], edges[1:], counts)]


def _stability(cv: float, what: str, rule: str) -> tuple[str, str, bool] | None:
    if not np.isfinite(cv):
        return None
    if cv < 0.1:
        return (f"The {what} is very stable", f"CV {cv:.2f} {rule} (very stable below 0.1)", False)
    if cv < 0.25:
        return (f"The {what} is fairly stable", f"CV {cv:.2f} {rule} (fairly stable below 0.25)", False)
    return (f"The {what} varies", f"CV {cv:.2f} {rule} (stable below 0.25)", True)


SETTINGS = (
    Setting("sdf_kernel_s", 1.0, "Spike density: Gaussian kernel SD (s)", "", "s", "",
            minimum=0.001),
    Setting("sdf_step_ms", 50.0, "Spike density: sampled every (ms)", "", "ms", "",
            advanced=True, minimum=0.1),
    Setting("burst_method", "intraburst limit", "Bursts: rule", "", "", "",
            choices=BURST_METHODS),
    Setting("max_intraburst_isi_s", 0.5, "Bursts: intervals inside a burst at most (s)", "", "s",
            ""),
    Setting("min_interburst_isi_s", 1.0, "Bursts (max interval): join bursts closer than (s)", "", "s",
            ""),
    Setting("min_burst_spikes", 2, "Bursts: at least this many spikes", "", "", "",
            minimum=2),
    Setting("burst_isi_ratio", 2.5, "Bursts (ISI ratio): inside a burst below the mean interval divided by", "", "",
            "", advanced=True, minimum=1.0),
    Setting("isi_hist_max_s", 1.0, "ISI histogram: up to (s)", "", "s", "", advanced=True),
    Setting("isi_hist_bin_ms", 10.0, "ISI histogram: bins (ms)", "", "ms", "", advanced=True),
    Setting("acg_max_s", 1.0, "Autocorrelogram: lags up to (s)", "", "s", advanced=True),
    Setting("acg_bin_ms", 10.0, "Autocorrelogram: bins (ms)", "", "ms", advanced=True),
    Setting("rhythm_min_hz", 0.05, "Rhythm: the spike density's peak above (Hz)", "Slower is the rate's drift.",
            "Hz", "Clampwise", advanced=True, minimum=0.0),
    Setting("rate_window_s", 10.0, "Rate stability: windows of (s)", "", "s", "Clampwise", advanced=True),
)

OUTPUTS = (
    Output("n_spikes", "Spikes in the train.", "count", label="Spikes", headline=True),
    Output("rate_hz", "Spikes / the time from the first to the last.", "Hz", "Dayan & Abbott 2001",
           label="Mean rate", headline=True),
    Output("isi_mean_s", "Mean interspike interval.", "s", label="ISI (mean)"),
    Output("isi_sd_s", "Standard deviation of the interspike intervals.", "s", label="ISI (SD)"),
    Output("isi_median_s", "Median interspike interval.", "s", label="ISI (median)", headline=True),
    Output("isi_cv", "SD / mean of the intervals.", "ratio", "Dayan & Abbott 2001", label="ISI CV", headline=True),
    Output("isi_cv2", "Mean of 2|I(n+1) - I(n)| / (I(n+1) + I(n)): variability between neighbours.", "ratio",
           "Holt et al. 1996", label="ISI CV2"),
    Output("isi_lv", "Local variation: regular below 0.5, random 0.5-1, bursty above 1.", "ratio",
           "Shinomoto et al. 2003", label="ISI LV", headline=True),
    Output("n_bursts", "Bursts by the rule set.", "count", "", label="Bursts", headline=True),
    Output("burst_rate_hz", "Bursts per second.", "Hz", label="Burst rate"),
    Output("burst_period_s", "Burst cycle period: mean onset to onset.", "s", "",
           label="Burst cycle period"),
    Output("burst_period_cv", "CV of the cycle periods.", "ratio", label="Burst period CV"),
    Output("burst_length_s", "Mean burst length, first to last spike.", "s", "", label="Burst length"),
    Output("spikes_per_burst", "Mean spikes per burst.", "count", "", label="Spikes per burst"),
    Output("intraburst_hz", "Mean intraburst frequency, (spikes - 1) / length.", "Hz", "",
           label="Intraburst frequency"),
    Output("burst_fraction", "Share of the spikes inside bursts.", "fraction", "", label="Burstiness"),
    Output("sdf_peak_hz", "The frequency at the peak of the spike density's Fourier amplitude (a firing rhythm).",
           "Hz", "", label="Rhythm (spike density)"),
    Output("rate_cv", "CV of the firing rate over the windows.", "ratio", label="Rate stability (CV)"),
    Output("evaluation", "Statements read from the train, each with its rule and numbers.", "",
           "", label="Evaluation", headline=True),
)


@register("spike_train", "Spike train", version=1, types={"cc_spontaneous", "spike_times"}, settings=SETTINGS,
          outputs=OUTPUTS, requires=(),
          description="Firing rate, interspike-interval statistics, bursts, spike density and its rhythm, for "
                      "spontaneous firing or a file of spike times.",
          source="Dayan & Abbott 2001; Shinomoto et al. 2003; Holt et al. 1996")
def run(recording: Recording, context, s) -> Result | None:
    units, origin = spike_times(recording)
    if not units or units[0].size < 3:
        return None
    out = Result("spike_train", 1)
    out.choose("spikes", origin, f"{units[0].size} spikes" + (f" in unit 1 of {len(units)}" if len(units) > 1 else ""))
    t = units[0]
    v = out.values
    v.update(isi_measures(t))
    span = t[-1] - t[0]
    v["rate_hz"] = (t.size - 1) / span if span > 0 else NAN
    ledger.note("intervals, CV, CV2 and LV; Gaussian spike density; bursts by "
                f"{s['burst_method']}", "the spike train")
    # bursts
    b = bursts(t, s["burst_method"], s["max_intraburst_isi_s"], s["min_interburst_isi_s"], s["min_burst_spikes"],
               s["burst_isi_ratio"])
    v["n_bursts"] = len(b)
    rows = []
    for a, z in b:
        length = t[z] - t[a]
        rows.append({"burst": len(rows) + 1, "onset_s": float(t[a]), "length_s": float(length), "spikes": z - a + 1,
                     "intraburst_hz": (z - a) / length if length > 0 else NAN})
    out.tables["bursts"] = rows
    if rows:
        onsets = np.array([r["onset_s"] for r in rows])
        per = np.diff(onsets)
        v.update({"burst_rate_hz": len(rows) / span if span > 0 else NAN,
                  "burst_period_s": float(per.mean()) if per.size else NAN,
                  "burst_period_cv": float(per.std(ddof=1) / per.mean()) if per.size >= 2 and per.mean() > 0 else NAN,
                  "burst_length_s": float(np.mean([r["length_s"] for r in rows])),
                  "spikes_per_burst": float(np.mean([r["spikes"] for r in rows])),
                  "intraburst_hz": float(np.nanmean([r["intraburst_hz"] for r in rows])),
                  "burst_fraction": sum(r["spikes"] for r in rows) / t.size})
    else:
        v.update({"burst_rate_hz": 0.0, "burst_period_s": NAN, "burst_period_cv": NAN, "burst_length_s": NAN,
                  "spikes_per_burst": NAN, "intraburst_hz": NAN, "burst_fraction": 0.0})
    rule = {"intraburst limit": f"runs of intervals of at most {s['max_intraburst_isi_s']:g} s, "
                                f"{s['min_burst_spikes']} spikes or more",
            "max interval": f"starting at an interval of at most {s['max_intraburst_isi_s'] / 2:g} s, going on while "
                            f"intervals stay within {s['max_intraburst_isi_s']:g} s, bursts closer than "
                            f"{s['min_interburst_isi_s']:g} s joined",
            "ISI ratio": f"intervals shorter than the mean interval / {s['burst_isi_ratio']:g}"}[s["burst_method"]]
    out.choose("bursts", s["burst_method"], rule)
    # the spike density and its rhythm
    end = float(t[-1]) + 3 * s["sdf_kernel_s"]
    step = s["sdf_step_ms"] / 1e3
    grid, rate = sdf(t, end, s["sdf_kernel_s"], step)
    out.tables["spike_density"] = [{"time_s": float(a), "rate_hz": float(r)} for a, r in zip(grid, rate)]
    if rate.size >= 16:
        amp = np.abs(np.fft.rfft(rate - rate.mean())) / rate.size
        freq = np.fft.rfftfreq(rate.size, step)
        out.tables["spike_density_spectrum"] = [{"frequency_hz": float(f), "amplitude_hz": float(a)}
                                                for f, a in zip(freq[1:], amp[1:])]
        band = freq[1:] >= s["rhythm_min_hz"]         # slower than this is the rate's drift, not a rhythm
        v["sdf_peak_hz"] = float(freq[1:][band][int(np.argmax(amp[1:][band]))]) if band.any() else NAN
    else:
        v["sdf_peak_hz"] = NAN
    # intervals: histogram, return map, autocorrelogram
    isi = np.diff(t)
    from .events import histogram

    out.tables["isi_histogram"] = histogram(isi, s["isi_hist_max_s"], s["isi_hist_bin_ms"] / 1e3)
    out.tables["return_map"] = [{"isi_s": float(a), "next_isi_s": float(b), "deviation_s": float(a - isi.mean())}
                                for a, b in zip(isi[:-1], isi[1:])]
    out.tables["autocorrelogram"] = autocorrelogram(t, s["acg_max_s"], s["acg_bin_ms"] / 1e3)
    # rate stability over windows
    w = s["rate_window_s"]
    n_w = int(span // w)
    if n_w >= 3:
        counts = np.array([np.sum((t >= t[0] + j * w) & (t < t[0] + (j + 1) * w)) for j in range(n_w)]) / w
        v["rate_cv"] = float(counts.std(ddof=1) / counts.mean()) if counts.mean() > 0 else NAN
    else:
        v["rate_cv"] = NAN
    # the evaluation
    say = [(f"{t.size} spikes", origin, False), (f"{len(rows)} bursts", f"by {s['burst_method']}: {rule}", False)]
    st = _stability(v["rate_cv"], "mean firing rate", f"of the rate over {n_w} windows of {w:g} s")
    if st:
        say.append(st)
    st = _stability(v.get("burst_period_cv", NAN), "burst cycle period", f"of {max(0, len(rows) - 1)} periods")
    if st:
        say.append(st)
    lv = v["isi_lv"]
    if np.isfinite(lv):
        kind = "regular" if lv < 0.5 else "random (Poisson-like)" if lv <= 1.0 else "bursty"
        say.append((f"Firing regularity: {kind}", f"LV {lv:.2f} (regular below 0.5, random 0.5-1, bursty above 1)", False))
    out.tables["evaluation"] = [{"statement": a, "why": b2, "warning": c} for a, b2, c in say]
    v["evaluation"] = "; ".join(a for a, _, _ in say)
    if len(units) > 1:
        out.tables["units"] = [{"unit": j + 1, **isi_measures(u),
                                "rate_hz": (u.size - 1) / (u[-1] - u[0]) if u.size >= 2 and u[-1] > u[0] else NAN,
                                "n_bursts": len(bursts(u, s["burst_method"], s["max_intraburst_isi_s"],
                                                       s["min_interburst_isi_s"], s["min_burst_spikes"],
                                                       s["burst_isi_ratio"]))}
                               for j, u in enumerate(units)]
    return out
