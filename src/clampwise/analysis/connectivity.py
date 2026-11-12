"""Connected or not: whether the recorded cell answers the stimulus more often than chance, with the
false-positive rate stated, and the connection's reliability, latency, jitter and size (D26, G1; the lab's
cell-by-cell connectivity runs, 2026-11-10).

The experiment: a cell is patched, and the light is aimed at labelled (opsin-positive) neighbours, one region at
a time (the targeted cell alone, another region, the whole field), or an electrode stimulates a pathway. A
response in the patched cell is synaptic, or it is not a response. The method:

- **Response windows against matched sham windows.** Every pulse of every sweep gets a response window after it
  (from the light's blank to ``window_ms``, never past the next pulse) and a baseline just before it. The same
  windows are taken at sham times in the same sweeps: times a window away from every stimulus, command change,
  test pulse and artifact. The measure in each window is the largest departure from its own baseline, in units of
  the sweep's baseline noise, so a sweep at -70 mV and one at -30 mV count alike, and the sham windows carry the
  same spontaneous activity the response windows do.
- **Connected: the pooled responses beat the pooled shams.** The median departure over every response window is
  compared with the median over one sham per window, drawn 2000 times; p is the share of draws at least as large
  as the real one (a shuffled-trigger surrogate). Connected when p is under ``alpha`` (0.05). One sweep per level,
  as in a step family with the light inside it, is fine: the windows are pooled over sweeps and pulses.
- **The false-positive rate is stated.** A window is a success when its departure beats the (1 - alpha) quantile
  of the sham windows, so by construction alpha of empty windows would be called successes; the successes chance
  alone would give are reported beside the success rate.
- **Latency and jitter** from the successes: the lab's onset (the first departure past twice the noise that then
  grows without returning to the baseline until the peak; Ahmed, Headley & Pare 2021), their median and SD.
  After light, an onset sooner than ``min_latency_ms`` is not a synapse's (a ChroME-driven spike alone takes 1-3
  ms, then the synaptic delay): such windows are counted apart, and the evoked analysis's direct-photocurrent
  judgement (the recorded cell answering the light itself) is reported with them.
- **Amplitude with and without failures**: the departure at the fixed peak time of the successes' average, over
  every window (with failures: the efficacy) and over the successes alone (the potency; Geracitano et al. 2007).
- **A spike in a window is a success** (the strongest answer there is) and is kept out of the amplitudes.
- **Which pattern lit the cell** comes from the patterned light's log kept with the recording, else from the
  file's name (Cell_Only, ROI1, Full_Field: the lab's own names), else it is not known. The cell-level analysis
  (``connectivity_cell``) puts the cell's files side by side, pattern by pattern.
"""

from __future__ import annotations

import re

import numpy as np

from ..core.recording import Recording
from .registry import Output, Setting, register
from .result import Flag, Result
from .tools import fits, kinetics, ledger

NAN = float("nan")
DRAWS = 2000
MIN_WINDOWS = 4              # response windows needed before a verdict
PATTERN_NAMES = (
    (re.compile(r"full[ _-]?field", re.IGNORECASE), "Full field"),
    (re.compile(r"cell[ _-]?only", re.IGNORECASE), "Cell only"),
    (re.compile(r"\b(roi|region)[ _-]?(\d+)", re.IGNORECASE), None),
)


def pattern_from_name(name: str) -> str:
    """The pattern a file's name says was lit: Full field, Cell only, ROI1 ... ("" when it says none)."""
    for rx, word in PATTERN_NAMES:
        m = rx.search(name)
        if m:
            return word if word else f"ROI{m.group(2)}"
    return ""


SETTINGS = (
    Setting("window_ms", 50.0, "Response window after each pulse (ms)", "The largest departure from the baseline "
            "just before the pulse is read in it (never past the next pulse); sham windows are as long.", "ms",
            "Clampwise", minimum=2.0),
    Setting("baseline_ms", 5.0, "Baseline before each window (ms)", "Each window's own level: the median of this "
            "much just before it.", "ms", advanced=True, minimum=1.0),
    Setting("blank_ms", 0.3, "Skip after each pulse (ms)", "The stimulus's own transient; a light artifact the "
            "evoked analysis recognised is skipped too.", "ms", advanced=True, minimum=0.0),
    Setting("alpha", 0.05, "False-positive rate", "A window is a success when it beats this share of the sham "
            "windows; the recording is connected when the pooled test's p is under it.", "", "Clampwise",
            minimum=0.001, maximum=0.5),
    Setting("n_sham", 40, "Sham windows per sweep", "", "", advanced=True, minimum=10),
    Setting("onset_noise_sd", 2.0, "Onset: departure past (noise SD)", "The lab's onset: the first departure past "
            "this many times the noise that then grows without returning to the baseline.", "",
            "Ahmed, Headley & Pare 2021", advanced=True, minimum=1.0),
    Setting("min_latency_ms", 1.5, "After light, a synapse answers no sooner than (ms)", "An onset sooner than "
            "this after a light pulse is the cell's own photocurrent or the light's mark, not a synapse.", "ms",
            "Clampwise (2026-09-28)", advanced=True, minimum=0.0),
)

OUTPUTS = (
    Output("connected", "yes when the pooled response windows beat the pooled sham windows (p under alpha) and "
           "the successes outnumber what chance gives; no when they do not; 'direct photocurrent' when the "
           "recorded cell answers the light itself (the test says nothing about a synapse); 'region unknown' for "
           "a mapping protocol with no record of the regions lit; 'too few windows' under 4 windows.", "",
           "shuffled-trigger surrogate", label="Connected", headline=True),
    Output("p", "The share of 2000 draws (one sham window per response window, their median) at least as large "
           "as the real windows' median departure.", "p", "shuffled-trigger surrogate", label="p (pooled windows)",
           headline=True),
    Output("false_positive_rate", "The share of empty windows that would be called successes at the threshold "
           "used (alpha): the threshold is the (1 - alpha) quantile of the sham windows.", "fraction",
           label="False-positive rate"),
    Output("chance_successes", "How many successes chance alone would give among the windows: alpha times their "
           "number.", "count", label="Successes by chance"),
    Output("success_rate", "The share of windows whose departure beats the sham threshold (or that hold a spike).",
           "fraction", label="Success rate", headline=True),
    Output("successes", "How many windows are successes.", "count", label="Successes"),
    Output("success_rate_low", "The lower end of the success rate's 95% interval (Wilson's score interval: sound "
           "for few windows, where the usual plus-or-minus is not).", "fraction",
           "Wilson 1927; Brown, Cai & DasGupta 2001", label="Success rate, 95% interval from"),
    Output("success_rate_high", "The upper end of that interval.", "fraction",
           "Wilson 1927; Brown, Cai & DasGupta 2001", label="Success rate, 95% interval to"),
    Output("smallest_detectable", "The smallest response a window could have held and still counted as a success: "
           "the sham threshold times the sweeps' median baseline noise. A connection whose responses are smaller "
           "than this cannot be told from none in this recording: 'not connected' means 'none this large'.",
           "signal units", "detection limits of a connection: Campagnola et al. 2022",
           label="Smallest detectable response"),
    Output("n_windows", "Response windows: every pulse of every sweep measured.", "count", label="Windows"),
    Output("n_sham_windows", "Sham windows pooled for the null distribution.", "count", label="Sham windows"),
    Output("latency_median", "Median onset of the successes after the stimulus (the lab's onset: past twice the "
           "noise, then growing to the peak); after light, onsets under the minimum latency are left out and "
           "counted in 'early onsets'.", "ms", "Ahmed, Headley & Pare 2021", label="Latency (median)",
           headline=True),
    Output("jitter", "SD of those onsets.", "ms", label="Jitter (SD of onsets)", headline=True),
    Output("n_early", "Successes whose onset came sooner after the light than a synapse can answer.", "count",
           label="Early onsets"),
    Output("n_in_blank", "Successes already under way where the light's blank ends: their onsets are hidden in "
           "the blank.", "count", label="Onsets inside the blank"),
    Output("efficacy", "Amplitude with failures: the departure at the fixed peak time (the peak of the successes' "
           "average), the mean over every window, signed as the response.", "signal units",
           "Geracitano et al. 2007", label="Amplitude with failures (efficacy)", headline=True),
    Output("potency", "Amplitude without failures: the same over the successes alone.", "signal units",
           "Geracitano et al. 2007", label="Amplitude of the successes (potency)"),
    Output("peak_time", "The fixed peak time the amplitudes are read at, after the stimulus.", "ms",
           label="Peak time"),
    Output("spike_probability", "The share of windows in which the cell fired (current clamp).", "fraction",
           label="Spike probability"),
    Output("spike_latency_median", "Median time from the stimulus to the spike's onset over the windows that "
           "fired; after light, under the minimum latency it is the cell's own photocurrent firing it.", "ms",
           label="Spike latency (median)"),
    Output("direct_photocurrent", "The evoked analysis's judgement: yes when the recorded cell itself answers the "
           "light (see the evoked analysis).", "", label="Direct photocurrent"),
    Output("pattern", "Which pattern lit the cell: from the patterned light's log with the recording, else the "
           "condition typed in Files, else from the file's name (Cell_Only, ROI1, Full_Field).", "",
           label="Pattern lit"),
    Output("pattern_source", "Where the pattern's name came from: the light's log, the condition typed in Files, "
           "the file's name, or nothing.", "", label="Pattern from"),
    Output("units", "The units of the amplitudes: the recorded signal's.", "", label="Amplitude units"),
    Output("z", "The window's largest departure from its baseline, over the sweep's baseline noise SD.", "ratio",
           level="sweep"),
    Output("threshold_z", "The sham threshold: the (1 - alpha) quantile of the sham windows' departures.", "ratio",
           label="Sham threshold (z)"),
)

LEVEL_WORDS = {"yes": "connected", "no": "not connected"}


def _windows(st, sweeps: list[int], n_pulses: int) -> np.ndarray:
    return np.vstack([np.asarray(st.times_s[i][:n_pulses], float) for i in sweeps])


@register("connectivity", "Connected or not", version=3, types={"cc_evoked", "vc_evoked"}, settings=SETTINGS,
          outputs=OUTPUTS, requires=("evoked",),
          description="Whether the recorded cell answers the stimulus more often than chance: every response "
                      "window against matched sham windows in the same sweeps, with the false-positive rate "
                      "stated; the successes' latency, jitter and amplitude with and without failures; a direct "
                      "photocurrent told from a synaptic answer; the pattern lit, for comparing a cell's files.",
          source="shuffled-trigger surrogate; Ahmed, Headley & Pare 2021 (onset); Geracitano et al. 2007 "
                 "(efficacy and potency)")
def run(recording: Recording, context, s) -> Result | None:
    from .evoked import spike_onset

    st = context.stimulus
    if not len(st):
        return None
    out = Result("connectivity", 1)
    ev = context.result("evoked")
    light = st.kind == "light"
    fs, t = recording.sample_rate_hz, recording.time_s
    vc = recording.is_voltage_clamp
    counts = [int(x.size) for x in st.times_s]
    n_pulses = int(np.bincount([c for c in counts if c]).argmax())
    sweeps = [i for i, c in enumerate(counts) if c == n_pulses]
    data = np.asarray(recording.read(recording.signal_channel, sweeps), float)
    ons = _windows(st, sweeps, n_pulses)
    blank = s["blank_ms"] / 1e3
    if ev is not None and np.isfinite(ev.values.get("light_artifact_blank_ms", NAN)):
        blank = max(blank, ev.values["light_artifact_blank_ms"] / 1e3)
    pre, gap = s["baseline_ms"] / 1e3, 0.5e-3
    # one window length for every pulse: the set window, or the gap to the next pulse
    isi = float(np.min(np.diff(ons, axis=1))) if n_pulses > 1 else np.inf
    win = min(s["window_ms"] / 1e3, isi - 1e-3)
    if win - blank < 1e-3:
        out.notes.append("the pulses come too close together for a response window")
        return out
    avoid = [(a.start_s, a.end_s) for a in getattr(context, "artifacts", []) if a.kind != "stimulus artifact"]
    n_sm = max(1, int(round(0.2e-3 * fs)))
    n_win = int(round((win - blank) * fs))
    i_blank, i_pre, i_gap = int(round(blank * fs)), int(round(pre * fs)), int(round(gap * fs))
    sustain = max(1, int(round(1e-3 * fs)))

    def window(y_s: np.ndarray, at: float) -> tuple[np.ndarray, float] | None:
        i0 = int(round(at * fs))
        b0, b1 = i0 - i_pre, i0 - i_gap
        r0, r1 = i0 + i_blank, i0 + i_blank + n_win
        if b0 < 0 or r1 > y_s.size or b1 - b0 < 2:
            return None
        base = float(np.median(y_s[b0:b1]))
        return y_s[r0:r1] - base, base

    rows_z = np.full((len(sweeps), n_pulses), NAN)
    spiked = np.zeros((len(sweeps), n_pulses), bool)
    spike_lat = np.full((len(sweeps), n_pulses), NAN)        # from the pulse to the spike's onset, s
    devs: dict[tuple[int, int], np.ndarray] = {}
    noises = np.full(len(sweeps), NAN)
    shams: list[np.ndarray] = []
    # sham windows are matched to the response windows: the same command level (the same step of a family
    # played under the light: a cell firing on a depolarizing step fires in its sham windows too), clear of
    # every stimulus, command change, test pulse and artifact
    cmd = recording.command
    cmd_rows = np.asarray(cmd, float)[sweeps] if cmd is not None else None
    unmatched = 0
    for k in range(len(sweeps)):
        y = data[k]
        y_s = kinetics.smooth(y, n_sm)
        b0, b1 = int(round(ons[k, 0] * fs)) - int(round(0.02 * fs)), int(round(ons[k, 0] * fs)) - i_gap
        seg = y[max(b0, 0):b1]
        x = np.arange(seg.size)
        resid = seg - np.polyval(np.polyfit(x, seg, 1), x) if seg.size > 3 else seg - np.median(seg)
        noise = kinetics.robust_noise(resid) or float(np.std(resid)) or NAN
        noises[k] = noise
        for j in range(n_pulses):
            w = window(y_s, float(ons[k, j]))
            if w is None or not np.isfinite(noise):
                continue
            dev, base = w
            devs[(k, j)] = dev
            rows_z[k, j] = float(np.max(np.abs(dev))) / noise
            if not vc:
                at = spike_onset(t, y, fs, ons[k, j] + blank, ons[k, j] + win, base)
                spiked[k, j] = np.isfinite(at)
                spike_lat[k, j] = at - ons[k, j] if np.isfinite(at) else NAN
        # sham times: a window away from every stimulus, and clear of command changes, test pulses, artifacts
        lo, hi = pre + 1e-3, t[-1] - win - 1e-3
        cand = np.linspace(lo, hi, 400) if hi > lo else np.array([])
        keep = [c for c in cand
                if np.all((c + win < ons[k] - pre) | (c - pre > ons[k] + win))
                and not any(a <= c + win and c - pre <= b for a, b in avoid)]
        if cmd_rows is not None and keep:
            c_row = cmd_rows[k]
            i_on = int(round(ons[k, 0] * fs))
            level = float(np.median(c_row[max(i_on - i_pre, 0):i_on + i_blank + n_win]))
            tol = 0.5 + 0.02 * abs(level)

            def same_level(c: float) -> bool:
                seg = c_row[max(int(round((c - pre) * fs)), 0):int(round((c + win) * fs))]
                return bool(seg.size) and bool(np.all(np.abs(seg - level) <= tol))

            matched = [c for c in keep if same_level(c)]
            if len(matched) >= 10:
                keep = matched
            else:
                unmatched += 1
        if len(keep) >= 5:
            picked = np.array(keep)[np.linspace(0, len(keep) - 1, min(int(s["n_sham"]), len(keep))).astype(int)]
            zs = []
            for c in picked:
                w = window(y_s, float(c))
                if w is not None and np.isfinite(noise):
                    zs.append(float(np.max(np.abs(w[0]))) / noise)
            shams.append(np.asarray(zs))
        else:
            shams.append(np.array([]))
    have = np.isfinite(rows_z)
    null = np.concatenate([z for z in shams if z.size]) if any(z.size for z in shams) else np.array([])
    if cmd_rows is not None:
        if unmatched == 0:
            out.choose("sham windows", "at the same command level as the response windows",
                       "the same step of the family, clear of every stimulus, command change and test pulse")
        else:
            out.notes.append(f"sham windows could not be matched to the command level in {unmatched} of "
                             f"{len(sweeps)} sweep(s) (under 10 places left on the same step): those sweeps' shams "
                             "come from the whole sweep")
    out.values.update({"n_windows": int(have.sum()), "n_sham_windows": int(null.size), "units": recording.signal_units,
                       "false_positive_rate": float(s["alpha"]),
                       "chance_successes": float(s["alpha"]) * int(have.sum())})
    if have.sum() == 0 or null.size < 40:
        out.notes.append("not tested: " + ("no response window could be measured" if have.sum() == 0 else
                                           f"only {null.size} sham windows (40 needed): the sweeps are too short, "
                                           "or too full of stimuli and command changes, for sham windows"))
        out.values["connected"] = "not tested"
        return out
    threshold = float(np.quantile(null, 1 - s["alpha"]))
    success = (rows_z > threshold) | spiked
    success &= have
    # the pooled test: the real windows' median against the median of one sham per window, drawn many times
    real = float(np.median(rows_z[have]))
    rng = np.random.default_rng(0)

    def draw_medians(pool: list[np.ndarray], n: int) -> np.ndarray:
        """``n`` medians, each over one sham drawn from every window's own sweep."""
        picks = np.empty((n, len(pool)))
        for i, z in enumerate(pool):
            src = z if z.size else null
            picks[:, i] = src[rng.integers(src.size, size=n)]
        return np.median(picks, axis=1)

    pool = [shams[k] for k in range(len(sweeps)) for j in range(n_pulses) if have[k, j]]
    draws = draw_medians(pool, DRAWS)
    p = float((1 + np.sum(draws >= real)) / (1 + DRAWS))
    # each pulse on its own, the same way
    pulse_rows = []
    for j in range(n_pulses):
        hv = have[:, j]
        if not hv.any():
            continue
        r_j = float(np.median(rows_z[hv, j]))
        d_j = draw_medians([shams[k] for k in range(len(sweeps)) if hv[k]], DRAWS // 4)
        low, high = fits.wilson(int(success[:, j].sum()), int(hv.sum()))
        pulse_rows.append({"pulse": j + 1, "n_windows": int(hv.sum()), "successes": int(success[:, j].sum()),
                           "success_rate": float(success[:, j].sum() / hv.sum()),
                           "success_rate_low": low, "success_rate_high": high,
                           "p": float((1 + np.sum(d_j >= r_j)) / (1 + d_j.size)),
                           "spikes": int(spiked[:, j].sum())})
    # the successes' direction and fixed peak time, from their average (aligned on the stimulus)
    succ_keys = [(k, j) for (k, j) in devs if success[k, j] and not spiked[k, j]]
    all_keys = [(k, j) for (k, j) in devs]
    stack = np.vstack([devs[key] for key in (succ_keys or all_keys)])
    avg = stack.mean(axis=0)
    i_pk = int(np.argmax(np.abs(avg)))
    direction = float(np.sign(avg[i_pk])) or 1.0
    half = max(1, int(round(0.5e-3 * fs)))
    amp = {key: float(direction * np.mean(devs[key][max(i_pk - half, 0):i_pk + half + 1])) for key in all_keys}
    t_pk = (i_blank + i_pk) / fs
    # the lab's onset in each success
    lat: dict[tuple[int, int], float] = {}
    early_keys: set[tuple[int, int]] = set()
    in_blank = 0
    for key in succ_keys:
        k, j = key
        dev = devs[key]
        d = dev * float(np.sign(dev[int(np.argmax(np.abs(dev)))]) or 1.0)
        peak = int(np.argmax(d))
        on = kinetics.find_onset(d, 0, peak + 1, s["onset_noise_sd"] * noises[k], sustain)
        if on is None:
            continue
        if blank > s["blank_ms"] / 1e3 and on <= 1:
            in_blank += 1                     # already under way where the light's blank ends: the onset is hidden
            continue
        latency = blank + on / fs
        if light and latency < s["min_latency_ms"] / 1e3:
            early_keys.add(key)
            continue
        lat[key] = latency
    early = len(early_keys)
    lats = np.array(list(lat.values()))
    n_succ = int(success.sum())
    rate_low, rate_high = fits.wilson(n_succ, int(have.sum()))
    measured = noises[np.isfinite(noises) & have.any(axis=1)]
    out.values.update({
        "p": p, "threshold_z": threshold, "successes": n_succ, "success_rate": float(n_succ / have.sum()),
        "success_rate_low": rate_low, "success_rate_high": rate_high,
        "smallest_detectable": float(threshold * np.median(measured)) if measured.size else NAN,
        "latency_median": float(np.median(lats)) * 1e3 if lats.size else NAN,
        "jitter": float(np.std(lats, ddof=1)) * 1e3 if lats.size > 1 else NAN, "n_early": early,
        "efficacy": float(np.mean([amp[key] for key in all_keys if not spiked[key]])) if any(not spiked[key] for key in all_keys) else NAN,
        "potency": float(np.mean([amp[key] for key in succ_keys])) if succ_keys else NAN,
        "peak_time": t_pk * 1e3, "spike_probability": float(spiked[have].mean()) if not vc else NAN,
    })
    direct = ev is not None and ev.values.get("direct_photocurrent") == "yes"
    fired = spike_lat[np.isfinite(spike_lat)]
    if light and fired.size >= max(2, have.sum() // 4) and float(np.median(fired)) < s["min_latency_ms"] / 1e3:
        direct = True                           # the light makes the cell fire sooner than a synapse could
        out.notes.append(f"the cell fires {float(np.median(fired)) * 1e3:.2f} ms after the light (median of "
                         f"{fired.size} windows), sooner than a synapse can answer: its own photocurrent reaches "
                         "threshold")
    out.values["spike_latency_median"] = float(np.median(fired)) * 1e3 if fired.size else NAN
    region_unknown = ev is not None and any(f.code == "region unknown" for f in ev.flags)
    out.values["direct_photocurrent"] = "yes" if direct else "no"
    if region_unknown:
        verdict = "region unknown"
    elif direct:
        verdict = "direct photocurrent"
    elif have.sum() < MIN_WINDOWS:
        verdict = "too few windows"
    else:
        # connected: the pooled test passes and the successes outnumber what chance gives (a median of three
        # windows can pass the pooled test while none of them beats the threshold)
        verdict = "yes" if p < s["alpha"] and n_succ > s["alpha"] * have.sum() else "no"
    out.values["connected"] = verdict
    # which pattern lit the cell
    pattern, source = "", "nothing"
    typed = str(getattr(getattr(recording, "identity", None), "condition", "") or "").strip()
    if ev is not None and ev.values.get("pattern"):
        pattern, source = str(ev.values["pattern"]), "the light's log"
    elif typed:
        pattern, source = typed, "the condition typed in Files"       # Daniel types conditions (2026-11-10)
    elif pattern_from_name(recording.source.stem):
        pattern, source = pattern_from_name(recording.source.stem), "the file's name"
    out.values.update({"pattern": pattern, "pattern_source": source})
    for j, row in enumerate(pulse_rows):
        mine = [v for (k, jj), v in lat.items() if jj == row["pulse"] - 1]
        row["latency_median_ms"] = float(np.median(mine)) * 1e3 if mine else NAN
        row["jitter_ms"] = float(np.std(mine, ddof=1)) * 1e3 if len(mine) > 1 else NAN
        row["efficacy"] = float(np.mean([amp[key] for key in all_keys if key[1] == row["pulse"] - 1 and not spiked[key]])) \
            if any(key[1] == row["pulse"] - 1 and not spiked[key] for key in all_keys) else NAN
        row["potency"] = float(np.mean([amp[key] for key in succ_keys if key[1] == row["pulse"] - 1])) \
            if any(key[1] == row["pulse"] - 1 for key in succ_keys) else NAN
    out.tables["pulses"] = pulse_rows
    out.values["n_in_blank"] = in_blank
    if in_blank:
        out.notes.append(f"{in_blank} success(es) already under way where the light's blank ends ({blank * 1e3:.1f} "
                         "ms): their onsets lie inside the blank and are not known")
    out.tables["windows"] = [{"sweep": sweeps[k], "pulse": j + 1, "z": rows_z[k, j],
                              "over_threshold": float(rows_z[k, j] / threshold) if threshold > 0 else NAN,
                              "success": bool(success[k, j]),
                              "spiked": bool(spiked[k, j]), "spike_latency_ms": spike_lat[k, j] * 1e3,
                              "latency_ms": lat.get((k, j), NAN) * 1e3,
                              "amplitude": amp.get((k, j), NAN), "early": (k, j) in early_keys}
                             for k in range(len(sweeps)) for j in range(n_pulses) if have[k, j]]
    # the sham windows themselves: what the response windows are judged against, to draw beside them
    out.tables["shams"] = [{"sweep": sweeps[k], "z": float(z)} for k in range(len(sweeps)) for z in shams[k]]
    what = {"yes": f"connected: the {int(have.sum())} response windows beat the sham windows (p = {p:.3f})",
            "no": f"not connected: the {int(have.sum())} response windows do no better than sham windows in the same "
                  f"sweeps (p = {p:.2f})",
            "direct photocurrent": "the recorded cell answers the light itself (a direct photocurrent, see the "
                                   "evoked analysis): this recording cannot say whether a synapse connects to it",
            "region unknown": "which region lit each sweep is unknown: the windows are pooled over regions, so "
                              f"the verdict (p = {p:.3f}) is about the run as a whole, not any one region",
            "too few windows": f"{int(have.sum())} response window(s): {MIN_WINDOWS} are needed before a verdict "
                               f"(p = {p:.3f} on so few means little)"}[verdict]
    out.choose("verdict", verdict, what)
    out.choose("success threshold", f"{threshold:.2f} x the noise", f"the {1 - s['alpha']:.0%} quantile of "
               f"{null.size} sham windows: {s['alpha']:.0%} of empty windows pass it, so chance alone gives about "
               f"{s['alpha'] * have.sum():.1f} of the {int(have.sum())} windows")
    if pattern:
        out.choose("pattern", pattern, f"from {source}")
    ledger.note("every response window against matched sham windows in the same sweeps, the pooled median tested "
                "by drawing one sham per window", "connected or not")
    if verdict == "no" and n_succ > s["alpha"] * have.sum() * 3 and have.sum() >= 4:
        out.flags.append(Flag("unreliable connection", f"{n_succ} of {int(have.sum())} windows are successes "
                              f"(chance: {s['alpha'] * have.sum():.1f}) but the pooled test does not reach p < "
                              f"{s['alpha']:g} (p = {p:.2f}): a weak or unreliable input, or spontaneous activity "
                              "in the sham windows", severity="note"))
    if early and lats.size == 0 and not direct:
        out.flags.append(Flag("onsets too soon", f"every success's onset came under {s['min_latency_ms']:g} ms after "
                              "the light: not a synapse's latency (the cell's own photocurrent, or the light's mark on "
                              "the electrode); no latency is given", source="Clampwise"))
    return out


# ------------------------------------------------------------------ the cell: its files side by side

CELL_OUTPUTS = (
    Output("n_files", "The cell's recordings with a stimulus that were tested.", "count", label="Files tested",
           level="cell"),
    Output("n_connected", "How many of them are connected.", "count", label="Connected files", level="cell",
           headline=True),
    Output("patterns_connected", "The patterns (regions lit) whose recordings are connected.", "",
           label="Patterns connected", level="cell", headline=True),
    Output("patterns_not", "The patterns whose recordings are not connected.", "", label="Patterns not connected",
           level="cell"),
    Output("summary", "Each file's verdict in words, in time order.", "", label="Summary", level="cell"),
    Output("comparison", "Each pair of patterns compared directly: their successes over windows by Fisher's exact "
           "test, the p values corrected over the pairs (Holm). One pattern passing its own test and another "
           "failing its own is not a difference between them: this test is.", "",
           "Fisher's exact test; Nieuwenhuis, Forstmann & Wagenmakers 2011", label="Patterns compared",
           level="cell"),
)


def _p_words(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def holm(p: list[float]) -> list[float]:
    """Holm's step-down correction of ``p`` over its own length, in the order given."""
    order = np.argsort(p)
    out, worst = [NAN] * len(p), 0.0
    for rank, i in enumerate(order):
        worst = max(worst, min(1.0, (len(p) - rank) * p[i]))
        out[i] = worst
    return out


@register("connectivity_cell", "Connected or not, over a cell's files", version=3, types={"cc_evoked", "vc_evoked"},
          outputs=CELL_OUTPUTS, level="cell",
          description="A cell's recordings with a stimulus side by side (the targeted cell alone, each region, "
                      "the full field): connected or not, success rate, latency and jitter, amplitude with and "
                      "without failures, and whether the cell answers the light itself.",
          source="Clampwise")
def run_cell(cell, s) -> Result | None:
    members = [m for m in cell.members if m.result("connectivity") is not None]
    if not members:
        return None
    out = Result("connectivity_cell", 2)

    def when(m):
        rec = m.recording
        st = getattr(rec, "start_time", None) if rec is not None else None
        return st.timestamp() if st is not None else 0.0

    rows = []
    for m in sorted(members, key=when):
        c = m.result("connectivity")
        v = c.values
        rows.append({"file": m.name, "pattern": v.get("pattern", ""), "connected": v.get("connected", ""),
                     "p": v.get("p", NAN), "success_rate": v.get("success_rate", NAN),
                     "success_rate_low": v.get("success_rate_low", NAN),
                     "success_rate_high": v.get("success_rate_high", NAN),
                     "successes": v.get("successes", NAN), "n_windows": v.get("n_windows", NAN),
                     "smallest_detectable": v.get("smallest_detectable", NAN),
                     "latency_median_ms": v.get("latency_median", NAN), "jitter_ms": v.get("jitter", NAN),
                     "efficacy": v.get("efficacy", NAN), "potency": v.get("potency", NAN),
                     "direct_photocurrent": v.get("direct_photocurrent", ""), "units": v.get("units", "")})
    out.tables["files"] = rows
    # every response window of every file, named by its pattern: one table to draw the patterns side by side from
    # (successes over windows, latencies, amplitudes); each response as a multiple of its own file's sham threshold,
    # so files recorded at different noise share one axis
    every = []
    for m in sorted(members, key=when):
        c = m.result("connectivity")
        pattern = c.values.get("pattern", "") or m.name
        for w in c.tables.get("windows", []):
            every.append({"file": m.name, "pattern": pattern, "sweep": w["sweep"], "pulse": w["pulse"],
                          "outcome": "success" if w["success"] else "failure", "z": w["z"],
                          "over_threshold": w.get("over_threshold", NAN), "latency_ms": w["latency_ms"],
                          "amplitude": w["amplitude"], "spiked": w["spiked"]})
    out.tables["windows"] = every
    yes = [r for r in rows if r["connected"] == "yes"]
    no = [r for r in rows if r["connected"] == "no"]
    out.values.update({"n_files": len(rows), "n_connected": len(yes),
                       "patterns_connected": ", ".join(r["pattern"] or r["file"] for r in yes),
                       "patterns_not": ", ".join(r["pattern"] or r["file"] for r in no),
                       "summary": "; ".join(f"{r['pattern'] or r['file']}: {r['connected']}"
                                            + (f" ({_p_words(r['p'])}, {r['success_rate']:.0%} successes"
                                               + (f", {r['latency_median_ms']:.1f} ms)" if np.isfinite(r['latency_median_ms'])
                                                  else ")") if np.isfinite(r["p"]) else "")
                                            for r in rows)})
    # the patterns against each other: successes over windows, Fisher's exact test, Holm over the pairs
    from scipy import stats as sps

    tested = [r for r in rows if r["connected"] in ("yes", "no") and np.isfinite(r["successes"])
              and np.isfinite(r["n_windows"]) and r["n_windows"] > 0]
    pairs = [(a, b) for i, a in enumerate(tested) for b in tested[i + 1:]]
    raw = [float(sps.fisher_exact([[int(a["successes"]), int(a["n_windows"] - a["successes"])],
                                   [int(b["successes"]), int(b["n_windows"] - b["successes"])]])[1]) for a, b in pairs]
    corrected = holm(raw) if raw else []
    out.tables["comparisons"] = [
        {"pattern_a": a["pattern"] or a["file"], "pattern_b": b["pattern"] or b["file"],
         "successes_a": int(a["successes"]), "windows_a": int(a["n_windows"]),
         "successes_b": int(b["successes"]), "windows_b": int(b["n_windows"]), "p": p_raw, "p_holm": p_adj,
         "different": bool(p_adj < 0.05)} for (a, b), p_raw, p_adj in zip(pairs, raw, corrected)]
    out.values["comparison"] = "; ".join(
        f"{r['pattern_a']} against {r['pattern_b']}: {r['successes_a']} of {r['windows_a']} against "
        f"{r['successes_b']} of {r['windows_b']} windows, p = {r['p_holm']:.3g}"
        + (" (different)" if r["different"] else " (no difference shown)") for r in out.tables["comparisons"])
    if out.tables["comparisons"]:
        out.choose("patterns compared", "Fisher's exact test on successes over windows, Holm over the pairs",
                   "a pattern passing its own test and another failing is not a difference between them")
    direct = [r for r in rows if r["direct_photocurrent"] == "yes"]
    if direct:
        out.flags.append(Flag("direct photocurrent", f"{len(direct)} of {len(rows)} file(s) show the recorded cell "
                              "answering the light itself (" + ", ".join(r["pattern"] or r["file"] for r in direct)
                              + "): this cell expresses the opsin or its membrane is lit; those files cannot test "
                              "a connection", severity="note", level="cell"))
    out.choose("files", f"{len(rows)} recording(s) with a stimulus, in time order", "every file of the cell the "
               "connectivity analysis ran on")
    return out
