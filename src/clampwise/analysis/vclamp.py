"""Voltage-step currents (B8): the I-V of a voltage-step family or staircase,
the leak, the zero-current potential, slope resistance, and the conductance's
Boltzmann fit.

- **Steps**: each level the command holds (the epoch table's, else the
  command's own runs of at least 20 ms) that is not the sweep's holding level;
  in a staircase (one sweep stepping through levels) every level after the
  first, each against the one before it.
- **Currents**: the level before the step (the 10 ms before it), the steady
  state (the mean of the step's last ``steady_ms``), the peak (the largest
  deviation from the level before, after the capacitive transient's
  ``transient_ms``).
- **Leak**: a straight line through the steady currents of the hyperpolarizing
  steps (at least ``leak_below_mv`` below the holding level), where
  voltage-gated channels are closed (the Axon Guide). It is taken off only
  when its slope can be told from zero (its 95% interval excludes 0);
  otherwise it is flagged and left in (the recorded rule, 2026-09-30; whether
  to stop removing it at all is an open question).
- **Zero-current potential**: where the steady I-V crosses 0 pA (straight
  lines between neighbouring levels). **Slope resistance**: the voltage
  between neighbouring levels over the steady current's change.
- **Conductance**: with a reversal potential typed (``e_rev_mv``), the
  leak-free peak current over the driving force, fitted with a Boltzmann
  (Hille 2001): V1/2, slope and the maximal conductance, with intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .registry import Output, Setting, register
from .result import Drawable, Flag, Result
from .tools import fits

NAN = float("nan")


@dataclass
class Step:
    sweep: int
    a: int
    b: int
    level: float
    before: float               # the level the step came from


def level_runs(recording, sweep: int, min_s: float = 0.02) -> list[tuple[int, int, float]]:
    """(start, end sample, level) of each stretch the command holds in a sweep."""
    fs = recording.sample_rate_hz
    if recording.epochs and sweep < len(recording.epochs) and recording.epochs[sweep]:
        runs: list[list] = []
        for e in recording.epochs[sweep]:
            a, b = int(round(e.start_s * fs)), int(round(e.end_s * fs))
            if b <= a or e.kind not in ("step", "hold"):
                continue
            if runs and abs(runs[-1][2] - e.level) < 0.25 and runs[-1][1] == a:
                runs[-1][1] = b
            else:
                runs.append([a, b, float(e.level)])
        return [(a, b, lv) for a, b, lv in runs if (b - a) / fs >= min_s]
    if recording.command is None:
        return []
    c = np.round(np.asarray(recording.command[sweep], float) * 2) / 2          # to 0.5 mV
    edges = np.flatnonzero(np.diff(c)) + 1
    bounds = np.r_[0, edges, c.size]
    return [(int(a), int(b), float(c[a])) for a, b in zip(bounds[:-1], bounds[1:]) if (b - a) / fs >= min_s]


def steps_of(recording) -> list[Step]:
    out = []
    for k in range(recording.n_sweeps):
        runs = level_runs(recording, k)
        if not runs:
            continue
        hold = runs[0][2]
        staircase = recording.n_sweeps == 1 and len({lv for _, _, lv in runs}) >= 4
        for i, (a, b, lv) in enumerate(runs[1:], start=1):
            if staircase or abs(lv - hold) >= 0.5:
                out.append(Step(k, a, b, lv, runs[i - 1][2]))
    return out


SETTINGS = (
    Setting("steady_ms", 20.0, "Steady state: the step's last (ms)", "The steady current is the mean over this "
            "much of the step's end.", "ms", "Clampwise", minimum=1.0),
    Setting("transient_ms", 2.0, "Capacitive transient: skip (ms)", "The peak is looked for after this much of "
            "the step, past the capacitive transient.", "ms", "Clampwise", minimum=0.0),
    Setting("leak_below_mv", 5.0, "Leak: steps at least this far below the holding (mV)", "The leak is a straight "
            "line through the steady currents of these steps, where voltage-gated channels are closed.", "mV",
            "the Axon Guide", minimum=0.0),
    Setting("e_rev_mv", NAN, "Reversal potential for the conductance (mV)", "Type the current's reversal "
            "potential (e.g. -90 mV for potassium) to get its conductance and Boltzmann fit; blank: no conductance.",
            "mV", "Hille 2001"),
)

OUTPUTS = (
    Output("n_steps", "Voltage steps measured.", "count", label="Steps"),
    Output("holding_current", "The current at the holding level before the steps (their median).", "pA",
           label="Holding current", headline=True),
    Output("leak_ns", "The leak's slope: a straight line through the hyperpolarizing steps' steady currents.", "nS",
           "the Axon Guide", label="Leak conductance", headline=True),
    Output("leak_r_mohm", "1 / the leak's slope: the input resistance where channels are closed.", "MOhm",
           label="Leak resistance"),
    Output("leak_removed", "Whether the leak was taken off: only when its slope can be told from zero (95% "
           "interval).", "", "", label="Leak taken off"),
    Output("zero_current_mv", "Where the steady I-V crosses 0 pA (straight lines between neighbouring levels).", "mV",
           label="Zero-current potential", headline=True),
    Output("i_steady_max", "The largest steady current (leak taken off when it was), signed.", "pA",
           label="Largest steady current"),
    Output("i_peak_max", "The largest peak current (leak taken off when it was), signed.", "pA",
           label="Largest peak current", headline=True),
    Output("g_max_ns", "The Boltzmann fit's maximal conductance (with a typed reversal potential).", "nS",
           "Hille 2001", label="Maximal conductance"),
    Output("v_half_mv", "The Boltzmann fit's half-activation voltage.", "mV", "Hille 2001", label="V1/2",
           headline=True),
    Output("slope_mv", "The Boltzmann fit's slope factor.", "mV", "Hille 2001", label="Slope factor"),
    Output("v_mv", "The step's command level.", "mV", level="step"),
    Output("from_mv", "The level it stepped from.", "mV", level="step"),
    Output("i_before_pa", "The current in the 10 ms before the step.", "pA", level="step"),
    Output("i_steady_pa", "The mean current over the step's last steady_ms.", "pA", level="step"),
    Output("i_peak_pa", "The current where it deviates most from the level before, after the transient.", "pA",
           level="step"),
    Output("i_leak_pa", "The leak line's current at this level (0 when the leak was not taken off).", "pA",
           level="step"),
    Output("i_steady_sub_pa", "The steady current with the leak taken off.", "pA", level="step"),
    Output("i_peak_sub_pa", "The peak current with the leak taken off.", "pA", level="step"),
    Output("g_ns", "The leak-free peak current over the driving force (V - the typed reversal).", "nS",
           level="step"),
    Output("r_slope_mohm", "The voltage from the level below over the steady current's change: the slope "
           "resistance between them.", "MOhm", level="step"),
)


@register("vc_iv", "Voltage-step currents (I-V)", version=1, types={"vc_steps"}, settings=SETTINGS, outputs=OUTPUTS,
          description="The current-voltage relation of a voltage-step family or staircase: steady and peak currents, "
                      "the leak (taken off only when it can be told from zero), the zero-current potential, slope "
                      "resistance, and with a typed reversal potential the conductance's Boltzmann fit.",
          source="the Axon Guide (leak); Hille 2001 (Boltzmann); Clampwise's windows")
def run(recording, context, s) -> Result:
    out = Result("vc_iv", 1)
    fs = recording.sample_rate_hz
    steps = steps_of(recording)
    if not steps:
        out.notes.append("no voltage step found in the command")
        return out
    sig = {k: np.asarray(recording.sweep(k), float) for k in {st.sweep for st in steps}}
    pre, steady_n, skip = int(0.010 * fs), max(1, int(s["steady_ms"] / 1e3 * fs)), int(s["transient_ms"] / 1e3 * fs)
    rows = []
    for st in steps:
        y = sig[st.sweep]
        before = float(np.mean(y[max(0, st.a - pre):st.a])) if st.a > 0 else NAN
        steady = float(np.mean(y[max(st.a, st.b - steady_n):st.b]))
        seg = y[min(st.a + skip, st.b - 1):st.b]
        peak = float(seg[int(np.argmax(np.abs(seg - before)))]) if seg.size and np.isfinite(before) else steady
        rows.append({"sweep": st.sweep, "v_mv": st.level, "from_mv": st.before, "i_before_pa": before,
                     "i_steady_pa": steady, "i_peak_pa": peak})
    v = np.array([r["v_mv"] for r in rows])
    i_ss = np.array([r["i_steady_pa"] for r in rows])
    hold = float(np.median([st.before for st in steps if recording.n_sweeps > 1] or [steps[0].before]))
    out.values["holding_current"] = float(np.nanmedian([r["i_before_pa"] for r in rows if r["from_mv"] == hold])) \
        if any(r["from_mv"] == hold for r in rows) else NAN
    # the leak
    use = v <= hold - s["leak_below_mv"]
    leak = fits.linear(v[use], i_ss[use], purpose="leak: steady current against voltage, hyperpolarizing steps") \
        if np.unique(v[use]).size >= 2 else None
    removed = False
    if leak is not None and leak.ok:
        g = leak["slope"]                                       # pA/mV = nS
        lo, hi = leak.ci.get("slope", (NAN, NAN))
        out.values.update({"leak_ns": g, "leak_r_mohm": 1e3 / g if g else NAN})
        removed = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
        if not removed:
            out.flags.append(Flag("leak not removed", f"the leak's slope ({g:.3g} nS, 95% interval {lo:.3g} to "
                                  f"{hi:.3g}) cannot be told from zero: it is left in", "95% interval excludes 0",
                                  "", severity="note"))
        out.choose("leak", f"{int(use.sum())} steps from {v[use].min():g} to {v[use].max():g} mV",
                   "a straight line through their steady currents" + ("; taken off" if removed else "; left in"))
    else:
        out.values.update({"leak_ns": NAN, "leak_r_mohm": NAN})
        out.notes.append(f"fewer than two hyperpolarizing levels {s['leak_below_mv']:g} mV or more below the "
                         f"holding ({hold:g} mV): no leak line")
    out.values["leak_removed"] = "yes" if removed else "no"
    for r in rows:
        il = float(leak.curve(r["v_mv"])) if removed else 0.0
        r.update({"i_leak_pa": il, "i_steady_sub_pa": r["i_steady_pa"] - il, "i_peak_sub_pa": r["i_peak_pa"] - il})
    # the zero-current potential and slope resistance, on the levels in order
    order = np.argsort(v)
    vs, iss = v[order], i_ss[order]
    zero = NAN
    for k in range(vs.size - 1):
        if np.sign(iss[k]) != np.sign(iss[k + 1]) and iss[k + 1] != iss[k]:
            zero = float(vs[k] - iss[k] * (vs[k + 1] - vs[k]) / (iss[k + 1] - iss[k]))
            break
    out.values["zero_current_mv"] = zero
    slope_r = {}
    for k in range(1, vs.size):
        di = iss[k] - iss[k - 1]
        if vs[k] != vs[k - 1] and di != 0:
            slope_r[order[k]] = (vs[k] - vs[k - 1]) / di * 1e3            # mV/pA = GOhm -> MOhm
    for k, r in enumerate(rows):
        r["r_slope_mohm"] = slope_r.get(k, NAN)
    sub_ss = np.array([r["i_steady_sub_pa"] for r in rows])
    sub_pk = np.array([r["i_peak_sub_pa"] for r in rows])
    out.values["i_steady_max"] = float(sub_ss[np.nanargmax(np.abs(sub_ss))])
    out.values["i_peak_max"] = float(sub_pk[np.nanargmax(np.abs(sub_pk))])
    # the conductance, with a reversal potential typed
    e = s["e_rev_mv"]
    for r in rows:
        r["g_ns"] = r["i_peak_sub_pa"] / (r["v_mv"] - e) if np.isfinite(e) and abs(r["v_mv"] - e) > 5 else NAN
    out.values.update({"g_max_ns": NAN, "v_half_mv": NAN, "slope_mv": NAN})
    if np.isfinite(e):
        gv = np.array([r["g_ns"] for r in rows])
        ok = np.isfinite(gv)
        fit = fits.boltzmann(v[ok], gv[ok], purpose="conductance against voltage")
        if fit.ok:
            out.values.update({"g_max_ns": fit["top"], "v_half_mv": fit["v_half"], "slope_mv": fit["slope"]})
            xs = np.linspace(v[ok].min(), v[ok].max(), 100)
            out.drawables.append(Drawable("curve", "g_ns: the Boltzmann fit", x=xs, y=fit.curve(xs),
                                          domain="current"))
            out.choose("conductance", f"reversal {e:g} mV (typed)", fit.describe())
        else:
            out.notes.append("the conductance's Boltzmann fit did not converge: " + "; ".join(fit.flags))
    else:
        out.notes.append("no reversal potential typed: no conductance (Analyze, Voltage-step currents)")
    out.values["n_steps"] = len(rows)
    out.tables["steps"] = rows
    for r, st in zip(rows, steps):
        out.drawables.append(Drawable("marker", "steady", x=np.array([(st.b - steady_n / 2) / fs]),
                                      y=np.array([r["i_steady_pa"]]), sweep=st.sweep, domain="time"))
    return out


# ------------------------------------------------------------------ the tonic current

def tonic_level(y: np.ndarray, down: bool) -> tuple[float, float]:
    """The holding level of a stretch with events on it: a Gaussian fitted to the
    all-points histogram's event-free side (above its mode when the events go
    down), its mean and SD (Glykys & Mody 2007)."""
    y = y[np.isfinite(y)]
    if y.size < 100:
        return NAN, NAN
    lo, hi = np.percentile(y, [0.5, 99.5])
    counts, edges = np.histogram(y, bins=100, range=(lo, hi))
    mids = (edges[:-1] + edges[1:]) / 2
    m = int(np.argmax(counts))
    side = slice(m, None) if down else slice(0, m + 1)
    x, c = mids[side], counts[side].astype(float)
    if x.size < 4:
        return float(mids[m]), NAN
    # the free side mirrored onto the other, so the fit sees a whole Gaussian
    xx = np.r_[2 * mids[m] - x[::-1], x] if down else np.r_[x, 2 * mids[m] - x[::-1]]
    cc = np.r_[c[::-1], c] if down else np.r_[c, c[::-1]]
    sd0 = float(np.std(y[(y >= mids[m]) if down else (y <= mids[m])])) or float(edges[1] - edges[0])
    fit = fits.gaussian(xx, cc, (float(mids[m]), sd0, float(counts[m])), purpose="the tonic level: the all-points "
                        "histogram's event-free side")
    return (fit["mu"], abs(fit["sd"])) if fit.ok else (float(mids[m]), NAN)


@register("tonic", "Tonic current", version=1, types={"vc_spontaneous"}, requires=("events",),
          settings=(Setting("bin_s", 10.0, "Level read every (s)", "The holding level is read on each stretch this "
                            "long.", "s", minimum=1.0),
                    Setting("drug_window_s", 60.0, "Drug: compare (s) before it and at its end", "The tonic current "
                            "is the shift between this much before the drug and the last this much of it.", "s",
                            minimum=5.0)),
          outputs=(Output("tonic_level", "The holding level: a Gaussian fitted to the all-points histogram's "
                          "event-free side, the median over the stretches.", "pA", "Glykys & Mody 2007",
                          label="Holding level (event-free)", headline=True),
                   Output("tonic_noise_sd", "That Gaussian's SD, the median: the noise without the events.", "pA",
                          label="Noise without events"),
                   Output("tonic_drift", "The last stretch's level minus the first's.", "pA", label="Level drift"),
                   Output("tonic_shift", "The tonic current: the level over the drug's last minute minus the "
                          "minute before it (drug_window_s), signed; the first drug typed.", "pA",
                          "Glykys & Mody 2007", label="Tonic current (shift with the drug)", headline=True),
                   Output("tonic_drug", "The drug the shift is read for.", "", label="Drug"),
                   Output("level_pa", "The stretch's event-free level.", "pA", level="stretch"),
                   Output("sd_pa", "Its Gaussian's SD.", "pA", level="stretch")),
          description="The event-free holding current over time (from the event-free side of the all-points "
                      "histogram), and the tonic current as the shift produced by a drug.",
          source="Glykys & Mody 2007")
def tonic(recording, context, s) -> Result:
    out = Result("tonic", 1)
    fs = recording.sample_rate_hz
    ev = context.results.get("events") if hasattr(context, "results") else None
    down = not (ev is not None and ev.values.get("direction") == "up")
    rows = []
    for k in range(recording.n_sweeps):
        y = np.asarray(recording.sweep(k), float)
        start = float(recording.sweep_start_times_s[k]) if k < recording.sweep_start_times_s.size else 0.0
        step = int(s["bin_s"] * fs)
        for a in range(0, y.size - step // 2, step):
            level, sd = tonic_level(y[a:a + step], down)
            rows.append({"sweep": k, "time_s": start + (a + min(step, y.size - a) / 2) / fs, "level_pa": level,
                         "sd_pa": sd})
            out.drawables.append(Drawable("marker", "event-free level", x=np.array([(a + min(step, y.size - a) / 2) / fs]),
                                          y=np.array([level]), sweep=k, domain="time"))
    out.tables["stretches"] = rows
    levels = np.array([r["level_pa"] for r in rows])
    ok = np.isfinite(levels)
    out.values.update({"tonic_level": float(np.median(levels[ok])) if ok.any() else NAN,
                       "tonic_noise_sd": float(np.nanmedian([r["sd_pa"] for r in rows])) if rows else NAN,
                       "tonic_drift": float(levels[ok][-1] - levels[ok][0]) if ok.sum() > 1 else NAN,
                       "tonic_shift": NAN, "tonic_drug": ""})
    drugs = list(getattr(recording.metadata, "drugs", []) or [])
    if not drugs:
        out.notes.append("no drug typed for this recording: type one with its time (Files, experiment details) to "
                         "read the tonic current as the shift it makes")
        return out
    d = drugs[0]
    t = np.array([r["time_s"] for r in rows])
    w = s["drug_window_s"]
    end = d.off_s if np.isfinite(d.off_s) else (t.max() if t.size else NAN)
    before = levels[ok & (t >= d.on_s - w) & (t < d.on_s)]
    after = levels[ok & (t > end - w) & (t <= end) & (t > d.on_s)]
    if before.size and after.size:
        out.values.update({"tonic_shift": float(np.median(after) - np.median(before)), "tonic_drug": d.name})
        out.choose("tonic current", f"{d.name}: the last {w:g} s of it against the {w:g} s before",
                   f"{before.size} and {after.size} stretches of {s['bin_s']:g} s")
    else:
        out.notes.append(f"{d.name}: not enough of the recording before it or during it to read the shift")
    return out
