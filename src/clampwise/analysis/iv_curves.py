"""A current-step family step by step, as NeuroExpress shows it.

NeuroExpress (Szucs; version 25.7.27) reads every property of a step family
against the injected current, one point per step, and gives each at rest by
extrapolating to 0 pA. Clampwise follows NeuroExpress, in
Clampwise's way, every definition stated. NeuroExpress's source is not
published, so each definition below is either confirmed against its exported
numbers on its own test file and 26 lab step families (the agreement was checked
during development) or written from a published method
and said so.

Per step (``steps`` table, one row per sweep):

- baseline: the mean of the 100 ms before the step; steady state: the mean of
  the step's last 50 ms; peak: the extreme of the step's direction over the
  whole step, on a 5 ms running mean (a 1 ms one left a 9-13% bias from noise
  on simulated passive cells);
- resistance at the peak and at the steady state, (V - V_baseline) / I, for
  steps of at least the reject range (10 pA) away from 0 pA;
- sag (mV): steady state minus peak, signed (positive on hyperpolarizing
  steps); relative sag: |sag| / |peak deflection|;
- rebound / afterdepolarization: the furthest the voltage goes past the
  baseline in the 200 ms after the step (1 ms running mean, spikes blanked),
  signed; relative to the steady-state deflection; sag and rebound at 0 pA
  and their slopes from a line through the non-firing steps from the midpoint
  up (NeuroExpress's test cell: sag slope -6.7 against its -5.8 mV/nA, relative
  sag 9.6 against 8.2%, relative rebound 11.3 against 11.7%);
- tau at the step's onset (10-95% of the way to the peak, as the headline tau)
  and at its offset (10-95% of the way back); capacitance tau / R;
- the first spike of each step: latency, threshold at 5 mV/ms (NeuroExpress's
  spike slope threshold), amplitude and half-width from it, time to peak, fall
  time, steepest rise and fall and where in the spike they come, the trough
  after it, and the kink at the onset (onset rapidness: the slope of dV/dt
  against V; Naundorf, Wolf & Volgushev 2006 *verify*).

At rest (0 pA): a straight line through each against current over the steps
between the midpoint (-70 pA) and the reject range (-10 pA), read at 0 pA
(confirmed on NeuroExpress's test cell: 117.1 against its 117.4 MOhm at the
steady state; at the peak 128 against its 134, NeuroExpress taking the raw
extreme, which noise pushes up); the V-I relation (peak and steady state
against current) a straight line from the midpoint to minus it, its slope the
resistance from the V-I fit (103 / 98 against its 104 / 99 MOhm).

On noisy 11-step families no definition tried reproduces
NeuroExpress's resistances closer than a median 5-7%, nor its sag slope
(33%): its handling of noise is not shown anywhere. Each Clampwise measure
says what it is.

Firing: rheobase, the I-O gain (spikes per nA from rheobase until the count
reaches the I-O limit, 8), the cumulative count up to the counting limit
(100 pA), latency at 1.3 times rheobase, the mean latency up to the counting
limit, where the lowest voltage between two spikes falls in their interval;
per step the ISI CV, the local variation LV (Shinomoto et al. 2003 *verify*),
CV2 (Holt et al. 1996 *verify*) and the burst fraction.

The evaluation: plain statements with the rule and the numbers beside each
(NeuroExpress's list). Where NeuroExpress's rule could be read off its own
numbers it is used: "the resistance measures differ by more than 30%" is the
peak resistance at rest above 1.3 times the peak V-I resistance (true of its
statements on all 25 lab files with a row). Rectification needs 3 steps
beyond the midpoint (NeuroExpress says it on its 70-step test cell, on none of
the lab's 11-step families). The rest of its cut-offs are not recoverable (its
"strong sag" follows no measure that can be rebuilt: it flags a 1.4% sag and
misses a 26 mV one), so the rules are Clampwise's, printed with each
statement, and settings. Added: a cell resting above -40 mV.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .result import Drawable, Flag
from .tools import fits, ledger, spikes

NAN = float("nan")


@dataclass
class Curves:
    rows: list[dict] = field(default_factory=list)
    values: dict = field(default_factory=dict)
    statements: list[tuple[str, str, bool]] = field(default_factory=list)   # (statement, rule, a warning)
    drawables: list[Drawable] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)
    moved: list[str] = field(default_factory=list)             # the points read where they were put on Check


MOVABLE = ("baseline", "peak", "steady", "rebound")


def parse_moved(text: str) -> dict[tuple[int, str], float]:
    """``"3:peak:0.12345;..."``: (sweep, which point) -> where it was moved to (s)."""
    out = {}
    for part in (text or "").split(";"):
        bits = part.strip().split(":")
        if len(bits) == 3 and bits[1] in MOVABLE:
            out[(int(bits[0]), bits[1])] = float(bits[2])
    return out


def format_moved(moved: dict[tuple[int, str], float]) -> str:
    return ";".join(f"{k}:{what}:{t:.5f}" for (k, what), t in sorted(moved.items()))


def _smooth(y: np.ndarray, n: int) -> np.ndarray:
    """A running mean of n samples, the ends held at their own level (zero padding
    would pull a trace at -87 mV towards 0 at its edges)."""
    if n <= 1 or y.size <= n:
        return y
    padded = np.pad(y, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(padded, np.ones(n) / n, mode="valid")


def line_at_zero(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> tuple[float, float, int]:
    """(value at 0, slope per unit of x, points) of a straight line through the
    finite points with lo <= x <= hi; NaN with fewer than two distinct x."""
    m = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if np.unique(np.round(x[m], 6)).size < 2:
        return NAN, NAN, int(m.sum())
    slope, icpt = np.polyfit(x[m], y[m], 1)
    return float(icpt), float(slope), int(m.sum())


def first_spike_shape(v: np.ndarray, d: np.ndarray, fs: float, x: spikes.Spike) -> dict:
    """The first spike's shape from the 5 mV/ms threshold (NeuroExpress's measures)."""
    t5, p = x.slope_threshold_index, x.peak_index
    if t5 < 0 or p <= t5:
        return {}
    v5, peak = float(v[t5]), float(v[p])
    amp = peak - v5
    out = {"threshold_5_mv": v5, "amplitude_5_mv": amp, "time_to_peak_ms": (p - t5) / fs * 1e3,
           "upstroke": x.upstroke, "downstroke": x.downstroke}
    half = v5 + amp / 2.0
    up = np.flatnonzero(v[t5:p + 1] >= half)
    stop = min(v.size, p + int(0.05 * fs))
    dn = np.flatnonzero(v[p:stop] <= half)
    out["half_width_5_ms"] = (p + dn[0] - (t5 + up[0])) / fs * 1e3 if up.size and dn.size else NAN
    back = np.flatnonzero(v[p:stop] <= v5)
    out["fall_time_ms"] = back[0] / fs * 1e3 if back.size else NAN
    if amp > 0 and x.upstroke_index >= 0:
        out["upstroke_phase"] = (float(v[x.upstroke_index]) - v5) / amp
    if amp > 0 and x.downstroke_index >= 0:
        out["downstroke_phase"] = (float(v[x.downstroke_index]) - v5) / amp
    if x.fast_trough_index >= 0:
        out["trough_mv"] = float(v[x.fast_trough_index])
        out["ahp_5_mv"] = v5 - float(v[x.fast_trough_index])
    # the kink: d2V/dt2 between the threshold and the steepest rise
    u = x.upstroke_index if x.upstroke_index > t5 else p
    d2 = np.gradient(d, 1e3 / fs)                      # mV/ms per ms
    seg = d2[t5:u + 1]
    if seg.size >= 2:
        k = t5 + int(np.argmax(seg))
        out["kink_max"] = float(d2[k])
        out["time_to_kink_ms"] = (k - t5) / fs * 1e3
        out["kink_mv"] = float(v[k])
        out["kink_phase_slope"] = float(d2[k] / d[k]) if d[k] > 0 else NAN
        out["initial_phase_slope"] = float(d2[t5] / d[t5]) if d[t5] > 0 else NAN
    return out


def without_spikes(v: np.ndarray, fs: float, a: int, b: int) -> np.ndarray:
    """The trace with every spike between a and b replaced by a straight line from
    its threshold to 5 ms after its peak (a rebound spike is not a rebound)."""
    y = v.copy()
    for x in spikes.find(v, fs, window=(a, b)):
        s0 = x.slope_threshold_index if x.slope_threshold_index >= 0 else x.threshold_index
        s1 = min(v.size - 1, x.peak_index + int(0.005 * fs))
        if s1 > s0:
            y[s0:s1 + 1] = np.linspace(y[s0], y[s1], s1 - s0 + 1)
    return y


def isi_stats(isi: np.ndarray) -> dict:
    """CV; LV (Shinomoto et al. 2003); CV2 (Holt et al. 1996): defined on 3+ intervals."""
    out = {"isi_cv": NAN, "isi_lv": NAN, "isi_cv2": NAN}
    if isi.size >= 2:
        out["isi_cv"] = float(np.std(isi, ddof=1) / isi.mean()) if isi.mean() > 0 else NAN
        a, b = isi[:-1], isi[1:]
        r = (a - b) / (a + b)
        out["isi_lv"] = float(3.0 * np.mean(r ** 2))
        out["isi_cv2"] = float(np.mean(2.0 * np.abs(b - a) / (a + b)))
    return out


def analyze(recording, step, rows: list[dict], all_spikes: list[list[spikes.Spike]], s: dict) -> Curves:
    out = Curves()
    fs = recording.sample_rate_hz
    i0, i1 = int(round(step.start_s * fs)), int(round(step.end_s * fs))
    nb = max(1, int(round(min(s["nex_baseline_s"], step.start_s - step.baseline[0]) * fs)))
    ne = max(1, int(round(min(s["nex_steady_s"], step.duration_s / 4) * fs)))
    smooth_n = max(1, int(round(0.001 * fs)))
    post_n = int(round(s["rebound_window_ms"] / 1e3 * fs))
    peak_n = max(1, int(round(0.005 * fs)))      # the peak on a 5 ms mean, as IPFX averages its peak
    reject, mid = s["reject_range_pa"], s["resistance_midpoint_pa"]
    cur = np.array([r["current_pa"] for r in rows])
    counts = np.array([r["spikes"] for r in rows])
    ledger.note(f"per step: baseline the {nb / fs * 1e3:.0f} ms before the step, steady state "
                f"its last {ne / fs * 1e3:.0f} ms, the peak over the whole step; resistance (V - Vbaseline) / I for steps "
                f"{reject:g} pA or more from 0 pA; at 0 pA a line through the steps from {mid:g} to -{reject:g} pA",
                "intrinsic properties, step by step")
    n = len(rows)
    vb, ve, vm, va = (np.full(n, NAN) for _ in range(4))
    tau_on, tau_off = np.full(n, NAN), np.full(n, NAN)
    shapes: list[dict] = [{} for _ in range(n)]
    lat = np.full(n, NAN)
    t = np.arange(recording.n_samples) / fs
    marks: list[dict] = [{} for _ in range(n)]           # where each measure was read, to draw on the trace
    moved = parse_moved(s.get("moved", ""))               # points moved by hand on Check, kept in the project
    for k in range(n):
        v = np.asarray(recording.sweep(k), float)
        vb[k] = float(v[i0 - nb:i0].mean())
        ve[k] = float(v[i1 - ne:i1].mean())
        sign = 1.0 if cur[k] > 0 else -1.0
        seg = v[i0:i1]
        sm = _smooth(seg, peak_n)
        vm[k] = float(sm.max() if sign > 0 else sm.min())
        clean = without_spikes(v, fs, i1, min(v.size - 1, i1 + post_n))
        after = _smooth(clean[i1:i1 + post_n], smooth_n)[smooth_n:-smooth_n or None]
        if after.size:
            j = int(after.argmin() if sign > 0 else after.argmax())
            va[k] = float(after[j]) - vb[k]
            marks[k]["rebound"] = ((i1 + smooth_n + j) / fs, va[k] + vb[k])
        marks[k].update({"baseline": ((i0 - nb / 2) / fs, vb[k]), "steady": ((i1 - ne / 2) / fs, ve[k]),
                         "peak": ((i0 + int(sm.argmax() if sign > 0 else sm.argmin())) / fs, vm[k])})
        # a point moved by hand is read where it was put, over the same span as its own reading
        sw = rows[k]["sweep"]
        rebound_abs = va[k] + vb[k]
        for what, span in (("baseline", nb), ("steady", ne), ("peak", peak_n), ("rebound", smooth_n)):
            at = moved.get((sw, what))
            if at is None:
                continue
            j = int(np.clip(round(at * fs), 0, v.size - 1))
            level = float(v[max(0, j - span // 2):j + span // 2 + 1].mean())
            if what == "baseline":
                vb[k] = level
            elif what == "steady":
                ve[k] = level
            elif what == "peak":
                vm[k] = level
            else:
                rebound_abs = level
            marks[k][what] = (at, level, True)
            out.moved.append(f"sweep {sw + 1} {what} at {at * 1e3:.1f} ms")
        if np.isfinite(rebound_abs):
            va[k] = rebound_abs - vb[k]
            if "rebound" in marks[k] and len(marks[k]["rebound"]) == 2:
                marks[k]["rebound"] = (marks[k]["rebound"][0], rebound_abs)
        if counts[k] == 0 and cur[k] < -reject:
            pk = i0 + int(np.argmin(_smooth(seg, smooth_n)))
            tau_on[k] = _tau(t, v, i0, pk, vb[k], s)
            back = i1 + int(np.argmax(_smooth(clean[i1:i1 + post_n], smooth_n)[:-smooth_n or None]))
            tau_off[k] = _tau(t, clean, i1, back, ve[k], s)
        sp = [x for x in all_spikes[k] if i0 <= x.peak_index < i1]
        if sp:
            lat[k] = (sp[0].threshold_index - i0) / fs * 1e3
            shapes[k] = first_spike_shape(v, spikes.dvdt(v, fs), fs, sp[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        far = np.abs(cur) >= reject
        r_peak = np.where(far & (counts == 0), (vm - vb) / cur * 1e3, NAN)
        r_end = np.where(far & (counts == 0), (ve - vb) / cur * 1e3, NAN)
        sag = np.where(counts == 0, ve - vm, NAN)
        sag_pct = np.where(far & (counts == 0), np.abs(ve - vm) / np.abs(vm - vb) * 100, NAN)
        adp_pct = np.where(far & (counts == 0), np.abs(va) / np.abs(ve - vb) * 100, NAN)
        cap = tau_on / r_end * 1e3
    for k, r in enumerate(rows):
        row = {"sweep": r["sweep"], "current_pa": float(cur[k]), "v_peak": vm[k], "v_steady_nex": ve[k],
               "r_peak": r_peak[k], "r_steady": r_end[k], "sag_mv": sag[k], "sag_pct": sag_pct[k],
               "rebound_mv": va[k], "rebound_pct": adp_pct[k], "tau_on_ms": tau_on[k], "tau_off_ms": tau_off[k],
               "capacitance_pf": cap[k], "latency_ms": lat[k]}
        row.update(shapes[k])
        out.rows.append(row)
        out.drawables += [Drawable("marker", what + (", moved by hand" if len(xy) == 3 else ""), x=np.array([xy[0]]),
                                   y=np.array([xy[1]]), sweep=r["sweep"], domain="time") for what, xy in marks[k].items()]
    v_ = out.values
    # the holding current: the recorded current before the step when the amplifier's
    # current is recorded (NeuroExpress's Irest), else the command's level there
    from ..core.recording import Role

    mon = next((c for c in recording.channels if c.role is Role.MONITOR and c.units.strip().lower() in ("pa", "na")),
               None)
    if mon is not None:
        x = np.asarray(recording.read(mon), float)[:, i0 - nb:i0]
        v_["holding_current"] = float(x.mean() * (1e3 if mon.units.strip().lower() == "na" else 1.0))
        v_["holding_current_source"] = f"the recorded current ({mon.name}) before the step"
    elif recording.command is not None:
        v_["holding_current"] = float(np.asarray(recording.command, float)[:, i0 - nb:i0].mean())
        v_["holding_current_source"] = "the command before the step (no current recorded)"
    # at rest: lines through the near-zero hyperpolarizing steps
    for key, y in (("r_peak0", r_peak), ("r_steady0", r_end)):
        v0, sl, npts = line_at_zero(cur, y, mid, -reject)
        v_[key], v_[key.replace("0", "_slope0")] = v0, sl * 1e3       # MOhm per nA
        v_["n_resistance_steps"] = npts
        if np.isfinite(v0):
            out.drawables.append(Drawable("curve", f"{key[:-1]}: the line to 0 pA", x=np.array([mid, 0.0]),
                                          y=np.array([v0 + sl * mid, v0]), domain="current"))
    v_["tau0"], _, _ = line_at_zero(cur, tau_on, mid, -reject)
    v_["tau_off0"], _, _ = line_at_zero(cur, tau_off, mid, -reject)
    # capacitance at rest: tau at rest over R at rest (NeuroExpress's test cell: 162
    # against its 157 pF; a line through each step's tau / R gives 192)
    r0, t0 = v_.get("r_steady0", NAN), v_["tau0"]
    v_["capacitance0"] = t0 / r0 * 1e3 if np.isfinite(t0) and np.isfinite(r0) and r0 > 0 else NAN
    # sag and rebound: a line through the non-firing steps from the midpoint up (the
    # reject range left out), as NeuroExpress's sag / ADP midpoint divides them
    near = (cur >= mid) & (np.abs(cur) >= reject) & (counts == 0)
    v_["sag_pct0"], _, _ = line_at_zero(cur[near], sag_pct[near], -np.inf, np.inf)
    v_["rebound_pct0"], _, _ = line_at_zero(cur[near], adp_pct[near], -np.inf, np.inf)
    _, v_["sag_slope"], _ = line_at_zero(cur[near] / 1e3, sag[near], -np.inf, np.inf)            # mV per nA
    _, v_["rebound_slope"], _ = line_at_zero(cur[near] / 1e3, va[near], -np.inf, np.inf)
    # the V-I relation: a straight line through the non-firing steps from the
    # midpoint to minus it (its slope the resistance from the V-I fit)
    sub = (counts == 0) & np.isfinite(vm) & np.isfinite(ve) & (cur >= mid) & (cur <= -mid)
    for key, y in (("peak", vm), ("steady", ve)):
        if np.unique(cur[sub]).size >= 2:
            p = np.polyfit(cur[sub], y[sub], 1)
            v_[f"r_vi_{key}0"] = float(p[0] * 1e3)
            v_[f"v0_{key}"] = float(p[1])
            xs = np.linspace(cur[sub].min(), cur[sub].max(), 2)
            out.drawables.append(Drawable("curve", f"{'v_peak' if key == 'peak' else 'v_steady_nex'}: the V-I line",
                                          x=xs, y=np.polyval(p, xs), domain="current"))
        else:
            v_[f"r_vi_{key}0"] = v_[f"v0_{key}"] = NAN
    _firing(out, cur, counts, lat, rows, all_spikes, i0, i1, fs, recording, s)
    _evaluate(out, cur, counts, r_end, sag_pct, va, vb, rows, all_spikes, i0, i1, fs, s)
    return out


def _tau(t, v, a, b, base, s) -> float:
    from .intrinsic import tau_fit

    tau, _, _ = tau_fit(t, v, a, b, base, s["tau_start_fraction"], s["tau_end_fraction"], 0, 0, 1.0 / (t[1] - t[0]))
    return tau


def _firing(out: Curves, cur, counts, lat, rows, all_spikes, i0, i1, fs, recording, s) -> None:
    v_ = out.values
    order = np.argsort(cur)
    firing = [k for k in order if cur[k] > 0 and counts[k] > 0]
    lim = s["io_spike_limit"]
    if firing:
        k0 = firing[0]
        upto = [k for k in order if cur[k] >= cur[k0]]
        stop = next((j for j, k in enumerate(upto) if counts[k] >= lim), len(upto) - 1)
        use = upto[:stop + 1]
        if len(use) >= 2:
            g = fits.linear(cur[use], counts[use].astype(float), purpose="I-O gain")
            v_["io_gain"] = g["slope"] * 1e3                      # spikes per nA
            v_["io_rheobase"] = float(-g["intercept"] / g["slope"]) if g["slope"] > 0 else NAN
        else:
            v_["io_gain"] = v_["io_rheobase"] = NAN
        lat_c = cur[firing]
        target = 1.3 * cur[k0]
        v_["latency_1p3"] = float(np.interp(target, lat_c, lat[firing])) if lat_c.size and target <= lat_c.max() else NAN
    else:
        v_["io_gain"] = v_["io_rheobase"] = v_["latency_1p3"] = NAN
    climit = s["cumulative_limit_pa"]
    within = (cur > 0) & (cur <= climit)
    v_["cumulative_spikes"] = int(counts[within].sum())
    lat_w = lat[within & (counts > 0)]
    v_["mean_latency_to_limit"] = float(lat_w.mean()) if lat_w.size else NAN
    # where the lowest voltage between two spikes falls in their interval; ISI statistics per step
    phases = []
    for k, r in enumerate(rows):
        sp = [x for x in all_spikes[k] if i0 <= x.peak_index < i1]
        v = None
        if len(sp) >= 2:
            v = np.asarray(recording.sweep(k), float)
            for a, b in zip(sp[:-1], sp[1:]):
                if b.peak_index - a.peak_index > 2:
                    m = a.peak_index + int(np.argmin(v[a.peak_index:b.peak_index]))
                    phases.append((m - a.peak_index) / (b.peak_index - a.peak_index))
        isi = np.diff([x.peak_index for x in sp]) / fs * 1e3 if len(sp) >= 2 else np.array([])
        out.rows[k].update(isi_stats(isi))
    v_["min_vm_phase"] = float(np.mean(phases)) if phases else NAN
    # the first spike, averaged over the steps that fire (NeuroExpress's "mean 1st spike")
    for key in ("threshold_5_mv", "amplitude_5_mv", "half_width_5_ms", "time_to_peak_ms", "fall_time_ms", "ahp_5_mv",
                "upstroke", "downstroke", "upstroke_phase", "downstroke_phase", "kink_max", "time_to_kink_ms",
                "kink_mv", "initial_phase_slope", "kink_phase_slope"):
        vals = np.array([r.get(key, NAN) for r in out.rows if r.get("latency_ms") == r.get("latency_ms")], float)
        vals = vals[np.isfinite(vals)]
        v_[f"first_{key}"] = float(vals.mean()) if vals.size else NAN


def _evaluate(out: Curves, cur, counts, r_end, sag_pct, va, vb, rows, all_spikes, i0, i1, fs, s) -> None:
    """Statements read from the data, each with its rule and numbers (NeuroExpress's evaluation)."""
    v_, say = out.values, out.statements
    reject = s["reject_range_pa"]
    neg = int(np.sum((cur < -reject) & (counts == 0)))
    if neg < s["min_negative_steps"]:
        say.append(("Too few hyperpolarizing steps for the resistance at rest",
                    f"{neg} non-firing step(s) below -{reject:g} pA; at least {s['min_negative_steps']} are needed", True))
    rest = float(np.nanmedian(vb)) if np.isfinite(vb).any() else NAN
    if np.isfinite(rest) and rest > s["depolarized_rest_mv"]:
        say.append(("The cell rests depolarized", f"{rest:.1f} mV before the steps (the rule: above "
                    f"{s['depolarized_rest_mv']:g} mV): its spikes and resistances are a damaged cell's", True))
    r0 = v_.get("r_steady0", NAN)
    mid = s["resistance_midpoint_pa"]
    hyper = (cur < -reject) & np.isfinite(r_end)
    beyond = int(np.sum(hyper & (cur < mid)))
    if np.isfinite(r0) and r0 > 0 and beyond >= s["rectification_min_steps"]:
        k = int(np.argmin(np.where(hyper, cur, np.inf)))
        drop = (r0 - r_end[k]) / r0
        if drop >= s["rectification_fraction"]:
            say.append(("Inward rectification, possibly a Kir current",
                        f"steady-state resistance {r_end[k]:.0f} MOhm at {cur[k]:g} pA against {r0:.0f} at 0 pA "
                        f"({100 * drop:.0f}% lower; the rule: {100 * s['rectification_fraction']:.0f}% or more, "
                        f"{beyond} steps beyond {mid:g} pA)", False))
    dep = (cur > reject) & np.isfinite(r_end)
    if np.isfinite(r0) and r0 > 0 and int(np.sum(dep & (cur > -mid))) >= s["rectification_min_steps"]:
        k = int(np.argmax(np.where(dep, cur, -np.inf)))
        drop = (r0 - r_end[k]) / r0
        if drop >= s["rectification_fraction"]:
            say.append(("Outward rectification",
                        f"steady-state resistance {r_end[k]:.0f} MOhm at +{cur[k]:g} pA against {r0:.0f} at 0 pA "
                        f"({100 * drop:.0f}% lower; the rule: {100 * s['rectification_fraction']:.0f}% or more)", False))
    # NeuroExpress's own rule, read off its numbers on 25 lab files (it held on all 25):
    # the peak resistance at rest more than 1.3 times the peak V-I resistance
    rp, rvp = v_.get("r_peak0", NAN), v_.get("r_vi_peak0", NAN)
    lim = 1 + s["resistance_disagree_fraction"]
    if np.isfinite(rp) and np.isfinite(rvp) and rvp > 0 and rp > lim * rvp:
        say.append(("The resistance measures differ by more than "
                    f"{100 * s['resistance_disagree_fraction']:.0f}%",
                    f"the peak resistance at rest {rp:.0f} MOhm against {rvp:.0f} from the V-I line ({rp / rvp:.2f} "
                    f"times; the rule: more than {lim:g} times)", True))
    if hyper.any():
        k = int(np.argmin(np.where((cur < -reject) & np.isfinite(sag_pct), cur, np.inf)))
        if np.isfinite(sag_pct[k]) and sag_pct[k] >= s["strong_sag_pct"]:
            say.append(("A strong voltage sag, possibly an h current",
                        f"{sag_pct[k]:.0f}% of the peak deflection at {cur[k]:g} pA, the most hyperpolarizing "
                        f"step (the rule: {s['strong_sag_pct']:g}% or more)", False))
        if np.isfinite(va[k]) and va[k] >= s["strong_rebound_mv"]:
            say.append(("A strong rebound, possibly a T current",
                        f"{va[k]:.1f} mV above the baseline after the {cur[k]:g} pA step, spikes left out (the rule: "
                        f"{s['strong_rebound_mv']:g} mV or more)", False))
    above = [k for k in np.argsort(cur) if cur[k] > 0 and counts[k] > 0]
    if not above:
        say.append(("No action potentials", "no step fired", True))
        return
    singles = sum(1 for k in above if counts[k] == 1)
    if len(above) >= 3 and singles / len(above) >= s["single_spike_fraction"]:
        say.append(("An initial single spike, possibly a D current",
                    f"{singles} of {len(above)} firing steps fire exactly one spike (the rule: "
                    f"{100 * s['single_spike_fraction']:.0f}% or more of at least 3)", False))
    gaps = []
    for k in above:
        sp = [x for x in all_spikes[k] if i0 <= x.peak_index < i1]
        if len(sp) >= 4:
            isi = np.diff([x.peak_index for x in sp])
            gaps.append(isi.max() / np.median(isi))
    pausing = sum(1 for g in gaps if g >= s["stutter_gap_ratio"])
    if gaps and pausing >= max(1, len(gaps) / 2):
        say.append(("The firing stutters",
                    f"{pausing} of {len(gaps)} steps with 4 or more spikes pause for {s['stutter_gap_ratio']:g} times "
                    f"their median interval or longer (longest {max(gaps):.1f} times; the rule: half the steps or "
                    "more)", False))
    top = counts[above]
    if top.size >= 3 and top[-1] <= s["block_fraction"] * top.max() and top.max() >= 3:
        say.append(("Possible depolarization block",
                    f"the largest step fires {top[-1]} spike(s) against {top.max()} at its best (the rule: "
                    f"{100 * s['block_fraction']:.0f}% of the best or fewer)", True))
    hw = v_.get("first_half_width_5_ms", NAN)
    if np.isfinite(hw) and hw > s["broad_spike_ms"]:
        say.append(("The spikes are broad", f"the first spikes' half-width is {hw:.2f} ms on average (the rule: "
                    f"over {s['broad_spike_ms']:g} ms)", True))
