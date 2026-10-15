"""Passive properties and the recording's own errors (B20).

**Membrane test** (voltage clamp, every sweep with a test step of 1-20 mV
lasting at least 2 ms). The method of Clampex's Membrane Test (the Axon Guide;
Gillis in Sakmann & Neher 1995), on each sweep:

- steady state: the mean of the last 20% of the step, and the holding current
  the mean of as long just before it (pCLAMP's I1 and I2, the pCLAMP 10 User
  Guide p. 192): a mean, so the noise dithers below one count of the
  digitizer, where a median snapped to whole counts (the model cell's 20 mV
  steps all read Rt 546.13 MOhm at 6.1 pA a count);
- the capacitive transient (current minus steady state) fitted with a single
  exponential from 80% to 20% of its peak (the peak itself is filtered, and a
  neuron's slow dendritic tail drags the extrapolation down), extrapolated back
  to the step's onset;
- Rs = dV / (extrapolated transient + steady-state step): at the onset all the
  current flows through Rs. The raw-peak Rs (dV / peak) is beside it; it reads
  high because the peak is filtered and falls between samples;
- Rt = dV / steady-state step; Rm = Rt - Rs;
- Cm two ways, both reported, because capacitance depends on how it is
  measured in a neuron that is not isopotential (Golowasch et al. 2009,
  J Neurophysiol 102:2161): from the time constant, tau (Rs + Rm) / (Rs Rm);
  and from the charge of the transient, Q (Rs + Rm)^2 / (Rm^2 dV) (the
  charge the Rs-(Rm || Cm) circuit's transient carries).

**Series-resistance error** (voltage clamp, every evoked response): the
voltage error I x Rs(uncompensated) at the response's peak and the clamp's time
constant Rs(uncompensated) x Cm (the Axon Guide; Sigworth in Sakmann & Neher;
Williams & Mitchell 2008, Nat Neurosci 11:790). The flag limits are proposals,
marked as such, for the user to decide.

**Bridge balance** (current clamp, step families): an instant voltage jump at
a step's onset is the drop across uncompensated electrode resistance, not the
membrane; it is measured and flagged (B20, item 5).
"""

from __future__ import annotations

import numpy as np

from ..core.recording import Recording
from .registry import EVOKED, Output, Setting, register
from .result import Flag, Result
from .tools import fits, ledger

NAN = float("nan")
VC_TYPES = {"membrane_test", "vc_evoked", "vc_spontaneous", "vc_steps", "dsi"}


def test_steps(recording: Recording, sweep: int, lo_mv: float, hi_mv: float, min_s: float) -> tuple[int, int, float] | None:
    """(onset index, end index, step in mV) of the sweep's first test step: a
    command step of ``lo_mv``-``hi_mv`` from the level before it, at least ``min_s`` long."""
    cmd = recording.command
    if cmd is None:
        return None
    c = cmd[sweep]
    fs = recording.sample_rate_hz
    d = np.flatnonzero(np.abs(np.diff(c)) > 0.25)
    edges = np.concatenate([[0], d + 1, [c.size]])
    for a, b in zip(edges[:-1], edges[1:]):
        if a == 0 or b - a < min_s * fs:
            continue
        before = float(np.median(c[max(0, a - int(0.002 * fs)):a]))
        step = float(np.median(c[a:b])) - before
        if lo_mv <= abs(step) <= hi_mv:
            return int(a), int(b), step
    return None


def measure_step(t: np.ndarray, current: np.ndarray, i0: int, i1: int, dv_mv: float, fs: float) -> dict:
    """Rs, Rm, Cm (two ways), tau and holding from one test step."""
    out = {"holding_pa": NAN, "rs_mohm": NAN, "rs_raw_mohm": NAN, "rm_mohm": NAN, "cm_tau_pf": NAN,
           "cm_charge_pf": NAN, "tau_ms": NAN, "fit_r2": NAN, "peak_bound": False, "rt_mohm": NAN}
    n = i1 - i0
    if n < 10:
        return out
    tail = max(3, n // 5)                               # pCLAMP's: the last 20% of the step, and as long before it
    holding = float(np.mean(current[max(0, i0 - tail):i0])) if i0 > 0 else float(current[0])
    out["holding_pa"] = holding
    steady = float(np.mean(current[i1 - tail:i1]))
    d_ss = steady - holding
    if d_ss * dv_mv > 0:
        out["rt_mohm"] = dv_mv / d_ss * 1e3                # the step over its steady current, as recorded
    sign = np.sign(dv_mv)
    y = current[i0:i1] - steady
    search = max(3, min(n // 3, int(round(0.005 * fs))))
    peak = int(np.argmax(sign * y[:search]))
    peak_value = float(y[peak] + steady - holding)
    if peak_value != 0:
        out["rs_raw_mohm"] = dv_mv / peak_value * 1e3
    dev = sign * y
    top = dev[peak]
    after = np.arange(peak + 1, n)
    below80 = after[dev[peak + 1:] <= 0.8 * top]
    start = int(below80[0]) if below80.size else peak + 1
    below20 = np.flatnonzero(dev[start:] <= 0.2 * top)
    stop = start + (int(below20[0]) + 1 if below20.size else n - tail - start)
    if stop - start < 4:
        below5 = np.flatnonzero(dev[start:] <= 0.05 * top)
        stop = start + (int(below5[0]) + 1 if below5.size else stop - start)
    stop = min(max(stop, start + 3), n)
    tt = t[i0 + start:i0 + stop] - t[i0]
    fit = fits.exp_decay(tt, y[start:stop], offset=0.0, purpose="membrane test: the capacitive transient, 80-20%")
    if not fit.ok or fit["tau"] <= 0:
        return out
    tau = fit["tau"]
    if tt.size and tt[0] / tau > 50:            # a fit to noise: extrapolating it back would overflow
        return out
    if tau < 2.0 / fs:
        # faster than the recording resolves (an open pipette or a seal, no cell: one lab recording read
        # tau 0.04 ms at 20 kHz): extrapolated back, it put Rs near 0 (1e-12 MOhm) in 392 sweeps; not measured
        return out
    a0 = fit["amp"] * np.exp(tt[0] / tau) if tt.size else fit["amp"]
    out["tau_ms"], out["fit_r2"] = tau * 1e3, fit.r2
    # a step too short for the transient to finish: take out what the fit says is left in the steady state
    t_ss = t[i1 - tail:i1] - t[i0]
    d_ss = d_ss - float(a0 * np.mean(np.exp(-t_ss / tau)))
    if a0 * sign <= 0 or d_ss * sign <= 0:
        return out
    onset_current = a0 + d_ss
    if sign * peak_value > sign * onset_current:          # cannot be less than the recorded peak
        onset_current = peak_value
        out["peak_bound"] = True
    rs = dv_mv / onset_current * 1e3                      # mV / pA = GOhm -> MOhm
    rt = dv_mv / d_ss * 1e3
    rm = rt - rs
    out["rs_mohm"], out["rm_mohm"] = rs, rm
    if rs > 0 and rm > 0:
        out["cm_tau_pf"] = tau * (rs + rm) / (rs * rm) * 1e6
        # the transient's charge, from the onset to the steady-state window (the fitted
        # exponential stands in for the first samples, which the filter rounds)
        q_fit = a0 * tau * (1 - np.exp(-(t[i1 - tail] - t[i0]) / tau))        # pA * s
        q_data = float(np.trapezoid(y[start:n - tail], t[i0 + start:i1 - tail] - t[i0])) if n - tail > start + 1 else 0.0
        q_head = a0 * tau * (1 - np.exp(-(t[i0 + start] - t[i0]) / tau))
        q = q_head + q_data if np.isfinite(q_data) else q_fit
        # Rs-(Rm||Cm): the transient carries Q = dV Cm Rm^2 / (Rs + Rm)^2
        out["cm_charge_pf"] = q * (rs + rm) ** 2 / (rm ** 2 * dv_mv) * 1e3      # pA*s/mV = nF -> pF
    return out


@register("membrane_test", "Membrane test: series resistance, input resistance, capacitance", version=1,
          types=VC_TYPES,
          description="Rs, Rm, Cm and the holding current from the voltage-clamp test step in each sweep, the way "
                      "Clampex's Membrane Test computes them; capacitance two ways, both named.",
          source="the Axon Guide; Gillis in Sakmann & Neher 1995; Golowasch et al. 2009 (capacitance)",
          settings=(
              Setting("min_step_mv", 1.0, "Test step at least (mV)", "", "mV", advanced=True, minimum=0.1),
              Setting("max_step_mv", 20.0, "Test step at most (mV)", "Larger command steps are the protocol, not a "
                      "test pulse.", "mV", advanced=True),
              Setting("min_step_ms", 2.0, "Test step at least (ms)", "", "ms", advanced=True),
              Setting("reference_sweeps", 3, "Compare the first and last sweeps (n)", "Rs change is the median of "
                      "the last n sweeps against the first n.", minimum=1, advanced=True),
              Setting("no_cell_below_mohm", 20.0, "No cell below (MOhm)", "A step whose total resistance is under "
                      "this is the pipette alone (open, in the bath): no Rs, Rm or Cm is reported for it "
                      "(Rs is reported once on the cell). The quality checks' input-resistance floor.", "MOhm",
                      advanced=True, minimum=0.0),
          ),
          outputs=(
              Output("rs", "Series (access) resistance: the step over the current at its onset (the transient fitted "
                     "from 80% to 20% of its peak, extrapolated back), median of the sweeps.", "MOhm",
                     "Clampex Membrane Test (the Axon Guide)", label="Series resistance (Rs)", headline=True),
              Output("rs_raw", "Rs from the recorded peak (reads high: the peak is filtered).", "MOhm",
                     label="Rs from the raw peak"),
              Output("rs_change", "Rs of the last sweeps over the first, minus 1.", "fraction",
                     label="Rs change over the recording", headline=True),
              Output("rm", "Membrane resistance: the total resistance from the steady state minus Rs, median.", "MOhm",
                     label="Membrane resistance (Rm)", headline=True),
              Output("cm_tau", "Capacitance from the transient's time constant, tau (Rs + Rm) / (Rs Rm).", "pF",
                     "two-compartment relation", label="Capacitance, from tau", headline=True),
              Output("cm_charge", "Capacitance from the transient's charge, Q (Rs + Rm)^2 / (Rm^2 dV).", "pF",
                     "Golowasch et al. 2009", label="Capacitance, from charge"),
              Output("tau", "Time constant of the capacitive transient.", "ms", label="Transient tau"),
              Output("holding", "Holding current before the test step, median of the sweeps.", "pA",
                     label="Holding current", headline=True),
              Output("rs_rm", "Rs over Rm: above 0.15 the clamp is no longer small against the cell.", "ratio",
                     "", label="Rs / Rm"),
          ))
def run_membrane_test(recording: Recording, context, s) -> Result | None:
    if not recording.is_voltage_clamp or recording.command is None:
        return None
    out = Result("membrane_test", 1)
    fs, t = recording.sample_rate_hz, recording.time_s
    rows = []
    for k in range(recording.n_sweeps):
        step = test_steps(recording, k, s["min_step_mv"], s["max_step_mv"], s["min_step_ms"] / 1e3)
        if step is None:
            continue
        i0, i1, dv = step
        m = measure_step(t, recording.sweep(k), i0, i1, dv, fs)
        m.update({"sweep": k, "step_mv": dv, "onset_s": i0 / fs,
                  "time_s": float(recording.sweep_start_times_s[k]) if k < recording.sweep_start_times_s.size else NAN})
        m["no_cell"] = bool(m["rt_mohm"] < s["no_cell_below_mohm"])
        if m["no_cell"]:                  # the pipette alone: its transient is no cell's (Rs 0-1.6 MOhm from such fits)
            for key in ("rs_mohm", "rs_raw_mohm", "rm_mohm", "cm_tau_pf", "cm_charge_pf", "tau_ms"):
                m[key] = NAN
        rows.append(m)
    if not rows:
        return None                       # no test step: nothing to say here (quality says Rs was not assessed)
    out.tables["sweeps"] = rows
    open_ = [r for r in rows if r["no_cell"]]
    if open_:
        rts = np.array([r["rt_mohm"] for r in open_])
        out.notes.append(f"no cell in {len(open_)} of {len(rows)} sweep(s): Rt {np.nanmin(rts):.1f}-{np.nanmax(rts):.1f} "
                         f"MOhm, under the {s['no_cell_below_mohm']:g} MOhm floor (the pipette alone): no Rs, Rm or Cm "
                         "given for them")

    def med(key, subset=None):
        vals = np.array([r[key] for r in (subset or rows)], float)
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if vals.size else NAN

    n = min(s["reference_sweeps"], len(rows))
    first, last = med("rs_mohm", rows[:n]), med("rs_mohm", rows[-n:])
    rs, rm = med("rs_mohm"), med("rm_mohm")
    out.values.update({"rs": rs, "rs_raw": med("rs_raw_mohm"), "rm": rm, "cm_tau": med("cm_tau_pf"),
                       "cm_charge": med("cm_charge_pf"), "tau": med("tau_ms"), "holding": med("holding_pa"),
                       "rs_change": (last / first - 1) if len(rows) >= 2 * n and first > 0 else NAN,
                       "rs_rm": rs / rm if rm and np.isfinite(rm) else NAN, "rs_first": first, "rs_last": last,
                       "n_sweeps": len(rows), "step_mv": float(rows[0]["step_mv"])})
    out.choose("test step", f"{rows[0]['step_mv']:+.0f} mV at {rows[0]['onset_s'] * 1e3:.0f} ms, in {len(rows)} of "
               f"{recording.n_sweeps} sweep(s)", "the first command step of 1-20 mV in each sweep")
    if any(r["peak_bound"] for r in rows):
        out.notes.append("in some sweeps the extrapolated onset current was below the recorded peak; the peak was "
                         "used (an upper bound on Rs)")
    ledger.note("membrane test as Clampex computes it (the Axon Guide)", "series resistance and capacitance")
    return out


@register("rs_error", "Series-resistance error of each response", version=1, types=EVOKED - {"cc_evoked"},
          requires=("membrane_test", "evoked"),
          description="How far the cell's voltage was from the command at each response's peak (I x Rs), and how "
                      "fast the clamp could follow (Rs x Cm), with the series resistance left after compensation.",
          source="the Axon Guide; Sigworth in Sakmann & Neher 1995; Williams & Mitchell 2008",
          settings=(
              Setting("compensation_pct", 0.0, "Rs compensation used (%)", "Enter the compensation set on the "
                      "amplifier; it is not stored in Clampex files.", "%", minimum=0.0, maximum=100.0),
              Setting("max_error_mv", 5.0, "Flag a voltage error above (mV)", "Recordings whose clamp error at the "
                      "peak exceeds this are flagged.", "mV",
                      "", advanced=True),
              Setting("max_clamp_tau_fraction", 0.5, "Flag a clamp time constant above (x the 10-90% rise)",
                      "A clamp slower than half the response's rise time distorts its kinetics.",
                      source="Williams & Mitchell 2008", advanced=True),
          ),
          outputs=(
              Output("voltage_error", "Uncompensated Rs times pulse 1's peak current: how far the cell's voltage moved "
                     "from the command at the peak.", "mV", "the Axon Guide", label="Voltage error at pulse 1's peak",
                     headline=True),
              Output("clamp_tau", "Uncompensated Rs times Cm: the time constant with which the clamp follows the cell.",
                     "ms", "the Axon Guide", label="Clamp time constant", headline=True),
              Output("rs_used", "The Rs these come from, and where it was measured.", "MOhm", label="Rs used"),
          ))
def run_rs_error(recording: Recording, context, s) -> Result | None:
    ev = context.result("evoked")
    if ev is None or not recording.is_voltage_clamp:
        return None
    out = Result("rs_error", 1)
    mt = context.result("membrane_test")
    rs, cm, where = NAN, NAN, ""
    if mt is not None and np.isfinite(mt.values.get("rs", NAN)):
        rs, cm, where = mt.values["rs"], mt.values.get("cm_tau", NAN), "the test step in the file"
    else:
        typed = recording.metadata.rs_typed_mohm
        if np.isfinite(typed[0]):
            rs, where = float(np.nanmean(typed)), recording.metadata.rs_words("typed (Clampex's Membrane Test)")
    if not np.isfinite(rs):
        out.notes.append("no Rs: the file has no test step and none was typed (Files > identity: Rs typed); the "
                         "voltage error is not known")
        return out
    unc = rs * (1 - s["compensation_pct"] / 100.0)
    i_peak = ev.values.get("p1_amplitude", NAN)                    # pA
    err = abs(i_peak) * unc * 1e-3 if np.isfinite(i_peak) else NAN  # pA * MOhm = uV -> mV
    tau_c = unc * cm * 1e-3 if np.isfinite(cm) else NAN            # MOhm * pF = us -> ms
    out.values.update({"voltage_error": err, "clamp_tau": tau_c, "rs_used": unc})
    out.choose("Rs", f"{rs:.1f} MOhm from {where}, {s['compensation_pct']:g}% compensated",
               "the file's own test step when it has one, else the typed value")
    if np.isfinite(err) and err > s["max_error_mv"]:
        out.flags.append(Flag("voltage error", f"the cell's voltage moved {err:.1f} mV from the command at pulse 1's "
                              f"peak ({abs(i_peak):.0f} pA through {unc:.1f} MOhm uncompensated)",
                              f"{s['max_error_mv']:g} mV", "the Axon Guide"))
    rise = ev.values.get("p1_rise_10_90", NAN)
    if np.isfinite(tau_c) and np.isfinite(rise) and tau_c > s["max_clamp_tau_fraction"] * rise:
        out.flags.append(Flag("slow clamp", f"the clamp time constant ({tau_c:.2f} ms) is more than "
                              f"{s['max_clamp_tau_fraction']:g} x pulse 1's 10-90% rise ({rise:.2f} ms): the rise and "
                              "peak are filtered by the clamp", f"{s['max_clamp_tau_fraction']:g} x rise",
                              "Williams & Mitchell 2008"))
    return out


@register("bridge", "Bridge balance (current clamp)", version=1, types={"cc_steps", "cc_evoked"},
          description="An instant voltage jump at a current step's onset is the electrode's unbalanced resistance, "
                      "not the membrane.",
          source="the Axon Guide (bridge balance)",
          settings=(Setting("max_jump_mv", 2.0, "Flag an instant jump above (mV)", "A larger instantaneous jump at the "
                            "step onset indicates incomplete bridge balance.",
                            "mV", ""),),
          outputs=(
              Output("jump", "The largest instant voltage change at a step's onset (within 0.2 ms), over the step's "
                     "current.", "mV", label="Instant jump at the step onset", headline=True),
              Output("unbalanced", "That jump over the step current: the resistance the bridge left in.", "MOhm",
                     label="Unbalanced resistance"),
          ))
def run_bridge(recording: Recording, context, s) -> Result | None:
    if recording.is_voltage_clamp or recording.command is None:
        return None
    out = Result("bridge", 1)
    fs = recording.sample_rate_hz
    cmd = recording.command
    best = (0.0, NAN, -1)
    for k in range(recording.n_sweeps):
        c = cmd[k]
        d = np.flatnonzero(np.abs(np.diff(c)) > 5.0)        # a current step of 5 pA or more
        if not d.size:
            continue
        i = int(d[0]) + 1
        di = float(c[i] - c[i - 1])
        v = recording.sweep(k)
        w = max(1, int(round(0.2e-3 * fs)))
        if i - 2 < 0 or i + w >= v.size:
            continue
        jump = float(np.median(v[i:i + w]) - np.median(v[max(0, i - 3 * w):i]))
        # the membrane charges with its time constant: in 0.2 ms it moves little; the bridge moves at once
        if abs(jump) > abs(best[0]):
            best = (jump, di, k)
    if best[2] < 0:
        return None
    jump, di, k = best
    out.values.update({"jump": abs(jump), "unbalanced": abs(jump / di) * 1e3 if di else NAN})
    out.choose("step", f"sweep {k + 1} ({di:+.0f} pA)", "the sweep with the largest instant jump")
    if abs(jump) > s["max_jump_mv"]:
        out.flags.append(Flag("bridge", f"an instant {abs(jump):.1f} mV jump at the step onset (sweep {k + 1}, "
                              f"{di:+.0f} pA): about {abs(jump / di) * 1e3:.0f} MOhm the bridge did not balance; "
                              "voltages during steps include it", f"{s['max_jump_mv']:g} mV",
                              "the Axon Guide"))
    return out
