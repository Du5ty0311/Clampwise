"""Thresholds from brief and slow current injections: the Allen Institute's
short squares and ramps (Gouwens et al. 2019; IPFX's short-square and ramp
features, measured here).

- **Short squares**: one brief current pulse per sweep (3 ms in the Allen
  protocol), its size stepping across sweeps. The threshold is the smallest
  pulse that fires an action potential within ``window_ms`` of the pulse's
  start (IPFX: the lowest amplitude among the sweeps with a spike); its
  resolution is how far below it the largest pulse that fired nothing was. The
  action potential at threshold is described on every sweep of that size,
  averaged (IPFX's common-amplitude sweeps).
- **Ramps**: the current rising steadily. The threshold is the current the
  ramp had reached at the first action potential's threshold (IPFX's
  threshold_i), with that spike's voltage threshold and its latency from the
  ramp's start; the median over the ramps.

Spikes are tools.spikes.find's (IPFX's detection, step for step).
"""

from __future__ import annotations

import numpy as np

from .registry import Output, Setting, register
from .result import Drawable, Flag, Result
from .routing import ramp_of
from .tools import spikes

NAN = float("nan")
SHAPE = ("threshold_mv", "peak_mv", "amplitude_mv", "half_width_ms", "upstroke", "downstroke", "fast_trough_mv")


def _shape(x: spikes.Spike) -> dict:
    return {k: float(getattr(x, k)) for k in SHAPE}


SHORT_OUTPUTS = (
    Output("ss_threshold_pa", "The smallest short pulse that fires an action potential (the lowest amplitude "
           "among the sweeps with a spike).", "pA", "Gouwens et al. 2019", label="Short-square threshold",
           headline=True),
    Output("ss_resolution_pa", "The gap between the threshold and the largest pulse that did not fire: the threshold's "
           "precision.", "pA", label="Short-square threshold: resolution"),
    Output("ss_width_ms", "The pulses' duration.", "ms", label="Pulse width"),
    Output("ss_n_at_threshold", "Sweeps at the threshold's size (the action potential is their average).", "count",
           label="Sweeps at threshold"),
    Output("ss_latency_ms", "From the pulse's start to the action potential's threshold, at threshold.", "ms",
           label="Latency at threshold"),
    Output("ss_ap_threshold_mv", "The action potential's threshold (dV/dt crossing 20 mV/ms), at threshold.", "mV",
           "Gouwens et al. 2019", label="AP threshold at threshold"),
    Output("ss_ap_peak_mv", "Its peak.", "mV", "Gouwens et al. 2019", label="AP peak at threshold"),
    Output("ss_ap_amplitude_mv", "Its peak minus its threshold.", "mV", label="AP amplitude at threshold"),
    Output("ss_ap_half_width_ms", "Its width at half height from the fast trough.", "ms", "Gouwens et al. 2019",
           label="AP half-width at threshold"),
    Output("ss_ap_upstroke", "Its steepest rise.", "mV/ms", "Gouwens et al. 2019", label="AP upstroke at threshold"),
    Output("ss_ap_downstroke", "Its steepest fall.", "mV/ms", "Gouwens et al. 2019", label="AP downstroke at threshold"),
    Output("amplitude_pa", "The pulse's size against the holding level.", "pA", level="sweep"),
    Output("width_ms", "The pulse's duration.", "ms", level="sweep"),
    Output("spikes", "Action potentials within the window after the pulse's start.", "count", level="sweep"),
    Output("latency_ms", "From the pulse's start to the first action potential's threshold.", "ms", level="sweep"),
)


@register("short_square", "Short-square threshold", version=1, types={"cc_short_square"}, outputs=SHORT_OUTPUTS,
          settings=(Setting("window_ms", 20.0, "Spikes counted up to (ms after the pulse's start)", "An action "
                            "potential evoked by a short pulse occurs within a few milliseconds of it.", "ms", "",
                            minimum=1.0),),
          description="The smallest brief (short-square) current pulse that fires the cell, to the protocol's "
                      "resolution, and the action potential it evokes.",
          source="Gouwens et al. 2019")
def short_square(recording, context, s) -> Result:
    out = Result("short_square", 1)
    st, fs, cmd = context.stimulus, recording.sample_rate_hz, recording.command
    rows, shapes = [], {}
    for k in range(recording.n_sweeps):
        if k >= len(st.times_s) or not len(st.times_s[k]):
            continue
        a = int(round(float(st.times_s[k][0]) * fs))
        w = float(st.widths_s[k][0])
        b = a + max(1, int(round(w * fs)))
        hold = float(np.median(cmd[k, max(0, a - int(0.01 * fs)):a])) if a > 0 else 0.0
        amp = float(np.median(cmd[k, a:b])) - hold
        v = np.asarray(recording.sweep(k), float)
        sp = [x for x in spikes.find(v, fs, window=(a, min(v.size, a + int(s["window_ms"] / 1e3 * fs)))) if x.full]
        rows.append({"sweep": k, "amplitude_pa": amp, "width_ms": w * 1e3, "spikes": len(sp),
                     "latency_ms": (sp[0].threshold_index - a) / fs * 1e3 if sp else NAN,
                     **(_shape(sp[0]) if sp else {key: NAN for key in SHAPE})})
        if sp:
            shapes[k] = sp[0]
            out.drawables.append(Drawable("marker", "spike threshold", x=np.array([sp[0].threshold_index / fs]),
                                          y=np.array([sp[0].threshold_mv]), sweep=k, domain="time"))
            out.drawables.append(Drawable("marker", "spike peak", x=np.array([sp[0].peak_index / fs]),
                                          y=np.array([sp[0].peak_mv]), sweep=k, domain="time"))
    out.tables["sweeps"] = rows
    if not rows:
        out.notes.append("no pulse found in any sweep")
        return out
    amps = np.round([r["amplitude_pa"] for r in rows], 1)
    fired = [r for r in rows if r["spikes"] > 0]
    out.values["ss_width_ms"] = float(np.median([r["width_ms"] for r in rows]))
    if not fired:
        out.values.update({"ss_threshold_pa": NAN, "ss_resolution_pa": NAN})
        out.flags.append(Flag("no spike", f"no pulse fired an action potential, up to {amps.max():g} pA: the threshold "
                              "is above the largest pulse", severity="note"))
        return out
    thr = float(min(round(r["amplitude_pa"], 1) for r in fired))
    silent = [a for a in amps if a < thr and not any(r["spikes"] and round(r["amplitude_pa"], 1) == a for r in rows)]
    out.values["ss_threshold_pa"] = thr
    out.values["ss_resolution_pa"] = float(thr - max(silent)) if silent else NAN
    if not silent:
        out.flags.append(Flag("threshold not bracketed", f"every pulse fired, down to {thr:g} pA: the threshold is at "
                              "or below it", severity="note"))
    at = [r for r in fired if round(r["amplitude_pa"], 1) == thr]
    out.values["ss_n_at_threshold"] = len(at)
    out.values["ss_latency_ms"] = float(np.mean([r["latency_ms"] for r in at]))
    for key, name in (("threshold_mv", "ss_ap_threshold_mv"), ("peak_mv", "ss_ap_peak_mv"),
                      ("amplitude_mv", "ss_ap_amplitude_mv"), ("half_width_ms", "ss_ap_half_width_ms"),
                      ("upstroke", "ss_ap_upstroke"), ("downstroke", "ss_ap_downstroke")):
        out.values[name] = float(np.nanmean([r[key] for r in at]))
    above = sorted({round(r["amplitude_pa"], 1) for r in rows if r["amplitude_pa"] >= thr and not r["spikes"]})
    if above:
        out.flags.append(Flag("silent above threshold", f"pulses of {', '.join(f'{a:g}' for a in above[:5])} pA, at or "
                              f"above the {thr:g} pA threshold, fired nothing (threshold near its edge, or run-down)",
                              severity="note"))
    out.choose("short-square threshold", f"{thr:g} pA" + (f", within {out.values['ss_resolution_pa']:g} pA"
                                                           if silent else ""),
               "the smallest pulse with a spike")
    return out


RAMP_OUTPUTS = (
    Output("ramp_threshold_pa", "The ramp current at the first action potential's threshold, the median over the "
           "ramps.", "pA", "Gouwens et al. 2019",
           label="Ramp threshold (current)", headline=True),
    Output("ramp_threshold_mv", "The first action potential's threshold voltage, the median over the ramps.", "mV",
           "Gouwens et al. 2019", label="Ramp threshold (voltage)"),
    Output("ramp_latency_s", "From the ramp's start to the first action potential's threshold, the median.", "s",
           label="Ramp latency"),
    Output("ramp_rate_pa_s", "How fast the current rises, the median over the ramps.", "pA/s", label="Ramp rate"),
    Output("ramp_n", "Ramps found.", "count", label="Ramps"),
    Output("ramp_n_firing", "Ramps that fired.", "count", label="Ramps that fired"),
    Output("rate_pa_s", "This ramp's rate.", "pA/s", level="sweep"),
    Output("threshold_pa", "The current at its first action potential's threshold.", "pA", level="sweep"),
    Output("latency_s", "From its start to the first action potential's threshold.", "s", level="sweep"),
    Output("spikes", "Action potentials during the ramp.", "count", level="sweep"),
)


@register("ramp", "Ramp threshold", version=1, types={"cc_ramp"}, outputs=RAMP_OUTPUTS,
          description="The current at which a slowly rising ramp first fires the cell, with that spike's voltage "
                      "threshold and latency.",
          source="Gouwens et al. 2019")
def ramp(recording, context, s) -> Result:
    out = Result("ramp", 1)
    fs, cmd = recording.sample_rate_hz, recording.command
    rows = []
    for k in range(recording.n_sweeps):
        found = ramp_of(recording, k)
        if found is None:
            continue
        a, b, rate = found
        hold = float(np.median(cmd[k, max(0, a - int(0.01 * fs)):a])) if a > 0 else 0.0
        v = np.asarray(recording.sweep(k), float)
        sp = [x for x in spikes.find(v, fs, window=(a, min(v.size, b))) if x.full]
        first = sp[0] if sp else None
        rows.append({"sweep": k, "rate_pa_s": rate, "spikes": len(sp),
                     "threshold_pa": float(cmd[k, first.threshold_index]) - hold if first else NAN,
                     "latency_s": (first.threshold_index - a) / fs if first else NAN,
                     "threshold_mv": first.threshold_mv if first else NAN})
        if first:
            out.drawables.append(Drawable("marker", "spike threshold", x=np.array([first.threshold_index / fs]),
                                          y=np.array([first.threshold_mv]), sweep=k, domain="time"))
    out.tables["ramps"] = rows
    fired = [r for r in rows if r["spikes"]]
    out.values.update({"ramp_n": len(rows), "ramp_n_firing": len(fired),
                       "ramp_rate_pa_s": float(np.median([r["rate_pa_s"] for r in rows])) if rows else NAN,
                       "ramp_threshold_pa": float(np.median([r["threshold_pa"] for r in fired])) if fired else NAN,
                       "ramp_threshold_mv": float(np.median([r["threshold_mv"] for r in fired])) if fired else NAN,
                       "ramp_latency_s": float(np.median([r["latency_s"] for r in fired])) if fired else NAN})
    if rows and not fired:
        out.flags.append(Flag("no spike", "no ramp fired an action potential", severity="note"))
    if fired:
        out.choose("ramp threshold", f"the median of {len(fired)} ramp(s)", "the current at the first spike's "
                   "threshold")
    return out
