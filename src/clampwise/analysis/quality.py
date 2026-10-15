"""Recording quality (B10): whether a recording is good enough to use, by
published limits, without removing anything. Every check is a flag with its
value, its limit and its source (flags are evidence, never an edit); a check
that cannot run says so, and "passed" is earned only by checks that ran.

The limits are the recorded ones (vault: Health Checks; Analysis Decisions,
"Voltage-clamp recording quality"):

Current clamp
- resting potential -85 to -55 mV;
- drift across sweeps: 5 mV (the span of 95% of the sweeps from 40 sweeps on);
- spontaneous action potentials before the protocol in 25% of sweeps or more;
- the first action potential's threshold above -20 mV (a voltage offset);
- the current monitor at baseline beyond +/-50 pA (bias current left on);
- noise, RMS over 1.5 ms: 70 uV, reported, not flagged (that number is the
  Allen pipeline's hardware; this rig's runs several times higher);
- noise, RMS over the 500 ms before the stimulus: 0.5 mV, flagged when 25% of
  sweeps exceed it (IPFX's per-sweep criteria, Ephys QC Criteria v1.1);
- drift within a sweep: 1 mV between the 500 ms before the protocol and the
  last 500 ms, only when the sweep has a quiet 500 ms after the protocol.

Voltage clamp
- holding current 200 pA at holdings below -50 mV (500 pA in the CA3 sIPSC DSI
  preset, chloride-rich internal); above -50 mV only a large inward current is
  judged (it cannot flow there through a leak);
- holding change within a recording: 50 pA (a gap-free file in 10 s pieces,
  each piece's level the mode of its all-points histogram so one-sided events
  do not pull it);
- baseline noise SD: 200 pA (Allen synaptic physiology);
- Rs above 25 MOhm or changing by more than 20%; Rs / Rm above 0.15 (Allen
  IPFX QC). Rs from the file's test step, else typed from Clampex's Membrane
  Test; without either, "series resistance not assessed".

Field recordings get no quality report (none of these apply).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.recording import Recording
from .baseline import pre_protocol_end_s
from .evoked import spike_onset
from .registry import Output, Setting, WHOLE_CELL, register
from .result import Flag, Result

NAN = float("nan")
IPFX = "Gouwens et al. 2019"
ALLEN_SYN = "Campagnola et al. 2022"


@dataclass
class Check:
    name: str
    value: str
    limit: str
    ok: bool | None             # None: not assessed
    source: str
    note: str = ""
    code: str = ""

    def row(self) -> dict:
        return {"check": self.name, "value": self.value, "limit": self.limit,
                "result": "not assessed" if self.ok is None else ("passed" if self.ok else "flagged"),
                "source": self.source, "note": self.note}


SETTINGS = (
    Setting("rest_low_mv", -85.0, "Resting potential at least (mV)", "", "mV", "", advanced=True),
    Setting("rest_high_mv", -55.0, "Resting potential at most (mV)", "Higher values indicate a depolarized cell or an "
            "uncorrected pipette offset.", "mV", ""),
    Setting("max_drift_cc_mv", 5.0, "Drift across sweeps, current clamp (mV)", "", "mV", "",
            advanced=True),
    Setting("max_drift_vc_pa", 50.0, "Drift across sweeps or within a gap-free file, voltage clamp (pA)", "Stricter limits (e.g. 30 pA) are also used.", "pA",
            "Michaud et al. 2024", advanced=True),
    Setting("max_holding_pa", 200.0, "Holding current at most (pA)", "Applies at holding potentials below -50 mV. "
            "The CA3 sIPSC DSI preset uses 500 pA.", "pA", "O'Keeffe et al. 2025; Cording et al. 2025"),
    Setting("max_rs_mohm", 25.0, "Series resistance at most (MOhm)", "", "MOhm",
            "Hartzell 2018; Cording et al. 2025; Chancey 2022"),
    Setting("max_rs_change", 0.20, "Rs change at most (fraction)", "", "fraction", "O'Keeffe 2025; Hartzell 2018",
            advanced=True),
    Setting("max_rs_rm", 0.15, "Rs / Rm at most", "", source=IPFX, advanced=True),
    Setting("max_noise_sd_pa", 200.0, "Baseline noise SD at most, voltage clamp (pA)", "", "pA", ALLEN_SYN, advanced=True),
    Setting("max_monitor_pa", 50.0, "Current monitor at baseline at most (pA)", "Larger values indicate bias current left on.", "pA",
            "", advanced=True),
    Setting("spontaneous_share", 0.25, "Spontaneous spikes before the protocol in at most (share of sweeps)", "",
            "fraction", "", advanced=True),
    Setting("max_threshold_mv", -20.0, "Action potential threshold at most (mV)", "Higher values indicate a voltage offset.", "mV",
            "", advanced=True),
    Setting("rin_low_mohm", 20.0, "Input resistance at least (MOhm)", "Lower values indicate a leaky seal or a cell that "
            "is not whole-cell.", "MOhm", "", advanced=True),
    Setting("rin_high_mohm", 1000.0, "Input resistance at most (MOhm)", "Higher values indicate a cell sealed only in part, "
            "a pipette clogging, or a very small cell (68 of the lab's 633 MD thalamic step families read above 1 GOhm).",
            "MOhm", "",
            advanced=True),
    Setting("max_prestim_noise_mv", 0.5, "Noise before the stimulus at most (mV RMS, 500 ms)", "", "mV", IPFX,
            advanced=True),
    Setting("flag_fast_noise", False, "Flag the 1.5 ms noise (70 uV)", "Off by default: the 70 uV limit assumes very "
            "low-noise acquisition hardware, and typical rigs exceed it several-fold.", source=IPFX, advanced=True),
    Setting("max_sweep_drift_mv", 1.0, "Drift within a sweep at most (mV)", "", "mV", IPFX, advanced=True),
    Setting("failing_share", 0.25, "A per-sweep check flags the recording when it fails in (share of sweeps)", "",
            "fraction", "", advanced=True),
)

OUTPUTS = (
    Output("passed", "Whether every quality check that could run passed; 'not checked' when none could.", "",
           label="Quality", headline=True),
    Output("n_flagged", "How many checks flagged the recording.", "count", label="Checks flagged", headline=True),
    Output("n_assessed", "How many checks could run.", "count", label="Checks run"),
    Output("check", "One quality check: its value, its limit and its reference.", "", level="recording"),
)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - x.mean()) ** 2))) if x.size else NAN


def _mode(y: np.ndarray, bin_size: float = 1.0) -> float:
    """The most populated value of an all-points histogram."""
    y = y[np.isfinite(y)]
    if y.size == 0:
        return NAN
    edges = np.arange(np.percentile(y, 0.5), np.percentile(y, 99.5) + bin_size, bin_size)
    if edges.size < 2:
        return float(np.median(y))
    counts, edges = np.histogram(y, edges)
    i = int(np.argmax(counts))
    return float((edges[i] + edges[i + 1]) / 2)


@register("quality", "Recording quality", version=1, types=WHOLE_CELL, settings=SETTINGS, outputs=OUTPUTS,
          requires=("baseline", "membrane_test", "bridge", "intrinsic"),
          description="Whether the recording meets published quality limits. Each flag reports its value, limit "
                      "and reference; flagged recordings are never removed.",
          source="Gouwens et al. 2019; Campagnola et al. 2022")
def run(recording: Recording, context, s) -> Result:
    out = Result("quality", 1)
    checks: list[Check] = []
    base = context.result("baseline")
    vc = recording.is_voltage_clamp
    if base is None or not base.values:
        checks.append(Check("Baseline", "–", "", None, "", "no baseline stretch could be measured"))
    elif not vc:
        _current_clamp(recording, context, s, base, checks)
    else:
        _voltage_clamp(recording, context, s, base, checks)
    flagged = [c for c in checks if c.ok is False]
    assessed = [c for c in checks if c.ok is not None]
    out.values.update({"passed": (not flagged) if assessed else None, "n_flagged": len(flagged),
                       "n_assessed": len(assessed)})
    out.tables["checks"] = [c.row() for c in checks]
    for c in flagged:
        out.flags.append(Flag(c.code or c.name.lower(), f"{c.name}: {c.value}" + (f" ({c.note})" if c.note else ""),
                              c.limit, c.source))
    for c in checks:
        if c.ok is None:
            out.notes.append(f"{c.name} not assessed: {c.note}")
    if not assessed:
        out.notes.append("no check could run: the recording is 'not checked', never 'passed'")
    return out


def _current_clamp(recording, context, s, base, checks: list[Check]) -> None:
    fs, t = recording.sample_rate_hz, recording.time_s
    level = base.values["level"]
    if np.isfinite(level):
        ok = s["rest_low_mv"] <= level <= s["rest_high_mv"]
        checks.append(Check("Resting potential", f"{level:.1f} mV", f"{s['rest_low_mv']:g} to {s['rest_high_mv']:g} mV",
                            ok, "", "" if ok else ("depolarized, or the pipette offset not zeroed"
                                                   if level > s["rest_high_mv"] else "unusually negative"), "rest"))
    else:                                    # a check with no data on it is not assessed
        checks.append(Check("Resting potential", "–", f"{s['rest_low_mv']:g} to {s['rest_high_mv']:g} mV", None, "",
                            "no baseline level could be measured"))
    intrinsic = context.result("intrinsic")
    rin = intrinsic.values.get("r_input_slope", NAN) if intrinsic is not None else NAN
    if np.isfinite(rin):
        ok = s["rin_low_mohm"] <= rin <= s["rin_high_mohm"]
        checks.append(Check("Input resistance", f"{rin:.0f} MOhm", f"{s['rin_low_mohm']:g} to {s['rin_high_mohm']:g} MOhm",
                            ok, "", "" if ok else ("a cell sealed only in part, a clogging pipette, or a very small cell"
                                                   if rin > s["rin_high_mohm"] else "a leaky seal"), "input resistance"))
    else:
        checks.append(Check("Input resistance", "–", f"{s['rin_low_mohm']:g} to {s['rin_high_mohm']:g} MOhm", None, "",
                            "no current steps in the recording" if intrinsic is None
                            else "too few subthreshold steps to measure it"))
    drift = base.values["drift_range"]
    if len(base.tables.get("sweeps", [])) >= 2 and np.isfinite(drift):
        checks.append(Check("Drift across sweeps", f"{drift:.2f} mV", f"{s['max_drift_cc_mv']:g} mV",
                            drift <= s["max_drift_cc_mv"], "",
                            f"steady drift {base.values['drift_trend']:+.1f} mV (last 3 sweeps minus first 3)", "drift"))
    else:
        checks.append(Check("Drift across sweeps", "–", f"{s['max_drift_cc_mv']:g} mV", None, "",
                            "one baseline level: nothing to compare it with"))
    # spontaneous action potentials before the protocol, and the first spike's threshold
    spont, threshold = 0, NAN
    for k in range(recording.n_sweeps):
        y = recording.sweep(k)
        end = pre_protocol_end_s(recording, context.stimulus, k)
        b0 = float(np.median(y[: max(int(end * fs), 3)]))
        at = spike_onset(t, y, fs, 0.0, end, b0)
        if np.isfinite(at):
            spont += 1
        if not np.isfinite(threshold):
            first = spike_onset(t, y, fs, 0.0, recording.sweep_duration_s, b0)
            if np.isfinite(first):
                threshold = float(y[int(first * fs)])
    share = spont / max(recording.n_sweeps, 1)
    checks.append(Check("Spontaneous spikes before the protocol", f"{spont} of {recording.n_sweeps} sweeps",
                        f"under {s['spontaneous_share']:.0%} of sweeps", share < s["spontaneous_share"],
                        "", "a cell unstable or depolarized at rest" if share >= s["spontaneous_share"]
                        else "", "spontaneous spikes"))
    if np.isfinite(threshold):
        checks.append(Check("First spike's threshold", f"{threshold:.1f} mV", f"at most {s['max_threshold_mv']:g} mV",
                            threshold <= s["max_threshold_mv"], "",
                            "" if threshold <= s["max_threshold_mv"] else "a voltage offset", "threshold"))
    else:
        checks.append(Check("First spike's threshold", "–", f"at most {s['max_threshold_mv']:g} mV", None,
                            "", "no action potential in the recording"))
    from ..core.recording import Role

    monitor = next((c for c in recording.channels_with(Role.MONITOR) if c.units.lower() == "pa"), None)
    if monitor is not None:
        m = recording.read(monitor)
        vals = [float(np.median(m[k, : max(int(pre_protocol_end_s(recording, context.stimulus, k) * fs), 3)]))
                for k in range(recording.n_sweeps)]
        bias = float(np.median(vals))
        checks.append(Check("Current monitor at baseline", f"{bias:.0f} pA", f"within ±{s['max_monitor_pa']:g} pA",
                            abs(bias) <= s["max_monitor_pa"], "",
                            "" if abs(bias) <= s["max_monitor_pa"] else "bias current left on, or the monitor not zeroed",
                            "bias current"))
    else:
        checks.append(Check("Current monitor at baseline", "–", f"±{s['max_monitor_pa']:g} pA", None, "",
                            "no current monitor was recorded"))
    # noise: 1.5 ms (reported) and the 500 ms before the stimulus (per sweep, on the share that fail)
    fast, slow, slow_fail, sweep_drift, drift_fail, quiet = [], [], 0, [], 0, 0
    for k in range(recording.n_sweeps):
        y = recording.sweep(k)
        end = pre_protocol_end_s(recording, context.stimulus, k)
        i1 = int(end * fs)
        # IPFX's 500 ms before the stimulus; with less than that, what there is (fewer samples, no bias)
        i0 = max(0, i1 - int(0.5 * fs))
        if i1 - i0 < 2:
            continue
        pre = y[i0:i1]
        fast.append(_rms(pre[: max(2, int(0.0015 * fs))]))           # IPFX's 1.5 ms noise epoch
        r = _rms(pre)
        slow.append(r)
        slow_fail += r > s["max_prestim_noise_mv"]
        # drift within a sweep: that window against the last 500 ms, only when the protocol is
        # over 500 ms (plus 100 ms to settle) before the sweep ends
        last_event = _protocol_end_s(recording, context.stimulus, k)
        if recording.sweep_duration_s - last_event >= 0.6:
            quiet += 1
            d = abs(float(np.mean(y[-int(0.5 * fs):])) - float(np.mean(pre)))
            sweep_drift.append(d)
            drift_fail += d > s["max_sweep_drift_mv"]
    if fast:
        v = float(np.median(fast)) * 1e3
        checks.append(Check("Noise, 1.5 ms RMS", f"{v:.0f} µV median", "70 µV",
                            (v <= 70) if s["flag_fast_noise"] else True, IPFX,
                            "" if s["flag_fast_noise"] else "reported, not flagged (Analyze: flag the 1.5 ms noise)",
                            "noise"))
    if slow:
        share = slow_fail / len(slow)
        checks.append(Check("Noise before the stimulus (500 ms RMS)",
                            f"{np.median(slow):.2f} mV median, {slow_fail}/{len(slow)} sweeps over {s['max_prestim_noise_mv']:g}",
                            f"{s['max_prestim_noise_mv']:g} mV in at most {s['failing_share']:.0%} of sweeps",
                            share <= s["failing_share"], IPFX, "", "unsettled"))
    else:
        checks.append(Check("Noise before the stimulus (500 ms RMS)", "–", f"{s['max_prestim_noise_mv']:g} mV",
                            None, IPFX, "no sweep has a baseline before its protocol"))
    if sweep_drift:
        share = drift_fail / len(sweep_drift)
        checks.append(Check("Drift within a sweep", f"{np.median(sweep_drift):.2f} mV median, {drift_fail}/{quiet} sweeps "
                            f"over {s['max_sweep_drift_mv']:g}", f"{s['max_sweep_drift_mv']:g} mV in under "
                            f"{s['failing_share']:.0%} of sweeps or fewer", share <= s["failing_share"], IPFX, "", "sweep drift"))
    else:
        checks.append(Check("Drift within a sweep", "–", f"{s['max_sweep_drift_mv']:g} mV", None, IPFX,
                            "no sweep has a quiet 500 ms after its protocol"))
    bridge = context.result("bridge")
    if bridge is not None and bridge.qc_flags:
        checks.append(Check("Bridge balance", f"{bridge.values['jump']:.1f} mV instant jump",
                            "2 mV", False, "the Axon Guide", bridge.qc_flags[0].text, "bridge"))


def _protocol_end_s(recording, stimulus, k: int) -> float:
    """When the last thing the protocol does in sweep ``k`` is over: the last
    command change, or the last stimulus plus 100 ms."""
    end = 0.0
    cmd = recording.command
    if cmd is not None:
        c = cmd[k]
        moved = np.flatnonzero(np.abs(np.diff(c)) > 0.5)
        if moved.size:
            end = (moved[-1] + 1) / recording.sample_rate_hz
    if stimulus is not None and k < len(stimulus.times_s) and stimulus.times_s[k].size:
        end = max(end, float(stimulus.times_s[k][-1]) + 0.1)
    return end


def _voltage_clamp(recording, context, s, base, checks: list[Check]) -> None:
    fs = recording.sample_rate_hz
    holding_v = recording.holding
    level = base.values["level"]
    mt = context.result("membrane_test")
    if mt is not None and np.isfinite(mt.values.get("holding", NAN)):
        level = mt.values["holding"]
    if np.isfinite(level):
        if np.isfinite(holding_v) and holding_v > -50:
            ok = level > -s["max_holding_pa"]                    # only a large inward current is judged
            note = (f"held at {holding_v:g} mV, where an outward holding current is expected; only an inward one is "
                    "judged" if ok else f"{level:.0f} pA inward at {holding_v:g} mV cannot flow through a leak there")
        else:
            ok = abs(level) <= s["max_holding_pa"]
            note = "" if ok else "a leaky seal, or a dying or lost cell"
        checks.append(Check("Holding current", f"{level:.0f} pA" + (f" at {holding_v:g} mV" if np.isfinite(holding_v)
                                                                     else ""),
                            f"{s['max_holding_pa']:g} pA", ok, "O'Keeffe 2025; Cording 2025; Hartzell 2018", note,
                            "holding"))
    else:                                    # a check with no data on it is not assessed
        checks.append(Check("Holding current", "–", f"{s['max_holding_pa']:g} pA", None,
                            "O'Keeffe 2025; Cording 2025; Hartzell 2018", "no baseline level could be measured"))
    # the holding's change: across sweeps, or within a gap-free file in 10 s pieces (mode of each)
    if recording.is_gap_free:
        y = recording.sweep(0)
        piece = int(10 * fs)
        modes = [_mode(y[a:a + piece]) for a in range(0, y.size - piece + 1, piece)]
        change = (max(modes) - min(modes)) if len(modes) >= 2 else NAN
        what = f"{change:.0f} pA over {len(modes)} pieces of 10 s (each the mode of its all-points histogram)"
        why_not = ("shorter than two pieces of 10 s: nothing to compare" if len(modes) < 2
                   else "no level could be read in its pieces")
    else:
        change = base.values["drift_range"] if len(base.tables.get("sweeps", [])) >= 2 else NAN
        what = f"{change:.0f} pA across sweeps"
        why_not = "one sweep: nothing to compare it with"
    if np.isfinite(change):
        checks.append(Check("Holding current change", what, f"{s['max_drift_vc_pa']:g} pA",
                            change <= s["max_drift_vc_pa"], "", "",
                            "holding change"))
    else:
        checks.append(Check("Holding current change", "–", f"{s['max_drift_vc_pa']:g} pA", None, "", why_not))
    noise = base.values.get("noise_sd", NAN)
    checks.append(Check("Baseline noise SD", f"{noise:.1f} pA", f"{s['max_noise_sd_pa']:g} pA",
                        noise <= s["max_noise_sd_pa"] if np.isfinite(noise) else None, ALLEN_SYN, "", "noise"))
    rs = rs_first = rs_last = rm = NAN
    where = ""
    if mt is not None and np.isfinite(mt.values.get("rs", NAN)):
        rs, rm, where = mt.values["rs"], mt.values.get("rm", NAN), "the file's test step"
        rs_first, rs_last = mt.values.get("rs_first", NAN), mt.values.get("rs_last", NAN)
    else:
        typed = recording.metadata.rs_typed_mohm
        if np.isfinite(typed[0]):
            rs_first, rs_last = typed[0], typed[1] if np.isfinite(typed[1]) else typed[0]
            rs, where = max(rs_first, rs_last), recording.metadata.rs_words("typed from Clampex's Membrane Test")
    if not np.isfinite(rs):
        checks.append(Check("Series resistance", "–", f"{s['max_rs_mohm']:g} MOhm", None,
                            "Hartzell 2018; Cording 2025", "no test step in the file and none typed (Files: Rs typed, "
                            "start and end)"))
        return
    checks.append(Check("Series resistance", f"{rs:.1f} MOhm ({where})", f"{s['max_rs_mohm']:g} MOhm",
                        rs <= s["max_rs_mohm"], "Hartzell 2018; Cording 2025; Chancey 2022", "", "Rs high"))
    if np.isfinite(rs_first) and np.isfinite(rs_last) and rs_first > 0:
        ch = rs_last / rs_first - 1
        checks.append(Check("Series resistance change", f"{ch:+.0%} ({rs_first:.1f} to {rs_last:.1f} MOhm)",
                            f"{s['max_rs_change']:.0%}", abs(ch) <= s["max_rs_change"], "O'Keeffe 2025; Hartzell 2018",
                            "", "Rs change"))
    if np.isfinite(rm) and rm > 0:
        checks.append(Check("Rs / Rm", f"{rs / rm:.3f}", f"{s['max_rs_rm']:g}", rs / rm <= s["max_rs_rm"], IPFX,
                            "", "Rs/Rm"))
