"""The cell's health sweep by sweep, in current clamp, in words (A5; the lab's photostim protocols, 2026-11-10).

A file's quality report (``quality``) judges the recording as a whole. A cell that is failing through a recording
(a cell resting far above -55 mV) needs to be called unhealthy sweep by sweep, so that no sweep of it yields
a confident number. Each sweep gives:

- **Vm before the protocol**: the median of the 20 ms before the first stimulus or command change, judged
  against the resting-potential limits (-85 to -55 mV; the lab's inclusion criterion is -60 mV or more negative:
  Ahmed, Headley & Pare 2021).
- **Input resistance from the test pulse** inside the sweep (the lab's photostim protocols put a -20 pA step of
  100 ms at 766 ms in every sweep; Gungor et al. 2014 used -10 pA; Royer et al. 2000, 10 pA): the voltage it
  settles to (the mean of its last fifth) against the 20 ms before it, over its current. The pulse is a command
  step of at most ``max_pulse_pa`` that is the same in every sweep. Without one, the sweep's own step (a step
  family with the light inside it) gives the resistance from its steady state (the last 100 ms), on sweeps that
  do not fire. Each sweep's resistance is judged against the first sweeps' median: the lab excludes a change of
  more than 10% over a long protocol (Popescu et al. 2007, 2010).
- **Spikes before the protocol**: the cell firing on its own.

A sweep is unhealthy when any check fails; the file is flagged when a quarter of its sweeps are, and every sweep
says in words what is wrong with it.
"""

from __future__ import annotations

import numpy as np

from ..core.recording import Recording
from .baseline import pre_protocol_end_s
from .registry import Output, Setting, register
from .result import Flag, Result
from .tools import ledger

NAN = float("nan")
CC_TYPES = {"cc_evoked", "cc_steps", "cc_short_square", "cc_ramp"}

SETTINGS = (
    Setting("rest_low_mv", -85.0, "Vm before the protocol at least (mV)", "", "mV", advanced=True),
    Setting("rest_high_mv", -55.0, "Vm before the protocol at most (mV)", "The lab's inclusion criterion is -60 mV "
            "(Ahmed, Headley & Pare 2021); -55 mV is the quality report's limit.", "mV",
            "Ahmed, Headley & Pare 2021"),
    Setting("max_pulse_pa", 50.0, "Test pulse at most (pA)", "A command step this small, the same in every sweep, "
            "is the test pulse the input resistance is read from.", "pA", advanced=True),
    Setting("max_rin_change", 0.10, "Rin change at most (fraction of the first sweeps' median)", "",
            "fraction", "Popescu et al. 2007, 2010", advanced=True),
    Setting("first_sweeps", 3, "Rin reference: the first (n) sweeps' median", "", "", advanced=True, minimum=1),
    Setting("failing_share", 0.25, "The file is flagged when this share of sweeps is unhealthy", "", "fraction",
            advanced=True),
)

OUTPUTS = (
    Output("healthy", "yes when fewer than the failing share of sweeps are unhealthy; no otherwise; 'not checked' "
           "when no check could run.", "", label="Cell healthy through the recording", headline=True),
    Output("unhealthy_sweeps", "Sweeps that fail a check.", "count", label="Unhealthy sweeps", headline=True),
    Output("n_sweeps", "Sweeps judged.", "count", label="Sweeps judged"),
    Output("v_before_median", "The median over sweeps of Vm before the protocol.", "mV", label="Vm before (median)"),
    Output("v_before_range", "How far Vm before the protocol moved across sweeps (max minus min).", "mV",
           label="Vm before (range)"),
    Output("rin_median", "The median over sweeps of the per-sweep input resistance.", "MOhm",
           label="Input resistance (median)", headline=True),
    Output("rin_change", "The last sweeps' median input resistance over the first sweeps', minus 1.", "fraction",
           "Popescu et al. 2007, 2010", label="Rin change over the recording"),
    Output("rin_from", "What the per-sweep input resistance was read from.", "", label="Rin read from"),
    Output("words", "What is wrong, in words: the checks that fail, and in how many sweeps.", "",
           label="In words", headline=True),
    Output("rin_mohm", "One sweep's input resistance.", "MOhm", level="sweep"),
)


def _test_pulse(recording: Recording, max_pa: float) -> tuple[int, int, float] | None:
    """(onset index, end index, level in pA) of a command step of at most ``max_pa`` that is the same in every
    sweep (its edges and level), 20 ms to 1 s long: the test pulse. None without one."""
    cmd = recording.command
    if cmd is None or recording.n_sweeps == 0:
        return None
    fs = recording.sample_rate_hz
    found = None
    for k in range(recording.n_sweeps):
        c = cmd[k]
        d = np.flatnonzero(np.abs(np.diff(c)) > 0.25)
        edges = np.concatenate([[0], d + 1, [c.size]])
        pulses = []
        for a, b in zip(edges[:-1], edges[1:]):
            if a == 0 or b == c.size or not (0.02 * fs <= b - a <= 1.0 * fs):
                continue
            before = float(np.median(c[max(0, a - int(0.002 * fs)):a]))
            level = float(np.median(c[a:b])) - before
            after = float(np.median(c[b:min(c.size, b + int(0.002 * fs))])) - before
            if 1.0 <= abs(level) <= max_pa and abs(after) < 0.5:
                pulses.append((int(a), int(b), level))
        if k == 0:
            found = pulses
        else:
            found = [p for p in found if any(abs(p[0] - q[0]) <= 2 and abs(p[1] - q[1]) <= 2
                                             and abs(p[2] - q[2]) < 0.5 for q in pulses)]
        if not found:
            return None
    return found[0] if found else None


def _own_step(recording: Recording, k: int) -> tuple[int, int, float] | None:
    """(onset, end, level) of the sweep's own largest command step, 20 ms or longer."""
    cmd = recording.command
    if cmd is None:
        return None
    c = cmd[k]
    fs = recording.sample_rate_hz
    hold = float(np.median(c[: max(2, c.size // 64)]))
    away = np.abs(c - hold) > 0.5
    if not away.any():
        return None
    edges = np.flatnonzero(np.diff(np.r_[0, away.astype(np.int8), 0]))
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 0.02 * fs]
    if not runs:
        return None
    a, b = max(runs, key=lambda r: abs(float(np.median(c[r[0]:r[1]])) - hold))
    return int(a), int(b), float(np.median(c[a:b])) - hold


@register("sweep_health", "Health, sweep by sweep (current clamp)", version=1, types=CC_TYPES, settings=SETTINGS,
          outputs=OUTPUTS, requires=("baseline",),
          description="Each sweep's membrane potential before the protocol, its own input resistance from the "
                      "test pulse (or its step) and whether it fired on its own, each judged and said in words, "
                      "so a cell failing through a recording is called unhealthy sweep by sweep.",
          source="Ahmed, Headley & Pare 2021 (rest); Popescu et al. 2007, 2010 (Rin stability); Gungor et al. "
                 "2014 (the test pulse)")
def run(recording: Recording, context, s) -> Result | None:
    from .evoked import spike_onset

    if recording.is_voltage_clamp:
        return None
    out = Result("sweep_health", 1)
    fs, t = recording.sample_rate_hz, recording.time_s
    st = context.stimulus
    pulse = _test_pulse(recording, s["max_pulse_pa"])
    rows = []
    for k in range(recording.n_sweeps):
        y = np.asarray(recording.sweep(k), float)
        end = pre_protocol_end_s(recording, st, k)
        b0 = max(0, int(end * fs) - int(0.02 * fs))
        b1 = max(b0 + 2, int(end * fs))
        v_before = float(np.median(y[b0:b1]))
        spikes = int(np.isfinite(spike_onset(t, y, fs, 0.0, end, v_before)))
        rin, rin_from = NAN, ""
        step = pulse if pulse is not None else _own_step(recording, k)
        if step is not None:
            a, b, level = step
            tail = max(3, (b - a) // 5) if pulse is not None else min(int(0.1 * fs), max(3, (b - a) // 4))
            before = float(np.mean(y[max(0, a - int(0.02 * fs)):a]))
            fired = np.isfinite(spike_onset(t, y, fs, a / fs, b / fs, before))
            if abs(level) >= 1.0 and not fired:
                dv = float(np.mean(y[b - tail:b])) - before
                rin = dv / level * 1e3
                rin_from = (f"the {level:+.0f} pA test pulse at {a / fs * 1e3:.0f} ms" if pulse is not None
                            else f"the sweep's own {level:+.0f} pA step (steady state)")
            elif fired:
                rin_from = "not read: the cell fired during the step"
        rows.append({"sweep": k, "time_s": float(recording.sweep_start_times_s[k]) if k < recording.sweep_start_times_s.size else NAN,
                     "v_before": v_before, "rin_mohm": rin, "rin_from": rin_from, "rin_change": NAN,
                     "spikes_before": spikes, "healthy": True, "words": ""})
    if not rows:
        return out
    rins = np.array([r["rin_mohm"] for r in rows])
    n_ref = int(s["first_sweeps"])
    first = rins[np.isfinite(rins)][:n_ref]
    ref = float(np.median(first)) if first.size else NAN
    counts = {"depolarized": 0, "very negative": 0, "rin moved": 0, "fired before the protocol": 0}
    for r in rows:
        words = []
        if r["v_before"] > s["rest_high_mv"]:
            words.append(f"depolarized: {r['v_before']:.0f} mV before the protocol, above {s['rest_high_mv']:g} mV "
                         "(a failing cell, or the pipette offset not zeroed)")
            counts["depolarized"] += 1
        elif r["v_before"] < s["rest_low_mv"]:
            words.append(f"very negative: {r['v_before']:.0f} mV before the protocol, below {s['rest_low_mv']:g} mV "
                         "(an amplifier offset?)")
            counts["very negative"] += 1
        if np.isfinite(r["rin_mohm"]) and np.isfinite(ref) and ref > 0:
            r["rin_change"] = r["rin_mohm"] / ref - 1
            if abs(r["rin_change"]) > s["max_rin_change"]:
                words.append(f"input resistance moved {r['rin_change']:+.0%} from the first sweeps' "
                             f"{ref:.0f} MOhm ({r['rin_mohm']:.0f} MOhm)")
                counts["rin moved"] += 1
        if r["spikes_before"]:
            words.append("fired on its own before the protocol")
            counts["fired before the protocol"] += 1
        r["healthy"] = not words
        r["words"] = "; ".join(words)
    out.tables["sweeps"] = rows
    bad = [r for r in rows if not r["healthy"]]
    v = np.array([r["v_before"] for r in rows])
    good_r = rins[np.isfinite(rins)]
    last = good_r[-n_ref:] if good_r.size else np.array([])
    out.values.update({
        "n_sweeps": len(rows), "unhealthy_sweeps": len(bad),
        "v_before_median": float(np.median(v)), "v_before_range": float(np.ptp(v)) if v.size > 1 else 0.0,
        "rin_median": float(np.median(good_r)) if good_r.size else NAN,
        "rin_change": (float(np.median(last)) / ref - 1) if good_r.size >= 2 * n_ref and np.isfinite(ref) and ref > 0 else NAN,
        "rin_from": next((r["rin_from"] for r in rows if np.isfinite(r["rin_mohm"])), "no test pulse and no step"),
    })
    said = [f"{word} in {n} of {len(rows)} sweep(s)" for word, n in counts.items() if n]
    healthy = len(bad) / len(rows) < s["failing_share"]
    out.values["healthy"] = "yes" if healthy else "no"
    out.values["words"] = ("; ".join(said) if said else "every sweep passes: Vm before the protocol "
                           f"{np.median(v):.0f} mV" + (f", input resistance {np.median(good_r):.0f} MOhm"
                                                       if good_r.size else ""))
    if not healthy:
        out.flags.append(Flag("unhealthy", f"the cell is unhealthy in {len(bad)} of {len(rows)} sweep(s): "
                              + "; ".join(said) + ". Numbers from these sweeps are not a healthy cell's",
                              f"{s['failing_share']:.0%} of sweeps", "Ahmed, Headley & Pare 2021; Popescu et al. 2007"))
    if pulse is not None:
        out.choose("input resistance", f"the {pulse[2]:+.0f} pA test pulse at {pulse[0] / fs * 1e3:.0f} ms in every "
                   "sweep", "the voltage it settles to (its last fifth) against the 20 ms before, over its current")
    elif good_r.size:
        out.choose("input resistance", "each sweep's own step", "no test pulse in the protocol: the step's steady "
                   "state (its last 100 ms) against the baseline, over its current, on sweeps that did not fire")
    ledger.note("Vm before the protocol, the test pulse's resistance and spontaneous spikes judged sweep by sweep",
                "health sweep by sweep")
    return out
