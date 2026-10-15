"""Where the stimulus is, in every sweep, and how that is known (B2).

The order of evidence, from the brief and the recorded decisions: a typed
override first (the user's word), then the file's own record (WinLTP's
stimulation settings), then a recorded trigger or light channel, then a
commanded digital output, then a second output (a light driver's analog
command), then the command itself (a step family is not a stimulus; a
brief command pulse is). Which one was used, and why the others were not,
is written into the result.

Not every toggling channel is a stimulus (Analysis Decisions, 2026-09-22): a
channel is rejected when it is a monitor (pA, nA, mV), is held high longer
than the longest stimulus pulse (a gate), toggles faster than 500 Hz or more
than 200 times in a sweep (a frame-sync line), or has edges that line up with
the command step. A channel known to carry light is exempt from the last two
(a photocurrent command is hundreds of ms, and light is often delivered
during a step). A channel carries light when its name says so, when it is one
of the inputs the lab's rig records the LED trigger on (IN 8, IN 14), when the file's rig's profile names it among its light inputs
(AM_Stim on a Polygon rig), or when the protocol's name does; a light pulse that climbs
slowly is a ramp and is read from its foot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..core.recording import Channel, Epoch, Recording, Role, Stimulus

MAX_PULSE_S = 0.050          # a TTL held longer than this is a gate, not a stimulus
MAX_TOGGLES = 200            # more edges than this in a sweep: a clock or frame sync
MAX_RATE_HZ = 500.0
TTL_MIN_V = 0.1              # a recorded line that never rises this far has nothing connected (a TTL read through
#                              a 10x attenuator is 0.45 V: an LED trigger recorded on IN 8)
RAMP_MIN_RISE_S = 0.020      # a light pulse taking longer than this from 10% to 90% is a ramp, read from its foot
COMMAND_PULSE_MAX_S = 0.005  # a command deviation this brief is a stimulus (a 10 ms+ test pulse or step is not)
LIGHT_WORDS = ("led", "light", "laser", "polygon", "dmd", "opto")
ELECTRICAL_WORDS = ("stim", "isolator", "am_stim", "amstim", "shock")
LAB_LIGHT_INPUTS = ("in 8", "in 14")   # the lab's rig records the LED driver's trigger on one of these


@dataclass
class Candidate:
    channel: Channel
    accepted: bool
    reason: str
    times_s: list[np.ndarray]
    widths_s: list[np.ndarray]
    kind: str = "unknown"              # "light", "electrical" or "unknown"
    kind_reason: str = ""              # how the kind was decided when the channel's name does not say


def _edges(trace: np.ndarray, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    """Indices where ``trace`` rises above ``high`` and later falls below ``low``:
    a Schmitt trigger, so noise on a recorded TTL line does not count as edges
    (a stimulator output recorded through an input read 95,575 toggles in one
    sweep when judged against a single threshold)."""
    above = trace > high
    below = trace < low
    # state changes only where the trace crosses the far threshold: of each run of
    # samples on the same side, the first one is the edge, and the state starts low
    changes = np.flatnonzero(above | below)
    kind = above[changes]
    keep = np.r_[True, kind[1:] != kind[:-1]] if kind.size else np.zeros(0, dtype=bool)
    idx, kind = changes[keep], kind[keep]
    if kind.size and not kind[0]:
        idx, kind = idx[1:], kind[1:]
    return idx[kind].astype(int), idx[~kind].astype(int)


def pulses_from_trace(trace: np.ndarray, fs: float, low: float = 0.4, high: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """(onsets_s, widths_s) of the pulses in one sweep of a TTL-like trace: a
    rise above 60% of the way from its low level to its high level, a fall
    below 40%, so a 3.3 V and a 5 V line read the same and noise does not toggle
    (``low``/``high`` move the two thresholds: a ramp is read from its foot)."""
    lo, hi = float(np.min(trace)), float(np.max(trace))
    if hi - lo <= 0:
        return np.array([]), np.array([])
    rises, falls = _edges(trace, lo + low * (hi - lo), lo + high * (hi - lo))
    if rises.size == 0:
        return np.array([]), np.array([])
    widths = []
    for r in rises:
        f = falls[falls > r]
        widths.append((f[0] - r) / fs if f.size else (trace.size - r) / fs)
    return rises / fs, np.asarray(widths)


def _named_light(channel: Channel) -> bool:
    return channel.role is Role.LIGHT or any(w in channel.name.lower() for w in LIGHT_WORDS)


def _kind(channel: Channel, protocol: str = "", light_inputs: dict[str, str] | None = None) -> tuple[str, str]:
    """("light" | "electrical" | "unknown", why) from the channel's name, else the
    lab's rule for its inputs, else the protocol's name ("blue_LED_pulse_train_8Hz_NA":
    the NA rig records the LED on IN 8, Analysis Decisions on the lab's rigs). The
    file's rig's own light inputs (rigs.Rig.light_inputs) come before the name's words:
    a rig's wiring is a fact its line names cannot overrule."""
    name = channel.name.lower().strip()
    if _named_light(channel):
        return "light", ""
    why = next((w for n, w in (light_inputs or {}).items() if n.lower().strip() == name), None)
    if why is not None:
        return "light", f"light: on this rig {channel.name} carries light ({why})"
    if any(w in name for w in ELECTRICAL_WORDS):
        return "electrical", ""
    if channel.recorded and name in LAB_LIGHT_INPUTS:
        return "light", f"light: {channel.name} is the input the lab's rig records the LED trigger on"
    proto = (protocol or "").lower()
    if any(w in proto for w in LIGHT_WORDS):
        return "light", f"light: the protocol's name '{protocol}' says so (the channel's does not)"
    if any(w in proto for w in ELECTRICAL_WORDS):
        return "electrical", f"electrical: the protocol's name '{protocol}' says so (the channel's does not)"
    return "unknown", ""


def _rise_s(trace: np.ndarray, fs: float) -> float:
    """How long the first pulse of one sweep takes from 10% to 90% of its height
    (0 for a square TTL, the ramp's duration for a ramp of light)."""
    lo, hi = float(np.min(trace)), float(np.max(trace))
    r10, _ = _edges(trace, lo + 0.05 * (hi - lo), lo + 0.1 * (hi - lo))
    r90, _ = _edges(trace, lo + 0.85 * (hi - lo), lo + 0.9 * (hi - lo))
    return float(r90[0] - r10[0]) / fs if r10.size and r90.size and r90[0] >= r10[0] else 0.0


def _ramp_pulses(trace: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """(onsets_s, widths_s) of the ramps in one sweep: each one's foot, extrapolated
    from where it crosses 2% and 10% of its height back to the baseline (noise on
    the line is far below 1%), to where it falls back."""
    on2, w2 = pulses_from_trace(trace, fs, 0.01, 0.02)
    on10, _ = pulses_from_trace(trace, fs, 0.05, 0.1)
    if on2.size == 0 or on10.size == 0:
        return on2, w2
    feet = np.array([t2 - (on10[np.searchsorted(on10, t2)] - t2) / 4.0 if np.searchsorted(on10, t2) < on10.size else t2
                     for t2 in on2])
    feet = np.maximum(feet, 0.0)
    return feet, w2 + (on2 - feet)


def _step_edges(epochs: list[Epoch], fs: float, holding: float = float("nan")) -> np.ndarray:
    """Where the command changes level: an epoch boundary counts only when the
    level on either side differs (Clampex lists epochs at the holding level, and a
    trigger timed to one of those boundaries is not the command, as in
    Baseline_DSI files)."""
    edges = []
    prev = holding
    for e in epochs:
        if not np.isfinite(e.level):
            continue
        if not np.isfinite(prev) or abs(e.level - prev) > 0.5:
            if np.isfinite(prev):
                edges.append(e.start_s)
        prev = e.level
    if epochs and np.isfinite(holding) and np.isfinite(prev) and abs(prev - holding) > 0.5:
        edges.append(epochs[-1].end_s)
    return np.array(edges)


def _judge(channel: Channel, recording: Recording, max_pulse_s: float,
           light_inputs: dict[str, str] | None = None) -> Candidate:
    trace = recording.read(channel)
    fs = recording.sample_rate_hz
    units = channel.units.strip().lower()
    kind, kind_reason = _kind(channel, recording.protocol, light_inputs)
    light = kind == "light"
    if channel.recorded and units in ("pa", "na", "mv") and channel.role is not Role.LIGHT:
        return Candidate(channel, False, f"in {channel.units}: a monitor, not a TTL", [], [], kind, kind_reason)
    if channel.recorded and units == "v" and float(np.ptp(trace)) < TTL_MIN_V:
        return Candidate(channel, False, f"never rises above {float(np.ptp(trace)):.3f} V (a line carrying a trigger "
                                         f"rises {TTL_MIN_V:g} V or more): nothing is connected to it", [], [], kind, kind_reason)
    times, widths = [], []
    ramp = light and float(np.median([_rise_s(trace[s], fs) for s in range(trace.shape[0])])) > RAMP_MIN_RISE_S
    for s in range(trace.shape[0]):
        t, w = _ramp_pulses(trace[s], fs) if ramp else pulses_from_trace(trace[s], fs)
        times.append(t)
        widths.append(w)
    counts = [t.size for t in times]
    if not any(counts):
        return Candidate(channel, False, "never toggles", times, widths, kind, kind_reason)
    longest = max((float(w.max()) for w in widths if w.size), default=0.0)
    if not light and longest > max_pulse_s:
        return Candidate(channel, False, f"held high for {longest * 1e3:.0f} ms, longer than the longest stimulus "
                                         f"pulse allowed ({max_pulse_s * 1e3:.0f} ms): a gate, not a stimulus", times, widths,
                         kind, kind_reason)
    if max(counts) > MAX_TOGGLES:
        return Candidate(channel, False, f"toggles {max(counts)} times in a sweep: a clock or frame-sync line", times, widths,
                         kind, kind_reason)
    rates = [1.0 / np.min(np.diff(t)) for t in times if t.size > 1]
    if rates and max(rates) > MAX_RATE_HZ:
        return Candidate(channel, False, f"toggles at {max(rates):.0f} Hz: a clock or frame-sync line", times, widths,
                         kind, kind_reason)
    if channel.recorded and not _named_light(channel) and counts and all(c == 1 for c in counts if c):
        # Clampex holds the first 1/64 of each sweep; a recorded input that pulses
        # once per sweep exactly where that ends marks the sweep's start (an imaging
        # or acquisition trigger: 18815022's IN 14, protocol Blue_LED_ImageSeq)
        mark = (recording.n_samples // 64) / fs
        if all(abs(float(t[0]) - mark) <= 3.0 / fs for t in times if t.size):
            return Candidate(channel, False, f"pulses once per sweep at {mark * 1e3:.1f} ms, where Clampex's held first "
                                             "1/64 of the sweep ends: it marks the sweep's start (an imaging or "
                                             "acquisition trigger), not a stimulus", times, widths, kind, kind_reason)
    if not light and recording.epochs:
        step = _step_edges(recording.epochs[0], fs, recording.holding)
        onsets = times[0]
        if step.size and onsets.size and all(np.min(np.abs(step - t)) < 2.0 / fs for t in onsets):
            return Candidate(channel, False, "its edges line up with the command step: the command, not a stimulus",
                             times, widths, kind, kind_reason)
    if ramp:
        return Candidate(channel, True, f"a ramp of light rising over {longest * 1e3:.0f} ms to {float(np.max(trace)):.2f} V, "
                                        "taken from its foot", times, widths, kind, kind_reason)
    if light and longest > max_pulse_s:
        return Candidate(channel, True, f"pulses found, the longest {longest * 1e3:.0f} ms (light may be held on)", times,
                         widths, kind, kind_reason)
    return Candidate(channel, True, "pulses found", times, widths, kind, kind_reason)


def parse_override(text: str) -> tuple[np.ndarray, np.ndarray] | None:
    """``"320, 425 @ 5"``: onsets in ms and a width in ms; ``"none"`` or blank: nothing."""
    text = (text or "").strip().lower()
    if not text or text == "none" or STEPS.match(text):
        return None
    width_ms = 1.0
    if "@" in text:
        text, w = text.split("@", 1)
        width_ms = float(w)
    onsets = np.array([float(x) for x in text.replace(";", ",").split(",") if x.strip()]) / 1e3
    return onsets, np.full(onsets.size, width_ms / 1e3)


STEPS = re.compile(r"^steps?\s+(?P<first>[-+]?[\d.]+)\s*:\s*(?P<inc>[-+]?[\d.]+)(?:\s*:\s*(?P<last>[-+]?[\d.]+))?"
                   r"\s*(?:pa)?\s*(?:@\s*(?P<a>[\d.]+)\s*-\s*(?P<b>[\d.]+)\s*(?:ms)?)?$")


def typed_steps(recording: Recording, text: str) -> Recording | None:
    """``"steps -200:50:500 @ 500-1500"``: a current-step family typed for a file
    that records no command (an ATF of the membrane potential alone): first,
    increment and last level in pA, one level per sweep, the step from ``a`` to
    ``b`` ms. Without times, the step's edges are read from the sweep with the
    most negative level: its steepest fall (the onset) and its steepest rise
    after it (the end), on a 1 ms running mean. The typed family becomes the
    recording's command; ``None`` when ``text`` is not a step family."""
    import copy

    from .tools.filters import moving_average

    m = STEPS.match((text or "").strip().lower())
    if m is None:
        return None
    n, fs = recording.n_sweeps, recording.sample_rate_hz
    first, inc = float(m["first"]), float(m["inc"])
    levels = first + inc * np.arange(n)
    notes = [f"current steps typed in: {first:g} pA by {inc:g} pA over the {n} sweeps (the file records no command)"]
    if m["last"] is not None and not np.isclose(levels[-1], float(m["last"])):
        notes.append(f"the typed last level, {float(m['last']):g} pA, is not where {n} sweeps end ({levels[-1]:g} pA): "
                     "the levels follow the sweeps")
    if m["a"] is not None:
        a, b = float(m["a"]) / 1e3, float(m["b"]) / 1e3
        notes.append(f"the step from {1e3 * a:g} to {1e3 * b:g} ms, as typed")
    else:
        k = int(np.argmin(levels))
        if levels[k] >= 0:
            raise ValueError("type the step's times (@ start-end ms): with no negative step they are not read from "
                             "the response")
        w = max(1, int(round(1e-3 * fs)))
        dv = np.diff(moving_average(recording.sweep(k), w))[w:-w]      # the mean's padded ends left out
        on = int(np.argmin(dv[:-1]))
        off = w + on + 1 + int(np.argmax(dv[on + 1:]))
        on += w
        a, b = on / fs, off / fs
        notes.append(f"the step from {1e3 * a:.1f} to {1e3 * b:.1f} ms, read from the response of sweep {k + 1} "
                     f"({levels[k]:g} pA): its steepest fall and steepest rise")
    t = recording.time_s
    inside = (t >= a) & (t < b)
    command = np.where(inside[None, :], levels[:, None], 0.0)
    out = copy.copy(recording)
    channels = [c for c in recording.channels if c.role is not Role.COMMAND]
    index = recording.samples.n_channels
    channels.append(Channel(index, "typed steps", "pA", Role.COMMAND, False,
                            role_reason="current steps typed in by the user (the file records no command)"))
    out.channels = channels
    out.samples = _WithChannel(recording.samples, command)
    out.epochs = [[Epoch("hold", 0.0, a, 0.0), Epoch("step", a, b, float(lv)), Epoch("hold", b, t[-1], 0.0)]
                  for lv in levels]
    out.holding = 0.0
    out.notes = list(recording.notes) + notes
    return out


class _WithChannel:
    """Another source's channels and one more, in memory (a typed command)."""

    def __init__(self, source, extra: np.ndarray) -> None:
        self._source, self._extra = source, extra
        self.n_channels, self.n_sweeps, self.n_samples = source.n_channels + 1, source.n_sweeps, source.n_samples

    def read(self, channel: int, sweeps=None) -> np.ndarray:
        if channel < self._source.n_channels:
            return self._source.read(channel, sweeps)
        return self._extra if sweeps is None else self._extra[sweeps]


def find_stimulus(recording: Recording, *, override: str = "", max_pulse_s: float = MAX_PULSE_S,
                  stimulus_channel: str = "", artifacts: list | None = None, rig=None,
                  light_inputs: dict[str, str] | None = None) -> Stimulus:
    """The stimulus of ``recording`` and how it was found; ``reason`` lists every
    channel considered and why it was or was not taken. ``rig``: what the rig's
    programs recorded (rig.evidence.Evidence): a line of unknown kind that fired
    while the patterned light waited on its external trigger is light; one whose
    name says electrical keeps it, and the light's session is said beside it.
    ``light_inputs``: the inputs that carry light on the file's rig (its profile),
    each with why."""
    n = recording.n_sweeps
    typed = parse_override(override)
    if typed is not None:
        onsets, widths = typed
        return Stimulus([onsets] * n, [widths] * n, "typed", "unknown", "", "typed for this recording "
                        "(an override replaces whatever the file records; flagged on every result)")
    if recording.stimulus.times_s and recording.stimulus.source:
        s = recording.stimulus
        s.reason = s.reason or f"the file's own record of the stimulus ({s.source})"
        return s
    considered: list[Candidate] = []
    order = [Role.TRIGGER, Role.LIGHT, Role.OTHER, Role.MONITOR]      # monitors last: considered, and rejected with the reason
    channels = [c for role in order for c in recording.channels_with(role)]
    if stimulus_channel:
        named = [c for c in channels if c.name.lower() == stimulus_channel.lower()]
        channels = named + [c for c in channels if c not in named]
    recorded_first = sorted(channels, key=lambda c: (not c.recorded, order.index(c.role) if c.role in order else 9))
    chosen = None
    for c in recorded_first:
        cand = _judge(c, recording, max_pulse_s, light_inputs)
        considered.append(cand)
        if cand.accepted and chosen is None and (not stimulus_channel or c.name.lower() == stimulus_channel.lower()
                                                 or not any(x.name.lower() == stimulus_channel.lower() for x in channels)):
            chosen = cand
    lines = [f"{c.channel.name}: {'taken' if c is chosen else 'not taken'} ({c.reason})" for c in considered]
    if chosen is None:
        pulses = command_pulses(recording)
        if pulses is not None:
            times, widths, name = pulses
            lines.append(f"{name}: taken (brief pulses of {COMMAND_PULSE_MAX_S * 1e3:g} ms or less in the command, "
                         "the last place a stimulus can be read from)")
            return Stimulus(times, widths, f"command {name}", "command pulse", name, "; ".join(lines))
        # nothing records it: the stimulator's own artifact in the signal
        arts = [a for a in (artifacts or []) if a.kind == "stimulus artifact"]
        if arts:
            def at(a, k):                       # where it is in sweep k (an artifact on its own clock moves)
                return a.start_s if a.times_s is None else float(a.times_s[k]) if k < a.times_s.size else np.nan

            times = [np.array(sorted(x for x in (at(a, k) for a in arts) if np.isfinite(x))) for k in range(n)]
            lines.append("the signal: " + "; ".join(a.describe(recording.signal_units) for a in arts) +
                         " (no channel records the stimulus; its artifact is taken as it)")
            return Stimulus(times, [np.full(t.size, 1e-4) for t in times], "artifact in the signal", "electrical", "",
                            "; ".join(lines))
        return Stimulus([np.array([])] * n, [np.array([])] * n, "", "unknown", "",
                        "no stimulus found: " + ("; ".join(lines) if lines else "no trigger, light or output channel"))
    kind = chosen.kind
    if chosen.kind_reason:
        lines.append(chosen.kind_reason)
    if rig is not None and rig.light_waiting:
        if kind == "unknown":
            kind = "light"
            lines.append(f"light: {rig.light_words()}, and this line fired")
        elif kind == "electrical":
            lines.append(f"{rig.light_words()}: if this line triggered it, the stimulus is light (its name says "
                         "electrical, which is kept)")
        elif chosen.kind_reason.startswith("light: on this rig"):
            lines.append(f"{rig.light_words()}: it agrees")
    source = ("trigger channel" if chosen.channel.recorded else "digital output") if kind != "light" else \
        ("light channel" if chosen.channel.recorded else "light output")
    return Stimulus(chosen.times_s, chosen.widths_s, f"{source} {chosen.channel.name}", kind, chosen.channel.name,
                    "; ".join(lines))


def command_pulses(recording: Recording) -> tuple[list, list, str] | None:
    """Brief pulses in the command (at most COMMAND_PULSE_MAX_S away from the
    holding level), when there are any: a stimulus given through the pipette.
    A step family, a test pulse or a DSI step lasts longer and is not one."""
    c = recording.channel(Role.COMMAND)
    if c is None:
        return None
    cmd = recording.read(c)
    fs = recording.sample_rate_hz
    times, widths, found = [], [], False
    for k in range(cmd.shape[0]):
        x = cmd[k]
        hold = float(np.median(x[: max(2, x.size // 64)]))
        away = np.abs(x - hold) > max(0.5, 5 * float(np.std(np.diff(x[: max(3, x.size // 64)]))))
        d = np.diff(away.astype(np.int8))
        starts = np.flatnonzero(d == 1) + 1
        ends = np.flatnonzero(d == -1) + 1
        if away[0]:
            starts = np.r_[0, starts]
        if away[-1]:
            ends = np.r_[ends, x.size]
        w = (ends - starts) / fs
        brief = w <= COMMAND_PULSE_MAX_S
        times.append(starts[brief] / fs)
        widths.append(w[brief])
        found = found or bool(brief.any())
    return (times, widths, c.name) if found else None


def describe(stimulus: Stimulus) -> str:
    if not len(stimulus):
        return "no stimulus"
    fixed = stimulus.fixed_times()
    counts = sorted({int(t.size) for t in stimulus.times_s})
    per = f"{counts[0]} pulse(s) per sweep" if len(counts) == 1 else f"{counts[0]}-{counts[-1]} pulses per sweep"
    where = "at the same times in every sweep" if fixed.size else "moving between sweeps"
    kind = {"light": "light", "electrical": "electrical", "unknown": "unknown kind",
            "command pulse": "a pulse in the command"}.get(stimulus.kind, stimulus.kind)
    text = f"{per}, {where}, {kind}, from the {stimulus.source}"
    if fixed.size > 1:
        isi = np.diff(fixed)
        text += f"; interval {isi[0] * 1e3:.0f} ms" + ("" if np.allclose(isi, isi[0], atol=1e-4) else " (irregular)")
    return text
