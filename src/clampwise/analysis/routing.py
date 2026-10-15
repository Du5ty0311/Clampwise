"""What kind of recording this is, and why (B2).

The type decides which analyses run. It is decided from what the file
records (the clamp mode, the command's epochs, the stimulus, the sweep
length and mode), and the evidence is written out so the screen can say
"a command step family, 33 step sizes, no stimulus". A protocol mapping (a
pattern on the protocol name, from a preset or typed) takes precedence, and
a typed override on the recording beats both.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field

import numpy as np

from ..core.recording import ClampMode, Recording, Role, Stimulus
from .stimulus import ELECTRICAL_WORDS, LIGHT_WORDS

DSI_MIN_STEP_S = 0.5           # a depolarizing step this long or longer, to -20 mV or above, is a DSI step
SPONTANEOUS_MIN_SWEEP_S = 10.0
MEMBRANE_TEST_MAX_SWEEP_S = 0.25
MEMBRANE_TEST_MAX_STEP_MV = 20.0


@dataclass(frozen=True)
class Mapping:
    """``pattern`` on the protocol name -> a type, and optionally which analyses
    run, which channel is the stimulus and whether it is light or electrical."""

    pattern: str
    recording_type: str
    analyses: tuple[str, ...] = ()
    stimulus_channel: str = ""
    stimulus_kind: str = ""

    def matches(self, protocol: str) -> bool:
        return fnmatch.fnmatchcase((protocol or "").lower(), self.pattern.lower())


@dataclass
class Decision:
    recording_type: str
    reason: str
    source: str                             # "detected", "mapping <pattern>", "typed"
    evidence: dict[str, str] = field(default_factory=dict)
    mapping: Mapping | None = None

    def describe(self) -> str:
        return f"{self.recording_type} ({self.source}): {self.reason}"


def _steps(recording: Recording) -> tuple[list[float], float]:
    """Distinct step levels of the command (relative to the holding) across the
    sweeps, and the longest step duration."""
    levels, longest = set(), 0.0
    for sweep in recording.epochs:
        for e in sweep:
            if e.kind == "step":
                levels.add(round(e.level, 3))
                longest = max(longest, e.end_s - e.start_s)
    if recording.epochs or recording.command is None:
        return sorted(levels), longest
    # no epoch table (an ATF): the step read from the command waveform itself (a
    # recorded monitor, so noisy), its level rounded to a resolution above the noise
    cmd = np.asarray(recording.command, float)
    fs = recording.sample_rate_hz
    for k in range(cmd.shape[0]):
        c = cmd[k]
        head = c[: max(8, c.size // 64)]
        hold = float(np.median(head))
        noise = float(1.4826 * np.median(np.abs(head - hold))) or 1e-9
        dev = np.abs(c - hold)
        if dev.max() < max(10 * noise, 1.0):
            continue
        on = np.flatnonzero(dev > 0.5 * dev.max())
        level = float(np.median(c[on]))
        step = max(1.0, 10 * noise)
        levels.add(round(round(level / step) * step, 3))
        longest = max(longest, (on[-1] - on[0] + 1) / fs)
    return sorted(levels), longest


RAMP_MIN_S = 0.1              # a command rising this long, steadily, by at least RAMP_MIN_PA is a ramp
RAMP_MIN_PA = 20.0


def ramp_of(recording: Recording, sweep: int) -> tuple[int, int, float] | None:
    """The ramp in one sweep's current command: (start, end sample, its rate in
    pA/s), from an epoch of kind "ramp", else the longest stretch away from the
    holding that a straight line fits (R squared at least 0.98) for at least
    RAMP_MIN_S, rising or falling by at least RAMP_MIN_PA."""
    fs = recording.sample_rate_hz
    if recording.epochs and sweep < len(recording.epochs):
        for e in recording.epochs[sweep]:
            if e.kind == "ramp" and e.end_s - e.start_s >= RAMP_MIN_S:
                a, b = int(round(e.start_s * fs)), int(round(e.end_s * fs))
                cmd = recording.command
                lo = float(np.median(cmd[sweep, max(0, a - int(0.01 * fs)):a])) if cmd is not None and a > 0 else 0.0
                return a, b, (e.level - lo) / (e.end_s - e.start_s) if e.end_s > e.start_s else 0.0
        if recording.epochs[sweep]:
            return None                         # Clampex's own table says what it played: no ramp
    if recording.command is None:
        return None
    c = np.asarray(recording.command[sweep], float)
    hold = float(np.median(c[: max(2, c.size // 64)]))
    away = np.abs(c - hold) > 1.0
    if not away.any():
        return None
    edges = np.flatnonzero(np.diff(np.r_[0, away.astype(np.int8), 0]))
    runs = list(zip(edges[::2], edges[1::2]))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    if (b - a) / fs < RAMP_MIN_S or abs(c[b - 1] - c[a]) < RAMP_MIN_PA:
        return None
    x = np.arange(a, b) / fs
    p = np.polyfit(x, c[a:b], 1)
    resid = c[a:b] - np.polyval(p, x)
    r2 = 1 - float(np.sum(resid ** 2)) / max(float(np.sum((c[a:b] - c[a:b].mean()) ** 2)), 1e-12)
    return (int(a), int(b), float(p[0])) if r2 >= 0.98 else None


def _injection(recording: Recording, stimulus: Stimulus) -> tuple[str, str] | None:
    """In current clamp, current put into the recorded cell itself: a brief pulse
    (a short square, Allen's 3 ms) or a ramp. Neither is a synaptic stimulus."""
    if recording.clamp_mode is not ClampMode.CURRENT_CLAMP or recording.command is None:
        return None
    from .stimulus import LIGHT_WORDS

    light = any(c.role is Role.LIGHT for c in recording.channels) or \
        any(w in (recording.protocol or "").lower() for w in LIGHT_WORDS)
    if stimulus.source.startswith("command") and not light:        # a pulse that drives a light is a stimulus
        widths = [w for ws in stimulus.widths_s for w in ws]
        return "cc_short_square", (f"brief current pulses through the pipette ({1e3 * float(np.median(widths)):.1f} ms): "
                                   "the cell's own threshold, not a synaptic stimulus")
    for k in range(min(recording.n_sweeps, 5)):
        found = ramp_of(recording, k)
        if found is not None:
            a, b, rate = found
            return "cc_ramp", (f"a current ramp of {rate:.3g} pA/s over {(b - a) / recording.sample_rate_hz:.2f} s "
                               f"(sweep {k + 1})")
    return None


def _is_field(recording: Recording, hint: str) -> bool:
    text = f"{recording.protocol} {recording.source.name} {hint}".lower()
    if any(w in text for w in ("lfp", "field", "fepsp")):
        return True
    return recording.header.get("recording_kind") == "field"


def detect(recording: Recording, stimulus: Stimulus, *, mode_hint: str = "") -> Decision:
    """The recording's type from its own evidence."""
    mode = recording.clamp_mode
    has_stim = len(stimulus) > 0
    evidence: dict[str, str] = {"clamp": mode.value, "stimulus": "yes" if has_stim else "no",
                                "sweeps": f"{recording.n_sweeps} x {recording.sweep_duration_s:g} s"
                                if not recording.is_gap_free else f"gap-free, {recording.duration_s:.0f} s"}
    if recording.header.get("recording_kind") == "spike_times":
        n = sum(len(u) for u in recording.header.get("spike_times_s", []))
        return Decision("spike_times", f"a file of spike times ({n} spikes)", "detected", evidence)
    if _is_field(recording, mode_hint):
        kind = "field_evoked" if has_stim else "field_spontaneous"
        return Decision(kind, "a field recording" + (" with a stimulus" if has_stim else " without a stimulus"),
                        "detected", evidence)
    if recording.header.get("event_driven") and mode is not ClampMode.UNKNOWN:
        kind = "vc_spontaneous" if mode is ClampMode.VOLTAGE_CLAMP else "cc_spontaneous"
        return Decision(kind, f"an event-driven file: {recording.n_sweeps} sweeps, each one triggered event",
                        "detected", evidence)
    levels, longest = _steps(recording)
    away = [lv for lv in levels if np.isfinite(recording.holding) and abs(lv - recording.holding) > 0.5] \
        if np.isfinite(recording.holding) else [lv for lv in levels if abs(lv) > 0.5]
    evidence["steps"] = f"{len(away)} step size(s)" if away else "no steps"
    vc = mode is ClampMode.VOLTAGE_CLAMP
    # DSI: voltage clamp, a long depolarizing step to -20 mV or above, from a holding at or below -40 mV
    if vc and away and recording.epochs:
        hold = recording.holding if np.isfinite(recording.holding) else min(levels, default=float("nan"))
        big = [e for sweep in recording.epochs for e in sweep
               if e.kind == "step" and e.end_s - e.start_s >= DSI_MIN_STEP_S and e.level >= -20.0 and hold <= -40.0]
        if big:
            return Decision("dsi", f"voltage clamp with a depolarizing step of {max(e.end_s - e.start_s for e in big):g} s "
                            f"to {max(e.level for e in big):g} mV from {hold:g} mV", "detected", evidence)
    if vc and not has_stim and recording.sweep_duration_s <= MEMBRANE_TEST_MAX_SWEEP_S and len(away) == 1 \
            and abs(away[0] - (recording.holding if np.isfinite(recording.holding) else 0)) <= MEMBRANE_TEST_MAX_STEP_MV:
        return Decision("membrane_test", f"voltage clamp, {recording.sweep_duration_s * 1e3:.0f} ms sweeps with one "
                        f"test step of {away[0] - (recording.holding if np.isfinite(recording.holding) else 0):g} mV "
                        "and no stimulus", "detected", evidence)
    if vc and recording.n_sweeps == 1 and not has_stim:
        from .vclamp import level_runs

        runs = level_runs(recording, 0)
        if len({lv for _, _, lv in runs}) >= 4:
            return Decision("vc_steps", f"a voltage staircase: {len(runs)} levels held in one sweep", "detected",
                            evidence)
    injected = _injection(recording, stimulus)
    if injected is not None:
        return Decision(injected[0], injected[1], "detected", evidence)
    if has_stim:
        kind = "vc_evoked" if vc else "cc_evoked"
        return Decision(kind, f"a stimulus was found ({'voltage' if vc else 'current'} clamp)", "detected", evidence)
    if len(away) >= 2:
        kind = "vc_steps" if vc else "cc_steps"
        return Decision(kind, f"a command step family, {len(away)} step sizes, no stimulus", "detected", evidence)
    if recording.is_gap_free or recording.sweep_duration_s >= SPONTANEOUS_MIN_SWEEP_S:
        kind = "vc_spontaneous" if vc else "cc_spontaneous"
        what = "a gap-free file" if recording.is_gap_free else f"sweeps of {recording.sweep_duration_s:g} s"
        return Decision(kind, f"no stimulus and {what} ({'voltage' if vc else 'current'} clamp)", "detected", evidence)
    if mode is ClampMode.UNKNOWN:
        return Decision("unknown", "the signal's units say neither current nor voltage clamp", "detected", evidence)
    proto = (recording.protocol or "").lower()
    if len(away) == 1 and recording.n_sweeps == 1 and longest >= 0.1:
        # a step protocol stopped after its first sweep (62 of the lab corpus's 5,143 files, 2026-10-09)
        return Decision("unknown", f"one sweep with one command step of {away[0]:g} {'mV' if vc else 'pA'} for {longest:.2f} s: "
                        "a step family needs at least two step sizes, so this file was stopped after its first sweep",
                        "detected", evidence)
    if len(away) == 1 and not vc and recording.sweep_duration_s <= MEMBRANE_TEST_MAX_SWEEP_S:
        return Decision("unknown", f"current clamp, {recording.sweep_duration_s * 1e3:.0f} ms sweeps with one pulse of "
                        f"{away[0]:g} pA: a seal test in current clamp, which no analysis measures (a membrane test is read "
                        "in voltage clamp)", "detected", evidence)
    if any(w in proto for w in LIGHT_WORDS + ELECTRICAL_WORDS):
        # the protocol was a stimulus protocol, but nothing in the file records when the stimulus came
        return Decision("unknown", f"the protocol '{recording.protocol}' is named for a stimulus but no channel records "
                        f"one ({stimulus.reason}); no step family, and sweeps too short for spontaneous activity",
                        "detected", evidence)
    return Decision("unknown", "no stimulus, no step family, and sweeps too short for spontaneous activity",
                    "detected", evidence)


def decide(recording: Recording, stimulus: Stimulus, *, mappings: list[Mapping] = (), typed: str = "",
           mode_hint: str = "") -> Decision:
    """Typed override, else the first matching protocol mapping, else detection."""
    detected = detect(recording, stimulus, mode_hint=mode_hint)
    if typed:
        return Decision(typed, f"set for this recording on Check (detected: {detected.recording_type}, "
                        f"{detected.reason})", "typed", detected.evidence)
    for m in mappings:
        if m.matches(recording.protocol):
            return Decision(m.recording_type, f"protocol '{recording.protocol}' matches the mapping "
                            f"'{m.pattern}' (detected: {detected.recording_type}, {detected.reason})",
                            f"mapping {m.pattern}", detected.evidence, m)
    return detected
