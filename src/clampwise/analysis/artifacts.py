"""What the signal shows that may be no channel's record: stimulus artifacts,
test pulses and the command's own changes: every artifact is identified, labelled
and flagged, and recorded where the software did not record it.

Read on the sweeps averaged (a time-locked artifact survives the average,
spontaneous activity does not), episodic recordings only:

- **stimulus artifact**: a jump between two samples of at least
  ``ARTIFACT_SD`` times the noise of the averaged sweep's sample-to-sample
  differences that starts at once (twice each change before it) and turns
  back within 0.5 ms, at the same sample in at least 80% of the sweeps; else
  each sweep's own sharpest jump, the same size in 80% of them (a stimulator
  on its own clock, looked for only where the command never changes). An
  extracellular stimulator's capacitive spike: an action potential's upstroke
  builds over several samples, and an evoked one jitters. With one or two
  sweeps (nothing to show it repeats) a jump must be 100 times the noise, else
  it is labelled an artifact and not taken as a stimulus.
- **test pulse**: a square step the command does not hold (a seal or bridge
  test from the amplifier or from a second output): a change of level, the
  mean over 10 ms against the 10 ms before (2 ms either side of the edge left
  out), of at least ``EDGE_SD`` times that measure's own robust spread, then the
  next change the other way 20 ms to 1 s later, 0.6 to 1.6 times its size, the
  level between them held (its last 40% varies by less than a quarter of its
  size) and the signal back afterwards. A synaptic response rises and decays;
  it holds no level. Edges within 30 ms after a command change are the
  command's.
- **command change**: where the recorded command changes level (the epoch
  table), its capacitive transient and charging with it.
- **step**: two such jumps the other way of each other and the same size,
  2 ms to 2 s apart: a step's edges (capacitive or bridge jumps), from an
  output the file does not record, or the recorded step where the signal shows
  it (flagged when that is not where the epoch table puts it).
- **artifact**: a sharp jump in one or two sweeps that cannot be shown to repeat.

A jump at a command change or at a test pulse's edge is that change's
transient, not a stimulus. Each artifact says where it is, how big, and why
it was called what it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.recording import Recording

ARTIFACT_SD = 25.0          # the jump against the noise of the averaged sweep's differences
EDGE_SD = 12.0              # a test pulse's edge against the spread of the level-change measure
LOCKED_FRACTION = 0.8       # of the sweeps that carry the artifact at the same sample
MAX_SWEEPS = 40             # sweeps read (evenly spaced) to find them
COMMAND_SETTLE_S = 0.02     # a command change's transient and charging, kept with it


@dataclass
class Artifact:
    kind: str                   # "stimulus artifact", "test pulse", "step", "command change", "artifact"
    start_s: float
    end_s: float
    size: float                 # in the signal's units (the jump, or the pulse's level against the baseline)
    recorded: bool              # whether a channel or the epoch table records it
    reason: str
    times_s: np.ndarray | None = None       # a stimulus artifact: where it is in each sweep
    jumps: np.ndarray | None = None         # a jump at one time: its signed size in each sweep read

    def describe(self, units: str = "") -> str:
        where = f"{self.start_s * 1e3:.1f} ms" if self.kind == "stimulus artifact" else \
            f"{self.start_s * 1e3:.1f}-{self.end_s * 1e3:.1f} ms"
        return f"{self.kind} at {where} ({self.size:+.3g} {units}): {self.reason}"


def _mean(x: np.ndarray, n: int) -> np.ndarray:
    """A running mean of n samples, the ends from fewer samples (no zero padding)."""
    if n <= 1:
        return x.astype(float)
    c = np.cumsum(np.r_[0.0, x.astype(float)])
    i = np.arange(x.size)
    lo, hi = np.maximum(i - n // 2, 0), np.minimum(i - n // 2 + n, x.size)
    return (c[hi] - c[lo]) / (hi - lo)


def command_edges(recording: Recording) -> np.ndarray:
    """Where the recorded command changes level, over every sweep (s)."""
    from .stimulus import _step_edges

    fs = recording.sample_rate_hz
    edges = [_step_edges(sweep, fs, recording.holding) for sweep in recording.epochs]
    return np.unique(np.round(np.concatenate(edges), 4)) if edges else np.array([])


def find(recording: Recording) -> list[Artifact]:
    if recording.is_gap_free or recording.n_sweeps < 1 or recording.n_samples < 100:
        return []
    fs = recording.sample_rate_hz
    idx = np.unique(np.linspace(0, recording.n_sweeps - 1, min(recording.n_sweeps, MAX_SWEEPS)).astype(int))
    x = np.asarray(recording.read(recording.signal_channel, list(idx)), float)
    x = x[np.isfinite(x).all(axis=1)] if x.ndim == 2 else x
    if x.size == 0:
        return []
    mean = x.mean(axis=0)
    t = np.arange(mean.size) / fs
    out = [Artifact("command change", e, min(t[-1], e + COMMAND_SETTLE_S), 0.0, True,
                    "the command changes level here (the epoch table); its transient and the charging that follows "
                    "go with it") for e in command_edges(recording)]
    near_command = [a.start_s for a in out]
    pulses = _test_pulses(mean, fs, near_command)
    out += pulses
    edges = near_command + [e for p in pulses for e in (p.start_s, p.end_s)]
    out += _steps_from_jumps(_stimulus_artifacts(recording, x, mean, fs, edges), near_command)
    return sorted(out, key=lambda a: a.start_s)


def _steps_from_jumps(jumps: list[Artifact], command: list[float]) -> list[Artifact]:
    """Two jumps the other way of each other and the same size, 2 ms to 2 s apart,
    are a step's edges (its capacitive or bridge jumps), not two stimuli: a step
    on an output the command does not record, or the recorded step where the
    signal shows it (said when the times differ)."""
    jumps = sorted(jumps, key=lambda a: a.start_s)

    def opposite(a: Artifact, b: Artifact) -> bool:
        if a.jumps is not None and b.jumps is not None and a.jumps.size == b.jumps.size:
            # sweep by sweep (a step family's edges change with its current, the mean may hide them)
            ok = (np.sign(a.jumps) == -np.sign(b.jumps)) & (np.abs(b.jumps) >= 0.5 * np.abs(a.jumps)) & \
                 (np.abs(b.jumps) <= 2.0 * np.abs(a.jumps))
            if ok.mean() >= LOCKED_FRACTION:
                return True
        return np.sign(a.size) != np.sign(b.size) and 0.67 <= abs(b.size / a.size) <= 1.5

    steps = [(a, b) for a, b in zip(jumps, jumps[1:]) if 0.002 <= b.start_s - a.start_s <= 2.0 and opposite(a, b)]
    paired = {id(x) for pair in steps for x in pair}
    out = [a for a in jumps if id(a) not in paired]
    cmd = sorted(command)
    for a, b in steps:
        dur = b.start_s - a.start_s
        match = next(((c0, c1) for c0, c1 in zip(cmd, cmd[1:]) if abs((c1 - c0) - dur) < 1e-3), None)
        why = (f"two jumps the other way of each other ({a.size:+.3g}, {b.size:+.3g}), {dur * 1e3:.1f} ms apart: the "
               "edges of a step, not a stimulus")
        if match is not None:
            why += (f"; the command's own step ({match[0] * 1e3:.1f}-{match[1] * 1e3:.1f} ms in the epoch table) shows in "
                    f"the signal {(match[0] - a.start_s) * 1e3:+.1f} ms from where the table puts it")
        else:
            why += "; no command the file records steps there (a second output, or the amplifier)"
        out.append(Artifact("step", a.start_s, b.end_s, a.size, match is not None, why))
    return out


def _sharp(y: np.ndarray, i: int, fs: float) -> tuple[bool, float, int]:
    """Whether the jump from sample i to i+1 is an artifact's: it starts at once
    (at least twice each of the three changes before it; an action potential's
    upstroke builds over samples) and turns back within 0.5 ms (a change the
    other way of at least 30% of it: a capacitive spike, one- or two-sided).
    Returns (it is, its size against the level before, samples until it turns)."""
    ms, turn = max(4, int(round(1e-3 * fs))), max(2, int(round(5e-4 * fs)))
    if i < ms or i + ms >= y.size:
        return False, 0.0, 0
    jump = float(y[i + 1] - y[i])
    if abs(jump) < 2 * float(np.max(np.abs(np.diff(y[i - 3:i + 1])))):
        return False, jump, 0
    after = np.diff(y[i + 1:i + 2 + turn])
    back = np.flatnonzero(np.sign(after) == -np.sign(jump))
    back = back[np.abs(after[back]) >= 0.3 * abs(jump)]
    if not back.size:
        return False, jump, 0
    pre = float(np.median(y[max(0, i - 2 * ms):i - 1]))
    seg = y[i + 1:i + 2 + turn] - pre
    return True, float(seg[np.argmax(np.abs(seg))]), int(back[0]) + 1


def _robust(d: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(d - np.median(d)))) or float(np.std(d)) or 1e-12


def _stimulus_artifacts(recording: Recording, x: np.ndarray, mean: np.ndarray, fs: float,
                        edges: list[float]) -> list[Artifact]:
    d = np.diff(mean)
    noise = _robust(d)
    ms = max(2, int(round(1e-3 * fs)))
    cand = np.flatnonzero(np.abs(d) > ARTIFACT_SD * noise)
    found: list[Artifact] = []
    k = 0
    while k < cand.size:
        group = cand[(cand >= cand[k]) & (cand < cand[k] + ms)]
        k += group.size
        i = int(group[0])                      # its onset (a two-sided artifact's larger swing may be its return)
        if any(abs(i / fs - e) < 3e-3 for e in edges):
            continue
        ok, peak, back = _sharp(mean, i, fs)
        if not ok:
            continue
        # time-locked: each sweep jumps at the same sample (a clipped artifact spreads over a few)
        steps = np.diff(x[:, max(0, i - 2):i + 4], axis=1)
        signed = steps[np.arange(steps.shape[0]), np.argmax(np.abs(steps), axis=1)]
        present = np.abs(signed) >= 0.3 * abs(d[i])
        if present.mean() < LOCKED_FRACTION:
            continue
        # a stimulus's artifact is the same in every sweep; a jump that changes size or sign with the sweep is a
        # step's edge (a family's current on an output the file does not record)
        typical = float(np.median(signed))
        same = (np.sign(signed) == np.sign(typical)) & (np.abs(signed) >= abs(typical) / 3) & \
            (np.abs(signed) <= 3 * abs(typical))
        # a sweep or two cannot show it repeats at all: never called a stimulus's
        repeated = x.shape[0] >= 3 or (x.shape[0] == 2 and abs(d[i]) >= 100 * noise)
        kind = "stimulus artifact" if repeated and same.mean() >= LOCKED_FRACTION else "artifact"
        why = (f"a jump of {abs(d[i]) / noise:.0f} times the noise between two samples, at once and turning back "
               f"within {back / fs * 1e3:.2f} ms, at the same sample in {present.mean():.0%} of the {x.shape[0]} "
               "sweep(s)")
        if not repeated:
            why += "; too few sweeps to show it repeats, so not taken as a stimulus's"
        elif kind == "artifact":
            why += "; its size or sign changes from sweep to sweep: a step's edge, not a stimulus's artifact"
        found.append(Artifact(kind, i / fs, (i + 1 + back) / fs, peak, False, why, jumps=signed))
    # a stimulator on its own clock only where Clampex plays nothing (the command never changes)
    if found or edges or recording.n_sweeps < 5 or recording.n_sweeps * recording.n_samples > 5e7:
        return found
    # not at one time: the stimulator on its own clock, so each sweep's own sharpest jump, the same size in most
    y = np.asarray(recording.signal, float)
    t = np.arange(y.shape[1] - 1) / fs
    away = np.ones(t.size, bool)
    for e in edges:
        away &= np.abs(t - e) >= 3e-3
    at, size = np.full(y.shape[0], np.nan), np.full(y.shape[0], np.nan)
    for s in range(y.shape[0]):
        dd = np.diff(y[s])
        big = np.where(away, np.abs(dd), 0.0)
        m = int(np.argmax(big))
        over = np.flatnonzero(big[max(0, m - 3):m + 1] > ARTIFACT_SD * _robust(dd))
        i = max(0, m - 3) + int(over[0]) if over.size else m      # the onset of the sharpest jump
        if abs(dd[i]) > ARTIFACT_SD * _robust(dd) and _sharp(y[s], i, fs)[0]:
            at[s], size[s] = (i + 1) / fs, dd[i]
    have = np.isfinite(at)
    if have.mean() >= LOCKED_FRACTION:
        typical = float(np.median(size[have]))
        same = have & (np.abs(size) >= 0.67 * abs(typical)) & (np.abs(size) <= 1.5 * abs(typical)) & \
            (np.sign(size) == np.sign(typical))
        if same.mean() >= LOCKED_FRACTION:
            at[~same] = np.nan
            spread = np.nanmax(at) - np.nanmin(at)
            found.append(Artifact("stimulus artifact", float(np.nanmedian(at)), float(np.nanmedian(at)) + 1e-3,
                                  typical, False,
                                  f"a jump between two samples in {same.mean():.0%} of the sweeps, at a different time in "
                                  f"each (over {spread * 1e3:.0f} ms: the stimulator ran on its own clock)", times_s=at))
    return found


def _test_pulses(mean: np.ndarray, fs: float, command: list[float]) -> list[Artifact]:
    from scipy.ndimage import median_filter

    # on 0.5 ms blocks, the level the median of 10 ms: a capacitive transient of a
    # millisecond or two at a pulse's edge cannot move it
    block = max(1, int(round(0.0005 * fs)))
    nb = mean.size // block
    gb, wb = 4, 20                                  # 2 ms left out either side, 10 ms read
    if nb < 2 * (gb + wb) + 2:
        return []
    y = mean[:nb * block].reshape(nb, block).mean(axis=1)
    med = median_filter(y, size=wb, mode="nearest")
    j = np.arange(gb + wb, nb - gb - wb)
    level = med[j + gb + wb // 2] - med[j - gb - wb // 2]
    spread = 1.4826 * float(np.median(np.abs(level - np.median(level)))) or 1e-12
    big = np.abs(level) > EDGE_SD * spread
    edges = []
    pos = np.flatnonzero(big)
    jump = np.abs(np.diff(_mean(mean, block)))
    reach = int(round(0.008 * fs))
    for r in (np.split(pos, np.flatnonzero(np.diff(pos) > 1) + 1) if pos.size else []):
        m = int(r[np.argmax(np.abs(level[r]))])
        at = int(j[m]) * block                      # then to the sample where it happens: the largest jump near it
        lo = max(0, at - reach)
        edges.append((lo + int(np.argmax(jump[lo:at + reach])), float(level[m])))
    g, w, n = 2 * gb * block, wb * block, mean.size
    sm = _mean(mean, max(1, int(round(0.001 * fs))))
    near = [k for k, (e, _) in enumerate(edges) if any(-0.002 < e / fs - c < COMMAND_SETTLE_S + 0.010 for c in command)]
    out = []
    for a, (e1, l1) in enumerate(edges):
        if a in near:
            continue
        # its pair: the next edge the other way, unless one this way as large comes first (then this is not the onset)
        b = next((k for k in range(a + 1, len(edges)) if np.sign(edges[k][1]) != np.sign(l1)
                  or abs(edges[k][1]) >= 0.6 * abs(l1)), None)
        if b is None or b in near or np.sign(edges[b][1]) == np.sign(l1):
            continue
        e2, l2 = edges[b]
        dur = (e2 - e1) / fs
        if not 0.020 <= dur <= 1.0 or not 0.6 <= abs(l2 / l1) <= 1.6:
            continue
        base = float(np.median(sm[max(0, e1 - w - g):e1 - g]))
        held = sm[e2 - g - int(0.4 * (e2 - e1)):e2 - g]
        if held.size < 3:
            continue
        lv = float(np.median(held)) - base
        flat = float(np.percentile(held, 95) - np.percentile(held, 5)) <= 0.25 * abs(lv)
        if np.sign(lv) != np.sign(l1) or abs(lv) < 0.25 * abs(l1) or not flat:
            continue
        back = sm[min(n - 1, e2 + 2 * g):min(n, e2 + 2 * g + 3 * w)]
        if back.size and abs(float(np.median(back)) - base) > 0.5 * abs(lv):
            continue
        out.append(Artifact("test pulse", e1 / fs, e2 / fs, lv, False,
                            f"a square step of {lv:+.3g} held for {dur * 1e3:.0f} ms and back, that the command "
                            "does not hold (a seal or bridge test from the amplifier or another output)"))
    return out


# ------------------------------------------------------------------ the declaration

from .registry import TYPES, Output, register  # noqa: E402  (after the finder, which the pipeline imports alone)
from .result import Drawable, Flag, Result  # noqa: E402

OUTPUTS = (
    Output("n_stimulus_artifacts", "Stimulus artifacts found in the signal (a jump between two samples, at the same "
           "sample in most sweeps, back within 1 ms).", "count", label="Stimulus artifacts"),
    Output("n_test_pulses", "Test pulses found in the signal that no channel records (a held square step and back).",
           "count", label="Unrecorded test pulses"),
    Output("test_pulse_level", "The first unrecorded test pulse's level against the baseline, in the signal's units.",
           "signal units", label="Test pulse: level"),
    Output("test_pulse_ms", "The first unrecorded test pulse's duration.", "ms", label="Test pulse: duration"),
    Output("n_command_changes", "Where the recorded command changes level.", "count", label="Command changes"),
    Output("n_steps", "Steps seen only by their edges: two equal jumps the other way of each other (a step on an "
           "output the file does not record, or the recorded step at another time than its record).", "count",
           label="Steps seen by their edges"),
    Output("n_other_artifacts", "Sharp jumps in a sweep or two, too few sweeps to call a stimulus's.", "count",
           label="Other artifacts"),
    Output("kind", "What it is: a stimulus artifact, a test pulse or a command change.", "", level="artifact"),
    Output("start_ms", "Where it starts in the sweep.", "ms", level="artifact"),
    Output("end_ms", "Where it ends (a stimulus artifact: back below half its size).", "ms", level="artifact"),
    Output("size", "Its size: the jump, or the test pulse's level against the baseline.", "signal units",
           level="artifact"),
    Output("recorded", "Whether a channel or the epoch table records it.", "", level="artifact"),
    Output("why", "Why it was called what it is, with the numbers.", "", level="artifact"),
)


@register("artifacts", "Artifacts and test pulses", version=1, types=tuple(t for t in TYPES if t != "spike_times"),
          outputs=OUTPUTS,
          description="What the signal shows besides the cell: stimulus artifacts, test pulses and the command's "
                      "changes, each labelled with why; one that no channel records is said, the stimulus is taken "
                      "from its artifact when nothing records it, and the time-locked test keeps sham times off all "
                      "of them.",
          source="Clampwise")
def run(recording, context, s) -> Result:
    out = Result("artifacts", 1)
    found = getattr(context, "artifacts", []) or []
    units = recording.signal_units
    stim = [a for a in found if a.kind == "stimulus artifact"]
    pulses = [a for a in found if a.kind == "test pulse"]
    steps = [a for a in found if a.kind == "step"]
    out.values.update({"n_stimulus_artifacts": len(stim), "n_test_pulses": len(pulses),
                       "n_steps": len(steps), "n_other_artifacts": sum(a.kind == "artifact" for a in found),
                       "n_command_changes": sum(a.kind == "command change" for a in found)})
    if pulses:
        out.values.update({"test_pulse_level": pulses[0].size,
                           "test_pulse_ms": (pulses[0].end_s - pulses[0].start_s) * 1e3})
    out.tables["artifacts"] = [{"kind": a.kind, "start_ms": a.start_s * 1e3, "end_ms": a.end_s * 1e3,
                                "size": a.size, "recorded": a.recorded, "why": a.reason} for a in found]
    for a in pulses:
        out.flags.append(Flag("unrecorded test pulse", a.describe(units) + "; no channel records it, so its command "
                              "is unknown and no resistance is read from it; kept out of the time-locked test",
                              source="Clampwise", severity="note"))
    if stim and context.stimulus.source == "artifact in the signal":
        out.flags.append(Flag("stimulus from its artifact", "no channel records the stimulus; it is taken from its "
                              "artifact in the signal: " + "; ".join(a.describe(units) for a in stim),
                              source="", severity="note"))
    for a in steps:
        if a.recorded:                          # the command's own step, where the signal shows it
            out.flags.append(Flag("command timing", a.describe(units) + ". Windows read from the epoch table's times "
                                  "may sit off the response", source="Clampwise", severity="flag"))
        else:
            out.flags.append(Flag("unrecorded step", a.describe(units) + "; kept out of the time-locked test",
                                  source="Clampwise", severity="note"))
    for a in [a for a in found if a.kind == "artifact"]:
        out.flags.append(Flag("artifact", a.describe(units), source="Clampwise", severity="note"))
    for a in found:
        if a.kind != "command change":
            out.drawables.append(Drawable("window", a.kind, start_s=a.start_s, end_s=a.end_s, domain="time"))
    if not found:
        out.notes.append("no stimulus artifact, test pulse or command change in the signal"
                         + (" (a continuous recording is not searched)" if recording.is_gap_free else ""))
    return out
