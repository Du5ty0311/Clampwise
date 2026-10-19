"""Which pattern of patterned light each sweep saw: the Polygon's log, matched to a recording's sweeps by time.

Every step Clampwise shows on the Polygon (and every Off, every take and let go) is written to ``polygon_log.csv``
in the cell's folder, timed to the millisecond on the PC's clock when the Polygon took it, with what it lit: the
steps' shapes on the field and on the camera's picture (pixels and um), their area in um^2, the fraction of the
field lit, the objective and the byte that lit a mirror. A recording's sweeps are timed on the same PC's clock (its
start, then each sweep's start), so each sweep (and each light pulse in it, when the recording says when they came)
is given the pattern shown at that moment, with how sure the match is: sure when no change of pattern came within
the clock's uncertainty of the sweep (0.1 s for Clampwise's own recordings, 1 s for another program's, whose start
is stamped when it opens the file), else said why not.

The match is written beside the recording as ``<recording>.light.json`` (a file of its own: the recording is never
touched) and read back by the analyses, which can group sweeps by the region lit.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOG_NAME = "polygon_log.csv"
COLUMNS = ["time", "unix_s", "event", "step", "lit", "field_fraction", "area_um2", "objective", "on_value",
           "shapes", "pattern"]
OWN_CLOCK_S = 0.1           # Clampwise's own recordings: the same clock, the start taken as the file is begun
OTHER_CLOCK_S = 1.0         # another program's: its start stamped as it opens the file, on the same PC


@dataclass
class Shown:
    """One line of the log: ``event`` shown, off, sequence (the steps loaded, one shown on each trigger: ``shapes``
    then holds each step's name and what it lights), taken or let go."""
    unix_s: float
    event: str
    step: str = ""
    lit: bool = False
    field_fraction: float = 0.0
    area_um2: float | None = None
    objective: str = ""
    on_value: int | None = None
    shapes: list = field(default_factory=list)
    pattern: str = ""
    coarse: bool = False         # timed to the second only (a log written before times had milliseconds)

    @property
    def known(self) -> bool:
        """What the Polygon showed is known from here on (not after it was let go, or before it was taken)."""
        return self.event in ("shown", "off", "sequence")


def entry(event: str, step=None, pattern: str = "", dose: dict | None = None, objective: str = "",
          on_value: int | None = None, when: float | None = None) -> Shown:
    dose = dose or {}
    return Shown(time.time() if when is None else when, event, step.name if step is not None else "",
                 event == "shown" and bool(dose.get("field_fraction", 1.0)), float(dose.get("field_fraction", 0.0)),
                 dose.get("area_um2"), objective, on_value, list(dose.get("shapes", [])), pattern)


def write(folder: str | Path, e: Shown) -> Path | None:
    """``e`` added to the log in ``folder`` (None when there is no folder to write in). A log begun before these
    columns is kept under another name and a new one begun."""
    folder = Path(folder) if folder else None
    if folder is None or not folder.is_dir():
        return None
    path = folder / LOG_NAME
    try:
        if path.exists():
            with path.open(encoding="utf-8", newline="") as f:
                head = next(csv.reader(f), [])
            if head != COLUMNS:
                k = 1
                while (folder / f"polygon_log (before {k}).csv").exists():
                    k += 1
                path.rename(folder / f"polygon_log (before {k}).csv")
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(COLUMNS)
            stamp = dt.datetime.fromtimestamp(e.unix_s)
            w.writerow([stamp.strftime("%Y-%m-%d %H:%M:%S.") + f"{stamp.microsecond // 1000:03d}", f"{e.unix_s:.3f}",
                        e.event, e.step, "yes" if e.lit else "no", f"{e.field_fraction:.6f}",
                        "" if e.area_um2 is None else f"{e.area_um2:.1f}", e.objective,
                        "" if e.on_value is None else e.on_value, json.dumps(e.shapes, separators=(",", ":")),
                        e.pattern])
        return path
    except OSError:
        return None


def read(path: str | Path) -> list[Shown]:
    """The log's lines in time order; also the log written before times had milliseconds (time, step, shapes,
    pattern), its times then to the second."""
    out: list[Shown] = []
    try:
        with Path(path).open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return out
    for r in rows:
        try:
            if "unix_s" in r and r["unix_s"]:
                area = r.get("area_um2") or ""
                on = r.get("on_value") or ""
                out.append(Shown(float(r["unix_s"]), r.get("event", "shown"), r.get("step", ""),
                                 r.get("lit") == "yes", float(r.get("field_fraction") or 0.0),
                                 float(area) if area else None, r.get("objective", ""), int(on) if on else None,
                                 json.loads(r.get("shapes") or "[]"), r.get("pattern", "")))
            else:
                t = dt.datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S").timestamp()
                off = r.get("step") == "off"
                out.append(Shown(t + 0.5, "off" if off else "shown", "" if off else r.get("step", ""), not off,
                                 pattern=r.get("pattern", ""), coarse=True))
        except (KeyError, ValueError, TypeError):
            continue
    out.sort(key=lambda e: e.unix_s)
    return out


def _state(e: Shown | None) -> dict:
    if e is None:
        return {"step": None, "lit": None, "why": "nothing was logged before it (Clampwise did not hold the "
                                                  "patterned light)"}
    if e.event == "off":
        return {"step": "off", "lit": False}
    if e.event == "sequence":
        return {"step": e.step, "lit": True, "sequence": [d.get("step", "") for d in e.shapes],
                "objective": e.objective, "on_value": e.on_value, "pattern": e.pattern}
    if e.event == "shown":
        return {"step": e.step, "lit": e.lit, "field_fraction": e.field_fraction, "area_um2": e.area_um2,
                "objective": e.objective, "on_value": e.on_value, "shapes": e.shapes, "pattern": e.pattern}
    return {"step": None, "lit": None, "why": "the patterned light had been let go (another program may have "
                                              "shown anything)" if e.event == "let go" else "not shown yet"}


def match(log: list[Shown], start: dt.datetime, sweep_starts_s, sweep_s: float, clock_s: float = OTHER_CLOCK_S,
          pulses_s: list | None = None) -> list[dict]:
    """Each sweep's pattern: what was shown at its start (and, when ``pulses_s`` gives each sweep's light pulses
    from its start, at each pulse), whether that is sure, and why not when it is not."""
    times = [e.unix_s for e in log]
    t_start = start.timestamp()
    out = []
    for k, s0 in enumerate(np.asarray(sweep_starts_s, float)):
        t0, t1 = t_start + float(s0), t_start + float(s0) + sweep_s
        i = bisect_right(times, t0)
        before = log[i - 1] if i else None
        j = bisect_right(times, t1)
        during = log[i:j]
        after = log[j] if j < len(log) else None
        rec = {"sweep": k + 1, "start_s": round(float(s0), 4), **_state(before)}
        slack = clock_s + (0.5 if (before is not None and before.coarse) else 0.0)
        margin = min(t0 - before.unix_s if before is not None else math.inf,
                     after.unix_s - t1 if after is not None else math.inf)
        if pulses_s is not None and k < len(pulses_s) and len(pulses_s[k]):
            per = []
            for j, p in enumerate(np.asarray(pulses_s[k], float)):
                tp = t0 + p
                q = bisect_right(times, tp)
                at = log[q - 1] if q else None
                gaps = ([tp - times[q - 1]] if q else []) + ([times[q] - tp] if q < len(log) else [])
                near = min(gaps) if gaps else math.inf
                one = {"at_s": round(float(p), 4), **_state(at), "sure": bool(at is not None and at.known and
                                                                               near > slack)}
                names = one.pop("sequence", None)
                if names:                         # the trigger steps it: pulse j of each sweep, step j (of n)
                    one.update(step=names[j % len(names)], by_trigger=True)
                per.append(one)
            rec["pulses"] = per
        if before is None or not before.known:
            rec.update(sure=False)
        elif during:
            names = ", ".join(e.step or e.event for e in during)
            rec.update(sure=False, why=f"the pattern changed during the sweep ({names})",
                       changed_at_s=[round(e.unix_s - t0, 3) for e in during])
            if rec.get("pulses") and all(p["sure"] for p in rec["pulses"]):
                rec["why"] += "; each light pulse's own pattern is sure"
        elif margin <= slack:
            rec.update(sure=False, why=f"a change of pattern came {margin:.2f} s from the sweep, within the clocks' "
                                       f"uncertainty ({slack:g} s)")
        else:
            rec.update(sure=True)
        rec["margin_s"] = None if math.isinf(margin) else round(margin, 3)
        out.append(rec)
    return out


def sidecar_path(recording: str | Path) -> Path:
    return Path(recording).with_suffix(".light.json")


def write_sidecar(recording: str | Path, sweeps: list[dict], clock_s: float, log_path: str | Path) -> Path:
    path = sidecar_path(recording)
    body = {"recording": Path(recording).name, "log": Path(log_path).name, "clock_uncertainty_s": clock_s,
            "matched": time.strftime("%Y-%m-%d %H:%M:%S"), "matched_by": "Clampwise (the log's times and the "
            "recording's, on the same PC's clock)", "sweeps": sweeps,
            "note": "Irradiance where a pattern lights is the same for any pattern at one LED command (each mirror "
                    "passes the same light); the power on the slice goes with the area lit."}
    path.write_text(json.dumps(body, indent=1), encoding="utf-8")
    return path


def find_log(recording: str | Path, folders=()) -> Path | None:
    """The Polygon's log for a recording: beside it, or in ``folders`` (the cell's)."""
    rec = Path(recording)
    for f in [rec.parent, *map(Path, folders)]:
        if (f / LOG_NAME).is_file():
            return f / LOG_NAME
    return None


def for_recording(recording, folders=(), own: bool | None = None, write_it: bool = True,
                  stimulus=None) -> list[dict] | None:
    """Each sweep's pattern for a loaded ``recording`` (core.recording.Recording): from its sidecar when there is
    one, else matched from the log (and the sidecar written beside it); None when there is no log or no clock.
    ``stimulus``: the light pulses found (else the recording's own), to give each pulse its pattern."""
    side = sidecar_path(recording.source)
    if side.is_file():
        try:
            return json.loads(side.read_text(encoding="utf-8"))["sweeps"]
        except (OSError, ValueError, KeyError):
            pass
    log_path = find_log(recording.source, folders)
    if log_path is None or recording.start_time is None:
        return None
    if own is None:
        own = Path(recording.source).stem[5:7] == "cw"                    # Clampwise's names: 26o15cw000
    clock = OWN_CLOCK_S if own else OTHER_CLOCK_S
    stim = stimulus if stimulus is not None else getattr(recording, "stimulus", None)
    pulses = stim.times_s if stim is not None and getattr(stim, "kind", "") == "light" and len(stim) else None
    sweeps = match(read(log_path), recording.start_time, recording.sweep_start_times_s, recording.sweep_duration_s,
                   clock, pulses)
    if write_it:
        try:
            write_sidecar(recording.source, sweeps, clock, log_path)
        except OSError:
            pass
    return sweeps


def for_sweeps(recording, sweeps: list[dict]) -> list[dict | None]:
    """The match for each of ``recording``'s sweeps by its start time (a recording of chosen sweeps keeps their
    times), None for a sweep the match does not hold."""
    by_start = {round(float(s.get("start_s", -1)), 3): s for s in sweeps}
    return [by_start.get(round(float(t), 3)) for t in np.asarray(recording.sweep_start_times_s, float)]


def by_step(sweeps: list[dict]) -> dict[str, list[int]]:
    """The sweeps (numbered from 1) of each pattern, sure ones only; ``"not sure"`` holds the rest."""
    groups: dict[str, list[int]] = {}
    for s in sweeps:
        key = (s.get("step") or "not known") if s.get("sure") else "not sure"
        groups.setdefault(key, []).append(int(s["sweep"]))
    return groups


def dose_notes(sweeps: list[dict]) -> list[str]:
    """What to keep in mind when the patterns' responses are compared: their areas, and that a full field reaches
    far more of the slice than a region on one cell."""
    areas: dict[str, tuple[float | None, float]] = {}
    for s in sweeps:
        if s.get("sure") and s.get("lit"):
            areas.setdefault(s["step"], (s.get("area_um2"), s.get("field_fraction") or 0.0))
    if len(areas) < 2:
        return []
    notes = []
    big = max(areas.items(), key=lambda kv: kv[1][1])
    for name, (a, frac) in areas.items():
        if name == big[0] or not frac:
            continue
        ratio = big[1][1] / frac
        if ratio >= 2:
            what = f" ({big[1][0]:.0f} vs {a:.0f} um^2)" if big[1][0] and a else ""
            notes.append(f"{big[0]} lights {ratio:.0f}x the area of {name}{what}: at one LED command the irradiance "
                         f"is the same, the power on the slice {ratio:.0f}x, and light scattered from it reaches "
                         "neighbours and dendrites the region spares.")
    return notes
