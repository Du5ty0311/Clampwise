"""The cell health timeline: series resistance, holding current (or level) and
noise across a cell's files on one clock, from the stored results (nothing is
re-analyzed), with a warning wherever a measure drifts past its limit.

The limits are proposed (the user decides; they are settings of the project,
"health"): Rs changing by more than 20% of its first value (a criterion many
whole-cell studies state), the holding current moving by more than 100 pA (or
the level by 5 mV in current clamp), and the noise more than doubling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LIMITS = {"rs_change_pct": (20.0, "%", "Rs changing by more than this % of its first value"),
          "holding_change": (100.0, "pA (5 mV in current clamp)", "the holding current moving by more "
                             "than this from its first value"),
          "noise_ratio": (2.0, "x", "the noise growing to more than this many times its first value")}
MEASURES = {"rs": ("Series resistance", "MOhm"), "holding": ("Holding", ""), "noise": ("Noise SD", "")}


@dataclass
class Timeline:
    rows: pd.DataFrame          # file, minute, measure, value, source
    warnings: list[str]
    units: str                  # the signal's units (holding and noise)


def limits(project) -> dict[str, float]:
    given = project.settings().get("health", {}) if project is not None else {}
    return {k: float(given.get(k, v[0])) for k, v in LIMITS.items()}


def timeline(project, paths: list, load) -> Timeline:
    """The cell's measures over its files: each file's sweeps placed at the file's start
    time plus the sweep's own time, in minutes from the cell's first file."""
    rows = []
    units = ""
    starts = {}
    for p in paths:
        try:
            rec = load(p)
        except Exception:
            continue
        units = units or rec.signal_units
        starts[p] = rec.start_time.timestamp() if rec.start_time is not None else np.nan
    if not starts:
        return Timeline(pd.DataFrame(columns=["file", "minute", "measure", "value", "source"]), [], units)
    t0 = np.nanmin(list(starts.values()))
    for p, start in starts.items():
        run = project.last_run(p, read_file=False)
        if run is None:
            continue
        off = (start - t0) if np.isfinite(start) else 0.0
        base = run.results.get("baseline")
        for r in (base.tables.get("sweeps", []) if base is not None else []):
            m = (off + float(r.get("time_s", 0.0) or 0.0)) / 60.0
            rows.append((p.name, m, "holding", r.get("sweep_level"), "baseline before each sweep's stimulus"))
            rows.append((p.name, m, "noise", r.get("sweep_noise_sd"), "baseline noise of each sweep"))
        mt = run.results.get("membrane_test")
        if mt is not None and mt.tables.get("sweeps"):
            for r in mt.tables["sweeps"]:
                rows.append((p.name, off / 60.0 + float(r.get("time_s", 0.0) or 0.0) / 60.0, "rs", r.get("rs_mohm"),
                             "the file's test step"))
        else:
            typed = project.metadata(p).rs_typed_mohm
            book = (float("nan"), float("nan"))
            if not all(np.isfinite(v) for v in typed) and project.features().get("rig_records", True):
                from ..rig import evidence as rig_evidence

                book = rig_evidence.lab_book_rs(p)              # what was typed stays; the Lab Book fills the rest
            for v, b in zip(typed, book):
                if np.isfinite(v):
                    rows.append((p.name, off / 60.0, "rs", v, "typed from Clampex's Membrane Test"))
                elif np.isfinite(b):
                    rows.append((p.name, off / 60.0, "rs", b,
                                 "Clampex's Membrane Test, read from the acquisition's Lab Book"))
    frame = pd.DataFrame(rows, columns=["file", "minute", "measure", "value", "source"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"]).sort_values("minute").reset_index(drop=True)
    return Timeline(frame, drift(frame, limits(project), units), units)


def drift(frame: pd.DataFrame, lim: dict[str, float], units: str) -> list[str]:
    """Where each measure first goes past its limit, against its first value."""
    out = []
    vc = units.lower() in ("pa", "na")
    for measure in ("rs", "holding", "noise"):
        s = frame[frame.measure == measure]
        if len(s) < 2:
            continue
        first = float(s.value.iloc[:3].median())
        if measure == "rs":
            past = s[np.abs(s.value - first) > lim["rs_change_pct"] / 100.0 * abs(first)]
            what = f"Rs changed by more than {lim['rs_change_pct']:g}% of its first value ({first:.1f} MOhm)"
        elif measure == "holding":
            limit = lim["holding_change"] if vc else 5.0
            past = s[np.abs(s.value - first) > limit]
            what = f"the holding moved more than {limit:g} {units} from its first value ({first:.0f} {units})"
        else:
            past = s[s.value > lim["noise_ratio"] * first]
            what = f"the noise grew past {lim['noise_ratio']:g} x its first value ({first:.2g} {units})"
        if len(past):
            r = past.iloc[0]
            out.append(f"{what} at {r.minute:.1f} min ({r.file}); {len(past)} of {len(s)} points past the limit "
                       "(limits: Settings, Health)")
    return out
