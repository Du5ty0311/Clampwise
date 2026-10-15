"""From a table and a recipe to what a panel draws, in B12's order of steps:
which rows (filters) -> which time window -> baseline -> bins -> one value
per cell -> the group centre and spread. Each step that changes the numbers
says so in the caption ("What this figure shows"), and the warnings say when
the figure would mislead (Weissgerber et al. 2015; Aarts et al. 2014; Cumming
2014).

The prepared frame has standard columns: x, y, y2, group, cell, animal,
panel (and rows, the table rows behind each point).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from .recipe import PLOTS, SUMMARIES, Recipe

OPERATORS = ("is", "is not", "<", "<=", ">", ">=", "contains", "has a value")
SUMMARY_WORDS = re.compile(r"^\s*(average|mean|st\.?\s*dev|stdev|sd|sem|s\.e\.m\.?|n|count|total|sum)\s*$", re.I)
ALL = "all"


@dataclass
class Prepared:
    points: pd.DataFrame                  # one row per point drawn
    summary: pd.DataFrame                 # group (and x) -> centre, lo, hi, n
    rows: pd.DataFrame                    # the table rows kept, before rolling up (for SuperPlots and n)
    level: str                            # "cell" or "row"
    caption: str = ""
    warnings: list[str] = field(default_factory=list)
    counts: dict = field(default_factory=dict)       # group -> (cells, animals, rows)


def _filter(frame: pd.DataFrame, filters: list) -> tuple[pd.DataFrame, list[str]]:
    said = []
    for col, op, value in filters:
        if col not in frame:
            continue
        s = frame[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        try:
            v = float(value) if numeric and op != "contains" else str(value)
        except ValueError:
            v = str(value)
        text = s.astype(str)
        keep = {"is": (s == v) if numeric else (text == v), "is not": (s != v) if numeric else (text != v),
                "<": s < v if numeric else text < v, "<=": s <= v if numeric else text <= v,
                ">": s > v if numeric else text > v, ">=": s >= v if numeric else text >= v,
                "contains": text.str.contains(str(value), case=False, regex=False),
                "has a value": s.notna() & (text.str.strip() != "")}[op]
        frame = frame[keep.fillna(False)]
        said.append(f"{col} {op}" + ("" if op == "has a value" else f" {value}"))
    return frame, said


def _categories(col: pd.Series) -> pd.Series:
    """A column's values as category names: whole numbers without their ".0"."""
    num = pd.to_numeric(col, errors="coerce")
    whole = num.notna() & (num == num.round())
    out = col.astype(str).where(col.notna(), "")
    out[whole] = num[whole].astype("int64").astype(str)
    return out


def _counts(rows: pd.DataFrame, r: Recipe, has_cells: bool, said: list, warn: list) -> "Prepared":
    """How many cells (or rows) fall in each category of x, per group; and each one's % of its group."""
    by = ["panel", "group", "xcat"]
    level = "cell" if (r.level == "cell" and has_cells) else "row"
    n = rows.groupby(by, sort=False)["cell"].nunique() if level == "cell" else rows.groupby(by, sort=False).size()
    points = n.rename("count").reset_index()
    totals = points.groupby(["panel", "group"])["count"].transform("sum")
    points["pct"] = 100.0 * points["count"] / totals
    points["y"] = points["pct"] if r.percent else points["count"]
    points["rows"] = rows.groupby(by, sort=False).size().values
    counts = {g: (int(sub["cell"].nunique()) if has_cells else 0,
                  int(sub["animal"][sub["animal"] != ""].nunique()) if r.animal else 0, len(sub))
              for g, sub in rows.groupby("group", sort=False)}
    what = "cells" if level == "cell" else "rows"
    if level == "row" and has_cells and rows.groupby(["group", "cell"]).size().max() > 1:
        warn.append("Counting rows, but cells give several rows: the counts are of rows (events, sweeps), not "
                    "cells. Choose 'Each point is: a cell' to count each cell once.")
    totals_said = "; ".join(f"{'' if g == ALL else g + ': '}{int(t)} {what}"
                            for g, t in points.groupby("group", sort=False)["count"].sum().items())
    caption = (f"{PLOTS.get(r.plot, ('', ''))[0]}. "
               + (f"Each {what[:-1]}'s share of its group (%) " if r.percent else f"How many {what} ")
               + f"in each category of {r.x}. Totals: {totals_said}. " + " ".join(said))
    return Prepared(points, pd.DataFrame(columns=["panel", "group", "centre", "lo", "hi", "n"]), rows, level,
                    caption.strip(), warn, counts)


def centre_spread(values, summary: str) -> tuple[float, float, float, int]:
    """(centre, low, high, n) of ``values`` by the chosen summary."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    if summary == "median_iqr":
        q1, m, q3 = np.percentile(v, [25, 50, 75])
        return float(m), float(q1), float(q3), n
    m = float(v.mean())
    if n < 2:
        return m, m, m, n
    sd = float(v.std(ddof=1))
    half = {"mean_sd": sd, "mean_sem": sd / np.sqrt(n),
            "mean_ci95": sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n)}[summary]
    return m, m - half, m + half, n


def _summarize(points: pd.DataFrame, by: list[str], summary: str, col: str = "y") -> pd.DataFrame:
    rows = []
    for key, sub in points.groupby(by, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        c, lo, hi, n = centre_spread(sub[col], summary)
        rows.append({**dict(zip(by, key)), "centre": c, "lo": lo, "hi": hi, "n": n})
    out = pd.DataFrame(rows, columns=by + ["centre", "lo", "hi", "n"])
    return out.sort_values(by).reset_index(drop=True) if "x" in by else out


def prepare(frame: pd.DataFrame, r: Recipe) -> Prepared:
    said: list[str] = []
    warn: list[str] = []
    frame, kept = _filter(frame, r.filters)
    if kept:
        said.append("Rows kept where " + " and ".join(kept) + ".")
    texts = frame.select_dtypes(exclude="number")
    summary_rows = texts.apply(lambda c: c.astype(str).str.match(SUMMARY_WORDS)).any(axis=1) if len(texts.columns) \
        else pd.Series(False, index=frame.index)
    if summary_rows.any():                          # a spreadsheet's own AVERAGE, SEM, n rows
        warn.append(f"{int(summary_rows.sum())} row(s) look like a spreadsheet's summaries (AVERAGE, STDEV, SEM, "
                    "n...), not measurements: leave them out with 'Keep only rows where' (a column that names the "
                    "cell, 'has a value').")
    std = pd.DataFrame(index=frame.index)
    for role in ("x", "y", "y2"):
        col = getattr(r, role)
        std[role] = pd.to_numeric(frame[col], errors="coerce") if col in frame else np.nan
    for role in ("group", "cell", "animal", "panel_by"):
        col = getattr(r, role)
        std["panel" if role == "panel_by" else role] = frame[col].astype(str).where(frame[col].notna(), "") \
            if col in frame else ""
    std["xcat"] = _categories(frame[r.x]) if r.x in frame else ""      # x as categories (counts, grouped bars)
    std["size"] = pd.to_numeric(frame[r.size_by], errors="coerce") if r.size_by in frame else np.nan
    std["err"] = pd.to_numeric(frame[r.err], errors="coerce").abs() if r.err in frame else np.nan
    if not r.group:
        std["group"] = ALL
    hidden = [g for g, look in (r.series or {}).items() if look.get("hidden")]
    if hidden:                                      # a group hidden from the figure is left out of its numbers too
        std = std[~std["group"].isin(hidden)]
        said.append("Not shown: " + ", ".join(hidden) + ".")
    has_cells = bool(r.cell) and r.cell in frame
    if has_cells and r.animal and r.animal in frame:     # a cell's name is unique only within its animal
        std["cell"] = std["animal"] + " / " + std["cell"]
    key_cols = ["panel", "group", "cell"] if has_cells else ["panel", "group"]
    q = r.question
    unit = re.search(r"\(([^)]*)\)$", r.x or "")
    xu = f" {unit.group(1)}" if unit else ""          # the x column's units, for the caption
    if q in ("time", "pulses", "relationship") and r.window and len(r.window) == 2:
        a, b = r.window
        std = std[(std.x >= a) & (std.x <= b)]
        said.append(f"X from {a:g} to {b:g}{xu}.")
    keep_gaps = {"example": ["x"], "relationship": ["x", "y"]}            # a blanked artifact stays a gap in a trace
    if q == "counts":
        return _counts(std[std.xcat != ""].copy(), r, has_cells, said, warn)
    rows = std.dropna(subset=keep_gaps.get(q, ["y"])).copy()
    if q == "pulses" and len(rows):
        first = rows.x.min()
        p1 = rows[rows.x == first].groupby(key_cols)["y"].mean().rename("p1")
        rows = rows.join(p1, on=key_cols)
        rows["y"] = rows["y"] / rows["p1"]
        said.append(f"Each pulse's value is over pulse {first:g} of the same {'cell' if has_cells else 'group'}.")
    if q == "time" and r.normalize and len(r.normalize) == 2 and len(rows):
        a, b = r.normalize
        base = rows[(rows.x >= a) & (rows.x < b)].groupby(key_cols)["y"].mean().rename("base")   # [a, b): the first 10 min
        rows = rows.join(base, on=key_cols)
        missing = rows["base"].isna() | (rows["base"] == 0)
        if missing.any():
            warn.append(f"{rows.loc[missing, 'cell' if has_cells else 'group'].nunique()} "
                        f"{'cell(s)' if has_cells else 'group(s)'} have no values from {a:g} to {b:g} and are "
                        "left out of the normalized figure.")
        rows = rows[~missing]
        rows["y"] = 100.0 * rows["y"] / rows["base"]
        said.append(f"Y is % of each {'cell' if has_cells else 'group'}'s own mean from {a:g} to {b:g}{xu}.")
    if q == "time" and r.bin and r.bin > 0 and len(rows):
        rows["x"] = (np.floor(rows.x / r.bin) + 0.5) * r.bin
        said.append(f"Values averaged into bins of {r.bin:g}{xu} (each point at its bin's centre).")
    if q == "time" and not (r.bin and r.bin > 0) and has_cells and rows["cell"].nunique() > 1 and len(rows):
        shared = rows.groupby("x")["cell"].nunique()
        if (shared > 1).mean() < 0.5:                # most x values belong to a single cell: n = 1 at each
            steps = rows.sort_values("x").groupby("cell")["x"].apply(lambda v: np.median(np.diff(v)) if len(v) > 1
                                                                     else np.nan)
            step = float(np.nanmedian(steps)) if steps.notna().any() else np.nan
            warn.append("The cells' x values do not line up, so most points stand for one cell: use 'Average into "
                        "time bins of'" + (f" (about {step:.3g}{xu}, the spacing of a cell's points)."
                                           if np.isfinite(step) else "."))
    level = "cell" if ((r.level == "cell" or r.plot == "superplot") and has_cells) else "row"
    by_x = q in ("time", "pulses") or (q == "relationship" and r.plot == "xy_error")
    if q == "compare" and r.plot == "grouped_bar":   # groups within the categories of x
        key_cols = key_cols[:2] + ["xcat"] + key_cols[2:]
    if level == "cell":
        group_by = key_cols + (["x"] if by_x else [])
        agg = {"y": "mean", "y2": "mean", "size": "mean", "err": "mean", "animal": "first"}
        if not by_x:
            agg["x"] = "mean"
        points = rows.groupby(group_by, sort=False).agg(agg).reset_index()
        points["rows"] = rows.groupby(group_by, sort=False).size().values
    else:
        points = rows.copy()
        points["rows"] = 1
        repeats = has_cells and rows.groupby(key_cols).size().max() > 1 and q in ("compare", "time", "pulses")
        if repeats:
            warn.append("Each point is a row, but some cells give several rows: the spread and n then count rows "
                        "(events, sweeps), not cells, and make the effect look surer than it is (pseudoreplication; "
                        "Aarts et al. 2014). Choose 'Each point is: a cell'.")
    extra = ["x"] if by_x else (["xcat"] if "xcat" in key_cols else [])
    summary = _summarize(points, ["panel", "group"] + extra, r.summary) if len(points) else \
        pd.DataFrame(columns=["panel", "group", "centre", "lo", "hi", "n"])
    counts = {}
    for g, sub in rows.groupby("group", sort=False):
        counts[g] = (sub["cell"].nunique() if has_cells else 0,
                     sub["animal"][sub["animal"] != ""].nunique() if r.animal else 0, len(sub))
    # the warnings when a figure would mislead
    if q in ("compare", "time", "pulses"):
        for g in counts:
            sub = points[points.group == g]
            n = sub["cell"].nunique() if level == "cell" else (sub.groupby("x").size().max() if by_x else len(sub))
            if 0 < n < 3:
                warn.append(f"{'The data' if g == ALL else g} has n = {n}: too few to show a spread.")
    if r.plot == "bar" and r.y_from_zero is False:
        warn.append("A bar's height means something only from zero: its y axis should start at zero.")
    if q == "distribution" and has_cells and level == "row" and r.group and \
            rows.groupby("group")["cell"].nunique().max() > 1:
        warn.append("Each curve pools the rows of several cells, which are not independent: compare the cells' "
                    "curves (or their medians), not pooled events (Aarts et al. 2014).")
    caption = _caption(r, points, summary, counts, level, has_cells, said, by_x)
    return Prepared(points.reset_index(drop=True), summary, rows, level, caption, warn, counts)


def _caption(r: Recipe, points, summary, counts, level, has_cells, said, by_x) -> str:
    """The figure legend's text, in the order a reader needs it."""
    parts = []
    plot = PLOTS.get(r.plot, ("", ""))[0]
    if r.question == "example":
        return " ".join([f"{plot}."] + list(r.notes) + (["Scale bars as labelled."] if r.scale_bars else []))
    if r.question == "distribution":
        per_cell = has_cells and (points.cell != "").any()
        parts.append(f"{plot}. " + ("Thin curves are single cells; the thick curve is the mean of its group's cell "
                                    "curves." if per_cell else "One curve per group."))
    elif r.plot == "superplot":
        parts.append(f"{plot}. Small dots are single rows, coloured by cell; large dots are each cell's mean.")
    elif r.plot == "xy_error":
        parts.append(f"{plot}. At each x, the centre and spread of the " +
                     ("cells (each the mean of its rows there)." if level == "cell" else "rows there (one per cell)."))
    elif by_x:
        parts.append(f"{plot}. " + ("Each cell gives one value at each x (the mean of its rows there)."
                                    if level == "cell" else "Each point is one row of the table."))
    else:
        parts.append(f"{plot}. Each point is " + ("one cell (the mean of its rows)." if level == "cell"
                                                  else "one row of the table."))
    if counts:
        ns = []
        for g, (cells, animals, n_rows) in counts.items():
            name = "" if g == ALL else f"{g}: "
            if has_cells:
                ns.append(f"{name}n = {cells} cell{'s' if cells != 1 else ''}" +
                          (f" from {animals} animal{'s' if animals != 1 else ''}" if animals else "") +
                          (f" ({n_rows} rows)" if level == "cell" and n_rows > cells else ""))
            else:
                ns.append(f"{name}n = {n_rows}")
        parts.append("; ".join(ns) + ".")
    if r.question in ("compare", "time", "pulses") or r.plot == "xy_error":
        name, meaning = SUMMARIES[r.summary]
        across = "cells" if level == "cell" else "rows"
        shape = {"time_course": "The line and shading show", "lines": "The thick line shows",
                 "profile": "The thick line and its bars show", "bar": "Bars and error bars show"}.get(
            r.plot, "Black marks and bars show")
        if r.plot == "box":
            parts.append("Boxes show the median and quartiles, whiskers the range within 1.5 IQR.")
        else:
            parts.append(f"{shape} the {name[0].lower() + name[1:]} across {across}: {meaning}.")
    if r.paired:
        parts.append("Lines join the same cell across conditions.")
    parts += said + list(r.notes)
    if r.fit:
        parts.append("The curve is a least-squares fit to the points shown.")
    text = " ".join(parts)
    if r.unit and r.unit != "cell":                 # one cell's episodes, sweeps...: named as they are
        text = re.sub(r"\bcell(s?)\b", lambda m: r.unit + m.group(1), text)
    return text
