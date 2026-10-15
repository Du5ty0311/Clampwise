"""Where a figure's table comes from, as a pandas DataFrame whose column names
carry their units ("Amplitude (pA)"):

- the project: one table per recording type (who each recording is of, then
  every recording-level number), the cells' results, and every analysis's
  own table (steps, pulses, events, the DSI time course...) with who each row
  is of; built from the export's column definitions, so a figure and the
  workbook name a number the same way;
- a CSV or Excel file: the sheet, the header row and the rows to skip under it
  chosen (a Clampwise workbook's units row is folded into the names);
- reshaping between wide (one column per condition) and long (one row per
  value) tables.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

IDENTITY = ("File", "Animal", "Slice", "Cell", "Condition", "Age", "Sex", "Genotype", "Region", "Cell type")
CELL_ID, ANIMAL_ID = "Cell ID", "Animal ID"


UNIT_WORDS = {"min": "min", "mv": "mV", "ms": "ms", "pa": "pA", "pct": "%", "s": "s", "hz": "Hz", "pf": "pF",
              "na": "nA", "mohm": "MOhm"}


def _named(title: str, units: str, signal: str = "") -> str:
    """"Title (units)": a unit word the title repeats is dropped ("Time min (min)" ->
    "Time (min)"), and the recording's own units replace "signal units"."""
    if signal:
        vc = signal.lower() in ("pa", "na")
        units = units.replace("signal units", signal).replace("pA or mV", signal) \
            .replace("charge units", "pC" if vc else "mV*ms")
    if units and title.endswith(f"({units})"):       # the title already says them
        return title
    words = title.split()
    if len(words) > 1 and UNIT_WORDS.get(words[-1].lower()) == units:
        title = " ".join(words[:-1])
    return f"{title} ({units})" if units else title


def _signal_units(sources) -> str:
    """The recordings' signal units when they all share one (read from each file's header)."""
    from ..io import load

    found = set()
    for src in sources:
        path = getattr(src, "path", None)
        if path is None:
            continue
        try:
            found.add(load(path).signal_units)
        except Exception:
            return ""
        if len(found) > 1:
            return ""
    return found.pop() if found else ""


def _with_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """A cell is its animal, slice and cell together; an animal its name."""
    if {"Animal", "Slice", "Cell"} <= set(frame.columns):
        parts = frame[["Animal", "Slice", "Cell"]].fillna("").astype(str)
        frame[CELL_ID] = parts.agg(" / ".join, axis=1).str.strip(" /")
        frame[ANIMAL_ID] = frame["Animal"].fillna("").astype(str)
    return frame


def project_tables(project) -> dict[str, Callable[[], pd.DataFrame]]:
    """Every table of ``project`` a figure can be made from, by name, each built when asked."""
    from .. import export
    from ..analysis import registry

    rows = export.collect(project)
    out: dict[str, Callable[[], pd.DataFrame]] = {}
    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r.recording_type, []).append(r)

    def recordings(group):
        cols = export.identity_columns() + export.value_columns(r.results for r in group)
        signal = _signal_units(group)
        frame = pd.DataFrame({_named(c.title, c.units, signal): [c.get(r) for r in group] for c in cols
                              if c.title not in ("Choices", "Flags")})
        return _with_ids(frame)

    for t in registry.TYPES:
        if by_type.get(t):
            out[f"Recordings: {registry.TYPE_WORDS.get(t, t)}"] = lambda g=by_type[t]: recordings(g)
    cells = export.cell_results(project, rows)
    for key, name, cols, body in export.tables(rows, cells):
        a = registry.load()[key]

        def table(cols=cols, body=body, key=key):
            frame: dict[str, list] = {c: [] for c in IDENTITY}
            for src, _ in body:                 # a recording (its identity), or a cell (its key as File)
                who, meta = getattr(src, "identity", None), getattr(src, "metadata", None)
                for c in IDENTITY:
                    attr = c.lower().replace(" ", "_")
                    v = src.name if c == "File" else getattr(who, attr, None) if hasattr(who, attr) else \
                        getattr(meta, attr, None)
                    frame[c].append(v or None)
            signal = _signal_units({id(src): src for src, _ in body}.values())
            for c in cols:
                o = registry.column(key, c)
                title = _named(o.title if o else c.replace("_", " ").capitalize(), o.units if o else "", signal)
                frame[title] = [export.plain(row.get(c)) for _, row in body]
            f = pd.DataFrame(frame).dropna(axis=1, how="all")
            return _with_ids(f)

        out[f"{a.label}: {name}"] = table
    return out


# --------------------------------------------------------------------------- files


def sheets(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return list(pd.ExcelFile(path).sheet_names)
    return []


def sniff(path: str | Path, sheet: str | None = None) -> tuple[int, int]:
    """(header row, rows to skip under it), counted from 0. A Clampwise workbook
    has a group row, the names, then the units: header 1, units folded in.
    Otherwise the names are the first row that is mostly words with numbers
    under it (a sheet often starts with a title and blank rows)."""
    path = Path(path)
    excel = path.suffix.lower() in (".xlsx", ".xlsm", ".xls")
    if excel and "Definitions" in sheets(path) and sheet not in ("Definitions", "Provenance"):
        return 1, 1
    try:
        raw = pd.read_excel(path, sheet_name=sheet or 0, header=None, nrows=30) if excel else \
            pd.read_csv(path, header=None, sep=None, engine="python", nrows=30, encoding="utf-8-sig")
    except Exception:
        return 0, 0
    filled = raw.notna().sum(axis=1)
    if not len(filled) or filled.max() == 0:
        return 0, 0
    for k in range(len(raw) - 1):
        row, below = raw.iloc[k], raw.iloc[k + 1]
        words = sum(isinstance(v, str) and not _number(v) for v in row if pd.notna(v))
        numbers = sum(_number(v) for v in below if pd.notna(v))
        if filled.iloc[k] >= 0.6 * filled.max() and words >= 0.6 * filled.iloc[k] and numbers >= 1:
            return k, 0
    return 0, 0


def _number(v) -> bool:
    try:
        float(str(v).replace(",", "."))
        return True
    except ValueError:
        return False


def read_table(path: str | Path, sheet: str | None = None, header: int | None = None,
               skip: int | None = None) -> pd.DataFrame:
    """A CSV, TSV or Excel sheet as a table; the units row of a Clampwise workbook
    is folded into the column names."""
    path = Path(path)
    h0, s0 = sniff(path, sheet)
    header = h0 if header is None else header
    skip = s0 if skip is None else skip
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        raw = pd.read_excel(path, sheet_name=sheet or 0, header=None)
    else:
        raw = pd.read_csv(path, header=None, sep=None, engine="python", encoding="utf-8-sig")
    names = [str(v).strip() if pd.notna(v) else f"Column {k + 1}" for k, v in enumerate(raw.iloc[header])]
    if skip == 1 and h0 == 1:                         # the units row
        units = raw.iloc[header + 1]
        names = [_named(n, str(u).strip()) if pd.notna(u) and str(u).strip() else n for n, u in zip(names, units)]
    body = raw.iloc[header + 1 + skip:].reset_index(drop=True)
    body.columns = _unique(names)
    body = body.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return typed(body)


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        k = seen.get(n, 0)
        out.append(n if not k else f"{n} ({k + 1})")
        seen[n] = k + 1
    return out


def typed(frame: pd.DataFrame) -> pd.DataFrame:
    """Columns that are numbers become numbers; the rest stay text."""
    out = frame.copy()
    for c in out.columns:
        num = pd.to_numeric(out[c], errors="coerce")
        given = out[c].notna() & (out[c].astype(str).str.strip() != "")
        if given.any() and num[given].notna().mean() >= 0.95:
            out[c] = num
        else:
            out[c] = out[c].where(out[c].isna(), out[c].astype(str))
    return out


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and frame[c].notna().any()]


def category_columns(frame: pd.DataFrame, most: int = 40) -> list[str]:
    """Columns that can name groups: text, or numbers with few values."""
    return [c for c in frame.columns if frame[c].notna().any() and frame[c].nunique() <= most
            and (not pd.api.types.is_numeric_dtype(frame[c]) or frame[c].nunique() <= 12)]


def guess(frame: pd.DataFrame) -> dict[str, str]:
    """Columns that look like the cell, the animal, the groups and time, by their names."""
    def find(*patterns, among=None):
        for p in patterns:
            for c in among if among is not None else frame.columns:
                if re.search(p, str(c), re.I):
                    return c
        return ""
    nums = numeric_columns(frame)
    return {"cell": CELL_ID if CELL_ID in frame else find(r"^cell\s*(id)?$", r"cell\s*#", r"\bcell\b(?!\s*type)"),
            "animal": ANIMAL_ID if ANIMAL_ID in frame else find(r"^animal|mouse|^rat\b|subject"),
            "group": find(r"^condition", r"group|treatment|genotype", among=category_columns(frame)),
            "x": find(r"^time\b", r"\btime (from|in|after)", r"minute|^min\b", r"^pulse", among=nums)}


def _block(column: str) -> int:
    """Which side-by-side block a column is in: repeated names were read as "Name (2)"."""
    m = re.search(r" \((\d+)\)$", str(column))
    return int(m.group(1)) if m else 1


def _base(column: str) -> str:
    return re.sub(r" \((\d+)\)$", "", str(column))


def to_long(frame: pd.DataFrame, values: list[str], name: str = "Condition", value: str = "Value",
            labels: list[str] | None = None) -> pd.DataFrame:
    """Conditions in separate columns -> one column of values and one naming the
    condition. Each value column keeps the other columns of its own block, so two
    tables side by side (the same headers twice, read as "Subject" and "Subject
    (2)") are stacked, each row with its own subject and cell."""
    labels = list(labels or []) + [str(v) for v in values][len(labels or []):]
    parts = []
    for v, lab in zip(values, labels):
        ids = [c for c in frame.columns if c not in values and _block(c) == _block(v)]
        part = frame[ids + [v]].rename(columns={c: _base(c) for c in ids}).rename(columns={v: value})
        part.insert(0, name, lab)
        parts.append(part.dropna(subset=[value]))
    out = pd.concat(parts, ignore_index=True)
    if len(out.columns) == 2:                       # nothing else to tell the rows apart
        out.insert(0, "Row", np.arange(len(out)) + 1)
    return out


def to_wide(frame: pd.DataFrame, index: list[str], columns: str, values: str) -> pd.DataFrame:
    """Long to wide: one column per value of ``columns`` (the mean where a cell repeats)."""
    return frame.pivot_table(index=index, columns=columns, values=values, aggfunc="mean").reset_index()
