"""Any graph of the Figures screen as an editable figure. What the graph drew
(its lines, points, bars, error bars, reference lines and axis titles, or its
heat map) is read back into a figure-builder recipe, so every graph opens in
the builder through one path, with nothing written per graph.

Named series keep their names; unnamed traces drawn alike (a step family's
first spikes, the events behind their average) become one group drawn trace by
trace; error bars join the series they sit on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from ..figure.recipe import Recipe, Spec, pack

LINES = {QtCore.Qt.SolidLine: "-", QtCore.Qt.DashLine: "--", QtCore.Qt.DotLine: ":", QtCore.Qt.DashDotLine: "-.",
         QtCore.Qt.NoPen: ""}
SYMBOLS = {"o": "o", "s": "s", "t": "v", "t1": "^", "d": "D", "p": "p", "h": "h", "star": "*", "x": "x", "+": "+"}


def _look(pen, symbol, brush) -> dict:
    pen = pg.mkPen(pen) if pen is not None else pg.mkPen(None)
    line = LINES.get(pen.style(), "-") if pen.style() != QtCore.Qt.NoPen else ""
    marker = SYMBOLS.get(symbol, "o") if symbol else "none"
    colour = pen.color() if line else pg.mkBrush(brush).color()
    return {"colour": colour.name().upper(), "line": line, "marker": marker,
            "width": round(max(0.3, pen.widthF() or 1.0), 2)}


def _named(x: str, y: str) -> tuple[str, str]:
    return (x or "X"), (y if y and y != x else f"{y or 'Y'} ")


def _y_name(left: str, title: str) -> str:
    """The y column's name: the axis title, or the graph's subject with the axis's units ("R peak (MOhm)");
    "Value (mV)" when the graph shows several measures."""
    subject = title.split(" vs.")[0].strip()
    if not subject or " and " in subject:
        subject = "Value"
    return f"{subject} ({left})" if left and " " not in left and "(" not in left else (left or subject)


def _traces(item) -> list[dict]:
    """What a plot item drew, one dict per drawn thing: x, y, name, look, and its kind."""
    out = []
    for it in item.items:
        if isinstance(it, pg.PlotDataItem):
            x, y = it.getData()
            if x is None or not len(x):
                continue
            o = it.opts
            out.append({"kind": "line", "x": np.asarray(x, float), "y": np.asarray(y, float), "name": it.name() or "",
                        "look": _look(o.get("pen"), o.get("symbol"), o.get("symbolBrush"))})
        elif isinstance(it, pg.ScatterPlotItem):
            x, y = it.getData()
            if x is None or not len(x):
                continue
            look = _look(None, it.opts.get("symbol") or "o", it.opts.get("brush"))
            labels = [d for d in it.data["data"]] if all(isinstance(d, str) for d in it.data["data"]) else []
            out.append({"kind": "line", "x": np.asarray(x, float), "y": np.asarray(y, float),
                        "name": it.name() or "", "look": look, "labels": labels})
        elif isinstance(it, pg.BarGraphItem):
            o = it.opts
            h = np.asarray(o.get("height"), float)
            if o.get("x0") is not None and o.get("x1") is not None:
                x = (np.asarray(o["x0"], float) + np.asarray(o["x1"], float)) / 2
            else:
                x = np.asarray(o.get("x"), float)
            look = _look(None, None, o.get("brush"))
            out.append({"kind": "bars", "x": x, "y": h, "name": it.name() if hasattr(it, "name") else "",
                        "look": {**look, "marker": "none", "bars": True}})
        elif isinstance(it, pg.ErrorBarItem):
            o = it.opts
            if o.get("x") is None or o.get("y") is None:
                continue
            height = o.get("height")
            err = np.asarray(height, float) / 2 if height is not None else \
                (np.asarray(o.get("top", 0), float) + np.asarray(o.get("bottom", 0), float)) / 2
            out.append({"kind": "error", "x": np.asarray(o["x"], float), "y": np.asarray(o["y"], float),
                        "err": np.broadcast_to(err, np.shape(o["x"])).astype(float),
                        "look": _look(o.get("pen"), None, None)})
    return out


def _attach_errors(drawn: list[dict]) -> None:
    """Each error bar joins the series drawn through the same points; one alone is its own series."""
    for e in [d for d in drawn if d["kind"] == "error"]:
        host = next((d for d in drawn if d["kind"] == "line" and d["x"].shape == e["x"].shape and "err" not in d
                     and np.allclose(d["x"], e["x"], equal_nan=True) and np.allclose(d["y"], e["y"], equal_nan=True)),
                    None)
        if host is not None:
            host["err"] = e["err"]
            drawn.remove(e)
        else:
            e.update(kind="line", name="", look={**e["look"], "marker": "none", "line": ""})


def _references(item) -> list[dict]:
    out = []
    for it in item.items:
        if isinstance(it, pg.InfiniteLine):
            text = it.label.format if getattr(it, "label", None) is not None else ""
            out.append({"axis": "x" if abs(it.angle % 180 - 90) < 1 else "y", "at": float(it.value()),
                        "label": text or ""})
    return out


def from_plot(item, title: str, source: str = "") -> Spec | None:
    """The graph drawn on pyqtgraph PlotItem ``item`` as a one-panel figure, or None when nothing is drawn."""
    drawn = _traces(item)
    _attach_errors(drawn)
    drawn = [d for d in drawn if d["kind"] in ("line", "bars")]
    if not drawn:
        return None
    x_name, y_name = _named(item.getAxis("bottom").labelText, _y_name(item.getAxis("left").labelText, title))
    # unnamed traces drawn alike are one group, drawn trace by trace
    names: dict[tuple, str] = {}
    series: dict[str, dict] = {}
    rows = []
    for k, d in enumerate(drawn):
        sig = (d["look"]["colour"], d["look"]["line"], d["look"]["marker"], d["kind"])
        name = d["name"] or names.setdefault(sig, f"Series {len(names) + 1}")
        series.setdefault(name, {**d["look"], **({} if d["name"] else {"legend": False})})
        frame = pd.DataFrame({x_name: d["x"], y_name: d["y"], "Series": name, "Trace": str(k + 1)})
        if "err" in d:
            frame["Error"] = d["err"]
        if d.get("labels"):
            frame["Label"] = d["labels"]
        rows.append(frame)
    frame = pd.concat(rows, ignore_index=True)
    per_series = frame.groupby("Series", sort=False)["Trace"].nunique()
    lined = any(d["look"]["line"] or d["kind"] == "bars" for d in drawn)
    r = Recipe(source={"kind": "graph", "table": title, "recording": source}, data=pack(frame),
               question="relationship", plot="line" if lined else "scatter", x=x_name, y=y_name, group="Series",
               cell="Trace" if (per_series > 1).any() else "", level="row",
               err="Error" if "Error" in frame else "", series=series,
               legend="best" if any(d["name"] for d in drawn) and len(series) > 1 else "none",
               ref_lines=_references(item))
    return Spec(panels=[r])


def from_heatmap(hm, title: str, source: str = "") -> Spec | None:
    """A heat map (``analysis.heatmaps.Heatmap``, as shown) as a one-panel heat-map figure."""
    if hm is None:
        return None
    m = np.asarray(hm.matrix, float)
    xs, ys = np.meshgrid(np.asarray(hm.x, float), np.asarray(hm.y, float))
    unit = lambda name, u: f"{name} ({u})" if u else (name or "")   # noqa: E731
    x_name, y_name = _named(unit(hm.x_label, hm.x_unit), unit(hm.y_label, hm.y_unit))
    value = hm.colour_label or "Value"
    frame = pd.DataFrame({x_name: xs.ravel(), y_name: ys.ravel(), value: m.ravel()})
    r = Recipe(source={"kind": "graph", "table": title, "recording": source}, data=pack(frame),
               question="relationship", plot="heatmap", x=x_name, y=y_name, y2=value, level="row", legend="none")
    return Spec(panels=[r])
