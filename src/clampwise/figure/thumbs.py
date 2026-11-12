"""A small picture of each kind of plot, to choose from: drawn by the same code that draws the figure, from a
made-up table of the shape that plot is for (cells in two conditions, a time course, pulses, two measures,
events, categories, a trace). Made once and kept; no Qt.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .recipe import PLOT_QUESTION, PLOTS, QUESTIONS, Recipe, Spec, Style, pack

_cache: dict[tuple, bytes] = {}


def _cells(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g, shift in (("control", 0.0), ("drug", 2.2)):
        for a in range(3):
            for c in range(3):
                base = 5 + shift + rng.normal(0, 0.9)
                for t in range(9):
                    rows.append({"Condition": g, "Animal": f"{g}{a}", "Cell": f"{g}{a}{c}", "Sex": "F" if a % 2 else "M",
                                 "Time": float(t), "Pulse": float(t % 5 + 1),
                                 "Value": base * (1 + 0.5 * (g == "drug") * (t > 3)) * (1 - 0.08 * (t % 5))
                                 + rng.normal(0, 0.5), "Other": base * 1.4 + rng.normal(0, 1.0),
                                 "Third": abs(rng.normal(3, 1)), "Kind": ("regular", "bursting", "fast")[(a + c + t) % 3]})
    return pd.DataFrame(rows)


def _trace() -> pd.DataFrame:
    t = np.linspace(-5, 60, 400)
    rng = np.random.default_rng(2)
    rows = []
    for name, amp in (("control", -1.0), ("drug", -0.55)):
        y = np.where(t > 2, amp * (1 - np.exp(-(t - 2) / 1.5)) * np.exp(-(t - 2) / 12), 0.0) + rng.normal(0, 0.02, t.size)
        rows.append(pd.DataFrame({"Time (ms)": t, "Current (nA)": y, "Trace": name, "Sweep": name}))
    return pd.concat(rows, ignore_index=True)


def _spikes() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for sweep in range(8):
        times = np.sort(np.r_[rng.uniform(0, 1, 3), rng.normal(0.42, 0.03, 6)])
        rows += [{"Time (s)": float(v), "Sweep": f"{sweep + 1}"} for v in times if 0 <= v <= 1]
    return pd.DataFrame(rows)


def sample(plot: str) -> Recipe:
    """A recipe that draws ``plot`` from a made-up table of the shape the plot is for."""
    q = PLOT_QUESTION.get(plot, "compare")
    if plot == "trace":
        return Recipe(data=pack(_trace()), question="example", plot="trace", x="Time (ms)", y="Current (nA)",
                      group="Trace", cell="Sweep", level="row", scale_bars=True, legend="none",
                      series={"control": {"legend": False}, "drug": {"legend": False}})
    if plot == "raster":
        return Recipe(data=pack(_spikes()), question="example", plot="raster", x="Time (s)", cell="Sweep", level="row",
                      legend="none")
    f = _cells()
    r = Recipe(data=pack(f), question=q, plot=plot, y="Value", group="Condition", cell="Cell", animal="Animal",
               legend="none")
    if q == "time":
        r.x = "Time"
    elif q == "pulses":
        r.x = "Pulse"
    elif q == "relationship":
        r.x, r.level = "Other", "cell"
        if plot == "heatmap":
            r.x, r.y, r.y2, r.level, r.group = "Time", "Pulse", "Value", "row", ""
        elif plot in ("line", "step", "xy_error"):
            r.x = "Time"
        elif plot == "bubble":
            r.size_by = "Third"
        elif plot == "hexbin":
            r.level = "row"
    elif q == "distribution":
        r.level = "row"
    elif q == "counts":
        r.x, r.y = "Kind", ""
    if plot == "proportion":
        r.event = "regular"
    if plot == "grouped_bar":
        r.x = "Sex"
    if plot == "estimation":
        r.stats = True
    return r


def png(plot: str, width_px: int = 132, height_px: int = 96) -> bytes:
    """The picture of ``plot`` as PNG bytes, without labels (the picture alone says which plot it is)."""
    key = (plot, width_px, height_px)
    if key not in _cache:
        from . import render

        dpi = 96.0
        r = sample(plot)
        spec = Spec([r], size=f"{width_px / dpi * 25.4:.2f}x{height_px / dpi * 25.4:.2f}", labels=False,
                    style=Style(font_size=4.0, marker_size=2.2, line_width=0.8, cap_size=1.0, axis_width=0.5))
        fig, _ = render.build(spec)
        for ax in fig.axes:                          # no words: only the picture
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, length=1.5)
            for t in list(ax.texts):
                t.set_visible(False)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        for t in list(fig.texts):
            t.set_visible(False)
        for ax in fig.axes:
            if ax.get_subplotspec() is not None:
                ax.set_position([0.08, 0.1, 0.86, 0.84]) if len(fig.axes) == 1 else None
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        _cache[key] = buf.getvalue()
    return _cache[key]


def catalogue() -> list[tuple[str, str, list[tuple[str, str, str]]]]:
    """(the question's key, its words, [(plot key, name, what it shows)]) for every kind of plot, grouped by what
    the figure is to show."""
    return [(key, title, [(p, PLOTS[p][0], PLOTS[p][1]) for p in plots]) for key, title, _why, _std, plots in QUESTIONS]
