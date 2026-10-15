"""Drawing a figure with matplotlib (what the builder shows is what exports)
and saving it: PDF, SVG, EPS, PNG or TIFF at a chosen resolution, a journal's
column width, fonts embedded (TrueType in PDF and EPS, text kept as text in
SVG). Every export carries the whole figure (``Spec``) inside it, so the file
reopens in the builder editable (``read_spec``).

The look comes from two places: the figure's ``Style`` (a theme: fonts, axis
lines, grid, ticks, background, colours) made into matplotlib's settings for
every panel, and each group's own look (``looks``: colour, marker, line,
width, filled or hollow, its name in the legend, its order), which every plot
type draws with.
"""

from __future__ import annotations

import base64
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import stats as fstats
from .prepare import ALL, Prepared, prepare
from .recipe import FITS, MARKERS, PALETTES, Recipe, Spec, Style

MARK_START, MARK_END = b"CLAMPWISE-FIGURE:", b":END-CLAMPWISE-FIGURE"
FORMATS = ("pdf", "svg", "eps", "png", "tiff")
COMPARE = ("points", "bar", "box", "violin", "raincloud", "dot", "hbar", "grouped_bar", "estimation", "superplot")
LINE_MARKERS = ("x", "+")                         # drawn with lines, not filled


@dataclass
class Drawn:
    recipe: Recipe
    prepared: Prepared | None
    comparisons: list
    note: str = ""            # why nothing was drawn


@dataclass
class Look:
    colour: str
    marker: str
    line: str
    width: float
    filled: bool
    label: str


def rc(style: Style) -> dict:
    """The theme as matplotlib's settings."""
    ink, bg = style.ink, style.background
    clear = bg == "transparent"
    spines = {"open": (True, True, False, False), "box": (True, True, True, True),
              "left-bottom-light": (True, True, False, False), "none": (False, False, False, False)}.get(
        style.spines, (True, True, False, False))
    ticks = style.tick_dir if style.tick_dir in ("in", "out") else "out"
    size = 0.0 if style.tick_dir == "none" else 3.0
    return {"font.family": "sans-serif", "font.sans-serif": [style.font, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": style.font_size, "axes.titlesize": style.font_size, "axes.labelsize": style.font_size,
            "xtick.labelsize": style.font_size * 0.9, "ytick.labelsize": style.font_size * 0.9,
            "legend.fontsize": style.font_size * 0.9, "lines.linewidth": style.line_width,
            "axes.linewidth": 0.5 if style.spines == "left-bottom-light" else 0.8,
            "axes.spines.left": spines[0], "axes.spines.bottom": spines[1], "axes.spines.top": spines[2],
            "axes.spines.right": spines[3], "axes.grid": style.grid != "none",
            "axes.grid.axis": style.grid if style.grid in ("x", "y") else "both", "grid.color": style.grid_colour,
            "grid.linewidth": 0.6, "axes.axisbelow": True,
            "xtick.direction": ticks, "ytick.direction": ticks, "xtick.major.size": size, "ytick.major.size": size,
            "xtick.minor.visible": style.minor_ticks, "ytick.minor.visible": style.minor_ticks,
            "text.color": ink, "axes.labelcolor": ink, "axes.edgecolor": ink, "xtick.color": ink, "ytick.color": ink,
            "axes.titleweight": "bold" if style.title_bold else "normal",
            "figure.facecolor": "none" if clear else bg, "axes.facecolor": "none" if clear else bg,
            "savefig.facecolor": "none" if clear else bg, "savefig.transparent": clear,
            "legend.frameon": style.legend_frame, "legend.edgecolor": ink,
            "legend.facecolor": "none" if clear else bg, "errorbar.capsize": style.cap_size,
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02}


def colours(style: Style, n: int) -> list[str]:
    pal = PALETTES.get(style.palette, PALETTES["okabe-ito"])
    return [pal[k % len(pal)] for k in range(max(n, 1))]


def ordered(groups: list[str], r: Recipe) -> list[str]:
    """The groups in the order asked (the rest as they come)."""
    return [g for g in r.order if g in groups] + [g for g in groups if g not in r.order]


def looks(r: Recipe, style: Style, groups: list[str]) -> dict[str, Look]:
    """Each group's look: its own where set (Groups' look), else the theme's."""
    pal = colours(style, len(groups))
    marks = [m for m in MARKERS if m != "none"]
    out = {}
    for k, g in enumerate(groups):
        d = (r.series or {}).get(g, {})
        out[g] = Look(d.get("colour") or pal[k], d.get("marker") or (marks[k % len(marks)] if style.vary_markers
                                                                      else "o"),
                      d.get("line", "-") if d.get("line") is not None else "-",
                      float(d.get("width") or style.line_width), bool(d.get("filled", True)),
                      d.get("label") or ("" if g == ALL else g))
    return out


def build(spec: Spec, fig: Figure | None = None) -> tuple[Figure, list[Drawn]]:
    """The whole figure: each panel prepared from its own data and drawn in its slot."""
    w, h = spec.size_mm()
    with plt.rc_context(rc(spec.style)):
        fig = fig or Figure()
        fig.clear()
        fig.set_size_inches(w / 25.4, h / 25.4)
        fig.set_facecolor("none" if spec.style.background == "transparent" else spec.style.background)
        fig.set_layout_engine("constrained")       # labels, titles and panel letters kept inside the figure
        cols = max(1, spec.columns)
        rows = -(-len(spec.panels) // cols)
        grid = fig.add_gridspec(rows, cols)
        out = []
        for k, r in enumerate(spec.panels):
            slot = grid[k // cols, k % cols]
            out.append(_panel(fig, slot, r, spec.style, k if spec.labels and len(spec.panels) > 1 else None))
    return fig, out


def _letter(k: int, style: str) -> str:
    a = chr(ord("A") + k)
    return {"a": a.lower(), "(A)": f"({a})", "A.": f"{a}."}.get(style, a)


def _panel(fig: Figure, slot, r: Recipe, style: Style, label_index: int | None) -> Drawn:
    frame = r.frame()
    needs_y = r.question not in ("example", "counts")
    if frame is None or (needs_y and not r.y) or (r.question == "counts" and not r.x):
        ax = fig.add_subplot(slot)
        ax.axis("off")
        ax.text(0.5, 0.5, r.source.get("needs") or "Choose a table and the columns to show", ha="center",
                va="center", transform=ax.transAxes, color="#777777", wrap=True)
        if r.title:
            ax.set_title(r.title)
        return Drawn(r, None, [], "no data")
    prep = prepare(frame, r)
    panels = [p for p in dict.fromkeys(prep.points.panel)] or [""]
    estimation = r.plot == "estimation" and r.question == "compare"
    sub = slot.subgridspec(1, len(panels) * (2 if estimation else 1),
                           width_ratios=[2, 1] * len(panels) if estimation else None)
    comparisons = []
    first = None
    for j, pv in enumerate(panels):
        share = first if first is not None and r.question not in ("counts",) else None
        ax = fig.add_subplot(sub[0, 2 * j if estimation else j], sharey=share)
        first = first or ax
        mine = Prepared(prep.points[prep.points.panel == pv], prep.summary[prep.summary.panel == pv],
                        prep.rows[prep.rows.panel == pv], prep.level, prep.caption, prep.warnings, prep.counts)
        comps = fstats.compare(mine.points, r.paired) if (r.stats or estimation) and r.question == "compare" \
            and r.plot != "grouped_bar" else []
        comparisons += comps
        _draw(ax, r, mine, style, comps)
        if pv:
            ax.set_title(f"{r.panel_by}: {pv}" if r.panel_by else pv, fontsize=style.font_size)
        if estimation:
            _estimation(fig.add_subplot(sub[0, 2 * j + 1]), r, mine, style, comps)
        if j == 0 and label_index is not None:
            ax.text(-0.28, 1.06, _letter(label_index, style.label_style), transform=ax.transAxes,
                    fontsize=style.font_size + 3, fontweight="bold", va="bottom", ha="left")
        if j == 0 and r.title:
            ax.set_title(r.title, fontsize=style.font_size, loc="left" if pv == "" else "center")
    return Drawn(r, prep, comparisons)


def _groups(points: pd.DataFrame, r: Recipe | None = None) -> list[str]:
    g = list(dict.fromkeys(points.group))
    return ordered(g, r) if r is not None else g


def _labels(ax, r: Recipe, default_x: str, default_y: str) -> None:
    ax.set_xlabel(r.x_label or default_x)
    ax.set_ylabel(r.y_label or default_y)


def _y_name(r: Recipe) -> str:
    if r.question == "pulses":
        return f"{_bare(r.y)} / pulse 1"
    if r.question == "time" and r.normalize:
        return f"{_bare(r.y)} (% of baseline)"
    return r.y


def _bare(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)$", "", name or "")


def _units(name: str) -> str:
    m = re.search(r"\(([^)]*)\)$", name or "")
    return m.group(1) if m else ""


def _jitter(n: int, seed: int, width: float = 0.12) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, n)


def _scatter(ax, x, y, look: Look, style: Style, size=None, alpha=None, zorder=3, label=None) -> None:
    """Points in a group's look: filled, hollow, or a line marker (x, +)."""
    s = style.marker_size ** 2 if size is None else size
    a = style.point_alpha if alpha is None else alpha
    if look.marker in LINE_MARKERS:
        ax.scatter(x, y, s=s, marker=look.marker, color=look.colour, linewidths=max(0.8, look.width), alpha=a,
                   zorder=zorder, label=label)
    elif look.filled:
        ax.scatter(x, y, s=s, marker=look.marker, color=look.colour, linewidths=0, alpha=a, zorder=zorder,
                   label=label)
    else:
        ax.scatter(x, y, s=s, marker=look.marker, facecolors="none", edgecolors=look.colour,
                   linewidths=max(0.6, look.width * 0.8), alpha=a, zorder=zorder, label=label)


def _spread(ax, x, s, look: Look, r: Recipe, style: Style, label=None, line=True) -> None:
    """A centre line with its spread: error bars, a band, or nothing, as asked."""
    ls = look.line or "-"
    if r.error_style == "band":
        ax.fill_between(x, s.lo, s.hi, color=look.colour, alpha=style.fill_alpha, lw=0)
    if r.error_style == "bar":
        ax.errorbar(x, s.centre, yerr=[s.centre - s.lo, s.hi - s.centre], color=look.colour, ls=ls if line else "",
                    lw=look.width * 1.5, marker=look.marker, ms=style.marker_size,
                    mfc=look.colour if look.filled else "none", mec=look.colour, label=label)
    else:
        ax.plot(x, s.centre, color=look.colour, ls=ls if line else "", lw=look.width * 1.5, marker=look.marker,
                ms=style.marker_size * 0.8, mfc=look.colour if look.filled else "none", mec=look.colour, label=label)


def _draw(ax, r: Recipe, p: Prepared, style: Style, comps: list) -> None:
    pts = p.points
    groups = _groups(pts, r)
    lk = looks(r, style, groups)
    q = r.question
    if q == "compare":
        if r.plot == "grouped_bar":
            _grouped(ax, r, p, style, groups, lk)
        elif r.plot == "hbar":
            _hbar(ax, r, p, style, groups, lk)
        else:
            _compare(ax, r, p, style, groups, lk, comps)
    elif q in ("time", "pulses"):
        for g in groups:
            s = p.summary[p.summary.group == g].sort_values("x")
            mine = pts[pts.group == g]
            look = lk[g]
            if r.plot in ("lines", "profile") and "cell" in mine and (mine.cell != "").any():
                for _, c in mine.groupby("cell"):
                    c = c.sort_values("x")
                    ax.plot(c.x, c.y, color=look.colour, alpha=0.35, lw=look.width * 0.6, ls=look.line or "-")
            if r.plot == "area":
                ax.fill_between(s.x, 0, s.centre, color=look.colour, alpha=style.fill_alpha, lw=0)
            _spread(ax, s.x, s, look, r, style, label=look.label or None)
        if q == "pulses":
            ax.axhline(1.0, color="#888888", lw=0.6, ls=":")
            xs = sorted(pts.x.dropna().unique())
            if len(xs) <= 20:
                ax.set_xticks(xs)
        if q == "time" and r.normalize:
            ax.axhline(100.0, color="#888888", lw=0.6, ls=":")
        _labels(ax, r, r.x, _y_name(r))
    elif q == "relationship":
        _relationship(ax, r, p, style, groups, lk)
    elif q == "distribution":
        _distribution(ax, r, p, style, groups, lk)
    elif q == "counts":
        _counts(ax, r, p, style, groups, lk)
    elif q == "example":
        _trace(ax, r, p, style)
    _axes(ax, r, p, style)


def _centre(p: Prepared, g: str, extra: dict | None = None) -> tuple[float, float, float]:
    s = p.summary[p.summary.group == g]
    for k, v in (extra or {}).items():
        s = s[s[k] == v]
    if not len(s):
        return np.nan, np.nan, np.nan
    return float(s.centre.iloc[0]), float(s.lo.iloc[0]), float(s.hi.iloc[0])


def _half_violin(ax, values, pos, look: Look, style: Style, side: int = 1, width: float = 0.35) -> None:
    v = ax.violinplot([values], positions=[pos], widths=2 * width, showextrema=False)
    for b in v["bodies"]:
        verts = b.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], pos, np.inf) if side > 0 else np.clip(verts[:, 0], -np.inf, pos)
        b.set_facecolor(look.colour)
        b.set_alpha(style.fill_alpha + 0.15)
        b.set_edgecolor("none")


def _compare(ax, r, p, style, groups, lk, comps) -> None:
    pts = p.points
    ms = style.marker_size
    pos = {g: i for i, g in enumerate(groups)}
    if r.plot == "superplot" and (p.rows.cell != "").any():
        cells = list(dict.fromkeys(p.rows.cell))
        ccol = dict(zip(cells, colours(style, len(cells))))
        for g in groups:
            rows = p.rows[p.rows.group == g]
            ax.scatter(pos[g] + _jitter(len(rows), 1, style.jitter * 2), rows.y, s=(ms * 0.7) ** 2,
                       c=[ccol[c] for c in rows.cell], alpha=0.45, lw=0)
            mine = pts[pts.group == g]
            ax.scatter(pos[g] + _jitter(len(mine), 2, style.jitter), mine.y, s=(ms * 1.6) ** 2,
                       c=[ccol[c] for c in mine.cell], edgecolors=style.ink, linewidths=0.6, zorder=3,
                       marker=lk[g].marker)
    for g in groups:
        mine = pts[pts.group == g]
        look = lk[g]
        values = mine.y.dropna()
        c, lo, hi = _centre(p, g)
        x = pos[g] + _jitter(len(mine), 0, style.jitter)
        if r.plot == "bar":
            ax.bar(pos[g], c, width=0.6, color=look.colour, alpha=style.fill_alpha + 0.1, edgecolor=look.colour,
                   lw=look.width)
        elif r.plot == "box" and len(values):
            props = {"color": look.colour, "linewidth": look.width}
            ax.boxplot([values], positions=[pos[g]], widths=0.5, showfliers=False, medianprops=props, boxprops=props,
                       whiskerprops=props, capprops=props)
        elif r.plot == "violin" and len(values) > 1:
            v = ax.violinplot([values], positions=[pos[g]], widths=0.7, showextrema=False)
            for b in v["bodies"]:
                b.set_facecolor(look.colour)
                b.set_alpha(style.fill_alpha)
        elif r.plot == "raincloud" and len(values) > 1:
            _half_violin(ax, values, pos[g] + 0.05, look, style, side=1)
            props = {"color": look.colour, "linewidth": look.width}
            ax.boxplot([values], positions=[pos[g] + 0.05], widths=0.08, showfliers=False, medianprops=props,
                       boxprops=props, whiskerprops=props, capprops=props)
            x = pos[g] - 0.2 + _jitter(len(mine), 0, style.jitter * 0.6)
        if r.plot != "superplot" and r.show_points:
            alpha = 0.3 if r.plot == "dot" else None
            _scatter(ax, x, mine.y, look, style, alpha=alpha)
        if r.plot not in ("box", "raincloud") and r.error_style != "none":
            at = pos[g] if r.plot == "dot" else pos[g] + 0.28
            ax.errorbar(at, c, yerr=[[c - lo], [hi - c]], color=look.colour if r.plot == "dot" else style.ink,
                        lw=1.2 if r.plot == "dot" else 1, marker="o" if r.plot == "dot" else "_",
                        ms=ms * 1.6 if r.plot == "dot" else 10, zorder=4)
    if r.paired and (pts.cell != "").any():
        for _, cpts in pts.groupby("cell"):
            if cpts.group.nunique() > 1:
                cpts = cpts.assign(pos=cpts.group.map(pos)).sort_values("pos")
                ax.plot(cpts.pos, cpts.y, color="#999999", lw=0.5, zorder=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([lk[g].label for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    _labels(ax, r, r.group, _y_name(r))
    if comps and r.stats:
        top = np.nanmax(pts.y) if len(pts) else 1.0
        span = (np.nanmax(pts.y) - np.nanmin(pts.y)) if len(pts) > 1 else abs(top) or 1.0
        for k, comp in enumerate(comps):
            if comp.reference not in pos or comp.other not in pos:
                continue
            y = top + span * (0.08 + 0.12 * k)
            a, b = pos[comp.reference], pos[comp.other]
            ax.plot([a, a, b, b], [y - span * 0.02, y, y, y - span * 0.02], color=style.ink, lw=0.7)
            ax.text((a + b) / 2, y, fstats.fmt_p(comp.p), ha="center", va="bottom", fontsize=style.font_size * 0.85)


def _hbar(ax, r, p, style, groups, lk) -> None:
    pts = p.points
    pos = {g: i for i, g in enumerate(groups)}
    for g in groups:
        look = lk[g]
        c, lo, hi = _centre(p, g)
        ax.barh(pos[g], c, height=0.6, color=look.colour, alpha=style.fill_alpha + 0.1, edgecolor=look.colour,
                lw=look.width)
        if r.error_style != "none":
            ax.errorbar(c, pos[g], xerr=[[c - lo], [hi - c]], color=style.ink, lw=1, fmt="none")
        mine = pts[pts.group == g]
        if r.show_points:
            _scatter(ax, mine.y, pos[g] + _jitter(len(mine), 0, style.jitter), look, style)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([lk[g].label for g in groups])
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.invert_yaxis()
    ax.set_ylabel(r.y_label or r.group)
    ax.set_xlabel(r.x_label or _y_name(r))


def _grouped(ax, r, p, style, groups, lk) -> None:
    """Groups side by side within each category of x."""
    pts = p.points
    cats = list(dict.fromkeys(pts.xcat))
    width = 0.8 / max(1, len(groups))
    for k, g in enumerate(groups):
        look = lk[g]
        offs = -0.4 + width * (k + 0.5)
        for i, cat in enumerate(cats):
            c, lo, hi = _centre(p, g, {"xcat": cat})
            if not np.isfinite(c):
                continue
            ax.bar(i + offs, c, width=width * 0.9, color=look.colour, alpha=style.fill_alpha + 0.15,
                   edgecolor=look.colour, lw=look.width, label=look.label if i == 0 else None)
            if r.error_style != "none":
                ax.errorbar(i + offs, c, yerr=[[c - lo], [hi - c]], color=style.ink, lw=0.9, fmt="none")
            mine = pts[(pts.group == g) & (pts.xcat == cat)]
            if r.show_points:
                _scatter(ax, i + offs + _jitter(len(mine), i, width * 0.25), mine.y, look, style)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    _labels(ax, r, r.x, _y_name(r))


def _estimation(ax, r, p, style, comps) -> None:
    """Ho et al. 2019: each group's difference from the first, its bootstrap
    distribution and 95% CI, on its own axis."""
    groups = _groups(p.points, r)
    lk = looks(r, style, groups)
    rng = np.random.default_rng(0)
    ref = p.points[p.points.group == groups[0]].y.dropna().to_numpy() if groups else np.array([])
    for k, c in enumerate(comps, start=1):
        other = p.points[p.points.group == c.other].y.dropna().to_numpy()
        if ref.size < 2 or other.size < 2:
            continue
        boots = other[rng.integers(0, other.size, (2000, other.size))].mean(1) - \
            ref[rng.integers(0, ref.size, (2000, ref.size))].mean(1)
        v = ax.violinplot([boots], positions=[k], widths=0.6, showextrema=False)
        for b in v["bodies"]:
            b.set_facecolor(lk[c.other].colour if c.other in lk else "#888888")
            b.set_alpha(0.35)
        ax.errorbar(k, c.difference, yerr=[[c.difference - c.ci[0]], [c.ci[1] - c.difference]], color=style.ink,
                    marker="o", ms=style.marker_size, capsize=0, lw=1.2)
    ax.axhline(0, color="#888888", lw=0.6, ls=":")
    ax.set_xticks(range(1, len(comps) + 1))
    ax.set_xticklabels([f"{c.other}\nminus\n{c.reference}" for c in comps], fontsize=style.font_size * 0.8)
    ax.set_xlim(0.4, len(comps) + 0.6)
    ax.set_ylabel(f"Difference ({_units(r.y)})" if _units(r.y) else "Difference")
    ax.yaxis.set_label_position("right")              # clear of the groups' axis on its left
    ax.yaxis.tick_right()
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)


def _fit(xs, ys, kind: str):
    from ..analysis.tools import fits

    x, y = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return None, ""
    order = np.argsort(x)
    x, y = x[order], y[order]
    grid = np.linspace(x.min(), x.max(), 200)
    if kind == "linear":
        f = fits.linear(x, y, purpose="figure: fitted line")
        eq = f"y = {f['slope']:.3g}x {f['intercept']:+.3g}"
        curve = f.curve(grid)
    elif kind == "exponential":
        f = fits.exp_decay(x, y, purpose="figure: fitted exponential")
        eq = f"y = {f['amp']:.3g}·e^(-(x-{x[0]:.3g})/{f['tau']:.3g}) {f['offset']:+.3g}"
        curve = f.curve(grid - x[0])
    elif kind == "boltzmann":
        f = fits.boltzmann(x, y, falling=bool(y[0] > y[-1]), purpose="figure: fitted Boltzmann")
        eq = f"Boltzmann: V½ {f['v_half']:.3g}, k {f['slope']:.3g}"
        curve = f.curve(grid)
    else:
        f = fits.hill(x, y, purpose="figure: fitted Hill")
        eq = f"Hill: EC50 {f['ec50']:.3g}, n {f['n']:.2g}"
        curve = f.curve(grid)
    if not f.ok:
        return None, f"{FITS[kind]}: the fit did not converge"
    return (grid, curve), f"{eq} (R² {f.r2:.2f})"


def regression_band(xs, ys, level: float = 0.95):
    """A straight line by least squares and the band where the true line lies with
    ``level`` confidence (the mean's confidence interval)."""
    from scipy import stats as sps

    x, y = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 200)
    fit = slope * grid + intercept
    resid = y - (slope * x + intercept)
    s = np.sqrt(np.sum(resid ** 2) / (x.size - 2))
    se = s * np.sqrt(1 / x.size + (grid - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    t = sps.t.ppf(0.5 + level / 2, x.size - 2)
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    return grid, fit, fit - t * se, fit + t * se, f"y = {slope:.3g}x {intercept:+.3g} (R² {r2:.2f}, 95% band)"


def _relationship(ax, r, p, style, groups, lk) -> None:
    pts = p.points
    if r.plot == "heatmap":
        _heatmap(ax, r, p)
        return
    if r.plot == "hexbin":
        ok = pts.dropna(subset=["x", "y"])
        hb = ax.hexbin(ok.x, ok.y, gridsize=int(np.clip(np.sqrt(len(ok)) / 1.5, 6, 40)), cmap="cividis", mincnt=1,
                       linewidths=0)
        ax.figure.colorbar(hb, ax=ax, label="Count", fraction=0.05, pad=0.03)
        _labels(ax, r, r.x, r.y)
        return
    sizes = pts["size"] if "size" in pts else pd.Series(np.nan, index=pts.index)
    smax = np.nanmax(np.abs(sizes)) if np.isfinite(sizes).any() else np.nan
    equations = []                                  # written in the corner, not in the legend: the plot keeps its room
    for g in groups:
        mine = pts[pts.group == g].sort_values("x")
        look = lk[g]
        name = None if (r.series or {}).get(g, {}).get("legend") is False else (look.label or None)
        if r.plot == "xy_error":
            s = p.summary[p.summary.group == g].sort_values("x")
            _spread(ax, s.x, s, look, r, style, label=name)
        elif r.plot == "line":
            _lines(ax, pts[pts.group == g], look, style, r, name)
        elif r.plot == "step":
            ax.step(mine.x, mine.y, where="mid", color=look.colour, ls=look.line or "-", lw=look.width, label=name)
        elif r.plot == "bubble" and np.isfinite(smax):
            area = (0.5 + 6.0 * np.abs(mine["size"].to_numpy(float)) / smax) * style.marker_size ** 2
            _scatter(ax, mine.x, mine.y, look, style, size=np.nan_to_num(area, nan=20), alpha=0.6, label=name)
        else:
            _scatter(ax, mine.x, mine.y, look, style, label=name)
        if r.plot in ("scatter", "line", "regression", "bubble") and "err" in mine and mine.err.notna().any():
            ok = mine.dropna(subset=["err"])
            ax.errorbar(ok.x, ok.y, yerr=ok.err, fmt="none", ecolor=look.colour, elinewidth=look.width,
                        capsize=style.cap_size, zorder=2)
        if r.plot == "regression":
            band = regression_band(mine.x, mine.y)
            if band is not None:
                grid, fit, lo, hi, eq = band
                ax.fill_between(grid, lo, hi, color=look.colour, alpha=style.fill_alpha, lw=0)
                ax.plot(grid, fit, color=look.colour, lw=look.width * 1.3, ls=look.line or "-")
                equations.append((eq, look.colour))
        elif r.fit:
            drawn, eq = _fit(mine.x, mine.y, r.fit)
            if drawn is not None:
                ax.plot(*drawn, color=look.colour, lw=look.width * 1.2, ls="--")
            if eq:
                equations.append((eq, look.colour))
    for k, (eq, c) in enumerate(equations):
        ax.text(0.02, 0.98 - 0.07 * k, eq, transform=ax.transAxes, va="top", ha="left", color=c,
                fontsize=style.font_size * 0.75).set_in_layout(False)      # inside the plot: no room taken
    if r.y2:
        twin = ax.twinx()
        twin.spines["right"].set_visible(True)
        m = pts.groupby("x")["y2"].mean().dropna()
        twin.plot(m.index, m.values, color="#777777", lw=style.line_width, ls="-.", label=r.y2)
        twin.set_ylabel(r.y2, color="#555555")
    _labels(ax, r, r.x, r.y)


def _lines(ax, mine, look: Look, style: Style, r: Recipe, name) -> None:
    """A group's points joined: one line per cell when cells are named (each trace of a group drawn on its
    own, in the table's order, named once), else in x order; or bars when the group's look asks for them."""
    if (r.series or {}).get(look_key(mine), {}).get("bars"):
        xs = np.sort(mine.x.dropna().unique())
        width = float(np.median(np.diff(xs))) * 0.9 if xs.size > 1 else 1.0
        ax.bar(mine.x, mine.y, width=width, color=look.colour, alpha=0.85, lw=0, label=name)
        return
    ls = look.line if look.line else "none"
    marker = None if look.marker == "none" else look.marker
    # a named cell's trace keeps the table's order (a phase plot loops back); otherwise points join in x order
    parts = [c for _, c in mine.groupby("cell", sort=False)] if r.cell and "cell" in mine else [mine.sort_values("x")]
    for k, c in enumerate(parts):
        ax.plot(c.x, c.y, color=look.colour, ls=ls, lw=look.width, marker=marker, ms=style.marker_size * 0.8,
                mfc=look.colour if look.filled else "none", mec=look.colour, label=name if k == 0 else None)


def look_key(mine: pd.DataFrame) -> str:
    return str(mine.group.iloc[0]) if len(mine) else ""


def _heatmap(ax, r, p) -> None:
    from ..analysis import heatmaps

    rows = p.rows.dropna(subset=["x", "y", "y2"])
    if not len(rows):
        return
    m = rows.pivot_table(index="y", columns="x", values="y2", aggfunc="mean")
    hm = heatmaps.Heatmap(m.to_numpy(float), m.columns.to_numpy(float), m.index.to_numpy(float), "", "", "", "",
                          "", r.y2, "")
    lo, hi = heatmaps.levels(hm)
    x0, x1, y0, y1 = hm.extent()
    im = ax.imshow(np.ma.masked_invalid(hm.matrix), origin="lower", aspect="auto", extent=(x0, x1, y0, y1),
                   cmap="cividis", vmin=lo, vmax=hi, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, label=r.y2, fraction=0.05, pad=0.03)
    _labels(ax, r, r.x, r.y)


def _distribution(ax, r, p, style, groups, lk) -> None:
    rows = p.rows.dropna(subset=["y"])
    if not len(rows):
        return
    if r.plot == "histogram":
        bins = np.histogram_bin_edges(rows.y, bins="auto")
        for g in groups:
            look = lk[g]
            ax.hist(rows[rows.group == g].y, bins=bins, histtype="stepfilled", color=look.colour,
                    alpha=style.fill_alpha, lw=0)
            ax.hist(rows[rows.group == g].y, bins=bins, histtype="step", color=look.colour, lw=look.width * 1.3,
                    ls=look.line or "-", label=look.label or None)
        _labels(ax, r, r.y, "Count")
        return
    if r.plot == "density":
        from scipy.stats import gaussian_kde

        grid = np.linspace(rows.y.min(), rows.y.max(), 300)
        for g in groups:
            v = rows[rows.group == g].y.to_numpy(float)
            if v.size > 2 and np.ptp(v) > 0:
                d = gaussian_kde(v)(grid)
                look = lk[g]
                ax.fill_between(grid, 0, d, color=look.colour, alpha=style.fill_alpha, lw=0)
                ax.plot(grid, d, color=look.colour, lw=look.width * 1.4, ls=look.line or "-", label=look.label or None)
        _labels(ax, r, r.y, "Density")
        return
    if r.plot == "qq":
        from scipy import stats as sps

        for g in groups:
            v = rows[rows.group == g].y.to_numpy(float)
            if v.size > 2:
                (theo, ordered_v), (slope, intercept, _r) = sps.probplot(v, dist="norm")
                look = lk[g]
                _scatter(ax, theo, ordered_v, look, style, label=look.label or None)
                ax.plot(theo, slope * theo + intercept, color=look.colour, lw=look.width, ls="--")
        _labels(ax, r, "Normal quantiles", f"{r.y} (ordered)")
        return
    grid = np.linspace(rows.y.min(), rows.y.max(), 300)
    for g in groups:
        mine = rows[rows.group == g]
        look = lk[g]
        cells = [c for _, c in mine.groupby("cell")] if (mine.cell != "").any() else [mine]
        curves = []
        for c in cells:
            v = np.sort(c.y.to_numpy(float))
            cdf = np.searchsorted(v, grid, side="right") / v.size
            curves.append(cdf)
            if len(cells) > 1:
                ax.plot(grid, cdf, color=look.colour, alpha=0.3, lw=look.width * 0.6)
        ax.plot(grid, np.mean(curves, axis=0), color=look.colour, lw=look.width * 1.6, ls=look.line or "-",
                label=(look.label or None) if len(cells) == 1 else f"{look.label + ': ' if look.label else ''}mean "
                f"of {len(cells)} cells")
    _labels(ax, r, r.y, "Cumulative fraction")
    ax.set_ylim(0, 1.02)


def _counts(ax, r, p, style, groups, lk) -> None:
    """Counts or percentages per category: bars (groups side by side), stacked bars, pies."""
    pts = p.points
    cats = list(dict.fromkeys(pts.xcat))
    cat_colours = dict(zip(cats, colours(style, len(cats))))
    for c in cats:                                  # a category's own colour, when set under Groups' look
        if c in (r.series or {}) and r.series[c].get("colour"):
            cat_colours[c] = r.series[c]["colour"]
    unit = "%" if r.percent else ""
    if r.plot in ("pie", "donut"):
        for k, g in enumerate(groups):
            mine = pts[pts.group == g].set_index("xcat").reindex(cats)
            vals = mine["count"].fillna(0).to_numpy(float)
            wedges, *_ = ax.pie(vals, center=(2.6 * k, 0), radius=1.0, colors=[cat_colours[c] for c in cats],
                                startangle=90, counterclock=False,
                                autopct=lambda v: f"{v:.0f}%" if v >= 4 else "", pctdistance=0.72 if r.plot == "pie"
                                else 0.82, wedgeprops={"width": 0.45 if r.plot == "donut" else 1.0,
                                                       "edgecolor": style.background if style.background !=
                                                       "transparent" else "white", "linewidth": 1},
                                textprops={"fontsize": style.font_size * 0.85})
            ax.text(2.6 * k, -1.35, lk[g].label or "", ha="center", va="top")
            if r.plot == "donut":
                ax.text(2.6 * k, 0, f"n = {int(vals.sum())}", ha="center", va="center")
        ax.set_aspect("equal")
        ax.set_xlim(-1.3, 2.6 * (len(groups) - 1) + 1.3)
        ax.set_ylim(-1.7, 1.3)
        ax.axis("off")
        from matplotlib.patches import Patch

        ax.legend(handles=[Patch(color=cat_colours[c], label=c) for c in cats], loc="center left",
                  bbox_to_anchor=(1.0, 0.5), frameon=style.legend_frame)
        return
    if r.plot == "stacked_bar":
        bottom = np.zeros(len(groups))
        for c in cats:
            vals = np.array([float(pts[(pts.group == g) & (pts.xcat == c)].y.sum()) for g in groups])
            bars = ax.bar(range(len(groups)), vals, bottom=bottom, color=cat_colours[c], label=c, width=0.6,
                          edgecolor=style.background if style.background != "transparent" else "white", lw=0.6)
            for b, v in zip(bars, vals):
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_y() + v / 2, f"{v:.0f}{unit}", ha="center",
                            va="center", fontsize=style.font_size * 0.8)
            bottom += vals
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([lk[g].label for g in groups])
        _labels(ax, r, r.group, "% of the group" if r.percent else "Count")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=style.legend_frame)   # beside, not on, the bars
        return
    width = 0.8 / max(1, len(groups))                # bars of counts: groups side by side in each category
    for k, g in enumerate(groups):
        look = lk[g]
        vals = [float(pts[(pts.group == g) & (pts.xcat == c)].y.sum()) for c in cats]
        bars = ax.bar(np.arange(len(cats)) - 0.4 + width * (k + 0.5), vals, width=width * 0.9, color=look.colour,
                      alpha=style.fill_alpha + 0.4, edgecolor=look.colour, lw=look.width, label=look.label or None)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}{unit}", ha="center", va="bottom",
                    fontsize=style.font_size * 0.8)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    _labels(ax, r, r.x, "% of the group" if r.percent else "Count")
    if len(groups) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=style.legend_frame)


def nice(v: float) -> float:
    """A round size for a scale bar near ``v``."""
    if not np.isfinite(v) or v <= 0:
        return 1.0
    e = 10 ** np.floor(np.log10(v))
    return float(min((1, 2, 5, 10), key=lambda m: abs(m * e - v)) * e)


def _trace(ax, r, p, style) -> None:
    rows = p.rows.dropna(subset=["x"])                 # NaN y: a blanked stretch, drawn as a gap
    groups = ordered(list(dict.fromkeys(rows.group)), r)
    lk = looks(r, style, groups)
    ends = []
    for g in groups:
        mine = rows[rows.group == g]
        look = lk[g]
        # a set drawn sweep by sweep: each its own thin line, in the table's order
        parts = [c for _, c in mine.groupby("cell", sort=False)] if r.cell and "cell" in mine             and mine["cell"].nunique() > 1 else [mine.sort_values("x")]
        many = len(parts) > 1
        for c in parts:
            if look.line == "" and look.marker != "none":     # points marked on a trace (rise, events)
                ax.plot(c.x, c.y, ls="none", marker=look.marker, ms=style.marker_size,
                        color=look.colour)
                continue
            ax.plot(c.x, c.y, color=look.colour, lw=look.width * (0.6 if many else 1.0), ls=look.line or "-",
                    alpha=0.6 if many else 1.0)
        mine = parts[-1]
        last = mine.dropna(subset=["y"]).tail(1)
        if look.label and len(last):                # named at its end: no legend over the traces
            ends.append([float(last.x.iloc[0]), float(last.y.iloc[0]), look.label, look.colour])
    gap = 0.07 * float(rows.y.max() - rows.y.min() or 1.0) * style.font_size / 8.0
    ends.sort(key=lambda e: e[1])
    ys = [e[1] for e in ends]
    for k in range(1, len(ys)):                     # names that would overlap pushed apart, centred where they were
        ys[k] = max(ys[k], ys[k - 1] + gap)
    shift = (sum(e[1] for e in ends) - sum(ys)) / len(ys) if ys else 0.0
    for e, y in zip(ends, ys):
        e[1] = y + shift
    for x, y, text, colour in ends:
        ax.text(x, y, f" {text}", color=colour, va="center", ha="left", fontsize=style.font_size * 0.9,
                clip_on=False)
    for m in r.markers:
        if m.get("end") is not None:
            ax.axvspan(m["at"], m["end"], color="#cccccc", alpha=0.6, lw=0)
    if r.scale_bars and len(rows):
        xr = rows.x.max() - rows.x.min()
        yr = rows.y.max() - rows.y.min()
        bx, by = nice(0.2 * xr), nice(0.2 * yr)
        x0, y0 = rows.x.max() - bx, rows.y.min()
        ax.plot([x0, x0 + bx], [y0, y0], color=style.ink, lw=1.5)
        ax.plot([x0 + bx, x0 + bx], [y0, y0 + by], color=style.ink, lw=1.5)
        ax.text(x0 + bx / 2, y0, f"{bx:g} {_units(r.x)}", ha="center", va="top", fontsize=style.font_size * 0.9)
        ax.text(x0 + bx, y0 + by / 2, f" {by:g} {_units(r.y)}", ha="left", va="center",
                fontsize=style.font_size * 0.9)
        ax.axis("off")
    else:
        _labels(ax, r, r.x, r.y)


def _axes(ax, r: Recipe, p: Prepared, style: Style) -> None:
    if r.question == "counts" and r.plot in ("pie", "donut"):
        return
    if r.x_log:
        ax.set_xscale("log")
    if r.y_log:
        ax.set_yscale("log")
    if len(r.x_range) == 2:
        ax.set_xlim(*r.x_range)
    if len(r.y_range) == 2:
        ax.set_ylim(*r.y_range)
    elif (r.y_from_zero if r.y_from_zero is not None else r.plot in ("bar", "grouped_bar", "count_bar",
                                                                       "stacked_bar")) and len(p.points):
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0), max(hi, 0)) if hi > 0 else ax.set_ylim(lo, 0)
    for m in r.markers if r.question != "example" else ():
        if m.get("end") is None:
            ax.axvline(m["at"], color="#555555", lw=0.7, ls="--")
        else:
            ax.axvspan(m["at"], m["end"], color="#dddddd", alpha=0.6, lw=0, zorder=0)
        if m.get("label"):                          # inside the axes, clear of the panel's title
            ax.text(m["at"], 0.99, f" {m['label']}", transform=ax.get_xaxis_transform(), ha="left", va="top",
                    fontsize=matplotlib.rcParams["font.size"] * 0.85, color="#555555")
    for line in r.ref_lines:
        if line.get("axis") == "x":
            ax.axvline(line["at"], color="#777777", lw=0.7, ls=":")
            where = (line["at"], 0.02, ax.get_xaxis_transform(), "left", "bottom")
        else:
            ax.axhline(line["at"], color="#777777", lw=0.7, ls=":")
            where = (0.01, line["at"], ax.get_yaxis_transform(), "left", "bottom")
        if line.get("label"):
            ax.text(where[0], where[1], f" {line['label']}", transform=where[2], ha=where[3], va=where[4],
                    fontsize=matplotlib.rcParams["font.size"] * 0.8, color="#555555")
    for a in r.annotations:
        if a.get("kind") == "arrow":
            ax.annotate(a.get("text", ""), xy=(a["x"], a["y"]), xytext=(a.get("x2", a["x"]), a.get("y2", a["y"])),
                        arrowprops={"arrowstyle": "->", "lw": 0.8})
        else:
            ax.text(a["x"], a["y"], a.get("text", ""))
    if r.x_rotate:
        for t in ax.get_xticklabels():
            t.set_rotation(r.x_rotate)
            t.set_ha("right")
    handles, _ = ax.get_legend_handles_labels()
    if r.legend == "none":
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    elif handles and ax.get_legend() is None:
        if r.legend == "outside":
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), ncol=max(1, style.legend_cols))
        else:
            ax.legend(loc=r.legend or "best", ncol=max(1, style.legend_cols))



# --------------------------------------------------------------------------- saving and reopening


def _mark(spec: Spec) -> str:
    return (MARK_START + base64.b64encode(zlib.compress(spec.to_json().encode("utf-8"))) + MARK_END).decode("ascii")


def export(spec: Spec, path: str | Path, dpi: int = 600) -> Path:
    """The figure in the format its extension names, the figure itself inside."""
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    if ext not in FORMATS + ("tif",):
        raise ValueError(f"Figures are saved as {', '.join(FORMATS)}, not .{ext}")
    fig, _ = build(spec)
    mark = _mark(spec)
    with plt.rc_context(rc(spec.style)):
        if ext == "png":
            fig.savefig(path, dpi=dpi, metadata={"Software": "Clampwise", "Description": mark})
        elif ext == "pdf":
            fig.savefig(path, metadata={"Creator": "Clampwise", "Subject": mark})
        elif ext == "svg":
            fig.savefig(path, metadata={"Creator": "Clampwise", "Description": mark})
        elif ext in ("tif", "tiff"):
            fig.savefig(path, dpi=dpi, pil_kwargs={"compression": "tiff_lzw", "tiffinfo": {270: mark}})
        else:
            fig.savefig(path)
            raw = path.read_bytes()
            nl = raw.index(b"\n") + 1
            path.write_bytes(raw[:nl] + b"%%ClampwiseFigure: " + mark.encode("ascii") + b"\n" + raw[nl:])
    return path


def save_spec(spec: Spec, path: str | Path) -> Path:
    """The builder's own file (.cwfig): the figure as JSON, its data inside."""
    path = Path(path)
    path.write_text(spec.to_json(), encoding="utf-8")
    return path


def read_spec(path: str | Path) -> Spec:
    """A figure saved by Clampwise (.cwfig, or any export of it), editable again."""
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix.lower() in (".cwfig", ".json"):
        return Spec.from_json(raw.decode("utf-8"))
    start = raw.find(MARK_START)
    end = raw.find(MARK_END, start)
    if start < 0 or end < 0:
        raise ValueError(f"{path.name} was not saved by Clampwise's figure builder, so it has no figure inside "
                         "(a picture of a plot can be digitized instead)")
    body = re.sub(rb"[^A-Za-z0-9+/=]", b"", raw[start + len(MARK_START):end])
    return Spec.from_json(zlib.decompress(base64.b64decode(body)).decode("utf-8"))
