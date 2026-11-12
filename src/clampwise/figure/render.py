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

Where each panel sits comes from ``page.layout`` (a grid whose panels can span
cells, panels placed freely in millimetres, insets). Each panel is drawn inside
its own rectangle and then fitted to it: its axes are shrunk until its tick
labels, titles and scale bars lie inside the rectangle, so a panel's rectangle
on the page is what the person arranged, whatever the panel holds.

Everything a person can select in the builder's window is tagged as it is
drawn (``elements.tag``): ``things`` lists them, ``hit`` finds the one under a
point, ``extent`` gives its outline.
"""

from __future__ import annotations

import base64
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import LogFormatter

from . import elements, journals, page
from . import stats as fstats
from .elements import tag, tag_of
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
    axes: list = field(default_factory=list)      # the panel's axes (the first is its main one)
    box: tuple = ()           # the panel's rectangle on the page: left, bottom, width, height as figure fractions
    letter: str = ""          # the letter drawn at its corner ("" when none)


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
    size = 0.0 if style.tick_dir == "none" else 2.5
    axis = float(style.axis_width or 0.8)
    # the font by its own name (not "sans-serif", which a text looks up again each time it is drawn: a figure
    # drawn outside these settings would come out in matplotlib's default font)
    return {"font.family": families(style), "font.sans-serif": families(style),
            "font.size": style.font_size, "axes.titlesize": style.size_of("title"),
            "axes.labelsize": style.size_of("label"),
            "xtick.labelsize": style.size_of("tick"), "ytick.labelsize": style.size_of("tick"),
            "legend.fontsize": style.size_of("legend"), "lines.linewidth": style.line_width,
            "axes.linewidth": min(axis, 0.5) if style.spines == "left-bottom-light" else axis,
            "xtick.major.width": axis, "ytick.major.width": axis, "xtick.minor.width": axis * 0.75,
            "ytick.minor.width": axis * 0.75, "axes.labelpad": 2.0, "axes.titlepad": 3.0,
            "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "legend.handlelength": 1.4,
            "legend.borderaxespad": 0.3, "legend.labelspacing": 0.25, "legend.handletextpad": 0.5,
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
            # text kept as text, editable in Illustrator or Inkscape; the page saved at its exact size
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": "standard",
            "savefig.pad_inches": 0.0}


_INSTALLED: set[str] = set()


def families(style: Style) -> list[str]:
    """The figure's font, then what stands in for it where it is not installed (only fonts this machine has:
    matplotlib warns, once a text, about each family of the list it cannot find)."""
    if not _INSTALLED:
        from matplotlib import font_manager

        _INSTALLED.update(font_manager.get_font_names())
    want = list(dict.fromkeys([style.font, "Arial", "Helvetica", "DejaVu Sans"]))
    return [f for f in want if f in _INSTALLED] or ["DejaVu Sans"]


def colours(style: Style, n: int) -> list[str]:
    pal = PALETTES.get(style.palette, PALETTES["okabe-ito"])
    return [pal[k % len(pal)] for k in range(max(n, 1))]


def ordered(groups: list[str], r: Recipe) -> list[str]:
    """The groups in the order asked (the rest as they come)."""
    return [g for g in r.order if g in groups] + [g for g in groups if g not in r.order]


def looks(r: Recipe, style: Style, groups: list[str]) -> dict[str, Look]:
    """Each group's look: its own where set (Groups' look), else the theme's."""
    # a group with a colour of its own for the whole figure keeps it; the others take the palette's colours
    # in order, passing over those
    fixed = {g: (style.group_colours or {})[g] for g in groups if g in (style.group_colours or {})}
    taken = {c.upper() for c in fixed.values()}
    whole = PALETTES.get(style.palette, PALETTES["okabe-ito"])
    left = [c for c in whole if c.upper() not in taken] or list(whole)
    pal, n = {}, 0
    for g in groups:
        if g in fixed:
            pal[g] = fixed[g]
        else:
            pal[g] = left[n % len(left)]
            n += 1
    marks = [m for m in MARKERS if m != "none"]
    out = {}
    for k, g in enumerate(groups):
        d = (r.series or {}).get(g, {})
        out[g] = Look(d.get("colour") or pal[g],
                      d.get("marker") or (marks[k % len(marks)] if style.vary_markers else "o"),
                      d.get("line", "-") if d.get("line") is not None else "-",
                      float(d.get("width") or style.line_width), bool(d.get("filled", True)),
                      d.get("label") or ("" if g == ALL else g))
    return out


LETTER_PAD_MM = 0.4       # the panel's letter this far inside its rectangle's top left corner
BOX_PAD_MM = 0.6          # what a panel draws stays this far inside its rectangle


def letters(spec: Spec) -> list[str]:
    """Each panel's letter: its own when set ("-": none), else the next of A, B, C... over the panels that take
    one (an inset takes none unless it is given one)."""
    out, k = [], 0
    many = sum(1 for i in range(len(spec.panels)) if not page.is_inset(spec.panels, i)) > 1
    for i, r in enumerate(spec.panels):
        own = (r.letter or "").strip()
        if own == "-" or not spec.labels:
            out.append("")
        elif own:
            out.append(own)
        elif page.is_inset(spec.panels, i) or not many:
            out.append("")
        else:
            out.append(_letter(k, spec.style.label_style))
            k += 1
    return out


def build(spec: Spec, fig: Figure | None = None) -> tuple[Figure, list[Drawn]]:
    """The whole figure: each panel prepared from its own data, drawn in its rectangle of the page and fitted
    to it; insets inside their panels; the page's own text."""
    from matplotlib.gridspec import GridSpec

    rects, w, h = page.layout(spec)
    style = spec.style
    names = letters(spec)
    with plt.rc_context(rc(style)):
        fig = fig or Figure()
        fig.clear()
        fig.set_size_inches(w / 25.4, h / 25.4)
        fig.set_facecolor("none" if style.background == "transparent" else style.background)
        fig.set_layout_engine("none")              # each panel is fitted to its own rectangle below
        out: list[Drawn | None] = [None] * len(spec.panels)
        groups = []
        for k, r in enumerate(spec.panels):
            if rects[k] is None:
                continue
            x, y, pw, ph = rects[k]
            box = (x / w, 1 - (y + ph) / h, pw / w, ph / h)
            grid = GridSpec(1, 1, figure=fig, left=box[0] + 0.12 * box[2], right=box[0] + 0.97 * box[2],
                            bottom=box[1] + 0.16 * box[3], top=box[1] + 0.92 * box[3])
            before = list(fig.axes)
            d = _panel(fig, grid[0, 0], r, style, k)
            d.axes = [a for a in fig.axes if a not in before]
            d.box, d.letter = box, names[k]
            mark = None
            if names[k]:
                dx, dy = (list(r.letter_offset) + [0.0, 0.0])[:2] if r.letter_offset else (0.0, 0.0)
                mark = fig.text(box[0] + (LETTER_PAD_MM + dx) / w, box[1] + box[3] - (LETTER_PAD_MM + dy) / h, names[k],
                                fontsize=style.size_of("letter"), fontweight="bold" if style.letter_bold else "normal",
                                va="top", ha="left")
                tag(mark, "letter", k)
            out[k] = d
            groups.append((d, grid, mark))
        for k, r in enumerate(spec.panels):        # insets: inside their panel's main axes
            if rects[k] is not None:
                continue
            parent = out[r.inset_of] if 0 <= r.inset_of < len(out) else None
            if parent is None or not parent.axes:
                out[k] = Drawn(r, None, [], "its panel is not drawn")
                continue
            ax = parent.axes[0].inset_axes(list(r.inset_rect) if len(r.inset_rect) == 4 else page.inset_default())
            ax._cw_panel = k
            ax.patch.set_alpha(0.0 if style.background == "transparent" else 1.0)
            d = _panel_on(ax, r, style, k)
            d.axes, d.letter = [ax], names[k]
            if names[k]:
                tag(ax.text(0.0, 1.02, names[k], transform=ax.transAxes, fontsize=style.size_of("letter"),
                            fontweight="bold" if style.letter_bold else "normal", va="bottom", ha="left"),
                    "letter", k)
            out[k] = d
        for d in out:
            for a in (d.axes if d is not None else ()):
                _pin(a, style)
        _fit_panels(fig, groups, w, h)
        for i, t in enumerate(spec.texts or []):
            if str(t.get("text", "")).strip():
                tag(fig.text(float(t.get("x", 0.0)) / w, 1 - float(t.get("y", 0.0)) / h, str(t["text"]),
                             fontsize=float(t.get("size") or style.size_of("annotation")),
                             fontweight="bold" if t.get("bold") else "normal", va="top", ha="left"), "figtext", -1, i)
    return fig, [d for d in out if d is not None]


def _pin(ax, style: Style) -> None:
    """The ticks' look written into the axes themselves. matplotlib makes ticks when they are first needed and
    takes their look from the settings in force at that moment: a tick made later (the window redrawing the
    figure, a check measuring it) would otherwise come out in matplotlib's own size, and the panel no longer
    fit its rectangle."""
    axis = float(style.axis_width or 0.8)
    length = 0.0 if style.tick_dir == "none" else 2.5
    way = style.tick_dir if style.tick_dir in ("in", "out") else "out"
    ax.tick_params(axis="both", which="major", labelsize=style.size_of("tick"), width=axis, length=length,
                   direction=way, colors=style.ink, pad=2.0, labelfontfamily=families(style))
    ax.tick_params(axis="both", which="minor", width=axis * 0.75, length=length * 0.6, direction=way,
                   colors=style.ink)


def _renderer(fig: Figure):
    canvas = fig.canvas
    return canvas.get_renderer() if hasattr(canvas, "get_renderer") else fig._get_renderer()


def _fit_panels(fig: Figure, groups: list, w_mm: float, h_mm: float) -> None:
    """Each panel's axes moved and shrunk so that all the panel draws (tick labels, titles, scale bars, the
    light's bar) lies inside its rectangle, under its letter. Measured as drawn, twice (the labels change a
    little as the axes do)."""
    renderer = _renderer(fig)
    fw, fh = fig.bbox.width, fig.bbox.height
    pad_x, pad_y = BOX_PAD_MM / w_mm, BOX_PAD_MM / h_mm
    for d, grid, mark in groups:
        l, b, w, h = d.box
        top = b + h - pad_y
        if mark is not None:                       # the letter's line is the panel's own top margin
            me = mark.get_window_extent(renderer)
            off = d.recipe.letter_offset or [0.0, 0.0]
            if abs(off[0]) < 3 and abs(off[1]) < 3:      # (a letter dragged away takes no room)
                top = min(top, me.y0 / fh - 0.4 * pad_y)
        want = [l + pad_x, b + pad_y, l + w - pad_x, top]
        for _ in range(6):
            boxes = [a.get_tightbbox(renderer) for a in d.axes if a.get_visible()]
            boxes = [bb for bb in boxes if bb is not None and np.all(np.isfinite(bb.extents))]
            if not boxes:
                break
            x0, y0 = min(bb.x0 for bb in boxes) / fw, min(bb.y0 for bb in boxes) / fh
            x1, y1 = max(bb.x1 for bb in boxes) / fw, max(bb.y1 for bb in boxes) / fh
            new = [grid.left + (want[0] - x0), grid.bottom + (want[1] - y0), grid.right - (x1 - want[2]),
                   grid.top - (y1 - want[3])]
            # never squeezed to nothing: at least a quarter of the rectangle stays plot
            if new[2] - new[0] < 0.25 * w:
                mid = (new[0] + new[2]) / 2
                new[0], new[2] = mid - 0.125 * w, mid + 0.125 * w
            if new[3] - new[1] < 0.25 * h:
                mid = (new[1] + new[3]) / 2
                new[1], new[3] = mid - 0.125 * h, mid + 0.125 * h
            moved = max(abs(new[0] - grid.left), abs(new[1] - grid.bottom), abs(new[2] - grid.right),
                        abs(new[3] - grid.top))
            grid.left, grid.bottom, grid.right, grid.top = new
            for a in d.axes:
                if a.get_subplotspec() is not None:
                    a._set_position(a.get_subplotspec().get_position(fig))
            if moved < 0.5 / max(fw, fh):
                break


def _letter(k: int, style: str) -> str:
    a = chr(ord("A") + k % 26)
    return {"a": a.lower(), "(A)": f"({a})", "A.": f"{a}."}.get(style, a)


def _empty(ax, r: Recipe) -> Drawn:
    ax.axis("off")
    ax.text(0.5, 0.5, r.source.get("needs") or "No data yet: select this panel and choose its data", ha="center",
            va="center", transform=ax.transAxes, color="#777777", wrap=True)
    if r.title:
        tag(ax.set_title(r.title), "title", getattr(ax, "_cw_panel", -1))
    return Drawn(r, None, [], "no data")


def _has_data(r: Recipe, frame) -> bool:
    needs_y = r.question not in ("counts",) and r.plot != "raster"
    return not (frame is None or (needs_y and not r.y) or (r.question == "counts" and not r.x)
                or (r.plot == "raster" and not r.x))


def comparisons_of(r: Recipe, p: Prepared, ordered_groups: bool = True) -> list:
    """The panel's groups compared, whatever it shows: one value per cell (a comparison of groups), or a share
    of each group (a proportion). [] for a panel that compares nothing."""
    if r.question == "compare" and r.plot != "grouped_bar":
        return fstats.compare(p.points, r.paired, test=r.test, pairs=r.pairs)
    if r.question == "counts" and r.plot == "proportion" and len(p.points):
        from .prepare import proportion_counts, proportion_event

        counts = proportion_counts(p.points, r)
        counts = {g: counts[g] for g in ordered(list(counts), r)}
        from .prepare import rows_word

        what = "cells" if p.level == "cell" else rows_word(r)
        return fstats.compare_proportions(counts, r.pairs, f"{what} “{proportion_event(p.points, r)}”")
    return []


def _panel_on(ax, r: Recipe, style: Style, k: int) -> Drawn:
    """A panel drawn on one axes already made (an inset): no split into side-by-side panels, no estimation axis,
    no command under it."""
    frame = r.frame()
    if not _has_data(r, frame):
        return _empty(ax, r)
    prep = prepare(frame, r)
    comps = comparisons_of(r, prep) if r.stats else []
    _draw(ax, r, prep, style, comps)
    if r.title:
        tag(ax.set_title(r.title, fontsize=style.size_of("title"), loc="left"), "title", k)
    return Drawn(r, prep, comps)


def _panel(fig: Figure, slot, r: Recipe, style: Style, k: int) -> Drawn:
    frame = r.frame()
    if not _has_data(r, frame):
        ax = fig.add_subplot(slot)
        ax._cw_panel = k
        return _empty(ax, r)
    prep = prepare(frame, r)
    panels = [p for p in dict.fromkeys(prep.points.panel)] or [""]
    estimation = r.plot == "estimation" and r.question == "compare"
    # a trace panel's command under it, a raster's histogram under it: a second, lower axes sharing x
    under = (r.question == "example" and r.plot == "trace" and r.command.get("show") and r.y2
             and "y2" in prep.rows and prep.rows.y2.notna().any()) or r.plot == "raster"
    share_h = float(r.command.get("share") or 0.22) if r.plot != "raster" else 0.38
    between = 0.06
    if under and r.plot == "trace" and r.scale_bars:       # the time bar's size is written under it: room for it
        slot_pt = max(slot.get_position(fig).height * fig.get_figheight() * 72.0, 1.0)
        between = max(between, 2.0 * (style.size_of("annotation") + 6.0) / slot_pt)
    sub = slot.subgridspec(2 if under else 1, len(panels) * (2 if estimation else 1),
                           width_ratios=[2, 1] * len(panels) if estimation else None,
                           height_ratios=[1 - share_h, share_h] if under else None,
                           hspace=between if under else None)
    comparisons = []
    first = None
    for j, pv in enumerate(panels):
        share = first if first is not None and r.question not in ("counts",) else None
        ax = fig.add_subplot(sub[0, 2 * j if estimation else j], sharey=share)
        ax._cw_panel = k
        first = first or ax
        mine = Prepared(prep.points[prep.points.panel == pv], prep.summary[prep.summary.panel == pv],
                        prep.rows[prep.rows.panel == pv], prep.level, prep.caption, prep.warnings, prep.counts)
        comps = comparisons_of(r, mine) if (r.stats or estimation) else []
        comparisons += comps
        lower = None
        if under:
            lower = fig.add_subplot(sub[1, j], sharex=ax)
            lower._cw_panel = k
        _draw(ax, r, mine, style, comps, lower)
        if pv:
            ax.set_title(f"{r.panel_by}: {pv}" if r.panel_by else pv, fontsize=style.size_of("title"))
        if estimation:
            right = fig.add_subplot(sub[0, 2 * j + 1])
            right._cw_panel = k
            _estimation(right, r, mine, style, comps)
        if j == 0 and r.title:
            # the title clear of the stimulus bars drawn over the plot
            over = max([float(bar.get("y", 1.02)) + float(bar.get("height", 0.035)) + (0.07 if bar.get("label")
                                                                                         else 0.0)
                        for bar in r.stim_bars if not bar.get("hidden")] or [1.0]) - 1.0
            pad = 3.0 + max(over, 0.0) * ax.get_position().height * fig.get_figheight() * 72.0
            tag(ax.set_title(r.title, fontsize=style.size_of("title"), loc="left" if pv == "" else "center",
                             pad=pad), "title", k)
    return Drawn(r, prep, comparisons)


def _each(ax, groups):
    """The groups in turn; whatever is drawn while one is current is tagged as that group's, so a click on it
    selects the group."""
    k = getattr(ax, "_cw_panel", -1)
    for g in groups:
        before = {id(a) for a in ax.get_children()}
        yield g
        for a in ax.get_children():
            if id(a) not in before and tag_of(a) is None:
                tag(a, "series", k, g)


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


def _draw(ax, r: Recipe, p: Prepared, style: Style, comps: list, lower=None) -> None:
    pts = p.points
    groups = _groups(pts, r)
    lk = looks(r, style, groups)
    q = r.question
    if r.plot == "raster":
        _raster(ax, lower, r, p, style)
        _axes(ax, r, p, style)
        return
    if q == "compare":
        if r.plot == "grouped_bar":
            _grouped(ax, r, p, style, groups, lk)
        elif r.plot == "hbar":
            _hbar(ax, r, p, style, groups, lk)
        else:
            _compare(ax, r, p, style, groups, lk, comps)
    elif q in ("time", "pulses"):
        for g in _each(ax, groups):
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
        _trace(ax, r, p, style, lower)
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
    for g in _each(ax, groups):
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
        _brackets(ax, r, comps, pos, top, span, style)


def _brackets(ax, r, comps, pos: dict, top: float, span: float, style) -> None:
    """A bracket over each pair compared, with its p value (or stars, or the person's own words), each where it
    was dragged to when it was."""
    own = {(str(b.get("a")), str(b.get("b"))): b for b in r.brackets or []}
    panel = getattr(ax, "_cw_panel", -1)
    for k, comp in enumerate(comps):
        if comp.reference not in pos or comp.other not in pos:
            continue
        mine = own.get((str(comp.reference), str(comp.other)), {})
        if mine.get("hidden"):
            continue
        y = float(mine["y"]) if mine.get("y") is not None else top + span * (0.08 + 0.14 * k)
        elements.draw_bracket(ax, pos[comp.reference], pos[comp.other], y, mine.get("text") or
                              elements.bracket_text(comp.p, r.stats_text), span * 0.02, panel,
                              f"{comp.reference}|{comp.other}", style.ink, style.size_of("annotation"),
                              max(0.5, style.axis_width * 0.9))


def _hbar(ax, r, p, style, groups, lk) -> None:
    pts = p.points
    pos = {g: i for i, g in enumerate(groups)}
    for g in _each(ax, groups):
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
    for k, g in enumerate(_each(ax, groups)):
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
    for g in _each(ax, groups):
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
                fontsize=style.size_of("annotation") * 0.85).set_in_layout(False)   # inside the plot: no room taken
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
        for g in _each(ax, groups):
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
        for g in _each(ax, groups):
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

        for g in _each(ax, groups):
            v = rows[rows.group == g].y.to_numpy(float)
            if v.size > 2:
                (theo, ordered_v), (slope, intercept, _r) = sps.probplot(v, dist="norm")
                look = lk[g]
                _scatter(ax, theo, ordered_v, look, style, label=look.label or None)
                ax.plot(theo, slope * theo + intercept, color=look.colour, lw=look.width, ls="--")
        _labels(ax, r, "Normal quantiles", f"{r.y} (ordered)")
        return
    grid = np.linspace(rows.y.min(), rows.y.max(), 300)
    for g in _each(ax, groups):
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
    if r.plot == "proportion":
        return _proportion(ax, r, p, style, groups, lk)
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
    for k, g in enumerate(_each(ax, groups)):
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


def _proportion(ax, r, p, style, groups, lk) -> None:
    """Each group's share in one category: a bar to the share (%), Wilson's 95% interval on it, k of n over it;
    with statistics on, a bracket over each pair compared (Fisher's exact test)."""
    from .prepare import proportion_counts, proportion_event

    counts = proportion_counts(p.points, r)
    event = proportion_event(p.points, r)
    pos = {g: i for i, g in enumerate(groups)}
    small = style.size_of("annotation")
    tops = []
    for g in _each(ax, groups):
        k, n = counts.get(g, (0, 0))
        if n <= 0:
            continue
        look = lk[g]
        share = 100.0 * k / n
        lo, hi = (100.0 * v for v in fstats.wilson(k, n))
        ax.bar(pos[g], share, width=0.6, color=look.colour, alpha=style.fill_alpha + 0.4, edgecolor=look.colour,
               lw=look.width)                         # (named under it on the x axis: no legend)
        if r.error_style != "none":
            ax.errorbar(pos[g], share, yerr=[[share - lo], [hi - share]], color=style.ink, lw=max(0.6, style.axis_width),
                        capsize=style.cap_size, fmt="none")
        ax.text(pos[g], hi if r.error_style != "none" else share, f"{k}/{n}", ha="center", va="bottom",
                fontsize=small, color=style.ink)
        tops.append(hi if r.error_style != "none" else share)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([lk[g].label for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    top = max(tops or [100.0])
    comps = comparisons_of(r, p) if r.stats else []
    shown = [c for c in comps if not next((b for b in r.brackets or [] if (str(b.get("a")), str(b.get("b"))) ==
                                           (str(c.reference), str(c.other))), {}).get("hidden")]
    from .prepare import rows_word

    full = max(100.0 if top > 60 else 0.0, top * 1.12 + 6.0)
    if len(r.y_range) != 2:
        ax.set_ylim(0, full * (1.0 + 0.16 * len(shown)))
    if shown:
        _brackets(ax, r, comps, pos, top + 0.06 * full, full * 1.14, style)
    _labels(ax, r, r.group, f"{event} (% of {'cells' if p.level == 'cell' else rows_word(r)})" if event else "%")


nice = elements.nice          # a round size for a scale bar (kept under its old name)


def _trace(ax, r, p, style, lower=None) -> None:
    """Recorded traces: each set a line (a set drawn sweep by sweep: thin lines), named at its end or by the
    potential written at its start, cut where asked (spikes truncated), with scale bars instead of axes and the
    command under them when the panel has one."""
    k = getattr(ax, "_cw_panel", -1)
    rows = p.rows.dropna(subset=["x"])                 # NaN y: a blanked stretch, drawn as a gap
    groups = ordered(list(dict.fromkeys(rows.group)), r)
    lk = looks(r, style, groups)
    clip = sorted(float(v) for v in r.clip_y) if len(r.clip_y) == 2 else None
    small = style.size_of("annotation")
    ends, starts = [], {}
    drop = 0.0
    if r.stack and len(groups) > 1:                  # each trace on its own level, the first on top
        spans_ = [float(rows[rows.group == g].y.max() - rows[rows.group == g].y.min()) for g in groups]
        drop = float(r.stack_gap) if r.stack_gap and r.stack_gap > 0 else 1.15 * max(
            [v for v in spans_ if np.isfinite(v)] or [1.0])
    for level, g in enumerate(_each(ax, groups)):
        mine = rows[rows.group == g]
        if drop:
            mine = mine.assign(y=mine.y - level * drop)
        look = lk[g]
        # a set drawn sweep by sweep: each its own thin line, in the table's order
        parts = [c for _, c in mine.groupby("cell", sort=False)] if r.cell and "cell" in mine \
            and mine["cell"].nunique() > 1 else [mine.sort_values("x")]
        many = len(parts) > 1
        for c in parts:
            y = c.y.to_numpy(float)
            if clip is not None:
                y = np.where((y < clip[0]) | (y > clip[1]), np.nan, y)
            if look.line == "" and look.marker != "none":     # points marked on a trace (rise, events)
                ax.plot(c.x, y, ls="none", marker=look.marker, ms=style.marker_size, color=look.colour)
                continue
            ax.plot(c.x, y, color=look.colour, lw=look.width * (0.6 if many else 1.0), ls=look.line or "-",
                    alpha=0.6 if many else 1.0)
        whole = parts[-1].dropna(subset=["y"])
        if len(whole):
            starts[g] = (float(whole.x.iloc[0]), float(whole.y.iloc[0]))
            if look.label and (r.series or {}).get(g, {}).get("legend") is not False:   # named at its end
                ends.append([float(whole.x.iloc[-1]), float(whole.y.iloc[-1]), look.label, look.colour, g])
    # a line of text's height in the data's units, from the plot area's height on the page (its place is not
    # final yet: a little more than it needs)
    lo_d, hi_d = ax.dataLim.intervaly
    drawn = float(hi_d - lo_d) if np.isfinite(hi_d - lo_d) and hi_d > lo_d else float(rows.y.max() - rows.y.min() or 1.0)
    h_pt = max(ax.get_position().height * ax.figure.get_figheight() * 72.0, 1.0)
    gap = max(0.07 * drawn * small / 7.2, 1.45 * small / h_pt * drawn * 1.25)

    def apart(ys: list[float]) -> list[float]:
        """Heights (sorted) pushed apart so texts there do not overlap, centred where they were."""
        out = list(ys)
        for i in range(1, len(out)):
            out[i] = max(out[i], out[i - 1] + gap)
        shift = (sum(ys) - sum(out)) / len(out) if out else 0.0
        return [y + shift for y in out]

    ends.sort(key=lambda e: e[1])
    for e, y in zip(ends, apart([e[1] for e in ends])):
        e[1] = y
    for x, y, text, colour, g in ends:
        tag(ax.text(x, y, f" {text}", color=colour, va="center", ha="left", fontsize=small, clip_on=False),
            "series", k, g, "name")
    if r.show_vm:                                    # the potential (or the holding current) at each trace's start
        span = float(rows.y.max() - rows.y.min() or 1.0)
        written: list[tuple[str, float]] = []
        labels: list[list] = []
        for g, text in (r.vm_labels or {}).items():
            if g not in starts or not str(text).strip():
                continue
            if any(t == str(text) and abs(y - starts[g][1]) < 0.04 * span for t, y in written):
                continue                             # traces that start together at one potential: written once
            same = sum(1 for g2, t2 in r.vm_labels.items() if t2 == text and g2 in starts
                       and abs(starts[g2][1] - starts[g][1]) < 0.04 * span)
            written.append((str(text), starts[g][1]))
            labels.append([starts[g][0], starts[g][1], str(text), lk[g].colour if same == 1 else style.ink, g])
        labels.sort(key=lambda e: e[1])
        for (x, _y, text, colour, g), y in zip(labels, apart([e[1] for e in labels])):
            tag(ax.text(x, y, f"{text} ", color=colour, va="center", ha="right", fontsize=small, clip_on=False),
                "vmlabel", k, g)
    for m in r.markers:
        if m.get("end") is not None:
            ax.axvspan(m["at"], m["end"], color="#cccccc", alpha=0.6, lw=0)
    if len(r.x_range) == 2:
        ax.set_xlim(*r.x_range)
    if len(r.y_range) == 2:
        ax.set_ylim(*r.y_range)
    elif clip is not None:
        lo, hi = ax.get_ylim()
        ax.set_ylim(max(lo, clip[0] - 0.03 * (clip[1] - clip[0])), min(hi, clip[1] + 0.03 * (clip[1] - clip[0])))
    if lower is not None:                            # the command each trace was recorded under, in the same colours
        for g in groups:
            mine = rows[rows.group == g]
            if "y2" not in mine or not mine.y2.notna().any():
                continue
            parts = [c for _, c in mine.groupby("cell", sort=False)] if r.cell and "cell" in mine \
                and mine["cell"].nunique() > 1 else [mine.sort_values("x")]
            for c in parts:
                line, = lower.plot(c.x, c.y2, color=lk[g].colour, lw=max(0.5, lk[g].width * 0.7))
                tag(line, "command", k, g)
        lower.margins(y=0.12)
        if r.scale_bars:
            lower.axis("off")
            span = lower.get_ylim()
            if abs(span[1] - span[0]) > 0:
                elements.draw_scale_bars(lower, {"show_x": False, "at": (0.99, 0.1), **{
                    kk: v for kk, v in (r.command or {}).items() if kk in ("y", "y_label", "width")}}, "",
                    _units(r.y2), k, style.ink, small, style.line_width, key="command")
        else:
            lower.set_ylabel(r.y2)
            lower.set_xlabel(r.x_label or r.x)
            ax.tick_params(labelbottom=False)
    if r.scale_bars and len(rows):
        ax.axis("off")
        if len(r.y_range) != 2 and not (r.scale_bar or {}).get("at"):
            lo, hi = ax.get_ylim()                   # room under the lowest trace for the bars: the vertical one
            room = 0.12 * (hi - lo)                  # ends under the traces (its length follows the room made)
            if (r.scale_bar or {}).get("show_y", True):
                for _ in range(4):
                    _bx, by = elements.scale_bar_sizes(r.scale_bar or {}, 1.0, abs(hi - lo) + room)
                    room = by + 0.08 * (hi - lo)
            ax.set_ylim(lo - min(room, 0.5 * (hi - lo)), hi)
        elements.draw_scale_bars(ax, r.scale_bar or {}, _units(r.x), _units(r.y), k, style.ink, small,
                                 style.line_width)
    elif lower is None:
        _labels(ax, r, r.x, r.y)
    else:
        ax.set_ylabel(r.y_label or r.y)


def _raster(ax, lower, r, p, style) -> None:
    """A raster: one row per sweep (the Cell column, else every row its own), a tick at each time in x; under it
    the rate in time bins over the sweeps (Hz when x is in s or ms, else counts per bin)."""
    k = getattr(ax, "_cw_panel", -1)
    rows = p.rows.dropna(subset=["x"])
    groups = ordered(list(dict.fromkeys(rows.group)), r)
    lk = looks(r, style, groups)
    sweeps = list(dict.fromkeys(rows.cell)) if (rows.cell != "").any() else [""]
    at = {s: i for i, s in enumerate(sweeps)}
    for g in _each(ax, groups):
        mine = rows[rows.group == g]
        ax.eventplot([mine[mine.cell == s].x.to_numpy(float) for s in sweeps], lineoffsets=[at[s] + 1 for s in sweeps],
                     linelengths=0.8, colors=lk[g].colour, linewidths=max(0.6, lk[g].width * 0.8))
    ax.set_ylim(len(sweeps) + 0.6, 0.4)             # sweep 1 on top
    ax.set_ylabel(r.y_label or ("Sweep" if len(sweeps) > 1 else ""))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=4))
    if lower is None:
        ax.set_xlabel(r.x_label or r.x)
        return
    ax.tick_params(labelbottom=False)
    lo, hi = (r.x_range if len(r.x_range) == 2 else (float(rows.x.min()), float(rows.x.max()))) if len(rows) else (0, 1)
    width = float(r.bin_width) if r.bin_width and r.bin_width > 0 else elements.nice((hi - lo) / 25 or 1.0)
    edges = np.arange(np.floor(lo / width) * width, hi + width, width)
    per_s = {"s": 1.0, "ms": 1e3, "min": 1 / 60.0}.get(_units(r.x))
    for g in _each(lower, groups):
        counts, _ = np.histogram(rows[rows.group == g].x, bins=edges)
        rate = counts / max(1, len(sweeps)) / width * per_s if per_s else counts
        lower.bar(edges[:-1], rate, width=width, align="edge", color=lk[g].colour, alpha=0.75, lw=0)
    lower.set_ylabel("Rate (Hz)" if per_s else "Count per bin")
    lower.set_xlabel(r.x_label or r.x)
    lower.set_ylim(bottom=0)


class _PlainLog(LogFormatter):
    """A log axis's labels as numbers are written (0.4, 2, 10, 300), never 4 × 10⁻¹; matplotlib's own rule
    decides which minor ticks get one (all of them on a narrow range, none on a wide one)."""

    def _num_to_string(self, x, vmin, vmax):
        return f"{x:g}"


def _axes(ax, r: Recipe, p: Prepared, style: Style) -> None:
    k = getattr(ax, "_cw_panel", -1)
    small = style.size_of("annotation")
    if r.question == "counts" and r.plot in ("pie", "donut"):
        return
    if r.x_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(_PlainLog(labelOnlyBase=False))
        ax.xaxis.set_minor_formatter(_PlainLog(labelOnlyBase=False))
    if r.y_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(_PlainLog(labelOnlyBase=False))
        ax.yaxis.set_minor_formatter(_PlainLog(labelOnlyBase=False))
    if len(r.x_range) == 2:
        ax.set_xlim(*r.x_range)
    if len(r.y_range) == 2 and r.plot != "raster":
        ax.set_ylim(*r.y_range)
    elif (r.y_from_zero if r.y_from_zero is not None else r.plot in ("bar", "grouped_bar", "count_bar",
                                                                       "stacked_bar")) and len(p.points):
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0), max(hi, 0)) if hi > 0 else ax.set_ylim(lo, 0)
    for i, m in enumerate(r.markers if r.question != "example" else ()):
        if m.get("end") is None:
            tag(ax.axvline(m["at"], color="#555555", lw=0.7, ls="--"), "marker", k, i)
        else:
            tag(ax.axvspan(m["at"], m["end"], color="#dddddd", alpha=0.6, lw=0, zorder=0), "marker", k, i)
        if m.get("label"):                          # inside the axes, clear of the panel's title
            tag(ax.text(m["at"], 0.99, f" {m['label']}", transform=ax.get_xaxis_transform(), ha="left", va="top",
                        fontsize=small, color="#555555"), "marker", k, i, "label")
    for i, line in enumerate(r.ref_lines):
        if line.get("axis") == "x":
            tag(ax.axvline(line["at"], color="#777777", lw=0.7, ls=":"), "refline", k, i)
            where = (line["at"], 0.02, ax.get_xaxis_transform(), "left", "bottom")
        else:
            tag(ax.axhline(line["at"], color="#777777", lw=0.7, ls=":"), "refline", k, i)
            where = (0.01, line["at"], ax.get_yaxis_transform(), "left", "bottom")
        if line.get("label"):
            tag(ax.text(where[0], where[1], f" {line['label']}", transform=where[2], ha=where[3], va=where[4],
                        fontsize=small, color="#555555"), "refline", k, i, "label")
    # the light (or a drug, or a shock) as a bar over the plot at its time
    elements.draw_stim_bars(ax, r.stim_bars, k, style.ink, small)
    for i, a in enumerate(r.annotations):
        size = float(a.get("size") or small)
        if a.get("kind") == "arrow":
            t = ax.annotate(a.get("text", ""), xy=(a["x"], a["y"]), xytext=(a.get("x2", a["x"]), a.get("y2", a["y"])),
                            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": a.get("colour") or style.ink},
                            fontsize=size, color=a.get("colour") or style.ink, annotation_clip=False)
        else:
            t = ax.text(a["x"], a["y"], a.get("text", ""), fontsize=size, color=a.get("colour") or style.ink,
                        fontweight="bold" if a.get("bold") else "normal", ha=a.get("ha", "left"),
                        va=a.get("va", "baseline"), clip_on=False)
        tag(t, "annotation", k, i)
    if r.zero_axes and ax.axison:                   # the axes through 0, 0 (a current-voltage plot)
        for side in ("left", "bottom"):
            ax.spines[side].set_position("zero")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.xaxis.set_label_coords(1.0, -0.02)
        ax.xaxis.label.set_horizontalalignment("right")
        ax.yaxis.set_label_coords(-0.02, 1.0)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_horizontalalignment("right")
    if r.x_rotate:
        for t in ax.get_xticklabels():
            t.set_rotation(r.x_rotate)
            t.set_ha("right")
    tag(ax.xaxis.label, "xlabel", k)
    tag(ax.yaxis.label, "ylabel", k)
    handles, _ = ax.get_legend_handles_labels()
    if r.legend == "none":
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    elif handles and ax.get_legend() is None:
        if len(r.legend_at) == 2:                   # where it was dragged to
            ax.legend(loc="upper left", bbox_to_anchor=tuple(r.legend_at), ncol=max(1, style.legend_cols),
                      borderaxespad=0.0)
        elif r.legend == "outside":
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), ncol=max(1, style.legend_cols))
        else:
            ax.legend(loc=r.legend or "best", ncol=max(1, style.legend_cols))
    if ax.get_legend() is not None:
        tag(ax.get_legend(), "legend", k)


# --------------------------------------------------------------------------- what can be selected

ORDER = ("letter", "figtext", "annotation", "bracket", "scalebar", "stimbar", "legend", "vmlabel", "title", "xlabel",
         "ylabel", "refline", "marker", "command", "series")


def things(fig: Figure) -> list:
    """(tag, artist) of everything drawn that a person can select, in drawing order."""
    return [(tag_of(a), a) for a in fig.findobj(lambda a: tag_of(a) is not None)]


def _box(artist, renderer):
    try:
        bb = artist.get_window_extent(renderer)
    except Exception:
        return None
    return bb if bb is not None and np.all(np.isfinite(bb.extents)) and bb.width >= 0 and bb.height >= 0 else None


def extent(fig: Figure, which, grow: float = 2.0) -> tuple[float, float, float, float] | None:
    """The outline (x0, y0, x1, y1 in the figure's pixels, y up) of the thing ``which`` (a Tag: every artist of
    its kind, panel and key), a little larger than it; None when it is not drawn."""
    renderer = _renderer(fig)
    boxes = [_box(a, renderer) for t, a in things(fig) if (t.kind, t.panel, t.key) == (which.kind, which.panel,
                                                                                      which.key)]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (min(b.x0 for b in boxes) - grow, min(b.y0 for b in boxes) - grow, max(b.x1 for b in boxes) + grow,
            max(b.y1 for b in boxes) + grow)


def _near(artist, x: float, y: float, tol: float, renderer) -> bool:
    """Whether the point is on ``artist``: within ``tol`` pixels of a line's path or a point, inside a patch or a
    text's box."""
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if isinstance(artist, Line2D):
        xy = artist.get_transform().transform(np.column_stack(artist.get_data(orig=False)).astype(float))
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if not len(xy):
            return False
        if len(xy) == 1 or artist.get_linestyle() in ("None", "none", " ", ""):
            return bool(np.min(np.hypot(xy[:, 0] - x, xy[:, 1] - y)) <= tol + artist.get_markersize() / 2)
        a, b = xy[:-1], xy[1:]
        d = b - a
        length = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
        t = np.clip(((x - a[:, 0]) * d[:, 0] + (y - a[:, 1]) * d[:, 1]) / length, 0, 1)
        return bool(np.min(np.hypot(a[:, 0] + t * d[:, 0] - x, a[:, 1] + t * d[:, 1] - y)) <= tol)
    if isinstance(artist, Patch):
        return bool(artist.contains_point((x, y), radius=tol))
    if isinstance(artist, Collection):
        offs = artist.get_offsets()
        if len(offs):
            xy = artist.get_offset_transform().transform(np.asarray(offs, float))
            if np.min(np.hypot(xy[:, 0] - x, xy[:, 1] - y)) <= tol + 3:
                return True
        bb = _box(artist, renderer)
        return bool(len(artist.get_paths()) and not len(offs) and bb is not None and bb.contains(x, y)
                    and any(pth.contains_point((x, y), artist.get_transform(), radius=tol)
                            for pth in artist.get_paths()[:400]))
    bb = _box(artist, renderer)
    return bool(bb is not None and bb.x0 - tol <= x <= bb.x1 + tol and bb.y0 - tol <= y <= bb.y1 + tol)


def hit(fig: Figure, drawn: list[Drawn], x: float, y: float, tol: float = 4.0):
    """The thing under the point (the figure's pixels, y up), as its Tag: the smallest and topmost first (a
    letter, a text, a bracket, the scale bars, a stimulus bar, the legend, a title, a group's curve), then an
    axis (its tick labels' strip), then the panel whose rectangle holds the point; None on the bare page."""
    renderer = _renderer(fig)
    found = things(fig)
    for kind in ORDER:
        for t, a in reversed(found):
            if t.kind == kind and a.get_visible() and _near(a, x, y, tol, renderer):
                return elements.Tag(t.kind, t.panel, t.key)
        if kind == "scalebar":                      # the corner the two bars enclose is theirs too
            for key in dict.fromkeys((t.panel, t.key) for t, _a in found if t.kind == kind):
                which = elements.Tag(kind, key[0], key[1])
                o = extent(fig, which, 0.0)
                if o is not None and o[0] <= x <= o[2] and o[1] <= y <= o[3]:
                    return which
    fw, fh = fig.bbox.width, fig.bbox.height
    order = sorted(range(len(drawn)), key=lambda i: not page_is_inset(drawn, i))      # insets lie on their panels
    for i in order:
        d = drawn[i]
        k = getattr(d.axes[0], "_cw_panel", i) if d.axes else i
        for ax in d.axes:
            bb = ax.get_window_extent(renderer)
            if bb.contains(x, y):
                return elements.Tag("panel", k)
            if ax.axison:
                tight = ax.get_tightbbox(renderer)
                if tight is not None and tight.contains(x, y):
                    if x < bb.x0 and bb.y0 <= y <= bb.y1:
                        return elements.Tag("yaxis", k)
                    if y < bb.y0 and bb.x0 <= x <= bb.x1:
                        return elements.Tag("xaxis", k)
    for i in order:
        d = drawn[i]
        if d.box and d.box[0] * fw <= x <= (d.box[0] + d.box[2]) * fw and d.box[1] * fh <= y <= (d.box[1] + d.box[3]) * fh:
            return elements.Tag("panel", getattr(d.axes[0], "_cw_panel", i) if d.axes else i)
    return None


def page_is_inset(drawn: list[Drawn], i: int) -> bool:
    return not drawn[i].box


def panel_outline(fig: Figure, d: Drawn) -> tuple[float, float, float, float] | None:
    """A panel's rectangle in the figure's pixels (an inset's: its axes)."""
    fw, fh = fig.bbox.width, fig.bbox.height
    if d.box:
        return d.box[0] * fw, d.box[1] * fh, (d.box[0] + d.box[2]) * fw, (d.box[1] + d.box[3]) * fh
    if d.axes:
        bb = d.axes[0].get_window_extent(_renderer(fig))
        return bb.x0, bb.y0, bb.x1, bb.y1
    return None


# --------------------------------------------------------------------------- the journal's check

def check(spec: Spec, fmt: str = "", built: tuple | None = None) -> list:
    """What an export of ``spec`` would get wrong for its journal (``journals.check``): the page's size, every
    text's size at final size, every line's width. With no journal chosen, only what does not print (text
    under 5 pt, hairlines). ``built``: the figure already drawn (fig, drawn), to save drawing it again."""
    from matplotlib.lines import Line2D
    from matplotlib.text import Text as MplText

    fig, drawn = built if built is not None else build(spec)
    w, h = spec.size_mm()
    panel_of = {}
    for d in drawn:
        for ax in d.axes:
            panel_of[id(ax)] = d.letter
    texts, lines = [], []
    for a in fig.findobj(lambda a: isinstance(a, (MplText, Line2D))):
        if not a.get_visible():
            continue
        where = panel_of.get(id(getattr(a, "axes", None)), "")
        if isinstance(a, MplText):
            if a.get_text().strip() and (a.axes is None or a.axes.axison or tag_of(a) is not None
                                         or a not in (a.axes.xaxis.label, a.axes.yaxis.label)):
                if a.axes is not None and not a.axes.axison and tag_of(a) is None:
                    continue                        # tick labels of an axis that is not drawn
                texts.append((a.get_text().strip(), float(a.get_fontsize()), where))
        elif a.axes is not None and (a.axes.axison or tag_of(a) is not None) and len(a.get_xdata()) \
                and a.get_linestyle() not in ("None", "none", " ", ""):
            lines.append((float(a.get_linewidth()), where))
    return journals.check(spec.journal, w, h, texts, lines, fmt)


# --------------------------------------------------------------------------- saving and reopening


def _mark(spec: Spec) -> str:
    return (MARK_START + base64.b64encode(zlib.compress(spec.to_json().encode("utf-8"))) + MARK_END).decode("ascii")


def export_dpi(spec: Spec) -> int:
    """The resolution a pixel export should have: the journal's for line art (text and lines must stay sharp),
    600 dpi with no journal chosen."""
    j = journals.JOURNALS.get(spec.journal)
    return int(j.dpi_line) if j else 600


def export(spec: Spec, path: str | Path, dpi: int | None = None, outline: bool = False) -> Path:
    """The figure in the format its extension names, at its exact size on the page, the figure itself inside
    (so it reopens editable). Vector files (PDF, SVG, EPS) keep their text as text, editable in Illustrator or
    Inkscape; ``outline`` turns the letters into shapes instead (SVG: paths; PDF and EPS: Type 3 glyphs), for a
    journal that asks for outlined fonts. ``dpi`` (PNG, TIFF): the journal's line-art resolution unless given."""
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    if ext not in FORMATS + ("tif",):
        raise ValueError(f"Figures are saved as {', '.join(FORMATS)}, not .{ext}")
    dpi = int(dpi or export_dpi(spec))
    mark = _mark(spec)
    look = rc(spec.style)
    if outline:
        look.update({"pdf.fonttype": 3, "ps.fonttype": 3, "svg.fonttype": "path"})
    with plt.rc_context(look):
        fig, _ = build(spec)
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
