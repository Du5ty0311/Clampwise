"""What a slice electrophysiology figure is made of beyond its curves, each a thing of its own that can be
selected, moved and edited: scale bars, the light's bar over a trace, the command under it, the membrane
potential written at a trace's start, a comparison's bracket, a panel's letter, text and arrows.

Each is described by a small dictionary kept in the panel's recipe (so it saves, reopens and undoes), drawn here
with matplotlib, and tagged (``tag``) so the builder's window can find what is under the mouse: the kind of
thing, the panel it belongs to, which one of its kind, and how it may be dragged. No Qt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# how a tagged thing may be dragged, and where its position is kept
MOVES = {
    "letter": "page",        # the panel's letter: an offset in mm from the panel's top left corner
    "legend": "axes",        # axes fractions of its upper left corner
    "scalebar": "axes",      # axes fractions of the corner where the two bars meet
    "stimbar": "axes_y",     # up and down only: the axes fraction of its lower edge
    "annotation": "data",    # x, y in the data's units
    "arrow_tail": "data",
    "bracket": "data_y",     # up and down only: y in the data's units
    "vmlabel": "",
    "figtext": "page",
    "panel": "page",
}
KIND_WORDS = {
    "panel": "Panel", "letter": "Panel letter", "title": "Panel title", "xlabel": "X axis title",
    "ylabel": "Y axis title", "xaxis": "X axis", "yaxis": "Y axis", "legend": "Legend", "scalebar": "Scale bars",
    "stimbar": "Stimulus bar", "command": "Command trace", "annotation": "Text", "arrow_tail": "Arrow",
    "bracket": "Comparison bracket", "series": "Group", "refline": "Reference line", "marker": "Mark along x",
    "vmlabel": "Potential label", "figtext": "Text on the page",
}


@dataclass(frozen=True)
class Tag:
    kind: str
    panel: int
    key: str = ""            # which one of its kind in the panel: an index, or a group's name
    part: str = ""           # for a thing drawn as several artists: which part this artist is


def tag(artist, kind: str, panel: int, key="", part: str = ""):
    """Mark ``artist`` as (part of) a selectable thing; returns the artist."""
    if artist is not None:
        artist._clampwise = Tag(kind, int(panel), str(key), part)
    return artist


def tag_of(artist) -> Tag | None:
    return getattr(artist, "_clampwise", None)


# ---------------------------------------------------------------- light

def wavelength_colour(nm: float) -> str:
    """The colour a light of ``nm`` is drawn in: its own hue, as figures show it (470 nm blue, 590 nm amber, 630 nm
    red); infrared (two-photon) lights in dark red, ultraviolet in violet."""
    nm = float(nm)
    if nm >= 700:
        return "#8B0000"
    if nm < 400:
        return "#7A00B8"
    # the visible spectrum, piecewise (after Bruton's approximation), kept dark enough to print
    if nm < 440:
        r, g, b = (440 - nm) / 40 * 0.6 + 0.2, 0.0, 1.0
    elif nm < 490:
        r, g, b = 0.0, (nm - 440) / 50 * 0.75, 1.0
    elif nm < 510:
        r, g, b = 0.0, 0.8, (510 - nm) / 20
    elif nm < 580:
        r, g, b = (nm - 510) / 70, 0.8, 0.0
    elif nm < 645:
        r, g, b = 1.0, (645 - nm) / 65 * 0.75, 0.0
    else:
        r, g, b = 0.9, 0.0, 0.0
    return "#{:02X}{:02X}{:02X}".format(*(int(round(255 * v)) for v in (r, g, b)))


LIGHTS = {"blue (470 nm)": 470, "cyan (505 nm)": 505, "green (530 nm)": 530, "amber (590 nm)": 590,
          "red (630 nm)": 630, "far red (660 nm)": 660, "infrared (1040 nm, two-photon)": 1040}
ELECTRICAL = "#000000"


def stim_colour(bar: dict) -> str:
    if bar.get("colour"):
        return bar["colour"]
    if bar.get("nm"):
        return wavelength_colour(bar["nm"])
    return ELECTRICAL if bar.get("kind") == "electrical" else wavelength_colour(470)


def draw_stim_bars(ax, bars: list, panel: int, ink: str, size: float) -> list:
    """Each stimulus as a bar over the plot at the time it was on (``start`` to ``end`` in x's units), in the
    light's colour; an electrical shock (no width) as a short tick. Drawn above the plot area, outside it, so it
    never covers the trace; its ``label`` is written to its left once. Returns the artists."""
    from matplotlib.patches import Rectangle

    out = []
    trans = ax.get_xaxis_transform()               # x in data units, y as a fraction of the axes
    x0, x1 = ax.get_xlim()
    least = abs(x1 - x0) * 0.004                   # a pulse too brief to see is drawn this wide (said in the caption)
    said: set[str] = set()
    for i, bar in enumerate(bars or []):
        if bar.get("hidden"):
            continue
        a = float(bar.get("start", 0.0))
        b = float(bar.get("end", a) if bar.get("end") is not None else a)
        y = float(bar.get("y", 1.02))
        h = float(bar.get("height", 0.035))
        colour = stim_colour(bar)
        if b - a <= 0 or bar.get("kind") == "electrical":            # a shock: a tick at its time
            line, = ax.plot([a, a], [y, y + h * 1.6], transform=trans, color=colour, lw=max(0.8, size / 8),
                            clip_on=False, solid_capstyle="butt")
            line.set_in_layout(True)
            out.append(tag(line, "stimbar", panel, i))
        else:
            w = max(b - a, least)
            patch = Rectangle((a, y), w, h, transform=trans, facecolor=colour, edgecolor=colour, linewidth=0.4,
                              clip_on=False)
            ax.add_patch(patch)
            out.append(tag(patch, "stimbar", panel, i))
        text = str(bar.get("label") or "")
        if text and text not in said:
            said.add(text)
            t = ax.text(a, y + h + 0.01, text, transform=trans, ha="left", va="bottom", fontsize=size,
                        color=colour if colour != "#FFFFFF" else ink, clip_on=False)
            out.append(tag(t, "stimbar", panel, i, "label"))
    return out


# ---------------------------------------------------------------- scale bars

def nice(v: float) -> float:
    """A round size for a scale bar near ``v``: 1, 2 or 5 times a power of ten."""
    if not np.isfinite(v) or v <= 0:
        return 1.0
    e = 10 ** np.floor(np.log10(v))
    return float(min((1, 2, 5, 10), key=lambda m: abs(m * e - v)) * e)


def scale_bar_sizes(bar: dict, x_range: float, y_range: float) -> tuple[float, float]:
    """The bars' lengths: those set, else a round value near a fifth of what is drawn."""
    bx = float(bar["x"]) if bar.get("x") else nice(0.2 * x_range)
    by = float(bar["y"]) if bar.get("y") else nice(0.2 * y_range)
    return bx, by


def _number(v: float) -> str:
    return f"{v:g}"


def draw_scale_bars(ax, bar: dict, x_units: str, y_units: str, panel: int, ink: str, size: float,
                    default_width: float = 1.0, key="") -> list:
    """An L of two bars with their sizes written: a horizontal one for x (time), a vertical one for y. Where
    the two meet is ``bar['at']`` (fractions of the plot area; the lower right corner of the data unless set).
    ``bar`` may set each length, each label, the line's width, which bars show, and the corner the L opens to
    (``corner``: "right" puts the vertical bar at the right end, "left" at the left)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0, x1)                            # the view is settled: the bars are placed in it, and drawing
    ax.set_ylim(y0, y1)                            # them must not stretch it
    bx, by = scale_bar_sizes(bar, abs(x1 - x0), abs(y1 - y0))
    fx, fy = bar.get("at") or (0.995, 0.02)
    cx, cy = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
    lw = float(bar.get("width") or max(default_width, 1.0))
    left = bar.get("corner") == "left"
    sx = 1.0 if (x1 > x0) else -1.0
    out = []
    hx = [cx, cx + sx * bx] if left else [cx - sx * bx, cx]
    if bar.get("show_x", True):
        line, = ax.plot(hx, [cy, cy], color=ink, lw=lw, solid_capstyle="butt", clip_on=False)
        out.append(tag(line, "scalebar", panel, key, "x"))
        label = bar.get("x_label") or f"{_number(bx)} {x_units}".strip()
        t = ax.annotate(label, xy=((hx[0] + hx[1]) / 2, cy), xytext=(0, -2), textcoords="offset points", ha="center",
                        va="top", fontsize=size, color=ink, annotation_clip=False)
        out.append(tag(t, "scalebar", panel, key, "x_label"))
    if bar.get("show_y", True):
        line, = ax.plot([cx, cx], [cy, cy + by], color=ink, lw=lw, solid_capstyle="butt", clip_on=False)
        out.append(tag(line, "scalebar", panel, key, "y"))
        label = bar.get("y_label") or f"{_number(by)} {y_units}".strip()
        t = ax.annotate(label, xy=(cx, cy + by / 2), xytext=(-2 if left else 2, 0), textcoords="offset points",
                        ha="right" if left else "left", va="center", fontsize=size, color=ink,
                        annotation_clip=False)
        out.append(tag(t, "scalebar", panel, key, "y_label"))
    return out


# ---------------------------------------------------------------- comparisons

def draw_bracket(ax, a: float, b: float, y: float, text: str, tick: float, panel: int, key, ink: str,
                 size: float, lw: float = 0.7) -> list:
    """A bracket from group position ``a`` to ``b`` at height ``y`` with its text above (a p value, stars, or
    words)."""
    line, = ax.plot([a, a, b, b], [y - tick, y, y, y - tick], color=ink, lw=lw, clip_on=False)
    t = ax.text((a + b) / 2, y, text, ha="center", va="bottom", fontsize=size, color=ink, clip_on=False)
    return [tag(line, "bracket", panel, key, "line"), tag(t, "bracket", panel, key, "text")]


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def bracket_text(p: float, how: str = "p") -> str:
    """What a bracket says: the exact p (the journals' rule), stars, or both."""
    if not np.isfinite(p):
        return ""
    exact = "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"
    return {"p": exact, "stars": stars(p), "both": f"{stars(p)} ({exact})"}.get(how, exact)
