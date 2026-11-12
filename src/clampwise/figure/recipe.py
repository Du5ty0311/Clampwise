"""A figure as data. ``Spec`` holds the panels (one ``Recipe`` each), how they
are laid out and the figure's size; ``Recipe`` holds what one panel shows and
how. Both turn into JSON and back, with the table each panel was drawn from
kept inside, so a saved figure reopens editable even when its source moved.

The words the window shows are here too, one place for each: the questions a
figure can answer (B12 item 1, each with why its standard plot is used), the
plot types, the group centre and spread choices, and every control's caption,
line and help (B12 item 2: named for what they do).
"""

from __future__ import annotations

import base64
import gzip
import io
import json
from dataclasses import asdict, dataclass, field, fields

import pandas as pd

FORMAT = "clampwise-figure"
VERSION = 2          # 2: free layout, journal sizes, separate text sizes, electrophysiology elements

# (key, title, why the standard plot is used, the standard plot, the other plots that answer it)
QUESTIONS = (
    ("compare", "Compare groups or conditions",
     "Every cell's value as a point with the group centre and spread: a bar alone hides how the cells are spread "
     "and how many there are (Weissgerber et al. 2015).", "points",
     ("points", "bar", "box", "violin", "raincloud", "dot", "hbar", "grouped_bar", "estimation", "superplot")),
    ("time", "Change over time",
     "Each cell's values averaged into time bins, shown relative to its own baseline, and the mean across cells "
     "with its spread: the standard LTP, wash-in and DSI time course.", "time_course",
     ("time_course", "lines", "area")),
    ("pulses", "Pulse-by-pulse profile",
     "Each pulse over pulse 1 against pulse number, one line per cell and the mean on top: paired pulses and "
     "trains.", "profile", ("profile",)),
    ("relationship", "Relationship between two measures",
     "A scatter, one point per cell (or per event), with a fitted curve if asked: input-output, I-V, amplitude "
     "against rise.", "scatter",
     ("scatter", "regression", "line", "step", "xy_error", "bubble", "hexbin", "heatmap")),
    ("distribution", "Distribution",
     "Cumulative distributions, one per cell or per condition: events from one cell are not independent, so "
     "compare cells, not pooled events (Aarts et al. 2014).", "cumulative",
     ("cumulative", "histogram", "density", "qq")),
    ("counts", "Counts and proportions",
     "How many cells fall in each category (responded or not, a cell type, a firing pattern), per group: bars of "
     "the counts or the percentages, every count written on its bar; or the share of each group in one category "
     "(connected, a success) with its 95% interval.", "count_bar",
     ("count_bar", "stacked_bar", "proportion", "pie", "donut")),
    ("example", "A raw example",
     "A trace or an averaged response with scale bars instead of axes; the sweep closest to the group's mean "
     "when picked for you.", "trace", ("trace", "raster")),
)
QUESTION = {q[0]: q for q in QUESTIONS}

PLOTS = {
    "points": ("Every point with the centre and spread", "each cell a dot, the group centre a line, its spread "
               "a bar"),
    "bar": ("Bar with every point", "a bar to the group mean, every cell shown on it"),
    "box": ("Box plot with every point", "median, quartiles and the range within 1.5 IQR; every cell shown"),
    "violin": ("Violin with every point", "the shape of the distribution, every cell shown"),
    "estimation": ("Estimation plot", "the cells of each group and, beside them, the difference from the first "
                   "group with its 95% CI (Ho et al. 2019)"),
    "superplot": ("SuperPlot", "every row coloured by its cell, each cell's mean on top; statistics on the cell "
                  "means (Lord et al. 2020)"),
    "time_course": ("Time course", "the mean across cells in each time bin, its spread shaded"),
    "lines": ("Every cell over time", "each cell a thin line, the mean across cells on top"),
    "profile": ("Pulse profile", "each pulse over pulse 1, one line per cell, the mean on top"),
    "scatter": ("Scatter", "one point per row or per cell, a fitted curve if asked"),
    "line": ("Line", "points joined in x order, one line per group (one per cell when cells are named)"),
    "xy_error": ("XY with error bars", "the centre and spread of y at each x, per group"),
    "heatmap": ("Heat map", "y against x, the colour a third column (the heat-map rules: missing stays blank)"),
    "cumulative": ("Cumulative distribution", "the fraction of values at or below each value, one curve per "
                   "cell or per group"),
    "histogram": ("Histogram", "how many values fall in each bin, one per group"),
    "trace": ("Trace", "a recorded trace or an averaged response, with scale bars"),
    "raster": ("Raster with its histogram", "one row of ticks per sweep at each spike or event, and under it the "
               "rate in time bins"),
    "raincloud": ("Raincloud", "half a violin, a box and every point beside it: the distribution, its summary and "
                  "the data at once"),
    "dot": ("Centre and spread only", "the group centre and its spread as a dot and whiskers, every point faint"),
    "hbar": ("Horizontal bars with every point", "bars across, for long group names"),
    "grouped_bar": ("Grouped bars", "groups side by side within each category of X (e.g. condition within sex)"),
    "area": ("Area under the time course", "the mean over time filled to zero, the spread shaded"),
    "regression": ("Regression with its 95% band", "a straight line through the points and the band where the "
                   "true line lies with 95% confidence"),
    "step": ("Step line", "points joined by steps, for counts or binned rates"),
    "bubble": ("Bubble chart", "a scatter whose point size is a third column"),
    "hexbin": ("Hexagonal bins", "a scatter of many points as the count in each hexagon"),
    "density": ("Density", "a smooth estimate of the distribution (Gaussian kernel), one curve per group"),
    "qq": ("Q-Q plot", "each group's quantiles against a normal distribution's: a straight line means normal"),
    "count_bar": ("Bars of counts", "how many in each category, groups side by side"),
    "stacked_bar": ("Stacked bars", "each group's categories stacked to its total (or to 100%)"),
    "proportion": ("Proportion with its 95% interval", "the share of each group in one category (connected, a "
                   "success): a bar, Wilson's interval, k of n written; groups compared by Fisher's exact test"),
    "pie": ("Pie", "each category's share of the whole, one pie per group"),
    "donut": ("Donut", "a pie with the total in the middle"),
}
PLOT_QUESTION = {p: q[0] for q in QUESTIONS for p in q[4]}       # any plot can be tried: its question follows

MARKERS = {"o": "circle", "s": "square", "^": "triangle up", "v": "triangle down", "D": "diamond", "p": "pentagon",
           "h": "hexagon", "*": "star", "x": "x", "+": "plus", "none": "no marker"}
LINES = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dash-dot", "": "no line"}
ERROR_STYLES = {"bar": "error bars", "band": "a shaded band", "none": "no spread shown"}

SUMMARIES = {
    "mean_sem": ("Mean ± SEM", "how precisely the mean is known; it shrinks as n grows"),
    "mean_sd": ("Mean ± SD", "how much the cells differ from each other"),
    "median_iqr": ("Median with IQR", "for skewed data: the middle value and the middle half"),
    "mean_ci95": ("Mean with 95% CI", "the range of means compatible with the data"),
}

# B12 item 5: the standard figure for each kind of analysis, why, and the source
STANDARD = (
    ("Evoked responses", "compare", "Pulse 1 amplitude per cell, control against drug: every cell shown, the "
     "groups' mean ± SEM, the difference with its 95% CI.", "Weissgerber et al. 2015; Ho et al. 2019"),
    ("Paired pulses and trains", "pulses", "Pn / P1 against pulse number, one line per cell, the mean on top.",
     "Zucker & Regehr 2002"),
    ("DSI / DSE", "time", "Each episode's responses as % of its own baseline against time from the step; the "
     "suppression against step duration.", "Pitler & Alger 1992; Wilson & Nicoll 2001"),
    ("Field potentials and LTP", "time", "fEPSP slope as % of the baseline before induction, binned per minute, "
     "mean ± SEM across slices, the induction marked.", "Bliss & Lomo 1973"),
    ("Spontaneous events", "distribution", "Cumulative distributions per cell (or each cell's median compared), "
     "never pooled events tested as if independent.", "Aarts et al. 2014"),
    ("Intrinsic properties", "relationship", "Firing rate against injected current (F-I), one line per cell; "
     "passive properties compared per cell.", "Gouwens et al. 2019"),
    ("Spectra and time-frequency", "relationship", "Power against frequency, or the time-frequency map (heat "
     "map), its units and windows stated.", "Cohen 2014"),
)

FITS = {"": "No fitted curve", "linear": "Straight line", "exponential": "Single exponential",
        "boltzmann": "Boltzmann", "hill": "Hill"}

PALETTES = {
    "okabe-ito": ("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"),
    "tol-bright": ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"),
    "greys": ("#000000", "#555555", "#999999", "#CCCCCC"),
    "clampwise greens": ("#2d6a4f", "#95a5a6", "#6fbf93", "#1b4332", "#b7e4c7"),
    "ibm colour-blind": ("#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#000000"),
    "tableau 10": ("#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
                   "#9C755F", "#BAB0AC"),
    "dark2": ("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666"),
    "set2 (soft)": ("#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"),
    "paired": ("#1F78B4", "#A6CEE3", "#33A02C", "#B2DF8A", "#E31A1C", "#FB9A99", "#FF7F00", "#FDBF6F"),
    "viridis": ("#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"),
    "red and blue": ("#B2182B", "#2166AC", "#EF8A62", "#67A9CF", "#FDDBC7", "#D1E5F0"),
    "black and red": ("#000000", "#D62728", "#7F7F7F", "#FF9896"),
}

# whole looks in one click; each sets the Style's fields it names (the rest stay)
THEMES = {
    "Journal (classic)": dict(palette="okabe-ito", font="Arial", font_size=8.0, line_width=1.0, marker_size=4.0,
                              spines="open", grid="none", tick_dir="out", background="#FFFFFF", ink="#000000",
                              legend_frame=False, vary_markers=False, title_bold=False, label_style="A"),
    "Minimal": dict(palette="okabe-ito", spines="left-bottom-light", grid="y", tick_dir="none", background="#FFFFFF",
                    ink="#333333", legend_frame=False, vary_markers=False),
    "Boxed": dict(spines="box", grid="none", tick_dir="in", background="#FFFFFF", ink="#000000", legend_frame=True),
    "Grey grid": dict(palette="tableau 10", spines="none", grid="both", tick_dir="none", background="#EBEBEB",
                      grid_colour="#FFFFFF", ink="#222222", legend_frame=False),
    "Bold presentation": dict(palette="tol-bright", font_size=14.0, line_width=2.0, marker_size=7.0, spines="open",
                              grid="none", tick_dir="out", background="#FFFFFF", ink="#000000", title_bold=True),
    "Dark slide": dict(palette="tol-bright", font_size=12.0, line_width=1.6, marker_size=6.0, spines="open",
                       grid="y", tick_dir="out", background="#1E1E1E", ink="#E8E8E8", grid_colour="#3A3A3A"),
    "Greyscale print": dict(palette="greys", vary_markers=True, spines="open", grid="none", background="#FFFFFF",
                            ink="#000000"),
    "High contrast (colour-blind)": dict(palette="ibm colour-blind", vary_markers=True, line_width=1.5,
                                         marker_size=5.0, spines="open", background="#FFFFFF", ink="#000000"),
}

# journal sizes, width x height in mm (single, one and a half, double column)
SIZES = {"single": (85.0, 70.0), "one and a half": (114.0, 85.0), "double": (174.0, 110.0)}

# every control: (caption, the line under it, the help behind its "?", with an example)
CONTROLS = {
    "source": ("Table", "Where the numbers come from: a Clampwise table, a CSV or Excel file, a saved figure, or a "
               "picture of a plot", "Pick a table of this project (every recording's numbers, or one analysis's "
               "table), or open a file. Example: the DSI time course table, or an Excel sheet of your own."),
    "filters": ("Keep only rows where...", "Leave out the rows you do not want in this figure",
                "Each condition keeps the rows that meet it; all must hold. Example: Condition is control, or "
                "Amplitude (pA) < -20."),
    "window": ("Time window", "Only x values from ... to ... (blank: all of it)",
               "For a time course: the stretch to show. Example: -10 to 40 min around the induction."),
    "bin": ("Average into time bins of", "Each cell's values averaged within bins of this width (blank: no bins)",
            "Bins make cells recorded at different times line up. Example: 1 min bins for an LTP series."),
    "normalize": ("Show y relative to a baseline", "Each cell's y as % of its own mean over this x range "
                  "(blank: as measured)", "Example: % of the first 10 min (0 to 10), the baseline before a drug."),
    "level": ("Each point is", "A cell (its rows averaged first) or a row of the table",
              "Rows from one cell are not independent: averaging them into one value per cell first keeps the "
              "spread and n honest (Aarts et al. 2014). Choose a row only when each row is its own cell."),
    "x": ("X", "The column along the bottom", "Example: Time (min), Pulse, Current (pA)."),
    "y": ("Y", "The column measured", "Example: Amplitude (pA), slope as % of baseline."),
    "err": ("Error bars from", "A column holding each point's +/- error (blank: none)",
            "For values whose spread was computed elsewhere. Example: the SEM column of a summary table."),
    "y2": ("Second Y (right axis)", "A second measure on its own axis at the right (blank: none)",
           "For two measures with different units over one x. Example: Rs beside the amplitude over time."),
    "group": ("Groups", "The column whose values are the groups or conditions compared",
              "Example: Condition (control, drug)."),
    "cell": ("Cell", "The column naming each cell, so its rows can be averaged and joined",
             "In a Clampwise table: Animal, slice and cell together."),
    "unit": ("Each replicate is a", "What the Cell column names, as the caption says it",
             "Usually a cell. One cell's DSI episodes are episodes; an LTP series, a slice. The caption counts "
             "them by this name."),
    "animal": ("Animal", "The column naming each animal, so cells from one animal are counted as such",
               "Cells from one animal are not independent; with this, n is said as cells and animals, and "
               "statistics can resample animals then cells."),
    "panel_by": ("Split into side-by-side panels by", "One panel per value of this column (blank: one panel)",
                 "Example: one panel per sex."),
    "plot": ("Plot", "How the numbers are drawn", "The standard plot for the question is chosen for you; the "
             "others answer the same question another way."),
    "summary": ("Group centre and spread", "What the line and the bars show",
                "Mean ± SEM: how precisely the mean is known. Mean ± SD: how much cells differ. Median with IQR: "
                "for skewed data. 95% CI: the range of means compatible with the data (Cumming 2014)."),
    "paired": ("Connect values from the same cell (repeated measures)", "A line joins a cell's values across "
               "conditions", "For a paired design (the same cells before and after a drug)."),
    "fit": ("Fitted curve", "A curve fitted to the points, its equation in the legend",
            "Straight line, single exponential, Boltzmann or Hill; least squares on the points shown."),
    "stats": ("Statistics", "The difference between groups with its 95% CI, and the test a reviewer asks for",
              "Effect sizes with confidence intervals by bootstrap over cells (or animals then cells), Welch's t "
              "and Mann-Whitney on cell values, paired t and Wilcoxon for paired designs."),
    "x_label": ("X axis title", "With its units in brackets", "Example: Time (min)."),
    "y_label": ("Y axis title", "With its units in brackets", "Example: fEPSP slope (% of baseline)."),
    "x_range": ("X range", "From ... to ... (blank: fit the data)", "Example: -10, 60."),
    "y_range": ("Y range", "From ... to ... (blank: fit the data)", "Example: 0, 200."),
    "log": ("Log axes", "A logarithmic x or y axis", "For values spanning orders of magnitude (intervals, rates)."),
    "y_from_zero": ("Y axis starts at zero", "A bar's height means something only from zero",
                    "On for bar charts; off lets other plots show where the values are."),
    "scale_bars": ("Scale bars instead of axes", "For traces: bars with their size written, no axes",
                   "Example: 50 pA and 20 ms bars."),
    "palette": ("Colours", "Colour-blind-safe sets by default", "Okabe-Ito and Tol's bright set read the same "
                "in the common colour-vision deficiencies."),
    "marker_size": ("Point size", "In points", ""),
    "line_width": ("Line width", "In points", ""),
    "font": ("Font", "The figure's font, embedded in PDF and EPS", "Arial or Helvetica for most journals."),
    "font_size": ("Text size", "In points; 7 to 9 for a printed figure", ""),
    "title": ("Panel title", "Above the panel (blank: none)", ""),
    "legend": ("Legend", "Where the legend sits (or none)", ""),
    "size": ("Figure size", "A journal's column width", "Single column 85 mm, one and a half 114 mm, double 174 "
             "mm."),
    "theme": ("Theme", "A whole look in one click: colours, fonts, axes, grid", "Each theme sets the colours, text "
              "size, axis lines, grid, ticks and background; change any of them after."),
    "spines": ("Axis lines", "Which sides of the plot have a line", "Open (left and bottom), a box, light, or none."),
    "grid": ("Grid", "Light lines across the plot at the ticks", "None, along y, along x, or both."),
    "tick_dir": ("Ticks", "Tick marks outside, inside, or none", ""),
    "background": ("Background", "The figure's colour; transparent for slides", "White for print; transparent "
                   "lets a slide's colour show through (PNG, SVG, PDF)."),
    "ink": ("Text and axis colour", "The colour of text, ticks and axis lines", "Black for print; light grey on a "
            "dark slide."),
    "label_style": ("Panel letters", "How panels are labelled", "A, a, (A) or A."),
    "legend_frame": ("Frame around the legend", "", ""),
    "legend_cols": ("Legend columns", "Entries side by side", ""),
    "title_bold": ("Bold titles", "", ""),
    "vary_markers": ("A different marker for each group", "For greyscale print and colour-blind readers", ""),
    "point_alpha": ("Point transparency", "1 is solid; lower shows overlapping points", ""),
    "jitter": ("Spread of the points", "How far points spread sideways so none hides another", ""),
    "cap_size": ("Error bar caps", "Width of the caps in points (0: none)", ""),
    "error_style": ("Spread drawn as", "Error bars, a shaded band, or none", ""),
    "show_points": ("Show every point", "The individual cells (or rows) behind the summary",
                    "Keep on for small samples: a summary alone hides the data (Weissgerber et al. 2015)."),
    "series": ("Groups' look", "Each group's colour, marker, line and name in the legend; their order",
               "Click a colour to choose it; order changes where each group sits."),
    "size_by": ("Point size from", "A column whose values set each point's size (bubble chart)", ""),
    "percent": ("Show percentages", "Each category's share of its group, instead of the count", ""),
    "row_unit": ("A row of the table is", "What one row is, in the caption's words (a response window, a sweep, an "
                 "event) when the points are rows and not cells", ""),
    "event": ("The category counted", "For a proportion: which category of x each group's share is of (connected, "
              "success); the first one when not named", "Wilson's score interval: Wilson 1927; Brown, Cai & "
              "DasGupta 2001"),
    "ref_lines": ("Reference lines", "Horizontal or vertical lines at given values, each with an optional label",
                  "A line at a value on either axis, with its name. Example: a horizontal line at 100 named "
                  "baseline. Add with + Add, remove with Remove selected, the right-click menu or Delete."),
    "x_rotate": ("X labels' angle", "Turn the x axis labels (degrees), for long group names", ""),
    "minor_ticks": ("Minor ticks", "Small ticks between the labelled ones", ""),
    "tick_size": ("Tick labels' size", "In points (0: a little under the text size)", ""),
    "label_size": ("Axis titles' size", "In points (0: the text size)", ""),
    "title_size": ("Panel titles' size", "In points (0: the text size)", ""),
    "letter_size": ("Panel letters' size", "In points (0: the text size plus 2)", "Nature: 8 pt bold lower case."),
    "annotation_size": ("Text and scale bars' size", "In points (0: a little under the text size)", ""),
    "legend_size": ("Legend's size", "In points (0: a little under the text size)", ""),
    "axis_width": ("Axis lines' width", "In points", "0.5 to 1 pt prints well."),
    "letter_bold": ("Bold panel letters", "", ""),
    "zero_axes": ("Axes cross at zero", "The axes drawn through 0, 0 instead of at the left and bottom",
                  "For a current-voltage plot, where the sign matters."),
    "clip_y": ("Cut the traces at", "From ... to ... in y's units: what goes beyond is not drawn (blank: all of it)",
               "Cut the action potentials when the subthreshold response is the point: 'spikes truncated' is "
               "then said in the caption. Example: -90, -20."),
    "stats_text": ("Brackets say", "The exact p value, stars, or both", "Journals ask for exact p values; "
                   "J Physiol does not accept asterisks alone."),
    "test": ("Test", "Which test's p value the brackets and the caption carry",
             "Automatic: Welch's t test for separate groups, the paired t test for the same cells in two "
             "conditions. The rank tests (Mann-Whitney, Wilcoxon) assume no normal distribution. The other of each "
             "pair is always reported beside."),
    "pairs": ("Compare", "Each group with the first, or every pair of groups", "p values are Holm-corrected over "
              "the comparisons made."),
    "bin_width": ("Histogram bins of", "The width of the rate histogram's bins under a raster, in x's units "
                  "(blank: chosen for you)", ""),
}


def pack(frame: pd.DataFrame | None) -> str:
    """A table as text inside a recipe: CSV, gzipped, base64."""
    if frame is None:
        return ""
    return base64.b64encode(gzip.compress(frame.to_csv(index=False).encode("utf-8"))).decode("ascii")


def unpack(text: str) -> pd.DataFrame | None:
    if not text:
        return None
    return pd.read_csv(io.StringIO(gzip.decompress(base64.b64decode(text)).decode("utf-8")))


@dataclass
class Recipe:
    """One panel: its data, the question, the plot and its look."""
    source: dict = field(default_factory=dict)   # {"kind": "project"|"file"|"figure"|"digitized", "table", "path"...}
    data: str = ""                               # the table it was drawn from (``pack``)
    question: str = "compare"
    plot: str = "points"
    x: str = ""
    y: str = ""
    y2: str = ""
    err: str = ""                                # a column of +/- errors drawn as bars on the points
    group: str = ""
    cell: str = ""
    animal: str = ""
    panel_by: str = ""
    level: str = "cell"                          # "cell" or "row"
    unit: str = "cell"                           # what one replicate is called in the caption (cell, episode...)
    filters: list = field(default_factory=list)  # [column, operator, value]
    window: list = field(default_factory=list)   # [from, to] on x
    bin: float = 0.0
    normalize: list = field(default_factory=list)  # [from, to] on x: y as % of each cell's mean there
    summary: str = "mean_sem"
    paired: bool = False
    fit: str = ""
    stats: bool = False
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    x_range: list = field(default_factory=list)
    y_range: list = field(default_factory=list)
    x_log: bool = False
    y_log: bool = False
    y_from_zero: bool | None = None              # None: on for bars, off otherwise
    scale_bars: bool = False
    legend: str = "best"                         # matplotlib location, or "none"
    annotations: list = field(default_factory=list)   # {"kind": "text"|"arrow", "x", "y", "text", "x2", "y2"}
    markers: list = field(default_factory=list)  # {"at": x, "end": x2 or None, "label": "drug"}: lines and spans on x
    notes: list = field(default_factory=list)    # sentences the caption adds (a layout says how it chose)
    series: dict = field(default_factory=dict)   # group -> {"colour", "marker", "line", "width", "filled", "label",
    #                                                       "hidden"}: each group's own look
    order: list = field(default_factory=list)    # the groups in the order drawn (blank: as in the table)
    show_points: bool = True
    error_style: str = "bar"                     # "bar", "band", "none"
    ref_lines: list = field(default_factory=list)   # {"axis": "x"|"y", "at": value, "label": ""}
    size_by: str = ""                            # a column for a bubble chart's point size
    percent: bool = False                        # counts as % of each group
    event: str = ""                              # a proportion: the category of x counted ("": the first)
    row_unit: str = ""                           # what one row of the table is, for the caption ("response window")
    x_rotate: float = 0.0
    # ---- where the panel sits (figure/page.py)
    span: list = field(default_factory=list)     # [columns, rows] it takes in the grid ([]: one cell)
    rect: list = field(default_factory=list)     # [x, y, width, height] mm on the page, when placed freely
    inset_of: int = -1                           # the panel it sits inside (-1: none)
    inset_rect: list = field(default_factory=list)   # [x, y, width, height] as fractions of that panel's plot area
    letter: str = ""                             # the panel's letter ("": the next one; "-": none)
    letter_offset: list = field(default_factory=list)   # [dx, dy] mm from the panel's top left corner, when moved
    legend_at: list = field(default_factory=list)    # [x, y] of the legend's upper left corner as axes fractions,
    #                                                  when dragged (then ``legend``'s location is not used)
    # ---- electrophysiology (figure/elements.py)
    scale_bar: dict = field(default_factory=dict)    # {"x", "y" lengths, "at": [fx, fy], "width", "x_label",
    #                                                  "y_label", "show_x", "show_y", "corner"}: blank is automatic
    stim_bars: list = field(default_factory=list)    # {"start", "end", "nm" or "colour", "kind", "label", "y",
    #                                                  "height", "hidden"}: the light's (or a shock's) time over a trace
    command: dict = field(default_factory=dict)      # {"show": bool, "share": 0.22}: the command under the trace
    #                                                  (from the y2 column of a trace panel's table)
    vm_labels: dict = field(default_factory=dict)    # trace -> what is written at its start ("-65 mV")
    show_vm: bool = True
    clip_y: list = field(default_factory=list)       # [low, high]: traces cut there (spikes truncated)
    stack: bool = False                              # traces one under the other instead of overlaid
    stack_gap: float = 0.0                           # how far apart, in y's units (0: a little over their range)
    zero_axes: bool = False                          # the axes cross at 0, 0 (I-V plots)
    brackets: list = field(default_factory=list)     # {"a", "b", "text", "y", "hidden"}: a comparison's bracket as
    #                                                  set by hand (the others are placed automatically)
    stats_text: str = "p"                            # what a bracket says: "p" (exact), "stars", "both"
    test: str = ""                                   # which test's p a bracket carries (stats.TESTS; "": the
    #                                                  usual one for the design: Welch's t, or the paired t)
    pairs: str = "first"                             # groups compared: each with the "first", or "all" pairs
    excluded: list = field(default_factory=list)     # rows of the table left out by hand (their row numbers)
    bin_width: float = 0.0                           # a raster's histogram bins, in x's units (0: automatic)

    def frame(self) -> pd.DataFrame | None:
        return unpack(self.data)


@dataclass
class Style:
    """The look, shared by every panel; saved and applied by name."""
    palette: str = "okabe-ito"
    marker_size: float = 4.0
    line_width: float = 1.0
    font: str = "Arial"
    font_size: float = 8.0
    fill_alpha: float = 0.25
    theme: str = "Journal (classic)"
    spines: str = "open"                         # "open", "box", "left-bottom-light", "none"
    grid: str = "none"                           # "none", "y", "x", "both"
    grid_colour: str = "#DDDDDD"
    tick_dir: str = "out"                        # "out", "in", "none"
    background: str = "#FFFFFF"                  # a colour, or "transparent"
    ink: str = "#000000"                         # text, ticks and axis lines
    legend_frame: bool = False
    legend_cols: int = 1
    title_bold: bool = False
    label_style: str = "A"                       # "A", "a", "(A)", "A."
    vary_markers: bool = False
    point_alpha: float = 0.85
    jitter: float = 0.12
    cap_size: float = 2.5
    minor_ticks: bool = False
    # each kind of text its own size in points; 0 follows ``font_size``
    tick_size: float = 0.0
    label_size: float = 0.0
    title_size: float = 0.0
    letter_size: float = 0.0
    annotation_size: float = 0.0
    legend_size: float = 0.0
    axis_width: float = 0.8
    letter_bold: bool = True
    group_colours: dict = field(default_factory=dict)   # a group's (a condition's) colour wherever it is drawn:
    #                                                     the same condition, the same colour, in every panel

    def apply_theme(self, name: str) -> None:
        for k, v in THEMES.get(name, {}).items():
            setattr(self, k, v)
        self.theme = name

    def size_of(self, what: str) -> float:
        """The size in points of ``what``: tick, label, title, letter, annotation or legend."""
        own = float(getattr(self, f"{what}_size", 0.0) or 0.0)
        if own > 0:
            return own
        return {"tick": 0.9, "label": 1.0, "title": 1.0, "annotation": 0.9, "legend": 0.9}.get(what, 1.0) \
            * self.font_size + (2.0 if what == "letter" else 0.0)

    def apply_journal(self, key: str) -> None:
        """The journal's font and sizes (figure/journals.py): text, panel letters and their case, least line."""
        from .journals import JOURNALS, MIN_LINE_PT

        j = JOURNALS.get(key)
        if j is None:
            return
        self.font, self.font_size = j.font, j.text_pt
        self.tick_size = self.label_size = self.title_size = self.annotation_size = self.legend_size = 0.0
        if j.min_pt and self.size_of("tick") < j.min_pt:         # the smallest text still within the journal's range
            self.tick_size = self.annotation_size = self.legend_size = j.min_pt
        self.letter_size, self.letter_bold = j.letter_pt, j.letter_bold
        self.label_style = j.letter_style if self.label_style in ("A", "a") else self.label_style
        self.line_width = max(self.line_width, j.min_line_pt or MIN_LINE_PT)
        self.axis_width = max(min(self.axis_width, 0.8), j.min_line_pt or MIN_LINE_PT)

@dataclass
class Spec:
    """A whole figure: its panels (in a grid, placed freely, or inset in one another: figure/page.py), the page
    they sit on, and the look they share."""
    panels: list[Recipe] = field(default_factory=lambda: [Recipe()])
    columns: int = 1
    size: str = "single"                         # a column width's name ("single", "one and a half", "double"),
    #                                              or "w x h" in mm for a page of any size
    labels: bool = True
    style: Style = field(default_factory=Style)
    format: str = FORMAT
    version: int = VERSION
    journal: str = ""                            # a journals.JOURNALS key: whose column widths ``size`` names
    #                                              ("": Clampwise's own 85 / 114 / 174 mm)
    height_mm: float = 0.0                       # the page's height (0: as high as its rows need)
    col_ratios: list = field(default_factory=list)   # each grid column's share of the width ([]: equal)
    row_ratios: list = field(default_factory=list)   # each grid row's share of the height
    texts: list = field(default_factory=list)    # words on the page itself: {"x", "y" mm, "text", "size", "bold"}
    snap_mm: float = 0.0                         # a panel dragged or resized lands on this grid (0: anywhere)

    def width_mm(self) -> float:
        if self.size in SIZES:
            from .journals import width_mm

            w = width_mm(self.journal, self.size) if self.journal else None
            return float(w if w else SIZES[self.size][0])
        return float(self.size.partition("x")[0])

    def page_mm(self, rows: int | None = None) -> tuple[float, float]:
        """The page's (width, height) in mm; its height the one set, else what ``rows`` grid rows need (never
        over the journal's limit)."""
        from . import page
        from .journals import JOURNALS

        if self.size not in SIZES:
            w, _, h = self.size.partition("x")
            return float(w), float(h)
        w = self.width_mm()
        if self.height_mm and self.height_mm > 0:
            return w, float(self.height_mm)
        rows = page.rows_used(self) if rows is None else rows
        h = page.auto_height(w, max(1, self.columns), rows, self.row_ratios)
        limit = JOURNALS[self.journal].max_height_mm if self.journal in JOURNALS else None
        return w, min(h, limit) if limit else h

    def size_mm(self) -> tuple[float, float]:
        return self.page_mm()

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "Spec":
        d = json.loads(text)
        if d.get("format") != FORMAT:
            raise ValueError("not a Clampwise figure")
        known = {f.name for f in fields(Recipe)}
        panels = [Recipe(**{k: v for k, v in p.items() if k in known}) for p in d.get("panels", [])]
        style = Style(**{k: v for k, v in d.get("style", {}).items() if k in {f.name for f in fields(Style)}})
        own = {f.name for f in fields(cls)} - {"panels", "style", "format", "version"}
        spec = cls(panels or [Recipe()], style=style, **{k: v for k, v in d.items() if k in own})
        if int(d.get("version", 1)) < 2:         # a figure saved before letters had their own size: as it looked
            spec.style.letter_size = spec.style.letter_size or spec.style.font_size + 3
        return spec
