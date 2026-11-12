"""The figure as a page (no window): panels that span, are placed freely or inset; journals' sizes and the
check; each panel fitted inside its rectangle; what a slice figure is made of (the light's bar, scale bars at
round values, the command under the trace, the potential at a trace's start, spikes cut, a raster, axes through
zero); things that can be selected and found under a point; the choice of test; rows left out by hand; one
colour per condition; an export at the exact size with its text kept as text."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from clampwise import simulate
from clampwise.analysis.stimulus import find_stimulus
from clampwise.figure import elements, journals, layouts, page, render, stats, thumbs, traces
from clampwise.figure.recipe import PLOTS, Recipe, Spec, pack

from test_figure import nested


def cells(effect: float = 3.0) -> pd.DataFrame:
    f = nested(effect=effect, animals=4, cells=3, events=6, seed=4)
    f["Time (min)"] = np.tile(np.arange(6.0), len(f) // 6)
    return f


def compare(**kw) -> Recipe:
    return Recipe(**{**dict(data=pack(cells()), question="compare", plot="points", y="Amplitude (pA)",
                            group="Condition", cell="Cell", animal="Animal"), **kw})


def light_panel(**kw) -> tuple[Recipe, object]:
    sim = simulate.photostim_steps(seed=3)
    rec = sim.recording
    rec.stimulus = find_stimulus(rec)
    sets = [traces.TraceSet("0 pA", "average", [["", 5, 1]]), traces.TraceSet("-80 pA", "average", [["", 3, 1]])]
    r = layouts.example_recipe(sets, traces.Window(20, 160, True, 0.0), lambda _f: rec, "sim", "Light-evoked PSPs")
    for k, v in kw.items():
        setattr(r, k, v)
    return r, rec


def inside(fig, d, slack: float = 1.5) -> bool:
    """Whether everything the panel draws lies inside its rectangle (to ``slack`` pixels)."""
    renderer = render._renderer(fig)
    x0, y0, x1, y1 = render.panel_outline(fig, d)
    for ax in d.axes:
        bb = ax.get_tightbbox(renderer)
        if bb is not None and not (bb.x0 >= x0 - slack and bb.y0 >= y0 - slack and bb.x1 <= x1 + slack
                                   and bb.y1 <= y1 + slack):
            return False
    return True


# ---------------------------------------------------------------- where the panels sit

def test_panels_flow_into_the_grid_and_span_cells():
    found, rows = page.cells([(2, 1), (1, 1), (1, 1), (1, 2), (1, 1)], 2)
    assert found == [(0, 0, 2, 1), (0, 1, 1, 1), (1, 1, 1, 1), (0, 2, 1, 2), (1, 2, 1, 1)] and rows == 4
    rects = page.grid_rects(found, 2, rows, 176.0, 200.0)
    assert rects[0][2] == pytest.approx(175.0) and rects[1][2] == pytest.approx((175.0 - page.GUTTER_MM) / 2)
    assert rects[3][3] == pytest.approx(2 * rects[4][3] + page.GUTTER_MM)             # two rows and the gutter
    wide = page.grid_rects(found[:3], 2, 2, 176.0, 100.0, col_ratios=(2, 1), row_ratios=(1, 3))
    assert wide[1][2] == pytest.approx(2 * wide[2][2]) and wide[1][3] == pytest.approx(3 * wide[0][3])
    assert page.cells([(5, 1)], 2)[0] == [(0, 0, 2, 1)]                              # never wider than the grid


def test_a_template_arranges_the_panels_and_free_placement_keeps_them_where_they_are():
    spec = Spec([compare(), compare(), compare(), compare()], size="double")
    page.apply_template(spec, "wide_top_3")
    rects, w, h = page.layout(spec)
    assert w == 174.0 and spec.columns == 3 and [r.span for r in spec.panels] == [[3, 1], [1, 1], [1, 1], [1, 1]]
    assert rects[0][2] == pytest.approx(w - 2 * page.MARGIN_MM) and rects[1][1] > rects[0][1] + rects[0][3]
    assert all(0 <= x and 0 <= y and x + rw <= w + 1e-6 and y + rh <= h + 1e-6 for x, y, rw, rh in rects)
    for key, t in page.TEMPLATES.items():                                            # every template lays out
        s = Spec([compare() for _ in range(t.n)], size="double")
        page.apply_template(s, key)
        got, tw, th = page.layout(s)
        assert len(got) == t.n and all(r is not None and r[2] > 10 and r[3] > 10 for r in got), key
        for i, a in enumerate(got):                                                  # and no two panels overlap
            for b in got[i + 1:]:
                assert a[0] + a[2] <= b[0] + 1e-6 or b[0] + b[2] <= a[0] + 1e-6 or a[1] + a[3] <= b[1] + 1e-6 \
                    or b[1] + b[3] <= a[1] + 1e-6, key
    before = [list(r) for r in rects]
    page.to_free(spec)
    assert [[round(v, 2) for v in r] for r in before] == [r.rect for r in spec.panels]
    spec.panels[1].rect = [10.0, 60.0, 70.0, 30.0]
    assert page.layout(spec)[0][1] == [10.0, 60.0, 70.0, 30.0] and page.layout(spec)[0][2] == spec.panels[2].rect
    page.to_grid(spec)
    assert all(r.rect == [] for r in spec.panels)


def test_rectangles_are_snapped_clamped_aligned_and_spaced():
    assert page.snap([10.2, 4.9, 20.4, 9.7], 1.0) == [10.0, 5.0, 21.0, 10.0]         # each edge on the grid
    assert page.snap([10.2, 4.9, 20.3, 9.6], 0) == [10.2, 4.9, 20.3, 9.6]
    assert page.clamp([-5, 300, 50, 50], 176, 120) == [0.0, 70.0, 50.0, 50.0]
    assert page.clamp([0, 0, 2, 500], 176, 120) == [0.0, 0.0, page.MIN_PANEL_MM, 120.0]
    r = [[10, 10, 40, 30], [70, 25, 20, 20], [130, 40, 30, 10]]
    assert [x[1] for x in page.arrange(r, "top")] == [10, 10, 10]
    assert [x[0] + x[2] for x in page.arrange(r, "right")] == [50, 50, 50]
    assert [x[2] for x in page.arrange(r, "same_width")] == [40, 40, 40]
    spread = page.arrange(r, "spread_x")
    gaps = [spread[1][0] - (spread[0][0] + spread[0][2]), spread[2][0] - (spread[1][0] + spread[1][2])]
    assert gaps[0] == pytest.approx(gaps[1]) and spread[0][0] == 10 and spread[2][0] + spread[2][2] == 160


def test_every_panel_is_drawn_inside_its_rectangle_with_its_letter_at_the_corner():
    trace, _rec = light_panel()
    spec = Spec([trace, compare(stats=True), compare(plot="box"),
                 Recipe(data=pack(cells()), question="time", plot="time_course", x="Time (min)", y="Amplitude (pA)",
                        group="Condition", cell="Cell", title="A long title above the panel")], size="double")
    page.apply_template(spec, "wide_top_3")
    fig, drawn = render.build(spec)
    assert [d.letter for d in drawn] == ["A", "B", "C", "D"] and all(inside(fig, d) for d in drawn)
    fw, fh = fig.bbox.width, fig.bbox.height
    marks = {t.panel: a for t, a in render.things(fig) if t.kind == "letter"}
    for k, d in enumerate(drawn):                    # the letter sits at the same place of every panel's rectangle
        x0, _y0, _x1, y1 = render.panel_outline(fig, d)
        bb = marks[k].get_window_extent(render._renderer(fig))
        assert bb.x0 - x0 == pytest.approx(render.LETTER_PAD_MM / 174.0 * fw, abs=1.0)
        assert y1 - bb.y1 == pytest.approx(render.LETTER_PAD_MM / spec.size_mm()[1] * fh, abs=1.5)
    spec.panels[1].letter, spec.panels[2].letter = "-", "x"       # none, and one of its own
    assert render.letters(spec) == ["A", "", "x", "B"]
    spec.labels = False
    assert render.letters(spec) == ["", "", "", ""]


def test_an_inset_sits_inside_its_panel_and_a_freely_placed_panel_where_it_was_put():
    big = compare()
    small = compare(plot="box", inset_of=0, inset_rect=[0.55, 0.55, 0.4, 0.4], legend="none")
    free = compare(rect=[100.0, 5.0, 60.0, 40.0])
    spec = Spec([big, small, free], size="double", height_mm=60.0)
    assert page.is_inset(spec.panels, 1) and not page.is_inset(spec.panels, 0)
    rects, w, h = page.layout(spec)
    assert rects[1] is None and rects[2] == [100.0, 5.0, 60.0, 40.0] and (w, h) == (174.0, 60.0)
    fig, drawn = render.build(spec)
    r = render._renderer(fig)
    outer, inner = drawn[0].axes[0].get_window_extent(r), drawn[1].axes[0].get_window_extent(r)
    assert outer.x0 < inner.x0 and inner.x1 <= outer.x1 + 1 and inner.y1 <= outer.y1 + 1
    assert inner.width == pytest.approx(0.4 * outer.width, rel=0.02)
    x0, y0, x1, y1 = render.panel_outline(fig, drawn[2])
    assert (x0, x1) == pytest.approx((100 / 174 * fig.bbox.width, 160 / 174 * fig.bbox.width))
    assert render.letters(spec) == ["A", "", "B"]                                  # an inset takes no letter
    spec.panels[1].inset_of = 7                                                      # its panel gone: a panel again
    assert not page.is_inset(spec.panels, 1) and page.layout(spec)[0][1] is not None


# ---------------------------------------------------------------- journals

def test_a_journal_sets_the_widths_the_text_and_what_is_checked():
    spec = Spec([compare()], journal="jneurosci", size="double")
    assert spec.width_mm() == 176.0 and Spec([compare()], journal="nature", size="single").width_mm() == 89.0
    assert Spec([compare()], size="one and a half").width_mm() == 114.0             # no journal: Clampwise's own
    spec.style.apply_journal("nature")
    s = spec.style
    assert (s.font, s.font_size, s.letter_size, s.label_style) == ("Arial", 6.5, 8.0, "a") and s.size_of("tick") >= 5.0
    tall = Spec([compare() for _ in range(8)], journal="nature", size="single")
    assert tall.size_mm()[1] == 170.0                                                # never over the journal's height
    assert {j.name for j in journals.JOURNALS.values()} >= {"J Neurosci", "eLife", "Nature", "Nature Neuroscience",
                                                            "J Physiol", "Cerebral Cortex"}
    assert not journals.JOURNALS["neuron"].verified                                  # said where it could not be read
    found = journals.check("nature", 200.0, 180.0, [("tiny", 4.0, "A"), ("fine", 6.0, "A"), ("huge", 12.0, "B")],
                           [(0.2, "A"), (1.0, "B")], "png")
    said = " | ".join(f.text for f in found)
    assert "200 mm wide" in said and "180 mm high" in said and "under 5 pt" in said and "over Nature's 7 pt" in said
    assert "thinner than 0.5 pt" in said and "not PNG" in said
    assert journals.check("nature", 89.0, 100.0, [("ok", 6.0, "")], [(0.8, "")], "pdf") == []
    assert [f.text for f in journals.check("", 500.0, 500.0, [("x", 3.0, "")], [])][0].startswith("1 piece(s) of text")


def test_the_check_reads_the_figure_as_drawn():
    spec = Spec([compare()], journal="jneurosci", size="single")
    spec.style.apply_journal("jneurosci")
    assert render.check(spec, "pdf") == []
    spec.style.tick_size = 3.5
    found = render.check(spec, "pdf")
    assert len(found) == 1 and found[0].serious and "under 5 pt" in found[0].text
    spec.style.tick_size, spec.size = 0.0, "230x60"
    assert "J Neurosci's widest is 176 mm" in render.check(spec)[0].text


# ---------------------------------------------------------------- what a slice figure is made of

def test_traces_come_with_the_light_s_bars_the_potential_and_the_command():
    r, rec = light_panel()
    f = r.frame()
    assert list(f.columns) == ["Time (ms)", "Voltage (mV)", "Trace", "Sweep", "Command (pA)"]
    assert r.y2 == "Command (pA)" and traces.command_of(f) == "Command (pA)"
    assert r.stim_bars == [{"start": 0.0, "end": 5.0, "nm": 470.0}, {"start": 105.0, "end": 110.0, "nm": 470.0}]
    assert r.vm_labels == {"0 pA": "-60 mV", "-80 pA": "-68 mV"}                    # where each trace sat
    assert r.command == {"show": False}                 # held steady through this window: nothing to draw under
    assert any("Bars over the traces: the light" in n for n in r.notes)
    # a step family: the command moves within the traces, so it is drawn under them
    sim = simulate.firing_steps()
    made = layouts.step_examples({"intrinsic": __import__("clampwise.analysis.pipeline", fromlist=["x"]).analyze(
        sim.recording).results["intrinsic"]}, sim.recording)
    steps = layouts.example_recipe(made[0], made[1], lambda _f: sim.recording, "steps")
    assert steps.command == {"show": True} and steps.stim_bars == []
    fig, drawn = render.build(Spec([steps], size="double"))
    assert len(drawn[0].axes) == 2                                                   # the traces, and their command
    assert {t.kind for t, _a in render.things(fig)} >= {"command", "scalebar", "vmlabel", "series"}
    upper, lower = (a.get_position() for a in drawn[0].axes)
    assert lower.y1 <= upper.y0 + 1e-6 and lower.height < upper.height
    assert inside(fig, drawn[0])


def test_the_light_s_bar_is_drawn_at_its_time_in_its_colour():
    blue = elements.wavelength_colour(470)
    assert blue[1:3] == "00" and blue[5:] == "FF" and elements.wavelength_colour(630)[1:3] == "FF"
    assert elements.wavelength_colour(1040) == "#8B0000" and elements.wavelength_colour(590)[5:] == "00"
    assert elements.stim_colour({"colour": "#123456", "nm": 470}) == "#123456"
    assert elements.stim_colour({"kind": "electrical"}) == "#000000"
    r, _rec = light_panel()
    r.stim_bars[0]["label"] = "470 nm, 5 ms"
    fig, drawn = render.build(Spec([r], size="double"))
    bars = [(t, a) for t, a in render.things(fig) if t.kind == "stimbar" and not t.part]
    assert [t.key for t, _a in bars] == ["0", "1"]
    first = bars[0][1]
    assert (first.get_x(), first.get_width()) == (0.0, 5.0) and first.get_y() > 1.0     # over the plot, at its time
    assert first.get_facecolor()[:3] == pytest.approx(tuple(int(blue[i:i + 2], 16) / 255 for i in (1, 3, 5)))
    assert any(t.part == "label" and a.get_text() == "470 nm, 5 ms" for t, a in render.things(fig))
    title = next(a for t, a in render.things(fig) if t.kind == "title")
    rr = render._renderer(fig)
    assert title.get_window_extent(rr).y0 >= first.get_window_extent(rr).y1 - 1       # the title clears the bars
    # a shock is a tick; a bar hidden is not drawn
    r.stim_bars = [{"start": 10.0, "end": 10.0, "kind": "electrical"}, {"start": 50.0, "end": 60.0, "hidden": True}]
    fig, _ = render.build(Spec([r], size="double"))
    drawn_bars = [a for t, a in render.things(fig) if t.kind == "stimbar"]
    assert len(drawn_bars) == 1 and list(drawn_bars[0].get_xdata()) == [10.0, 10.0]


def test_scale_bars_have_round_lengths_that_can_be_set_and_moved():
    assert [elements.nice(v) for v in (0.37, 1.4, 7.9, 23.0, 180.0)] == [0.5, 1.0, 10.0, 20.0, 200.0]
    r, _rec = light_panel()
    fig, drawn = render.build(Spec([r], size="double"))
    ax = drawn[0].axes[0]
    parts = {t.part: a for t, a in render.things(fig) if t.kind == "scalebar" and t.key == ""}
    xs, ys = parts["x"].get_xdata(), parts["y"].get_ydata()
    assert xs[1] - xs[0] == 50.0 and parts["x_label"].get_text() == "50 ms"          # a fifth of 180 ms, rounded
    assert ys[1] - ys[0] in (1.0, 2.0) and parts["y_label"].get_text().endswith(" mV")
    lowest = min(float(np.nanmin(line.get_ydata())) for t, line in render.things(fig)
                 if t.kind == "series" and hasattr(line, "get_ydata"))
    assert ys[0] < lowest                                                            # under the traces, not on them
    assert not ax.axison
    r.scale_bar = {"x": 20.0, "y": 5.0, "x_label": "20 ms", "y_label": "5 mV", "at": [0.5, 0.5], "width": 2.0,
                   "corner": "left"}
    fig, drawn = render.build(Spec([r], size="double"))
    parts = {t.part: a for t, a in render.things(fig) if t.kind == "scalebar" and t.key == ""}
    ax = drawn[0].axes[0]
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    assert list(parts["x"].get_xdata()) == pytest.approx([(x0 + x1) / 2, (x0 + x1) / 2 + 20.0])
    assert list(parts["y"].get_ydata()) == pytest.approx([(y0 + y1) / 2, (y0 + y1) / 2 + 5.0])
    assert parts["x"].get_linewidth() == 2.0 and parts["y_label"].get_ha() == "right"
    r.scale_bar = {"show_y": False}
    fig, _ = render.build(Spec([r], size="double"))
    assert {t.part for t, _a in render.things(fig) if t.kind == "scalebar" and t.key == ""} == {"x", "x_label"}
    r.scale_bars = False                                                             # axes instead
    fig, drawn = render.build(Spec([r], size="double"))
    assert drawn[0].axes[0].axison and not any(t.kind == "scalebar" for t, _a in render.things(fig))


def test_traces_stacked_cut_and_labelled():
    r, _rec = light_panel(stack=True)
    fig, _ = render.build(Spec([r], size="double"))
    lines = {t.key: a for t, a in render.things(fig) if t.kind == "series" and not t.part and hasattr(a, "get_ydata")}
    top, low = np.nanmedian(lines["0 pA"].get_ydata()), np.nanmedian(lines["-80 pA"].get_ydata())
    assert top - low > 3.0                                                           # one under the other
    labels = {t.key: a.get_text().strip() for t, a in render.things(fig) if t.kind == "vmlabel"}
    assert labels == {"0 pA": "-60 mV", "-80 pA": "-68 mV"}
    r.show_vm, r.stack, r.clip_y = False, False, [-1.0, 1.5]
    fig, _ = render.build(Spec([r], size="double"))
    assert not any(t.kind == "vmlabel" for t, _a in render.things(fig))
    ys = np.concatenate([a.get_ydata() for t, a in render.things(fig) if t.kind == "series" and not t.part
                         and hasattr(a, "get_ydata")])
    assert np.nanmax(ys) <= 1.5 and np.isnan(ys).any()                               # what lies beyond is not drawn


def test_a_raster_with_its_histogram_and_axes_through_zero():
    r = thumbs.sample("raster")
    fig, drawn = render.build(Spec([r]))
    upper, lower = drawn[0].axes
    assert lower.get_ylabel() == "Rate (Hz)" and upper.get_ylim()[0] > upper.get_ylim()[1]      # sweep 1 on top
    n = len(r.frame())
    width = lower.patches[0].get_width()
    assert sum(p.get_height() for p in lower.patches) * width * 8 == pytest.approx(n, rel=1e-6)   # 8 sweeps, in Hz
    iv = Recipe(data=pack(pd.DataFrame({"V (mV)": [-80.0, -40.0, 0.0, 40.0], "I (pA)": [-60.0, -20.0, 5.0, 50.0]})),
                question="relationship", plot="line", x="V (mV)", y="I (pA)", level="row", zero_axes=True)
    fig, drawn = render.build(Spec([iv]))
    ax = drawn[0].axes[0]
    assert ax.spines["left"].get_position() == "zero" and ax.spines["bottom"].get_position() == "zero"
    assert not ax.spines["top"].get_visible()


def test_every_plot_has_a_picture_to_choose_it_by():
    for _q, _words, plots in thumbs.catalogue():
        for plot, name, what in plots:
            assert name and what and plot in PLOTS
            assert thumbs.png(plot)[:8] == b"\x89PNG\r\n\x1a\n", plot
    assert thumbs.png("bar") is thumbs.png("bar")                                    # made once


# ---------------------------------------------------------------- selecting, finding, moving

def test_everything_drawn_can_be_found_under_a_point():
    r = compare(stats=True, title="Amplitudes", annotations=[{"kind": "text", "x": 0.5, "y": -18.0, "text": "note"}],
                ref_lines=[{"axis": "y", "at": -20.0, "label": "rest"}])
    spec = Spec([r, compare(plot="box", legend="none")], columns=2, size="double")
    fig, drawn = render.build(spec)
    kinds = {t.kind for t, _a in render.things(fig)}
    assert kinds >= {"letter", "title", "xlabel", "ylabel", "annotation", "bracket", "series", "refline"}

    def middle(which):
        x0, y0, x1, y1 = render.extent(fig, which, 0.0)
        return (x0 + x1) / 2, (y0 + y1) / 2

    for kind, key in (("title", ""), ("ylabel", ""), ("xlabel", ""), ("letter", ""), ("annotation", "0")):
        assert render.hit(fig, drawn, *middle(elements.Tag(kind, 0, key))) == elements.Tag(kind, 0, key), kind
    text = next(a for t, a in render.things(fig) if (t.kind, t.part) == ("bracket", "text"))
    bb = text.get_window_extent(render._renderer(fig))
    assert render.hit(fig, drawn, (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2) == elements.Tag("bracket", 0, "control|drug")
    ax = drawn[0].axes[0]
    dot = next(a for t, a in render.things(fig) if t == elements.Tag("series", 0, "drug") and len(a.get_offsets()))
    px = dot.get_offset_transform().transform(dot.get_offsets())[0]
    assert render.hit(fig, drawn, px[0], px[1]) == elements.Tag("series", 0, "drug")
    bb = ax.get_window_extent(render._renderer(fig))
    assert render.hit(fig, drawn, bb.x0 + 3, bb.y1 - 3) == elements.Tag("panel", 0)             # the plot's own ground
    assert render.hit(fig, drawn, bb.x0 - 12, (bb.y0 + bb.y1) / 2) == elements.Tag("yaxis", 0)  # its tick labels
    assert render.hit(fig, drawn, (bb.x0 + bb.x1) / 2, bb.y0 - 6) == elements.Tag("xaxis", 0)
    other = drawn[1].axes[0].get_window_extent(render._renderer(fig))
    assert render.hit(fig, drawn, other.x0 + 3, other.y1 - 3) == elements.Tag("panel", 1)
    assert render.hit(fig, drawn, -50, -50) is None
    assert render.extent(fig, elements.Tag("legend", 0)) is None                     # none drawn: nothing to outline


def test_a_legend_a_bracket_and_a_letter_stay_where_they_were_put():
    time = Recipe(data=pack(cells()), question="time", plot="time_course", x="Time (min)", y="Amplitude (pA)",
                  group="Condition", cell="Cell", legend_at=[0.1, 0.5])
    both = compare(stats=True, brackets=[{"a": "control", "b": "drug", "y": 5.0, "text": "my own test, p = 0.01"}],
                   letter_offset=[5.0, 2.0])
    spec = Spec([time, both], columns=2, size="double")
    fig, drawn = render.build(spec)
    rr = render._renderer(fig)
    leg, ax = drawn[0].axes[0].get_legend(), drawn[0].axes[0]
    corner = ax.transAxes.inverted().transform((leg.get_window_extent(rr).x0, leg.get_window_extent(rr).y1))
    assert corner == pytest.approx([0.1, 0.5], abs=0.03)
    parts = {t.part: a for t, a in render.things(fig) if t.kind == "bracket"}
    assert parts["text"].get_text() == "my own test, p = 0.01" and parts["line"].get_ydata()[1] == 5.0
    mark = next(a for t, a in render.things(fig) if t == elements.Tag("letter", 1))
    x0, _y0, _x1, y1 = render.panel_outline(fig, drawn[1])
    bb = mark.get_window_extent(rr)
    assert bb.x0 - x0 == pytest.approx((render.LETTER_PAD_MM + 5.0) / 174.0 * fig.bbox.width, abs=1.0)
    both.brackets = [{"a": "control", "b": "drug", "hidden": True}]
    fig, _ = render.build(spec)
    assert not any(t.kind == "bracket" for t, _a in render.things(fig))


# ---------------------------------------------------------------- the statistics a person chooses

def test_the_test_is_a_choice_and_every_pair_can_be_compared():
    f = cells()
    third = f[f.Condition == "drug"].assign(Condition="wash")
    third["Amplitude (pA)"] += 1.0
    third["Cell"] = third["Cell"].str.replace("drug", "wash")
    three = pd.concat([f, third], ignore_index=True)
    from clampwise.figure import prepare

    pts = prepare.prepare(three, Recipe(question="compare", y="Amplitude (pA)", group="Condition", cell="Cell",
                                        animal="Animal")).points
    first = stats.compare(pts)
    assert [(c.reference, c.other) for c in first] == [("control", "drug"), ("control", "wash")]
    assert first[0].test == "Welch's t test" and first[0].other_test == "Mann-Whitney U"
    every = stats.compare(pts, pairs="all")
    assert [(c.reference, c.other) for c in every] == [("control", "drug"), ("control", "wash"), ("drug", "wash")]
    assert all(c.p >= c.p_raw for c in every)                                        # Holm over the three
    rank = stats.compare(pts, test="mann_whitney")
    assert rank[0].test == "Mann-Whitney U" and rank[0].p_raw == pytest.approx(first[0].other_p)
    assert rank[0].other_p == pytest.approx(first[0].p_raw)
    student = stats.compare(pts, test="student")[0]
    from scipy import stats as sps

    a, b = pts[pts.group == "control"].y, pts[pts.group == "drug"].y
    assert student.p_raw == pytest.approx(sps.ttest_ind(b, a, equal_var=True).pvalue)
    assert "Mann-Whitney U" in stats.methods(rank) and set(stats.TESTS) >= {"", "welch", "student", "mann_whitney",
                                                                           "paired_t", "wilcoxon"}
    # the bracket carries the chosen test's p, as an exact value, stars or both
    r = compare(stats=True, test="mann_whitney", stats_text="both")
    fig, drawn = render.build(Spec([r]))
    text = next(a for t, a in render.things(fig) if (t.kind, t.part) == ("bracket", "text")).get_text()
    assert text.startswith("*") and "p " in text and drawn[0].comparisons[0].test == "Mann-Whitney U"
    assert elements.bracket_text(0.0312) == "p = 0.031" and elements.bracket_text(0.0002, "stars") == "***"
    assert elements.bracket_text(0.2, "both") == "n.s. (p = 0.20)"


def test_rows_left_out_by_hand_leave_the_figure_and_are_said():
    r = compare()
    fig, drawn = render.build(Spec([r]))
    n = drawn[0].prepared.counts["control"]
    r.excluded = list(range(0, 6))                                                   # one cell's six rows
    fig, drawn = render.build(Spec([r]))
    after = drawn[0].prepared.counts["control"]
    assert after[0] == n[0] - 1 and after[2] == n[2] - 6
    assert "6 rows of the table left out by hand" in drawn[0].prepared.caption


def test_a_condition_keeps_one_colour_in_every_panel():
    spec = Spec([compare(), Recipe(data=pack(cells()[cells().Condition == "drug"].assign(Condition="drug").pipe(
        lambda d: pd.concat([d, d.assign(Condition="wash", Cell=d.Cell + "w")]))), question="compare", plot="points",
        y="Amplitude (pA)", group="Condition", cell="Cell")], columns=2)
    a = render.looks(spec.panels[0], spec.style, ["control", "drug"])
    b = render.looks(spec.panels[1], spec.style, ["drug", "wash"])
    assert a["drug"].colour != b["drug"].colour                                      # each panel on its own
    spec.style.group_colours = {"drug": a["drug"].colour}
    b = render.looks(spec.panels[1], spec.style, ["drug", "wash"])
    assert b["drug"].colour == a["drug"].colour and b["wash"].colour not in (a["drug"].colour,)
    assert render.looks(spec.panels[0], spec.style, ["control", "drug"])["control"].colour != a["drug"].colour
    spec.panels[1].series = {"drug": {"colour": "#123456"}}                          # a panel's own choice still wins
    assert render.looks(spec.panels[1], spec.style, ["drug", "wash"])["drug"].colour == "#123456"


# ---------------------------------------------------------------- export

def test_a_pdf_is_the_exact_size_with_its_text_kept_as_text(tmp_path):
    trace, _rec = light_panel()
    spec = Spec([trace, compare(stats=True)], columns=2, journal="jneurosci", size="double")
    spec.style.apply_journal("jneurosci")
    w, h = spec.size_mm()
    out = render.export(spec, tmp_path / "fig.pdf")
    raw = out.read_bytes()
    box = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
    assert box is not None
    assert float(box.group(3)) / 72 * 25.4 == pytest.approx(w, abs=0.05)             # 176 mm, to a twentieth
    assert float(box.group(4)) / 72 * 25.4 == pytest.approx(h, abs=0.05)
    assert b"/Subtype /TrueType" in raw or b"/FontFile2" in raw                      # the font itself, not outlines
    assert b"/Type3" not in raw and render.read_spec(out).to_json() == spec.to_json()
    svg = render.export(spec, tmp_path / "fig.svg").read_text(encoding="utf-8")
    assert "<text" in svg and "50 ms" in svg                                         # words stay words
    assert float(re.search(r'<svg[^>]*\bwidth="([\d.]+)pt"', svg).group(1)) / 72 * 25.4 == pytest.approx(w, abs=0.05)
    shapes = render.export(spec, tmp_path / "shapes.svg", outline=True).read_text(encoding="utf-8")
    assert "<text" not in shapes                                                     # asked for: letters as shapes
    assert b"/Type3" in render.export(spec, tmp_path / "shapes.pdf", outline=True).read_bytes()
    assert render.export_dpi(spec) == 1200 and render.export_dpi(Spec([compare()])) == 600
    from PIL import Image

    png = render.export(spec, tmp_path / "fig.png", dpi=300)
    with Image.open(png) as im:
        assert im.size[0] == pytest.approx(w / 25.4 * 300, abs=2) and im.size[1] == pytest.approx(h / 25.4 * 300, abs=2)


def test_a_figure_of_the_old_format_reopens_as_it_looked():
    import json

    old = json.loads(Spec([compare(), compare()], columns=2).to_json())
    old["version"] = 1
    for key in ("journal", "height_mm", "col_ratios", "row_ratios", "texts", "snap_mm"):
        old.pop(key)
    for p in old["panels"]:
        for key in ("span", "rect", "inset_of", "stim_bars", "scale_bar", "test", "excluded"):
            p.pop(key)
    back = Spec.from_json(json.dumps(old))
    assert len(back.panels) == 2 and back.journal == "" and back.style.letter_size == back.style.font_size + 3
    fig, drawn = render.build(back)
    assert len(drawn) == 2 and back.size_mm()[0] == 85.0


def test_a_proportion_with_its_interval_and_fishers_exact_test():
    """The share of each group in one category: Wilson's interval (checked against published values), Newcombe's
    interval of a difference (his own example), Fisher's exact test, Holm over the pairs; drawn as bars with k of
    n over each, a bracket per pair, and said in the caption and the Methods text."""
    from scipy import stats as sps

    from clampwise.analysis.tools import fits
    from clampwise.figure import methods, stats

    # Wilson: 0 of 10 gives 0 to 27.8% (Brown, Cai & DasGupta 2001 discuss this case); the analyses' own agree
    lo, hi = stats.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=5e-4)
    assert stats.wilson(4, 22) == pytest.approx(fits.wilson(4, 22), abs=1e-9)
    assert stats.wilson(81, 263) == pytest.approx((0.2553, 0.3662), abs=5e-4)      # Newcombe 1998 (Stat Med 17:857)
    assert all(np.isnan(v) for v in stats.wilson(0, 0))
    # Newcombe's hybrid score interval, his worked example: 56/70 - 48/80 = 0.20, 0.0524 to 0.3339
    assert stats.newcombe(48, 80, 56, 70) == pytest.approx((0.0524, 0.3339), abs=5e-4)
    rows = [{"Pattern": pat, "Outcome": "success" if i < k else "failure"}
            for pat, k, n in (("Full field", 12, 22), ("ROI1", 1, 20), ("ROI2", 3, 22)) for i in range(n)]
    r = Recipe(data=pack(pd.DataFrame(rows)), question="counts", plot="proportion", x="Outcome", group="Pattern",
               level="row", event="success", stats=True, pairs="first")
    fig, drawn = render.build(Spec([r]))
    d = drawn[0]
    ax = fig.axes[0]
    bars = [b for b in ax.patches if b.get_width() == pytest.approx(0.6)]
    assert [round(b.get_height(), 2) for b in bars] == [round(100 * 12 / 22, 2), 5.0, round(100 * 3 / 22, 2)]
    assert {t.get_text() for t in ax.texts} >= {"12/22", "1/20", "3/22"}
    assert ax.get_ylabel() == "success (% of rows)" and ax.get_legend() is None
    assert [(c.reference, c.other, c.counts) for c in d.comparisons] == [
        ("Full field", "ROI1", (12, 22, 1, 20)), ("Full field", "ROI2", (12, 22, 3, 22))]
    raw = [float(sps.fisher_exact([[12, 10], [1, 19]])[1]), float(sps.fisher_exact([[12, 10], [3, 19]])[1])]
    assert [c.p_raw for c in d.comparisons] == pytest.approx(raw)
    assert [c.p for c in d.comparisons] == pytest.approx(stats.holm(raw)) and d.comparisons[0].p < 0.01
    said = d.comparisons[0].sentence()
    assert "1 of 20 against 12 of 22" in said and "Fisher's exact test p" in said and "Newcombe" in said
    brackets = [t for t, _a in render.things(fig) if t.kind == "bracket" and t.part == "line"]
    assert [b.key for b in brackets] == ["Full field|ROI1", "Full field|ROI2"]
    assert "Wilson's 95% score interval" in d.prepared.caption and "Full field: 12 of 22" in d.prepared.caption
    assert any("trials of that recording" in w for w in d.prepared.warnings)       # windows are not cells
    text = methods.paragraph(drawn, None)
    assert "Fisher's exact test, Holm-corrected" in text and "Newcombe 1998" in text
    # the category counted is a choice (the first one when none is named); cells counted once each when named
    r.event = "failure"
    _fig, drawn = render.build(Spec([r]))
    assert drawn[0].comparisons[0].counts == (10, 22, 19, 20)
    cells = pd.DataFrame([{"Group": g, "Cell": f"{g}{c}", "Connected": "yes" if c < k else "no", "Sweep": s}
                          for g, k, n in (("near", 6, 8), ("far", 1, 9)) for c in range(n) for s in range(3)])
    r2 = Recipe(data=pack(cells), question="counts", plot="proportion", x="Connected", group="Group", cell="Cell",
                level="cell", stats=True)
    _fig, drawn = render.build(Spec([r2]))
    assert drawn[0].comparisons[0].counts == (6, 8, 1, 9) and not drawn[0].prepared.warnings
    assert "share of cells" in drawn[0].prepared.caption
