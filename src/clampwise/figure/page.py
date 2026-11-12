"""Where each panel sits on the page, in millimetres from the page's top left.

A figure's panels are laid out one of two ways, panel by panel:

- **in the grid**: the page is cut into columns and rows (each with its own share of the width and height), and
  a panel takes the next free cell, or several (it can span columns and rows);
- **placed freely**: the panel has its own rectangle on the page (``Recipe.rect``), put there by a template, by
  dragging, or by typing its millimetres.

An inset has no place of its own on the page: it sits inside another panel (``Recipe.inset_of``), at a rectangle
given as fractions of that panel's plot area.

Everything here is arithmetic on rectangles ``[x, y, width, height]`` (mm): the cells, the templates of common
figures, aligning and distributing, snapping to a grid, keeping a panel on the page. No drawing, no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass

GUTTER_MM = 3.0          # between panels in the grid
MARGIN_MM = 0.5          # round the page
MIN_PANEL_MM = 12.0      # a panel no smaller than this, either way
ROW_ASPECT = 0.72        # a grid row's height over a column's width, when the height is left to Clampwise
ROW_MOST_MM = 58.0       # and never higher than this: a row of wide panels is not a tall one


def spans(panels) -> list[tuple[int, int]]:
    """(columns, rows) each panel spans in the grid (1, 1 unless set)."""
    out = []
    for r in panels:
        s = list(getattr(r, "span", []) or [])
        out.append((max(1, int(s[0])) if len(s) > 0 else 1, max(1, int(s[1])) if len(s) > 1 else 1))
    return out


def cells(spans_: list[tuple[int, int]], columns: int, skip: set[int] = frozenset()) -> tuple[list, int]:
    """Each panel's cell (column, row, columns spanned, rows spanned), panels flowing left to right, top to
    bottom into the first free cells that hold them; and the number of rows used. Panels in ``skip`` (insets,
    freely placed ones) take no cell (None)."""
    columns = max(1, int(columns))
    taken: set[tuple[int, int]] = set()
    out, rows = [], 0
    for k, (cs, rs) in enumerate(spans_):
        if k in skip:
            out.append(None)
            continue
        cs = min(cs, columns)
        row = 0
        while True:
            placed = False
            for col in range(columns - cs + 1):
                want = {(col + i, row + j) for i in range(cs) for j in range(rs)}
                if not (want & taken):
                    taken |= want
                    out.append((col, row, cs, rs))
                    rows = max(rows, row + rs)
                    placed = True
                    break
            if placed:
                break
            row += 1
    return out, max(rows, 1)


def _shares(ratios, n: int) -> list[float]:
    r = [float(v) for v in list(ratios or [])[:n] if v and float(v) > 0]
    r += [1.0] * (n - len(r))
    total = sum(r)
    return [v / total for v in r]


def auto_height(width_mm: float, columns: int, rows: int, row_ratios=()) -> float:
    """The page's height when it is left to Clampwise: rows as high as ``ROW_ASPECT`` of a column's width."""
    columns, rows = max(1, columns), max(1, rows)
    col_w = (width_mm - 2 * MARGIN_MM - GUTTER_MM * (columns - 1)) / columns
    return 2 * MARGIN_MM + rows * min(col_w * ROW_ASPECT, ROW_MOST_MM) + GUTTER_MM * (rows - 1)


def grid_rects(cells_: list, columns: int, rows: int, width_mm: float, height_mm: float, col_ratios=(),
               row_ratios=(), gutter: float = GUTTER_MM, margin: float = MARGIN_MM) -> list:
    """The rectangle (mm) of each cell of ``cells``; None stays None."""
    cw = _shares(col_ratios, columns)
    rh = _shares(row_ratios, rows)
    inner_w = width_mm - 2 * margin - gutter * (columns - 1)
    inner_h = height_mm - 2 * margin - gutter * (rows - 1)
    xs = [margin + sum(cw[:i]) * inner_w + gutter * i for i in range(columns)]
    ys = [margin + sum(rh[:j]) * inner_h + gutter * j for j in range(rows)]
    out = []
    for c in cells_:
        if c is None:
            out.append(None)
            continue
        col, row, cs, rs = c
        w = sum(cw[col:col + cs]) * inner_w + gutter * (cs - 1)
        h = sum(rh[row:row + rs]) * inner_h + gutter * (rs - 1)
        out.append([xs[col], ys[row], w, h])
    return out


def layout(spec) -> tuple[list, float, float]:
    """(each panel's rectangle in mm, the page's width, its height). A panel placed freely has its own
    rectangle; an inset has None (it is drawn inside its parent); the others take the grid's cells."""
    panels = spec.panels
    free = {k for k, r in enumerate(panels) if len(getattr(r, "rect", []) or []) == 4}
    inset = {k for k, r in enumerate(panels) if is_inset(panels, k)}
    cell_list, rows = cells(spans(panels), spec.columns, skip=free | inset)
    width, height = spec.page_mm(rows)
    rects = grid_rects(cell_list, max(1, spec.columns), rows, width, height, spec.col_ratios, spec.row_ratios)
    for k in free - inset:
        rects[k] = [float(v) for v in panels[k].rect]
    return rects, width, height


def is_inset(panels, k: int) -> bool:
    """Whether panel ``k`` sits inside another panel that exists (and is not itself an inset)."""
    j = getattr(panels[k], "inset_of", -1)
    return isinstance(j, int) and 0 <= j < len(panels) and j != k and not (
        isinstance(getattr(panels[j], "inset_of", -1), int) and 0 <= getattr(panels[j], "inset_of", -1) < len(panels))


def rows_used(spec) -> int:
    panels = spec.panels
    skip = {k for k, r in enumerate(panels) if len(getattr(r, "rect", []) or []) == 4 or is_inset(panels, k)}
    return cells(spans(panels), spec.columns, skip=skip)[1]


# ---------------------------------------------------------------- templates of common figures

@dataclass(frozen=True)
class Template:
    key: str
    name: str
    line: str
    columns: int
    spans: tuple                  # (columns, rows) per panel, in order
    col_ratios: tuple = ()
    row_ratios: tuple = ()

    @property
    def n(self) -> int:
        return len(self.spans)


TEMPLATES = {t.key: t for t in (
    Template("one", "One panel", "A single panel filling the page.", 1, ((1, 1),)),
    Template("two_across", "Two side by side", "Two panels of equal width.", 2, ((1, 1), (1, 1))),
    Template("two_down", "Two stacked", "One panel above the other.", 1, ((1, 1), (1, 1))),
    Template("three_across", "Three in a row", "Three panels of equal width.", 3, ((1, 1),) * 3),
    Template("grid_2x2", "Two by two", "Four panels of equal size.", 2, ((1, 1),) * 4),
    Template("grid_3x2", "Three by two", "Six panels of equal size.", 3, ((1, 1),) * 6),
    Template("wide_top_2", "Wide on top, two below", "Example traces across the page, two summaries under them.", 2,
             ((2, 1), (1, 1), (1, 1))),
    Template("wide_top_3", "Wide on top, three below", "Example traces across the page, three summaries under "
             "them: the usual connectivity or evoked-response figure.", 3, ((3, 1), (1, 1), (1, 1), (1, 1)),
             row_ratios=(1.15, 1.0)),
    Template("tall_left_2", "Tall on the left, two on the right", "A step family or an image beside two graphs.", 2,
             ((1, 2), (1, 1), (1, 1)), col_ratios=(1.2, 1.0)),
    Template("traces_left_3", "Traces on the left, three graphs on the right", "A wide column of example traces "
             "and a narrow column of three small graphs.", 3, ((2, 3), (1, 1), (1, 1), (1, 1))),
    Template("wide_left_2x2", "One wide, then two by two", "A wide panel and four small ones beside it.", 4,
             ((2, 2), (1, 1), (1, 1), (1, 1), (1, 1))),
    Template("two_rows_3_2", "Three above, two below", "Three panels in the first row, two wider ones under "
             "them.", 6, ((2, 1), (2, 1), (2, 1), (3, 1), (3, 1))),
)}


def apply_template(spec, key: str) -> None:
    """The figure's panels in the template's arrangement: its columns, each panel's span, the rows' and columns'
    shares. Panels the template has no place for keep flowing after it, one cell each; a panel placed freely
    comes back into the grid; insets stay insets."""
    t = TEMPLATES[key]
    spec.columns = t.columns
    spec.col_ratios, spec.row_ratios = list(t.col_ratios), list(t.row_ratios)
    k = 0
    for i, r in enumerate(spec.panels):
        if is_inset(spec.panels, i):
            continue
        r.rect = []
        r.span = list(t.spans[k]) if k < t.n else [1, 1]
        k += 1


def to_free(spec) -> None:
    """Every panel given its own rectangle where the grid has it now: the start of placing them freely."""
    rects, _w, _h = layout(spec)
    for r, rect in zip(spec.panels, rects):
        if rect is not None:
            r.rect = [round(v, 2) for v in rect]


def to_grid(spec) -> None:
    """Every panel back into the grid's cells (its own rectangle forgotten)."""
    for r in spec.panels:
        r.rect = []


# ---------------------------------------------------------------- arithmetic on rectangles

def clamp(rect, width_mm: float, height_mm: float, least: float = MIN_PANEL_MM) -> list:
    """``rect`` kept on the page and no smaller than ``least`` either way."""
    x, y, w, h = (float(v) for v in rect)
    w, h = min(max(w, least), width_mm), min(max(h, least), height_mm)
    return [min(max(x, 0.0), width_mm - w), min(max(y, 0.0), height_mm - h), w, h]


def snap(rect, step_mm: float) -> list:
    """``rect`` with its edges on a grid of ``step_mm`` (0: as it is)."""
    if not step_mm or step_mm <= 0:
        return [float(v) for v in rect]
    x, y, w, h = (float(v) for v in rect)
    x0, y0 = round(x / step_mm) * step_mm, round(y / step_mm) * step_mm
    x1, y1 = round((x + w) / step_mm) * step_mm, round((y + h) / step_mm) * step_mm
    return [x0, y0, max(x1 - x0, step_mm), max(y1 - y0, step_mm)]


ALIGN = {"left": "Align left edges", "right": "Align right edges", "top": "Align top edges",
         "bottom": "Align bottom edges", "centre_x": "Centre on one vertical line",
         "centre_y": "Centre on one horizontal line", "same_width": "Make the same width",
         "same_height": "Make the same height", "spread_x": "Space evenly across", "spread_y": "Space evenly down"}


def arrange(rects: list, how: str) -> list:
    """Rectangles aligned, sized or spaced (``ALIGN``'s keys), the first one the reference for aligning and
    sizing; spacing keeps the outermost two where they are and makes the gaps between equal."""
    rs = [[float(v) for v in r] for r in rects]
    if len(rs) < 2:
        return rs
    ref = rs[0]
    if how == "left":
        for r in rs:
            r[0] = ref[0]
    elif how == "right":
        for r in rs:
            r[0] = ref[0] + ref[2] - r[2]
    elif how == "top":
        for r in rs:
            r[1] = ref[1]
    elif how == "bottom":
        for r in rs:
            r[1] = ref[1] + ref[3] - r[3]
    elif how == "centre_x":
        for r in rs:
            r[0] = ref[0] + ref[2] / 2 - r[2] / 2
    elif how == "centre_y":
        for r in rs:
            r[1] = ref[1] + ref[3] / 2 - r[3] / 2
    elif how == "same_width":
        for r in rs:
            r[2] = ref[2]
    elif how == "same_height":
        for r in rs:
            r[3] = ref[3]
    elif how in ("spread_x", "spread_y") and len(rs) > 2:
        i, size = (0, 2) if how == "spread_x" else (1, 3)
        order = sorted(range(len(rs)), key=lambda k: rs[k][i])
        first, last = rs[order[0]], rs[order[-1]]
        room = (last[i] + last[size]) - first[i] - sum(rs[k][size] for k in order)
        gap = room / (len(rs) - 1)
        at = first[i]
        for k in order:
            rs[k][i] = at
            at += rs[k][size] + gap
    return rs


def inset_default() -> list:
    """Where a new inset sits in its parent's plot area: the upper right quarter (fractions of the plot)."""
    return [0.56, 0.54, 0.4, 0.4]
