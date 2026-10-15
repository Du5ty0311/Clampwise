"""The figure builder (B12) without its window: a figure is a ``Spec`` (its
panels' recipes, layout and size, with the data each panel was made from),
so it can be saved, reopened editable, embedded in its own exports, and
tested without Qt.

- ``recipe``: the Spec and each panel's Recipe; the questions a figure can
  answer; every control's caption, line and help (the window is built from it).
- ``data``: where a table comes from (the project, a CSV or Excel file, a
  digitized plot) and reshaping.
- ``prepare``: filters, time window, bins, baseline, the level each point is
  (rows rolled up to cells), the caption ("What this figure shows") and the
  warnings when a figure would mislead.
- ``stats``: comparisons at the right level (B23).
- ``render``: matplotlib drawing and export (PDF, SVG, EPS, PNG, TIFF).
- ``digitize``: a table from a picture of a plot, said to be an estimate.
- ``methods``: the Methods paragraph for what a figure shows.
- ``layouts``: paper-figure layouts and the example-trace picker.
"""
