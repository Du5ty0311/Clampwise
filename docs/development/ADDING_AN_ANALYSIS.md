# Adding an analysis

An analysis is one Python module in `src/clampwise/analysis/` that declares itself to the registry. From the
declaration the window builds its settings panel, Results shows its definitions, the exports define every column,
the project cache keys its results and Help lists it. The analysis writes no interface code and imports no Qt.

## 1. Write the module

```python
"""One paragraph on the method and where each choice comes from (the paper, the lab's decision and its date)."""

from __future__ import annotations

import numpy as np

from ..core.recording import Recording
from .registry import Output, Setting, register
from .result import Flag, Result
from .tools import ledger

SETTINGS = (
    Setting("window_ms", 50.0, "Response window (ms)", "One line of help, in the lab's words.", "ms",
            "the source of the default", minimum=1.0),
    Setting("alpha", 0.05, "False-positive rate", "", "", "Clampwise", advanced=True),
)

OUTPUTS = (
    Output("connected", "What the number means, as the lab would say it.", "", "the published method",
           label="Connected", headline=True),
    Output("z", "A column of a table below the recording level.", "ratio", level="sweep"),
)


@register("connectivity", "Connected or not", version=1, types={"cc_evoked", "vc_evoked"}, settings=SETTINGS,
          outputs=OUTPUTS, requires=("evoked",), description="One sentence for Help.", source="the papers")
def run(recording: Recording, context, s) -> Result | None:
    out = Result("connectivity", 1)
    ...
    out.values["connected"] = "yes"                       # every recording-level output, NaN when not measured
    out.tables["windows"] = [{"sweep": 0, "z": 4.2}]       # rows of a lower level
    out.flags.append(Flag("unhealthy", "what is wrong, in full", "the limit", "the source"))
    out.choose("verdict", "yes", "why this stands for the recording")   # nothing is chosen silently
    ledger.note("what was done, in one line", "connected or not")       # goes into the Methods text
    return out
```

- `types` are the recording types in `registry.TYPES`; `routing.py` decides which one a recording is.
- `requires` names the analyses whose results this one reads through `context.result("evoked")`; they run first.
- `context.stimulus` is where the stimulus is in every sweep; `context.artifacts` what the signal shows besides
  the cell (test pulses, command changes); `context.shared` the shared settings (a sweep selection, the rig's
  light inputs).
- Return `None` when the analysis does not apply (a current-clamp file for a voltage-clamp measure): nothing is
  shown. Return a `Result` with a note when it applies but nothing could be measured.
- A `Flag` is evidence, never an edit: the value stays, the flag says what is wrong, its limit and its source.
  `severity="note"` describes the data; `"flag"` (the default) is a quality problem.
- Every value and every table column needs a definition: recording-level outputs in `OUTPUTS`; columns of the
  tables in `columns.TABLES[<analysis key>]` (or an `Output` with `level="sweep"` and so on).
- **Bump `version`** whenever the method changes: it invalidates the project caches, so old numbers are never
  mixed with new ones.

## 2. List the module

Add it to `registry.MODULES`, in the order it should appear in Results. Modules are listed, not discovered, so
the frozen build includes them.

## 3. Known answers

Every measure gets a simulated recording whose true value is exact (`clampwise.simulate`), a section in
`tools/audit/known_answers.py` returning `Row(section, case, measure, truth, measured, tolerance)`, listed in
`SECTIONS`, and a test that runs it:

```python
def test_the_known_answers_pass():
    rows = known_answers.run(["connectivity"])
    bad = [r for r in rows if not r.ok]
    assert not bad, "\n".join(f"{r.case} {r.measure}: true {r.truth} measured {r.measured}" for r in bad)
```

`python tools/audit/known_answers.py connectivity` prints the same rows. Where Clampfit can make the same measure
on a real file, compare and write the agreement into `docs/VALIDATION.md`.

## 4. Cell-level analyses

An analysis over a cell's files (several recordings of one cell, on one clock) registers with `level="cell"` and a
`run(cell, s)` that reads `cell.members` (each with `.recording`, `.results` and `.result(key)`); it never
re-analyzes a recording. See `connectivity_cell` in `connectivity.py` and `dsi_cell` in `dsi.py`.

## 5. Before committing

- `tests/`: the full suite takes about 25 minutes (`python -m pytest -q tests`); one membrane-test test is known
  to be flaky.
- A CHANGELOG entry under "not released yet", in the lab's words, with numbers from real files where there are any.
- Keep each file's own line endings (`git ls-files --eol`): the tree is mixed.
