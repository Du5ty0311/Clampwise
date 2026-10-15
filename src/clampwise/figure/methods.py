"""The Methods paragraph for a figure: the analyses its numbers come from (what
each does and its source, the settings changed from the defaults, and what the
methods ledger recorded as applied: filters, fits, windows), then what the
figure shows and the statistics. Written from the registry, the stored
results and the figure itself, never from memory.
"""

from __future__ import annotations

from .. import __version__
from . import stats as fstats


def _analyses_used(project, names: set[str]) -> list[str]:
    """The analyses behind the project tables a figure was drawn from."""
    from .. import export
    from ..analysis import registry

    if project is None or not names:
        return []
    reg = registry.load()
    wanted = [k for k, a in reg.items() if any(n == a.label or n.startswith(f"{a.label}:") for n in names)]
    if any(n.startswith("Recordings:") for n in names):
        types = {n.split(": ", 1)[1] for n in names if n.startswith("Recordings:")}
        wanted += [k for k, a in reg.items() if any(registry.TYPE_WORDS.get(t, t) in types for t in a.types)]
    rows = export.collect(project)
    sources = list(rows) + list(export.cell_results(project, rows))     # a cell's results (DSI over its files)
    out = []
    for key in dict.fromkeys(wanted):
        a = reg[key]
        results = [r.results[key] for r in sources if key in r.results and not r.results[key].error]
        if not results:
            continue
        ledger = list(dict.fromkeys(m for res in results for m in res.methods))
        changed = sorted({f"{s.label.lower()} {res.settings.get(s.key)}{(' ' + s.units) if s.units else ''}"
                          for res in results for s in a.settings if s.key in res.changed_settings})
        text = f"{a.label}: {a.description.rstrip('.')}" + (f" ({a.source})" if a.source else "") + "."
        if changed:
            text += " Settings changed from the defaults: " + "; ".join(changed) + "."
        if ledger:
            text += " Applied: " + "; ".join(ledger[:8]) + ("; and others" if len(ledger) > 8 else "") + "."
        out.append(text)
    return out


def paragraph(drawn: list, project=None) -> str:
    """Methods for the figure ``drawn`` (render.build's output)."""
    names = {d.recipe.source.get("table", "") for d in drawn if d.recipe.source.get("kind") == "project"}
    parts = []
    used = _analyses_used(project, names)
    if used:
        parts.append(f"Recordings were analyzed with Clampwise {__version__}. " + " ".join(used))
    files = sorted({d.recipe.source.get("path", "") for d in drawn if d.recipe.source.get("kind") == "file"})
    if files:
        parts.append("Data were read from " + ", ".join(p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in files) + ".")
    if any(d.recipe.source.get("kind") == "digitized" for d in drawn):
        parts.append("Some values were digitized from a published figure; they are estimates limited by the "
                     "picture's resolution.")
    for k, d in enumerate(drawn):
        if d.prepared is None:
            continue
        label = f"({chr(ord('A') + k)}) " if len(drawn) > 1 else ""
        parts.append(label + d.prepared.caption)
    comps = [c for d in drawn for c in d.comparisons]
    if comps:
        parts.append(fstats.methods(comps))
    return " ".join(parts)
