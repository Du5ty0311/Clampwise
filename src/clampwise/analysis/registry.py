"""The registry: adding an analysis is one file (B18, item 9).

An analysis declares itself here with ``@register``: the recording types it
applies to, its settings (default, units, one-line help, source, whether it
is an "Advanced" setting), its outputs with definitions, its version (bumped
by hand whenever the method changes, which invalidates cached results), its
source, and the function that runs it. From that declaration the window
builds the settings panel, Results shows the definitions, the exports define
every column, the cache keys its results and the manual lists it. The
analysis itself writes no interface code.

``docs/development/ADDING_AN_ANALYSIS.md`` walks through one.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable

log = logging.getLogger("clampwise")

# Recording types (routing.py decides which one a recording is).
TYPES = ("membrane_test", "cc_steps", "vc_steps", "cc_short_square", "cc_ramp", "cc_evoked", "vc_evoked", "dsi",
         "cc_spontaneous", "vc_spontaneous", "field_evoked", "field_spontaneous", "spike_times", "unknown")
WHOLE_CELL = frozenset(t for t in TYPES if not t.startswith("field") and t not in ("unknown", "spike_times"))
FIELD = frozenset(t for t in TYPES if t.startswith("field"))
TYPE_WORDS = {
    "membrane_test": "Membrane test", "cc_steps": "Current steps", "vc_steps": "Voltage steps",
    "cc_short_square": "Short squares", "cc_ramp": "Current ramp",
    "cc_evoked": "Evoked, current clamp", "vc_evoked": "Evoked, voltage clamp", "dsi": "DSI",
    "cc_spontaneous": "Spontaneous, current clamp", "vc_spontaneous": "Spontaneous, voltage clamp",
    "field_evoked": "Field, evoked", "field_spontaneous": "Field, spontaneous", "spike_times": "Spike times",
    "unknown": "Unknown", "": "",
}
EVOKED = frozenset({"cc_evoked", "vc_evoked", "dsi"})
LEVELS = ("event", "pulse", "sweep", "recording", "cell", "animal")


@dataclass(frozen=True)
class Setting:
    """One setting of an analysis, as the window and the manual show it."""

    key: str
    default: Any
    label: str
    help: str = ""                      # one line, the lab's words
    units: str = ""
    source: str = ""                    # where the default comes from
    advanced: bool = False              # under "Advanced" unless someone changes it
    choices: tuple = ()                 # for a pick list
    minimum: float | None = None
    maximum: float | None = None
    hidden: bool = False                # kept by the app (e.g. the event review), never a control

    def validate(self, value: Any) -> Any:
        if self.choices and value not in self.choices:
            raise ValueError(f"{self.label}: {value!r} is not one of {', '.join(map(str, self.choices))}")
        if isinstance(self.default, bool):
            return bool(value)
        if isinstance(self.default, (int, float)) and not isinstance(value, bool):
            v = float(value)
            if self.minimum is not None and v < self.minimum:
                raise ValueError(f"{self.label}: {v:g} is below the minimum {self.minimum:g}")
            if self.maximum is not None and v > self.maximum:
                raise ValueError(f"{self.label}: {v:g} is above the maximum {self.maximum:g}")
            return int(v) if isinstance(self.default, int) else v
        return value


@dataclass(frozen=True)
class Output:
    """One number or table column an analysis produces, with its definition."""

    key: str
    definition: str
    units: str = ""
    source: str = ""                    # the published method, or "Clampwise" with the reason in the definition
    level: str = "recording"            # event, pulse, sweep, recording, cell, animal
    verified_against: str = "simulation"
    label: str = ""                     # what the screen calls it, in the lab's words
    headline: bool = False              # shown in the overview before anything is opened

    @property
    def title(self) -> str:
        """The label, or the key made readable when none was given."""
        if self.label:
            return self.label
        words = self.key.replace("_", " ")
        return words[:1].upper() + words[1:]


@dataclass(frozen=True)
class Analysis:
    key: str
    label: str
    version: int
    types: frozenset[str]
    run: Callable
    settings: tuple[Setting, ...] = ()
    outputs: tuple[Output, ...] = ()
    description: str = ""
    source: str = ""
    modality: str = "whole_cell"        # "whole_cell" or "field": which workbook
    requires: tuple[str, ...] = ()      # other analyses whose results this one reads
    visible_settings: int = 7           # how many settings show before "Advanced"
    # "recording": run on one recording by the pipeline; "cell": run on a cell's
    # recordings together (their per-recording results on one clock), after them
    level: str = "recording"

    def defaults(self) -> dict[str, Any]:
        return {s.key: s.default for s in self.settings}

    def resolve(self, given: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
        """Settings with defaults filled in and values checked; the keys that
        differ from their default (what "Advanced (2 changed)" counts)."""
        given = dict(given or {})
        unknown = sorted(set(given) - {s.key for s in self.settings})
        if unknown:
            raise ValueError(f"{self.label}: unknown setting(s) {', '.join(unknown)}")
        out, changed = {}, []
        for s in self.settings:
            value = s.validate(given[s.key]) if s.key in given else s.default
            out[s.key] = value
            if s.key in given and value != s.default:
                changed.append(s.key)
        return out, changed

    def output(self, key: str) -> Output | None:
        for o in self.outputs:
            if o.key == key:
                return o
        return None


REGISTRY: dict[str, Analysis] = {}
LOAD_ERRORS: list[str] = []

# Modules that register analyses when imported: listed, not discovered, so a
# frozen build includes them and results come out in a fixed order.
MODULES: tuple[str, ...] = (
    "clampwise.analysis.baseline",
    "clampwise.analysis.artifacts",
    "clampwise.analysis.evoked",
    "clampwise.analysis.passive",
    "clampwise.analysis.quality",
    "clampwise.analysis.events",
    "clampwise.analysis.dsi",
    "clampwise.analysis.dsi_events",
    "clampwise.analysis.intrinsic",
    "clampwise.analysis.thresholds",
    "clampwise.analysis.vclamp",
    "clampwise.analysis.field",
    "clampwise.analysis.spike_train",
)
_loaded = False


def register(key: str, label: str, *, version: int, types: Iterable[str], settings: Iterable[Setting] = (),
             outputs: Iterable[Output] = (), description: str = "", source: str = "",
             modality: str = "whole_cell", requires: Iterable[str] = (), visible_settings: int = 7,
             level: str = "recording"):
    """Decorator: register ``fn(recording, context, settings) -> Result``."""
    types = frozenset(types)
    bad = types - set(TYPES)
    if bad:
        raise ValueError(f"{key}: unknown recording type(s) {sorted(bad)}")

    def wrap(fn):
        REGISTRY[key] = Analysis(key, label, version, types, fn, tuple(settings), tuple(outputs), description,
                                 source, modality, tuple(requires), visible_settings, level)
        return fn

    return wrap


def load() -> dict[str, Analysis]:
    global _loaded
    if not _loaded:
        for name in MODULES:
            try:
                importlib.import_module(name)
            except ImportError as exc:        # a missing module costs its analyses, not the app
                LOAD_ERRORS.append(f"{name}: {exc}")
        _loaded = True
    return REGISTRY


def for_type(recording_type: str, *, disabled: Iterable[str] = (), only: Iterable[str] | None = None) -> list[Analysis]:
    """The analyses that apply to a recording type, in registration order,
    with their requirements first."""
    disabled = set(disabled)
    wanted = set(only) if only is not None else None
    chosen = [a for a in load().values() if a.key not in disabled and recording_type in a.types
              and a.level == "recording" and (wanted is None or a.key in wanted)]
    ordered: list[Analysis] = []

    def add(a: Analysis) -> None:
        if a in ordered:
            return
        for req in a.requires:
            dep = REGISTRY.get(req)
            if dep is not None and dep not in ordered:
                add(dep)
        ordered.append(a)

    for a in chosen:
        add(a)
    return ordered


def cell_analyses(*, disabled: Iterable[str] = ()) -> list[Analysis]:
    """The analyses run on a cell's recordings together, in registration order."""
    disabled = set(disabled)
    return [a for a in load().values() if a.level == "cell" and a.key not in disabled]


def column(analysis: str, key: str) -> Output | None:
    """A result table's column: the analysis's own output below the recording
    level, else columns.py's (its own tables', then the shared ones)."""
    from . import columns

    a = load().get(analysis)
    own = next((o for o in (a.outputs if a else ()) if o.key == key and o.level != "recording"), None)
    if own is not None:
        return own
    mine = {o.key: o for o in columns.TABLES.get(analysis, ())}
    return mine.get(key) or columns.COMMON.get(key)


def definitions_by_analysis() -> dict[str, dict[str, Output]]:
    """Each analysis's outputs, by key (two analyses may use the same key)."""
    return {k: {o.key: o for o in a.outputs} for k, a in load().items()}


def definitions() -> dict[str, Output]:
    """Every output of every analysis, by key (the export definitions)."""
    out: dict[str, Output] = {}
    for a in load().values():
        for o in a.outputs:
            out[o.key] = o
    return out
