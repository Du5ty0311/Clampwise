"""What an analysis returns, and what the pipeline wraps it in.

A ``Result`` carries the recording-level values (keys match the analysis's
declared outputs), tables at the lower levels (pulses, sweeps, events), the
flags (evidence, never an edit: the value stays, the flag says what is wrong,
its limit and its source), notes, the methods the ledger recorded, the
settings used, and drawables (fits and windows) for the trace.

A ``Flag`` is the one badge style of the interface (B0, item 15): a short
word for the badge, the full reason, the limit and the source for the
tooltip and the details panel.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Flag:
    code: str                           # short: "Rs high", "no response", "fit at limit"
    text: str                           # what is wrong, in full
    limit: str = ""                     # the limit it crossed, with units
    source: str = ""                    # where the limit comes from
    level: str = "recording"            # recording, sweep, pulse, event
    where: str = ""                     # which sweep / pulse / event, when not the recording
    severity: str = "flag"              # "flag" (QC) or "note" (describes the data, not a problem)

    def describe(self) -> str:
        text = f"{self.code}: {self.text}"
        if self.limit:
            text += f" (limit {self.limit})"
        if self.source:
            text += f" [{self.source}]"
        return text


@dataclass
class Drawable:
    """Something to draw over the trace: a fitted curve, a window, a marker."""

    kind: str                           # "curve", "window", "marker", "level"
    label: str
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    start_s: float = float("nan")
    end_s: float = float("nan")
    level: float = float("nan")
    sweep: int | None = None            # None: on the average
    # what x is: "time" (s in the sweep, drawn on the trace), "current" (pA, beside
    # the per-step curves), "event" (ms around an event); "" for results kept
    # before it was said, which are not drawn
    domain: str = ""


@dataclass
class Result:
    key: str
    version: int
    values: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, list[dict]] = field(default_factory=dict)     # level name -> rows
    flags: list[Flag] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    changed_settings: list[str] = field(default_factory=list)
    drawables: list[Drawable] = field(default_factory=list)
    # Which pulse, sweep, window or baseline stands for the recording, and why
    # (Part A: nothing is chosen silently).
    choices: dict[str, str] = field(default_factory=dict)
    error: str = ""                     # set when the analysis raised: a software error, not the recording's
    # compact numeric series another level reads (e.g. charge in 10 ms blocks for
    # a cell's DSI); not shown as a table
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    from_cache: bool = False            # taken from the project's cache rather than computed now

    @property
    def qc_flags(self) -> list[Flag]:
        return [f for f in self.flags if f.severity == "flag"]

    def choose(self, what: str, which: str, why: str) -> None:
        self.choices[what] = f"{which}: {why}"


@dataclass
class Provenance:
    """Where a number came from (B18, item 6): enough to make it again."""

    file: str
    content_hash: str
    app_version: str
    analyzed_at: str
    analyses: dict[str, int]            # key -> version
    settings: dict[str, dict]           # key -> settings used
    preset: str = ""
    python: str = ""
    libraries: dict[str, str] = field(default_factory=dict)

    @classmethod
    def now(cls, file: str, content_hash: str, app_version: str, analyses: dict[str, int],
            settings: dict[str, dict], preset: str = "") -> "Provenance":
        import platform

        libs = {}
        for name in ("numpy", "scipy", "pandas", "pyabf"):
            try:
                libs[name] = getattr(__import__(name), "__version__", "?")
            except ImportError:
                libs[name] = "not installed"
        return cls(file, content_hash, app_version, dt.datetime.now().isoformat(timespec="seconds"),
                   dict(analyses), settings, preset, platform.python_version(), libs)

    def to_dict(self) -> dict:
        return {"file": self.file, "sha256": self.content_hash, "app_version": self.app_version,
                "analyzed_at": self.analyzed_at, "analyses": self.analyses, "settings": self.settings,
                "preset": self.preset, "python": self.python, "libraries": self.libraries}


@dataclass
class RecordingResult:
    """Everything the pipeline produced for one recording."""

    recording: Any                      # the Recording
    recording_type: str
    type_reason: str
    stimulus_summary: str
    results: dict[str, Result] = field(default_factory=dict)
    provenance: Provenance | None = None
    notes: list[str] = field(default_factory=list)
    cache_shared: dict[str, dict] = field(default_factory=dict)    # key -> what its cache entry is keyed on

    @property
    def flags(self) -> list[Flag]:
        return [f for r in self.results.values() for f in r.qc_flags]

    @property
    def methods(self) -> list[str]:
        seen: list[str] = []
        for r in self.results.values():
            for m in r.methods:
                if m not in seen:
                    seen.append(m)
        return seen

    def values(self) -> dict[str, Any]:
        """Every recording-level value, keyed ``<analysis>.<output>``."""
        out = {}
        for key, r in self.results.items():
            for k, v in r.values.items():
                out[f"{key}.{k}"] = v
        return out

    def quality(self) -> str:
        """passed / flagged / not checked: the one badge per recording (B10)."""
        if "quality" not in self.results:
            return "not checked"
        return "flagged" if self.results["quality"].qc_flags else "passed"
