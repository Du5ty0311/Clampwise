"""The analysis cascade for one recording: find the stimulus, decide the type,
run the analyses that apply, keep the provenance (Nylen & Wallisch 2017's
import-preprocess-analyze-summarize pipeline, as one function).

Nothing here knows the window. The same call serves the window, the command
line (``clampwise analyze``) and the Python API, so they give the same
numbers by construction.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import Any

from .. import __version__
from ..core.recording import Recording
from . import registry, routing, stimulus as stimulus_finder
from .result import Flag, Provenance, RecordingResult, Result
from .tools import ledger

log = logging.getLogger("clampwise")


@dataclass
class Context:
    """What every analysis may read besides the recording: the stimulus, the
    type decision, earlier results, and shared settings."""

    stimulus: Any
    decision: routing.Decision
    results: dict[str, Result] = field(default_factory=dict)
    shared: dict[str, Any] = field(default_factory=dict)        # e.g. sweep selection, stretch
    artifacts: list = field(default_factory=list)               # artifacts.find: what the signal shows

    def result(self, key: str) -> Result | None:
        return self.results.get(key)


@dataclass
class Settings:
    """Per-analysis settings, the shared ones, and the preset's name."""

    by_analysis: dict[str, dict[str, Any]] = field(default_factory=dict)
    shared: dict[str, Any] = field(default_factory=dict)
    preset: str = ""
    mappings: list[routing.Mapping] = field(default_factory=list)
    disabled: set[str] = field(default_factory=set)
    only: set[str] | None = None
    # per-recording overrides: typed stimulus times, typed type, the signal channel
    stimulus_override: str = ""
    type_override: str = ""
    stimulus_channel: str = ""
    rig: Any = None                       # what the rig's programs recorded of it (rig.evidence.Evidence), when read

    def for_analysis(self, key: str) -> dict[str, Any]:
        return dict(self.by_analysis.get(key, {}))


@dataclass
class Found:
    """What a recording is, before any analysis runs: as analyzed (typed steps added, the chosen sweeps), what
    its signal shows, its stimulus and its type."""

    recording: Recording
    full: Recording
    chosen_sweeps: str
    artifacts: list
    stimulus: Any
    decision: routing.Decision


def find(recording: Recording, settings: Settings | None = None) -> Found:
    """The first steps of every analysis, alone: the same for the analysis and for any screen that shows what a
    recording is before it is analyzed (Check), so the two never disagree."""
    settings = settings or Settings()
    recording = stimulus_finder.typed_steps(recording, settings.stimulus_override) or recording
    full = recording
    chosen_sweeps = str(settings.shared.get("sweeps", "") or "").strip()
    if chosen_sweeps and not recording.is_gap_free:
        from ..core.recording import parse_sweeps

        recording = recording.select_sweeps(parse_sweeps(chosen_sweeps, recording.n_sweeps))
    from . import artifacts

    found = artifacts.find(recording)
    stim = stimulus_finder.find_stimulus(recording, override=settings.stimulus_override,
                                         stimulus_channel=settings.stimulus_channel, artifacts=found, rig=settings.rig,
                                         light_inputs=settings.shared.get("light_inputs"))
    decision = routing.decide(recording, stim, mappings=settings.mappings, typed=settings.type_override)
    return Found(recording, full, chosen_sweeps, found, stim, decision)


def analyze(recording: Recording, settings: Settings | None = None, cache=None) -> RecordingResult:
    """``cache`` is a project store (or anything with ``cache_lookup``): an
    analysis whose file, settings and version it already holds is not re-run."""
    settings = settings or Settings()
    first = find(recording, settings)
    recording, full, chosen_sweeps = first.recording, first.full, first.chosen_sweeps
    found, stim, decision = first.artifacts, first.stimulus, first.decision
    context = Context(stim, decision, shared=dict(settings.shared), artifacts=found)
    out = RecordingResult(recording, decision.recording_type, decision.describe(),
                          stimulus_finder.describe(stim))
    if stim.source == "typed":
        out.notes.append("stimulus times were typed in; the file's own record was not used")
    if settings.rig is not None:
        out.notes += settings.rig.notes()
    if recording is not full:
        out.notes.append(f"sweeps chosen for this recording: {chosen_sweeps} ({recording.n_sweeps} of "
                         f"{full.n_sweeps}); sweep numbers in its tables count the chosen sweeps from 0")
    only = settings.only
    if decision.mapping is not None and decision.mapping.analyses:
        only = set(decision.mapping.analyses) if only is None else only & set(decision.mapping.analyses)
    versions, used = {}, {}
    # what a result depends on beyond its own settings, so the cache keys on it too: the type and the
    # stimulus it was given (its times too: the finder has no version of its own), and the full keys of
    # the results it reads (a changed baseline re-runs them)
    given = {"type": decision.recording_type, "stimulus": [stim.source, stim.kind, stim.channel, _digest(stim)]}
    for analysis in registry.for_type(decision.recording_type, disabled=settings.disabled, only=only):
        try:
            resolved, changed = analysis.resolve(settings.for_analysis(analysis.key))
        except ValueError as exc:
            out.notes.append(f"{analysis.label} not run: {exc}")
            continue
        shared = {**settings.shared, "given": given,
                  "reads": {k: [versions[k], used[k], out.cache_shared[k]["reads"]]
                            for k in analysis.requires if k in versions}}
        out.cache_shared[analysis.key] = shared
        cached = cache.cache_lookup(recording.content_hash, analysis, resolved, shared) if cache else None
        if cached is not None:
            out.results[analysis.key] = cached
            context.results[analysis.key] = cached
            versions[analysis.key] = analysis.version
            used[analysis.key] = resolved
            continue
        with ledger.recording() as methods:
            try:
                result = analysis.run(recording, context, resolved)
            except Exception as exc:           # one analysis never fails the recording
                log.exception("%s failed on %s", analysis.label, recording.source.name)
                result = Result(analysis.key, analysis.version,
                                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
                result.flags.append(Flag("software error", f"{analysis.label} could not be computed "
                                         f"({type(exc).__name__}: {exc}). A problem in Clampwise, not in the "
                                         "recording; the details are in the error log.", severity="flag"))
        if result is None:
            continue
        result.key, result.version = analysis.key, analysis.version
        result.methods = list(methods)
        result.settings, result.changed_settings = resolved, changed
        out.results[analysis.key] = result
        context.results[analysis.key] = result
        versions[analysis.key] = analysis.version
        used[analysis.key] = resolved
    out.provenance = Provenance.now(str(recording.source), recording.content_hash, __version__, versions, used,
                                    settings.preset)
    return out


def _digest(stim) -> str:
    """A short hash of every sweep's stimulus onsets and widths."""
    import hashlib

    import numpy as np

    h = hashlib.sha256()
    for k in range(max(len(stim.times_s), len(stim.widths_s))):
        for part in (stim.times_s, stim.widths_s):
            h.update(np.asarray(part[k] if k < len(part) else [], np.float64).tobytes() + b"|")
    return h.hexdigest()[:16]


def analyze_file(path, settings: Settings | None = None) -> RecordingResult:
    from ..io import load

    return analyze(load(path), settings)
