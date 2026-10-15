"""The Python API (B18, item 8): the same numbers as the window, from a script.

    import clampwise
    results = clampwise.analyze(r"D:\\DSI_data\\260929", preset="CA3 sIPSC DSI")
    for r in results:
        print(r.recording.source.name, r.recording_type, r.values())

``analyze`` takes a file, a list of files or a folder; with a folder it uses
(or creates) the project in that folder, so results are cached and kept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .analysis import pipeline
from .analysis.result import RecordingResult
from .io import find_recordings, group_recordings, load


def analyze(target: str | Path | Iterable[str | Path], *, preset: str | None = None, project: bool = True,
            force: bool = False) -> list[RecordingResult]:
    """Analyze one file, several files, or every recording under a folder."""
    if isinstance(target, (str, Path)):
        t = Path(target)
        paths = find_recordings(t) if t.is_dir() else [t]
        folder = t if t.is_dir() else None
    else:
        paths = [Path(p) for p in target]
        folder = None
    paths, _ = group_recordings(paths)
    if folder is not None and project:
        from .project import Project

        store = Project.open(folder)
        try:
            if preset:
                store.apply_preset(preset)
            store.add_recordings(paths)
            return [store.analyze(p, force=force) for p in paths]
        finally:
            store.close()
    settings = pipeline.Settings(preset=preset or "")
    if preset:
        from .project.presets import BUILTIN

        if preset not in BUILTIN:
            raise ValueError(f"the preset '{preset}' is not built in; open the project folder to use a saved one")
        p = BUILTIN[preset]
        settings = pipeline.Settings(by_analysis=dict(p.settings), shared=dict(p.shared), preset=p.name,
                                     mappings=list(p.mappings))
    return [pipeline.analyze(load(p), settings) for p in paths]
