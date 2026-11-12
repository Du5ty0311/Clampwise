"""Where the tests' optional recordings are; a test whose recording is not there is skipped.

- The demo recordings (NeuroExpress's test files, which Clampwise ships): ``CLAMPWISE_DEMOS``, else the copy
  Clampwise keeps in ``%LOCALAPPDATA%\\Clampwise\\Demos`` once its demos have been opened.
- A lab's own recordings: the folders ``CLAMPWISE_LAB_DATA`` names (separated by ``;``), searched by file name.
- Particular recordings some tests need, by role: ``CLAMPWISE_LAB_FILES`` (a JSON file), else
  ``tests/lab_files.local.json``, which git ignores, so a lab's file names never enter the repository::

      {"light, whole field": "D:/data/.../Full_Field_....abf", "light, region 1": "...", "light, region 2": "...",
       "step family": "...", "unknown kind": "..."}

  The light files and the step family are one cell's, in one folder.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest


def _demos() -> Path:
    if os.environ.get("CLAMPWISE_DEMOS"):
        return Path(os.environ["CLAMPWISE_DEMOS"])
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Clampwise" / "Demos"


DEMOS = _demos()


def lab_roots() -> list[Path]:
    parts = [p.strip() for p in os.environ.get("CLAMPWISE_LAB_DATA", "").split(";")]
    return [Path(p) for p in parts if p and Path(p).is_dir()]


def find_all(pattern: str) -> list[Path]:
    """Every match of ``pattern`` (a file or folder name, or a path ending, ``**`` allowed) under the lab's folders."""
    out: list[Path] = []
    for root in lab_roots():
        out += [Path(h) for h in glob.glob(str(root / "**" / pattern), recursive=True)]
    return out


def find(pattern: str) -> Path | None:
    hits = find_all(pattern)
    return hits[0] if hits else None


def _local() -> dict:
    import json

    path = Path(os.environ.get("CLAMPWISE_LAB_FILES") or Path(__file__).with_name("lab_files.local.json"))
    try:
        return {k: Path(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError):
        return {}


LOCAL = _local()


def lab_file(role: str) -> Path:
    """The recording named for ``role`` on this machine (a path that does not exist when none is named)."""
    return LOCAL.get(role, Path("not-named") / role)


def lab_files(*roles: str) -> bool:
    """Whether every one of ``roles`` is named and its file is on this machine."""
    return all(lab_file(r).is_file() for r in roles)


def needs(pattern: str) -> Path:
    p = find(pattern)
    if p is None:
        pytest.skip(f"{pattern} is not among the recordings CLAMPWISE_LAB_DATA names")
    return p
