"""Reading files: Clampex ``.abf`` (ABF1 and ABF2), WinLTP sweep files, and
Axon Text Files (``.atf``, Clampfit's export). One ``Recording`` per file (or per
WinLTP series); headers first, samples on demand (ATF, being text, is read
whole).
"""

from __future__ import annotations

from pathlib import Path

from ..core.recording import Recording
from . import abf, atf, spiketimes, winltp

SUPPORTED_SUFFIXES = {".abf"} | winltp.SUFFIXES | atf.SUFFIXES | spiketimes.SUFFIXES


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def load(path: str | Path) -> Recording:
    """The recording in ``path``; a WinLTP sweep file loads its whole series."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".abf":
        return abf.load(p)
    if suffix in winltp.SUFFIXES:
        return winltp.load(p)
    if suffix in atf.SUFFIXES:
        return atf.load(p)
    if suffix in spiketimes.SUFFIXES:
        return spiketimes.load(p)
    raise ValueError(f"{p.name}: not a Clampex .abf, Axon text .atf, spike-time .asc or WinLTP sweep file")


def recording_path(path: str | Path) -> Path:
    """The name a recording is known by: the file, or the WinLTP series."""
    p = Path(path)
    return winltp.series_path(p) if p.suffix.lower() in winltp.SUFFIXES else p


def group_recordings(paths) -> tuple[list[Path], list[str]]:
    """One path per recording to analyze (ABF files as they are; WinLTP files
    collapsed to one per series, processed copies skipped), and notes saying
    what was left out and why."""
    abfs = sorted(Path(p) for p in paths if Path(p).suffix.lower() in {".abf"} | atf.SUFFIXES | spiketimes.SUFFIXES)
    wl = [Path(p) for p in paths if Path(p).suffix.lower() in winltp.SUFFIXES]
    chosen, notes = winltp.choose_series(wl)
    return abfs + chosen, notes


def find_recordings(folder: str | Path, recursive: bool = True) -> list[Path]:
    """Every supported file under ``folder``."""
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and is_supported(p))


__all__ = ["SUPPORTED_SUFFIXES", "find_recordings", "group_recordings", "is_supported", "load",
           "recording_path", "abf", "atf", "spiketimes", "winltp"]
