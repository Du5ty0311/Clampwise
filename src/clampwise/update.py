"""Updating Clampwise from its GitHub releases.

The newest release is read from GitHub's API (``releases/latest``: drafts and pre-releases are left out); when its
version is above this copy's, the user can update. The release's zip is downloaded and checked against the SHA-256
GitHub lists for it (a download that does not match is thrown away); it is unpacked beside the program, and a small
PowerShell script waits for Clampwise to close, swaps the program's folder for the new one (the old one kept until
the new one is in place, put back if the swap fails), and starts the new copy. Projects and settings live outside the
program's folder and are not touched. Nothing is installed without the user asking.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import __version__

REPO = "Du5ty0311/Clampwise"
ASSET = "Clampwise-windows.zip"
EXE = "Clampwise.exe"


def api_url() -> str:
    return os.environ.get("CLAMPWISE_UPDATE_URL") or f"https://api.github.com/repos/{REPO}/releases/latest"


@dataclass
class Release:
    version: str                  # "1.0.1"
    tag: str                      # "v1.0.1"
    notes: str
    page: str                     # the release's page on GitHub
    url: str                      # the zip
    size: int
    sha256: str


def parse_version(text: str) -> tuple[int, ...]:
    """'v1.2.3' or '1.2.3' -> (1, 2, 3); anything else -> ()."""
    m = re.fullmatch(r"v?(\d+(?:\.\d+)*)", text.strip())
    return tuple(int(p) for p in m.group(1).split(".")) if m else ()


def is_newer(version: str, current: str = __version__) -> bool:
    new, now = parse_version(version), parse_version(current)
    return bool(new) and new > now


def read_release(doc: dict) -> Release | None:
    """The release in GitHub's JSON, or None when it has no zip with a checksum (it is then not offered)."""
    tag = str(doc.get("tag_name") or "")
    if not parse_version(tag):
        return None
    for a in doc.get("assets") or []:
        if a.get("name") != ASSET:
            continue
        digest = str(a.get("digest") or "")
        if not digest.startswith("sha256:"):
            return None
        return Release(tag.lstrip("v"), tag, str(doc.get("body") or ""), str(doc.get("html_url") or ""),
                       str(a.get("browser_download_url") or ""), int(a.get("size") or 0), digest.split(":", 1)[1])
    return None


def latest(timeout: float = 8.0) -> Release | None:
    """The newest release on GitHub; raises OSError when GitHub cannot be reached."""
    req = urllib.request.Request(api_url(), headers={"Accept": "application/vnd.github+json",
                                                     "User-Agent": f"Clampwise/{__version__}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return read_release(json.loads(r.read().decode("utf-8")))


def download(release: Release, folder: Path, progress=None, cancelled=lambda: False) -> Path:
    """The release's zip in ``folder``, its SHA-256 checked; ``progress(done, total)`` as it comes."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"Clampwise-{release.version}.zip"
    part = target.with_suffix(".part")
    h = hashlib.sha256()
    req = urllib.request.Request(release.url, headers={"User-Agent": f"Clampwise/{__version__}"})
    done = 0
    with urllib.request.urlopen(req, timeout=30) as r, open(part, "wb") as out:
        total = int(r.headers.get("Content-Length") or release.size or 0)
        while True:
            if cancelled():
                out.close()
                part.unlink(missing_ok=True)
                raise RuntimeError("cancelled")
            block = r.read(1 << 20)
            if not block:
                break
            out.write(block)
            h.update(block)
            done += len(block)
            if progress is not None:
                progress(done, total)
    if h.hexdigest().lower() != release.sha256.lower():
        part.unlink(missing_ok=True)
        raise RuntimeError("the download does not match the checksum GitHub lists for it: not used")
    part.replace(target)
    return target


def install_folder() -> Path | None:
    """The program's own folder when running as the built program (an update replaces it), else None."""
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else None


SWAP = r"""param([int]$WaitPid, [string]$Install, [string]$Staged, [string]$Exe, [string]$Log, [switch]$NoLaunch)
# Clampwise's update: wait for it to close, swap its folder for the new one, start the new copy.
function Say($t) { Add-Content -Path $Log -Value ("{0:s} {1}" -f (Get-Date), $t) }
$old = "$Install.old"
try {
    if ($WaitPid) { Wait-Process -Id $WaitPid -Timeout 120 -ErrorAction SilentlyContinue }
    if (Test-Path $old) { Remove-Item -Recurse -Force $old }
    $moved = $false
    for ($i = 0; $i -lt 60 -and -not $moved; $i++) {
        try { Rename-Item -Path $Install -NewName (Split-Path $old -Leaf) -ErrorAction Stop; $moved = $true }
        catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $moved) { throw "the old copy is still in use" }
    try { Move-Item -Path $Staged -Destination $Install -ErrorAction Stop }
    catch { Rename-Item -Path $old -NewName (Split-Path $Install -Leaf); throw }
    Remove-Item -Recurse -Force $old -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force (Split-Path $Staged) -ErrorAction SilentlyContinue
    Say "updated $Install"
} catch { Say ("update failed, the old copy kept: " + $_) }
if (-not $NoLaunch) { Start-Process -FilePath (Join-Path $Install $Exe) }
"""


def update_now(launch: bool = False) -> str:
    """The whole update without a window (``Clampwise.exe --update``, for whoever looks after a lab's PCs): the newest
    release checked, downloaded, checked against its checksum and swapped in once this process has ended. What
    happened is written to the update log and returned."""
    install = install_folder()
    if install is None:
        result = "this copy runs from source: update it with git pull"
    else:
        r = latest()
        if r is None or not is_newer(r.version):
            result = f"Clampwise {__version__} is the newest version"
        else:
            staged = stage(download(r, log_path().parent / "updates"), install)
            start_swap(staged, install, launch=launch)
            result = f"updating Clampwise {__version__} to {r.version} once this copy has closed"
    log = log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')} --update: {result}\n")
    return result


def log_path() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Clampwise" / "update.log"


def stage(zip_path: Path, install: Path) -> Path:
    """The new copy unpacked beside ``install`` (same drive, so the swap is a rename); its folder. Raises
    PermissionError when the program's folder cannot be written (installed for all users: update by hand)."""
    staging = install.parent / f".{install.name}-update"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(staging)
    inner = staging / "Clampwise"
    if not (inner / EXE).is_file():
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"the download holds no Clampwise\\{EXE}")
    return inner


def start_swap(staged: Path, install: Path, pid: int | None = None, launch: bool = True) -> subprocess.Popen:
    """The swap started in its own process; it waits for ``pid`` (this program) to end. The caller then quits."""
    log = log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    script = log.with_name("update.ps1")
    script.write_text(SWAP, encoding="utf-8")
    args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", str(script),
            "-WaitPid", str(pid if pid is not None else os.getpid()), "-Install", str(install), "-Staged", str(staged),
            "-Exe", EXE, "-Log", str(log)] + ([] if launch else ["-NoLaunch"])
    # no window, and its own process group: it outlives Clampwise, which it waits for
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(args, creationflags=flags, close_fds=True)
