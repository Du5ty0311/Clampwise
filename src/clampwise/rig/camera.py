"""The camera, through Infinity Analyze's settings and the pictures it saves (read only).

Infinity Analyze (Teledyne Lumenera) owns the rig's camera while it runs, so Clampwise does not open the camera:
it reads the program's settings (``%LOCALAPPDATA%\\Teledyne Lumenera\\InfinityAnalyze\\InfinityAnalyze.ini``: the
camera model, the capture folder, the folder last saved to) and the pictures themselves. A cell's
pictures are typically saved (the pipette, its fluorescence) into the cell's data folder before recording it, so a recording's folder
holds its cell's pictures. Each TIFF is read for its size, its time, and the XMP block Infinity Analyze writes into
it (exposure, gain, the length calibration with its name), with no image library.
"""

from __future__ import annotations

import configparser
import datetime as dt
import os
import re
import struct
import xml.etree.ElementTree as ET
import copy
from dataclasses import dataclass, field
from pathlib import Path

from .files import modified, read_shared

SETTINGS = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / \
    "Teledyne Lumenera" / "InfinityAnalyze" / "InfinityAnalyze.ini"
PICTURES = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")
LUMEN = "{Lumenera/}"
MIN_UM_PER_PX = 0.01            # below this a calibration is doubtful (a 40x objective gives about 0.1-0.2 um a pixel)


@dataclass
class Settings:
    camera: str = ""
    capture_folder: str = ""
    last_saved_folder: str = ""
    file_format: str = ""


@dataclass
class Picture:
    path: Path
    taken: dt.datetime | None = None
    width: int = 0
    height: int = 0
    camera: str = ""
    exposure_ms: float | None = None
    gain: float | None = None
    calibration: str = ""                       # the calibration's name in Infinity Analyze
    um_per_px: float | None = None
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = [self.path.name]
        if self.width:
            parts.append(f"{self.width} x {self.height} px")
        if self.exposure_ms is not None:
            parts.append(f"exposure {self.exposure_ms:.3g} ms")
        if self.gain is not None:
            parts.append(f"gain {self.gain:.3g}")
        if self.um_per_px is not None:
            parts.append(f"{self.um_per_px:.4g} um per pixel ({self.calibration or 'calibration'})")
        return ", ".join(parts)


def read_settings(path: str | Path | None = None) -> Settings | None:
    """Infinity Analyze's settings, or None when it is not installed here."""
    path = Path(path or SETTINGS)
    if not path.is_file():
        return None
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(read_shared(path).decode("utf-8", errors="replace"))
    except (OSError, configparser.Error):
        return None

    def first(key: str, section: str | None = None) -> str:
        for name in ([section] if section else parser.sections()):
            if parser.has_option(name, key):
                return parser.get(name, key).strip()
        return ""

    return Settings(camera=first("ModelName", "Camera") if parser.has_section("Camera") else first("ModelName"),
                    capture_folder=first("OutputFileFolder"), last_saved_folder=first("LastWriteFolder"),
                    file_format=first("OutputFileFormat"))


# ---- TIFF tags, read directly (ImageWidth 256, ImageLength 257, Make 271, Model 272, DateTime 306, XMP 700)
_TYPES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8, 16: 8}


def tiff_tags(data: bytes) -> dict[int, object]:
    if len(data) < 8 or data[:2] not in (b"II", b"MM"):
        return {}
    order = "<" if data[:2] == b"II" else ">"
    if struct.unpack(order + "H", data[2:4])[0] != 42:
        return {}
    offset = struct.unpack(order + "I", data[4:8])[0]
    if offset + 2 > len(data):
        return {}
    count = struct.unpack(order + "H", data[offset:offset + 2])[0]
    out: dict[int, object] = {}
    for k in range(count):
        at = offset + 2 + 12 * k
        if at + 12 > len(data):
            break
        tag, kind, n = struct.unpack(order + "HHI", data[at:at + 8])
        size = _TYPES.get(kind, 1) * n
        raw = data[at + 8:at + 12] if size <= 4 else data[struct.unpack(order + "I", data[at + 8:at + 12])[0]:][:size]
        if kind == 2 or tag == 700:
            out[tag] = raw[:size]
        elif kind == 3:
            out[tag] = struct.unpack(order + "H", raw[:2])[0]
        elif kind == 4:
            out[tag] = struct.unpack(order + "I", raw[:4])[0]
    return out


def _number(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


_READ: dict[tuple, "Picture"] = {}          # each picture read once while it is unchanged (Live looks every 2 s)


def read_picture(path: str | Path) -> Picture:
    path = Path(path)
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and key in _READ:
        return copy.deepcopy(_READ[key])
    pic = _read_picture(path)
    if key is not None:
        if len(_READ) > 2000:
            _READ.clear()
        _READ[key] = copy.deepcopy(pic)
    return pic


def _read_picture(path: Path) -> Picture:
    pic = Picture(path)
    stamp = modified(path)
    if stamp:
        pic.taken = dt.datetime.fromtimestamp(stamp)
    if path.suffix.lower() not in (".tif", ".tiff"):
        return pic
    try:
        tags = tiff_tags(read_shared(path))
    except OSError:
        return pic
    pic.width, pic.height = int(tags.get(256, 0) or 0), int(tags.get(257, 0) or 0)
    if 306 in tags:
        try:
            pic.taken = dt.datetime.strptime(bytes(tags[306]).decode("ascii", "replace").strip("\x00 "), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
    pic.camera = " ".join(bytes(tags[t]).decode("ascii", "replace").strip("\x00 ") for t in (271, 272) if t in tags)
    xmp = bytes(tags.get(700, b""))
    if xmp:
        _read_xmp(pic, xmp)
    return pic


def _read_xmp(pic: Picture, xmp: bytes) -> None:
    text = xmp.decode("utf-8", errors="replace")
    text = text[text.find("<x:xmpmeta"):text.rfind("</x:xmpmeta>") + len("</x:xmpmeta>")]
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return
    for el in root.iter():
        if el.tag == LUMEN + "Camera":
            pic.exposure_ms, pic.gain = _number(el.get(LUMEN + "Exposure")), _number(el.get(LUMEN + "Gain"))
        elif el.tag == LUMEN + "Calibration" and el.get(LUMEN + "Type", "Length") == "Length":
            pic.calibration = el.get(LUMEN + "Name", "")
            source, dest = _number(el.get(LUMEN + "SourceDimension")), _number(el.get(LUMEN + "DestinationDimension"))
            unit = (el.get(LUMEN + "DestinationUnit") or "").replace("μ", "µ")
            to_um = {"µm": 1.0, "um": 1.0, "nm": 1e-3, "mm": 1e3}.get(unit)
            if source and dest and to_um:
                pic.um_per_px = dest * to_um / source
                if pic.um_per_px < MIN_UM_PER_PX:
                    pic.notes.append(f"the calibration '{pic.calibration}' gives {pic.um_per_px:.3g} um per pixel "
                                     f"({source:g} px for {dest:g} {unit}), far finer than a microscope camera "
                                     "resolves: check it in Infinity Analyze")


def pictures(folder: str | Path) -> list[Picture]:
    """The pictures in a folder (a cell's, beside its recordings), oldest first."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    found = [read_picture(p) for p in folder.iterdir() if p.suffix.lower() in PICTURES and p.is_file()]
    return sorted(found, key=lambda p: p.taken or dt.datetime.min)


def for_recording(path: str | Path) -> list[Picture]:
    """The cell's pictures: those saved in the recording's own folder."""
    return pictures(Path(path).parent)


def newest(folders) -> Picture | None:
    """The latest picture saved in any of ``folders`` (the capture folder, the folder last saved to)."""
    found = [p for f in folders if f for p in pictures(f)]
    return max(found, key=lambda p: p.taken or dt.datetime.min) if found else None


def watched_folders(settings: Settings | None) -> list[Path]:
    if settings is None:
        return []
    return [Path(re.sub(r"[\\/]+", lambda _: os.sep, f)) for f in (settings.last_saved_folder, settings.capture_folder)
            if f]
