"""The camera's picture as Clampwise keeps it: the scale (um a pixel at each objective, from a calibration or the
camera's own pixel), the slice's compass, a snapshot as an uncompressed TIFF of the camera's own pixels with how it was
taken written into it and the scale as its resolution (ImageJ and Fiji read it), the gain that keeps a shorter
exposure as bright, and auto exposure's measure of the picture. Nothing here changes a picture: it is shown and saved
as the camera gives it (Waters 2009, J Cell Biol 185:1135; Rossner & Yamada 2004 on keeping originals).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import struct
from pathlib import Path

import numpy as np


def is_grey(rgb: np.ndarray) -> bool:
    """A monochrome camera's frame: one grey channel seen as three (a view, not a copy)."""
    return rgb.ndim == 3 and rgb.strides[2] == 0


def lum8(rgb: np.ndarray) -> np.ndarray:
    """The brightness, uint8: the grey channel itself, else the channels' mean."""
    if is_grey(rgb):
        return rgb[:, :, 0]
    return (rgb.sum(axis=2, dtype=np.uint16) // 3).astype(np.uint8)


OBJECTIVES = (2.5, 5.0, 10.0, 20.0, 40.0, 63.0)     # the slider's stops; each microscope's own can be set


def parse_objectives(text: str) -> list[float]:
    """'2.5, 5, 20x, 63' -> [2.5, 5.0, 20.0, 63.0] (sorted, no repeats); the standard list when nothing is readable."""
    out = set()
    for part in str(text).replace(";", ",").split(","):
        part = part.strip().lower().rstrip("x").strip()
        try:
            v = float(part)
        except ValueError:
            continue
        if v > 0:
            out.add(v)
    return sorted(out) or list(OBJECTIVES)


# the sensor's pixel, um, by camera model (the maker's specifications): a nominal scale before any calibration
PIXEL_UM = {"INFINITY8-3M": 4.5,            # Sony IMX429, 1944 x 1472, 4.5 x 4.5 um (Lumenera; Edmund Optics)
            # Basler ace (Basler's specifications), each for its monochrome (um, gm) and colour (uc, gc) versions
            # and their variants (MED): Sony IMX249 and IMX174, 1920 x 1200, 5.86 um; IMX264 and IMX250,
            # 2448 x 2048, 3.45 um; PYTHON 1300, 1280 x 1024, 4.8 um
            "ACA1920-40": 5.86, "ACA1920-155": 5.86, "ACA2440-35": 3.45, "ACA2440-75": 3.45, "ACA1300-200": 4.8}


def pixel_um_of(model: str) -> float:
    """The sensor's pixel (um) for a camera model as its program or the camera names it ('INFINITY8-3M',
    'acA1920-40um', 'Basler acA1920-40umMED'), else 0."""
    key = str(model or "").upper().replace("LUMENERA", "").replace("BASLER", "").strip()
    if key in PIXEL_UM:
        return PIXEL_UM[key]
    m = re.match(r"(ACA\d+-\d+)[A-Z]", key)                  # a Basler ace: its version's letters after the model
    return PIXEL_UM.get(m.group(1), 0.0) if m else 0.0


def objective_scale(mag: float, calibrated: dict, pixel_um: float = 0.0, adapter: float = 1.0) -> tuple[float, str]:
    """um a pixel with the ``mag`` objective, and how it is known. Its own calibration (a stage micrometer measured
    with it) first; else another objective's, scaled by the ratio of their magnifications (the camera sees each
    objective's image through the same tube and adapter, so um a pixel goes as 1 / magnification: the nearest
    calibrated objective is used); else nominal, from the sensor's own pixel: ``pixel_um`` / (``mag`` x the camera
    adapter's magnification), which holds as far as the objective's and the adapter's magnifications are their
    stated ones; else 0, not known."""
    cal = {float(k): float(v) for k, v in calibrated.items() if float(v) > 0}
    if cal.get(float(mag)):
        return cal[float(mag)], f"calibrated at {mag:g}x"
    if cal:
        ref = min(cal, key=lambda m: abs(np.log(m / mag)))
        return cal[ref] * ref / mag, f"estimated from the {ref:g}x calibration"
    if pixel_um > 0 and adapter > 0:
        return pixel_um / (mag * adapter), (f"nominal: the camera's {pixel_um:g} um pixels / ({mag:g}x objective x "
                                            f"{adapter:g}x camera adapter)")
    return 0.0, "not calibrated"


# a slice's plane: its two axes in the picture (the first the one turned, the second put on a side of it), and the
# axis cut across, out of the picture
PLANES = {"horizontal": (("rostral", "caudal"), ("medial", "lateral"), ("dorsal", "ventral")),
          "coronal": (("dorsal", "ventral"), ("medial", "lateral"), ("rostral", "caudal")),
          "sagittal": (("dorsal", "ventral"), ("rostral", "caudal"), ("medial", "lateral"))}
# how the directions are named: rostral-caudal as such or as anterior-posterior, in words or by their letters
NAMINGS = ("Rostral, caudal", "Anterior, posterior", "Letters: R C D V M L", "Letters: A P D V M L")


def direction_name(direction: str, naming: str = NAMINGS[0]) -> str:
    """``direction`` (rostral, caudal, dorsal, ventral, medial or lateral) as ``naming`` (one of NAMINGS) names it."""
    anterior, letters = NAMINGS.index(naming) in (1, 3), NAMINGS.index(naming) in (2, 3)
    if anterior:
        direction = {"rostral": "anterior", "caudal": "posterior"}.get(direction, direction)
    return direction[0].upper() if letters else direction.capitalize()


def in_words(naming: str) -> str:
    """``naming`` in words: rostral-caudal kept as such or as anterior-posterior."""
    return NAMINGS[1] if NAMINGS.index(naming) in (1, 3) else NAMINGS[0]


def axis_name(pair: tuple[str, str], naming: str = NAMINGS[0]) -> str:
    """An axis by its two ends' letters, as atlases write it (RC or AP, DV, ML)."""
    letters = NAMINGS[3] if NAMINGS.index(naming) in (1, 3) else NAMINGS[2]
    return "".join(direction_name(d, letters) for d in pair)


def compass(up_deg: float, side_right: bool = True, plane: str = "horizontal",
            naming: str = NAMINGS[0]) -> dict[str, tuple[float, float]]:
    """The slice's directions in the picture, as unit vectors (x to the right, y down, as the picture's rows run),
    named as ``naming`` names them, the first the plane's turned direction. The plane's first axis
    (rostral in a horizontal slice; dorsal in a coronal or a sagittal one) points ``up_deg`` clockwise from up, its
    other end opposite; the second axis's first end (medial in a horizontal or a coronal slice, rostral in a sagittal
    one) a quarter turn clockwise of it (to its right, as one faces it) or anticlockwise. The axis the slice is cut
    across (dorsal-ventral in a horizontal slice, rostral-caudal in a coronal one, medial-lateral in a sagittal one)
    is out of the picture and not drawn."""
    (a0, a1), (b0, b1), _ = PLANES[plane]
    a = np.radians(float(up_deg) % 360.0)
    rx, ry = float(np.sin(a)), float(-np.cos(a))
    mx, my = (-ry, rx) if side_right else (ry, -rx)
    clean = lambda v: 0.0 if abs(v) < 1e-12 else v
    n = lambda d: direction_name(d, naming)
    return {n(a0): (clean(rx), clean(ry)), n(a1): (clean(-rx), clean(-ry)),
            n(b0): (clean(mx), clean(my)), n(b1): (clean(-mx), clean(-my))}


def scale_bar(um_per_px: float, width_px: int) -> tuple[float, int]:
    """(length in um, length in px) of a round scale bar about a fifth of the picture's width."""
    target = um_per_px * width_px / 5
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000):
        if step >= target * 0.6:
            return float(step), int(round(step / um_per_px))
    return 2000.0, int(round(2000 / um_per_px))


def write_tiff(path: str | Path, rgb: np.ndarray, info: dict, um_per_px: float | None = None) -> Path:
    """An uncompressed RGB TIFF of ``rgb`` with ``info`` (JSON) as its description and, when calibrated, its
    resolution in pixels a centimetre (the scale ImageJ reads)."""
    path = Path(path)
    h, w, _ = rgb.shape
    desc = (json.dumps(info, default=str) + "\0").encode("utf-8")
    entries = []                                  # (tag, type, count, value or bytes)
    data_at = 8
    extra = bytearray()
    n_tags = 13 if um_per_px else 10
    ifd_size = 2 + 12 * n_tags + 4
    base = data_at + ifd_size

    def put(blob: bytes) -> int:
        offset = base + len(extra)
        extra.extend(blob)
        if len(extra) % 2:
            extra.append(0)
        return offset

    bits = put(struct.pack("<3H", 8, 8, 8))
    desc_at = put(desc)
    res = None
    if um_per_px:
        per_cm = 1e4 / um_per_px
        res = put(struct.pack("<2I", int(round(per_cm * 1000)), 1000))
    pixels = rgb.astype(np.uint8).tobytes()
    strip = base + len(extra)
    entries += [(256, 4, 1, w), (257, 4, 1, h), (258, 3, 3, bits), (259, 3, 1, 1), (262, 3, 1, 2),
                (270, 2, len(desc), desc_at), (273, 4, 1, strip), (277, 3, 1, 3), (278, 4, 1, h),
                (279, 4, 1, len(pixels))]
    if res is not None:
        entries += [(282, 5, 1, res), (283, 5, 1, res), (296, 3, 1, 3)]
    entries.sort()
    ifd = struct.pack("<H", len(entries))
    for tag, typ, count, value in entries:
        if typ == 3 and count == 1:
            ifd += struct.pack("<HHIHH", tag, typ, count, value, 0)
        else:
            ifd += struct.pack("<HHII", tag, typ, count, value)
    ifd += struct.pack("<I", 0)
    path.write_bytes(b"II*\0" + struct.pack("<I", data_at) + ifd + bytes(extra) + pixels)
    return path


def snapshot_name(folder: str | Path, label: str = "") -> Path:
    """A new file name in ``folder``: the time, and the label (the objective, GFP) when given."""
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{stamp}_{label}".rstrip("_").replace(" ", "_")
    p = Path(folder) / f"{stem}.tif"
    k = 2
    while p.exists():
        p = Path(folder) / f"{stem}_{k}.tif"
        k += 1
    return p


def gain_for(gain: float, brighter: float, kind: str, low: float, high: float) -> tuple[float, float]:
    """The gain that makes the picture ``brighter`` times as bright, kept in [low, high], and how much brighter
    that is: ``kind`` is how the camera counts gain ("linear", a multiplier; "db"; "raw", its own steps, not
    changed)."""
    if kind == "linear":
        new = min(max(gain * brighter, low), high)
        return new, new / gain if gain > 0 else 1.0
    if kind == "db":
        new = min(max(gain + 20.0 * float(np.log10(max(brighter, 1e-9))), low), high)
        return new, float(10.0 ** ((new - gain) / 20.0))
    return gain, 1.0



TARGET = 0.80 * 255                  # auto exposure: the brightest 0.5% at 80% of full scale, nothing clipped
AUTO_MAX_MS = 1000.0                 # auto exposure goes no longer (a longer one is typed): a dark picture is no light


def stats(raw: np.ndarray, bins: int = 64) -> dict:
    """The histogram of the frame's brightness, its brightest 0.5% and how much of it is clipped at either end, from
    a count of the 256 levels (15 ms on the rig camera's grey frame; a sort of every pixel took 217 ms). A colour
    frame is read every second pixel each way (the same histogram, a quarter of the work)."""
    if not is_grey(raw) and raw.shape[0] * raw.shape[1] > 1_000_000:
        raw = raw[::2, ::2]
    lum = lum8(raw)
    counts = np.bincount(lum.ravel(), minlength=256)
    total = max(int(counts.sum()), 1)
    hist = counts.reshape(bins, 256 // bins).sum(axis=1)
    p995 = float(np.searchsorted(np.cumsum(counts), 0.995 * total))
    mean = float((counts * np.arange(256)).sum() / total)
    if is_grey(raw):
        clipped, black = counts[255] / total, counts[0] / total
    else:
        top = raw.max(axis=2)
        clipped = float(np.count_nonzero(top == 255)) / top.size
        black = float(np.count_nonzero(top == 0)) / top.size
    return {"hist": hist, "counts": counts, "p995": p995, "mean": mean, "clipped": float(clipped),
            "black": float(black)}


def next_exposure(ms: float, p995: float, low: float, high: float, clipped: float = 0.0,
                  step: float | None = None) -> float:
    """An exposure that brings the brightest 0.5% to 80% of full scale (halved while anything is clipped), at most
    ``step`` times longer or shorter than ``ms`` when given (a dark picture asks for 100 times more at once)."""
    if clipped > 0.001 or p995 >= 254:
        new = ms / 2
    else:
        new = ms * TARGET / max(p995, 1.0)
    if step:
        new = min(max(new, ms / step), ms * step)
    return float(min(max(new, low), high))
