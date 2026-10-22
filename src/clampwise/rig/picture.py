"""The camera's picture as Clampwise shows and saves it: beside the exposure and gain, contrast, brightness,
sharpness, clarity and the rest, each easy to change.

The adjustments change what is shown, never the picture taken: a snapshot keeps the camera's own pixels, with how
it was taken (exposure, gain, frames averaged) and the adjustments that were on written into it, and the length
calibration as the TIFF's resolution (ImageJ and Fiji read it as the scale). Each adjustment is one number with a
neutral value, so any of them can be reset, and a set of them can be kept by name (a preset: brightfield, GFP).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import struct
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
from scipy import ndimage


@dataclass
class Adjust:
    """What is shown: each at its neutral value changes nothing."""

    black: float = 0.0            # the levels (the histogram's handles): this grey level and below shown black,
    white: float = 255.0          # this one and above white, those between stretched over the whole range
    brightness: float = 0.0       # -1..1: added to every pixel (a fraction of the full range)
    contrast: float = 1.0         # 0..3: stretched about mid-grey
    gamma: float = 1.0            # 0.2..5: below 1 lifts the shadows (dim fluorescence), above darkens them
    saturation: float = 1.0       # 0..3: 0 grey, 1 as taken
    sharpness: float = 0.0        # 0..3: an unsharp mask of fine detail (1 px)
    clarity: float = 0.0          # 0..3: local contrast (an unsharp mask 20 px wide)
    auto_levels: bool = False     # stretch the darkest 0.5% to black and the brightest 0.5% to white
    red: float = 1.0              # 0..3 each: the colour balance (white balance sets them)
    green: float = 1.0
    blue: float = 1.0

    def neutral(self) -> bool:
        return self == Adjust()

    def changed(self) -> dict:
        base = Adjust()
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != getattr(base, f.name)}


RANGES = {"brightness": (-1.0, 1.0), "contrast": (0.0, 3.0), "gamma": (0.2, 5.0), "saturation": (0.0, 3.0),
          "sharpness": (0.0, 3.0), "clarity": (0.0, 3.0), "red": (0.0, 3.0), "green": (0.0, 3.0), "blue": (0.0, 3.0)}


def is_grey(rgb: np.ndarray) -> bool:
    """A monochrome camera's frame: one grey channel seen as three (a view, not a copy)."""
    return rgb.ndim == 3 and rgb.strides[2] == 0


def lum8(rgb: np.ndarray) -> np.ndarray:
    """The brightness, uint8: the grey channel itself, else the channels' mean."""
    if is_grey(rgb):
        return rgb[:, :, 0]
    return (rgb.sum(axis=2, dtype=np.uint16) // 3).astype(np.uint8)


def _levels(x: np.ndarray, a: Adjust) -> np.ndarray:
    """``x`` (0-1, as taken) with the levels set: ``a.black`` and below to 0, ``a.white`` and above to 1."""
    if a.black == 0.0 and a.white == 255.0:
        return x
    lo, hi = float(a.black) / 255.0, float(a.white) / 255.0
    return np.clip((x - lo) / max(hi - lo, 1.0 / 255.0), 0.0, 1.0)


def _point_lut(a: Adjust, gain: float) -> np.ndarray:
    """What ``apply`` does to one channel's 256 levels when only point operations are set."""
    x = _levels(np.arange(256, dtype=np.float32) / 255.0, a) * gain
    x = np.clip((x - 0.5) * a.contrast + 0.5 + a.brightness, 0.0, 1.0)
    if a.gamma != 1.0:
        x = x ** (1.0 / a.gamma)
    return np.round(x * 255.0).astype(np.uint8)


def apply(rgb: np.ndarray, a: Adjust) -> np.ndarray:
    """The picture (H, W, 3) uint8 as shown with ``a``; the input is not changed. Brightness, contrast, gamma and the
    colour balance alone (saturation too, on a grey picture) are a lookup of each level: 5 ms on the rig camera's
    2.9 MP frame, where working in floating point took 43 ms."""
    if a.neutral():
        return rgb
    if not (a.auto_levels or a.sharpness or a.clarity) and (a.saturation == 1.0 or is_grey(rgb)):
        if is_grey(rgb) and a.red == a.green == a.blue:
            g = _point_lut(a, a.red)[rgb[:, :, 0]]
            return np.broadcast_to(g[:, :, None], rgb.shape)
        out = np.empty(rgb.shape, np.uint8)
        for c, gain in enumerate((a.red, a.green, a.blue)):
            out[:, :, c] = _point_lut(a, gain)[rgb[:, :, c]]
        return out
    x = _levels(rgb.astype(np.float32) / 255.0, a)
    x = x * np.array([a.red, a.green, a.blue], np.float32)
    if a.auto_levels:
        lo, hi = np.percentile(x, (0.5, 99.5))
        x = (x - lo) / max(hi - lo, 1e-6)
    if a.sharpness:
        x = x + a.sharpness * (x - ndimage.gaussian_filter(x, (1.0, 1.0, 0)))
    if a.clarity:
        lum = x.mean(axis=2, keepdims=True)
        x = x + a.clarity * 0.5 * (lum - ndimage.gaussian_filter(lum, (20.0, 20.0, 0)))
    if a.saturation != 1.0:
        lum = x.mean(axis=2, keepdims=True)
        x = lum + a.saturation * (x - lum)
    x = (x - 0.5) * a.contrast + 0.5 + a.brightness
    x = np.clip(x, 0.0, 1.0)
    if a.gamma != 1.0:
        x = x ** (1.0 / a.gamma)
    return np.round(x * 255.0).astype(np.uint8)


def white_balance(rgb: np.ndarray) -> tuple[float, float, float]:
    """Red, green and blue gains that make the picture's mean grey (point the camera at an empty, bright field)."""
    means = rgb.reshape(-1, 3).astype(np.float64).mean(axis=0)
    means = np.maximum(means, 1.0)
    return tuple(float(means.mean() / m) for m in means)


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
    named as ``naming`` names them, the first the plane's turned direction (drawn stronger). The plane's first axis
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


def preset_to_text(a: Adjust) -> str:
    return json.dumps(asdict(a))


def preset_from_text(text: str) -> Adjust:
    d = json.loads(text)
    return fold_tone(Adjust(**{f.name: d[f.name] for f in fields(Adjust) if f.name in d}))


def fold_tone(a: Adjust) -> Adjust:
    """Brightness and contrast (kept before 1.8, which set them with sliders) as the levels that do the same: the
    histogram's black and white handles are now the one place they are set. Exact while the stretch is a
    contrast of 1 or more (a lower one, which greys black, is taken as the nearest the levels can show)."""
    if a.brightness == 0.0 and a.contrast == 1.0:
        return a
    c = max(float(a.contrast), 1e-3)
    span = float(a.white) - float(a.black)
    black = a.black + span * (0.5 - (0.5 + a.brightness) / c)
    white = a.black + span * (0.5 + (0.5 - a.brightness) / c)
    black = min(max(black, 0.0), 254.0)
    white = min(max(white, black + 1.0), 255.0)
    return replace(a, black=round(black, 2), white=round(white, 2), brightness=0.0, contrast=1.0)


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


# ---- microscopy corrections and aids, for patching (sources in Help: Waters 2009 JCB 185:1135; Lambert & Waters
# 2014, Methods Cell Biol 123; Murphy & Davidson 2012; Allen, Allen & Travis 1981 and Inoue 1981 for video-enhanced
# contrast; Stuart, Dodt & Sakmann 1993 for infrared video microscopy of slices; Rossner & Yamada 2004 on originals)

LUTS = ("colour", "grey", "green", "magenta", "fire")
TARGET = 0.80 * 255                  # auto exposure: the brightest 0.5% at 80% of full scale, nothing clipped
AUTO_MAX_MS = 1000.0                 # auto exposure goes no longer (a longer one is typed): a dark picture is no light
AUTO_STEP = 4.0                      # each of its steps at most 4 times longer or shorter
AUTO_BUDGET_S = 3.0                  # and it stops trying after about this long


@dataclass
class Corrections:
    """What is done to each frame before it is shown (never to the frame saved)."""

    dark: np.ndarray | None = None          # the camera's own offset: frames with the light blocked, averaged
    background: np.ndarray | None = None    # an empty or defocused field, averaged
    mode: str = "off"                       # "off", "subtract" (video-enhanced contrast), "flat" (flat-field)
    lut: str = "colour"                     # how a picture is coloured on screen
    show_clipped: bool = False              # saturated pixels painted red (the camera's own full scale): off, the
                                            # picture shown raw unless asked for
    overlay: np.ndarray | None = None       # a kept picture (fluorescence), 0-1, laid over the live one
    overlay_colour: tuple = (0, 255, 0)
    overlay_opacity: float = 0.6
    overlay_on: bool = True

    def active(self) -> list[str]:
        out = []
        if self.dark is not None:
            out.append("dark frame subtracted")
        if self.background is not None and self.mode == "subtract":
            out.append("background subtracted (video-enhanced contrast)")
        if self.background is not None and self.mode == "flat":
            out.append("flat-field corrected")
        if self.lut != "colour":
            out.append(f"shown in {self.lut}")
        if self.overlay is not None and self.overlay_on:
            out.append(f"overlay at {self.overlay_opacity:.0%}")
        return out


def _lut(name: str) -> np.ndarray:
    x = np.linspace(0, 1, 256)
    if name == "green":
        rgb = np.stack([0 * x, x, 0 * x], 1)
    elif name == "magenta":
        rgb = np.stack([x, 0 * x, x], 1)
    elif name == "fire":
        rgb = np.stack([np.clip(3 * x, 0, 1), np.clip(3 * x - 1, 0, 1), np.clip(3 * x - 2, 0, 1)], 1)
    else:
        rgb = np.stack([x, x, x], 1)
    return np.round(rgb * 255).astype(np.uint8)


def corrected(raw: np.ndarray, c: Corrections) -> np.ndarray:
    """The frame with the dark frame and the background (or flat field) taken out, uint8."""
    if c.dark is None and (c.background is None or c.mode == "off"):
        return raw
    x = raw.astype(np.float32)
    if c.dark is not None and c.dark.shape == x.shape:
        x = np.maximum(x - c.dark, 0)
    bg = c.background
    if bg is not None and bg.shape == x.shape and c.mode != "off":
        if c.dark is not None and c.dark.shape == bg.shape:
            bg = np.maximum(bg - c.dark, 0)
        if c.mode == "subtract":
            x = x - bg + float(bg.mean())
        else:
            x = x / np.maximum(bg / max(float(bg.mean()), 1e-6), 1e-3)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def shown(raw: np.ndarray, c: Corrections, a: Adjust) -> np.ndarray:
    """The frame as shown: corrected, adjusted, coloured, the overlay laid over it, clipped pixels in red."""
    x = apply(corrected(raw, c), a)
    if c.lut != "colour":
        x = _lut(c.lut)[lum8(x)]
    if c.overlay is not None and c.overlay_on and c.overlay.shape == x.shape[:2]:
        m = (c.overlay * c.overlay_opacity)[..., None]
        x = np.round(x * (1 - m) + np.array(c.overlay_colour, np.float32) * m).astype(np.uint8)
    if c.show_clipped:
        clipped = raw[:, :, 0] == 255 if is_grey(raw) else (raw >= 255).any(axis=2)
        if clipped.any():
            x = x.copy()
            x[clipped] = (255, 0, 0)
    return x


def display_table(c: Corrections, a: Adjust) -> np.ndarray | None:
    """For a grey frame: what ``shown`` does, as a table of the 256 levels (256, 3) uint8, when it can be one (point
    adjustments only, no correction or overlay); the screen then draws the camera's own pixels through it. None when
    the picture needs ``shown`` itself. A frame with a clipped pixel took 51 ms to draw through ``shown`` (red pixels
    make a colour copy); through the table, a few."""
    if a.auto_levels or a.sharpness or a.clarity or c.mode != "off" and c.background is not None or c.dark is not None:
        return None
    if c.overlay is not None and c.overlay_on:
        return None
    levels = np.arange(256, dtype=np.uint8)[:, None, None]                  # the 256 levels as a grey picture
    grey = np.broadcast_to(levels, (256, 1, 3))
    table = np.ascontiguousarray(apply(grey, a)).reshape(256, 3)
    if c.lut != "colour":
        table = _lut(c.lut)[(table.sum(axis=1, dtype=np.uint16) // 3).astype(np.uint8)]
    if c.show_clipped:
        table = table.copy()
        table[255] = (255, 0, 0)
    return table


def overlay_from(raw: np.ndarray, c: Corrections) -> np.ndarray:
    """A kept picture as an overlay: its brightness above its own background, 0-1 (the brightest 0.5% at 1)."""
    lum = corrected(raw, Corrections(dark=c.dark)).astype(np.float32).mean(axis=2)
    lum = lum - ndimage.median_filter(lum[::8, ::8], size=9).repeat(8, 0).repeat(8, 1)[:lum.shape[0], :lum.shape[1]]
    top = float(np.percentile(lum, 99.5))
    return np.clip(lum / max(top, 1e-6), 0, 1).astype(np.float32)


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


BINNINGS = (1, 2, 4)


def bin_sum(rgb: np.ndarray, n: int) -> np.ndarray:
    """``n`` x ``n`` pixels added into one, clipped at full scale: n squared times the light a pixel (a dim
    fluorescent picture as bright at 1/n squared of the exposure, so that many times faster), at 1/n of the
    resolution each way. Added from strided views, never a reshaped copy (3.6 ms for 2 x 2 on the rig camera's
    2.9 MP frame; a reshape and sum took 18 ms)."""
    if n <= 1:
        return rgb
    grey = is_grey(rgb)
    x = rgb[:, :, 0] if grey else rgb
    h, w = x.shape[0] // n * n, x.shape[1] // n * n
    acc = x[0:h:n, 0:w:n].astype(np.uint16)
    for i in range(n):
        for j in range(n):
            if i or j:
                acc += x[i:h:n, j:w:n]
    out = np.minimum(acc, 255).astype(np.uint8)
    return np.broadcast_to(out[:, :, None], out.shape + (3,)) if grey else out


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
