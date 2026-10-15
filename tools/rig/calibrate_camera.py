"""The camera's length calibration from a picture of a stage micrometer. The divisions' direction and period come from the picture's 2D
Fourier transform, the period is refined from the divisions themselves (a line profile across them, each
division's centre, a straight line through their positions), and um per pixel is the micrometer's spacing over that
period. The picture is read by Qt (no new dependency); a copy with the divisions found marked is written beside
the report, never over the picture.

    python tools/rig/calibrate_camera.py <picture.tif> <spacing in um> [--out folder]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui


def gray(path: Path) -> np.ndarray:
    image = QtGui.QImage(str(path))
    if image.isNull():
        raise ValueError(f"{path.name} cannot be read as a picture")
    image = image.convertToFormat(QtGui.QImage.Format_Grayscale8)
    ptr = image.constBits()
    ptr.setsize(image.bytesPerLine() * image.height())
    rows = np.frombuffer(ptr, np.uint8).reshape(image.height(), image.bytesPerLine())
    return rows[:, :image.width()].astype(float)


def direction_and_period(img: np.ndarray) -> tuple[float, float]:
    """(angle of the line across the divisions in radians, period in px), from the strongest non-zero peak of the
    2D spectrum of the picture's central square."""
    h, w = img.shape
    n = min(h, w)
    sq = img[(h - n) // 2:(h + n) // 2, (w - n) // 2:(w + n) // 2]
    sq = (sq - sq.mean()) * np.outer(np.hanning(n), np.hanning(n))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(sq)))
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - c, xx - c)
    spec[r < 4] = 0                                   # the slow shading, not the divisions
    spec[r > n // 3] = 0                              # nor pixel noise (a division is wider than 3 px)
    y, x = np.unravel_index(np.argmax(spec), spec.shape)
    fy, fx = (y - c) / n, (x - c) / n
    return float(np.arctan2(fy, fx)), float(1.0 / np.hypot(fy, fx))


def profile(img: np.ndarray, angle: float, band: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """The mean brightness along a line through the centre at ``angle``, over a band ``band`` px wide."""
    h, w = img.shape
    cy, cx = h / 2, w / 2
    half = int(0.45 * min(h, w))                      # inside the picture whatever the angle
    s = np.arange(-half, half)
    u, v = np.cos(angle), np.sin(angle)
    total = np.zeros(s.size)
    for o in np.linspace(-band / 2, band / 2, band):
        xs = np.clip(np.round(cx + s * u - o * v).astype(int), 0, w - 1)
        ys = np.clip(np.round(cy + s * v + o * u).astype(int), 0, h - 1)
        total += img[ys, xs]
    return s.astype(float), total / band


def refine(s: np.ndarray, y: np.ndarray, period: float) -> tuple[float, np.ndarray, float]:
    """(period, division centres along the line, SD of the residuals in px): each division found near where the
    rough period puts it (the profile's darker or brighter extreme, whichever stands out), a straight line through
    their positions."""
    y = y - np.convolve(y, np.ones(int(3 * period)) / int(3 * period), mode="same")      # take the shading off
    sign = 1.0 if np.max(y) - np.median(y) > np.median(y) - np.min(y) else -1.0
    z = sign * y
    first = int(np.argmax(z[:int(period) + 1]))
    centres = []
    k = 0
    while True:
        guess = first + k * period
        if guess >= z.size - period / 2:
            break
        a, b = max(0, int(guess - period / 3)), min(z.size, int(guess + period / 3) + 1)
        i = a + int(np.argmax(z[a:b]))
        if 0 < i < z.size - 1:                         # the peak's centre between samples (a parabola)
            d = z[i - 1] - 2 * z[i] + z[i + 1]
            i = i + (0.5 * (z[i - 1] - z[i + 1]) / d if d else 0.0)
        centres.append(float(i))
        k += 1
    index = np.arange(len(centres))
    fit = np.polyfit(index, centres, 1)
    residual = np.asarray(centres) - np.polyval(fit, index)
    return float(fit[0]), s[0] + np.asarray(centres), float(np.std(residual))


def measure(path: Path, spacing_um: float, out: Path | None = None) -> dict:
    img = gray(path)
    angle, rough = direction_and_period(img)
    s, y = profile(img, angle)
    # thin lines far apart put more power in a harmonic than in the divisions' own frequency: the shortest multiple
    # of the rough period whose divisions sit on a straight line (residuals under a tenth of it) is theirs
    for k in (1, 2, 3, 4):
        period, centres, sd = refine(s, y, rough * k)
        if sd < 0.1 * period:
            rough *= k
            break
    result = {"picture": path.name, "size_px": img.shape[::-1], "angle_deg": float(np.degrees(angle)) % 180,
              "rough_period_px": rough, "period_px": period, "divisions": len(centres), "residual_sd_px": sd,
              "um_per_px": spacing_um / period, "px_per_um": period / spacing_um}
    if out is not None:
        image = QtGui.QImage(str(path)).convertToFormat(QtGui.QImage.Format_RGB32)
        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 40, 40), 4))
        h, w = img.shape
        u, v = np.cos(angle), np.sin(angle)
        for c in centres:                              # a short red mark across the line at each division found
            x, yv = w / 2 + c * u, h / 2 + c * v
            painter.drawLine(QtCore.QPointF(x + 80 * v, yv - 80 * u), QtCore.QPointF(x - 80 * v, yv + 80 * u))
        painter.end()
        out.mkdir(parents=True, exist_ok=True)
        image.save(str(out / f"{path.stem}_divisions.png"))
    return result


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    r = measure(Path(argv[0]), float(argv[1]), out)
    print(f"{r['picture']}: {r['divisions']} divisions at {r['angle_deg']:.1f} deg, {r['period_px']:.3f} px apart "
          f"(rough {r['rough_period_px']:.2f}; residual SD {r['residual_sd_px']:.2f} px)")
    print(f"  {r['um_per_px']:.5f} um per pixel = {r['px_per_um']:.4f} px per um")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
