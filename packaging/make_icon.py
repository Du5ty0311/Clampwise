"""Draw the Clampwise icon: a white action potential on the app's deep green.

    python packaging/make_icon.py

Each size is drawn at its own scale (a downsampled spike blurs at 16 px).
Writes packaging/clampwise.ico (the exe) and app/resources/icon.png (the
window). Needs Pillow, a development tool only.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
RESOURCES = HERE.parent / "src" / "clampwise" / "app" / "resources"
GREEN = (45, 106, 79)            # theme.LIGHT.accent
SIZES = (16, 24, 32, 48, 64, 128, 256)

# The trace in a unit square, y up: rest, upstroke, peak, repolarization,
# undershoot, recovery.
TRACE = ((0.10, 0.34), (0.36, 0.34), (0.42, 0.40), (0.50, 0.88), (0.56, 0.86), (0.64, 0.30),
         (0.70, 0.22), (0.80, 0.30), (0.90, 0.34))


def draw(size: int) -> Image.Image:
    scale = 4                                  # drawn large and reduced once, for smooth edges
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.2), fill=GREEN)
    width = max(2 * scale, int(s * (0.09 if size <= 24 else 0.065)))
    d.line([(x * s, (1 - y) * s) for x, y in TRACE], fill="white", width=width, joint="curve")
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    images = [draw(n) for n in SIZES]
    images[-1].save(HERE / "clampwise.ico", sizes=[(n, n) for n in SIZES], append_images=images[:-1])
    RESOURCES.mkdir(exist_ok=True)
    images[-1].save(RESOURCES / "icon.png")


if __name__ == "__main__":
    main()
