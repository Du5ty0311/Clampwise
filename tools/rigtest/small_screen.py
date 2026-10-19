"""Nothing cut off on a small screen: Clampwise's window at 1366 x 768 (a laptop's), Live's camera and patterned
light views (shown, not taken: no device is touched), and the camera's own window; each widget smaller than it needs
to be is listed, and a screenshot of each saved.

    python -I tools/rigtest/small_screen.py [--offscreen] [--size 1366 768]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402


def cut_off(root) -> list[str]:
    """The buttons, ticks and labels shown narrower or lower than they need (their text cut), outside a scroll area
    that would let them be reached."""
    from pyqtgraph.Qt import QtWidgets

    out = []
    for w in root.findChildren((QtWidgets.QAbstractButton, QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox)):
        if not w.isVisibleTo(root) or not w.isVisible():
            continue
        need = w.minimumSizeHint()
        if w.width() + 1 < need.width() or w.height() + 1 < need.height():
            text = getattr(w, "text", lambda: "")() or w.objectName() or type(w).__name__
            out.append(f"{type(w).__name__} '{text}': {w.width()} x {w.height()} px, needs {need.width()} x "
                       f"{need.height()}")
        # inside the window's visible area?
        g = w.mapTo(root, w.rect().bottomRight())
        if g.x() > root.width() + 1 or g.y() > root.height() + 1:
            area = w.parentWidget()
            while area is not None and not isinstance(area, QtWidgets.QAbstractScrollArea):
                area = area.parentWidget()
            if area is None:
                text = getattr(w, "text", lambda: "")() or type(w).__name__
                out.append(f"{type(w).__name__} '{text}' lies outside the window (no scrolling reaches it)")
    return out


def main() -> int:
    p = _common.parser(__doc__)
    p.add_argument("--size", type=int, nargs=2, default=[1366, 768])
    args = p.parse_args()
    window, app = _common.open_window(args, tuple(args.size))
    window.go("live")
    live = window.screens["live"]
    found = {}
    for role in ("camera", "light"):
        view = live.views[role]
        view.set_own(True)                             # its panel shown; the device is not taken
        _common.pump(0.5)
        found[f"Live, {role} view"] = cut_off(window)
        print(f"Live, {role} view: {_common.screenshot(window, f'small-{role}', args.out)}")
    for role, size in (("camera", None), ("light", (640, 480))):
        view = live.views[role]
        live.toggle_popped(role)                       # the view becomes a window of its own
        _common.pump(0.8)
        if size is not None:
            view.resize(*size)
            _common.pump(0.3)
        where = f"the {role} view's own window at {view.width()} x {view.height()}"
        found[where] = cut_off(view)
        print(f"{where}: {_common.screenshot(view, f'small-{role}-window', args.out)}")
        live.toggle_popped(role)
        _common.pump(0.3)
    for role in ("camera", "light"):
        live.views[role].set_own(False)
    bad = 0
    for where, items in found.items():
        print(f"{where}: " + ("nothing cut off" if not items else f"{len(items)} cut off"))
        for i in items:
            print(f"  {i}")
        bad += len(items)
    _common.leave_safely()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
