"""The patterned light's own window in Live: its field as the mirrors show it (what is lit, the steps to choose and
light), with what the mirrors do and the two buttons that change it. It needs no calibration."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ...rig import patterns
from .. import widgets
from ..widgets import box, label
from .polygon_panel import CHOSEN, LIT, PolygonPanel


class FieldView(QtWidgets.QWidget):
    """The patterned light's own window in Live: its field as the mirrors show it, in its 16:10 shape. What is lit
    now in white (a step, the checkerboard, a calibration picture), the pattern's steps outlined and numbered (yellow
    chosen, blue lit); a click chooses a step, Ctrl+click adds one or takes it out, a double-click lights it. Above,
    what the mirrors do and the two buttons that change it. Needs no calibration: it is the field itself, not the
    camera's picture of it."""

    def __init__(self, light: PolygonPanel) -> None:
        super().__init__()
        self.light = light
        lay = box("v", "xs")
        self.state = widgets.Badge("muted", "")
        self.state.setWordWrap(True)
        lay.addWidget(self.state)
        go = box("h", "s")
        self.on_button = widgets.primary("Mirrors on", light.project, "The chosen step(s) on the mirrors (lit "
                                         "whenever the LED fires)")
        self.off_button = widgets.button("All mirrors off", light.off, "No mirror lit: nothing reaches the slice")
        go.addWidget(self.on_button)
        go.addWidget(self.off_button)
        go.addStretch(1)
        lay.addLayout(go)
        self.view = pg.GraphicsLayoutWidget()
        self.view.setToolTip("The patterned light's field, as its mirrors show it: white is lit now. Click a step to "
                             "choose it, Ctrl+click for more, double-click to light it.")
        self.box = self.view.addViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.box.setMouseEnabled(False, False)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.box.addItem(self.image)
        edge = pg.PlotCurveItem([0, patterns.FIELD_W, patterns.FIELD_W, 0, 0], [0, 0, patterns.FIELD_H,
                                                                              patterns.FIELD_H, 0],
                                pen=pg.mkPen((128, 128, 128), width=1, style=QtCore.Qt.DashLine))
        self.box.addItem(edge)
        self.box.setRange(QtCore.QRectF(0, 0, patterns.FIELD_W, patterns.FIELD_H), padding=0.02)
        self.view.scene().sigMouseClicked.connect(self._clicked)
        lay.addWidget(self.view, 1)
        lay.addWidget(label("Its other controls (pattern, steps, calibration) are in the Patterned light section, "
                            "at the right.", "muted", wrap=True))
        self.setLayout(lay)
        self._items: list = []
        self._drawn = None                            # what was drawn last: nothing redrawn while it is the same
        light.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        """The state's words and the buttons as the panel has them; the field drawn again only when it changed."""
        p = self.light
        self.state.set(*p.state._shown)
        self.on_button.setEnabled(p.on_button.isEnabled())
        self.off_button.setEnabled(p.off_button.isEnabled())
        chosen = tuple(p._chosen_rows())
        key = (id(p.steps), len(p.steps), chosen, frozenset(p._lit), id(p.shown_mask))
        if key == self._drawn:
            return
        self._drawn = key
        m = p.shown_mask
        lit = np.zeros((patterns.ROWS // 2, patterns.COLUMNS // 2), np.uint8) if m is None else \
            (m[::2, ::2] * np.uint8(255)).astype(np.uint8)          # every 2nd mirror: the screen shows no more
        self.image.setImage(lit, autoLevels=False, levels=(0, 255))
        self.image.setRect(QtCore.QRectF(0, 0, patterns.FIELD_W, patterns.FIELD_H))
        for item in self._items:
            self.box.removeItem(item)
        self._items = []
        many = len(p.steps) > 100
        for k, step in enumerate(p.steps):
            bright, on = k in chosen, step.name in p._lit
            if many and not (bright or on):
                continue
            colour = LIT if on else CHOSEN
            pen = pg.mkPen(colour if bright or on else colour + (110,), width=2 if bright or on else 1)
            for shape in step.shapes:
                pts = shape.outline()
                item = pg.PlotCurveItem(pts[:, 0] * patterns.FIELD_W, pts[:, 1] * patterns.FIELD_H, pen=pen)
                self.box.addItem(item)
                self._items.append(item)
            if step.shapes and (len(p.steps) <= 25 or bright or on):
                cu, cv = step.shapes[0].centre()
                text = pg.TextItem(step.name, color=colour if bright or on else colour + (150,), anchor=(0.5, 0.5))
                text.setPos(cu * patterns.FIELD_W, cv * patterns.FIELD_H)
                self.box.addItem(text)
                self._items.append(text)

    def step_at(self, u: float, v: float) -> int | None:
        """The step covering (u, v) of the field (0 to 1 each way; the last made when several overlap)."""
        for k in range(len(self.light.steps) - 1, -1, -1):
            if self.light.steps[k].covers(u, v):
                return k
        return None

    def click_at(self, u: float, v: float, add: bool = False, light_it: bool = False) -> int | None:
        """A click at (u, v) of the field: the step there chosen (``add``: added or taken out), lit when asked."""
        row = self.step_at(u, v)
        if row is None:
            return None
        self.light.choose_at(row, add=add and not light_it)
        if light_it:
            self.light.project()
        return row

    def _clicked(self, event) -> None:
        if not self.box.sceneBoundingRect().contains(event.scenePos()):
            return
        at = self.box.mapSceneToView(event.scenePos())
        u, v = at.x() / patterns.FIELD_W, at.y() / patterns.FIELD_H
        if 0 <= u <= 1 and 0 <= v <= 1:
            add = bool(event.modifiers() & QtCore.Qt.ControlModifier)
            self.click_at(u, v, add=add, light_it=event.double())
