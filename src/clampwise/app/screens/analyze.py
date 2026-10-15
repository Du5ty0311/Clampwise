"""Analyze: the analyses these recordings call for, their settings, and Run
(B0 item 8, B18 item 9).

Nothing here is written per analysis: each section is built from the
analysis's declaration in the registry. A section shows what the analysis
does and the method it follows, a switch to run it or not, the few settings
people change, and the rest under "Advanced (n changed)". A changed setting is
marked and has "Reset to default". Settings are kept in the project, so a lab
member who opens it sees the same; changing one makes the next run re-measure
only the analyses it affects.
"""

from __future__ import annotations

from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from ...analysis import registry
from .. import theme, widgets
from ..widgets import box, label
from .base import Screen
from .files import TYPE_WORDS


class SettingControl(QtWidgets.QWidget):
    """One declared setting as a control with its label, hint, units and
    "Reset to default"; writes to the project as it changes."""

    changed = QtCore.Signal()

    def __init__(self, screen, analysis: registry.Analysis, setting: registry.Setting, value) -> None:
        super().__init__()
        self.screen, self.analysis, self.setting = screen, analysis, setting
        s = setting
        if isinstance(s.default, bool):
            self.control = QtWidgets.QCheckBox(s.label)
            self.control.setChecked(bool(value))
            self.control.toggled.connect(self._write)
        elif s.choices:
            self.control = QtWidgets.QComboBox()
            self.control.addItems([str(c) for c in s.choices])
            self.control.setCurrentText(str(value))
            self.control.currentTextChanged.connect(self._write)
        elif isinstance(s.default, int):
            self.control = QtWidgets.QSpinBox()
            self.control.setRange(int(s.minimum if s.minimum is not None else -10**6),
                                  int(s.maximum if s.maximum is not None else 10**6))
            self.control.setValue(int(value))
            self.control.editingFinished.connect(self._write)
        elif isinstance(s.default, float):
            self.control = QtWidgets.QDoubleSpinBox()
            d = abs(float(s.default))
            self.control.setDecimals(3 if 0 < d < 0.1 else 2 if 0 < d < 1 else 1)
            self.control.setRange(float(s.minimum if s.minimum is not None else -1e6),
                                  float(s.maximum if s.maximum is not None else 1e6))
            self.control.setValue(float(value))
            if s.units:
                self.control.setSuffix(f" {s.units}")
            self.control.editingFinished.connect(self._write)
        else:
            self.control = QtWidgets.QLineEdit(str(value))
            self.control.editingFinished.connect(self._write)
        self.control.setToolTip(s.help + (f"\nDefault: {s.default}{' ' + s.units if s.units else ''}"
                                          f"{'; ' + s.source if s.source else ''}"))
        lay = box("v", "xs")
        top = box("h", "xs")
        if not isinstance(s.default, bool):
            top.addWidget(label(s.label))
        self.mark = label("", "hint")
        top.addWidget(self.mark)
        top.addStretch(1)
        self.reset = widgets.button("Reset to default", self._reset, quiet=True)
        top.addWidget(self.reset)
        lay.addLayout(top)
        lay.addWidget(self.control)
        if s.help:
            lay.addWidget(label(s.help, "hint", wrap=True))
        self.setLayout(lay)
        self._refresh(value)

    def value(self):
        c, s = self.control, self.setting
        if isinstance(c, QtWidgets.QCheckBox):
            return c.isChecked()
        if isinstance(c, QtWidgets.QComboBox):
            text = c.currentText()
            return type(s.default)(text) if not isinstance(s.default, str) else text
        if isinstance(c, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            return c.value()
        return c.text()

    def _write(self, *_) -> None:
        project = self.screen.window.project
        if project is None:
            return
        v = self.value()
        try:
            if v == self.setting.default:
                project.reset_setting(self.analysis.key, self.setting.key)
            else:
                project.set_setting(self.analysis.key, self.setting.key, v)
        except ValueError as exc:
            self.screen.window.say(str(exc))
            return
        self._refresh(v)
        self.changed.emit()

    def _reset(self) -> None:
        project = self.screen.window.project
        if project is not None:
            project.reset_setting(self.analysis.key, self.setting.key)
        c, d = self.control, self.setting.default
        c.blockSignals(True)
        if isinstance(c, QtWidgets.QCheckBox):
            c.setChecked(bool(d))
        elif isinstance(c, QtWidgets.QComboBox):
            c.setCurrentText(str(d))
        elif isinstance(c, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            c.setValue(d)
        else:
            c.setText(str(d))
        c.blockSignals(False)
        self._refresh(d)
        self.changed.emit()

    def _refresh(self, v) -> None:
        differs = v != self.setting.default
        self.mark.setText("changed" if differs else "")
        self.reset.setVisible(differs)

    @property
    def differs(self) -> bool:
        return self.value() != self.setting.default


class AnalysisSection(QtWidgets.QFrame):
    """One analysis: what it does, its method, run it or not, its settings."""

    def __init__(self, screen, analysis: registry.Analysis, values: dict, enabled: bool, types_here: list[str]) -> None:
        super().__init__()
        self.setObjectName("surface")
        self.analysis = analysis
        lay = box("v", "s", "m")
        top = box("h", "s")
        self.run = QtWidgets.QCheckBox(analysis.label)
        self.run.setChecked(enabled)
        self.run.setToolTip("Run this analysis on the recordings it applies to")
        self.run.toggled.connect(lambda on: screen.set_enabled(analysis.key, on))
        top.addWidget(self.run, 1)
        where = ", ".join(sorted(TYPE_WORDS.get(t, t) for t in types_here))
        if analysis.level == "cell":
            where = f"a cell's files together ({where})"
        top.addWidget(label(where, "hint"))
        lay.addLayout(top)
        if analysis.description:
            lay.addWidget(label(analysis.description, "muted", wrap=True))
        if analysis.source:
            lay.addWidget(label(f"Reference: {analysis.source}", "hint", wrap=True))
        visible = [s for s in analysis.settings if not s.advanced and not s.hidden][:analysis.visible_settings]
        rest = [s for s in analysis.settings if s not in visible and not s.hidden]
        self.controls: list[SettingControl] = []
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(theme.SPACE["l"])
        grid.setVerticalSpacing(theme.SPACE["s"])
        for i, s in enumerate(visible):
            c = SettingControl(screen, analysis, s, values.get(s.key, s.default))
            self.controls.append(c)
            grid.addWidget(c, i // 2, i % 2)
        lay.addLayout(grid)
        if analysis.key == "events":
            # one template for the whole experiment, made from its own averaged events
            row = box("h", "s")
            origin = values.get("template_origin", "")
            self.template_note = label(f"Template from {origin}." if origin else
                                       "The template is the default shape. Make one for the experiment from its "
                                       "recordings' averaged events, and analyze every recording with it.", "hint",
                                       wrap=True)
            row.addWidget(self.template_note, 1)
            row.addWidget(widgets.button("Make the experiment's template", screen.make_template,
                                         "From the averaged events of every analyzed recording: one template, applied "
                                         "identically to all of them"))
            lay.addLayout(row)
        self.advanced = None
        self.advanced_controls: list[SettingControl] = []
        if rest:
            self.advanced = widgets.Advanced()
            for s in rest:
                c = SettingControl(screen, analysis, s, values.get(s.key, s.default))
                c.changed.connect(self._count)
                self.controls.append(c)
                self.advanced_controls.append(c)
                self.advanced.add(c)
            lay.addWidget(self.advanced)
            self._count()
        self.setLayout(lay)

    def _count(self) -> None:
        if self.advanced is not None:
            self.advanced.set_changed(sum(c.differs for c in self.advanced_controls))


class AnalyzeScreen(Screen):
    key = "analyze"
    title = "Analyze"
    purpose = "The analyses that apply to these recordings, and their settings. Defaults follow published " \
              "methods; change only what your experiment requires."

    def __init__(self, window) -> None:
        super().__init__(window)
        self.presets = QtWidgets.QComboBox()
        self.presets.setToolTip("A named set of settings for an experiment type; its name is printed in every export")
        self.apply_preset = widgets.button("Apply preset", self._apply_preset)
        self.preset_more = widgets.button("Presets", tooltip="Save the settings as a preset, or import and export "
                                          "presets as files")
        menu = QtWidgets.QMenu(self.preset_more)
        menu.addAction("Save the current settings as a preset...", self.save_preset)
        menu.addAction("Export the chosen preset to a file...", self.export_preset)
        menu.addAction("Import a preset from a file...", self.import_preset)
        self.preset_more.setMenu(menu)
        self.run_button = widgets.primary("Analyze", lambda: window.analyze_all(),
                                          "Run the analyses on every recording in the project, in the background; "
                                          "results whose file and settings did not change are kept")
        for w in (self.presets, self.apply_preset, self.preset_more, self.run_button):
            self.add_action(w)
        self.stack = QtWidgets.QStackedWidget()
        self.empty = widgets.EmptyState("No recordings yet", "Open a folder in Files: the analyses its recordings call "
                                        "for are listed here with their settings.",
                                        [("Go to Files", lambda: window.go("files"), True)])
        self.stack.addWidget(self.empty)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.stack.addWidget(self.scroll)
        self.body(self.stack)
        dl = box("v", "s", "m")
        dl.addWidget(label("How settings work", "heading"))
        dl.addWidget(label("Each analysis shows the settings people change; the rest are under Advanced, which says how "
                           "many differ from their default. Settings are saved in the project. A changed setting makes "
                           "the next run measure again only the analyses it affects; everything else is read from the "
                           "project's saved results.", "muted", wrap=True))
        dl.addStretch(1)
        self.details.setLayout(dl)

    def on_project(self) -> None:
        p = self.window.project
        if p is None or not p.recordings():
            self.stack.setCurrentIndex(0)
            self.run_button.setVisible(False)
            self.presets.setEnabled(False)
            self.apply_preset.setEnabled(False)
            return
        self.run_button.setVisible(True)
        self.presets.setEnabled(True)
        self.apply_preset.setEnabled(True)
        n = len(p.recordings())
        self.run_button.setText(f"Analyze {n} recording{'s' if n != 1 else ''}")
        self.presets.blockSignals(True)
        self.presets.clear()
        self.presets.addItems(list(p.presets()))
        current = p.meta("preset") or ""
        if current:
            self.presets.setCurrentText(current)
        self.presets.blockSignals(False)
        if self.is_current():
            self._build()
        else:
            self._stale = True               # built when Analyze is shown
            self.stack.setCurrentIndex(1)    # never the empty state's button beside the run button

    def on_shown(self) -> None:
        if getattr(self, "_stale", False):
            self._stale = False
            self._build()

    def _build(self) -> None:
        p = self.window.project
        types = sorted({r["recording_type"] for r in p.summaries() if r["recording_type"]})
        settings = p.settings()
        disabled = set(settings.get("shared", {}).get("disabled_analyses", []))
        page = QtWidgets.QWidget()
        lay = box("v", "m")
        if not types:
            lay.addWidget(label("These recordings are not analyzed yet, so their types are not known. Every analysis "
                                "is listed; after the first run only those that apply are.", "muted", wrap=True))
        shown = []
        for a in registry.load().values():
            here = [t for t in types if t in a.types] if types else sorted(a.types)
            if types and not here:
                continue
            shown.append(a.key)
            lay.addWidget(AnalysisSection(self, a, settings.get(a.key, {}), a.key not in disabled, here))
        lay.addStretch(1)
        page.setLayout(lay)
        self.scroll.setWidget(page)
        self.stack.setCurrentIndex(1)

    def set_enabled(self, key: str, on: bool) -> None:
        p = self.window.project
        shared = p.settings().get("shared", {})
        disabled = set(shared.get("disabled_analyses", []))
        (disabled.discard if on else disabled.add)(key)
        p.set_setting("shared", "disabled_analyses", sorted(disabled))

    def _apply_preset(self) -> None:
        p = self.window.project
        name = self.presets.currentText()
        if p is None or not name:
            return
        p.apply_preset(name)
        self.window.say(f"Preset '{name}' applied: its settings are now the project's. Analyze to measure with them.")
        self._build()

    def save_preset(self, name: str | None = None) -> None:
        """The project's settings now, kept as a preset of this name."""
        from ...project.presets import Preset

        p = self.window.project
        if p is None:
            return
        if name is None:
            name, ok = QtWidgets.QInputDialog.getText(self, "Save as a preset", "Name (e.g. the experiment type):")
            if not ok or not name.strip():
                return
        settings = p.settings()
        shared = settings.pop("shared", {})
        p.save_preset(Preset(name.strip(), f"Saved from {p.name}", settings, shared, p.mappings(),
                             source=f"project {p.name}"))
        self.window.say(f"Preset '{name.strip()}' saved in this project; Presets, Export shares it as a file.")
        self._build()

    def export_preset(self, path: str | None = None) -> None:
        import json

        p = self.window.project
        name = self.presets.currentText()
        if p is None or not name:
            return
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export the preset", f"{name}.cwpreset",
                                                            "Clampwise preset (*.cwpreset)")
        if path:
            Path(path).write_text(json.dumps(p.presets()[name].to_dict(), indent=1), encoding="utf-8")
            self.window.say(f"Wrote {Path(path).name}: anyone can import it (Analyze, Presets).")

    def import_preset(self, path: str | None = None) -> None:
        import json

        from ...project.presets import Preset

        p = self.window.project
        if p is None:
            return
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import a preset", "",
                                                            "Clampwise preset (*.cwpreset *.json)")
        if not path:
            return
        try:
            preset = Preset.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except Exception as exc:
            QtWidgets.QMessageBox.information(self, "Not a preset", f"{Path(path).name}: {exc}")
            return
        p.save_preset(preset)
        self._build()
        self.presets.setCurrentText(preset.name)
        self.window.say(f"Imported '{preset.name}': {preset.describe()}. Apply preset to use it.")

    def make_template(self) -> None:
        p = self.window.project
        if p is None:
            return
        tpl = p.make_event_template()
        if tpl.ok:
            self.window.say(f"Template made from {tpl.origin()}: rise tau {tpl.rise_tau_ms:.2f} / decay "
                            f"{tpl.decay_tau_ms:.1f} ms, set for every recording. Analyze to find events with it.")
        else:
            self.window.say(f"No template made: {tpl.note}.")
        self._build()
