"""Files: open recordings and say who each one is (B1, B10).

Overview first: one row per recording with its identity, its type and one
quality badge (passed, flagged, not checked). The details panel edits the
selected recordings' identity; every field says whether it was typed or
read from the path, and the edit can be undone (Ctrl+Z).

With the cell's Notes.txt read and episodes worked out (features), each file
shows its label from the notes, its episode and its rig, with any doubt in the
tooltip. "Overview" shows every animal, slice and cell (analyzed, flagged,
reviewed, exported, what is missing) and the audit trail: every edit and
decision, who made it, each undoable, the whole exportable.
"""

from __future__ import annotations

import datetime
import json

from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from ...analysis.registry import TYPE_WORDS
from ...core.recording import Identity
from .. import theme, widgets
from ..widgets import box, label
from .base import Screen

COLUMNS = ("File", "Animal", "Slice", "Cell", "Condition", "Label", "Episode", "Type", "Quality", "Review", "Rig",
           "Analyzed")
MISSING = (("age", "age"), ("sex", "sex"), ("drugs", "drugs"), ("internal_solution", "internal"))


class FilesScreen(Screen):
    key = "files"
    title = "Files"
    purpose = "Open a folder of recordings. Each recording is listed with its animal, slice, cell and condition, " \
              "its recording type and its quality status."

    def __init__(self, window) -> None:
        super().__init__(window)
        self.open_button = widgets.button("Open folder…", window.open_folder, "Open a folder as a project (Ctrl+O)")
        self.analyze_button = widgets.primary("Analyze", window.analyze_all,
                                              "Analyze every recording in the project, in the background")
        more = widgets.button("More", tooltip="Less common actions")
        menu = QtWidgets.QMenu(more)
        menu.addAction("Add files…", window.add_files)
        menu.addAction("Open the demos", window.open_demos)
        menu.addAction("Re-analyze everything (ignore saved results)", lambda: window.analyze_all(force=True))
        menu.addAction("Remove the selected recordings from the project", self._remove)
        menu.addSeparator()
        menu.addAction("Name the rig of the selected recordings...", self._name_rig)
        menu.addSeparator()
        self.watch_action = menu.addAction("Watch this folder for new recordings")
        self.watch_action.setCheckable(True)
        self.watch_action.setToolTip("While Clampex writes into the folder, each finished recording is added and "
                                     "analyzed")
        self.watch_action.toggled.connect(window.set_watching)
        more.setMenu(menu)
        self.overview_button = widgets.button("Overview", self._toggle_overview, "Every animal, slice and cell, and "
                                              "the audit trail of every edit")
        self.overview_button.setCheckable(True)
        for w in (self.open_button, self.overview_button, more, self.analyze_button):
            self.add_action(w)

        self.stack = QtWidgets.QStackedWidget()
        self.empty = widgets.EmptyState(
            "No recordings yet",
            "Open the folder containing an experiment's recordings. Clampwise saves the analysis in a project file in "
            "that folder, so work resumes where it was left. Supported formats: Axon .abf and .atf, WinLTP sweep files "
            "and spike-time files (.asc, .spk).",
            [("Open a folder…", window.open_folder, True), ("Open the demos", window.open_demos, False)])
        self.stack.addWidget(self.empty)
        listing = QtWidgets.QWidget()
        lay = box("v", "s")
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Filter by file, animal, slice, cell, condition or type")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        lay.addWidget(self.filter)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setSortingEnabled(True)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)     # the file name is never cut
        hdr.setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._selected)
        self.table.doubleClicked.connect(lambda _: window.go("check"))
        lay.addWidget(self.table, 1)
        self.count = label("", "muted")
        lay.addWidget(self.count)
        listing.setLayout(lay)
        self.stack.addWidget(listing)
        self.overview = Overview(window)
        self.stack.addWidget(self.overview)
        self.body(self.stack)
        self._rows: list[dict] = []
        self._build_details()

    # ---- details: identity of the selection
    def _build_details(self) -> None:
        lay = box("v", "m", "m")
        lay.addWidget(label("Recording identity", "heading"))
        self.who = label("Select one or more recordings.", "muted", wrap=True)
        lay.addWidget(self.who)
        self.edits: dict[str, QtWidgets.QLineEdit] = {}
        self.sources: dict[str, QtWidgets.QLabel] = {}
        for key, caption, hint in (
                ("animal", "Animal", ""), ("slice", "Slice", ""), ("cell", "Cell", ""),
                ("condition", "Condition", "e.g. control, carbachol, after TTX")):
            e = QtWidgets.QLineEdit()
            e.setPlaceholderText("blank keeps what is there")
            self.edits[key] = e
            f = widgets.field(caption, e, hint)
            lay.addWidget(f)
            src = label("", "hint", wrap=True)
            self.sources[key] = src
            lay.addWidget(src)
        self.apply = widgets.button("Apply to the selected recordings", self._apply_identity,
                                    "Typed values replace the ones read from the folder names; Ctrl+Z undoes")
        lay.addWidget(self.apply)
        lay.addWidget(widgets.button("Naming rules...", self._rules, "See and edit how animal, slice, cell and "
                                     "condition are read from folder and file names", quiet=True))
        # what the experiment was: typed, never invented (B1, B20)
        self.experiment = widgets.Advanced("Experiment details")
        self.meta_edits: dict[str, QtWidgets.QLineEdit] = {}
        for key, caption, hint in (
                ("rs_start", "Rs at the start (MOhm)", "Clampex's Membrane Test; checked against the quality limits "
                                                       "when the file has no test pulse"),
                ("rs_end", "Rs at the end (MOhm)", ""),
                ("sex", "Sex", ""), ("age", "Age", "e.g. P35, 8 weeks"), ("genotype", "Genotype", ""),
                ("region", "Region", "e.g. BLA, CA3"), ("cell_type", "Cell type", "e.g. pyramidal, PV interneuron"),
                ("temperature_c", "Temperature (°C)", "")):
            e = QtWidgets.QLineEdit()
            e.setPlaceholderText("blank keeps what is there")
            self.meta_edits[key] = e
            self.experiment.add(widgets.field(caption, e, hint))
        self.apply_meta = widgets.button("Apply experiment details", self._apply_metadata,
                                         "Applied to the selected recordings; analyze again to use them")
        self.experiment.add(self.apply_meta)
        lay.addWidget(self.experiment)
        lay.addStretch(1)
        scroll_body = QtWidgets.QWidget()
        scroll_body.setLayout(lay)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(scroll_body)
        outer = box("v")
        outer.addWidget(scroll)
        self.details.setLayout(outer)
        self._enable_identity(False)

    def _enable_identity(self, on: bool) -> None:
        for e in list(self.edits.values()) + list(self.meta_edits.values()):
            e.setEnabled(on)
        self.apply.setEnabled(on)
        self.apply_meta.setEnabled(on)

    # ---- data
    def on_project(self) -> None:
        self.watch_action.blockSignals(True)
        self.watch_action.setChecked(self.window.watching)
        self.watch_action.blockSignals(False)
        p = self.window.project
        if p is None:
            self.stack.setCurrentWidget(self.empty)
            self.analyze_button.setEnabled(False)
            self.analyze_button.setVisible(False)          # the empty state's button is the one primary
            return
        self._rows = p.summaries()
        if self.overview_button.isChecked() and self._rows:
            self.overview.refresh()
            self.stack.setCurrentWidget(self.overview)
        else:
            self.stack.setCurrentIndex(1 if self._rows else 0)
            # filled while hidden behind the Overview, the badges are placed again once the list shows
            QtCore.QTimer.singleShot(0, self.table.doItemsLayout)
        self.analyze_button.setVisible(bool(self._rows))
        n = len(self._rows)
        self.analyze_button.setText(f"Analyze {n} recording{'s' if n != 1 else ''}")
        self.analyze_button.setEnabled(n > 0)
        self._fill()

    def _fill(self) -> None:
        selected = {self.table.item(r.row(), 0).data(QtCore.Qt.UserRole)
                    for r in self.table.selectionModel().selectedRows()} if self.table.rowCount() else set()
        self.table.setSortingEnabled(False)
        org = self.window.project.organization
        features = self.window.project.features()
        # which columns show is settled before the badges are placed, so none is drawn where a column was
        self.table.setColumnHidden(COLUMNS.index("Review"), not features["file_review"])
        self.table.setColumnHidden(COLUMNS.index("Episode"), org is None or not features["session_grouping"])
        self.table.setColumnHidden(COLUMNS.index("Label"), not any(r["label"] for r in self._rows))
        self.table.setColumnHidden(COLUMNS.index("Rig"), org is None or not any(
            m.rig is not None or m.doubts for m in org.rigs.values()))
        self.table.setRowCount(len(self._rows))
        for i, row in enumerate(self._rows):
            flags = org.note_flags(row["path"]) if org is not None else []
            doubts = org.episode_doubts(row["path"]) if org is not None else []
            episode = org.describe(row["path"]) if org is not None else ""
            match = org.rigs.get(Path(row["path"])) if org is not None else None
            rig = match.rig.name if match is not None and match.rig is not None else ""
            values = (row["name"], row["animal"], row["slice"], row["cell"], row["condition"],
                      ("⚠ " if flags and row["label"] else "") + row["label"], ("⚠ " if doubts else "") + episode,
                      TYPE_WORDS.get(row["recording_type"], row["recording_type"]), "", row["review"],
                      ("⚠ " if match is not None and match.doubts else "") + rig,
                      row["analyzed_at"].replace("T", " ")[:16] if row["analyzed_at"] else "not yet")
            for j, v in enumerate(values):
                item = QtWidgets.QTableWidgetItem(v)
                if j == 0:
                    item.setData(QtCore.Qt.UserRole, str(row["path"]))
                    item.setToolTip(str(row["path"]))
                elif j == COLUMNS.index("Review") and v:
                    item.setToolTip(f"Marked {v} on Check" + (f": {row['review_reason']}" if row["review_reason"] else ""))
                elif j == COLUMNS.index("Label") and v:
                    item.setToolTip(f"From {row['label_source'] or 'typed'}" + "".join(f"\n⚠ {d}" for d in flags))
                elif j == COLUMNS.index("Episode") and v:
                    item.setToolTip("Episode number and this file's part in it (before, step, after)"
                                    + "".join(f"\n⚠ {d}" for d in doubts))
                elif j == COLUMNS.index("Rig") and match is not None:
                    item.setToolTip(match.why + "".join(f"\n⚠ {d}" for d in match.doubts))
                self.table.setItem(i, j, item)
            status, word = widgets.QUALITY.get(row["quality"], ("muted", row["quality"]))
            badge = widgets.Badge(status, word, "Quality checks: " + (
                "not run yet; analyze the recording" if row["quality"] == "not checked" else row["quality"]))
            # the badge shows the word; the item under it carries it only for sorting and the filter
            sort_item = QtWidgets.QTableWidgetItem()
            sort_item.setData(QtCore.Qt.UserRole + 1, word)
            sort_item.setData(QtCore.Qt.AccessibleTextRole, word)
            self.table.setItem(i, COLUMNS.index("Quality"), sort_item)
            self.table.setCellWidget(i, COLUMNS.index("Quality"), badge)
            if str(row["path"]) in selected:
                self.table.selectRow(i)
        self.table.setSortingEnabled(True)
        # ResizeToContents does not measure cell widgets: give the badge column the badge's own width
        q = COLUMNS.index("Quality")
        widest = max((self.table.cellWidget(i, q).sizeHint().width() for i in range(self.table.rowCount())
                      if self.table.cellWidget(i, q) is not None), default=0)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(q, QtWidgets.QHeaderView.Fixed)
        header.resizeSection(q, max(widest + theme.SPACE["m"], header.sectionSizeHint(q)))
        self.table.doItemsLayout()          # the badges follow the columns' new widths at once
        self._apply_filter(self.filter.text())
        flagged = sum(r["quality"] == "flagged" for r in self._rows)
        reviewing = features["file_review"]
        reviewed = sum(bool(r["review"]) for r in self._rows)
        cells = len({(r["animal"], r["slice"], r["cell"]) for r in self._rows})
        episodes = org.count() if org is not None and features["session_grouping"] else 0
        flags = [f for n in (org.notes.values() if org is not None else []) for f in n.flags]
        self.count.setText(f"{len(self._rows)} recording(s) from {cells} cell(s)"
                           + (f" · {episodes} episode(s)" if episodes else "")
                           + (f" · {flagged} flagged" if flagged else "")
                           + (f" · {reviewed} reviewed" if reviewing and reviewed else "")
                           + (f" · ⚠ {len(flags)} difference(s) from Notes.txt" if flags else ""))
        self.count.setToolTip("\n".join(flags))

    def _apply_filter(self, text: str) -> None:
        words = text.lower().split()
        for i in range(self.table.rowCount()):
            hay = " ".join((self.table.item(i, j).text() or str(self.table.item(i, j).data(QtCore.Qt.UserRole + 1) or "")
                            if self.table.item(i, j) else "") for j in range(len(COLUMNS))).lower()
            self.table.setRowHidden(i, not all(w in hay for w in words))

    def _toggle_overview(self) -> None:
        self.on_project()

    def _name_rig(self) -> None:
        """Name the rig the selected files were recorded on; its preset applies to its files after."""
        from ...project import rigs

        paths = self.selected_paths()
        if not paths:
            self.window.say("Select the recordings of one rig first.")
            return
        known = rigs.Rigs()
        name, ok = QtWidgets.QInputDialog.getItem(self, "Name the rig", "This rig's name (e.g. Rig 1, the opto rig):",
                                                  list(known.rigs) or ["Rig 1"], 0, True)
        if not ok or not name.strip():
            return
        try:
            rig = known.learn(name.strip(), [rigs.fingerprint(p) for p in paths])
        except ValueError as exc:
            QtWidgets.QMessageBox.information(self, "Not one rig", str(exc))
            return
        presets = ["(no preset)"] + list(self.window.project.presets())
        preset, ok = QtWidgets.QInputDialog.getItem(self, "The rig's preset", f"Settings for {rig.name}'s files "
                                                    "(applied when Rig-aware presets is on):", presets,
                                                    max(0, presets.index(rig.preset)) if rig.preset in presets else 0,
                                                    False)
        if ok:
            known.set_preset(rig.name, "" if preset == "(no preset)" else preset)
        text, typed = QtWidgets.QInputDialog.getText(
            self, "Inputs that carry light", f"The recorded inputs that carry light on {rig.name}, whatever their "
            "names say (comma-separated; empty for none):", text=", ".join(rig.light_inputs))
        if typed:
            today = datetime.date.today().isoformat()
            known.set_light_inputs(rig.name, {n: rig.light_inputs.get(n) or f"the rig's profile, {today}"
                                              for n in (x.strip() for x in text.split(",")) if n})
        self.window.say(f"{rig.name}: {rigs.Fingerprint(rig.format, rig.digitizer, rig.amplifier, rig.channels[0], rig.units[0]).describe()}"
                        + (f"; preset {preset}" if ok and preset != "(no preset)" else ""))
        rigs._cache.clear()
        self.window.organize()

    def selected_paths(self) -> list[Path]:
        return [Path(self.table.item(r.row(), 0).data(QtCore.Qt.UserRole))
                for r in self.table.selectionModel().selectedRows()]

    def _selected(self) -> None:
        paths = self.selected_paths()
        self._enable_identity(bool(paths))
        for e in self.edits.values():
            e.clear()
        if not paths:
            self.who.setText("Select one or more recordings.")
            for s in self.sources.values():
                s.setText("")
            return
        p = self.window.project
        for e in self.meta_edits.values():
            e.clear()
            e.setPlaceholderText("blank keeps what is there")
        if len(paths) == 1:
            meta = p.metadata(paths[0])
            shown = {"rs_start": meta.rs_typed_mohm[0], "rs_end": meta.rs_typed_mohm[1], "sex": meta.sex,
                     "age": meta.age, "genotype": meta.genotype, "region": meta.region, "cell_type": meta.cell_type,
                     "temperature_c": meta.temperature_c}
            for k, v in shown.items():
                text = f"{v:g}" if isinstance(v, float) and v == v else (v if isinstance(v, str) else "")
                self.meta_edits[k].setPlaceholderText(text or "not set")
            ident = p.identity(paths[0])
            self.who.setText(paths[0].name)
            for k, e in self.edits.items():
                e.setPlaceholderText(getattr(ident, k) or "not set")
                src = ident.sources.get(k, "")
                self.sources[k].setText(f"from {src}" if src and src != "typed" else ("typed" if src else ""))
        else:
            self.who.setText(f"{len(paths)} recordings: a typed field applies to all of them; blanks leave each "
                             "as it is.")
            for k, e in self.edits.items():
                e.setPlaceholderText("blank keeps each one's own")
                self.sources[k].setText("")
        self.window.select(paths[0])

    def _apply_metadata(self) -> None:
        typed = {k: e.text().strip() for k, e in self.meta_edits.items() if e.text().strip()}
        if not typed:
            self.window.say("Nothing typed: the experiment details are unchanged.")
            return
        numbers = {}
        for k in ("rs_start", "rs_end", "temperature_c"):
            if k in typed:
                try:
                    numbers[k] = float(typed[k].replace(",", "."))
                except ValueError:
                    self.window.say(f"'{typed[k]}' is not a number; nothing was applied.")
                    return
        p = self.window.project
        paths = self.selected_paths()
        for path in paths:
            meta = p.metadata(path)
            for k in ("sex", "age", "genotype", "region", "cell_type"):
                if k in typed:
                    setattr(meta, k, typed[k])
            if "temperature_c" in numbers:
                meta.temperature_c = numbers["temperature_c"]
            if "rs_start" in numbers or "rs_end" in numbers:
                a, b = meta.rs_typed_mohm
                meta.rs_typed_mohm = (numbers.get("rs_start", a), numbers.get("rs_end", b))
            p.set_metadata(path, meta)
        self.window.say(f"Experiment details applied to {len(paths)} recording(s). Analyze again to use them.")
        self._selected()

    def _rules(self) -> None:
        from ..dialogs import RulesDialog

        p = self.window.project
        if p is None:
            return
        examples = self.selected_paths() or [r.path for r in p.recordings()[:4]]
        dlg = RulesDialog(self, p.rules, examples, p.folder)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            changed = p.set_rules(dlg.result_rules)
            self.window.say(f"Naming rules saved: {changed} recording(s) read differently now (typed values kept).")
            self.window.project_changed()

    def select_path(self, path) -> None:
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).data(QtCore.Qt.UserRole) == str(path):
                self.table.selectRow(i)
                return

    def _apply_identity(self) -> None:
        typed = Identity(**{k: e.text().strip() for k, e in self.edits.items()})
        if not any((typed.animal, typed.slice, typed.cell, typed.condition)):
            self.window.say("Nothing typed: the identity is unchanged.")
            return
        self.window.set_identity(self.selected_paths(), typed)

    def _remove(self) -> None:
        paths = self.selected_paths()
        if not paths:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Remove from the project?",
            f"Remove {len(paths)} recording(s) from this project? The files on disk are not touched; their saved "
            "results and decisions in the project are removed.")
        if answer == QtWidgets.QMessageBox.Yes:
            for p in paths:
                self.window.project.remove_recording(p)
            self.window.project_changed()


def _change(kind: str, before, after) -> str:
    """An audit entry's change in a few words."""
    if isinstance(before, dict) and isinstance(after, dict):
        diff = [f"{k}: {before.get(k)!r} -> {after.get(k)!r}" for k in after if k != "sources"
                and str(before.get(k)) != str(after.get(k))]          # NaN is NaN: not a change
        return "; ".join(diff)[:160] or "no change"
    if before is None and after is not None:
        return f"added {json.dumps(after, default=str)[:150]}"
    if after is None and before is not None:
        return f"removed {json.dumps(before, default=str)[:150]}"
    return f"{json.dumps(before, default=str)[:70]} -> {json.dumps(after, default=str)[:70]}"


class Overview(QtWidgets.QWidget):
    """Every animal, slice and cell, and the audit trail."""

    HEAD = ("Animal / slice / cell", "Files", "Analyzed", "Flagged", "Reviewed", "Episodes", "Exported",
            "Missing details")

    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        lay = box("v", "s")
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(self.HEAD)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemDoubleClicked.connect(self._open_cell)
        lay.addWidget(self.tree, 3)
        self.nudge = label("", "muted", wrap=True)
        lay.addWidget(self.nudge)
        head = box("h", "s")
        head.addWidget(label("Audit trail", "heading"))
        head.addStretch(1)
        self.undo_button = widgets.button("Undo this change", self._undo, "Put back what the selected entry changed; "
                                          "the undoing is recorded too")
        head.addWidget(self.undo_button)
        head.addWidget(widgets.button("Export the audit trail...", self._export, "Every entry as a CSV file",
                                      quiet=True))
        lay.addLayout(head)
        self.audit = QtWidgets.QTableWidget(0, 6)
        self.audit.setHorizontalHeaderLabels(["When", "What", "Which", "Change", "By", "Undone"])
        self.audit.verticalHeader().setVisible(False)
        self.audit.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.audit.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.audit.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.audit.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.audit.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.audit, 2)
        self.setLayout(lay)

    def refresh(self) -> None:
        p = self.window.project
        self.tree.clear()
        self.audit.setRowCount(0)
        if p is None:
            return
        org = p.organization
        last_export = p.meta("last_export") or ""
        cells: dict[tuple, list[dict]] = {}
        for r in p.summaries():
            cells.setdefault((r["animal"] or "(no animal)", r["slice"] or "(no slice)", r["cell"] or "(no cell)"),
                             []).append(r)
        nodes: dict[tuple, QtWidgets.QTreeWidgetItem] = {}
        missing_total = 0
        for (animal, slice_, cell), rows in sorted(cells.items()):
            meta = p.metadata(rows[0]["path"])
            missing = [w for k, w in MISSING if not getattr(meta, k)]
            missing_total += bool(missing)
            analyzed = [r["analyzed_at"] for r in rows if r["analyzed_at"]]
            exported = "yes" if last_export and analyzed and max(analyzed) <= last_export else \
                ("not since the last analysis" if last_export and analyzed else "no")
            key = "/".join(x for x in (rows[0]["animal"], rows[0]["slice"], rows[0]["cell"]))
            eps = len(org.episodes.get(key, [])) if org is not None else 0
            values = [cell, str(len(rows)), str(len(analyzed)), str(sum(r["quality"] == "flagged" for r in rows)),
                      str(sum(bool(r["review"]) for r in rows)), str(eps or ""), exported, ", ".join(missing)]
            parent = None
            for depth, name in enumerate((animal, slice_)):
                k = (animal, slice_)[:depth + 1]
                if k not in nodes:
                    nodes[k] = QtWidgets.QTreeWidgetItem([name] + [""] * (len(self.HEAD) - 1))
                    (parent.addChild(nodes[k]) if parent is not None else self.tree.addTopLevelItem(nodes[k]))
                parent = nodes[k]
            item = QtWidgets.QTreeWidgetItem(values)
            item.setData(0, QtCore.Qt.UserRole, str(rows[0]["path"]))
            parent.addChild(item)
            for k in ((animal,), (animal, slice_)):            # counts add up the tree
                for col in (1, 2, 3, 4):
                    n = int(nodes[k].text(col) or 0) + int(values[col] or 0)
                    nodes[k].setText(col, str(n))
        self.tree.expandAll()
        for c in range(len(self.HEAD)):
            self.tree.resizeColumnToContents(c)
        self.nudge.setText(f"⚠ {missing_total} cell(s) have no age, sex, drugs or internal solution recorded: type "
                           "them under Experiment details (or in the cell's Notes.txt)." if missing_total else "")
        entries = p.audit()
        self.audit.setRowCount(len(entries))
        for i, d in enumerate(entries):
            which = Path(d["target"]).name if ("/" in d["target"] or "\\" in d["target"]) else d["target"]
            for j, v in enumerate((d["made_at"].replace("T", " ")[:19], d["kind"], which,
                                   _change(d["kind"], d["before"], d["after"]), d["by"], "yes" if d["undone"] else "")):
                item = QtWidgets.QTableWidgetItem(v)
                if j == 0:
                    item.setData(QtCore.Qt.UserRole, d["id"])
                self.audit.setItem(i, j, item)

    def _open_cell(self, item, _col) -> None:
        path = item.data(0, QtCore.Qt.UserRole)
        if path:
            self.window.select(Path(path))
            self.window.go("check")

    def _undo(self) -> None:
        rows = self.audit.selectionModel().selectedRows()
        if not rows or self.window.project is None:
            return
        entry = self.audit.item(rows[0].row(), 0).data(QtCore.Qt.UserRole)
        self.window.say(self.window.project.revert(int(entry)) + ".")
        self.window.project_changed()

    def _export(self) -> None:
        p = self.window.project
        if p is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export the audit trail",
                                                        str(Path(p.folder) / f"{p.name} - audit trail.csv"),
                                                        "CSV (*.csv)")
        if path:
            self.window.say(f"Wrote {p.write_audit(path).name}.")
