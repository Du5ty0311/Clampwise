"""One project per experiment (B18, items 4-6): a SQLite file in the
experiment's folder holding the recordings, their identity and metadata,
the settings in force, the review decisions (accepted and rejected events,
overrides, exclusions with their reasons, notes), the presets, and a cache
of results keyed by the file's content, the settings and the analysis
version. A lab member who opens the project sees what its owner saw.

Python's built-in ``sqlite3``; no new dependency. The file is
``Clampwise project.clampwise``. Opening a folder that holds a Synaptoscope
project (``Synaptoscope project.synaptoscope``) imports it (``legacy.py``)
and leaves the old file as it was.

Results are stored twice: the recording-level values as JSON (for lists and
tables without unpickling) and the whole ``Result`` pickled (tables,
drawables, flags), so reopening a project is a read, not a re-analysis.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import pickle
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .. import __version__
from ..core.recording import DrugEvent, ExperimentMetadata, Identity, Note, Recording, file_hash
from ..core.identity import Rules
from ..analysis import pipeline, registry, routing
from ..analysis.result import Provenance, RecordingResult, Result

PROJECT_FILE = "Clampwise project.clampwise"
LEGACY_PROJECT_FILE = "Synaptoscope project.synaptoscope"
FORMAT = "clampwise-project"
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, name TEXT NOT NULL, sha256 TEXT, size INTEGER, mtime REAL,
    added TEXT NOT NULL, header TEXT);
CREATE TABLE IF NOT EXISTS identity (
    recording_id INTEGER PRIMARY KEY REFERENCES recordings(id) ON DELETE CASCADE,
    animal TEXT DEFAULT '', slice TEXT DEFAULT '', cell TEXT DEFAULT '', condition TEXT DEFAULT '',
    label TEXT DEFAULT '', sources TEXT DEFAULT '{}');
CREATE TABLE IF NOT EXISTS metadata (
    recording_id INTEGER PRIMARY KEY REFERENCES recordings(id) ON DELETE CASCADE, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS overrides (
    recording_id INTEGER PRIMARY KEY REFERENCES recordings(id) ON DELETE CASCADE,
    stimulus_override TEXT DEFAULT '', type_override TEXT DEFAULT '', stimulus_channel TEXT DEFAULT '',
    sweeps TEXT DEFAULT '', stretch TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY, recording_id INTEGER REFERENCES recordings(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, target TEXT NOT NULL, reason TEXT DEFAULT '', made_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY, recording_id INTEGER REFERENCES recordings(id) ON DELETE CASCADE,
    time_s REAL, text TEXT NOT NULL, source TEXT DEFAULT 'typed', made_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS settings (analysis TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
    PRIMARY KEY (analysis, key));
CREATE TABLE IF NOT EXISTS mappings (
    id INTEGER PRIMARY KEY, pattern TEXT NOT NULL, recording_type TEXT NOT NULL, analyses TEXT DEFAULT '',
    stimulus_channel TEXT DEFAULT '', stimulus_kind TEXT DEFAULT '', position INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS presets (name TEXT PRIMARY KEY, json TEXT NOT NULL, saved TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY, recording_id INTEGER REFERENCES recordings(id) ON DELETE CASCADE,
    sha256 TEXT NOT NULL, analysis TEXT NOT NULL, version INTEGER NOT NULL, settings_hash TEXT NOT NULL,
    values_json TEXT NOT NULL, flags INTEGER NOT NULL, blob BLOB NOT NULL, made_at TEXT NOT NULL, app_version TEXT,
    UNIQUE (sha256, analysis, version, settings_hash));
CREATE TABLE IF NOT EXISTS cell_runs (
    cell TEXT PRIMARY KEY, files TEXT NOT NULL, blob BLOB NOT NULL, made_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY, made_at TEXT NOT NULL, kind TEXT NOT NULL, target TEXT, before TEXT, after TEXT,
    by TEXT DEFAULT 'you', undone INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS runs (
    recording_id INTEGER PRIMARY KEY REFERENCES recordings(id) ON DELETE CASCADE,
    made_at TEXT NOT NULL, recording_type TEXT, type_reason TEXT, stimulus_summary TEXT, notes TEXT,
    provenance TEXT, result_ids TEXT NOT NULL);
"""


def settings_hash(resolved: dict[str, Any], shared: dict[str, Any] | None = None) -> str:
    """One hash of an analysis's resolved settings (and the shared ones), order-free."""
    text = json.dumps({"settings": resolved, "shared": shared or {}}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _metadata_from(d: dict) -> ExperimentMetadata:
    d = dict(d)
    d["drugs"] = [DrugEvent(**x) for x in d.get("drugs", [])]
    d["notes"] = [Note(**x) for x in d.get("notes", [])]
    d["rs_typed_mohm"] = tuple(d.get("rs_typed_mohm", (float("nan"), float("nan"))))
    return ExperimentMetadata(**d)


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


@dataclass
class RecordingRow:
    id: int
    path: Path
    name: str
    sha256: str
    size: int
    added: str


@dataclass
class Overrides:
    stimulus_override: str = ""
    type_override: str = ""
    stimulus_channel: str = ""
    sweeps: str = ""                # "135-200, 455": sweeps to analyze (blank: all)
    stretch: str = ""               # "30-90": seconds of a gap-free file to analyze (blank: all)


@dataclass
class Decision:
    id: int
    kind: str                       # "event_accept", "event_reject", "event_add", "exclude", "sweep_exclude"
    target: str                     # what it applies to: an event time, a sweep number, a cell...
    reason: str
    made_at: str


class Project:
    """An experiment's store. ``Project.open(folder)`` creates or opens it."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.folder = self.path.parent
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self._init_meta()
        self.rules = self._load_rules()
        self._by = "you"                     # who the audit trail says made an edit
        self.organization = None             # the notes and episodes (organize.py), when worked out

    # ---- opening
    @classmethod
    def open(cls, folder: str | Path, *, import_legacy: bool = True) -> "Project":
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        path = folder / PROJECT_FILE
        existed = path.is_file()
        project = cls(path)
        if not existed and import_legacy and (folder / LEGACY_PROJECT_FILE).is_file():
            from . import legacy

            notes = legacy.import_session(project, folder / LEGACY_PROJECT_FILE)
            project.set_meta("imported_from", str(folder / LEGACY_PROJECT_FILE))
            project.set_meta("import_notes", json.dumps(notes))
        return project

    @staticmethod
    def is_project(folder: str | Path) -> bool:
        folder = Path(folder)
        return (folder / PROJECT_FILE).is_file() or (folder / LEGACY_PROJECT_FILE).is_file()

    def close(self) -> None:
        self.db.close()

    def _init_meta(self) -> None:
        if self.meta("format") is None:
            with self.db:
                self.db.executemany("INSERT INTO meta VALUES (?, ?)", [
                    ("format", FORMAT), ("schema", str(SCHEMA_VERSION)), ("created", _now()),
                    ("app_version", __version__), ("name", self.folder.name), ("preset", "")])

    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row["value"]

    def set_meta(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    @property
    def name(self) -> str:
        return self.meta("name") or self.folder.name

    # ---- recordings
    def add_recordings(self, paths: Iterable[str | Path], *, hash_files: bool = True) -> list[RecordingRow]:
        out = []
        with self.db:
            for p in paths:
                p = Path(p).resolve()
                existing = self.db.execute("SELECT id FROM recordings WHERE path = ?", (str(p),)).fetchone()
                if existing is None:
                    st = p.stat() if p.exists() else None
                    self.db.execute(
                        "INSERT INTO recordings (path, name, sha256, size, mtime, added) VALUES (?, ?, ?, ?, ?, ?)",
                        (str(p), p.name, file_hash(p) if hash_files and p.exists() else "",
                         st.st_size if st else 0, st.st_mtime if st else 0.0, _now()))
                    rid = self.db.execute("SELECT id FROM recordings WHERE path = ?", (str(p),)).fetchone()["id"]
                    parsed = self.rules.parse(p, stop_at=(self.folder,))
                    self.db.execute("INSERT OR IGNORE INTO identity (recording_id, animal, slice, cell, condition, "
                                    "label, sources) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (rid, parsed.animal, parsed.slice, parsed.cell, parsed.condition, "",
                                     json.dumps(parsed.sources)))
                out.append(self.recording(p))
        return out

    def recordings(self) -> list[RecordingRow]:
        rows = self.db.execute("SELECT * FROM recordings ORDER BY path").fetchall()
        return [RecordingRow(r["id"], Path(r["path"]), r["name"], r["sha256"] or "", r["size"] or 0, r["added"])
                for r in rows]

    def recording(self, path: str | Path) -> RecordingRow:
        r = self.db.execute("SELECT * FROM recordings WHERE path = ?", (str(Path(path).resolve()),)).fetchone()
        if r is None:
            raise KeyError(f"{path} is not in the project")
        return RecordingRow(r["id"], Path(r["path"]), r["name"], r["sha256"] or "", r["size"] or 0, r["added"])

    def _rid(self, path: str | Path) -> int:
        return self.recording(path).id

    def remove_recording(self, path: str | Path) -> None:
        with self.db:
            self.db.execute("DELETE FROM recordings WHERE id = ?", (self._rid(path),))

    def _changed_on_disk(self, path: str | Path) -> bool:
        """Whether the file's size or modification time differ from when it was hashed
        (a re-recorded file must not be read from the old results)."""
        p = Path(path).resolve()
        r = self.db.execute("SELECT size, mtime FROM recordings WHERE path = ?", (str(p),)).fetchone()
        try:
            st = p.stat()
        except OSError:
            return False
        return r is None or int(r["size"] or 0) != st.st_size or abs(float(r["mtime"] or 0.0) - st.st_mtime) > 1e-6

    def refresh_hash(self, path: str | Path) -> str:
        """Re-hash a file that may have changed (a re-recorded file is picked up,
        and its cached results no longer match)."""
        p = Path(path).resolve()
        h = file_hash(p)
        with self.db:
            self.db.execute("UPDATE recordings SET sha256 = ?, size = ?, mtime = ? WHERE path = ?",
                            (h, p.stat().st_size if p.exists() else 0, p.stat().st_mtime if p.exists() else 0.0, str(p)))
        return h

    # ---- identity rules: kept in the project, seen and edited by the user (B1)
    def _load_rules(self) -> Rules:
        from ..core.identity import Rule

        text = self.meta("identity_rules")
        if not text:
            return Rules()
        try:
            return Rules([Rule(r["name"], r["pattern"], r.get("applies_to", "folders")) for r in json.loads(text)])
        except (ValueError, KeyError, TypeError):
            return Rules()

    def set_rules(self, rules: list) -> int:
        """Keep ``rules`` and read every recording's identity again with them;
        what was typed stays. Returns how many identities changed."""
        import re

        for r in rules:
            re.compile(r.pattern)                       # a bad pattern is refused before anything changes
        before = json.loads(self.meta("identity_rules") or "[]")
        self.set_meta("identity_rules", json.dumps([{"name": r.name, "pattern": r.pattern,
                                                     "applies_to": r.applies_to} for r in rules]))
        self._audit("naming rules", "project", before, json.loads(self.meta("identity_rules")))
        self.rules = Rules(list(rules))
        changed = 0
        for row in self.recordings():
            old = self.identity(row.path)
            parsed = self.rules.parse(row.path, stop_at=(self.folder,))
            new = Identity(sources=dict(parsed.sources))
            for k in ("animal", "slice", "cell", "condition"):
                if old.sources.get(k) == "typed":
                    setattr(new, k, getattr(old, k))
                    new.sources[k] = "typed"
                else:
                    setattr(new, k, getattr(parsed, k))
            new.label = old.label
            if any(getattr(new, k) != getattr(old, k) for k in ("animal", "slice", "cell", "condition")):
                changed += 1
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO identity (recording_id, animal, slice, cell, condition, label, "
                                "sources) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (row.id, new.animal, new.slice, new.cell, new.condition, new.label,
                                 json.dumps(new.sources)))
        return changed

    # ---- identity and metadata
    def identity(self, path: str | Path) -> Identity:
        r = self.db.execute("SELECT * FROM identity WHERE recording_id = ?", (self._rid(path),)).fetchone()
        if r is None:
            return Identity()
        return Identity(r["animal"], r["slice"], r["cell"], r["condition"], r["label"], json.loads(r["sources"] or "{}"))

    def set_identity(self, path: str | Path, typed: Identity) -> Identity:
        """What is typed wins; blanks keep the parsed value; each field says its source."""
        before = self.identity(path)
        merged = Rules.merge(before, typed)
        self._write_identity(path, merged)
        self._audit("identity", path, asdict(before), asdict(merged))
        return merged

    def _write_identity(self, path: str | Path, ident: Identity) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO identity (recording_id, animal, slice, cell, condition, label, "
                            "sources) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (self._rid(path), ident.animal, ident.slice, ident.cell, ident.condition, ident.label,
                             json.dumps(ident.sources)))

    def metadata(self, path: str | Path) -> ExperimentMetadata:
        r = self.db.execute("SELECT json FROM metadata WHERE recording_id = ?", (self._rid(path),)).fetchone()
        if r is None:
            return ExperimentMetadata()
        return _metadata_from(json.loads(r["json"]))

    def set_metadata(self, path: str | Path, meta: ExperimentMetadata) -> None:
        d = asdict(meta)
        self._audit("metadata", path, asdict(self.metadata(path)), d)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO metadata (recording_id, json) VALUES (?, ?)",
                            (self._rid(path), json.dumps(d, default=str)))

    # ---- overrides and decisions
    def overrides(self, path: str | Path) -> Overrides:
        r = self.db.execute("SELECT * FROM overrides WHERE recording_id = ?", (self._rid(path),)).fetchone()
        if r is None:
            return Overrides()
        return Overrides(r["stimulus_override"], r["type_override"], r["stimulus_channel"], r["sweeps"], r["stretch"])

    def set_overrides(self, path: str | Path, **fields) -> Overrides:
        current = self.overrides(path)
        before = asdict(current)
        for k, v in fields.items():
            if not hasattr(current, k):
                raise ValueError(f"no override called {k}")
            setattr(current, k, v or "")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO overrides VALUES (?, ?, ?, ?, ?, ?)",
                            (self._rid(path), current.stimulus_override, current.type_override,
                             current.stimulus_channel, current.sweeps, current.stretch))
        self._audit("overrides", path, before, asdict(current))
        return current

    def add_decision(self, path: str | Path, kind: str, target: str, reason: str = "") -> Decision:
        with self.db:
            cur = self.db.execute("INSERT INTO decisions (recording_id, kind, target, reason, made_at) VALUES "
                                  "(?, ?, ?, ?, ?)", (self._rid(path), kind, target, reason, _now()))
        self._audit("decision added", path, None, {"kind": kind, "target": target, "reason": reason})
        return Decision(cur.lastrowid, kind, target, reason, _now())

    def decisions(self, path: str | Path, kind: str | None = None) -> list[Decision]:
        q = "SELECT * FROM decisions WHERE recording_id = ?" + (" AND kind = ?" if kind else "") + " ORDER BY id"
        args = (self._rid(path), kind) if kind else (self._rid(path),)
        return [Decision(r["id"], r["kind"], r["target"], r["reason"], r["made_at"])
                for r in self.db.execute(q, args).fetchall()]

    def remove_decision(self, decision_id: int) -> None:
        r = self.db.execute("SELECT d.*, r.path FROM decisions d LEFT JOIN recordings r ON r.id = d.recording_id "
                            "WHERE d.id = ?", (decision_id,)).fetchone()
        with self.db:
            self.db.execute("DELETE FROM decisions WHERE id = ?", (decision_id,))
        if r is not None:
            self._audit("decision removed", r["path"], {"kind": r["kind"], "target": r["target"],
                                                        "reason": r["reason"]}, None)

    def add_note(self, path: str | Path | None, text: str, time_s: float | None = None, source: str = "typed") -> int:
        rid = self._rid(path) if path is not None else None
        with self.db:
            cur = self.db.execute("INSERT INTO notes (recording_id, time_s, text, source, made_at) VALUES "
                                  "(?, ?, ?, ?, ?)", (rid, time_s, text, source, _now()))
        return cur.lastrowid

    def notes(self, path: str | Path | None = None) -> list[dict]:
        if path is None:
            rows = self.db.execute("SELECT * FROM notes ORDER BY id").fetchall()
        else:
            rows = self.db.execute("SELECT * FROM notes WHERE recording_id = ? ORDER BY id", (self._rid(path),)).fetchall()
        return [dict(r) for r in rows]

    # ---- settings, mappings, presets
    def settings(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in self.db.execute("SELECT * FROM settings").fetchall():
            out.setdefault(r["analysis"], {})[r["key"]] = json.loads(r["value"])
        return out

    def set_setting(self, analysis: str, key: str, value: Any) -> None:
        a = registry.load().get(analysis)
        if a is not None and analysis != "shared":
            a.resolve({key: value})                 # validates
        before = self.settings().get(analysis, {}).get(key)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?, ?)", (analysis, key, json.dumps(value)))
        self._audit("setting", f"{analysis}.{key}", before, value)

    def reset_setting(self, analysis: str, key: str | None = None) -> None:
        before = self.settings().get(analysis, {})
        self._audit("setting", f"{analysis}.{key}" if key else analysis,
                    before.get(key) if key else before, None)
        with self.db:
            if key is None:
                self.db.execute("DELETE FROM settings WHERE analysis = ?", (analysis,))
            else:
                self.db.execute("DELETE FROM settings WHERE analysis = ? AND key = ?", (analysis, key))

    def mappings(self) -> list[routing.Mapping]:
        rows = self.db.execute("SELECT * FROM mappings ORDER BY position").fetchall()
        return [routing.Mapping(r["pattern"], r["recording_type"], tuple(json.loads(r["analyses"] or "[]")),
                                r["stimulus_channel"], r["stimulus_kind"]) for r in rows]

    def set_mappings(self, mappings: Iterable[routing.Mapping]) -> None:
        with self.db:
            self.db.execute("DELETE FROM mappings")
            for i, m in enumerate(mappings):
                self.db.execute("INSERT INTO mappings (pattern, recording_type, analyses, stimulus_channel, "
                                "stimulus_kind, position) VALUES (?, ?, ?, ?, ?, ?)",
                                (m.pattern, m.recording_type, json.dumps(list(m.analyses)), m.stimulus_channel,
                                 m.stimulus_kind, i))

    def save_preset(self, preset) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO presets VALUES (?, ?, ?)",
                            (preset.name, json.dumps(preset.to_dict()), _now()))

    def presets(self) -> dict:
        from .presets import Preset, BUILTIN

        out = dict(BUILTIN)
        for r in self.db.execute("SELECT * FROM presets ORDER BY name").fetchall():
            p = Preset.from_dict(json.loads(r["json"]))
            out[p.name] = p
        return out

    def delete_preset(self, name: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM presets WHERE name = ?", (name,))

    def apply_preset(self, name: str) -> None:
        """The preset's settings and mappings become the project's; its name is
        recorded and printed in every export."""
        preset = self.presets()[name]
        self._audit("preset applied", name, {"settings": self.settings(), "preset": self.meta("preset") or ""},
                    {"preset": name})
        with self.db:
            self.db.execute("DELETE FROM settings")
            for analysis, values in preset.settings.items():
                for k, v in values.items():
                    self.db.execute("INSERT INTO settings VALUES (?, ?, ?)", (analysis, k, json.dumps(v)))
            for k, v in preset.shared.items():
                self.db.execute("INSERT INTO settings VALUES (?, ?, ?)", ("shared", k, json.dumps(v)))
        if preset.mappings:
            self.set_mappings(preset.mappings)
        self.set_meta("preset", name)

    def _rig_preset(self, s: pipeline.Settings, path) -> None:
        """The file's rig's preset (rigs.py), over the project's settings; a file's own
        overrides still win. Its name goes in the provenance as the preset. The rig's
        light inputs go in the shared settings (the stimulus reads them; the cache keys on them)."""
        from . import rigs

        try:
            m = rigs.match_file(path)
        except Exception:
            return
        if m.rig is not None and m.rig.light_inputs:
            s.shared = {**s.shared, "light_inputs": dict(m.rig.light_inputs)}
        preset = self.presets().get(m.rig.preset) if m.rig is not None and m.rig.preset else None
        if preset is None:
            return
        s.by_analysis = {k: dict(v) for k, v in s.by_analysis.items()}
        for analysis, values in preset.settings.items():
            s.by_analysis.setdefault(analysis, {}).update(values)
        s.shared = {**s.shared, **preset.shared}
        s.preset = f"{preset.name} (rig {m.rig.name})"

    def pipeline_settings(self, path: str | Path | None = None) -> pipeline.Settings:
        """The project's settings as the pipeline takes them, with the
        recording's own overrides when ``path`` is given."""
        all_settings = self.settings()
        shared = all_settings.pop("shared", {})
        s = pipeline.Settings(by_analysis=all_settings, shared=shared, preset=self.meta("preset") or "",
                              mappings=self.mappings())
        disabled = shared.get("disabled_analyses")
        if disabled:
            s.disabled = set(disabled)
        if path is not None and self.features().get("rig_presets"):
            self._rig_preset(s, path)
        if path is not None:
            o = self.overrides(path)
            s.stimulus_override, s.type_override, s.stimulus_channel = o.stimulus_override, o.type_override, o.stimulus_channel
            s.shared["sweeps"], s.shared["stretch"] = o.sweeps, o.stretch
            # a typed stimulus, type or stimulus channel changes every result: part of the cache key
            for k in ("stimulus_override", "type_override", "stimulus_channel"):
                if getattr(o, k):
                    s.shared[k] = getattr(o, k)
            # typed experiment details (Rs typed from the Membrane Test, temperature...) change results:
            # they are part of what the cache keys on
            meta = self.db.execute("SELECT json FROM metadata WHERE recording_id = ?", (self._rid(path),)).fetchone()
            if meta is not None:
                s.shared["metadata"] = hashlib.sha256(meta["json"].encode()).hexdigest()[:16]
            edits = self.event_edits(path)
            if edits:
                s.by_analysis = {k: dict(v) for k, v in s.by_analysis.items()}
                s.by_analysis.setdefault("events", {})["edits"] = edits
            moved = self.moved_points(path)
            if moved:
                s.by_analysis = {k: dict(v) for k, v in s.by_analysis.items()}
                s.by_analysis.setdefault("intrinsic", {})["moved"] = moved
        return s

    # ---- points moved by hand on Check: kept as decisions, re-applied at every analysis
    POINT_MOVED = "point moved"

    def moved_points(self, path: str | Path) -> str:
        """The recording's moved points, in the intrinsic analysis's saved form."""
        from ..analysis.iv_curves import format_moved, parse_moved

        return format_moved(parse_moved(";".join(d.target for d in self.decisions(path, self.POINT_MOVED))))

    def move_point(self, path: str | Path, sweep: int, what: str, time_s: float | None, reason: str = "") -> None:
        """Read sweep ``sweep``'s ``what`` (baseline, peak, steady, rebound) at
        ``time_s``; ``None`` puts it back where the analysis finds it."""
        for d in self.decisions(path, self.POINT_MOVED):
            if d.target.split(":")[:2] == [str(int(sweep)), what]:
                self.remove_decision(d.id)
        if time_s is not None:
            self.add_decision(path, self.POINT_MOVED, f"{int(sweep)}:{what}:{float(time_s):.5f}", reason)

    def reset_points(self, path: str | Path) -> int:
        found = self.decisions(path, self.POINT_MOVED)
        for d in found:
            self.remove_decision(d.id)
        return len(found)

    # ---- optional features (features.py): on or off per project
    def features(self) -> dict[str, bool]:
        from .features import FEATURES

        saved = json.loads(self.meta("features") or "{}")
        return {f.key: bool(saved.get(f.key, f.default)) for f in FEATURES}

    def set_feature(self, key: str, on: bool) -> None:
        before = self.features().get(key)
        self.set_meta("features", json.dumps({**self.features(), key: bool(on)}))
        self._audit("feature", key, before, bool(on))

    # ---- the audit trail: every edit and decision, who made it, undoable, exportable
    @contextmanager
    def acting(self, by: str):
        """Edits inside the block are said to be by ``by`` ("Notes.txt", "rig profile")."""
        before, self._by = self._by, by
        try:
            yield
        finally:
            self._by = before

    def _audit(self, kind: str, target, before, after) -> None:
        with self.db:
            self.db.execute("INSERT INTO audit (made_at, kind, target, before, after, by) VALUES (?, ?, ?, ?, ?, ?)",
                            (_now(), kind, str(target), json.dumps(before, default=str),
                             json.dumps(after, default=str), self._by))

    def audit(self) -> list[dict]:
        """Every entry, newest first: when, what, which, before, after, by whom, undone."""
        out = []
        for r in self.db.execute("SELECT * FROM audit ORDER BY id DESC").fetchall():
            d = dict(r)
            d["before"], d["after"] = json.loads(d["before"] or "null"), json.loads(d["after"] or "null")
            out.append(d)
        return out

    def revert(self, entry_id: int) -> str:
        """Undo one audit entry (put back what it changed); the undoing is audited too."""
        r = self.db.execute("SELECT * FROM audit WHERE id = ?", (entry_id,)).fetchone()
        if r is None or r["undone"]:
            return "nothing to undo"
        kind, target = r["kind"], r["target"]
        before, after = json.loads(r["before"] or "null"), json.loads(r["after"] or "null")
        with self.acting(f"undo of #{entry_id}"):
            if kind == "identity":
                self._write_identity(target, Identity(**before))
                self._audit("identity", target, after, before)
            elif kind == "metadata":
                self.set_metadata(target, _metadata_from(before))
            elif kind == "overrides":
                self.set_overrides(target, **before)
            elif kind == "setting":
                analysis, _, key = target.partition(".")
                if key and before is not None:
                    self.set_setting(analysis, key, before)
                elif key:
                    self.reset_setting(analysis, key)
                else:
                    for k, v in (before or {}).items():
                        self.set_setting(analysis, k, v)
            elif kind == "feature":
                self.set_feature(target, bool(before) if before is not None else True)
            elif kind == "decision added":
                for d in self.decisions(target, after["kind"]):
                    if d.target == after["target"]:
                        self.remove_decision(d.id)
                        break
            elif kind == "decision removed":
                self.add_decision(target, before["kind"], before["target"], before.get("reason", ""))
            elif kind == "preset applied":
                with self.db:
                    self.db.execute("DELETE FROM settings")
                    for analysis, values in before["settings"].items():
                        for k, v in values.items():
                            self.db.execute("INSERT INTO settings VALUES (?, ?, ?)", (analysis, k, json.dumps(v)))
                self.set_meta("preset", before.get("preset", ""))
                self._audit("preset applied", before.get("preset", ""), after, before)
            elif kind == "naming rules":
                from ..core.identity import Rule

                self.set_rules([Rule(x["name"], x["pattern"], x.get("applies_to", "folders")) for x in before])
            else:
                return f"a {kind} cannot be undone here"
        with self.db:
            self.db.execute("UPDATE audit SET undone = 1 WHERE id = ?", (entry_id,))
        return f"undid #{entry_id}: {kind} of {Path(target).name if kind not in ('setting', 'feature') else target}"

    def write_audit(self, path: str | Path) -> Path:
        """The audit trail as CSV (UTF-8, opens in Excel)."""
        import csv

        path = Path(path)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["#", "When", "What", "Which", "Before", "After", "By", "Undone"])
            for d in self.audit():
                w.writerow([d["id"], d["made_at"], d["kind"], d["target"], json.dumps(d["before"]),
                            json.dumps(d["after"]), d["by"], "yes" if d["undone"] else ""])
        return path

    # ---- the file review on Check: good or bad, by hand, with a reason; evidence, never an edit
    FILE_REVIEW = "file review"

    def file_review(self, path: str | Path) -> tuple[str, str]:
        """("good" or "bad", the reason), or ("", "") when not reviewed."""
        found = self.decisions(path, self.FILE_REVIEW)
        return (found[-1].target, found[-1].reason) if found else ("", "")

    def set_file_review(self, path: str | Path, verdict: str, reason: str = "") -> None:
        """Mark the recording "good" or "bad"; "" clears the mark."""
        if verdict not in ("", "good", "bad"):
            raise ValueError(f"a review is good or bad, not {verdict!r}")
        for d in self.decisions(path, self.FILE_REVIEW):
            self.remove_decision(d.id)
        if verdict:
            self.add_decision(path, self.FILE_REVIEW, verdict, reason)

    # ---- the event review (B4): kept as decisions, re-applied at every analysis
    EVENT_REJECTED, EVENT_ADDED = "event rejected", "event added"

    def event_edits(self, path: str | Path) -> str:
        """The recording's reviewed events, in the events analysis's saved form."""
        from ..analysis.events import format_edits

        def parse(d):
            sweep, _, t = d.target.partition(":")
            return int(sweep), float(t)

        rejected = [parse(d) for d in self.decisions(path, self.EVENT_REJECTED)]
        added = [parse(d) for d in self.decisions(path, self.EVENT_ADDED)]
        return format_edits(rejected, added)

    def review_event(self, path: str | Path, sweep: int, time_s: float, action: str, reason: str = "") -> None:
        """``action``: "reject" or "restore" a detected event, "add" one by
        hand or "remove" an added one. The next analysis applies it."""
        target = f"{int(sweep)}:{float(time_s):.5f}"
        kind = {"reject": self.EVENT_REJECTED, "restore": self.EVENT_REJECTED,
                "add": self.EVENT_ADDED, "remove": self.EVENT_ADDED}[action]
        if action in ("reject", "add"):
            self.add_decision(path, kind, target, reason)
            return
        for d in self.decisions(path, kind):
            s, _, t = d.target.partition(":")
            if int(s) == int(sweep) and abs(float(t) - float(time_s)) <= 0.001:
                self.remove_decision(d.id)

    # ---- the result cache
    def cache_lookup(self, sha256: str, analysis: registry.Analysis, resolved: dict, shared: dict) -> Result | None:
        if not sha256:
            return None
        r = self.db.execute("SELECT blob FROM results WHERE sha256 = ? AND analysis = ? AND version = ? AND "
                            "settings_hash = ?", (sha256, analysis.key, analysis.version,
                                                  settings_hash(resolved, shared))).fetchone()
        if r is None:
            return None
        try:
            result = pickle.loads(r["blob"])
        except Exception:
            return None
        result.from_cache = True
        return result

    def cache_store(self, rid: int, sha256: str, result: Result, shared: dict) -> int:
        if not sha256 or result.error:
            return 0
        with self.db:
            cur = self.db.execute(
                "INSERT OR REPLACE INTO results (recording_id, sha256, analysis, version, settings_hash, values_json, "
                "flags, blob, made_at, app_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rid, sha256, result.key, result.version, settings_hash(result.settings, shared),
                 json.dumps(result.values, default=str), len(result.qc_flags), pickle.dumps(result, protocol=5),
                 _now(), __version__))
        return cur.lastrowid

    def clear_cache(self) -> int:
        with self.db:
            n = self.db.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            self.db.execute("DELETE FROM results")
            self.db.execute("DELETE FROM runs")
        return int(n)

    def cache_stats(self) -> dict:
        n, size = self.db.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(blob)), 0) FROM results").fetchone()
        return {"results": int(n), "bytes": int(size)}

    # ---- analyzing through the store
    def analyze(self, path: str | Path, recording: Recording | None = None, *, force: bool = False) -> RecordingResult:
        """Analyze one of the project's recordings, taking each analysis from the
        cache when its file, settings and version match, and keep the run."""
        from ..io import load

        row = self.recording(path)
        rec = recording if recording is not None else load(row.path)
        rec.identity = self.identity(path)
        evidence = None
        if self.features().get("rig_records", True):            # what the rig's programs recorded of it
            from ..rig import evidence as rig_evidence

            evidence = rig_evidence.for_recording(rec, path=row.path)
            evidence = None if evidence.empty else evidence
        rec.metadata = self._merged_metadata(path, rec, evidence)
        sha = row.sha256 if row.sha256 and not self._changed_on_disk(path) else self.refresh_hash(path)
        rec._hash = sha
        settings = self.pipeline_settings(path)
        if evidence is not None:
            settings.rig = evidence
            settings.shared["rig"] = evidence.digest()             # a changed record re-runs what it changes
        cache = None if force else self
        result = pipeline.analyze(rec, settings, cache=cache)
        ids = []
        for key, res in result.results.items():
            shared = result.cache_shared[key]          # what the pipeline keyed this result on
            if getattr(res, "from_cache", False):
                r = self.db.execute("SELECT id FROM results WHERE sha256 = ? AND analysis = ? AND version = ? AND "
                                    "settings_hash = ?", (sha, key, res.version,
                                                          settings_hash(res.settings, shared))).fetchone()
                if r is not None:
                    ids.append(r["id"])
            else:
                ids.append(self.cache_store(row.id, sha, res, shared))
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (row.id, _now(), result.recording_type, result.type_reason, result.stimulus_summary,
                             json.dumps(result.notes), json.dumps(result.provenance.to_dict()), json.dumps(ids)))
        return result

    def last_results(self, analysis: str) -> list[tuple[str, Result]]:
        """(recording name, its last result of ``analysis``) for every recording
        analyzed, read from the store without opening the recordings."""
        out = []
        for row in self.recordings():
            run = self.db.execute("SELECT result_ids FROM runs WHERE recording_id = ?", (row.id,)).fetchone()
            if run is None:
                continue
            for rid in json.loads(run["result_ids"]):
                r = self.db.execute("SELECT analysis, blob FROM results WHERE id = ?", (rid,)).fetchone()
                if r is not None and r["analysis"] == analysis:
                    try:
                        out.append((row.name, pickle.loads(r["blob"])))
                    except Exception:
                        pass
        return out

    # ---- cells: a cell's recordings analyzed together (B5's DSI from events, ...)
    def cells(self) -> dict[str, list[RecordingRow]]:
        """The project's recordings grouped by cell (animal / slice / cell)."""
        from ..analysis.cells import cell_key

        out: dict[str, list[RecordingRow]] = {}
        for row in self.recordings():
            out.setdefault(cell_key(self.identity(row.path), row.path), []).append(row)
        return out

    def analyze_cells(self, only: Iterable[str] | None = None) -> dict:
        """Run the cell-level analyses on every cell (or ``only`` those keys) from
        its recordings' last results, and keep them. Nothing is re-analyzed."""
        from ..analysis.cells import Cell, Member, analyze_cell

        settings = self.settings()
        disabled = settings.get("shared", {}).get("disabled_analyses", [])
        done = {}
        for key, rows in self.cells().items():
            if only is not None and key not in set(only):
                continue
            members = []
            for row in rows:
                try:
                    run = self.last_run(row.path)
                except Exception:
                    run = None
                if run is not None:
                    members.append(Member(row.name, run.recording, run.results, run.recording_type))
            if not members:
                continue
            result = analyze_cell(Cell.build(key, members), settings, disabled)
            if not result.results:
                with self.db:
                    self.db.execute("DELETE FROM cell_runs WHERE cell = ?", (key,))
                continue
            for m in members:                     # the samples are not kept with the result
                m.recording = None
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO cell_runs VALUES (?, ?, ?, ?)",
                                (key, json.dumps(result.files), pickle.dumps(result, protocol=5), _now()))
            done[key] = result
        return done

    def cell_result(self, path: str | Path):
        """The stored cell-level results of the cell ``path`` belongs to, or None."""
        from ..analysis.cells import cell_key

        key = cell_key(self.identity(path), self.recording(path).path)
        r = self.db.execute("SELECT blob FROM cell_runs WHERE cell = ?", (key,)).fetchone()
        if r is None:
            return None
        try:
            return pickle.loads(r["blob"])
        except Exception:
            return None

    def make_event_template(self):
        """One template for the experiment from its recordings' averaged events
        set for every recording when it can
        be made. Returns the ExperimentTemplate, with a note when it could not."""
        from ..analysis.events import experiment_template

        tpl = experiment_template([r for _, r in self.last_results("events")])
        if tpl.ok:
            for key, value in tpl.settings().items():
                self.set_setting("events", key, value)
        return tpl

    def _merged_metadata(self, path, rec: Recording, evidence=None) -> ExperimentMetadata:
        typed = self.metadata(path)
        file_notes = list(rec.metadata.notes)
        merged = ExperimentMetadata(**{k: v for k, v in asdict(typed).items() if k not in ("drugs", "notes")})
        merged.drugs = list(typed.drugs)
        merged.notes = file_notes + list(typed.notes)
        merged.rs_typed_mohm = typed.rs_typed_mohm
        typed_rs = merged.rs_typed_mohm
        if evidence is not None and not all(math.isfinite(x) for x in typed_rs):
            # what was typed stays; the Lab Book's Membrane Tests fill only what was not
            book = evidence.rs()
            if any(not math.isfinite(t) and math.isfinite(b) for t, b in zip(typed_rs, book)):
                start, end = (t if math.isfinite(t) else b for t, b in zip(typed_rs, book))
                if not math.isfinite(start):
                    start, end = end, float("nan")              # one value: the one Rs there is
                merged.rs_typed_mohm = (start, end)
                merged.rs_source = "the acquisition's Lab Book" + (
                    ", where nothing was typed" if any(math.isfinite(x) for x in typed_rs) else "")
        return merged

    def last_run(self, path: str | Path, recording: Recording | None = None, *,
                 read_file: bool = True) -> RecordingResult | None:
        """The stored result of the last analysis of ``path``, without re-analyzing
        (the Recording is loaded on demand unless given; not at all, and None in
        the result, without ``read_file``: an export needs only the numbers)."""
        from ..io import load

        row = self.recording(path)
        run = self.db.execute("SELECT * FROM runs WHERE recording_id = ?", (row.id,)).fetchone()
        if run is None:
            return None
        results: dict[str, Result] = {}
        for rid in json.loads(run["result_ids"]):
            r = self.db.execute("SELECT blob FROM results WHERE id = ?", (rid,)).fetchone()
            if r is not None:
                res = pickle.loads(r["blob"])
                res.from_cache = True
                results[res.key] = res
        prov = json.loads(run["provenance"]) if run["provenance"] else None
        rec = recording if recording is not None else load(row.path) if read_file else None
        if rec is not None:
            rec.identity = self.identity(path)
        out = RecordingResult(rec, run["recording_type"], run["type_reason"], run["stimulus_summary"], results,
                              Provenance(prov["file"], prov["sha256"], prov["app_version"], prov["analyzed_at"],
                                         prov["analyses"], prov["settings"], prov.get("preset", ""),
                                         prov.get("python", ""), prov.get("libraries", {})) if prov else None,
                              json.loads(run["notes"] or "[]"))
        return out

    def summaries(self) -> list[dict]:
        """One row per recording for the Files list: identity, type, quality
        badge and when it was last analyzed, from the store alone."""
        out = []
        for row in self.recordings():
            ident = self.identity(row.path)
            run = self.db.execute("SELECT * FROM runs WHERE recording_id = ?", (row.id,)).fetchone()
            flags = 0
            has_quality = False
            if run is not None:
                for rid in json.loads(run["result_ids"]):
                    r = self.db.execute("SELECT analysis, flags FROM results WHERE id = ?", (rid,)).fetchone()
                    if r is not None and r["analysis"] == "quality":
                        has_quality = True
                        flags += r["flags"]
            verdict, reason = self.file_review(row.path)
            out.append({"review": verdict, "review_reason": reason, "label": ident.label,
                        "label_source": ident.sources.get("label", ""), "path": row.path, "name": row.name, "animal": ident.animal, "slice": ident.slice,
                        "cell": ident.cell, "condition": ident.condition,
                        "recording_type": run["recording_type"] if run else "",
                        "analyzed_at": run["made_at"] if run else "",
                        "quality": ("flagged" if flags else "passed") if has_quality else "not checked"})
        return out
