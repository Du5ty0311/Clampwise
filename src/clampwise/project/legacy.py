"""Nothing made under the old name is lost (B17, item 4): Synaptoscope's
session and project files, presets and protocol mappings are read into a
Clampwise project.

A Synaptoscope session is JSON: ``recordings`` (each with a source path, an
``identity`` dictionary and a sweep selection), the Setup ``settings``, and
``statistics`` (groupings and exclusions with their reasons). The identity
carries animal, slice, cell, condition, label, sex, age, genotype, region,
cell type, free notes, the typed stimulus override, the typed recording type,
drug windows (``"baseline 0-10; WIN 10-25"`` in minutes), typed Rs values,
event edits (rejected and added events as JSON), the induction time and
note. All of that lands in the store as identity, metadata, overrides,
decisions and notes. The old Setup settings are kept on the project as a
record (``meta: legacy_settings``), not applied: the analyses' settings are
declared anew, and a silent mapping could change numbers.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..core.recording import DrugEvent, ExperimentMetadata, Identity, Note
from ..analysis import routing
from .presets import Preset

LEGACY_SESSION_FORMATS = ("synaptoscope-session", "clampwise-session")
LEGACY_PRESET_FORMATS = ("synaptoscope-analysis-preset", "clampwise-analysis-preset")


def parse_drug_windows(text: str) -> list[DrugEvent]:
    """``"baseline 0-10; WIN 10-25"`` (minutes) -> drug events in seconds."""
    out = []
    for chunk in re.split(r"[;\n]+", text or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(-?\d+(?:\.\d+)?)", chunk)
        if not m:
            continue
        name = (chunk[:m.start()] + chunk[m.end():]).strip(" :,=") or f"window {len(out) + 1}"
        a, b = float(m.group(1)) * 60, float(m.group(2)) * 60
        out.append(DrugEvent(name, min(a, b), max(a, b), "typed (Synaptoscope drug window)"))
    return out


def parse_rs_typed(text: str) -> tuple[float, float]:
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text or "")]
    if not nums:
        return float("nan"), float("nan")
    return nums[0], nums[1] if len(nums) > 1 else nums[0]


def import_session(project, path: str | Path) -> list[str]:
    """Read a Synaptoscope session or project file into ``project``. Returns
    notes on what was imported and what could not be."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{path.name} could not be read: {exc}"]
    if not isinstance(doc, dict) or doc.get("format") not in LEGACY_SESSION_FORMATS:
        return [f"{path.name} is not a Synaptoscope session"]
    notes = [f"imported {path.name} (saved {doc.get('saved', '?')} by Synaptoscope {doc.get('app_version', '?')})"]
    entries = doc.get("recordings", []) or []
    missing = []
    for entry in entries:
        source = str(entry.get("source", ""))
        if not source or source.startswith("published:"):
            continue
        p = Path(source)
        if not p.is_absolute():
            p = path.parent / p
        if not p.exists():
            missing.append(p.name)
        project.add_recordings([p], hash_files=p.exists())
        ident = entry.get("identity", {}) or {}
        typed = Identity(animal=str(ident.get("animal", "")), slice=str(ident.get("slice", "")),
                         cell=str(ident.get("cell", "")), condition=str(ident.get("condition", "")),
                         label=str(ident.get("label", "")))
        project.set_identity(p, typed)
        meta = ExperimentMetadata(age=str(ident.get("age", "")), sex=str(ident.get("sex", "")),
                                  genotype=str(ident.get("genotype", "")), region=str(ident.get("region", "")),
                                  cell_type=str(ident.get("cell_type", "")),
                                  rs_typed_mohm=parse_rs_typed(str(ident.get("rs_typed", ""))),
                                  drugs=parse_drug_windows(str(ident.get("drug_windows", ""))))
        for key in ("notes", "induction_note", "virus", "injection_site", "projection_target", "hemisphere",
                    "recording_date", "stim_intensity", "stim_intensity_units"):
            v = ident.get(key)
            if v:
                meta.notes.append(Note(0.0, f"{key.replace('_', ' ')}: {v}", "Synaptoscope identity"))
        if ident.get("induction_min") is not None:
            meta.drugs.append(DrugEvent("induction", float(ident["induction_min"]) * 60, float("nan"),
                                        "typed (Synaptoscope induction)"))
        project.set_metadata(p, meta)
        project.set_overrides(p, stimulus_override=str(ident.get("stim_override", "")),
                              type_override=str(ident.get("recording_type", "")), sweeps=str(entry.get("sweeps", "")))
        edits = ident.get("event_edits")
        if edits:
            try:
                data = json.loads(edits)
            except ValueError:
                data = {}
            for s, t in data.get("rejected", []):
                project.add_decision(p, "event_reject", f"sweep {int(s)} at {float(t):.4f} s", "rejected in Synaptoscope")
            for s, t in data.get("added", []):
                project.add_decision(p, "event_add", f"sweep {int(s)} at {float(t):.4f} s", "added in Synaptoscope")
    stats = doc.get("statistics", {}) or {}
    for ex in stats.get("exclusions", []) or []:
        key, reason = str(ex.get("key", "")), str(ex.get("reason", ""))
        matched = [r for r in project.recordings() if r.name == key or r.path.stem == key]
        for r in matched:
            project.add_decision(r.path, "exclude", key, reason or "excluded in Synaptoscope")
        if not matched:
            project.add_note(None, f"Synaptoscope exclusion of '{key}' ({reason}): no recording by that name",
                             source="import")
    if doc.get("settings"):
        project.set_meta("legacy_settings", json.dumps(doc["settings"]))
        notes.append("the Synaptoscope settings are kept on the project as a record (Settings > Imported), not "
                     "applied: Clampwise's analyses declare their own")
    if doc.get("watch_folder"):
        project.set_meta("watch_folder", str(doc["watch_folder"]))
    notes.append(f"{len(entries)} recording(s)" + (f"; {len(missing)} not found on disk: {', '.join(missing[:5])}"
                                                  if missing else ""))
    return notes


def import_preset(path: str | Path) -> Preset:
    """A Synaptoscope analysis preset as a Clampwise preset whose own settings
    are kept under ``legacy``; its mappings carry over."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("format") not in LEGACY_PRESET_FORMATS:
        raise ValueError(f"{Path(path).name} is not a Synaptoscope preset")
    mappings = [routing.Mapping(m.get("pattern", ""), m.get("recording_type", "unknown"),
                                tuple(m.get("analyses", ())), m.get("stimulus_channel", ""), m.get("stimulus_kind", ""))
                for m in d.get("mappings", []) if m.get("pattern")]
    return Preset(d.get("name", Path(path).stem), d.get("description", ""), {}, {}, mappings, 1, False,
                  "imported from Synaptoscope; its settings are kept under 'legacy' and not applied",
                  dict(d.get("settings", {})))


def import_protocol_map(path: str | Path) -> list[routing.Mapping]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for m in (d.get("mappings", []) if isinstance(d, dict) else []):
        if m.get("pattern"):
            out.append(routing.Mapping(m["pattern"], m.get("recording_type", "unknown"), tuple(m.get("analyses", ())),
                                       m.get("stimulus_channel", ""), m.get("stimulus_kind", "")))
    return out


def old_app_folders() -> dict[str, Path]:
    """Where Synaptoscope kept things on this machine (read once, never written)."""
    appdata = Path(os.environ.get("APPDATA") or Path.home()) / "Synaptoscope"
    local = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Synaptoscope"
    return {"presets": appdata / "presets", "protocol_map": appdata / "protocol_map.json",
            "figure_styles": appdata / "styles", "figure_templates": appdata / "figure_templates",
            "stimulus_library": local / "stimulus library"}
