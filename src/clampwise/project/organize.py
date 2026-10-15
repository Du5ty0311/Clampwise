"""A project's organization: each cell folder's Notes.txt and each cell's
episodes, worked out from the files' headers (off the window's thread: nothing
here writes), then the notes applied to the store (``apply_notes``: labels and
metadata where nothing was typed, recorded in the audit trail as by Notes.txt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import notes as notes_mod
from . import sessions


@dataclass
class Organization:
    notes: dict[Path, notes_mod.Notes] = field(default_factory=dict)        # folder -> its notes
    episodes: dict[str, list[sessions.Episode]] = field(default_factory=dict)   # cell -> its episodes
    of: dict[Path, list[tuple[str, sessions.Episode, str]]] = field(default_factory=dict)  # file -> (cell, ep, role)
    rigs: dict[Path, object] = field(default_factory=dict)          # file -> rigs.Match

    def describe(self, path) -> str:
        """'03 step' or '02 after, 03 before': sorts episodes together."""
        return ", ".join(f"{ep.index:02d} {role}" for _, ep, role in self.of.get(Path(path), []))

    def note_flags(self, path) -> list[str]:
        """What the notes say about this file that is in doubt or disagrees with it."""
        n = self.notes.get(Path(path).parent)
        if n is None:
            return []
        mine = [e.describe() for e in n.entry_for(Path(path))]
        return [f for f in n.flags if any(f.startswith(m) for m in mine)]

    def episode_doubts(self, path) -> list[str]:
        return list(dict.fromkeys(d for _, ep, _ in self.of.get(Path(path), []) for d in ep.doubts))

    def doubts(self, path) -> list[str]:
        return list(dict.fromkeys(self.note_flags(path) + self.episode_doubts(path)))

    def count(self) -> int:
        return sum(len(v) for v in self.episodes.values())


def work_out(rows: list[tuple[Path, str, str]], load, use_notes: bool = True, group: bool = True,
             find_rigs: bool = False) -> Organization:
    """``rows``: (file, cell key, condition) for every recording. Reads Notes.txt in
    each folder that has one, and groups each cell's files into episodes."""
    org = Organization()
    by_folder: dict[Path, list[Path]] = {}
    by_cell: dict[str, list[Path]] = {}
    conditions = {}
    for path, cell, condition in rows:
        path = Path(path)
        by_folder.setdefault(path.parent, []).append(path)
        by_cell.setdefault(cell or str(path.parent), []).append(path)
        conditions[path] = condition
    if use_notes:
        for folder, files in by_folder.items():
            n = notes_mod.read(folder, files)
            if n is not None:
                notes_mod.check(n, load)
                org.notes[folder] = n
    if find_rigs:
        from . import rigs

        profiles = rigs.Rigs()
        for path, _cell, _cond in rows:
            try:
                org.rigs[Path(path)] = rigs.match_file(path, profiles)
            except Exception:
                continue
    if group:
        for cell, files in by_cell.items():
            folders = {p.parent for p in files}
            n = org.notes.get(next(iter(folders))) if len(folders) == 1 else None
            eps = sessions.group(sorted(files), load, n, conditions)
            org.episodes[cell] = eps
            for ep in eps:
                for p, role in ep.members:
                    org.of.setdefault(Path(p), []).append((cell, ep, role))
    return org


def apply_notes(project, org: Organization) -> int:
    """Each named file's label, and the cell's age, sex, region, cell type and drugs,
    where nothing was typed. Returns how many recordings changed."""
    from ..core.recording import DrugEvent, Identity

    known = {Path(r.path) for r in project.recordings()}
    changed = 0
    with project.acting("Notes.txt"):
        for folder, n in org.notes.items():
            for path in (p for p in known if p.parent == folder):
                ident = project.identity(path)
                entries = n.entry_for(path)
                touched = False
                if entries and (not ident.label or ident.sources.get("label", "").startswith("Notes.txt")):
                    label = "; ".join(e.label for e in entries)
                    if label != ident.label:
                        new = Identity(**{**ident.__dict__, "label": label,
                                          "sources": {**ident.sources,
                                                      "label": f"Notes.txt, line {entries[0].line}"}})
                        project._write_identity(path, new)
                        project._audit("identity", path, ident.__dict__, new.__dict__)
                        touched = True
                meta = project.metadata(path)
                before = (meta.age, meta.sex, meta.region, meta.cell_type, len(meta.drugs))
                meta.age = meta.age or n.age
                meta.sex = meta.sex or n.sex
                meta.region = meta.region or n.region
                meta.cell_type = meta.cell_type or n.cell_type
                have = {d.name.lower() for d in meta.drugs}
                meta.drugs += [DrugEvent(d, 0.0, float("nan"), "Notes.txt (in the bath throughout)") for d in n.drugs
                               if d.lower() not in have]
                if (meta.age, meta.sex, meta.region, meta.cell_type, len(meta.drugs)) != before:
                    project.set_metadata(path, meta)
                    touched = True
                changed += touched
    return changed
