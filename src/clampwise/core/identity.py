"""Recording identity from folder and file names, by rules the user can see
and edit (B1). Anything typed wins over anything parsed, and every field says
where it came from.

A rule is a name and a regular expression with named groups (``animal``,
``slice``, ``cell``, ``condition``) tried against the path's parts from the
file upward. The defaults cover the lab's layouts: ``Animal\\Slice\\Cell\\file``
folders, and ``a12_s2_c3_cc_0001.abf`` file names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .recording import Identity

FIELDS = ("animal", "slice", "cell", "condition")


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: str
    applies_to: str = "folders"        # "folders": each folder name above the file; "file": the file's stem

    def match(self, text: str) -> dict[str, str]:
        m = re.search(self.pattern, text, re.IGNORECASE)
        if not m:
            return {}
        return {k: v for k, v in m.groupdict().items() if v}


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("slice folder", r"^(?P<slice>slice\s*\d+)$", "folders"),
    Rule("cell folder", r"^(?P<cell>cell\s*\d+|pair\d*)$", "folders"),
    Rule("file name a_s_c", r"^(?P<animal>a\d+)_(?P<slice>s\d+)_(?P<cell>c\d+)(?:_(?P<condition>[a-z]+))?", "file"),
)


@dataclass
class Rules:
    rules: list[Rule] = field(default_factory=lambda: list(DEFAULT_RULES))

    def parse(self, path: Path | str, stop_at: tuple[Path, ...] = ()) -> Identity:
        """Identity from the path: the file's stem by the file rules, then each
        folder above it (nearest first) by the folder rules; the first folder
        above a slice or cell folder that no rule names is the animal. The
        walk stops at a folder in ``stop_at`` (the project folder), which is
        itself still read: an experiment opened at its animal folder keeps it."""
        path = Path(path)
        out = Identity()
        sources: dict[str, str] = {}

        def take(found: dict[str, str], where: str) -> None:
            for k, v in found.items():
                if k in FIELDS and not getattr(out, k):
                    setattr(out, k, v.strip())
                    sources[k] = where

        for rule in self.rules:
            if rule.applies_to == "file":
                take(rule.match(path.stem), f"file name ({rule.name})")
        stops = {Path(s).resolve() for s in stop_at}
        folders = []
        for p in path.parents:
            if not p.name:
                break
            folders.append(p)
            if p.resolve() in stops:
                break
        for folder in folders:
            matched = False
            for rule in self.rules:
                if rule.applies_to == "folders":
                    found = rule.match(folder.name)
                    if found:
                        take(found, f"folder '{folder.name}' ({rule.name})")
                        matched = True
            if not matched and (out.slice or out.cell) and not out.animal:
                out.animal = folder.name
                sources["animal"] = f"folder '{folder.name}' (the folder above the slice or cell)"
                break
        out.sources = sources
        return out

    @staticmethod
    def merge(parsed: Identity, typed: Identity) -> Identity:
        """What was typed wins; blanks keep the parsed value."""
        out = Identity(sources=dict(parsed.sources))
        for k in (*FIELDS, "label"):
            t = getattr(typed, k)
            if t:
                setattr(out, k, t)
                out.sources[k] = "typed"
            else:
                setattr(out, k, getattr(parsed, k))
        return out
