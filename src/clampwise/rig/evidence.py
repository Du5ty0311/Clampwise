"""What the rig's programs recorded about one recording (what was applied, detected rather than typed in): the
patterned light's session running when it was made, and the
acquisition's Lab Book record of it (the protocol, the Membrane Tests before and after). The analyses use it beside
what the file itself records, never instead, and say so: a stimulus line's kind when its name does not tell, Rs
when the file has no test step and none was typed.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass

from . import labbook, polyscan

NAN = float("nan")
log = logging.getLogger("clampwise")
LEAST_FOLDERS = 2               # a Lab Book record is the file's when its name and at least its slice and cell match


@dataclass
class Evidence:
    session: polyscan.Session | None = None
    record: labbook.FileRecord | None = None

    @property
    def empty(self) -> bool:
        return self.session is None and self.record is None

    @property
    def light_waiting(self) -> bool:
        """The patterned light was waiting on an external trigger while the recording was made."""
        return self.session is not None and self.session.externally_triggered

    def rs(self) -> tuple[float, float]:
        """(before, after): Rs in MOhm from the Membrane Tests logged before and after the file; NaN where none."""
        if self.record is None:
            return NAN, NAN
        before = self.record.before.values.get("ra_mohm", NAN) if self.record.before else NAN
        after = self.record.after.values.get("ra_mohm", NAN) if self.record.after else NAN
        return before, after

    def light_words(self) -> str:
        s = self.session
        if s is None:
            return ""
        return (f"the patterned light's session {s.name} was running" +
                (", waiting on its external trigger" if s.externally_triggered else "") +
                (f" (pattern sequence {s.sequence}" + (f", {s.regions} regions" if s.regions else "") + ")"
                 if s.sequence else ""))

    def notes(self) -> list[str]:
        out = []
        if self.session is not None:
            out.append("Patterned light: " + self.session.describe(named=False))
        if self.record is not None:
            line = f"Acquisition's Lab Book: {self.record.name} {self.record.outcome}"
            if self.record.protocol:
                line += f", protocol {self.record.protocol.rsplit(chr(92), 1)[-1]}"
            for when, test in (("before", self.record.before), ("after", self.record.after)):
                if test is not None and test.values:
                    line += f"; Membrane Test {when}: {test.describe()}"
            out.append(line)
        return out

    def digest(self) -> str:
        """What of it the results' cache keys on: the Rs it gives (the light's session changes the stimulus's
        kind, which every result is already keyed on)."""
        before, after = self.rs()
        if not (math.isfinite(before) or math.isfinite(after)):
            return ""
        return hashlib.sha256(f"{before:.6g}|{after:.6g}".encode()).hexdigest()[:16]


def lab_book_rs(path, labbook_path=None) -> tuple[float, float]:
    """(before, after) Rs of the file at ``path`` from the acquisition's Lab Book alone (no recording needed)."""
    try:
        book = labbook.current(labbook_path)
        record = book.record(path, least=LEAST_FOLDERS) if book is not None else None
    except Exception:
        log.exception("the Lab Book could not be read for %s", path)
        record = None
    return Evidence(record=record).rs()


def for_recording(recording, *, path=None, labbook_path=None, polyscan_root=None) -> Evidence:
    """The rig's records of ``recording`` (``path``: where it is, when not its own source); none on a PC
    without the rig's programs."""
    ev = Evidence()
    start = getattr(recording, "start_time", None)
    try:                                  # a record that cannot be read is no evidence, never a failed analysis
        if start is not None:
            ev.session = polyscan.running_at(start, polyscan_root)
        book = labbook.current(labbook_path)
        if book is not None:
            ev.record = book.record(path or recording.source, least=LEAST_FOLDERS)
    except Exception:
        log.exception("the rig's records could not be read for %s", path or getattr(recording, "source", ""))
    return ev
