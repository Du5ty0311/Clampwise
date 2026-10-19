"""One look at the whole rig: the amplifier, what Clampex last did, the sweeps of the file it is writing (each one's
test pulse), the Polygon's session, the camera's last picture, and the rig programs' windows. Made off the window's
thread (``Watch.look``), shown by the Live screen."""

from __future__ import annotations

import collections
import datetime as dt
import time
from dataclasses import dataclass, field

from . import camera, growing, labbook, multiclamp, polyscan, windows
from .files import modified


@dataclass
class Snapshot:
    at: float = field(default_factory=time.time)
    amplifier: multiclamp.AmplifierState | None = None
    clampex_open: bool = False
    data_folder: str = ""                                # where Clampex writes (its Lab Book's last word)
    last_file: labbook.FileRecord | None = None
    last_test: labbook.MembraneTest | None = None
    light: polyscan.Session | None = None                # the newest session; running when its program is open
    light_open: bool = False
    picture: camera.Picture | None = None
    camera_open: bool = False
    windows: dict[str, list[windows.AppWindow]] = field(default_factory=dict)
    writing: str = ""                                    # the file whose sweeps are read as it is written (path)
    sweeps: list[growing.SweepTest] = field(default_factory=list)      # its sweeps completed since the last look
    telegraph: object = None                             # MultiClamp's telegraph for the channel read (rig.telegraph)

    def running(self, role: str) -> bool:
        return bool(self.windows.get(role))


FILE_MARGIN_S = 120.0       # a file changed this long before watching began is still read (a recording under way)


class Watch:
    """Looks at the rig when asked. The amplifier's connection is opened, used and closed in the thread that
    calls ``look`` (Commander's library wants one thread); the files are read again only when they change."""

    def __init__(self, commander: multiclamp.Commander | None = None, *, labbook_path=None, polyscan_root=None,
                 camera_settings=None, find_windows=windows.rig_windows, read_telegraph=None) -> None:
        # None: each program's usual place, looked up when read (tests and other PCs point elsewhere)
        self.commander = commander or multiclamp.Commander()
        self.labbook_path, self.polyscan_root, self.camera_settings = labbook_path, polyscan_root, camera_settings
        self.find_windows = find_windows
        if read_telegraph is None and commander is None:      # the rig's own (tests pass theirs, or none)
            from .telegraph import read as read_telegraph
        self.read_telegraph = read_telegraph
        self.slow_every_s = 2.0
        self.telegraph_every_s = 1.0           # the telegraph repeats Commander's own mode and scales: once a second
        self._telegraph, self._telegraph_at = None, -1e9
        # the first look shows the programs and the amplifier at once; the files (the Lab Book is a second's
        # read) come with the next, half a second later
        self._slow_at = time.monotonic() - self.slow_every_s + 0.5
        self._slow: dict = {}
        self.folder = ""                    # a folder watched beside the acquisition's own (the open project's)
        self._since = time.time() - FILE_MARGIN_S
        self._growing: growing.Growing | None = None
        self.commands: collections.deque = collections.deque()     # the amplifier's controls asked for, run here
        self.controls = None                                        # rig.amplifier_control, when allowed
        growing.warm()                     # the first sweep is not kept waiting for the analysis code to load

    def look(self) -> Snapshot:
        snap = Snapshot()
        while self.commands:               # in this thread, which holds Commander's connection, before reading
            if self.controls is None:
                from .amplifier_control import Controls

                self.controls = Controls(self.commander)
            command, said = self.commands.popleft()
            said(self.controls.run(command))
            self._telegraph_at = -1e9                     # a control sent: the telegraph read again at once
        snap.windows = self.find_windows()
        snap.clampex_open, snap.light_open = snap.running("acquisition"), snap.running("light")
        snap.camera_open = snap.running("camera")
        snap.amplifier = self.commander.read()             # it looks for Commander's window before any call
        amp = snap.amplifier
        if amp is not None and amp.connected and amp.device is not None and self.read_telegraph is not None:
            # what Commander telegraphs to an acquisition program: the mode, each output's scale and gain, the filter, the command's sensitivity
            if time.monotonic() - self._telegraph_at >= self.telegraph_every_s:
                self._telegraph_at = time.monotonic()
                self._telegraph = self.read_telegraph(amp.device.serial, amp.device.channel, timeout=0.3)
            snap.telegraph = self._telegraph
        if time.monotonic() - self._slow_at >= self.slow_every_s:
            self._slow_at = time.monotonic()
            self._slow = self._files()
        snap.data_folder = self._slow.get("data_folder", "")
        snap.last_file, snap.last_test = self._slow.get("last_file"), self._slow.get("last_test")
        snap.light, snap.picture = self._slow.get("light"), self._slow.get("picture")
        self._recorded(snap)
        return snap

    def _recorded(self, snap: Snapshot) -> None:
        """The newest file changed since watching began, in the acquisition's folder or the watched one: its sweeps
        completed since the last look. A file left with sweeps not yet read (a look reads at most 20) is finished
        before the next one is taken, so none of its sweeps is skipped."""
        if self._growing is not None:
            snap.writing = str(self._growing.path)
            snap.sweeps = self._growing.look()
            if snap.sweeps:
                return
        found = [p for p in (growing.newest(f, self._since) for f in {snap.data_folder, self.folder} if f) if p]
        if found:
            newest = max(found, key=modified)
            if self._growing is None or self._growing.path != newest:
                self._growing = growing.Growing(newest)
                snap.writing = str(newest)
                snap.sweeps = self._growing.look()

    def _files(self) -> dict:
        out: dict = {}
        book = labbook.current(self.labbook_path)
        if book is not None:
            folders = [e.path for e in book.entries if e.kind == "data_folder"]
            out["data_folder"] = folders[-1] if folders else ""
            out["last_file"] = book.files[-1] if book.files else None
            tests = [e for e in book.entries if e.kind == "test_value"]
            if tests:
                # the last Membrane Test that logged values: its values together
                last = None
                for e in book.entries:
                    if e.kind == "test_start":
                        last = labbook.MembraneTest(e.line)
                    elif e.kind == "test_value":
                        last = last or labbook.MembraneTest(e.line)
                        last.values[e.key] = e.value
                        out["last_test"] = last
        try:
            out["light"] = polyscan.latest(self.polyscan_root)
        except OSError:
            out["light"] = None
        settings = camera.read_settings(self.camera_settings)
        out["picture"] = camera.newest(camera.watched_folders(settings)) if settings else None
        return out

    def close(self) -> None:
        self.commander.close()


def since(when: dt.datetime | None, now: dt.datetime | None = None) -> str:
    """When, in words a glance takes in: 12 s ago, 4 min ago, 2 h 5 min ago, 12 days ago, on 9 Jun 2025."""
    if when is None:
        return ""
    s = max(0, int(((now or dt.datetime.now()) - when).total_seconds()))
    if s < 60:
        return f"{s} s ago"
    if s < 3600:
        return f"{s // 60} min ago"
    if s < 48 * 3600:
        return f"{s // 3600} h {s % 3600 // 60} min ago"
    if s < 60 * 86400:
        return f"{s // 86400} days ago"
    return f"on {when.day} {when:%b %Y}"
