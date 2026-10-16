"""Updating from the GitHub releases: the newest release read, offered only when newer and with a checksum, the zip
checked, unpacked beside the program, and the program's folder swapped (the old one kept if the swap cannot run)."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile

import pytest

from clampwise import update


def _release_doc(tag="v1.2.0", digest=None, url="https://example.org/Clampwise-windows.zip", name=update.ASSET):
    asset = {"name": name, "size": 10, "browser_download_url": url}
    if digest is not None:
        asset["digest"] = digest
    return {"tag_name": tag, "body": "What changed.", "html_url": "https://example.org/release", "assets": [asset]}


def test_versions_compare_as_numbers():
    assert update.parse_version("v1.10.2") == (1, 10, 2) and update.parse_version("beta") == ()
    assert update.is_newer("1.0.1", "1.0.0") and update.is_newer("v1.10.0", "1.9.9")
    assert not update.is_newer("1.0.0", "1.0.0") and not update.is_newer("0.9", "1.0.0")
    assert not update.is_newer("nightly", "1.0.0")


def test_a_release_is_offered_only_with_its_zip_and_checksum():
    r = update.read_release(_release_doc(digest="sha256:" + "ab" * 32))
    assert r is not None and r.version == "1.2.0" and r.sha256 == "ab" * 32 and r.notes == "What changed."
    assert update.read_release(_release_doc(digest=None)) is None             # no checksum: not offered
    assert update.read_release(_release_doc(digest="sha256:00", name="other.zip")) is None
    assert update.read_release(_release_doc(tag="latest", digest="sha256:00")) is None


def _zip(tmp_path, with_exe=True):
    z = tmp_path / "Clampwise-windows.zip"
    with zipfile.ZipFile(z, "w") as f:
        if with_exe:
            f.writestr("Clampwise/Clampwise.exe", b"new program")
        f.writestr("Clampwise/_internal/data.txt", b"new data")
    return z


def test_the_newest_release_is_read_and_its_zip_checked(tmp_path, monkeypatch):
    z = _zip(tmp_path)
    good = hashlib.sha256(z.read_bytes()).hexdigest()
    api = tmp_path / "latest.json"
    api.write_text(json.dumps(_release_doc(digest="sha256:" + good, url=z.as_uri())))
    monkeypatch.setenv("CLAMPWISE_UPDATE_URL", api.as_uri())
    r = update.latest()
    seen = []
    got = update.download(r, tmp_path / "dl", progress=lambda d, t: seen.append(d))
    assert got.read_bytes() == z.read_bytes() and seen and seen[-1] == z.stat().st_size
    r.sha256 = "0" * 64                                                        # tampered or broken: thrown away
    with pytest.raises(RuntimeError, match="checksum"):
        update.download(r, tmp_path / "dl2")
    assert not list((tmp_path / "dl2").iterdir())


def test_the_new_copy_is_unpacked_beside_the_program(tmp_path):
    install = tmp_path / "Programs" / "Clampwise"
    install.mkdir(parents=True)
    staged = update.stage(_zip(tmp_path), install)
    assert staged.parent.parent == install.parent and (staged / "Clampwise.exe").read_bytes() == b"new program"
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(RuntimeError, match="no Clampwise"):                   # not a Clampwise zip: nothing staged
        update.stage(_zip(other, with_exe=False), install)


@pytest.mark.skipif(sys.platform != "win32", reason="the swap is a Windows PowerShell script")
def test_the_swap_replaces_the_folder_or_keeps_the_old_one(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    install = tmp_path / "Programs" / "Clampwise"
    install.mkdir(parents=True)
    (install / "Clampwise.exe").write_bytes(b"old program")
    staged = update.stage(_zip(tmp_path), install)
    update.start_swap(staged, install, pid=0, launch=False).wait(60)
    assert (install / "Clampwise.exe").read_bytes() == b"new program"
    assert not install.with_name("Clampwise.old").exists() and not (install.parent / ".Clampwise-update").exists()
    assert "updated" in update.log_path().read_text()
    # a file of the old copy held open: the folder cannot move, so nothing is changed and the reason is logged
    staged = update.stage(_zip(tmp_path), install)
    (install / "held.txt").write_text("in use")
    with open(install / "held.txt"):
        update.start_swap(staged, install, pid=0, launch=False).wait(90)
    assert (install / "held.txt").exists() and "old copy kept" in update.log_path().read_text()


def test_settings_offers_a_newer_release_and_never_installs_from_source(qapp, monkeypatch):
    from clampwise.app import theme
    from clampwise.app.window import MainWindow

    theme.apply(qapp, theme.LIGHT)
    w = MainWindow(theme.LIGHT, persist=False)
    s = w.screens["settings"]
    newer = update.Release("99.0.0", "v99.0.0", "notes", "https://example.org/r", "https://example.org/z", 1, "00")
    monkeypatch.setattr(update, "latest", lambda: newer)
    s.check_updates(quiet=False)
    assert _wait(qapp, lambda: s.do_update.isVisibleTo(s))
    assert s.do_update.text() == "Update to 99.0.0" and "99.0.0 is available" in s.update_note.text()
    s.install_update()                                                        # the tests run from source
    assert "runs from source" in s.update_note.text()
    monkeypatch.setattr(update, "latest", lambda: update.Release("0.1", "v0.1", "", "", "", 1, "00"))
    s.check_updates(quiet=False)
    assert _wait(qapp, lambda: "is the newest version" in s.update_note.text()) and not s.do_update.isVisibleTo(s)
    assert not s.updates_wanted()                                             # a test window never checks by itself
    w.close()


@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait(qapp, until, seconds=8.0):
    import time

    end = time.time() + seconds
    while time.time() < end and not until():
        qapp.processEvents()
        time.sleep(0.02)
    return until()
