"""Every test sees an empty rig: the rig's programs' records (the acquisition's Lab Book, the patterned light's
sessions, the camera's settings, the named rigs) are looked for in an empty folder, never on the PC the tests run on, so no
result depends on whose rig it is. Tests of the rig pass their own records."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_rig_records(tmp_path_factory, monkeypatch):
    from clampwise.rig import camera, labbook, polyscan

    empty = tmp_path_factory.mktemp("no_rig")
    monkeypatch.setattr(labbook, "DEFAULT", empty / "Clampex LabBook.rtf")
    monkeypatch.setattr(polyscan, "ROOT", empty / "Mightex")
    monkeypatch.setattr(camera, "SETTINGS", empty / "InfinityAnalyze.ini")
    monkeypatch.setattr(polyscan, "program_running", lambda: False)        # not this PC's PolyScan
    monkeypatch.setenv("CLAMPWISE_RIGS", str(empty / "rigs.json"))         # nor its rig profiles
    from clampwise.rig import lucam

    monkeypatch.setattr(lucam, "found_path", lambda: empty / "camera_found.json")   # nor its camera's note
    from clampwise.project import rigs

    rigs._cache.clear()
    labbook._cache.clear()
    polyscan._folders.clear()
    polyscan._cache.clear()
