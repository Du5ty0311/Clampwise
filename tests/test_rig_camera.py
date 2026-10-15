"""Infinity Analyze's settings and pictures read: the camera and folders from its settings, each TIFF's size,
time, camera, exposure, gain and length calibration (a doubtful calibration said, not corrected), and a
recording's cell pictures from its own folder."""

from __future__ import annotations

import datetime as dt
import struct

import pytest

from clampwise.rig import camera

XMP = """<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:lumen="Lumenera/">
   <lumen:Camera lumen:Exposure="33.750999" lumen:Gain="9.189453"/>
   <lumen:Calibration lumen:Name="{name}" lumen:Type="Length" lumen:SourceDimension="{px}" lumen:SourceUnit="px"
    lumen:DestinationDimension="{length}" lumen:DestinationUnit="{unit}"/>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


def _tiff(path, *, width=1944, height=1472, when="2026:10:01 15:47:46", xmp=""):
    """A little-endian TIFF with ImageWidth, ImageLength, Make, Model, DateTime and XMP tags (no pixels needed)."""
    entries, blobs = [], b""
    texts = {271: b"INFINITY\x00", 272: b"INFINITY8-3C\x00", 306: when.encode() + b"\x00", 700: xmp.encode("utf-8")}
    n = 2 + len(texts)
    data_at = 8 + 2 + 12 * n + 4
    for tag, value in ((256, width), (257, height)):
        entries.append(struct.pack("<HHII", tag, 4, 1, value))
    for tag, raw in texts.items():
        entries.append(struct.pack("<HHII", tag, 7 if tag == 700 else 2, len(raw), data_at + len(blobs)))
        blobs += raw
    path.write_bytes(b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", n) + b"".join(entries) + b"\x00" * 4 + blobs)
    return path


def test_infinity_analyze_s_settings_say_the_camera_and_its_folders(tmp_path):
    ini = tmp_path / "InfinityAnalyze.ini"
    ini.write_text("[Images]\nLastWriteFolder=E:/lab_data/R007/Slice1/cell2\n[Camera]\nModelName=INFINITY8-3C\n"
                   "OutputFileFormat=TIFF\nOutputFileFolder=C:/Users/x/Pictures/INFINITY ANALYZE\nCaptureBitDepth=8\n"
                   "CaptureBitDepth=16\n[Calibration]\nLastSelectedCalibrationUnits=\\x3bcm\n", encoding="utf-8")
    s = camera.read_settings(ini)
    assert (s.camera, s.file_format, s.last_saved_folder) == ("INFINITY8-3C", "TIFF", "E:/lab_data/R007/Slice1/cell2")
    assert [p.name for p in camera.watched_folders(s)] == ["cell2", "INFINITY ANALYZE"]
    assert camera.read_settings(tmp_path / "missing.ini") is None


def test_a_picture_is_read_for_its_size_time_camera_exposure_and_calibration(tmp_path):
    p = camera.read_picture(_tiff(tmp_path / "pipette.tif", xmp=XMP.format(name="Cal 40x", px=1000, length=200,
                                                                            unit="\u03bcm")))
    assert (p.width, p.height, p.camera) == (1944, 1472, "INFINITY INFINITY8-3C")
    assert p.taken == dt.datetime(2026, 10, 1, 15, 47, 46)
    assert p.exposure_ms == pytest.approx(33.751, abs=1e-4) and p.gain == pytest.approx(9.189453)
    assert p.calibration == "Cal 40x" and p.um_per_px == pytest.approx(0.2) and not p.notes
    assert "0.2 um per pixel (Cal 40x)" in p.describe()


def test_a_doubtful_calibration_is_said_not_corrected(tmp_path):
    # this rig's own calibration reads 2430 px for 1 um: 0.0004 um a pixel, far finer than a camera on a microscope
    p = camera.read_picture(_tiff(tmp_path / "f.tif", xmp=XMP.format(name="Cal#1_8202026", px=2430, length=1,
                                                                      unit="\u03bcm")))
    assert p.um_per_px == pytest.approx(1 / 2430) and "check it in Infinity Analyze" in p.notes[0]


def test_a_recording_s_cell_pictures_are_those_in_its_folder(tmp_path):
    cell = tmp_path / "R007" / "Slice1" / "cell2"
    cell.mkdir(parents=True)
    _tiff(cell / "pipette.tif", when="2026:10:01 15:47:46")
    _tiff(cell / "Pipette_fluorescence.tif", when="2026:10:01 15:48:37")
    (cell / "26o01004.abf").write_bytes(b"ABF2")
    (cell / "notes.txt").write_text("not a picture")
    found = camera.for_recording(cell / "26o01004.abf")
    assert [p.path.name for p in found] == ["pipette.tif", "Pipette_fluorescence.tif"]
    assert camera.newest([cell]).path.name == "Pipette_fluorescence.tif"
    assert camera.pictures(tmp_path / "nowhere") == [] and camera.tiff_tags(b"not a tiff") == {}
