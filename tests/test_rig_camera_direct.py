"""The camera without Infinity Analyze: taken only on request and never while its program is
open, frames demosaiced to colour and averaged for dim fluorescence, let go at once; the picture's adjustments change
what is shown, never what is saved; a snapshot is a TIFF with how it was taken and its scale. A stand-in plays
Lumenera's library (tests never load the real one)."""

from __future__ import annotations

import ctypes
import json
import time

import numpy as np
import pytest

from clampwise.rig import camera, lucam, picture


class FakeLucam:
    """Lumenera's library as a stand-in: one 8 x 6 colour camera whose raw frames count up."""

    def __init__(self, mono: bool = False) -> None:
        self.calls, self.frames, self.rate = [], 0, None
        # as the camera's program left it on the rig (2026-10-14): contrast 3.4, brightness -5
        self.props = {lucam.EXPOSURE: 37.18, lucam.GAIN: 1.2, lucam.BRIGHTNESS: -5.0, lucam.CONTRAST: 3.4,
                      lucam.GAMMA: 1.0, lucam.SATURATION: 1.0,
                      lucam.COLOR_FORMAT: lucam.CF_MONO if mono else 8}            # 8: a Bayer (colour) sensor

    def LucamNumCameras(self):
        return 1

    def LucamCameraOpen(self, index):
        self.calls.append("open")
        return 7

    def LucamCameraClose(self, handle):
        self.calls.append("close")
        return 1

    def LucamQueryVersion(self, handle, v):
        v._obj.serialnumber = 25050256
        return 1

    def LucamGetFormat(self, handle, fmt, rate):
        f = fmt._obj
        f.width, f.height, f.pixelFormat, f.subSampleX, f.subSampleY = 8, 6, lucam.PF_8, 1, 1
        rate._obj.value = 26.8
        return 1

    def LucamGetProperty(self, handle, prop, value, flags):
        value._obj.value = self.props.get(prop.value, 0.0)
        return 1

    def LucamSetProperty(self, handle, prop, value, flags):
        self.props[prop.value] = value.value
        return 1

    def LucamPropertyRange(self, handle, prop, low, high, default, flags):
        low._obj.value, high._obj.value, default._obj.value = {lucam.EXPOSURE: (0.016, 500.0, 10.0),
                                                               lucam.GAIN: (1.0, 1.2, 1.0)}.get(prop.value, (0, 1, 0))
        return 1

    def LucamEnumAvailableFrameRates(self, handle, count, rates):
        if count:
            rates[0], rates[1] = 31.4, 62.8
        return 2

    def LucamSetFormat(self, handle, fmt, rate):
        self.rate = rate.value
        return 1

    def LucamStreamVideoControl(self, handle, control, window):
        self.calls.append("stream" if control == lucam.START_STREAMING else "stop")
        return 1

    def LucamTakeVideo(self, handle, n, raw):
        size = ctypes.sizeof(raw) // n
        for k in range(n):                    # each frame of the stream a level that counts up, inside 8 bits
            time.sleep(0.005)
            self.frames += 1
            ctypes.memset(ctypes.addressof(raw) + k * size, (10 * self.frames) % 200, size)
        return 1

    def LucamConvertFrameToRgb24(self, handle, rgb, raw, w, h, pf, conv):
        level = ctypes.cast(raw, ctypes.POINTER(ctypes.c_ubyte))[0]
        out = np.frombuffer(rgb, np.uint8).reshape(h.value, w.value, 3)
        out[:] = (level, level + 1, level + 2)                             # B, G, R as Windows keeps them
        return 1

    def LucamGetLastError(self):
        return 0


def test_the_camera_is_taken_only_when_its_program_is_closed_and_let_go_at_once():
    fake = FakeLucam()
    held = lucam.Camera(fake, owner_check=lambda: True)
    assert not held.open() and "close it first" in held.status and fake.calls == []
    cam = lucam.Camera(fake, owner_check=lambda: False)
    assert cam.open() and cam.serial == 25050256 and cam.exposure_ms == pytest.approx(37.18)
    s = cam.setting(lucam.EXPOSURE)
    assert (s.low, s.high) == (pytest.approx(0.016), 500.0)
    cam.close()
    assert fake.calls == ["open", "close"] and cam.handle is None


def test_the_camera_s_own_processing_is_neutral_while_held_and_put_back_after():
    fake = FakeLucam()
    cam = lucam.Camera(fake, owner_check=lambda: False)
    cam.open()
    assert fake.props[lucam.CONTRAST] == 1.0 and fake.props[lucam.BRIGHTNESS] == 0.0
    assert lucam.found_path().is_file()                                    # noted, should Clampwise stop early
    cam.close()
    assert fake.props[lucam.CONTRAST] == pytest.approx(3.4) and fake.props[lucam.BRIGHTNESS] == -5.0
    assert not lucam.found_path().exists()
    # a session that stopped without letting go: the next one puts back what the camera had before it
    lucam.found_path().write_text('{"1": 3.4, "0": -5.0}', encoding="utf-8")
    fake.props.update({lucam.CONTRAST: 1.0, lucam.BRIGHTNESS: 0.0})
    cam = lucam.Camera(fake, owner_check=lambda: False)
    cam.open()
    cam.close()
    assert fake.props[lucam.CONTRAST] == pytest.approx(3.4) and fake.props[lucam.BRIGHTNESS] == -5.0


def test_frames_are_the_video_stream_colour_and_averaged_and_the_exposure_changes_on_the_fly():
    fake = FakeLucam()
    cam = lucam.Camera(fake, owner_check=lambda: False)
    cam.open()
    f = cam.frame()
    assert f.rgb.shape == (6, 8, 3) and tuple(f.rgb[0, 0]) == (12, 11, 10)          # BGR turned to RGB
    assert fake.rate == pytest.approx(62.8) and cam.frame_rate == pytest.approx(62.8)   # the fastest the camera has
    f = cam.frame(average=4)                                                        # frames 2-5: 20..50, mean 35
    assert f.averaged == 4 and tuple(f.rgb[0, 0]) == (37, 36, 35)
    assert fake.calls.count("stream") == 1
    cam.set_exposure(250.0, 1.1)                                                    # the stream goes on
    cam.frame()
    assert fake.props[lucam.EXPOSURE] == 250.0 and fake.props[lucam.GAIN] == pytest.approx(1.1)
    assert fake.calls.count("stream") == 1
    cam.close()
    assert fake.calls[-2:] == ["stop", "close"]


def test_a_monochrome_camera_s_frames_stay_one_grey_channel():
    fake = FakeLucam(mono=True)
    cam = lucam.Camera(fake, owner_check=lambda: False)
    cam.open()
    f = cam.frame()
    assert cam.mono and f.rgb.shape == (6, 8, 3) and tuple(f.rgb[0, 0]) == (10, 10, 10)
    assert picture.is_grey(f.rgb)                                                   # three channels, one copy
    f = cam.frame(average=4)                                                        # 20..50: 35
    assert tuple(f.rgb[2, 3]) == (35, 35, 35)
    s = picture.stats(f.rgb)
    assert s["p995"] == 35 and s["hist"].sum() == 48 and s["clipped"] == 0
    shown = picture.apply(f.rgb, picture.Adjust(contrast=2.0))                      # a lookup, still grey
    assert picture.is_grey(shown) and tuple(shown[0, 0]) == tuple(picture.apply(
        np.ascontiguousarray(f.rgb), picture.Adjust(contrast=2.0, sharpness=1e-12))[0, 0])
    cam.close()


def test_adjustments_change_what_is_shown_and_each_resets_to_neutral():
    rng = np.random.default_rng(0)
    rgb = rng.integers(40, 120, (60, 80, 3)).astype(np.uint8)
    assert picture.apply(rgb, picture.Adjust()) is rgb                               # neutral: untouched
    brighter = picture.apply(rgb, picture.Adjust(brightness=0.2))
    assert brighter.mean() > rgb.mean() + 40 and rgb.mean() < 81                    # the original is unchanged
    grey = picture.apply(rgb, picture.Adjust(saturation=0.0))
    assert np.all(grey[..., 0] == grey[..., 1]) and np.all(grey[..., 1] == grey[..., 2])
    stretched = picture.apply(rgb, picture.Adjust(auto_levels=True))
    assert stretched.min() == 0 and stretched.max() == 255
    for name in ("contrast", "gamma", "sharpness", "clarity"):
        a = picture.Adjust(**{name: 2.0})
        assert not np.array_equal(picture.apply(rgb, a), rgb) and a.changed() == {name: 2.0}
    kept = picture.preset_from_text(picture.preset_to_text(picture.Adjust(gamma=0.6, clarity=1.5)))
    assert kept == picture.Adjust(gamma=0.6, clarity=1.5)


def test_white_balance_and_the_scale_bar():
    tinted = np.zeros((10, 10, 3), np.uint8) + np.array([200, 100, 50], np.uint8)
    r, g, b = picture.white_balance(tinted)
    balanced = picture.apply(tinted, picture.Adjust(red=r, green=g, blue=b))
    assert np.ptp(balanced[0, 0].astype(int)) <= 1
    assert picture.scale_bar(0.16, 1944) == (50.0, 312)                              # a fifth of 311 um: 50 um


def test_each_objective_s_scale_follows_one_calibration_by_its_magnification():
    assert picture.parse_objectives("63, 2.5x; 20 , 20, nonsense") == [2.5, 20.0, 63.0]
    assert picture.parse_objectives("") == list(picture.OBJECTIVES)
    assert picture.objective_scale(20, {}) == (0.0, "not calibrated")
    um, how = picture.objective_scale(20, {20: 0.2})
    assert um == 0.2 and how == "calibrated at 20x"
    um, how = picture.objective_scale(2.5, {20: 0.2})                                # 8x less magnified: 8x more
    assert um == pytest.approx(1.6) and how == "estimated from the 20x calibration"
    um, how = picture.objective_scale(63, {20: 0.2, 40: 0.098})                       # the nearest calibration
    assert um == pytest.approx(0.098 * 40 / 63) and "40x" in how


def test_the_compass_turns_with_rostral_and_puts_medial_on_the_side_chosen():
    c = picture.compass(0, side_right=True)                                         # screen: x right, y down
    assert c["Rostral"] == (0.0, -1.0) and c["Caudal"] == (0.0, 1.0)
    assert c["Medial"] == (1.0, 0.0) and c["Lateral"] == (-1.0, 0.0)
    c = picture.compass(90, side_right=True)                                        # rostral right: medial down
    assert c["Rostral"] == (1.0, 0.0) and c["Medial"] == (0.0, 1.0)
    c = picture.compass(0, side_right=False)
    assert c["Medial"] == (-1.0, 0.0) and c["Lateral"] == (1.0, 0.0)
    assert list(picture.compass(45)) == ["Rostral", "Caudal", "Medial", "Lateral"]  # no dorsal or ventral


def test_the_compass_draws_the_two_axes_in_the_slice_s_plane():
    c = picture.compass(0, True, "coronal")                                         # rostral-caudal cut across
    assert list(c) == ["Dorsal", "Ventral", "Medial", "Lateral"]
    assert c["Dorsal"] == (0.0, -1.0) and c["Medial"] == (1.0, 0.0)
    c = picture.compass(0, False, "sagittal")                                       # medial-lateral cut across
    assert list(c) == ["Dorsal", "Ventral", "Rostral", "Caudal"] and c["Rostral"] == (-1.0, 0.0)
    assert list(picture.compass(0, True, "sagittal", "Anterior, posterior")) == ["Dorsal", "Ventral", "Anterior",
                                                                                 "Posterior"]
    assert list(picture.compass(0, True, "horizontal", "Letters: A P D V M L")) == ["A", "P", "M", "L"]
    assert list(picture.compass(0, True, "coronal", "Letters: R C D V M L")) == ["D", "V", "M", "L"]
    across = {p: picture.axis_name(picture.PLANES[p][2]) for p in picture.PLANES}
    assert across == {"horizontal": "DV", "coronal": "RC", "sagittal": "ML"}
    assert picture.axis_name(picture.PLANES["coronal"][2], "Anterior, posterior") == "AP"


def test_before_any_calibration_a_pixel_s_size_follows_from_the_camera_s_own_pixel():
    assert picture.pixel_um_of("INFINITY8-3M") == 4.5 and picture.pixel_um_of("Lumenera INFINITY8-3M") == 4.5
    assert picture.pixel_um_of("unknown camera") == 0.0
    um, how = picture.objective_scale(20, {}, pixel_um=4.5)                          # 4.5 um / 20x = 0.225 um
    assert um == pytest.approx(0.225) and how.startswith("nominal")
    um, _ = picture.objective_scale(63, {}, pixel_um=4.5, adapter=0.63)              # a 0.63x camera adapter
    assert um == pytest.approx(4.5 / (63 * 0.63))
    assert picture.objective_scale(20, {20: 0.21}, pixel_um=4.5) == (0.21, "calibrated at 20x")   # measured wins


def test_a_snapshot_keeps_the_camera_s_pixels_how_it_was_taken_and_its_scale(tmp_path):
    rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    path = picture.write_tiff(picture.snapshot_name(tmp_path, "40x GFP"), rgb,
                              {"exposure_ms": 250.0, "gain": 1.1, "averaged": 4,
                               "shown_with": picture.Adjust(gamma=0.6).changed()}, um_per_px=0.16)
    assert path.name.endswith("_40x_GFP.tif")
    tags = camera.tiff_tags(path.read_bytes())
    assert (tags[256], tags[257]) == (8, 6)
    info = json.loads(tags[270].decode("utf-8").rstrip("\0"))
    assert info["exposure_ms"] == 250.0 and info["shown_with"] == {"gamma": 0.6}
    from pyqtgraph.Qt import QtGui

    back = QtGui.QImage(str(path))
    assert (back.width(), back.height()) == (8, 6) and QtGui.QColor(back.pixel(1, 0)).getRgb()[:3] == (3, 4, 5)
    assert back.dotsPerMeterX() == pytest.approx(1e6 / 0.16, rel=1e-3)               # the scale, as ImageJ reads it


def _field(h=60, w=80):
    """An uneven field (illumination falling off to the right) with one small dark cell in it."""
    shade = np.linspace(200, 100, w)[None, :, None] * np.ones((h, 1, 3))
    cell = shade.copy()
    cell[25:35, 35:45] -= 30
    return shade.astype(np.float32), np.clip(cell, 0, 255).astype(np.uint8)


def test_background_subtraction_flattens_the_field_and_keeps_the_cell():
    # video-enhanced contrast (Allen et al. 1981; Inoue 1981): the background taken out, the cell left
    bg, frame = _field()
    out = picture.corrected(frame, picture.Corrections(background=bg, mode="subtract")).astype(float)
    outside = np.delete(out[:, :, 0], np.s_[25:35], axis=0)
    assert np.ptp(outside) <= 1                                                   # flat where there is no cell
    assert out[30, 40, 0] == pytest.approx(out[0, 40, 0] - 30, abs=1)              # the cell's contrast kept
    flat = picture.corrected(frame, picture.Corrections(background=bg, mode="flat")).astype(float)
    assert np.ptp(np.delete(flat[:, :, 0], np.s_[25:35], axis=0)) <= 1


def test_the_dark_frame_is_taken_out_and_clipped_pixels_show_red():
    raw = np.full((4, 4, 3), 50, np.uint8)
    raw[0, 0] = 255
    dark = np.full((4, 4, 3), 10, np.float32)
    assert picture.corrected(raw, picture.Corrections(dark=dark))[1, 1, 0] == 40
    shown = picture.shown(raw, picture.Corrections(), picture.Adjust())
    assert tuple(shown[0, 0]) == (255, 0, 0) and tuple(shown[1, 1]) == (50, 50, 50)


def test_a_kept_fluorescence_picture_is_laid_over_the_live_one():
    dim = np.full((40, 40, 3), 20, np.uint8)
    dim[18:22, 18:22] = 120                                                       # one labelled cell
    over = picture.overlay_from(dim, picture.Corrections())
    assert over[20, 20] == pytest.approx(1.0, abs=0.05) and over[2, 2] == pytest.approx(0.0, abs=0.05)
    live = np.full((40, 40, 3), 100, np.uint8)
    c = picture.Corrections(overlay=over, overlay_colour=(0, 255, 0), overlay_opacity=0.5, show_clipped=False)
    out = picture.shown(live, c, picture.Adjust())
    assert tuple(out[20, 20]) == (50, 178, 50) and tuple(out[2, 2]) == (100, 100, 100)
    green = picture.shown(live, picture.Corrections(lut="green"), picture.Adjust())
    assert tuple(green[0, 0]) == (0, 100, 0)


def test_auto_exposure_brings_the_brightest_to_80_percent_and_backs_off_when_clipped():
    rgb = np.full((10, 10, 3), 100, np.uint8)
    s = picture.stats(rgb)
    assert s["p995"] == 100 and s["clipped"] == 0 and s["hist"].sum() == 100
    assert picture.next_exposure(20.0, s["p995"], 0.016, 500.0) == pytest.approx(20.0 * 204 / 100)
    assert picture.next_exposure(20.0, 255.0, 0.016, 500.0, clipped=0.2) == 10.0
    assert picture.next_exposure(400.0, 10.0, 0.016, 500.0) == 500.0                 # held to the camera's range
