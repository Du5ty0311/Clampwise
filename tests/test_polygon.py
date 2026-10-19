"""The Polygon: its patterns on the diamond mirrors, PolyScan's sequences, the driver, and the light test and
calibration run against a simulated Polygon under a simulated camera."""

from __future__ import annotations

import ctypes
import struct
import time

import numpy as np
import pytest

from clampwise.rig import patterns
from clampwise.rig.polygon import Polygon


@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ------------------------------------------------------------------ patterns

def test_a_round_spot_is_round_on_the_field_not_on_the_mirror_grid():
    lit = patterns.spot(0.5, 0.5, 0.2)[0].mirrors()
    rows, cols = np.nonzero(lit)
    across = (cols.max() - cols.min() + 1)                       # mirror widths
    down = (rows.max() - rows.min() + 1) / 2                     # rows are half a mirror apart
    assert across == pytest.approx(0.2 * patterns.FIELD_W, rel=0.03)
    assert down == pytest.approx(across, rel=0.03)               # as tall as wide, on the slice


def test_a_grid_is_numbered_from_the_top_left_along_the_rows_as_polyscan_numbers_it():
    steps = patterns.grid(10, 10)
    assert [s.name for s in steps[:3]] == ["1", "2", "3"] and steps[-1].name == "100"
    assert steps[0].shapes[0].points == pytest.approx((0.0, 0.0, 0.1, 0.1))
    assert steps[1].shapes[0].points == pytest.approx((0.1, 0.0, 0.2, 0.1))      # 2 is right of 1
    assert steps[10].shapes[0].points == pytest.approx((0.0, 0.1, 0.1, 0.2))     # 11 starts the second row
    total = sum(s.mirrors().sum() for s in steps)
    assert total == pytest.approx(patterns.ROWS * patterns.COLUMNS, rel=0.002)  # the cells tile the field


def test_the_image_sent_lights_with_the_byte_the_light_test_found():
    mask = patterns.grid(2, 2)[0].mirrors()
    on255, on0 = patterns.image(mask, 255), patterns.image(mask, 0)
    assert on255.shape == (1140, 912) and on255.dtype == np.uint8 and on255.flags["C_CONTIGUOUS"]
    assert set(np.unique(on255[mask])) == {255} and set(np.unique(on255[~mask])) == {0}
    assert set(np.unique(on0[mask])) == {0} and set(np.unique(on0[~mask])) == {255}


def test_an_image_file_is_lit_where_it_is_bright(tmp_path):
    from PIL import Image

    a = np.zeros((100, 160), np.uint8)
    a[:, :80] = 255                                              # the left half white
    Image.fromarray(a).save(tmp_path / "half.png")
    lit = patterns.from_image(tmp_path / "half.png")[0].mirrors()
    assert lit[:, :400].all() and not lit[:, 520:].any()
    assert not patterns.from_image(tmp_path / "half.png", invert=True)[0].mirrors()[:, :400].any()


def _region(kind: bytes, number: int, name: str, box) -> bytes:
    t = b"PolyScan.Utilities.C" + kind
    return (bytes([len(t)]) + t + struct.pack("<i", number) + bytes([len(name)]) + name.encode() + b"\x00" +
            struct.pack("<i", -1) + struct.pack("<f", 0.0036) + b"\x01" + struct.pack("<4f", *box) +
            struct.pack("<i", 0))


def test_polyscan_s_later_sequences_give_their_regions_in_order(tmp_path):
    data = b"\x070.0.0.1" + b"someone@somewhere.edu PC-IDS" + b"\x00" * 20 + struct.pack("<i", 3)
    data += _region(b"Rectangle", 0, "EWA", (0, 0, 1, 1))                        # the field itself: left out
    data += _region(b"Rectangle", 1, "ROI1", (0.0, 0.0, 0.1, 0.1))
    data += _region(b"Ellipse", 2, "ROI2", (0.1, 0.0, 0.2, 0.1))
    data += b"\x00" * 40
    (tmp_path / "Grid.seq").write_bytes(data)
    steps, notes = patterns.read_seq(tmp_path / "Grid.seq")
    assert [s.name for s in steps] == ["ROI1", "ROI2"] and not notes
    assert steps[1].shapes[0].kind == "ellipse" and steps[1].shapes[0].points == pytest.approx((0.1, 0, 0.2, 0.1))
    assert not any("somewhere" in s.describe() for s in steps)


def test_polyscan_s_earlier_sequences_give_each_step_s_cells_and_regions(tmp_path):
    def step(cols, rows, cells, regions=b"", nreg=0):
        per_row = (cols + 7) // 8
        bits = bytearray(rows * per_row)
        for i, j in cells:
            bits[i * per_row + j // 8] |= 0x80 >> (j % 8)
        return struct.pack("<3i", cols, rows, 1) + bytes(bits) + struct.pack("<iffi", 1, 1.0, 1.0, nreg) + regions

    data = b"\x88\x88\x88\x88" + struct.pack("<i", 3)
    data += step(10, 10, [(0, 0)])                                               # cell 1
    data += step(10, 10, [(9, 9)])                                               # cell 100
    data += step(1, 1, [], _region(b"Ellipse", 1, "ROI1", (0.4, 0.4, 0.6, 0.6)), 1)
    (tmp_path / "old.seq").write_bytes(data)
    steps, notes = patterns.read_seq(tmp_path / "old.seq")
    assert len(steps) == 3 and not notes
    assert steps[0].shapes[0].points == pytest.approx((0, 0, 0.1, 0.1))
    assert steps[1].shapes[0].points == pytest.approx((0.9, 0.9, 1.0, 1.0))
    assert steps[2].shapes[0].kind == "ellipse"


def test_the_fit_recovers_how_the_field_lies_under_the_camera_mirrored_and_turned():
    a = np.radians(180.0)                                        # PolyScan turns its images round: anything goes
    true = [[-0.004 * np.cos(a), 0.004 * np.sin(a), 0.9], [0.004 * np.sin(a), 0.006 * np.cos(a), 0.8]]
    cam = np.array([[50.0, 40.0], [180.0, 45.0], [60.0, 150.0], [170.0, 140.0], [110.0, 90.0]])
    c = patterns.Calibration(to_field=true)
    u, v = c.field_of(cam[:, 0], cam[:, 1])
    fitted, residual = patterns.fit_affine(cam, np.column_stack([u, v]))
    assert np.allclose(fitted, true) and residual < 1e-6
    x, y = patterns.Calibration(to_field=fitted).camera_of(u, v)
    assert np.allclose(np.column_stack([x, y]), cam)


def test_a_spot_is_found_where_it_is_lit_and_not_in_noise():
    rng = np.random.default_rng(1)
    off = 20 + rng.normal(0, 2, (160, 200))
    on = off.copy()
    yy, xx = np.mgrid[0:160, 0:200]
    on += 80 * (((xx - 130.0) ** 2 + (yy - 50.0) ** 2) < 36)
    found = patterns.find_spot(on, off)
    assert found is not None and found == pytest.approx((130, 50), abs=1.0)
    assert patterns.find_spot(off + rng.normal(0, 2, off.shape), off) is None


# ------------------------------------------------------------------ the driver

class FakeSDK:
    """The SDK as the Polygon answers: one device, numbered 1; the last image shown kept."""

    def __init__(self, devices=1):
        self.devices, self.calls, self.image, self.mode = devices, [], None, None

    def MTPLG_InitDevice(self, n):
        self.calls.append(("init", n))
        return self.devices

    def MTPLG_ConnectDev(self, i):
        self.calls.append(("connect", i))
        return 0 if i == 1 and self.devices else -5

    def MTPLG_GetDevModuleNo(self, i, buf):
        buf.value = b"DSIK3EEEEEEEEEEE99999999999999"                            # holds the serial
        return 0

    def MTPLG_SetDevDisplayMode(self, i, mode):
        self.mode = mode
        return 0

    def MTPLG_SetDevStaticImageFromMemory(self, i, ptr, order):
        n = patterns.ROWS * patterns.COLUMNS
        self.image = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), (n,)).copy().reshape(
            patterns.ROWS, patterns.COLUMNS)
        self.calls.append(("image", i, order))
        return 0

    def MTPLG_GetDevRunStatus(self, i, state, timeout):
        state._obj.Temperature = 41
        return 0

    def MTPLG_DisconnectDev(self, i):
        self.calls.append(("disconnect", i))
        return 0

    def MTPLG_UnInitDevice(self):
        self.calls.append(("uninit",))
        return 0


def test_the_polygon_is_not_taken_while_polyscan_holds_it():
    sdk = FakeSDK()
    p = Polygon(sdk, polyscan_running=lambda: True)
    assert not p.open() and "own program" in p.status and sdk.calls == []


def test_the_polygon_is_device_1_shows_still_pictures_and_never_tells_its_serial():
    sdk = FakeSDK()
    p = Polygon(sdk, polyscan_running=lambda: False)
    assert p.open() and ("connect", 1) in sdk.calls and sdk.mode == 0
    assert "9999" not in p.status and "9999" not in p.model
    step = patterns.grid(2, 2)[3]
    assert p.show(step, on_value=255) and p.shown == "4"
    assert sdk.image.shape == (1140, 912) and sdk.image[-1, -1] == 255 and sdk.image[0, 0] == 0
    assert p.off(255) and not sdk.image.any() and p.shown is None
    assert p.temperature() == 41
    p.close()
    assert sdk.calls[-2:] == [("disconnect", 1), ("uninit",)] and not p.connected


def test_no_polygon_found_is_said_and_nothing_is_left_open():
    sdk = FakeSDK(devices=0)
    p = Polygon(sdk, polyscan_running=lambda: False)
    assert not p.open() and "none found" in p.status and sdk.calls[-1] == ("uninit",)


# ------------------------------------------------------------------ the light test and calibration, simulated

class SimRig:
    """A Polygon under a camera: the camera's picture lit where the mirrors shown light the slide. The field lies
    turned round and mirrored (as nothing guarantees otherwise), and a mirror lights with ``on``."""

    def __init__(self, on: int):
        self.on = on
        self.cal = patterns.Calibration(to_field=[[-1 / 170, 0.0, 190 / 170], [0.0, -1 / 120, 140 / 120]])
        self.h, self.w = 160, 200

    def picture(self, image: np.ndarray | None) -> np.ndarray:
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(float)
        u, v = self.cal.field_of(xx, yy)
        inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        out = np.full((self.h, self.w), 15.0)
        if image is not None:
            r = np.clip((v * (patterns.FIELD_H + 0.5) * 2 - 1).round().astype(int), 0, patterns.ROWS - 1)
            c = np.clip((u * (patterns.FIELD_W + 0.5) - 0.5 - 0.5 * (r % 2)).round().astype(int), 0,
                        patterns.COLUMNS - 1)
            lit = (image[r, c] == self.on) & inside
            out[lit] = 180.0
        return out


@pytest.mark.parametrize("on", [255, 0])
def test_the_light_test_finds_which_byte_lights_and_the_calibration_where_the_field_lies(qapp, on, monkeypatch,
                                                                                         tmp_path):
    from types import SimpleNamespace

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets

    from clampwise.app.screens import polygon_panel
    from clampwise.rig import lucam

    monkeypatch.setattr(polygon_panel, "SETTLE_S", 0.0)
    rig, sdk = SimRig(on), FakeSDK()
    panel = QtWidgets.QWidget()
    panel.box = pg.ViewBox()
    panel.window = SimpleNamespace(say=lambda *_: None)
    panel.folder = lambda: str(tmp_path)
    panel.exposure = SimpleNamespace(value=lambda: 1.0)
    panel._thread = SimpleNamespace(exposure_ms=1.0, average=1)
    panel._shown = 0

    def frame():
        g = rig.picture(sdk.image).astype(np.uint8)
        panel.last = lucam.Frame(np.broadcast_to(g[:, :, None], g.shape + (3,)), 1.0, 1.0)
        panel._shown += 1

    frame()
    dlg = polygon_panel.PolygonPanel(panel, make=lambda: Polygon(sdk, polyscan_running=lambda: False))
    dlg.take()                                                                    # found on its own thread
    assert dlg._opening and dlg.wait() and dlg.polygon.connected and not dlg._opening
    dlg.pulsed.setChecked(False)                                                  # the light on all the time
    dlg.calibrate()
    for _ in range(200):
        if dlg._cal is None:
            break
        dlg.wait()
        frame()
        frame()
        dlg._calibration_tick()
    dlg.wait()
    assert dlg._cal is None and "Calibrated with 5 spots" in dlg.cal_status.text()
    assert dlg.calibration.on_value == on and dlg.calibration.light_tested
    assert np.allclose(dlg.calibration.to_field, rig.cal.to_field, atol=0.01)
    assert sdk.image is not None and not (sdk.image == on).any()                 # left with nothing lit
    # a grid's step 1 projected lands where the field's top-left cell is under the camera
    dlg.source.setCurrentText("Grid")
    dlg.rows.setValue(2)
    dlg.cols.setValue(2)
    dlg.make_steps()
    dlg.list.setCurrentRow(0)
    dlg.project()
    assert dlg.wait()
    frame()
    lit = rig.picture(sdk.image) > 100
    ys, xs = np.nonzero(lit)
    x, y = rig.cal.camera_of(np.array([0.25]), np.array([0.25]))
    assert xs.mean() == pytest.approx(x[0], abs=3) and ys.mean() == pytest.approx(y[0], abs=3)
    assert (tmp_path / "polygon_log.csv").read_text(encoding="utf-8").splitlines()[1].split(",")[1] == "1"
    dlg.release()
    assert dlg.wait() and ("uninit",) in sdk.calls


def test_the_light_test_says_so_when_the_light_is_off(qapp, monkeypatch):
    from types import SimpleNamespace

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets

    from clampwise.app.screens import polygon_panel
    from clampwise.rig import lucam

    monkeypatch.setattr(polygon_panel, "SETTLE_S", 0.0)
    sdk = FakeSDK()
    panel = QtWidgets.QWidget()
    panel.box, panel.window, panel.folder = pg.ViewBox(), SimpleNamespace(say=lambda *_: None), lambda: ""
    panel.exposure, panel._thread, panel._shown = SimpleNamespace(value=lambda: 1.0), SimpleNamespace(
        exposure_ms=1.0, average=1), 0
    dark = np.full((40, 50), 15, np.uint8)
    panel.last = lucam.Frame(np.broadcast_to(dark[:, :, None], (40, 50, 3)), 1.0, 1.0)
    dlg = polygon_panel.PolygonPanel(panel, make=lambda: Polygon(sdk, polyscan_running=lambda: False))
    dlg.take()
    assert dlg.wait()
    dlg.pulsed.setChecked(False)
    dlg.calibrate()
    for _ in range(20):
        if dlg._cal is None:
            break
        dlg.wait()
        panel._shown += 3
        time.sleep(0.005)                                                          # the settling (3 exposures)
        dlg._calibration_tick()
    assert dlg._cal is None and "is the" in dlg.cal_status.text() and "LED" in dlg.cal_status.text()
    assert not dlg.calibration.light_tested


def test_a_calibration_made_at_one_binning_holds_at_another():
    full = patterns.Calibration(to_field=[[0.004, 0.001, -0.1], [-0.0005, 0.006, 0.05]], camera_size=(1944, 1472))
    binned = full.at_size((972, 736))
    xb, yb = np.array([10.0, 400.0, 900.0]), np.array([20.0, 300.0, 700.0])
    xf, yf = 2 * xb + 0.5, 2 * yb + 0.5                                          # a binned pixel's centre
    assert np.allclose(np.column_stack(binned.field_of(xb, yb)), np.column_stack(full.field_of(xf, yf)))
    assert full.at_size((1944, 1472)) is full


def test_finding_the_polygon_never_holds_the_window_up(qapp):
    """Its SDK takes 25 s to find it on the rig: Use here returns at once and the window goes on meanwhile."""
    from types import SimpleNamespace

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

    from clampwise.app.screens import polygon_panel

    class SlowSDK(FakeSDK):
        def MTPLG_InitDevice(self, n):
            time.sleep(2.0)
            return super().MTPLG_InitDevice(n)

    panel = QtWidgets.QWidget()
    panel.box, panel.window, panel.folder = pg.ViewBox(), SimpleNamespace(say=lambda *_: None), lambda: ""
    panel.last, panel._thread, panel._shown = None, None, 0
    dlg = polygon_panel.PolygonPanel(panel, make=lambda: Polygon(SlowSDK(), polyscan_running=lambda: False))
    beats = []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: beats.append(time.monotonic()))
    timer.start(20)
    t0 = time.monotonic()
    dlg.take()
    assert time.monotonic() - t0 < 0.2
    assert dlg.busy() and "Finding the patterned light" in dlg.busy_text.text() and not dlg.cal_button.isEnabled()
    assert dlg.wait(10.0) and dlg.polygon.connected
    assert not dlg.busy() and dlg.cal_button.isEnabled() and dlg.list.isEnabled()      # ready: usable again
    gaps = np.diff(beats)
    assert len(beats) > 40 and gaps.max() < 0.3                                  # the window answered all along
    timer.stop()


def test_lit_and_unlit_can_be_swapped_by_hand_and_the_step_is_sent_again(qapp):
    from types import SimpleNamespace

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets

    from clampwise.app.screens import polygon_panel

    sdk = FakeSDK()
    panel = QtWidgets.QWidget()
    panel.box, panel.window, panel.folder = pg.ViewBox(), SimpleNamespace(say=lambda *_: None), lambda: ""
    panel.last, panel._thread, panel._shown = None, None, 0
    dlg = polygon_panel.PolygonPanel(panel, make=lambda: Polygon(sdk, polyscan_running=lambda: False))
    dlg.take()
    assert dlg.wait()
    dlg.make_steps()                                                              # the full field
    dlg.project()
    assert dlg.wait() and (sdk.image == 255).all()                                # lit by 255, as assumed
    dlg.swap.setChecked(True)
    assert dlg.wait() and (sdk.image == 0).all() and dlg.calibration.on_value == 0  # sent again, lit by 0
    dlg.off()
    assert dlg.wait() and (sdk.image == 255).all()                                # off: every mirror away


def test_the_calibration_waits_for_the_protocol_s_pulses_when_the_light_is_pulsed(qapp, monkeypatch):
    """The LED lit only in the protocol's pulses (one frame in four here): each test picture waits for a pulse and
    reads the frame it lit; one that no pulse lights (its mirrors away) is read dark after the wait."""
    from types import SimpleNamespace

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets

    from clampwise.app.screens import polygon_panel
    from clampwise.rig import lucam

    monkeypatch.setattr(polygon_panel, "SETTLE_S", 0.0)
    monkeypatch.setattr(polygon_panel, "PULSE_WAIT_S", 0.6)
    rig, sdk = SimRig(0), FakeSDK()                                               # lit by 0, as the rig's may be
    panel = QtWidgets.QWidget()
    panel.box, panel.window, panel.folder = pg.ViewBox(), SimpleNamespace(say=lambda *_: None), lambda: ""
    panel.exposure, panel._thread, panel._shown = SimpleNamespace(value=lambda: 1.0), SimpleNamespace(
        exposure_ms=1.0, average=1), 0

    def frame():
        flash = panel._shown % 4 == 3
        g = rig.picture(sdk.image if flash else None).astype(np.uint8)
        panel.last = lucam.Frame(np.broadcast_to(g[:, :, None], g.shape + (3,)), 1.0, 1.0)
        panel._shown += 1

    frame()
    dlg = polygon_panel.PolygonPanel(panel, make=lambda: Polygon(sdk, polyscan_running=lambda: False))
    dlg.take()
    assert dlg.wait() and dlg.pulsed.isChecked()                                  # pulsed unless unticked
    dlg.calibrate()
    end = time.monotonic() + 30
    while dlg._cal is not None and time.monotonic() < end:
        dlg.wait()
        frame()
        time.sleep(0.01)
        dlg._calibration_tick()
    assert dlg._cal is None and "Calibrated with 5 spots" in dlg.cal_status.text(), dlg.cal_status.text()
    assert dlg.calibration.on_value == 0 and dlg.swap.isChecked()
    assert np.allclose(dlg.calibration.to_field, rig.cal.to_field, atol=0.01)


def test_a_region_drawn_over_a_neuron_is_lit_and_follows_it_when_moved(qapp):
    from types import SimpleNamespace

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets

    from clampwise.app.screens import polygon_panel
    from clampwise.rig import lucam

    rig, sdk = SimRig(0), FakeSDK()
    panel = QtWidgets.QWidget()
    panel.box, panel.window, panel.folder = pg.ViewBox(), SimpleNamespace(say=lambda *_: None), lambda: ""
    panel._thread, panel._shown = None, 0
    g = np.zeros((rig.h, rig.w), np.uint8)
    panel.last = lucam.Frame(np.broadcast_to(g[:, :, None], g.shape + (3,)), 1.0, 1.0)
    dlg = polygon_panel.PolygonPanel(panel, make=lambda: Polygon(sdk, polyscan_running=lambda: False))
    dlg.calibration = patterns.Calibration(0, True, rig.cal.to_field, (rig.w, rig.h), 0.1)
    dlg.take()
    assert dlg.wait()
    dlg.source.setCurrentText("Drawn on the picture")
    dlg.add_shape("ellipse")
    roi = dlg.drawn[0]
    roi.setPos((60, 50))
    roi.setSize((30, 30))
    dlg.make_steps()
    dlg.project()
    assert dlg.wait()
    lit = rig.picture(sdk.image) > 100                                            # where the slide is lit
    ys, xs = np.nonzero(lit)
    assert xs.mean() == pytest.approx(75, abs=2) and ys.mean() == pytest.approx(65, abs=2)
    assert lit.sum() == pytest.approx(np.pi * 15 ** 2, rel=0.25)                  # the ellipse, not more
    roi.setPos((120, 90))                                                         # dragged onto another cell
    roi.sigRegionChangeFinished.emit(roi)
    assert dlg.wait()
    ys, xs = np.nonzero(rig.picture(sdk.image) > 100)
    assert xs.mean() == pytest.approx(135, abs=2) and ys.mean() == pytest.approx(105, abs=2)
