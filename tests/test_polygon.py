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



def test_a_map_fitted_to_every_spot_found_beats_three_and_is_checked_on_spots_left_out():
    # spots found a few pixels off (focus, scattered light): three fix the map exactly, their errors and all; eleven
    # average them out. Each spot is checked against the map fitted to the others, never one it helped to make
    true = patterns.Calibration(to_field=[[-1 / 1700, 0.0, 0.8], [0.0, -1 / 1100, 0.9]])
    rng = np.random.default_rng(3)
    uv = np.array([(u, v) for v in (0.35, 0.45, 0.55, 0.65) for u in (0.4, 0.5, 0.6)][:11])
    x, y = true.camera_of(uv[:, 0], uv[:, 1])
    found = np.column_stack([x, y]) + rng.normal(0, 3.0, (len(uv), 2))

    def worst(to_field):
        xx, yy = patterns.Calibration(to_field=to_field).camera_of(uv[:, 0], uv[:, 1])
        return float(np.hypot(xx - x, yy - y).max())

    three, _ = patterns.fit_affine(found[:3], uv[:3])
    every, residual, held = patterns.fit_held_out(found, uv)
    assert worst(every) < worst(three) / 2 and residual > 0
    held_err = np.hypot(*(held - found).T)
    fit_err = np.hypot(*(np.column_stack(patterns.Calibration(to_field=every).camera_of(uv[:, 0], uv[:, 1]))
                         - found).T)
    assert held.shape == (11, 2) and (held_err >= fit_err - 1e-9).all()     # left out: never flattered
    # one spot cut by the picture's edge, its centre pulled 35 px inwards (32 at the rig): left out, the map unharmed
    edge = found.copy()
    edge[1, 0] += 35.0
    to_field, _, held, kept = patterns.fit_without_strays(edge, uv)
    assert kept.sum() == 10 and not kept[1] and worst(to_field) < worst(three) / 2
    assert patterns.fit_without_strays(found, uv)[3].all()                   # none left out when none strays


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

    @staticmethod
    def _mirrors(ptr) -> np.ndarray:
        """What the Polygon does with what it is sent (as measured on the rig): reads a bit a mirror, eight a byte,
        from the start of it, and lights the mirrors whose bit is set (255 here, 0 not)."""
        n = patterns.ROWS * patterns.COLUMNS // 8
        data = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), (n,)).copy()
        bits = np.unpackbits(data.reshape(patterns.ROWS, patterns.COLUMNS // 8), axis=1, bitorder="big")
        return (bits * 255).astype(np.uint8)

    def MTPLG_SetDevStaticImageFromMemory(self, i, ptr, order):
        self.image = self._mirrors(ptr)
        self.calls.append(("image", i, order))
        return 0

    def MTPLG_GetDevRunStatus(self, i, state, timeout):
        state._obj.Temperature = 41
        state._obj.SeqIdx = getattr(self, "seq_idx", 0)
        return 0

    def MTPLG_SetDevPtnSetting(self, i, setting):
        self.setting, self.patterns = setting, {}
        return 0

    def MTPLG_SetDevPtnFromMemory(self, i, k, ptr):
        self.patterns[k] = self._mirrors(ptr).ravel()
        return 0

    def MTPLG_StartPattern(self, i):
        self.calls.append(("start",))
        return 0

    def MTPLG_StopPattern(self, i):
        self.calls.append(("stop",))
        return 0

    def MTPLG_DisconnectDev(self, i):
        self.calls.append(("disconnect", i))
        return 0

    def MTPLG_UnInitDevice(self):
        self.calls.append(("uninit",))
        return 0


def test_a_pattern_reaches_the_mirrors_as_drawn_a_bit_a_mirror():
    # the Polygon reads a bit a mirror, eight a byte (measured on the rig, 2026-10-23): sent a byte a mirror, only the
    # first eighth of an image reached it, as fine stripes over the whole field. The left half lights the left half.
    sdk = FakeSDK()
    p = Polygon(sdk, polyscan_running=lambda: False)
    assert p.open()
    left = np.zeros((patterns.ROWS, patterns.COLUMNS), bool)
    left[:, : patterns.COLUMNS // 2] = True
    assert p.send(patterns.image(left, 255))
    assert (sdk.image[:, : patterns.COLUMNS // 2] == 255).all() and not sdk.image[:, patterns.COLUMNS // 2:].any()
    corner = np.zeros_like(left)
    corner[1000:, 900:] = True                                                     # the last mirrors read too
    assert p.send(patterns.image(corner, 255)) and (sdk.image == 255).sum() == corner.sum()
    assert sdk.image[1139, 911] == 255 and sdk.image[0, 0] == 0
    p.close()


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

    def __init__(self, on: int, to_field=None):
        self.on = on
        self.cal = patterns.Calibration(to_field=to_field or [[-1 / 170, 0.0, 190 / 170], [0.0, -1 / 120, 140 / 120]])
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
    monkeypatch.setattr(polygon_panel, "LOOK_S", 0.0)
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
    assert dlg.led == "auto"                                                      # found out: on all the time
    dlg.calibrate()
    for _ in range(200):
        if dlg._cal is None:
            break
        dlg.wait()
        frame()
        frame()
        dlg._calibration_tick()
    dlg.wait()
    assert dlg._cal is None and "Calibrated with 13 spots" in dlg.cal_status.text()
    assert dlg.calibration.on_value == on and dlg.calibration.light_tested
    assert np.allclose(dlg.calibration.to_field, rig.cal.to_field, atol=0.01)
    assert sdk.image is not None and not (sdk.image == on).any()                 # left with nothing lit
    # a grid's step 1 projected lands where the field's top-left cell is under the camera
    dlg.source.setCurrentText("Grid")
    dlg.grid_on_picture.setChecked(False)                                        # over the whole field
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
    from clampwise.rig import light_record

    log = light_record.read(tmp_path / "polygon_log.csv")
    assert [e.event for e in log][:2] == ["taken", "shown"] and log[1].step == "1" and log[1].lit
    assert log[1].field_fraction == pytest.approx(0.25, abs=0.01)
    # the check grid: spots the fit did not use land where the map puts them
    assert "Check: 8 of 8 spots found" in dlg.cal_status.text() and len(dlg.calibration.check) == 13
    assert dlg.calibration.check_errors_px().max() < 2.0
    assert dlg.calibration.measured and "measured today at" in dlg._cal_text()  # when: the map holds until moved
    dlg.release()
    assert dlg.wait() and ("uninit",) in sdk.calls
    assert not (sdk.image == on).any()                                           # let go with nothing lit
    assert [e.event for e in light_record.read(tmp_path / "polygon_log.csv")][-1] == "let go"


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
    dlg.led = "steady"
    dlg.calibrate()
    for _ in range(20):
        if dlg._cal is None:
            break
        dlg.wait()
        panel._shown += 3
        time.sleep(0.005)                                                          # the settling (3 exposures)
        dlg._calibration_tick()
    assert dlg._cal is None and "Is the LED on" in dlg.cal_status.text()
    assert not dlg.calibration.light_tested


def test_a_calibration_made_at_one_binning_holds_at_another():
    full = patterns.Calibration(to_field=[[0.004, 0.001, -0.1], [-0.0005, 0.006, 0.05]], camera_size=(1944, 1472))
    binned = full.at_size((972, 736))
    xb, yb = np.array([10.0, 400.0, 900.0]), np.array([20.0, 300.0, 700.0])
    xf, yf = 2 * xb + 0.5, 2 * yb + 0.5                                          # a binned pixel's centre
    assert np.allclose(np.column_stack(binned.field_of(xb, yb)), np.column_stack(full.field_of(xf, yf)))
    assert full.at_size((1944, 1472)) is full


def test_letting_go_still_puts_the_mirrors_off_when_python_is_closing(qapp):
    """The worker the calls run on is shut down as Python closes; a release then must still light nothing and let
    the patterned light go (it once left a pattern lit after a script ended)."""
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
    assert dlg.wait() and dlg.polygon.connected
    dlg._pool.shutdown(wait=True)                                                  # as at Python's exit
    dlg.release()
    assert not dlg.polygon.connected and ("uninit",) in sdk.calls
    assert sdk.image is not None and not (sdk.image == dlg.calibration.on_value).any()   # nothing lit


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
    assert dlg.wait() and dlg.led == "auto"                                       # found out: in pulses
    dlg.calibrate()
    assert dlg._cal["pulsed"] is None
    end = time.monotonic() + 30
    while dlg._cal is not None and time.monotonic() < end:
        dlg.wait()
        frame()
        time.sleep(0.01)
        dlg._calibration_tick()
    assert dlg._cal is None and "Calibrated with 13 spots" in dlg.cal_status.text(), dlg.cal_status.text()
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


# ------------------------------------------------------------------ calibrations by objective, areas, the light log

def test_a_calibration_is_kept_for_its_objective_and_refused_at_another():
    cals = patterns.Calibrations()
    key10, key40 = patterns.objective_key(10), patterns.objective_key(40, 1.0)
    assert key40 == "40x, adapter 1x" and patterns.objective_key(None) == ""
    c = cals.at(key10)
    c.on_value, c.light_tested, c.objective = 0, True, key10
    c.to_field = [[0.004, 0.0, 0.1], [0.0, 0.006, 0.2]]
    cals.keep(c)
    assert cals.at(key10).ready and not cals.at(key40).ready
    assert cals.at(key40).on_value == 0 and cals.at(key40).light_tested         # the byte is the Polygon's own
    assert "10x" in cals.refusal(key40) and "calibrate with this objective" in cals.refusal(key40)
    assert cals.refusal(key10) == ""
    c.measured = "2026-10-20 14:05"
    again = patterns.Calibrations.from_dict(cals.to_dict())
    assert again.at(key10).to_field == c.to_field and again.on_value == 0
    assert again.at(key10).measured == "2026-10-20 14:05" == again.at(key10).at_size((10, 10)).measured
    old = patterns.Calibrations.from_dict({"on_value": 0, "light_tested": True, "to_field": c.to_field})
    assert old.on_value == 0 and not old.maps                                     # its objective was not noted


def test_a_pattern_out_of_focus_is_told_from_one_in_focus():
    # half the field lit: half the picture bright when focused; an even glow when not (the rig, 2026-10-22: mean 52
    # of 105 over 81% of the picture, at 2.5x on a mirror), so no calibration spot can show
    sharp = np.zeros((100, 120), np.float32)
    sharp[:, :60] = 105.0
    glow = np.full((100, 120), 52.0, np.float32) + np.random.default_rng(1).normal(0, 2, (100, 120))
    assert patterns.sharpness(sharp) > 0.9 and patterns.sharpness(glow) < patterns.SHARP


def test_how_old_a_calibration_is_said_and_an_old_one_is_to_be_checked():
    import datetime as dt

    now = dt.datetime(2026, 10, 22, 9, 30)
    assert patterns.age_words("2026-10-22 08:15", now) == "today at 08:15"
    assert patterns.age_words("2026-10-21 17:00", now) == "yesterday"
    assert patterns.age_words("2026-10-19 17:00", now).startswith("3 days ago (check it again")
    assert patterns.age_words("", now) == "" and patterns.age_words("not a date", now) == ""


def test_the_check_grid_and_the_scale_follow_the_binning():
    full = patterns.Calibration(to_field=[[0.004, 0.0, 0.0], [0.0, 0.006, 0.0]], camera_size=(1944, 1472),
                                check=[[100.5, 200.5, 102.5, 200.5]], um_per_px=0.1)
    binned = full.at_size((972, 736))
    assert binned.um_per_px == pytest.approx(0.2)
    assert binned.check_errors_px()[0] * binned.um_per_px == pytest.approx(full.check_errors_px()[0] * 0.1)


def test_a_step_s_area_on_the_slice_is_worked_out_through_the_calibration():
    cal = patterns.Calibration(to_field=[[1 / 1000, 0.0, 0.0], [0.0, 1 / 625, 0.0]])   # the field: 1000 x 625 px
    um = 0.5                                                                     # so 500 x 312.5 um
    whole = patterns.full_field()[0]
    d = patterns.dose(whole, whole.mirrors(), cal, um)
    assert d["field_fraction"] == pytest.approx(1.0) and d["area_um2"] == pytest.approx(500 * 312.5, rel=1e-3)
    spot = patterns.spot(0.5, 0.5, 0.04)[0]                                      # 40 px = 20 um across
    d = patterns.dose(spot, spot.mirrors(), cal, um)
    assert d["area_um2"] == pytest.approx(np.pi * 10 ** 2, rel=0.05)
    assert d["shapes"][0]["area_um2"] == pytest.approx(np.pi * 10 ** 2, rel=0.01)
    assert d["shapes"][0]["centre_um"] == pytest.approx([250, 156.25])
    assert patterns.dose(spot, spot.mirrors(), patterns.Calibration(), 0.0)["area_um2"] is None


def test_the_light_log_is_timed_to_the_millisecond_and_an_older_log_is_kept_aside(tmp_path):
    from clampwise.rig import light_record

    (tmp_path / "polygon_log.csv").write_text("time,step,shapes (field: 0-1 across, down),pattern\n"
                                              "2026-10-19 10:00:00,1,\"rect(0, 0, 1, 1)\",Grid\n", encoding="utf-8")
    old = light_record.read(tmp_path / "polygon_log.csv")
    assert old[0].coarse and old[0].step == "1" and old[0].event == "shown"
    step = patterns.spot(0.5, 0.5, 0.1)[0]
    e = light_record.entry("shown", step, "Spot", {"field_fraction": 0.01, "area_um2": 78.5, "shapes": [{"k": 1}]},
                           "40x, adapter 1x", 0, when=1_790_000_000.123)
    light_record.write(tmp_path, e)
    assert (tmp_path / "polygon_log (before 1).csv").exists()
    back = light_record.read(tmp_path / "polygon_log.csv")
    assert len(back) == 1 and back[0].unix_s == pytest.approx(1_790_000_000.123, abs=1e-3)
    assert back[0].step == "Spot" and back[0].area_um2 == 78.5 and back[0].on_value == 0 and back[0].lit
    assert back[0].shapes == [{"k": 1}] and back[0].objective == "40x, adapter 1x"


def test_each_sweep_gets_the_pattern_shown_and_how_sure_that_is():
    import datetime as dt

    from clampwise.rig import light_record as lr

    t0 = dt.datetime(2026, 10, 26, 14, 0, 0)
    s = t0.timestamp()
    roi = patterns.Step("Cell 1", [patterns.Shape("ellipse", (0.4, 0.4, 0.5, 0.5))])
    log = [lr.entry("taken", when=s - 60), lr.entry("shown", roi, when=s - 30, dose={"field_fraction": 0.01}),
           lr.entry("shown", patterns.full_field()[0], when=s + 10.5, dose={"field_fraction": 1.0}),
           lr.entry("off", when=s + 25.0), lr.entry("let go", when=s + 40)]
    # sweeps of 2 s every 5 s: 0, 5, 10 (the change at 10.5 s within it), 15, 20 (Off at 25.0: 3 s after it), 45
    starts = [0, 5, 10, 15, 20, 45]
    out = lr.match(log, t0, starts, 2.0, clock_s=1.0, pulses_s=[[0.375, 0.475]] * 3 + [[0.375]] * 3)
    assert [o["step"] for o in out] == ["Cell 1", "Cell 1", "Cell 1", "Full field", "Full field", None]
    assert out[0]["sure"] and out[1]["sure"] and not out[2]["sure"] and "changed during" in out[2]["why"]
    assert out[2]["pulses"][0]["step"] == "Cell 1" and not out[2]["pulses"][0]["sure"]  # 0.125 s from the change
    assert out[3]["sure"] and out[4]["sure"] and out[4]["margin_s"] == pytest.approx(3.0)
    assert not out[5]["sure"] and "let go" in out[5]["why"]
    tight = lr.match(log, t0, [8.0], 2.0, clock_s=1.0)                           # ends 0.5 s before the change
    assert not tight[0]["sure"] and "uncertainty" in tight[0]["why"]
    groups = lr.by_step(out)
    assert groups == {"Cell 1": [1, 2], "not sure": [3, 6], "Full field": [4, 5]}


def test_comparing_a_full_field_with_a_region_says_what_differs():
    from clampwise.rig import light_record as lr

    sweeps = [{"sweep": 1, "sure": True, "lit": True, "step": "Full field", "field_fraction": 1.0,
               "area_um2": 150000.0},
              {"sweep": 2, "sure": True, "lit": True, "step": "Cell 1", "field_fraction": 0.002, "area_um2": 300.0}]
    notes = lr.dose_notes(sweeps)
    assert len(notes) == 1 and "500x" in notes[0] and "irradiance is the same" in notes[0]
    assert lr.dose_notes(sweeps[:1]) == []


def test_a_recording_s_sweeps_are_matched_once_and_kept_beside_it(tmp_path):
    import datetime as dt
    import json
    from types import SimpleNamespace

    from clampwise.rig import light_record as lr

    t0 = dt.datetime(2026, 10, 26, 14, 0, 0)
    lr.write(tmp_path, lr.entry("shown", patterns.full_field()[0], when=t0.timestamp() - 5,
                                dose={"field_fraction": 1.0}))
    rec = SimpleNamespace(source=tmp_path / "26o26cw000.abf", start_time=t0, sweep_start_times_s=np.array([0.0, 5.0]),
                          sweep_duration_s=2.0, stimulus=None)
    out = lr.for_recording(rec)
    assert [o["step"] for o in out] == ["Full field", "Full field"] and all(o["sure"] for o in out)
    side = json.loads((tmp_path / "26o26cw000.light.json").read_text(encoding="utf-8"))
    assert side["clock_uncertainty_s"] == lr.OWN_CLOCK_S and len(side["sweeps"]) == 2
    (tmp_path / lr.LOG_NAME).unlink()
    assert lr.for_recording(rec) == out                                          # read back from beside it
    rec.source = tmp_path / "other.abf"
    assert lr.for_recording(rec) is None                                         # no log, no sidecar


def test_evoked_responses_are_measured_for_each_region_lit(tmp_path):
    """Sweeps alternating between a region on the cell and its control beside it (a fifth of the response,
    scattered light): the evoked analysis measures each pattern's sweeps on their own."""
    from clampwise import simulate
    from clampwise.analysis import pipeline
    from clampwise.rig import light_record as lr

    sim = simulate.evoked(onsets_s=(0.2,), amplitudes=(-50.0,), n_sweeps=20, seed=3)
    rec = sim.recording
    rec.source = tmp_path / "26101cw000.abf"                                     # Clampwise's own: its clock
    signal = rec.samples._arrays[0]
    signal[1::2] = -20.0 + (signal[1::2] + 20.0) * 0.2
    t0 = rec.start_time.timestamp()
    cell = patterns.Step("Cell 1", [patterns.Shape("ellipse", (0.4, 0.4, 0.45, 0.47))])
    ctrl = patterns.Step("Cell 1 control", [patterns.Shape("ellipse", (0.6, 0.4, 0.65, 0.47))])
    for k, start in enumerate(rec.sweep_start_times_s):
        lr.write(tmp_path, lr.entry("shown", ctrl if k % 2 else cell, when=t0 + start - 0.5,
                                    dose={"field_fraction": 0.002, "area_um2": 180.0}))
    e = pipeline.analyze(rec).results["evoked"]
    assert not e.error, e.error
    rows = {r["pattern"]: r for r in e.tables["by_pattern"]}
    assert set(rows) == {"Cell 1", "Cell 1 control"} and rows["Cell 1"]["sweeps"] == 10
    assert rows["Cell 1"]["p1_amplitude"] == pytest.approx(-50, rel=0.15)
    assert rows["Cell 1 control"]["p1_amplitude"] == pytest.approx(-10, rel=0.3)
    assert not (tmp_path / "26101cw000.light.json").exists()                     # the analysis writes nothing


def test_the_steps_can_be_stepped_by_the_polygon_s_own_trigger_and_off_stops_them():
    from clampwise.rig import polygon

    sdk = FakeSDK()
    p = Polygon(sdk, polyscan_running=lambda: False)
    assert p.open()
    a, b = patterns.grid(1, 2)
    assert p.run_sequence([a.mirrors(), b.mirrors()], on_value=0, shown_us=20_000)
    s = sdk.setting
    assert (s.PtnNumber, s.TrigType, s.ExposureTime, s.TrigEdge, s.BitDepth) == (2, polygon.TRIGGER_EXTERNAL, 20_000,
                                                                                 polygon.EDGE_RISE, 1)
    assert sdk.mode == polygon.DISPLAY_SEQUENCE and ("start",) in sdk.calls and p.sequence
    first = sdk.patterns[1].reshape(patterns.ROWS, patterns.COLUMNS)
    assert (first[:, :400] == 0).all() and (first[:, 520:] == 255).all()          # lit by 0: the left half
    assert p.sequence_index() == 0
    assert p.off(0) and ("stop",) in sdk.calls and sdk.mode == polygon.DISPLAY_STATIC and not p.sequence
    assert (sdk.image == 255).all()                                              # still, nothing lit


def test_each_pulse_stepped_by_the_trigger_is_given_its_own_region(tmp_path):
    from clampwise import simulate
    from clampwise.analysis import pipeline
    from clampwise.rig import light_record as lr

    sim = simulate.evoked(onsets_s=(0.2, 0.3), amplitudes=(-50.0, -10.0), n_sweeps=10, seed=4)
    rec = sim.recording
    rec.source = tmp_path / "26101cw001.abf"
    e = lr.entry("sequence", None, "Drawn on the picture", {"field_fraction": 0.002}, "40x, adapter 1x", 255,
                 when=rec.start_time.timestamp() - 5)
    e.step, e.shapes = "Cell 1 | Cell 1 control", [{"step": "Cell 1"}, {"step": "Cell 1 control"}]
    lr.write(tmp_path, e)
    from clampwise.analysis import stimulus as finder

    out = lr.for_recording(rec, write_it=False, stimulus=finder.find_stimulus(rec))
    assert all(o["sure"] for o in out)
    assert [p["step"] for p in out[0]["pulses"]] == ["Cell 1", "Cell 1 control"] and out[0]["pulses"][1]["by_trigger"]
    ev = pipeline.analyze(rec).results["evoked"]
    assert [r["pattern"] for r in ev.tables["pulses"]] == ["Cell 1", "Cell 1 control"]
    assert any("pulse 2 Cell 1 control" in n for n in ev.notes)


def test_projecting_on_triggers_loads_every_step_and_logs_them_as_one_sequence(qapp, tmp_path):
    from clampwise.app.screens import polygon_panel
    from clampwise.rig import light_record as lr

    sdk = FakeSDK()
    panel = _panel_with_scale(tmp_path)
    dlg = polygon_panel.PolygonPanel(panel, make=lambda: Polygon(sdk, polyscan_running=lambda: False))
    dlg.take()
    assert dlg.wait()
    dlg.source.setCurrentText("Grid")
    dlg.rows.setValue(1)
    dlg.cols.setValue(3)
    dlg.make_steps()
    dlg.on_trigger.setChecked(True)
    dlg.held_ms.setValue(30.0)
    dlg.project()
    assert dlg.wait() and len(sdk.patterns) == 3 and sdk.setting.ExposureTime == 30_000
    seq = [e for e in lr.read(tmp_path / lr.LOG_NAME) if e.event == "sequence"]
    assert len(seq) == 1 and seq[0].step == "1 | 2 | 3" and [d["step"] for d in seq[0].shapes] == ["1", "2", "3"]
    dlg.off()
    assert dlg.wait() and not dlg.polygon.sequence and not sdk.image.any()


def _panel_with_scale(tmp_path, um=0.5, mag=40.0):
    from types import SimpleNamespace

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets

    from clampwise.rig import lucam

    panel = QtWidgets.QWidget()
    panel.box, panel.window, panel.folder = pg.ViewBox(), SimpleNamespace(say=lambda *_: None), lambda: str(tmp_path)
    panel._thread, panel._shown = None, 0
    panel.magnification = mag
    panel.scale = lambda: (um, "calibrated")
    adapter = QtWidgets.QDoubleSpinBox()
    adapter.setValue(1.0)
    panel.dialog = SimpleNamespace(adapter=adapter)
    g = np.zeros((160, 200), np.uint8)
    panel.last = lucam.Frame(np.broadcast_to(g[:, :, None], g.shape + (3,)), 1.0, 1.0)
    return panel


def test_a_cell_s_region_is_placed_by_its_centre_and_diameter_with_a_control_beside_it(qapp, tmp_path):
    import json

    from clampwise.app.screens import polygon_panel

    panel = _panel_with_scale(tmp_path)
    dlg = polygon_panel.PolygonPanel(panel, make=lambda: Polygon(FakeSDK(), polyscan_running=lambda: False))
    rig = SimRig(255)
    dlg.calibration = patterns.Calibration(255, True, rig.cal.to_field, (rig.w, rig.h), 0.1, "40x, adapter 1x")
    dlg.source.setCurrentText("Drawn on the picture")
    dlg.roi_name.setText("Cell 1")
    dlg.roi_um.setValue(15.0)                                                    # 30 px at 0.5 um a pixel
    roi = dlg.place_at(80.0, 60.0)
    assert roi.name == "Cell 1" and tuple(roi.size()) == pytest.approx((30, 30))
    assert dlg._centre(roi) == pytest.approx((80, 60))
    dlg.control_um.setValue(40.0)                                                # 80 px to its right
    ctrl = dlg.add_control()
    assert ctrl.name == "Cell 1 control" and dlg._centre(ctrl) == pytest.approx((160, 60))
    assert [s.name for s in dlg.steps] == ["Cell 1", "Cell 1 control"]
    path = dlg.save_regions()
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["regions"][0]["centre_um"] == [40.0, 30.0] and body["regions"][1]["control"]
    dlg.clear_shapes()
    assert dlg.load_regions() == 2 and dlg._centre(dlg.drawn[1]) == pytest.approx((160, 60))
    assert getattr(dlg.drawn[1], "control", False) and dlg.drawn[0].name == "Cell 1"


def test_another_objective_uses_its_own_calibration_or_says_why_not(qapp, tmp_path):
    from clampwise.app.screens import polygon_panel

    panel = _panel_with_scale(tmp_path, mag=10.0)
    dlg = polygon_panel.PolygonPanel(panel, make=lambda: Polygon(FakeSDK(), polyscan_running=lambda: False))
    rig = SimRig(255)
    c = patterns.Calibration(255, True, rig.cal.to_field, (rig.w, rig.h), 0.1, dlg._objective())
    dlg.calibrations.keep(c)
    dlg._use_objective()
    assert dlg.calibration.ready and "Calibrated at 10x" in dlg.cal_status.text()
    panel.magnification = 40.0
    dlg._use_objective()
    assert not dlg.calibration.ready and "calibrated at 10x, adapter 1x, not at 40x" in dlg.cal_status.text()
    dlg.source.setCurrentText("Drawn on the picture")
    dlg.add_shape("ellipse")
    dlg.make_steps()
    assert not dlg.steps and "calibrate with this objective" in dlg.pattern_note.text()


def test_a_recording_s_end_leaves_nothing_lit(qapp, tmp_path):
    from clampwise.app.screens import polygon_panel

    sdk = FakeSDK()
    panel = _panel_with_scale(tmp_path)
    dlg = polygon_panel.PolygonPanel(panel, make=lambda: Polygon(sdk, polyscan_running=lambda: False))
    dlg.take()
    assert dlg.wait()
    dlg.make_steps()                                                             # the full field
    dlg.project()
    assert dlg.wait() and (sdk.image == 255).all() and dlg.off_after.isChecked()
    dlg.after_run([])
    assert dlg.wait() and not sdk.image.any()
    dlg.off_after.setChecked(False)
    dlg.project()
    dlg.after_run([])
    assert dlg.wait() and (sdk.image == 255).all()                               # unticked: left as it is


# ------------------------------------------------------------------ a field larger than the picture; cells chosen

# the rig's case: every mirror lit covers the whole picture; the picture sees the middle 0.4 of the field, mirrored
OVERFILLED = [[-0.002, 0.0, 0.7], [0.0, 0.0025, 0.3]]


def test_steps_lit_together_a_grid_over_the_picture_and_what_overfills_it():
    cal = patterns.Calibration(to_field=OVERFILLED, camera_size=(200, 160))
    assert patterns.picture_outside_field(cal, (200, 160)) is False
    assert patterns.picture_outside_field(patterns.Calibration(to_field=SimRig(255).cal.to_field), (200, 160))
    cells = patterns.grid_on_picture(10, 10, cal, (200, 160))
    assert [c.name for c in cells[:3]] == ["1", "2", "3"] and len(cells) == 100
    u, v = cal.field_of(np.array([130.0]), np.array([88.0]))                     # row 6, column 7 of the picture
    assert [c.name for c in cells if c.covers(u[0], v[0])] == ["57"]
    both = patterns.together([cells[0], cells[56]])
    assert both.name == "1 + 57" and len(both.shapes) == 2
    assert both.mirrors().sum() == cells[0].mirrors().sum() + cells[56].mirrors().sum()
    image = patterns.Step("picture", mask=np.zeros((patterns.ROWS, patterns.COLUMNS), bool))
    image.mask[:10] = True
    assert patterns.together([image, cells[0]]).mirrors().sum() == image.mask.sum() + cells[0].mirrors().sum()
    assert patterns.spread_px([[0, 0], [10, 10], [20, 20]]) < 1e-6 < patterns.spread_px([[0, 0], [10, 0], [0, 10]])


def _sim_panel(rig, sdk, tmp_path):
    from types import SimpleNamespace

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets

    from clampwise.app.screens import polygon_panel
    from clampwise.rig import lucam

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
    dlg.take()
    assert dlg.wait()
    return dlg, frame


def test_the_calibration_finds_a_field_larger_than_the_picture(qapp, monkeypatch, tmp_path):
    from clampwise.app.screens import polygon_panel

    monkeypatch.setattr(polygon_panel, "SETTLE_S", 0.0)
    rig, sdk = SimRig(255, OVERFILLED), FakeSDK()
    dlg, frame = _sim_panel(rig, sdk, tmp_path)
    dlg.led = "steady"
    dlg.calibrate()
    for _ in range(300):
        if dlg._cal is None:
            break
        dlg.wait()
        frame()
        frame()
        dlg._calibration_tick()
    dlg.wait()
    text = dlg.cal_status.text()
    assert dlg._cal is None and "Calibrated with 13 spots" in text, text     # the centre, a closer ring of 4, the check
    assert np.allclose(dlg.calibration.to_field, OVERFILLED, atol=0.002)
    assert "Check: 8 of 8 spots found" in text and dlg.calibration.check_errors_px().max() < 2.0
    assert not (sdk.image == 255).any()
    dlg.release()
    assert dlg.wait()


def test_cells_of_a_grid_chosen_on_the_picture_are_lit_together_and_nothing_else(qapp, tmp_path):
    from clampwise.rig import light_record as lr

    rig, sdk = SimRig(255, OVERFILLED), FakeSDK()
    dlg, frame = _sim_panel(rig, sdk, tmp_path)
    dlg.calibration = patterns.Calibration(255, True, OVERFILLED, (rig.w, rig.h), 0.1)
    dlg.show()
    dlg.source.setCurrentText("Grid")
    dlg.make_steps()                                                             # 10 x 10 over the picture
    assert len(dlg.steps) == 100 and "Over the camera's picture" in dlg.pattern_note.text()
    assert dlg.step_at(10, 8) == 0 and dlg.step_at(130, 88) == 56
    dlg.choose_at(0)                                                             # a click on cell 1
    dlg.choose_at(56, add=True)                                                  # Ctrl+click on cell 57
    assert dlg._chosen_rows() == [0, 56]
    dlg.project()
    assert dlg.wait()
    lit = rig.picture(sdk.image) > 100
    cell1, cell57 = np.zeros_like(lit), np.zeros_like(lit)
    cell1[0:16, 0:20], cell57[80:96, 120:140] = True, True
    near = np.zeros_like(lit)
    near[0:17, 0:21], near[79:97, 119:141] = True, True
    assert lit[cell1].mean() > 0.8 and lit[cell57].mean() > 0.8                  # both cells lit
    assert not lit[~near].any()                                                  # and nothing else
    assert dlg._lit == {"1", "57"}
    filled = [i for i in dlg.outline_items if type(i).__name__ == "QGraphicsPolygonItem"]
    assert len(filled) == 2                                                      # drawn as lit on the picture
    shown = [e for e in lr.read(tmp_path / lr.LOG_NAME) if e.event == "shown"][-1]
    assert shown.step == "1 + 57" and [d["step"] for d in shown.shapes] == ["1", "57"]
    assert shown.field_fraction == pytest.approx(2 * 0.04 * 0.04, rel=0.15)     # two cells of a 0.4 x 0.4 picture
    dlg.choose_at(56, add=True)                                                  # Ctrl+click again: taken out
    assert dlg._chosen_rows() == [0]
    dlg.off()
    assert dlg.wait() and dlg._lit == set() and not (sdk.image == 255).any()
    dlg.release()
    assert dlg.wait()


def test_an_image_step_is_laid_over_the_picture_where_its_mirrors_light(qapp, tmp_path):
    from pyqtgraph.Qt import QtCore

    rig, sdk = SimRig(255, OVERFILLED), FakeSDK()
    dlg, _ = _sim_panel(rig, sdk, tmp_path)
    dlg.calibration = patterns.Calibration(255, True, OVERFILLED, (rig.w, rig.h), 0.1)
    dlg.show()
    mask = np.zeros((patterns.ROWS, patterns.COLUMNS), bool)
    mask[::2, ::2] = True
    dlg.steps = [patterns.Step("picture", mask=mask)]
    dlg.list.addItems(["picture"])
    dlg.list.setCurrentRow(0)
    images = [i for i in dlg.outline_items if type(i).__name__ == "ImageItem"]
    assert len(images) == 1
    for (c, r), (u, v) in (((0, 0), (0.0, 0.0)), ((456, 570), (1.0, 1.0)), ((228, 285), (0.5, 0.5))):
        p = images[0].mapToParent(QtCore.QPointF(c, r))
        x, y = dlg.calibration.camera_of(np.array([u]), np.array([v]))
        assert (p.x(), p.y()) == pytest.approx((x[0], y[0]), abs=0.5)
    dlg.release()
    assert dlg.wait()


def test_a_plain_click_or_next_after_several_chosen_lights_that_step_alone(qapp, tmp_path):
    """Found by the end-to-end check of 1.6.0: after a Ctrl+click, a plain click, a double-click or Next kept the
    earlier steps chosen and lit them too (1 + 57 chosen, Next lit 1 + 58)."""
    import pyqtgraph as pg

    rig, sdk = SimRig(255, OVERFILLED), FakeSDK()
    dlg, _ = _sim_panel(rig, sdk, tmp_path)
    dlg.calibration = patterns.Calibration(255, True, OVERFILLED, (rig.w, rig.h), 0.1)
    picture = pg.ImageItem(np.zeros((rig.h, rig.w), np.uint8))
    dlg.panel.box.addItem(picture)
    dlg.show()
    dlg.source.setCurrentText("Grid")
    dlg.make_steps()
    dlg.choose_at(0)
    dlg.choose_at(56, add=True)
    dlg.choose_at(44)                                                            # a plain click on cell 45
    assert dlg._chosen_rows() == [44]
    dlg.choose_at(0)
    dlg.choose_at(56, add=True)
    dlg.step_by(1)                                                               # Next
    assert dlg.wait() and dlg._chosen_rows() == [57] and dlg._lit == {"58"}
    lit = rig.picture(sdk.image) > 100
    assert not lit[0:17, 0:21].any()                                             # cell 1 no longer lit
    # the camera's picture enlarged hides this panel: what is lit stays drawn while the patterned light is held
    dlg.hide()
    assert any(type(i).__name__ == "QGraphicsPolygonItem" for i in dlg.outline_items)
    # Fit fits the picture, not the field larger than it (the field's edge is not counted)
    dlg.panel.box.autoRange(padding=0)
    (x0, x1), (y0, y1) = dlg.panel.box.viewRange()
    assert x1 - x0 < 1.2 * rig.w or y1 - y0 < 1.2 * rig.h
    dlg.release()
    assert dlg.wait() and not dlg.outline_items                                  # let go: nothing drawn


def test_the_field_s_frame_moved_turned_or_mirrored_moves_the_map_with_it():
    cal = SimRig(255).cal                                                        # turned round and mirrored
    frame = patterns.frame_of(cal)
    u, v = np.array([0.0, 1.0, 0.3, 0.5]), np.array([0.0, 0.2, 1.0, 0.5])
    x0, y0 = cal.camera_of(u, v)
    assert np.allclose(patterns.moved(cal.to_field, frame, frame), cal.to_field)
    cx, cy, angle, w, h = frame
    assert (cx, cy) == pytest.approx(tuple(float(a[0]) for a in cal.camera_of(np.array([0.5]), np.array([0.5]))))
    shifted = patterns.Calibration(to_field=patterns.moved(cal.to_field, frame, (cx + 7, cy - 3, angle, w, h)))
    x, y = shifted.camera_of(u, v)
    assert np.allclose(x, x0 + 7) and np.allclose(y, y0 - 3)                    # every point moved with it
    turned = patterns.Calibration(to_field=patterns.moved(cal.to_field, frame, (cx, cy, angle + 90, w, h)))
    x, y = turned.camera_of(u, v)
    assert np.allclose(x - cx, -(y0 - cy)) and np.allclose(y - cy, x0 - cx)     # a quarter turn about the centre
    larger = patterns.Calibration(to_field=patterns.moved(cal.to_field, frame, (cx, cy, angle, 2 * w, 2 * h)))
    assert larger.field_area_px() == pytest.approx(4 * cal.field_area_px())
    across = patterns.flipped(cal.to_field, frame, across=True)
    x, y = patterns.Calibration(to_field=across).camera_of(1 - u, v)
    assert np.allclose(x, x0) and np.allclose(y, y0)                             # left for right
    assert np.allclose(patterns.flipped(across, frame, across=True), cal.to_field)
    down = patterns.Calibration(to_field=patterns.flipped(cal.to_field, frame, across=False))
    x, y = down.camera_of(u, 1 - v)
    assert np.allclose(x, x0) and np.allclose(y, y0)                             # top for bottom
    # no map yet: the field unturned in the middle of the picture, as large as fits
    guess = patterns.Calibration(to_field=patterns.guessed((200, 160)))
    x, y = guess.camera_of(np.array([0.0, 1.0, 0.5]), np.array([0.0, 1.0, 0.5]))
    assert (x[2], y[2]) == pytest.approx((100, 80)) and x[1] - x[0] == pytest.approx(200)
    assert (y[1] - y[0]) / (x[1] - x[0]) == pytest.approx(patterns.FIELD_H / patterns.FIELD_W)


def test_the_alignment_target_shows_which_way_round_the_field_lies():
    t = patterns.alignment_target()
    lit = lambda u, v: t.covers(u, v)                                            # noqa: E731
    assert lit(0.62, 0.5) and not lit(0.38, 0.5)                                 # the long arm along u, not back
    assert lit(0.5, 0.40) and not lit(0.5, 0.60)                                 # the short arm up, not down
    assert not lit(0.5, 0.30) and lit(0.66, 0.5)                                 # one arm longer than the other
    assert lit(0.001, 0.5) and lit(0.5, 0.999) and lit(0.15, 0.15) and lit(0.85, 0.85)   # the edge and spots
    m = t.mirrors()
    assert 0.02 < m.mean() < 0.15                                                # a target, not a full field


def test_the_field_placed_by_hand_is_dragged_onto_the_target_kept_or_put_back(qapp, tmp_path):
    from clampwise.app.screens import polygon_panel

    sdk, rig = FakeSDK(), SimRig(255)
    panel = _panel_with_scale(tmp_path)
    dlg = polygon_panel.PolygonPanel(panel, make=lambda: Polygon(sdk, polyscan_running=lambda: False))
    dlg.take()
    assert dlg.wait()
    frame = patterns.frame_of(rig.cal)
    cx, cy, angle, w, h = frame
    off = patterns.moved(rig.cal.to_field, frame, (cx + 12, cy, angle, w, h))   # moved since: 12 px to the right
    old = patterns.Calibration(255, True, off, (rig.w, rig.h), 0.2, dlg._objective(), [[1, 1, 1, 1]], 0.5,
                               "2026-10-01 09:00")
    dlg.calibrations.keep(old)
    dlg._use_objective()
    assert dlg.adjust() and dlg.wait()
    assert dlg._lit == {"alignment target"} and (sdk.image == 255).sum() == dlg._target.mirrors().sum()
    assert "not kept yet" in dlg.cal_ready.text() and dlg.calibrations.maps[dlg._objective()] is old
    assert any(isinstance(i, pg_curve()) for i in dlg.outline_items)             # the target drawn through the map
    roi = dlg._adjust["roi"]
    assert roi.parentItem() is panel.box.childGroup and dlg._roi_frame() == pytest.approx(dlg._adjust["frame"])
    roi.setPos(roi.pos() + pg_point(-12, 0))                                     # dragged back onto the lit target
    assert np.allclose(dlg.calibration.to_field, rig.cal.to_field, atol=1e-9)
    dlg.end_adjust(keep=True)
    assert dlg.wait() and not (sdk.image == 255).any() and roi.parentItem() is None
    kept = dlg.calibrations.maps[dlg._objective()]
    assert kept is dlg.calibration and kept.by_hand and not kept.check and kept.measured != old.measured
    assert "Placed by hand at 40x" in dlg.cal_ready.text() and "Placed by hand" in dlg._cal_text()
    assert patterns.Calibration.from_dict(kept.to_dict()).by_hand                # kept between sessions
    # the buttons: a nudge, a turn and a mirror, then Cancel puts back the map in use
    assert dlg.adjust()
    dlg.nudge(dx=dlg.nudge_step(), turn=polygon_panel.NUDGE_DEG, grow=1 + polygon_panel.NUDGE)
    dlg.flip(across=True)
    assert not np.allclose(dlg.calibration.to_field, kept.to_field)
    dlg.adjust_from_start()
    assert np.allclose(dlg.calibration.to_field, kept.to_field)
    dlg.nudge(dy=5)
    dlg.end_adjust(keep=False)
    assert dlg.calibration is kept and np.allclose(kept.to_field, rig.cal.to_field, atol=1e-9)
    # another objective, never calibrated: placing starts from a guess in the middle of the picture
    panel.magnification = 10.0
    dlg._use_objective()
    assert not dlg.calibration.ready and dlg.adjust() and dlg.calibration.ready
    assert dlg._roi_frame()[:2] == pytest.approx((rig.w / 2, rig.h / 2))
    panel.magnification = 40.0
    dlg._use_objective()                                                         # the objective changed: not kept
    assert dlg._adjust is None and dlg.calibration is kept and "10x, adapter 1x" not in dlg.calibrations.maps
    dlg.release()
    assert dlg.wait()


def pg_curve():
    import pyqtgraph as pg

    return pg.PlotCurveItem


def pg_point(x, y):
    from pyqtgraph.Qt import QtCore

    return QtCore.QPointF(x, y)
