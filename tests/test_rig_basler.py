"""A Basler camera without pylon Viewer: taken only on request and never while pylon Viewer is open, frames from its
free-running stream (the newest each time, grey or RGB, averaged for dim fluorescence), what was changed on the camera
put back when it is let go. A stand-in plays pylon's C library as its header declares it (tests never load the real
one); which camera the rig has is told by its USB ID."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from clampwise.rig import basler, cameras, hardware, lucam, picture

OK = 0
FAIL = -2147483648 + 5                  # an HRESULT failure, as pylon returns them


class FakePylon:
    """pylon's C library as a stand-in: one 8 x 6 ace whose frames count up, a few each time it is waited for."""

    def __init__(self, pixel_format="Mono12", formats=("Mono8", "Mono12"), opens=True) -> None:
        self.calls: list[str] = []
        self.opens, self.formats = opens, formats
        self.features = {"DeviceModelName": "acA1920-40um", "DeviceSerialNumber": "12345678",
                         "PixelFormat": pixel_format, "ExposureAuto": "Continuous", "GainAuto": "Off",
                         "AcquisitionMode": "Continuous", "TriggerSelector": "FrameStart", "TriggerMode": "On",
                         "AcquisitionFrameRateEnable": "1", "ExposureTime": 10000.0, "Gain": 0.0,
                         "Width": 8, "Height": 6, "ResultingFrameRate": 41.0}
        self.ranges = {"ExposureTime": (34.0, 10_000_000.0), "Gain": (0.0, 24.0)}
        self.buffers: dict[int, tuple[int, int]] = {}            # handle -> (address, size)
        self.queued: list[tuple[int, int]] = []                  # (buffer handle, context)
        self.output: list[tuple[int, int, int]] = []             # (buffer handle, context, status)
        self.level, self.burst, self.started = 0, 1, False

    # ---- the system and the device
    def PylonInitialize(self):
        self.calls.append("init")
        return OK

    def PylonTerminate(self):
        self.calls.append("terminate")
        return OK

    def PylonEnumerateDevices(self, n):
        n._obj.value = 1
        return OK

    def PylonCreateDeviceByIndex(self, index, handle):
        handle._obj.value = 77
        return OK

    def PylonDeviceOpen(self, handle, mode):
        self.calls.append("open")
        assert mode == basler.ACCESS
        return OK if self.opens else FAIL

    def PylonDeviceClose(self, handle):
        self.calls.append("close")
        return OK

    def PylonDestroyDevice(self, handle):
        self.calls.append("destroy")
        return OK

    # ---- features
    def PylonDeviceFeatureIsAvailable(self, handle, name):
        return name.decode() in self.features or name.decode() == "PayloadSize"

    def PylonDeviceFeatureIsReadable(self, handle, name):
        return self.PylonDeviceFeatureIsAvailable(handle, name)

    def PylonDeviceFeatureIsWritable(self, handle, name):
        return name.decode() in self.features and not (self.started and name.decode() in ("PixelFormat", "Width"))

    def PylonDeviceFeatureToString(self, handle, name, buf, n):
        value = str(self.features[name.decode()]).encode()
        ctypes.memmove(buf, value + b"\0", len(value) + 1)
        return OK

    def PylonDeviceFeatureFromString(self, handle, name, value):
        name, value = name.decode(), value.decode()
        if name == "PixelFormat" and value not in self.formats:
            return FAIL
        self.features[name] = value
        return OK

    def _range(self, name, out, which):
        low, high = self.ranges[name]
        out._obj.value = {"": self.features[name], "Min": low, "Max": high}[which]
        return OK

    def PylonDeviceGetFloatFeature(self, handle, name, out):
        return self._range(name.decode(), out, "") if name.decode() in self.ranges else \
            (setattr(out._obj, "value", float(self.features[name.decode()])) or OK)

    def PylonDeviceGetFloatFeatureMin(self, handle, name, out):
        return self._range(name.decode(), out, "Min")

    def PylonDeviceGetFloatFeatureMax(self, handle, name, out):
        return self._range(name.decode(), out, "Max")

    def PylonDeviceSetFloatFeature(self, handle, name, value):
        self.features[name.decode()] = value
        return OK

    def PylonDeviceGetIntegerFeature(self, handle, name, out):
        name = name.decode()
        if name == "PayloadSize":
            depth = 3 if self.features["PixelFormat"].startswith(("RGB", "BGR")) else \
                1 if self.features["PixelFormat"].endswith("8") else 2
            out._obj.value = self.features["Width"] * self.features["Height"] * depth
        else:
            out._obj.value = int(self.features[name])
        return OK

    def PylonDeviceSetBooleanFeature(self, handle, name, value):
        self.features[name.decode()] = "1" if value else "0"
        return OK

    def PylonDeviceExecuteCommandFeature(self, handle, name):
        self.calls.append(name.decode())
        self.started = name == b"AcquisitionStart"
        return OK

    # ---- the stream
    def PylonDeviceGetNumStreamGrabberChannels(self, handle, n):
        n._obj.value = 1
        return OK

    def PylonDeviceGetStreamGrabber(self, handle, index, grabber):
        grabber._obj.value = 88
        return OK

    def PylonStreamGrabberOpen(self, grabber):
        self.calls.append("grabber open")
        return OK

    def PylonStreamGrabberClose(self, grabber):
        self.calls.append("grabber close")
        return OK

    def PylonStreamGrabberGetWaitObject(self, grabber, wait):
        wait._obj.value = 99
        return OK

    def PylonStreamGrabberSetMaxNumBuffer(self, grabber, n):
        return OK

    def PylonStreamGrabberSetMaxBufferSize(self, grabber, size):
        self.size = size.value
        return OK

    def PylonStreamGrabberPrepareGrab(self, grabber):
        return OK

    def PylonStreamGrabberFinishGrab(self, grabber):
        self.calls.append("finish")
        return OK

    def PylonStreamGrabberRegisterBuffer(self, grabber, address, size, hbuf):
        h = 1000 + len(self.buffers)
        self.buffers[h] = (address.value, size.value)
        hbuf._obj.value = h
        return OK

    def PylonStreamGrabberDeregisterBuffer(self, grabber, hbuf):
        del self.buffers[hbuf.value]
        return OK

    def PylonStreamGrabberQueueBuffer(self, grabber, hbuf, context):
        self.queued.append((hbuf.value, context.value))
        return OK

    def PylonWaitObjectWait(self, wait, timeout, ready):
        # the camera fills ``burst`` queued buffers, each frame's level 10 more than the last (16 bits when Mono12)
        for _ in range(min(self.burst, len(self.queued))):
            h, context = self.queued.pop(0)
            address, size = self.buffers[h]
            self.level += 10
            fmt = self.features["PixelFormat"]
            if fmt == "Mono12":
                data = np.full(size // 2, self.level << 4, "<u2").tobytes()
            elif fmt.startswith(("RGB", "BGR")):
                data = np.tile(np.array([self.level, self.level + 1, self.level + 2], np.uint8), size // 3).tobytes()
            else:
                data = bytes([self.level % 256]) * size
            ctypes.memmove(address, data, size)
            self.output.append((h, context, basler.GRABBED))
        ready._obj.value = bool(self.output)
        return OK

    def PylonStreamGrabberRetrieveResult(self, grabber, result, ready):
        if not self.output:
            ready._obj.value = False
            return OK
        h, context, status = self.output.pop(0)
        r = result._obj
        r.Context, r.hBuffer, r.pBuffer, r.Status = context, h, self.buffers[h][0], status
        ready._obj.value = True
        return OK

    def PylonStreamGrabberCancelGrab(self, grabber):
        self.output += [(h, c, 3) for h, c in self.queued]                 # Canceled
        self.queued = []
        return OK


def _camera(**kw):
    fake = FakePylon(**kw)
    return fake, basler.Camera(fake, owner_check=lambda: False)


def test_the_grab_result_s_first_fields_sit_where_pylon_c_puts_them():
    # PylonGrabResult_t: Context, hBuffer, pBuffer (pointers), then Status; the same in pylon 5 and 6
    assert [basler.GrabResult.Context.offset, basler.GrabResult.hBuffer.offset, basler.GrabResult.pBuffer.offset,
            basler.GrabResult.Status.offset] == [0, 8, 16, 24]
    assert ctypes.sizeof(basler.GrabResult) >= 112                     # room for the whole of either version


def test_the_camera_is_taken_only_when_pylon_viewer_is_closed_and_let_go_as_found():
    fake = FakePylon()
    held = basler.Camera(fake, owner_check=lambda: True)
    assert not held.open() and "pylon Viewer" in held.status and fake.calls == []
    fake, cam = _camera()
    assert cam.open() and (cam.model, cam.serial) == ("acA1920-40um", 12345678)
    assert cam.mono and fake.features["PixelFormat"] == "Mono8"           # 8-bit frames while held
    assert fake.features["ExposureAuto"] == "Off" and fake.features["TriggerMode"] == "Off"
    assert fake.features["AcquisitionFrameRateEnable"] == "0"             # free-running at its fastest
    s = cam.setting(lucam.EXPOSURE)                                       # in ms, as the camera's us / 1000
    assert (s.value, s.low, s.high) == (10.0, pytest.approx(0.034), 10000.0)
    assert cam.setting(lucam.GAIN).high == 24.0
    cam.close()
    assert (fake.features["PixelFormat"], fake.features["ExposureAuto"], fake.features["TriggerMode"],
            fake.features["AcquisitionFrameRateEnable"]) == ("Mono12", "Continuous", "On", "1")      # put back
    assert fake.calls[-3:] == ["close", "destroy", "terminate"] and cam.handle is None


def test_a_camera_another_program_holds_is_said_and_pylon_shut_down():
    fake, cam = _camera(opens=False)
    assert not cam.open() and "pylon Viewer" in cam.status
    assert fake.calls == ["init", "open", "destroy", "terminate"]


def test_frames_are_the_newest_of_the_stream_grey_and_averaged_and_the_exposure_changes_on_the_fly():
    fake, cam = _camera()
    cam.open()
    f = cam.frame()
    assert f.rgb.shape == (6, 8, 3) and picture.is_grey(f.rgb) and tuple(f.rgb[0, 0]) == (10, 10, 10)
    assert cam.frame_rate == 41.0 and fake.calls.count("AcquisitionStart") == 1
    fake.burst = 3                                                        # frames waiting: the newest is shown
    assert tuple(cam.frame().rgb[0, 0]) == (40, 40, 40)
    assert len(fake.queued) == basler.BUFFERS                             # every buffer back with the camera
    fake.burst = 1
    f = cam.frame(average=4)                                              # 50..80: 65
    assert f.averaged == 4 and tuple(f.rgb[2, 3]) == (65, 65, 65)
    cam.set_exposure(250.0, 0.0)                                          # a gain of 0 dB is a gain
    assert fake.features["ExposureTime"] == 250000.0 and fake.features["Gain"] == 0.0
    cam.set_exposure(1e9, 99.0)                                           # kept inside the camera's range
    assert fake.features["ExposureTime"] == 10_000_000.0 and fake.features["Gain"] == 24.0
    cam.frame()
    assert fake.calls.count("AcquisitionStart") == 1
    cam.close()
    assert fake.buffers == {} and fake.queued == []                       # every buffer taken back and let go
    assert fake.calls[-6:] == ["AcquisitionStop", "finish", "grabber close", "close", "destroy", "terminate"]


def test_twelve_bit_frames_are_shown_as_eight_and_a_colour_camera_gives_rgb():
    fake, cam = _camera(formats=("Mono12",))                              # no Mono8: its 12 bits, shifted
    cam.open()
    assert cam._format == "Mono12" and tuple(cam.frame().rgb[0, 0]) == (10, 10, 10)
    cam.close()
    fake, cam = _camera(pixel_format="BayerRG8", formats=("BayerRG8", "BGR8"))
    cam.open()
    assert not cam.mono and fake.features["PixelFormat"] == "BGR8"
    assert tuple(cam.frame().rgb[0, 0]) == (12, 11, 10)                   # BGR turned to RGB
    cam.close()
    assert fake.features["PixelFormat"] == "BayerRG8"


def test_frames_that_never_come_are_said():
    fake, cam = _camera()
    cam.open()
    fake.burst = 0
    cam.exposure_ms = 1.0
    assert cam.frame() is None and "no frame came" in cam.status
    cam.close()


def test_the_rig_s_camera_decides_the_library_and_its_pixel_the_scale():
    basler_rig = [(r"USB\VID_2676&PID_BA02\12345678", "Basler acA1920-40um")]
    lumenera_rig = [(r"USB\VID_1724&PID_0753\0025050256", "INFINITY USB 3.0 Camera")]
    assert cameras.kind(lambda: basler_rig) == "basler" and cameras.kind(lambda: lumenera_rig) == "lumenera"
    d = hardware.identify(*basler_rig[0])
    assert (d.role, d.name, d.serial) == ("camera", "Basler acA1920-40um", "12345678")
    assert hardware.identify(r"USB\VID_2676&PID_BA02\12345678", "Basler USB3 Vision Camera").name == "Basler camera"
    devs = hardware.detect(devices=lambda: basler_rig, running=lambda: {"pylonviewer.exe"}, commander=lambda: False)
    assert devs[0].state == "in use by the camera's program" and devs[0].name == "Basler acA1920-40um"
    for model in ("acA1920-40um", "acA1920-40umMED", "Basler acA1920-40uc", "acA1920-40gm"):
        assert picture.pixel_um_of(model) == 5.86
    assert picture.pixel_um_of("acA1920-400um") == 0.0 and picture.pixel_um_of("INFINITY8-3M") == 4.5


def test_pylon_s_c_library_is_found_in_the_newest_pylon_installed(tmp_path, monkeypatch):
    for version, name in (("pylon 5", "PylonC_MD_VC120.dll"), ("pylon 6", "PylonC_v6_1.dll")):
        folder = tmp_path / "Basler" / version / "Runtime" / "x64"
        folder.mkdir(parents=True)
        (folder / name).write_bytes(b"")
        (folder / "PylonCamEmu_v6_1_TL.dll").write_bytes(b"")            # pylon's others are not it
    monkeypatch.delenv("CLAMPWISE_PYLON_DLL", raising=False)
    monkeypatch.delenv("PYLON_DEV_DIR", raising=False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    assert basler.find_library().name == "PylonC_v6_1.dll"


def test_the_camera_check_says_each_step(tmp_path, monkeypatch):
    from clampwise.app import main

    monkeypatch.setattr(main, "log_folder", lambda: tmp_path)
    monkeypatch.setattr(hardware, "usb_devices", lambda: [(r"USB\VID_2676&PID_BA02\12345678", "Basler acA1920-40um")])
    fake = FakePylon()
    monkeypatch.setattr(cameras, "make", lambda: basler.Camera(fake, owner_check=lambda: False))
    assert main.main(["clampwise", "--camera-check"]) == 0
    report = (tmp_path / "camera-check.txt").read_text(encoding="utf-8")
    assert "taken through: basler" in report and "model acA1920-40um" in report
    assert "10 pictures of 8 x 6 px" in report and report.rstrip().endswith("worked")
    assert fake.calls[-1] == "terminate"
    fake = FakePylon(opens=False)
    assert main.main(["clampwise", "--camera-check"]) == 1
    assert "FAIL the camera was not taken" in (tmp_path / "camera-check.txt").read_text(encoding="utf-8")
