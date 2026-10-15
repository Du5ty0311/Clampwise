"""The Digidata 1440A, through pCLAMP's own library, from a 32-bit process (acquisition
without Clampex, as the camera is used without its own program).

``AxDD1400.dll`` is 32-bit, so this file runs under a 32-bit Python (``%LOCALAPPDATA%\\Clampwise\\py32``), standard
library only, and talks to Clampwise over its standard input and output: one JSON object a line each way.

The library (pCLAMP 10.7, read 2026-10-15, never changed) exports stdcall functions to find and open the device;
``DD1400_OpenDevice`` returns a C++ object whose methods (thiscall: the object in ECX, the callee pops its
arguments) do the rest, in this order of its virtual table: 1 Reset, 2 GetDeviceInfo, 3 SetSerialNumber,
4 GetBufferGranularity, 5 SetProtocol, 6 GetProtocol, 7 StartAcquisition, 8 StopAcquisition, 9 IsAcquiring,
10 GetAIPosition, 11 GetAOPosition, 12 GetAIValue, 13 GetDIValue, 14 SetAOValue, 15 SetDOValue,
16 SetTrigThreshold, 17 GetTrigThreshold, 18 ReadTelegraphs, 19 GetTimeAtStartOfAcquisition,
20 GetCalibrationParams, 21 SetCalibrationParams, 22 GetPowerOnData, 23 SetPowerOnData, 24 GetEepromParams,
25 SetEepromParams, 26 GetLastErrorText, 27 GetLastError. The order and each method's argument bytes were read
from the library's code; they match the calls John Dempster's WinWCP makes (``dd1440.pas``, SESComponentsXE,
University of Strathclyde), whose structures are followed here. Never while Clampex is open: one program owns the
Digidata. EEPROM, calibration and serial-number writers are never called.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
import subprocess
import sys
from ctypes import wintypes

LIBRARY = r"C:\Program Files (x86)\Molecular Devices\pCLAMP10.7\AxDD1400.dll"
MAX_AI, MAX_AO = 16, 4
METHODS = {"Reset": (1, 0), "GetDeviceInfo": (2, 1), "GetBufferGranularity": (4, 0), "SetProtocol": (5, 1),
           "GetProtocol": (6, 1), "StartAcquisition": (7, 0), "StopAcquisition": (8, 0), "IsAcquiring": (9, 0),
           "GetAIPosition": (10, 1), "GetAOPosition": (11, 1), "GetAIValue": (12, 2), "GetDIValue": (13, 1),
           "SetAOValue": (14, 2), "SetDOValue": (15, 1), "ReadTelegraphs": (18, 3),
           "GetTimeAtStartOfAcquisition": (19, 1), "GetCalibrationParams": (20, 1), "GetPowerOnData": (22, 1),
           "GetLastErrorText": (26, 2), "GetLastError": (27, 0)}     # name: (slot, 4-byte arguments)


class Info(ctypes.Structure):          # DD1400_Info, packed: 115 bytes a device (FindDevices' stride)
    _pack_ = 1
    _fields_ = [("vendor_id", ctypes.c_uint16), ("product_id", ctypes.c_uint16), ("serial", ctypes.c_uint32),
                ("name", ctypes.c_char * 32), ("firmware", ctypes.c_char * 16),
                ("input_buffer_samples", ctypes.c_uint32), ("output_buffer_samples", ctypes.c_uint32),
                ("ai_channels", ctypes.c_uint32), ("ao_channels", ctypes.c_uint32),
                ("telegraphs", ctypes.c_uint32), ("do_channels", ctypes.c_uint32),
                ("min_period_us", ctypes.c_double), ("max_period_us", ctypes.c_double),
                ("period_quantum_us", ctypes.c_double), ("min_prequeue", ctypes.c_uint32),
                ("max_prequeue", ctypes.c_uint32), ("scope_out_bit", ctypes.c_uint16), ("usb1", ctypes.c_uint8)]


class Calibration(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("adc_gains", ctypes.c_int16 * MAX_AI), ("adc_offsets", ctypes.c_int16 * MAX_AI),
                ("dac_gains", ctypes.c_float * MAX_AO), ("dac_offsets", ctypes.c_int16 * MAX_AO)]


class Buffer(ctypes.Structure):        # DATABUFFER: one link of a circular list of transfer buffers
    _pack_ = 1


Buffer._fields_ = [("n", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("data", ctypes.c_void_p),
                   ("data_flags", ctypes.c_void_p), ("next", ctypes.POINTER(Buffer)),
                   ("prev", ctypes.POINTER(Buffer))]


class Protocol(ctypes.Structure):      # DD1400_Protocol (the library keeps 146 bytes; the tail is zero padding)
    _pack_ = 1
    _fields_ = [("period_us", ctypes.c_double), ("flags", ctypes.c_uint32),
                ("scope_out_ai", ctypes.c_int32), ("scope_out_threshold", ctypes.c_int16),
                ("scope_out_hysteresis", ctypes.c_int16), ("scope_out_positive", ctypes.c_uint8),
                ("n_ai", ctypes.c_uint32), ("ai", ctypes.c_int32 * MAX_AI), ("ai_buffers", ctypes.c_void_p),
                ("n_ai_buffers", ctypes.c_uint32), ("di_enable", ctypes.c_uint8), ("unused1", ctypes.c_uint8 * 3),
                ("n_ao", ctypes.c_uint32), ("ao", ctypes.c_int32 * MAX_AO), ("ao_buffers", ctypes.c_void_p),
                ("n_ao_buffers", ctypes.c_uint32), ("do_enable", ctypes.c_uint8), ("unused2", ctypes.c_uint8 * 3),
                ("chunks_per_second", ctypes.c_uint32), ("terminal_count", ctypes.c_uint32),
                ("save_ai", ctypes.c_uint8), ("save_ao", ctypes.c_uint8), ("padding", ctypes.c_uint8 * 9)]


def clampex_open() -> bool:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Clampex.exe", "/NH"], capture_output=True, text=True,
                         creationflags=0x08000000).stdout
    return "clampex.exe" in out.lower()


class _Thunks:
    """thiscall from ctypes: a stdcall stub ``f(this, args...)`` that moves ``this`` into ECX and jumps to the
    method, which pops its own arguments (pop eax; pop ecx; push eax; mov eax, target; jmp eax)."""

    def __init__(self) -> None:
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.VirtualAlloc.restype = ctypes.c_void_p
        k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
        self.page = k.VirtualAlloc(None, 4096, 0x3000, 0x40)          # MEM_COMMIT|MEM_RESERVE, EXECUTE_READWRITE
        self.used = 0

    def make(self, target: int, n_args: int):
        code = b"\x58\x59\x50\xb8" + struct.pack("<I", target) + b"\xff\xe0"
        at = self.page + self.used
        ctypes.memmove(at, code, len(code))
        self.used += 16
        return ctypes.WINFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, *[ctypes.c_uint32] * n_args)(at)


class Digidata:
    def __init__(self, library: str = LIBRARY) -> None:
        folder = os.path.dirname(library)
        os.add_dll_directory(folder)                                   # its USB layer (wdapi1210.dll) beside it
        self.dll = ctypes.WinDLL(library)
        self.dll.DD1400_CountDevices.restype = ctypes.c_uint32
        self.dll.DD1400_FindDevices.restype = ctypes.c_uint32
        self.dll.DD1400_FindDevices.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)]
        self.dll.DD1400_OpenDevice.restype = ctypes.c_void_p
        self.dll.DD1400_OpenDevice.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)]
        self.dll.DD1400_CloseDevice.argtypes = [ctypes.c_void_p]
        self.dll.DD1400_GetErrorText.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_uint32]
        self.obj = None
        self.m: dict = {}

    def error_text(self, code: int) -> str:
        buf = ctypes.create_string_buffer(256)
        self.dll.DD1400_GetErrorText(code, buf, 256)
        return buf.value.decode("ascii", "replace")

    def find(self) -> list[Info]:
        n = self.dll.DD1400_CountDevices()
        if n == 0:
            return []
        infos = (Info * 8)()
        err = ctypes.c_int32(0)
        found = self.dll.DD1400_FindDevices(infos, 8, ctypes.byref(err))
        if err.value:
            raise RuntimeError(f"finding the Digidata: {self.error_text(err.value)} ({err.value:#x})")
        return list(infos[:min(found, 8)])

    def open(self, serial: int) -> None:
        err = ctypes.c_int32(0)
        obj = self.dll.DD1400_OpenDevice(serial, ctypes.byref(err))
        if not obj or err.value:
            raise RuntimeError(f"opening the Digidata: {self.error_text(err.value)} ({err.value:#x})")
        self.obj = obj
        vtable = ctypes.c_uint32.from_address(obj).value
        thunks = _Thunks()
        for name, (slot, n) in METHODS.items():
            fn = thunks.make(ctypes.c_uint32.from_address(vtable + 4 * slot).value, n)
            self.m[name] = (fn, n)

    def call(self, name: str, *args) -> int:
        fn, n = self.m[name]
        return fn(self.obj, *[a if isinstance(a, int) else ctypes.cast(a, ctypes.c_void_p).value or 0
                              for a in args])

    def ok(self, name: str, *args) -> None:
        if not self.call(name, *args) & 0xFF:
            buf = ctypes.create_string_buffer(256)
            self.call("GetLastErrorText", buf, 256)
            raise RuntimeError(f"{name}: {buf.value.decode('ascii', 'replace')}")

    def close(self) -> None:
        if self.obj:
            if self.call("IsAcquiring") & 0xFF:
                self.call("StopAcquisition")
            self.dll.DD1400_CloseDevice(self.obj)
            self.obj = None


def _info_dict(i: Info) -> dict:
    return {k: (getattr(i, k).decode("ascii", "replace") if isinstance(getattr(i, k), bytes) else getattr(i, k))
            for k, _ in Info._fields_}


def describe(dd: Digidata) -> dict:
    """What the device is, nothing played: its record, calibration, power-on outputs, telegraphs, inputs now."""
    found = dd.find()
    if not found:
        return {"found": 0}
    dd.open(found[0].serial)
    info = Info()
    dd.ok("GetDeviceInfo", ctypes.pointer(info))
    cal = Calibration()
    dd.ok("GetCalibrationParams", ctypes.pointer(cal))
    power = (ctypes.c_uint8 * 16)()
    dd.call("GetPowerOnData", power)
    tele = (ctypes.c_int16 * 4)()
    dd.call("ReadTelegraphs", 0, tele, 4)
    ai = []
    for ch in range(min(info.ai_channels, MAX_AI)):
        v = ctypes.c_int16()
        dd.ok("GetAIValue", ch, ctypes.pointer(v))
        ai.append(v.value)
    return {"found": len(found), "info": _info_dict(info),
            "calibration": {"adc_gains": list(cal.adc_gains), "adc_offsets": list(cal.adc_offsets),
                            "dac_gains": [round(g, 6) for g in cal.dac_gains], "dac_offsets": list(cal.dac_offsets)},
            "power_on": {"digital": struct.unpack_from("<I", power)[0], "analog": list(struct.unpack_from("<4h", power, 4))},
            "telegraphs": list(tele), "ai_counts": ai, "acquiring": bool(dd.call("IsAcquiring") & 0xFF),
            "granularity": dd.call("GetBufferGranularity")}


N_OUT = MAX_AO + 1                     # each output frame: the four analog outputs, then the digital outputs
SUB = 60                               # frames a transfer buffer (as WinWCP: a multiple of every channel count)


def _ring(total: int, sub: int) -> tuple:
    """A circular list of transfer buffers over one block of ``total`` int16 samples, each ``sub`` long."""
    data = (ctypes.c_int16 * total)()
    flags = (ctypes.c_int16 * total)()
    n = total // sub
    links = (Buffer * n)()
    for i in range(n):
        links[i].n, links[i].flags = sub, 0
        links[i].data = ctypes.addressof(data) + 2 * i * sub
        links[i].data_flags = ctypes.addressof(flags)
        links[i].next = ctypes.pointer(links[(i + 1) % n])
        links[i].prev = ctypes.pointer(links[(i - 1) % n])
    return data, flags, links


def record(dd: Digidata, req: dict) -> dict:
    """Acquire ``frames`` frames of the analog inputs ``ai`` every ``period_us`` into ``out`` (int16, the inputs
    interleaved, as the device gives them) while the outputs play ``waveform`` (int16 frames of N_OUT values, the
    codes already calibrated; absent: every output held at ``hold`` codes). Progress goes out as JSON lines; a
    file ``out + '.stop'`` ends it early. Outputs are put back to ``hold`` at the end."""
    info = Info()
    dd.ok("GetDeviceInfo", ctypes.pointer(info))
    ai = [int(c) for c in req["ai"]]
    frames = int(req["frames"])
    hold = [int(v) for v in req["hold"]]                       # N_OUT codes: 4 analog, then the digital word
    held = struct.pack(f"<{N_OUT}h", *hold)
    wave = open(req["waveform"], "rb") if req.get("waveform") else None      # read a block at a time, as played
    n_wave = os.path.getsize(req["waveform"]) // (2 * N_OUT) if wave is not None else 0

    sub_ai, sub_ao = SUB * len(ai) * N_OUT, SUB * len(ai) * N_OUT
    ai_total = (info.input_buffer_samples * 2 // sub_ai) * sub_ai
    ao_total = (info.output_buffer_samples * 2 // sub_ao) * sub_ao
    ai_data, ai_flags, ai_links = _ring(ai_total, sub_ai)
    ao_data, ao_flags, ao_links = _ring(ao_total, sub_ao)
    ao_frames = ao_total // N_OUT

    repeat = bool(req.get("repeat")) and n_wave > 0            # the waveform played over and over (a test pulse)

    def fill(lo: int, hi: int) -> None:
        """Frames ``lo`` to ``hi`` of the run into their slots of the output ring: the waveform's, then the holding."""
        while lo < hi:
            slot = lo % ao_frames
            n = min(hi - lo, ao_frames - slot)
            block = b""
            if repeat:
                while len(block) < 2 * N_OUT * n:
                    at = (lo + len(block) // (2 * N_OUT)) % n_wave
                    wave.seek(2 * N_OUT * at)
                    block += wave.read(2 * N_OUT * min(n - len(block) // (2 * N_OUT), n_wave - at))
            elif lo < n_wave:
                wave.seek(2 * N_OUT * lo)
                block = wave.read(2 * N_OUT * min(n, n_wave - lo))
            block += held * (n - len(block) // (2 * N_OUT))
            ctypes.memmove(ctypes.addressof(ao_data) + 2 * N_OUT * slot, block, len(block))
            lo += n

    fill(0, ao_frames)
    p = Protocol()
    p.period_us = float(req["period_us"])
    p.n_ai = len(ai)
    for i, c in enumerate(ai):
        p.ai[i] = c
    p.ai_buffers, p.n_ai_buffers = ctypes.addressof(ai_links), len(ai_links)
    p.n_ao = MAX_AO
    for c in range(MAX_AO):
        p.ao[c] = c
    p.ao_buffers, p.n_ao_buffers = ctypes.addressof(ao_links), len(ao_links)
    p.do_enable = 1
    p.chunks_per_second = 20
    stop = req["out"] + ".stop"
    pos = ctypes.c_int64(0)
    done = filled = 0
    ai_frames = ai_total // len(ai)
    filled = ao_frames
    ring = int(req.get("ring_frames") or 0)
    with open(req["out"], "wb") as out:
        dd.ok("SetProtocol", ctypes.pointer(p))
        dd.ok("StartAcquisition")
        start = ctypes.create_string_buffer(64)
        dd.call("GetTimeAtStartOfAcquisition", start)
        last_said = 0
        try:
            while done < frames and not os.path.exists(stop):
                dd.ok("GetAIPosition", ctypes.pointer(pos))
                new = min(pos.value, frames) - done
                if new > ai_frames:
                    raise RuntimeError(f"the inputs ran {new - ai_frames} frames ahead of the copy: samples lost")
                at, left = done % ai_frames, new           # contiguous stretches of the ring, copied as bytes
                put = done
                while left:
                    n = min(left, ai_frames - at)
                    if ring:                               # a file of ``ring`` frames, written round (a test)
                        n = min(n, ring - put % ring)
                        out.seek(2 * len(ai) * (put % ring))
                    out.write(ctypes.string_at(ctypes.addressof(ai_data) + 2 * at * len(ai), 2 * n * len(ai)))
                    at, left, put = (at + n) % ai_frames, left - n, put + n
                if new:
                    out.flush()                            # on disk at once: the scope draws as it is recorded
                done += new
                dd.ok("GetAOPosition", ctypes.pointer(pos))
                if filled < pos.value + ao_frames:         # refill the slots the device has played
                    fill(filled, pos.value + ao_frames)
                    filled = pos.value + ao_frames
                if done - last_said >= int(float(req.get("say_s", 0.25)) * 1e6 / p.period_us) or done >= frames:
                    print(json.dumps({"frames": done}), flush=True)
                    last_said = done
                ctypes.windll.kernel32.Sleep(20)
        finally:
            dd.call("StopAcquisition")
            for c in range(MAX_AO):
                dd.call("SetAOValue", c, hold[c] & 0xFFFF)
            dd.call("SetDOValue", hold[MAX_AO] & 0xFFFF)
            if wave is not None:
                wave.close()
    return {"done": done, "start": list(start.raw[:32])}


def main() -> None:
    if struct.calcsize("P") != 4:
        print(json.dumps({"error": "the Digidata's library is 32-bit: run this with a 32-bit Python"}), flush=True)
        return
    if clampex_open():
        print(json.dumps({"error": "Clampex is open: it owns the Digidata"}), flush=True)
        return
    dd = Digidata()
    try:
        for line in sys.stdin:
            line = line.strip().lstrip("﻿")
            if not line:
                continue
            req = json.loads(line)
            try:
                if req["cmd"] == "describe":
                    out = describe(dd)
                elif req["cmd"] == "record":
                    if clampex_open():                                     # opened while this helper was kept
                        raise RuntimeError("Clampex is open: it owns the Digidata")
                    if dd.obj is None:
                        describe(dd)
                    out = record(dd, req)
                elif req["cmd"] == "quit":
                    break
                else:
                    out = {"error": f"unknown command {req['cmd']}"}
            except Exception as exc:                                       # said to Clampwise, the device closed
                out = {"error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(out), flush=True)
    finally:
        dd.close()


if __name__ == "__main__":
    main()
