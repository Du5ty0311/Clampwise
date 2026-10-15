"""Read ABF files with Axon's own library (pCLAMP's ABFFIO.dll, the format's owner), from a 32-bit Python:
each sweep's values as Axon scales them and its start time. A reference for Clampwise's files that is neither
Clampwise nor pyabf. Standard library only.

    %LOCALAPPDATA%\\Clampwise\\py32\\python.exe -I tools/rig/axon_read.py <file.abf> <channel> [<channel> ...]

Prints one JSON object: the episodes, the samples an episode, and per channel and episode the mean, SD, minimum,
maximum, the first 5 values and the start time (ms). The calls and the header as Axon's ABF File Support Pack
(installed with pCLAMP, ABF_FileSupportPack.zip: ABFFILES.H, ABFHEADR.H) declares them: ABF_ReadOpen,
ABF_ReadChannel, ABF_GetStartTime, ABF_Close (WINAPI); the ABFFileHeader (packed, 9965 bytes) with its versions
(2.06, pCLAMP 10.6 and later) and its size set, as its constructor does. (pCLAMP's Axabffio32.dll reads ABF 1 only:
it gives error 1001, unknown file type, on Clampex's own ABF2 files.)
"""

import ctypes
import json
import math
import struct
import sys

LIBRARY = r"C:\Program Files (x86)\Molecular Devices\pCLAMP10.7\ABFFIO.dll"
HEADER_BYTES = 9965                      # sizeof(ABFFileHeader), ABFHEADR.H (pack 1)


def main() -> None:
    path, channels = sys.argv[1], [int(c) for c in sys.argv[2:]]
    import os

    os.add_dll_directory(os.path.dirname(LIBRARY))
    dll = ctypes.WinDLL(LIBRARY)
    header = ctypes.create_string_buffer(HEADER_BYTES + 65536)   # room past it: a newer library may fill more
    struct.pack_into("<f", header, 0, 2.06)            # fFileVersionNumber = ABF_CURRENTVERSION
    struct.pack_into("<f", header, 28, 2.06)           # fHeaderVersionNumber
    struct.pack_into("<i", header, 9445, HEADER_BYTES)  # lHeaderSize = sizeof(ABFFileHeader)
    handle, max_samples, max_episodes, err = ctypes.c_int(), ctypes.c_uint(), ctypes.c_uint32(), ctypes.c_int()
    ok = dll.ABF_ReadOpen(path.encode("mbcs"), ctypes.byref(handle), 0, header, ctypes.byref(max_samples),
                           ctypes.byref(max_episodes), ctypes.byref(err))
    if not ok:
        print(json.dumps({"error": f"ABF_ReadOpen failed, error {err.value}"}))
        return
    out = {"episodes": max_episodes.value, "samples": max_samples.value, "channels": {}}
    buf = (ctypes.c_float * max(max_samples.value, 1))()
    try:
        for ch in channels:
            rows = []
            for ep in range(1, max_episodes.value + 1):
                n = ctypes.c_uint(max_samples.value)
                if not dll.ABF_ReadChannel(handle, header, ch, ctypes.c_uint32(ep), buf, ctypes.byref(n),
                                            ctypes.byref(err)):
                    rows.append({"error": err.value})
                    continue
                v = list(buf[: n.value])
                mean = sum(v) / len(v)
                sd = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))
                t = ctypes.c_double()
                dll.ABF_GetStartTime(handle, header, ch, ctypes.c_uint32(ep), ctypes.byref(t), ctypes.byref(err))
                rows.append({"n": n.value, "mean": mean, "sd": sd, "min": min(v), "max": max(v), "first": v[:5],
                             "start_ms": t.value})
            out["channels"][str(ch)] = rows
    finally:
        dll.ABF_Close(handle, ctypes.byref(err))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
