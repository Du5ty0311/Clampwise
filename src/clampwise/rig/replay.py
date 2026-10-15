"""Replaying a recorded folder as if live (B19's "done when": "a tool that copies its files in with their original
timing"). Each ABF file is written into the watched folder as Clampex writes one: its header sections first with
the counts not yet filled in, then sweep after sweep at the recorded interval (a gap-free file in stretches of a
second), then what follows the samples and the finished header. The files start with the gaps that separated them
when recorded; ``speed`` shortens both. The recorded files are only read (with every sharing flag).

    python -m clampwise.rig.replay <recorded folder> <watched folder> [--speed 10]
"""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

from . import growing
from .files import read_shared

GAP_FREE_STRETCH_S = 1.0


def plan(source: str | Path) -> list[tuple[float, Path]]:
    """(seconds after the first file's start, path) of each ABF file in ``source``, in the order they were recorded."""
    from ..io.abf import read_header

    timed = []
    for p in sorted(Path(source).glob("*.abf")):
        try:
            h = read_header(p)
        except (OSError, ValueError):
            continue
        timed.append((h.start_time.timestamp() if h.start_time else p.stat().st_mtime, p))
    timed.sort()
    return [(t - timed[0][0], p) for t, p in timed] if timed else []


def write_as_recorded(src: Path, dst: Path, *, speed: float = 1.0, sleep=time.sleep) -> None:
    """Write ``src`` into ``dst`` as Clampex writes a file, at its own pace (``speed`` times faster)."""
    data = read_shared(src)
    head = data[:growing.SECTIONS + 16 * 18]
    block, point_bytes, entries = struct.unpack_from("<IIq", head, growing.SECTIONS + 16 * growing.DATA)
    start = block * growing.BLOCK
    channels = struct.unpack_from("<IIq", head, growing.SECTIONS + 16 * growing.ADC)[2]
    protocol = struct.unpack_from("<I", head, growing.SECTIONS)[0] * growing.BLOCK
    mode, interval_us = struct.unpack_from("<hf", data, protocol)
    per_episode = struct.unpack_from("<i", data, protocol + 22)[0]
    if mode == growing.EPISODIC:
        step = per_episode * point_bytes
        pause = struct.unpack_from("<f", data, protocol + 62)[0] or per_episode / channels * interval_us / 1e6
    else:
        step = int(GAP_FREE_STRETCH_S * 1e6 / interval_us) * channels * point_bytes
        pause = GAP_FREE_STRETCH_S
    end = start + entries * point_bytes
    unfinished = bytearray(data[:start])            # the counts are filled in when the file is finished
    struct.pack_into("<I", unfinished, 12, 0)
    struct.pack_into("<q", unfinished, growing.SECTIONS + 16 * growing.DATA + 8, 0)
    for i in growing.AFTER_DATA:
        struct.pack_into("<IIq", unfinished, growing.SECTIONS + 16 * i, 0, 0, 0)
    with open(dst, "wb") as fh:
        fh.write(bytes(unfinished))
        fh.flush()
        for at in range(start, end, step):
            sleep(pause / speed)
            fh.write(data[at:min(at + step, end)])
            fh.flush()
        fh.write(data[end:])                        # the synch array, the tags
        fh.seek(0)
        fh.write(data[:start])                      # the finished header
        fh.flush()


def replay(source: str | Path, target: str | Path, *, speed: float = 1.0, sleep=time.sleep, say=print) -> list[Path]:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    began, written = time.monotonic(), []
    for at, p in plan(source):
        wait = at / speed - (time.monotonic() - began)
        if wait > 0:
            sleep(wait)
        say(f"{time.monotonic() - began:7.1f} s  {p.name}")
        write_as_recorded(p, target / p.name, speed=speed, sleep=sleep)
        written.append(target / p.name)
    return written


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    speed = 1.0
    if "--speed" in args:
        i = args.index("--speed")
        speed = float(args[i + 1])
        del args[i:i + 2]
    if len(args) != 2:
        print(__doc__)
        return 2
    source, target = Path(args[0]), Path(args[1])
    if source.resolve() == target.resolve():
        print("the watched folder must be another folder than the recorded one")
        return 2
    replay(source, target, speed=speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
