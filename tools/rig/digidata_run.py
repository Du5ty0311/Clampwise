"""Play a Clampex protocol on the Digidata and record it, as Clampex would, from the command line
(Clampex closed: one program holds the Digidata).

    python tools/rig/digidata_run.py <protocol.pro> <folder> [--channel 1] [--serial <the 700B's>] [--seconds 20]

The work is ``clampwise.rig.recorder.Recorder``'s (the one Live's Record uses); Ctrl+C stops it early, the file
keeping the sweeps complete by then.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from clampwise.rig.recorder import Recorder               # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol", type=Path)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--serial", default=None, help="the amplifier's serial (found from Commander when left out)")
    ap.add_argument("--seconds", type=float, default=None, help="gap-free: the run's length")
    a = ap.parse_args()
    rec = Recorder(a.serial, a.channel)
    t0 = time.time()
    rec.start(a.protocol, a.folder, a.seconds)
    said = ""
    try:
        while rec.status.busy:
            if rec.status.message != said:
                said = rec.status.message
                print(f"{time.time() - t0:6.1f} s  {rec.status.state}: {said}", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        rec.stop()
    st = rec.wait()
    for note in st.notes:
        print("note:", note)
    print(f"{time.time() - t0:6.1f} s  {st.state}: {st.message} ({st.frames} of {st.total} frames)", flush=True)
    if st.state == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
