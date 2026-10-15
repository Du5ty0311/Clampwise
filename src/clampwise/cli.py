"""The command line (B18, item 8): ``clampwise analyze <folder> --preset DSI``
gives the same numbers as the window, because it runs the same pipeline
through the same project store.

    clampwise analyze <folder or file> [--preset NAME] [--out DIR] [--workers N] [--force]
    clampwise info <file>
    clampwise version
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import __version__


def _analyze(args) -> int:
    from .analysis import batch
    from .io import find_recordings, group_recordings
    from .project import Project

    target = Path(args.target)
    if target.is_dir():
        folder = target
        paths = find_recordings(folder)
    elif target.is_file():
        folder = target.parent
        paths = [target]
    else:
        print(f"{target} is not a file or folder", file=sys.stderr)
        return 2
    paths, notes = group_recordings(paths)
    for n in notes:
        print(n)
    project = Project.open(folder)
    try:
        if args.preset:
            if args.preset not in project.presets():
                print(f"no preset called '{args.preset}'; available: {', '.join(project.presets())}", file=sys.stderr)
                return 2
            project.apply_preset(args.preset)
        project.add_recordings(paths)
        print(f"{len(paths)} recording(s) in {folder}")
        outcomes = []
        for done, total, outcome in _progress(batch.run(folder, paths, force=args.force, workers=args.workers)):
            status = "cached" if outcome.from_cache else ("ok" if outcome.ok else "FAILED")
            print(f"[{done}/{total}] {outcome.path.name}: {status} {outcome.recording_type} "
                  f"({outcome.flags} flag(s), {outcome.seconds:.1f} s)")
            if not outcome.ok:
                print("    " + outcome.error.splitlines()[0])
            outcomes.append(outcome)
        out = Path(args.out) if args.out else None
        if out:
            out.mkdir(parents=True, exist_ok=True)
            rows = []
            for p in paths:
                run = project.last_run(p)
                if run is None:
                    continue
                row = {"file": p.name, "recording_type": run.recording_type, "type_reason": run.type_reason,
                       "stimulus": run.stimulus_summary, "flags": " | ".join(f.describe() for f in run.flags),
                       "methods": " | ".join(run.methods)}
                row.update(run.values())
                rows.append(row)
                (out / f"{p.stem}.provenance.json").write_text(json.dumps(run.provenance.to_dict(), indent=1),
                                                                encoding="utf-8")
            if rows:
                keys = list(dict.fromkeys(k for r in rows for k in r))
                with open(out / "recordings.csv", "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=keys)
                    w.writeheader()
                    w.writerows(rows)
                print(f"wrote {out / 'recordings.csv'} ({len(rows)} rows) and the provenance files")
        failed = [o for o in outcomes if not o.ok]
        return 1 if failed else 0
    finally:
        project.close()


def _progress(iterator):
    done = 0
    for outcome in iterator:
        done += 1
        yield done, getattr(iterator, "total", 0) or done, outcome


def _info(args) -> int:
    from .io import load

    r = load(args.file)
    print(r.describe())
    print(f"mode {r.mode}; start {r.start_time}; sweeps {r.n_sweeps}; samples per sweep {r.n_samples}")
    for c in r.channels:
        print(f"  {c.describe()}: {c.role_reason}")
    for n in r.notes:
        print(f"  note: {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clampwise", description="Clampwise: analysis for slice electrophysiology")
    sub = parser.add_subparsers(dest="command")
    a = sub.add_parser("analyze", help="analyze a file or every recording under a folder, through its project")
    a.add_argument("target")
    a.add_argument("--preset", default="")
    a.add_argument("--out", default="")
    a.add_argument("--workers", type=int, default=None)
    a.add_argument("--force", action="store_true", help="ignore cached results")
    i = sub.add_parser("info", help="what a file holds")
    i.add_argument("file")
    sub.add_parser("version")
    args = parser.parse_args(argv)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "info":
        return _info(args)
    if args.command == "version":
        print(f"clampwise {__version__}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
