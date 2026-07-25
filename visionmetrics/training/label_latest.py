"""Jump straight to labeling the most recently recorded live session.

Every live demo session (`webserver.py`) auto-records itself to an .mp4 next
to its report (or under `recordings/` if run standalone). Once a session ends
there was previously no direct path from "the recording is saved" to "I'm
labeling it" — you had to remember the file path and paste it into option 6.
This finds that file for you.

Run:
    python -m visionmetrics.training.label_latest
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import prep

ROOT = Path(__file__).resolve().parents[2]
SEARCH_DIRS = ("results", "recordings")


def find_latest_recording(root: Path = ROOT) -> Path | None:
    candidates: list[Path] = []
    for d in SEARCH_DIRS:
        folder = root / d
        if folder.exists():
            candidates.extend(folder.glob("*.mp4"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser(description="Label the most recently recorded live session.")
    ap.add_argument("--sample-seconds", type=float, default=5.0,
                    help="store one whole frame every N seconds")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--aspect", type=float, default=0.30)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    video = find_latest_recording()
    if video is None:
        print(f"[label-latest] no recordings found under {' or '.join(SEARCH_DIRS)}/. "
              "Run a live demo session first (it records automatically).")
        return 1

    print(f"[label-latest] most recent recording -> {video}")
    return prep.run(str(video), sample_seconds=a.sample_seconds, fov_h_deg=a.fov,
                     conf=a.conf, aspect=a.aspect, config_path=a.config,
                     open_browser=not a.no_open)


if __name__ == "__main__":
    raise SystemExit(main())
