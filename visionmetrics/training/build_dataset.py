"""Merge all collected sessions into the master training dataset + report coverage.

Run by whoever owns the code (you) after pulling Hector's session files:

    python -m visionmetrics.training.build_dataset

It reads every CSV in data/raw_sessions/ (plus the legacy data/engagement_data.csv
if present), normalises them to one schema, writes data/engagement_dataset.csv, and
prints a coverage report — how many looking/away samples you have per distance tier
and per condition (glasses/headwear), flagging where you're thin so you know what to
collect next. Then train with:  python -m visionmetrics.training.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import dataset

DEFAULT_SESSIONS_DIR = "data/raw_sessions"
DEFAULT_LEGACY = "data/engagement_data.csv"
DEFAULT_OUTPUT = "data/engagement_dataset.csv"


def gather_inputs(sessions_dir: str, legacy: str | None) -> list[str]:
    paths = sorted(str(p) for p in Path(sessions_dir).glob("*.csv"))
    if legacy and Path(legacy).exists():
        paths.append(legacy)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the master engagement dataset.")
    ap.add_argument("--sessions", default=DEFAULT_SESSIONS_DIR,
                    help="folder of per-session CSVs from the collector")
    ap.add_argument("--legacy", default=DEFAULT_LEGACY,
                    help="legacy flat CSV to fold in (set '' to skip)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="master dataset to write")
    a = ap.parse_args()

    inputs = gather_inputs(a.sessions, a.legacy or None)
    if not inputs:
        print(f"No input CSVs found in {a.sessions!r} (or legacy {a.legacy!r}).")
        return 1

    print(f"Merging {len(inputs)} file(s):")
    for p in inputs:
        print(f"  - {p}")
    df = dataset.merge(inputs)
    before = len(df)
    df = dataset.dedupe(df)
    if before != len(df):
        print(f"Deduped {before - len(df)} exact-duplicate row(s) "
              f"({before} -> {len(df)}).")

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output, index=False)
    print(f"\nWrote {len(df)} rows -> {a.output}\n")
    print(dataset.coverage_text(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
