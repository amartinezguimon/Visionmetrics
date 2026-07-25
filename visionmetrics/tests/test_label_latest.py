"""Tests for finding the most recently recorded live session (pure filesystem
logic — the actual labeling path is prep.py, already exercised manually)."""

from __future__ import annotations

import os
import time

from visionmetrics.training.label_latest import find_latest_recording


def test_no_recordings_returns_none(tmp_path):
    assert find_latest_recording(tmp_path) is None


def test_finds_newest_across_results_and_recordings_dirs(tmp_path):
    results = tmp_path / "results"
    recordings = tmp_path / "recordings"
    results.mkdir()
    recordings.mkdir()

    older = results / "demo_20260101-000000.mp4"
    older.write_bytes(b"old")
    newer = recordings / "session_20260201-000000.mp4"
    newer.write_bytes(b"new")

    now = time.time()
    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now, now))

    assert find_latest_recording(tmp_path) == newer


def test_ignores_non_mp4_files(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "demo_report.json").write_text("{}")
    video = results / "demo_clip.mp4"
    video.write_bytes(b"data")

    assert find_latest_recording(tmp_path) == video
