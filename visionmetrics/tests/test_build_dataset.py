"""Tests for build_dataset.gather_inputs — the file-selection logic that feeds
the training dataset. Regression coverage for the bug where the live web
dashboard's `*_detections.csv` sibling files (a different schema, for the YOLO
fine-tuning pipeline) got swept into the engagement dataset merge and crashed
`dataset.normalize` (no yaw/pitch/label columns)."""

from __future__ import annotations

from visionmetrics.training import build_dataset, dataset


def _touch(path, text="yaw,pitch,distance,label\n0.0,0.0,0.3,1\n"):
    path.write_text(text, encoding="utf-8")


def test_gather_inputs_excludes_detections_siblings(tmp_path):
    sessions = tmp_path / "raw_sessions"
    sessions.mkdir()
    _touch(sessions / "live_20260101-000000.csv")
    _touch(sessions / "live_20260101-000000_detections.csv",
           "id,box,conf\n1,\"[0,0,10,10]\",0.9\n")   # different schema entirely
    _touch(sessions / "live_20260102-000000.csv")

    inputs = build_dataset.gather_inputs(str(sessions), legacy=None)

    assert len(inputs) == 2
    assert all(not p.endswith("_detections.csv") for p in inputs)


def test_gather_inputs_appends_legacy_last(tmp_path):
    sessions = tmp_path / "raw_sessions"
    sessions.mkdir()
    _touch(sessions / "live_20260101-000000.csv")
    legacy = tmp_path / "engagement_data.csv"
    _touch(legacy)

    inputs = build_dataset.gather_inputs(str(sessions), legacy=str(legacy))

    assert inputs[-1] == str(legacy)


def test_gather_inputs_output_is_mergeable(tmp_path):
    """The whole point: what gather_inputs returns must not crash dataset.merge,
    even when a detections sibling with a totally different schema is present."""
    sessions = tmp_path / "raw_sessions"
    sessions.mkdir()
    _touch(sessions / "live_20260101-000000.csv")
    _touch(sessions / "live_20260101-000000_detections.csv",
           "id,box,conf\n1,\"[0,0,10,10]\",0.9\n")

    inputs = build_dataset.gather_inputs(str(sessions), legacy=None)
    merged = dataset.merge(inputs)   # would raise ValueError before the fix

    assert len(merged) == 1
