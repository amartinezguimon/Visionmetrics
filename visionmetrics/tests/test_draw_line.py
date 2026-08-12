"""Tests for the pure helpers of the far-line drawing tool (no OpenCV)."""

import pytest

from visionmetrics.edge.tools.draw_line import (
    load_config_dict,
    merge_far_line,
    normalize_line,
    save_far_line,
)


def test_normalize_line_scales_and_clamps():
    pts = [(0, 0), (700, 500)]  # second endpoint is out of a 640x480 frame
    norm = normalize_line(pts, 640, 480)
    assert norm[0] == [0.0, 0.0]
    assert norm[1] == [1.0, 1.0]   # clamped back into the frame


def test_normalize_line_requires_exactly_two_points():
    with pytest.raises(ValueError):
        normalize_line([(0, 0)], 640, 480)
    with pytest.raises(ValueError):
        normalize_line([(0, 0), (1, 1), (2, 2)], 640, 480)


def test_merge_preserves_existing_keys_and_does_not_mutate():
    cfg = {"store_name": "X", "counting_region": {"polygon": [[0, 0]]}}
    out = merge_far_line(cfg, [[0.05, 0.45], [0.95, 0.55]])
    assert out["store_name"] == "X"
    assert out["counting_region"] == {"polygon": [[0, 0]]}   # untouched
    assert out["far_line"]["line"] == [[0.05, 0.45], [0.95, 0.55]]
    assert "far_line" not in cfg   # original untouched


def test_save_far_line_roundtrips_through_disk(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"store_name": "Y"}', encoding="utf-8")
    save_far_line(p, [[0.0, 0.5], [1.0, 0.5]])
    loaded = load_config_dict(p)
    assert loaded["store_name"] == "Y"                 # preserved
    assert loaded["far_line"]["line"] == [[0.0, 0.5], [1.0, 0.5]]
