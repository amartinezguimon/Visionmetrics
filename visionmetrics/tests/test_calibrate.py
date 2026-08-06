"""Tests for the pure helpers of the store calibration tool (no OpenCV/camera).

No test file existed for this tool before — it used to carry its own
duplicated (and untested) angle math; now it reuses geometry.py/HeadPoseAnalyzer
directly (covered by test_geometry.py / test_pipeline.py) and only these small
pure JSON/derivation helpers are specific to it.
"""

from __future__ import annotations

from visionmetrics.edge.tools.calibrate import (
    compute_derived_zone,
    load_config_dict,
    merge_calibration,
    save_calibration,
)


def test_compute_derived_zone_uses_degrees_not_ratio_scale():
    # A realistic capture: looking ~20 deg left, ~18 deg right, ~2 deg up at centre.
    z = compute_derived_zone(-20.0, 18.0, 2.0,
                              tolerance_yaw_deg=15.0, tolerance_pitch_deg=10.0)
    assert z["yaw_min"] == -35.0    # min(-20, 18) - 15
    assert z["yaw_max"] == 33.0     # max(-20, 18) + 15
    assert z["pitch_min"] == -8.0   # 2 - 10
    assert z["pitch_max"] == 12.0   # 2 + 10


def test_compute_derived_zone_handles_either_capture_order():
    # Operator could capture "right" with a numerically smaller value than
    # "left" depending on which side of 0 the display sits — min/max must not
    # assume yaw_left < yaw_right.
    a = compute_derived_zone(yaw_left=10.0, yaw_right=-5.0, pitch_center=0.0)
    b = compute_derived_zone(yaw_left=-5.0, yaw_right=10.0, pitch_center=0.0)
    assert a == b


def test_merge_calibration_preserves_existing_keys_and_does_not_mutate():
    existing = {"counting_region": {"polygon": [[0, 0], [1, 0], [1, 1]]}, "store_name": "old"}
    calibration = {"store_name": "new", "engagement_zone": {"yaw_center": 1.2}}
    out = merge_calibration(existing, calibration)
    assert out["store_name"] == "new"                       # calibration wins on shared keys
    assert out["counting_region"] == existing["counting_region"]  # draw_zone's own data survives
    assert "counting_region" not in calibration              # original arg untouched


def test_load_config_dict_missing_returns_empty(tmp_path):
    assert load_config_dict(tmp_path / "nope.json") == {}


def test_save_calibration_merges_with_existing_file(tmp_path):
    p = tmp_path / "store_config.json"
    save_calibration(p, {"store_name": "A", "counting_region": {"polygon": [[0, 0], [1, 0], [1, 1]]}})
    save_calibration(p, {"store_name": "A", "engagement_zone": {"yaw_center": 3.0}})

    saved = load_config_dict(p)
    assert saved["engagement_zone"] == {"yaw_center": 3.0}
    assert saved["counting_region"]["polygon"] == [[0, 0], [1, 0], [1, 1]]   # not clobbered
