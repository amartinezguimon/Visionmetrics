"""Tests for config loading and the edge<->cloud schema (pure)."""

from visionmetrics.edge.agent.config import DeviceConfig
from visionmetrics.shared.schema import SCHEMA_VERSION, MetricBucket


def test_empty_config_uses_prototype_defaults():
    cfg = DeviceConfig.from_dict({})
    assert cfg.camera.fov_h_deg == 70.0
    assert cfg.vision.face_width_m == 0.16
    assert cfg.engagement.count_threshold_s == 3.0
    assert cfg.uplink.enabled is False


def test_partial_config_overrides_only_given_keys():
    cfg = DeviceConfig.from_dict({
        "device": {"store_name": "Joyeria Centro"},
        "camera": {"source": "rtsp://10.0.0.5/stream", "fov_h_deg": 90.0},
        "engagement": {"count_threshold_s": 7.0},
    })
    assert cfg.device.store_name == "Joyeria Centro"
    assert cfg.camera.source == "rtsp://10.0.0.5/stream"
    assert cfg.camera.fov_h_deg == 90.0
    assert cfg.engagement.count_threshold_s == 7.0
    assert cfg.engagement.frame_buffer_size == 3  # untouched default


def test_unknown_keys_are_ignored():
    cfg = DeviceConfig.from_dict({"camera": {"source": 1, "nonsense": 123}})
    assert cfg.camera.source == 1


def test_calibration_path_read():
    cfg = DeviceConfig.from_dict({"calibration": {"config_path": "configs/x.json"}})
    assert cfg.calibration_config_path == "configs/x.json"


def test_metric_bucket_idempotency_key():
    b = MetricBucket(
        schema_version=SCHEMA_VERSION, device_id="dev-1", store_id="store-1",
        window_start="2026-06-08T14:00", window_end="2026-06-08T15:00",
        passersby=120, engaged=18, engagement_rate=15.0,
        total_attention_s=240.5,
    )
    assert b.idempotency_key == "dev-1:2026-06-08T14:00"


def test_metric_bucket_roundtrip():
    b = MetricBucket(
        schema_version=SCHEMA_VERSION, device_id="dev-1", store_id="store-1",
        window_start="2026-06-08T14:00", window_end="2026-06-08T15:00",
        passersby=120, engaged=18, engagement_rate=15.0,
        total_attention_s=240.5,
    )
    assert MetricBucket.from_dict(b.to_dict()) == b
