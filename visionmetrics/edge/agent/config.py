"""Typed device configuration loaded from device.yaml.

One place to read every per-deployment setting. Loading is strict-ish: unknown
sections are ignored, missing ones fall back to defaults that mirror the
prototype's old hardcoded values, so an empty config still runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .engagement import EngagementParams


@dataclass
class DeviceInfo:
    device_id: str = "dev-local"
    store_id: str = "store-local"
    store_name: str = "VisionMetrics AI"


@dataclass
class CameraConfig:
    source: Any = 0                  # int index | rtsp url | file path
    fov_h_deg: float = 70.0
    reconnect_delay_s: float = 2.0


@dataclass
class VisionConfig:
    face_width_m: float = 0.16
    yolo_conf_min: float = 0.45
    # 0.4 (was 0.75): keep boxes up to ~2.5x wider-than-tall. In a crowd you only
    # see many people head-and-shoulders (box is wide + short, h/w < 0.75), so the
    # old 0.75 floor discarded real people YOLO had already detected. 0.4 still
    # rejects genuinely flat non-people (benches, bags, reflections).
    aspect_ratio_min: float = 0.4
    # Passerby counting (foot traffic): a track counts as a real person once it
    # has PERSISTED for a few frames AND has either moved or shown a face — which
    # rejects flickery false boxes and static furniture (a chair never does both).
    passerby_min_frames: int = 8     # must persist this many frames before counting
    passerby_motion_px: int = 40     # movement (px) that proves a faceless track is a person
    # Ignore people whose bounding box is shorter than this fraction of the frame
    # height — i.e. too far away to be a real customer (across the street, a
    # transverse pavement). 0.0 = off. Set per store (e.g. 0.15) alongside the
    # counting zone; complements it where perspective makes far people fall inside
    # the polygon. Applies to BOTH foot traffic and engagement.
    passerby_min_height_frac: float = 0.0
    head_crop_frac: float = 0.45
    head_upscale: int = 4
    face_skip_frames: int = 3
    pose_enabled: bool = True
    pose_skip_frames: int = 8
    torso_neutral_span: float = 0.40
    torso_min_visibility: float = 0.40
    # Tracking robustness (anti double-count on ByteTrack id switches):
    track_buffer_frames: int = 60       # ByteTrack keeps a lost track this long (~2s @30fps)
    reassoc_grace_frames: int = 45      # adopt a new id into a lost track within this window
    reassoc_min_iou: float = 0.30       # box overlap required to treat them as the same person


@dataclass
class ModelPaths:
    yolo: str = "yolov8n.pt"
    face: str = "models/face_landmarker.task"
    pose: str = "models/pose_landmarker_lite.task"
    engagement: str = "models/engagement_model.pth"


@dataclass
class UplinkConfig:
    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    flush_interval_s: float = 60.0          # how often the sender thread drains the buffer
    window_s: float = 3600.0                # metric bucket length (hourly by default)
    heartbeat_interval_s: float = 30.0      # liveness ping cadence
    buffer_path: str = "data/uplink_buffer.sqlite"


@dataclass
class DeviceConfig:
    device: DeviceInfo = field(default_factory=DeviceInfo)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    engagement: EngagementParams = field(default_factory=EngagementParams)
    models: ModelPaths = field(default_factory=ModelPaths)
    uplink: UplinkConfig = field(default_factory=UplinkConfig)
    calibration_config_path: str | None = "configs/store_config.json"

    @classmethod
    def load(cls, path: str | Path) -> "DeviceConfig":
        """Load and validate a device.yaml. Missing keys use prototype defaults."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "DeviceConfig":
        calib = (raw.get("calibration") or {}).get("config_path", "configs/store_config.json")
        return cls(
            device=_build(DeviceInfo, raw.get("device")),
            camera=_build(CameraConfig, raw.get("camera")),
            vision=_build(VisionConfig, raw.get("vision")),
            engagement=_build(EngagementParams, raw.get("engagement")),
            models=_build(ModelPaths, raw.get("models")),
            uplink=_build(UplinkConfig, raw.get("uplink")),
            calibration_config_path=calib,
        )


def _build(dc_type, data: dict | None):
    """Construct a dataclass from a dict, ignoring unknown keys."""
    if not data:
        return dc_type()
    known = {f.name for f in fields(dc_type)}
    return dc_type(**{k: v for k, v in data.items() if k in known})
