"""Wire a DeviceConfig into a ready-to-run EngagementPipeline.

Keeps construction (which needs model files on disk) out of the pipeline itself,
so the pipeline stays unit-testable with fakes while this module handles the
real YOLO / MediaPipe / PyTorch wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

from .classifier import CURRENT_FEATURE_SCHEMA, EngagementClassifier
from .config import DeviceConfig
from .models_bootstrap import ensure_models
from .pipeline import EngagementPipeline
from .tracking import ReconcileParams
from .vision.detector import PersonDetector
from .vision.face import HeadPoseAnalyzer
from .vision.pose import TorsoAnalyzer
from .zone import CountingRegion, EngagementZone, GazeReference


def _load_calibration(config: DeviceConfig) -> dict | None:
    path = config.calibration_config_path
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_zone(config: DeviceConfig) -> EngagementZone | None:
    """Load the per-store calibration zone (the window's angular span), or None."""
    raw = _load_calibration(config)
    return EngagementZone.from_config(raw.get("derived")) if raw else None


def load_gaze_reference(config: DeviceConfig) -> GazeReference:
    """Load the per-store window-centre direction used to re-centre the classifier.

    Absent calibration => (0, 0) (no shift), preserving uncalibrated behaviour.
    """
    raw = _load_calibration(config)
    return GazeReference.from_config(raw.get("engagement_zone")) if raw else GazeReference()


def load_counting_region(config: DeviceConfig) -> CountingRegion | None:
    """Load the per-store counting polygon, or None (count everywhere)."""
    raw = _load_calibration(config)
    return CountingRegion.from_config(raw.get("counting_region")) if raw else None


def build_pipeline(config: DeviceConfig) -> EngagementPipeline:
    ensure_models(config)   # self-bootstrap: fetch missing MediaPipe task files
    v = config.vision
    detector = PersonDetector(
        config.models.yolo, conf_min=v.yolo_conf_min, aspect_ratio_min=v.aspect_ratio_min,
        track_buffer=v.track_buffer_frames,
    )
    head_pose = HeadPoseAnalyzer(
        config.models.face, face_width_m=v.face_width_m, head_crop_frac=v.head_crop_frac,
        head_upscale=v.head_upscale, skip_frames=v.face_skip_frames,
    )
    torso = TorsoAnalyzer(
        config.models.pose, neutral_span=v.torso_neutral_span,
        min_visibility=v.torso_min_visibility, skip_frames=v.pose_skip_frames,
    )
    classifier = EngagementClassifier.load(config.models.engagement)
    if classifier.feature_schema != CURRENT_FEATURE_SCHEMA:
        print(
            f"[build] *** AVISO IMPORTANTE ***  {config.models.engagement} se entrenó con "
            f"la definición de características antigua (schema {classifier.feature_schema}), "
            f"pero este código calcula yaw/pitch con la nueva (schema {CURRENT_FEATURE_SCHEMA}, "
            f"solvePnP en grados — ver geometry.py). Las predicciones de este modelo NO SON "
            f"FIABLES hasta reentrenar con datos recogidos con el pipeline actual. "
            f"Ejecuta build_dataset + train tras recopilar sesiones nuevas."
        )
    return EngagementPipeline(
        detector=detector, head_pose=head_pose, torso=torso, classifier=classifier,
        zone=load_zone(config),
        gaze_reference=load_gaze_reference(config),
        counting_region=load_counting_region(config),
        engagement_params=config.engagement,
        fov_h_deg=config.camera.fov_h_deg,
        passerby_min_frames=v.passerby_min_frames,
        passerby_motion_px=v.passerby_motion_px,
        passerby_min_height_frac=v.passerby_min_height_frac,
        zone_soft_margin=config.engagement.zone_soft_margin,
        reconcile_params=ReconcileParams(
            grace_frames=v.reassoc_grace_frames, min_iou=v.reassoc_min_iou,
        ),
    )
