"""Head-pose analysis (MediaPipe Face Landmarker + solvePnP) — Layers 2 & 3.

Crops the head region (top fraction of the person bbox), upscales it so far
faces become detectable, runs MediaPipe, and turns the resulting landmarks
into (yaw, pitch, distance, dist_m) via `geometry.solve_head_pose` (real 3D
rotation) + the shared camera model, then smooths yaw/pitch per track with a
One-Euro filter so per-frame landmark jitter doesn't reach the classifier.

Per-track frame-skip cache: MediaPipe is the expensive call, so we only run it
every N frames per person and reuse the last result in between (matches the
prototype's behavior).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .. import camera_model, geometry
from ..one_euro import OneEuroFilter


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    distance: float            # normalised cheekbone width (classifier feature)
    dist_m: float              # real distance in metres
    nose_px: tuple[int, int]   # nose position in original-frame coords (for debug draw)


_UPSCALE_TARGET_PX = 220   # cap on the upscaled crop's longer side — see upscale_factor()


def upscale_factor(roi_w: int, roi_h: int, head_upscale: float,
                    target_px: float = _UPSCALE_TARGET_PX) -> float:
    """How much to upscale a head crop before handing it to MediaPipe.

    `head_upscale` exists to make a FAR/small crop big enough to detect (a
    ~30px far-face crop needs the full configured factor). A close-up crop
    that's already large gains nothing from also being multiplied up — it's
    pure wasted work for the same landmarks, and with a crowd this is where
    the per-person cost actually goes. Never upscale PAST `target_px` on the
    longer side, whatever `head_upscale` says; a genuinely small/far crop is
    nowhere near that cap and still gets the full configured factor, unchanged
    from before. Pure — no cv2/MediaPipe — so it's directly unit-testable.
    """
    if roi_w <= 0 or roi_h <= 0:
        return 1.0
    return min(head_upscale, max(1.0, target_px / max(roi_w, roi_h)))


def most_centred_face(faces):
    """Pick the face whose nose sits nearest the horizontal centre of the crop.

    The head ROI is derived from ONE person's bounding box, so that person's
    head is roughly centred in it. When two people stand very close (a couple,
    a hug) an adjacent face can intrude into the crop; MediaPipe would otherwise
    return an arbitrary one and the pose would flicker between the two. Choosing
    the most-centred face attributes the crop to its rightful owner. With a
    single face this is a no-op.
    """
    best, best_d = faces[0], abs(faces[0][geometry.NOSE].x - 0.5)
    for f in faces[1:]:
        d = abs(f[geometry.NOSE].x - 0.5)
        if d < best_d:
            best, best_d = f, d
    return best


class HeadPoseAnalyzer:
    def __init__(
        self, model_path: str, *, face_width_m: float, head_crop_frac: float,
        head_upscale: int, skip_frames: int, pad: int = 30,
        min_detection_confidence: float = 0.25,
        one_euro_min_cutoff: float = 1.0, one_euro_beta: float = 0.007,
    ):
        base = python.BaseOptions(model_asset_path=model_path)
        opts = vision.FaceLandmarkerOptions(
            base_options=base, num_faces=2,   # detect up to 2 so we can disambiguate
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
        )
        self._detector = vision.FaceLandmarker.create_from_options(opts)
        self.face_width_m = face_width_m
        self.head_crop_frac = head_crop_frac
        self.head_upscale = head_upscale
        self.skip_frames = skip_frames
        self.pad = pad
        self._one_euro_min_cutoff = one_euro_min_cutoff
        self._one_euro_beta = one_euro_beta
        self._cache: dict[int, tuple[int, HeadPose | None]] = {}
        # Per-track smoothing state (yaw filter, pitch filter) — NEVER shared
        # across two tracks; a fresh pair is created the first time a track
        # id is seen and dropped in forget() so a ByteTrack id reassignment
        # can never smear one person's motion history onto another's.
        self._filters: dict[int, tuple[OneEuroFilter, OneEuroFilter]] = {}

    def head_region(self, frame, bbox) -> tuple[int, int, int, int]:
        """Compute the padded head ROI (top `head_crop_frac` of the bbox)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        head_y2 = y1 + int((y2 - y1) * self.head_crop_frac)
        return (
            max(0, x1 - self.pad), max(0, y1 - self.pad),
            min(w, x2 + self.pad), min(h, head_y2 + self.pad),
        )

    def analyze(self, frame, bbox, track_id: int, frame_idx: int,
                focal_px: float, now: float) -> HeadPose | None:
        """Return the head pose for one person, using the frame-skip cache.

        ``now``: wall-clock (or, for a recorded file, video-time) seconds —
        feeds the per-track One-Euro filters, which smooth by SPEED, not by
        frame count, so behaviour doesn't change if the frame rate does.
        """
        idx, cached = self._cache.get(track_id, (-(10**9), None))
        if frame_idx - idx < self.skip_frames:
            return cached

        rx1, ry1, rx2, ry2 = self.head_region(frame, bbox)
        roi = frame[ry1:ry2, rx1:rx2]
        pose = None
        if roi.size > 0:
            frame_h, frame_w = frame.shape[:2]
            camera_matrix = camera_model.intrinsics_from_focal(focal_px, frame_w, frame_h)
            pose = self._detect(roi, (rx1, ry1, rx2, ry2), camera_matrix, track_id, now)
        self._cache[track_id] = (frame_idx, pose)
        return pose

    def _detect(self, roi, region, camera_matrix: np.ndarray,
                track_id: int, now: float) -> HeadPose | None:
        rx1, ry1, rx2, ry2 = region
        roi_h, roi_w = roi.shape[:2]
        s = upscale_factor(roi_w, roi_h, self.head_upscale)
        up = cv2.resize(roi, (max(1, round(roi_w * s)), max(1, round(roi_h * s))),
                        interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
        result = self._detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return None
        lm = most_centred_face(result.face_landmarks)

        pose_3d = geometry.solve_head_pose(
            lm, (rx1, ry1), (roi_w, roi_h), camera_matrix,
            face_width_m=self.face_width_m,
        )
        if pose_3d is None:
            return None
        yaw, pitch, _roll, _reproj_err = pose_3d
        yaw, pitch = self._smooth(track_id, now, yaw, pitch)

        distance = geometry.cheekbone_width(lm)   # unrelated to the angle above
        dist_m = camera_model.distance_metres(
            distance, rx2 - rx1, camera_matrix[0, 0], face_width_m=self.face_width_m)
        nose_px = (rx1 + int(lm[geometry.NOSE].x * roi_w),
                   ry1 + int(lm[geometry.NOSE].y * roi_h))
        return HeadPose(yaw, pitch, distance, dist_m, nose_px)

    def _smooth(self, track_id: int, now: float, yaw: float, pitch: float) -> tuple[float, float]:
        yaw_f, pitch_f = self._filters.setdefault(
            track_id,
            (OneEuroFilter(min_cutoff=self._one_euro_min_cutoff, beta=self._one_euro_beta),
             OneEuroFilter(min_cutoff=self._one_euro_min_cutoff, beta=self._one_euro_beta)),
        )
        return yaw_f.filter(now, yaw), pitch_f.filter(now, pitch)

    def forget(self, track_id: int) -> None:
        self._cache.pop(track_id, None)
        self._filters.pop(track_id, None)
