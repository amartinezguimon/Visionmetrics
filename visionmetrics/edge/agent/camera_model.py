"""Pinhole camera model — convert normalised face width to a real distance.

Previously duplicated between main.py and calibrate.py. The focal length is
derived once from the camera's horizontal field of view and frame width; the
distance estimate follows the standard pinhole relation:

    focal_px = (frame_width / 2) / tan(fov_h / 2)
    dist_m   = real_face_width_m * focal_px / face_width_px

``fov_h`` and ``face_width_m`` are per-camera/per-deployment values and must
come from device config — NOT hardcoded. A wrong FOV silently corrupts every
distance, which then corrupts the zone filter, so this is the #1 thing to get
right when onboarding a new camera.
"""

from __future__ import annotations

import math

import numpy as np

# Clamp distance estimates to a sane physical range (metres).
DIST_MIN_M = 0.1
DIST_MAX_M = 8.0
_EPS = 1e-6


def focal_length_px(frame_width_px: int, fov_h_deg: float) -> float:
    """Focal length in pixels from horizontal FOV and frame width."""
    return (frame_width_px / 2.0) / math.tan(math.radians(fov_h_deg / 2.0))


def intrinsics_matrix(frame_w_px: int, frame_h_px: int, fov_h_deg: float) -> np.ndarray:
    """The 3x3 camera matrix K for solvePnP, from the SAME horizontal FOV used
    everywhere else in this file (one focal length, one source of truth).

    Assumes square pixels (fx == fy) and the principal point at the exact
    frame centre — there is no per-camera lens calibration (a checkerboard
    calibration) in this system, so this is a deliberate simplification, not
    an oversight: it ignores lens distortion (a cheap wide-angle lens bends
    lines and biases angles near the frame edges) and any real optical-centre
    offset. Good enough for a moderate-FOV, no-recollection-of-images system;
    see solve_head_pose's docstring for the honest limit of that assumption.

        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    """
    fx = fy = focal_length_px(frame_w_px, fov_h_deg)
    cx, cy = frame_w_px / 2.0, frame_h_px / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def intrinsics_from_focal(focal_px: float, frame_w_px: int, frame_h_px: int) -> np.ndarray:
    """Same 3x3 camera matrix as `intrinsics_matrix`, from an ALREADY-COMPUTED
    focal length in pixels instead of re-deriving it from FOV — for callers
    (e.g. `vision/face.py`) that only receive `focal_px` (computed once per
    session by the pipeline via `focal_length_px`), so the FOV-to-focal-length
    conversion stays in exactly one place."""
    cx, cy = frame_w_px / 2.0, frame_h_px / 2.0
    return np.array([[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def distance_metres(
    face_width_norm: float,
    source_region_width_px: int,
    focal_px: float,
    *,
    face_width_m: float,
) -> float:
    """Estimate real distance to the face, in metres.

    face_width_norm        : cheekbone width normalised to the region MediaPipe saw.
    source_region_width_px : width (original, non-upscaled pixels) of that region
                             — the head-crop bbox width, or the full frame width.
    focal_px               : from :func:`focal_length_px`.
    face_width_m           : assumed real face width (per-deployment config).
    """
    face_width_px = face_width_norm * source_region_width_px
    dist = (face_width_m * focal_px) / (face_width_px + _EPS)
    return float(min(max(dist, DIST_MIN_M), DIST_MAX_M))
