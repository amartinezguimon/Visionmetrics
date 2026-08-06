"""Tests for head-pose geometry (pure math, no camera/MediaPipe).

The solvePnP path is verified by a SYNTHETIC ROUND TRIP, not just "it doesn't
crash": build a KNOWN 3D rotation, project the canonical face model through it
with a known camera, feed the resulting 2D points back into `solve_head_pose`,
and assert the recovered angle matches the known input. This is the strongest
verification available without a real labelled photo — it catches sign flips,
axis mixups, and coordinate-mapping bugs directly, rather than trusting the
Euler-decomposition formula by inspection alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from visionmetrics.edge.agent import camera_model as cm
from visionmetrics.edge.agent import geometry as g

FACE_WIDTH_M = 0.16
FRAME_W, FRAME_H = 640, 480
FOV_H_DEG = 70.0


@dataclass
class P:
    x: float
    y: float


def make_landmarks(nose, top, chin, lcheek, rcheek):
    """Build a sparse landmark list indexed at the points geometry reads."""
    lm = [P(0, 0)] * 460
    lm[g.NOSE] = nose
    lm[g.TOP] = top
    lm[g.CHIN] = chin
    lm[g.LEFT_CHEEK] = lcheek
    lm[g.RIGHT_CHEEK] = rcheek
    return lm


# ── cheekbone_width (unrelated to the solvePnP change) ─────────────────────

def test_cheekbone_width_is_the_normalised_span():
    lm = make_landmarks(
        nose=P(0.5, 0.5), top=P(0.5, 0.2), chin=P(0.5, 0.8),
        lcheek=P(0.4, 0.5), rcheek=P(0.6, 0.5),
    )
    assert math.isclose(g.cheekbone_width(lm), 0.2, rel_tol=1e-9)


# ── synthetic round-trip helpers ────────────────────────────────────────────

def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _euler_deg_to_R(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Inverse of geometry._rotation_matrix_to_euler_deg's convention
    (R = Rz(roll) @ Ry(yaw) @ Rx(pitch)), for building known-rotation fixtures."""
    y, p, r = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    return _rot_z(r) @ _rot_y(y) @ _rot_x(p)


def _synthetic_landmarks(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0,
                          *, depth_m: float = 0.7, dx_frac: float = 0.0, dy_frac: float = 0.0):
    """Project the canonical face model through a KNOWN rotation/translation
    and a known camera, and wrap the result as landmarks normalised to the
    FULL FRAME (i.e. as if the "head crop" were the whole frame at (0,0)) —
    exactly the object `solve_head_pose` expects, built from ground truth
    instead of a real detector."""
    camera_matrix = cm.intrinsics_matrix(FRAME_W, FRAME_H, FOV_H_DEG)
    model = g.canonical_face_model(FACE_WIDTH_M)
    rvec, _ = cv2.Rodrigues(_euler_deg_to_R(yaw_deg, pitch_deg, roll_deg))
    tvec = np.array([dx_frac * depth_m, dy_frac * depth_m, depth_m], dtype=np.float64)
    projected, _ = cv2.projectPoints(model, rvec, tvec, camera_matrix, np.zeros((4, 1)))
    px = projected.reshape(-1, 2)
    lm = [P(x / FRAME_W, y / FRAME_H) for x, y in px]
    landmarks = make_landmarks(nose=lm[0], top=lm[1], chin=lm[2], lcheek=lm[3], rcheek=lm[4])
    return landmarks, camera_matrix


# ── solve_head_pose: the real correctness check ─────────────────────────────

def test_solve_head_pose_recovers_zero_rotation():
    landmarks, K = _synthetic_landmarks(0.0, 0.0, 0.0)
    result = g.solve_head_pose(landmarks, (0.0, 0.0), (FRAME_W, FRAME_H), K,
                                face_width_m=FACE_WIDTH_M)
    assert result is not None
    yaw, pitch, roll, err = result
    assert abs(yaw) < 1.0 and abs(pitch) < 1.0 and abs(roll) < 1.0
    assert err < 1e-3   # noiseless synthetic projection — near-zero reprojection error


def test_solve_head_pose_recovers_known_yaw():
    for known_yaw in (-30.0, -15.0, 15.0, 30.0):
        landmarks, K = _synthetic_landmarks(known_yaw, 0.0, 0.0)
        result = g.solve_head_pose(landmarks, (0.0, 0.0), (FRAME_W, FRAME_H), K,
                                    face_width_m=FACE_WIDTH_M)
        assert result is not None, f"solve failed for yaw={known_yaw}"
        yaw, pitch, roll, err = result
        assert math.isclose(yaw, known_yaw, abs_tol=1.5), f"yaw {yaw} != {known_yaw}"
        assert abs(pitch) < 2.0
        assert err < 1e-3


def test_solve_head_pose_recovers_known_pitch():
    for known_pitch in (-20.0, 20.0):
        landmarks, K = _synthetic_landmarks(0.0, known_pitch, 0.0)
        result = g.solve_head_pose(landmarks, (0.0, 0.0), (FRAME_W, FRAME_H), K,
                                    face_width_m=FACE_WIDTH_M)
        assert result is not None
        yaw, pitch, roll, err = result
        assert math.isclose(pitch, known_pitch, abs_tol=1.5), f"pitch {pitch} != {known_pitch}"
        assert abs(yaw) < 2.0


def test_solve_head_pose_yaw_sign_matches_right_cheek_side():
    # A rotation that turns the model's +X (RIGHT_CHEEK) side towards the
    # camera must have a CONSISTENT, non-ambiguous sign — whatever that sign
    # is, it must not flip between two otherwise-identical calls.
    left_landmarks, K = _synthetic_landmarks(-25.0, 0.0, 0.0)
    right_landmarks, _ = _synthetic_landmarks(25.0, 0.0, 0.0)
    left_result = g.solve_head_pose(left_landmarks, (0.0, 0.0), (FRAME_W, FRAME_H), K,
                                     face_width_m=FACE_WIDTH_M)
    right_result = g.solve_head_pose(right_landmarks, (0.0, 0.0), (FRAME_W, FRAME_H), K,
                                      face_width_m=FACE_WIDTH_M)
    assert left_result[0] < 0 < right_result[0]


def test_solve_head_pose_maps_through_a_head_crop_correctly():
    """The exact same face, seen only through a cropped+offset ROI (as the
    real pipeline always does — see HeadPoseAnalyzer.head_region), must
    recover the SAME angle as seeing it in the full, uncropped frame. This is
    the specific gotcha flagged before implementing this: solving PnP directly
    in crop-pixel space (instead of mapping back to frame-space first) would
    silently use the wrong effective focal length."""
    camera_matrix = cm.intrinsics_matrix(FRAME_W, FRAME_H, FOV_H_DEG)
    model = g.canonical_face_model(FACE_WIDTH_M)
    rvec, _ = cv2.Rodrigues(_euler_deg_to_R(18.0, 5.0, 0.0))
    tvec = np.array([0.05, 0.02, 0.7], dtype=np.float64)   # off-centre, so the crop isn't (0,0)
    projected, _ = cv2.projectPoints(model, rvec, tvec, camera_matrix, np.zeros((4, 1)))
    px = projected.reshape(-1, 2)

    full_frame_result = g.solve_head_pose(
        make_landmarks(*(P(x / FRAME_W, y / FRAME_H) for x, y in px)),
        (0.0, 0.0), (FRAME_W, FRAME_H), camera_matrix, face_width_m=FACE_WIDTH_M)

    # Now express the SAME points as landmarks inside an arbitrary head-crop
    # ROI (offset + smaller size + "upscaled" — normalised coords don't care).
    rx1, ry1, roi_w, roi_h = 150.0, 80.0, 220.0, 260.0
    crop_landmarks = make_landmarks(*(P((x - rx1) / roi_w, (y - ry1) / roi_h) for x, y in px))
    crop_result = g.solve_head_pose(
        crop_landmarks, (rx1, ry1), (roi_w, roi_h), camera_matrix, face_width_m=FACE_WIDTH_M)

    assert full_frame_result is not None and crop_result is not None
    assert math.isclose(full_frame_result[0], crop_result[0], abs_tol=0.05)
    assert math.isclose(full_frame_result[1], crop_result[1], abs_tol=0.05)


def test_solve_head_pose_none_on_degenerate_input():
    # All 5 points identical -> no real geometry to solve; must not raise,
    # must not fabricate a confident-looking angle from noise.
    lm = make_landmarks(*(P(0.5, 0.5) for _ in range(5)))
    K = cm.intrinsics_matrix(FRAME_W, FRAME_H, FOV_H_DEG)
    result = g.solve_head_pose(lm, (0.0, 0.0), (FRAME_W, FRAME_H), K, face_width_m=FACE_WIDTH_M)
    assert result is None


def test_solve_head_pose_rejects_high_reprojection_error():
    landmarks, K = _synthetic_landmarks(0.0, 0.0, 0.0)
    # Corrupt one point far from where a real face landmark could plausibly be.
    landmarks[g.NOSE] = P(0.95, 0.05)
    result = g.solve_head_pose(landmarks, (0.0, 0.0), (FRAME_W, FRAME_H), K,
                                face_width_m=FACE_WIDTH_M, max_reprojection_error_px=15.0)
    assert result is None


# ── unrelated geometry helpers (unchanged behaviour) ────────────────────────

def test_relative_neck_yaw_none_for_edge_on_shoulders():
    assert g.relative_neck_yaw(0.5, 0.50, 0.49) is None  # span 0.01 < 0.02


def test_relative_neck_yaw_zero_when_head_aligned():
    val = g.relative_neck_yaw(nose_x=0.5, left_shoulder_x=0.6, right_shoulder_x=0.4)
    assert abs(val) < 1e-6


def test_torso_confidence_clamps_to_unit_range():
    assert g.torso_confidence(0.7, 0.1, neutral_span=0.4) == 1.0   # wide -> full
    assert g.torso_confidence(0.5, 0.5, neutral_span=0.4) == 0.0   # edge-on -> 0
    assert math.isclose(g.torso_confidence(0.6, 0.4, neutral_span=0.4), 0.5)
