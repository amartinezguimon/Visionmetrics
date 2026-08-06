"""Head-pose geometry — the single source of truth for face angles.

This logic was previously DUPLICATED across the prototype (the old monolithic
agent and the calibration tool). Having two copies caused a real bug once (one
used eye corners while the other used cheekbones, producing a systematic zone
mismatch). It now lives here, once — every consumer (the live agent, the
training-data collector, and the calibration tool) goes through this module,
so they can never silently drift apart again.

WHY solvePnP INSTEAD OF THE OLD FLAT 2D RATIO
MediaPipe already hands back 468 3D-ish face landmarks; the previous version
of this file threw that away and estimated yaw/pitch from a flat 2D ratio
(nose position relative to the cheekbone midpoint). That ratio degrades badly
at extreme angles and on small/far faces, where a couple of noisy pixels swing
the ratio a lot. `solve_head_pose` instead fits a real 3D rotation (via
`cv2.solvePnP`) between a generic canonical face model and 5 already-validated
2D landmarks, using the camera's real per-deployment intrinsics — the accuracy
gain costs microseconds (PnP over 5 points), not frames.

BREAKING CHANGE, DELIBERATE (see ROADMAP): yaw/pitch now mean "real rotation
in degrees" instead of "a unitless 2D ratio" — a completely different numeric
scale. A model trained on the old ratios cannot correctly interpret angles
from this module; `classifier.py`'s `feature_schema` buffer + the load-time
check in `build.py` refuse to run that mismatched combination silently.

The functions are pure and framework-agnostic (no OpenCV/MediaPipe import at
module scope beyond cv2/numpy, no camera): they take any sequence of
landmarks where each landmark exposes ``.x``/``.y`` floats normalised to the
image that was fed to the face detector, and image coordinates already
converted into ORIGINAL FRAME pixels (see the docstring on `solve_head_pose`
for exactly why that mapping step matters).
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence

import cv2
import numpy as np

# MediaPipe Face Landmarker indices we rely on. Cheekbones (234/454) are used
# instead of eye corners so glasses/sunglasses never interfere — kept for BOTH
# the width->distance proxy AND (now) the solvePnP point set, so there is only
# ONE set of "which landmarks do we trust" decisions in this file, not two.
NOSE = 1
TOP = 10          # top of forehead
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

_EPS = 1e-6


class Landmark(Protocol):
    """Anything with normalised x/y coordinates (e.g. a MediaPipe landmark)."""
    x: float
    y: float


def cheekbone_width(landmarks: Sequence[Landmark]) -> float:
    """Normalised cheekbone-to-cheekbone width — the distance proxy fed to
    `camera_model.distance_metres` and to the engagement classifier. Unrelated
    to yaw/pitch (unaffected by the solvePnP change below); glasses-robust
    because cheekbones, unlike eye corners, sit clear of any frame."""
    l_cheek, r_cheek = landmarks[LEFT_CHEEK], landmarks[RIGHT_CHEEK]
    return abs(r_cheek.x - l_cheek.x)


def canonical_face_model(face_width_m: float) -> np.ndarray:
    """Generic 3D face model (metres), origin at the nose tip, for solvePnP.

    Axes: +X towards RIGHT_CHEEK (454) — the same "larger image x" side already
    validated by this module's own tests — +Y up (towards TOP/forehead), +Z
    towards the camera (the nose tip is the most forward point on a face).

    The cheek-to-cheek width comes straight from `face_width_m` — the SAME
    assumed face size already used for the distance estimate elsewhere in this
    codebase, so there is one face-size assumption, not two independently
    guessed ones. The other proportions (nose-to-chin, nose-to-forehead, how
    far the cheeks sit behind the nose) are reasoned generic head proportions,
    NOT a lab face scan — an approximation in the same honest spirit as
    `face_width_m` itself. `solvePnP` is tolerant of proportion error in the
    recovered ROTATION (it mostly biases the estimated depth/scale instead);
    what would be a real bug is a wrong LEFT/RIGHT or UP/DOWN sign, which
    `test_geometry.py`'s synthetic-rotation round-trip checks directly.
    """
    hw = face_width_m / 2.0
    return np.array([
        (0.0, 0.0, 0.0),                            # NOSE (1) — origin, most forward
        (0.0,  0.60 * face_width_m, -0.50 * hw),    # TOP / forehead (10)
        (0.0, -0.65 * face_width_m, -0.35 * hw),    # CHIN (152)
        (-hw, -0.05 * face_width_m, -0.60 * hw),    # LEFT_CHEEK (234)
        (hw, -0.05 * face_width_m, -0.60 * hw),     # RIGHT_CHEEK (454)
    ], dtype=np.float64)


def _rotation_matrix_to_euler_deg(rot: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from a 3x3 rotation matrix — the standard
    X-Y-Z Euler decomposition (R = Rz(roll) @ Ry(yaw) @ Rx(pitch))."""
    sy = math.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    if sy < 1e-6:   # gimbal-lock edge case (near-vertical looking straight up/down)
        pitch = math.atan2(-rot[1, 2], rot[1, 1])
        yaw = math.atan2(-rot[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(rot[2, 1], rot[2, 2])
        yaw = math.atan2(-rot[2, 0], sy)
        roll = math.atan2(rot[1, 0], rot[0, 0])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def solve_head_pose(
    landmarks: Sequence[Landmark],
    roi_offset_px: tuple[float, float],
    roi_size_px: tuple[float, float],
    camera_matrix: np.ndarray,
    *,
    face_width_m: float,
    max_reprojection_error_px: float = 15.0,
) -> tuple[float, float, float, float] | None:
    """Real 3D head rotation via `cv2.solvePnP` — replaces the old flat 2D
    ratio. Returns ``(yaw_deg, pitch_deg, roll_deg, reprojection_error_px)``,
    or ``None`` when the solve is too unreliable to trust (see below).

    ``roi_offset_px``/``roi_size_px`` are the ORIGINAL (non-upscaled) pixel
    offset and size of the region MediaPipe actually saw (e.g. the upscaled,
    padded head crop). Landmarks come back normalised [0,1] relative to
    whatever image was fed to MediaPipe; since normalised coordinates don't
    care about resolution, ``landmark.x * roi_w`` already undoes any upscale
    and lands in the ORIGINAL crop's pixel grid, and adding the crop's own
    frame offset maps it into ORIGINAL FRAME pixel coordinates — the SAME
    space ``camera_matrix`` (built from the full frame's width/height and the
    camera's real FOV) was defined in. Solving PnP directly in "upscaled crop"
    pixel space instead would silently use the wrong effective focal length —
    this mapping step is what avoids that.

    No lens-distortion coefficients are used (``camera_matrix`` in turn assumes
    no calibration board — see its own docstring): a cheap wide-angle lens's
    real barrel distortion will still bias angles somewhat near the frame
    edges. Documented, not hidden — fixing it needs a checkerboard calibration
    per camera model, out of scope here.

    A reprojection-error sanity check stands in for a full RANSAC pass (see
    ROADMAP): a face too close to edge-on, or any other near-degenerate point
    configuration, can make solvePnP converge on a flipped/wrong pose. Instead
    of trusting that blindly, the solved pose is projected back and compared
    to what was actually detected; a high mismatch returns ``None`` (handled
    identically to "no face found" by every caller) rather than feeding a
    garbage angle to the classifier.
    """
    rx1, ry1 = roi_offset_px
    roi_w, roi_h = roi_size_px
    if roi_w <= 0 or roi_h <= 0:
        return None

    def to_frame_px(lm: Landmark) -> tuple[float, float]:
        return (rx1 + lm.x * roi_w, ry1 + lm.y * roi_h)

    image_points = np.array([
        to_frame_px(landmarks[NOSE]), to_frame_px(landmarks[TOP]),
        to_frame_px(landmarks[CHIN]), to_frame_px(landmarks[LEFT_CHEEK]),
        to_frame_px(landmarks[RIGHT_CHEEK]),
    ], dtype=np.float64)

    # Degenerate-input guard, BEFORE solving: 5 (near-)coincident 2D points have
    # no real geometry in them, yet EPnP doesn't raise on that — it just returns
    # some solution, and the reprojection-error check below can't catch it
    # (that "solution" reprojects right back to the same coincident points with
    # ~zero error). A real detected face — even a tiny, far, 15px-wide one —
    # always has its 5 landmarks spread well beyond this.
    bbox_diag = math.hypot(*(image_points.max(axis=0) - image_points.min(axis=0)))
    if bbox_diag < 3.0:
        return None

    model_points = canonical_face_model(face_width_m)
    dist_coeffs = np.zeros((4, 1))   # see docstring: no per-camera lens calibration

    # EPnP (not the ITERATIVE default): ITERATIVE's own DLT initial guess needs
    # >= 6 points and raises below that; EPnP works from n >= 4 and handles our
    # 5-point set (nose, forehead, chin, both cheeks) directly.
    ok, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return None

    projected, _ = cv2.projectPoints(model_points, rvec, tvec, camera_matrix, dist_coeffs)
    err_px = float(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1).mean())
    if not math.isfinite(err_px) or err_px > max_reprojection_error_px:
        return None

    rot_mat, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _rotation_matrix_to_euler_deg(rot_mat)
    if not all(math.isfinite(v) for v in (yaw, pitch, roll)):
        return None
    return yaw, pitch, roll, err_px


def relative_neck_yaw(
    nose_x: float, left_shoulder_x: float, right_shoulder_x: float,
    *, min_span: float = 0.02,
) -> float | None:
    """Head-vs-torso yaw: how far the head is turned relative to the body axis.

    Near 0 = head aligned with torso; positive = head turned right of the body.
    Returns ``None`` when the shoulders are nearly edge-on (span too small to
    trust). Camera-position independent — useful as a calibration signal.
    Unrelated to the yaw/pitch change above (still a 2D ratio on purpose: this
    one is a coarse walk-by damping signal, not fed to the classifier).
    """
    span = left_shoulder_x - right_shoulder_x
    if abs(span) <= min_span:
        return None
    shoulder_mid_x = (left_shoulder_x + right_shoulder_x) / 2.0
    return (nose_x - shoulder_mid_x) / (abs(span) + _EPS)


def torso_confidence(
    left_shoulder_x: float, right_shoulder_x: float, *, neutral_span: float,
) -> float:
    """How squarely the torso faces the camera, in [0, 1].

    1.0 = shoulders fully facing the camera (wide span); 0.0 = edge-on.
    ``neutral_span`` is the expected shoulder span when facing the camera.
    """
    span = left_shoulder_x - right_shoulder_x
    return max(0.0, min(span, neutral_span)) / neutral_span
