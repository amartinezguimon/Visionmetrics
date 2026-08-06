"""Tests for the pinhole camera model (pure)."""

import math

import numpy as np

from visionmetrics.edge.agent import camera_model as cm


def test_focal_length_known_value():
    # 90 deg FOV on a 1000px wide frame => focal = 500 / tan(45) = 500.
    assert math.isclose(cm.focal_length_px(1000, 90.0), 500.0, rel_tol=1e-9)


def test_distance_inverse_to_face_width():
    focal = cm.focal_length_px(1280, 70.0)
    near = cm.distance_metres(0.20, 640, focal, face_width_m=0.16)
    far = cm.distance_metres(0.05, 640, focal, face_width_m=0.16)
    assert far > near  # smaller normalised face => farther away


def test_distance_clamped_to_physical_range():
    focal = cm.focal_length_px(1280, 70.0)
    huge = cm.distance_metres(1e-9, 640, focal, face_width_m=0.16)
    tiny = cm.distance_metres(10.0, 640, focal, face_width_m=0.16)
    assert huge == cm.DIST_MAX_M
    assert tiny == cm.DIST_MIN_M


def test_intrinsics_matrix_shape_and_focal():
    K = cm.intrinsics_matrix(1280, 720, 70.0)
    assert K.shape == (3, 3)
    expected_focal = cm.focal_length_px(1280, 70.0)
    assert math.isclose(K[0, 0], expected_focal, rel_tol=1e-9)   # fx
    assert math.isclose(K[1, 1], expected_focal, rel_tol=1e-9)   # fy == fx (square pixels)


def test_intrinsics_matrix_principal_point_is_frame_centre():
    K = cm.intrinsics_matrix(1280, 720, 70.0)
    assert math.isclose(K[0, 2], 640.0)   # cx
    assert math.isclose(K[1, 2], 360.0)   # cy
    assert K[2, 2] == 1.0


def test_intrinsics_from_focal_matches_intrinsics_matrix():
    focal = cm.focal_length_px(1280, 70.0)
    a = cm.intrinsics_matrix(1280, 720, 70.0)
    b = cm.intrinsics_from_focal(focal, 1280, 720)
    assert np.allclose(a, b)
