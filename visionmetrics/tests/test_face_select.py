"""Tests for most_centred_face — picking the right face when two are in a crop
— and upscale_factor, the per-frame compute-saving cap on head-crop upscaling."""

from __future__ import annotations

from visionmetrics.edge.agent import geometry
from visionmetrics.edge.agent.vision.face import most_centred_face, upscale_factor


class _LM:
    def __init__(self, x):
        self.x = x
        self.y = 0.0


def _face(nose_x):
    """A fake landmark list where only the NOSE index's x matters here."""
    lm = [_LM(0.0) for _ in range(max(geometry.NOSE, geometry.RIGHT_CHEEK) + 1)]
    lm[geometry.NOSE] = _LM(nose_x)
    return lm


def test_single_face_is_returned_unchanged():
    f = _face(0.2)
    assert most_centred_face([f]) is f


def test_picks_the_centred_face_over_an_edge_one():
    centred = _face(0.52)     # near the crop centre (0.5) -> the box owner
    intruder = _face(0.05)    # off to the side -> an adjacent person
    assert most_centred_face([intruder, centred]) is centred
    assert most_centred_face([centred, intruder]) is centred


def test_ties_keep_the_first():
    a = _face(0.4)
    b = _face(0.6)            # equidistant from 0.5
    assert most_centred_face([a, b]) is a


# ── upscale_factor ───────────────────────────────────────────────────────

def test_small_far_crop_gets_the_full_configured_factor():
    # A tiny ~30px crop is nowhere near the target cap — unchanged behaviour.
    assert upscale_factor(30, 30, head_upscale=4, target_px=220) == 4.0


def test_large_close_crop_is_barely_upscaled():
    # A crop already bigger than the target needs no real upscaling at all —
    # this is the actual compute saved for close-up people in a crowd.
    f = upscale_factor(300, 300, head_upscale=4, target_px=220)
    assert 1.0 <= f < 1.1


def test_mid_size_crop_scales_just_enough_to_reach_the_target():
    f = upscale_factor(100, 80, head_upscale=4, target_px=220)
    assert f == 2.2   # 220 / 100 (the longer side)


def test_never_upscales_below_1x():
    # A crop already larger than target_px even at head_upscale=1 must not be
    # shrunk — 1.0 is the floor, never < 1.
    assert upscale_factor(500, 400, head_upscale=4, target_px=220) == 1.0


def test_degenerate_zero_size_roi_is_safe():
    assert upscale_factor(0, 50, head_upscale=4) == 1.0
    assert upscale_factor(50, 0, head_upscale=4) == 1.0
