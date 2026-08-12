"""Tests for VideoSource using a synthetic video file (no real camera)."""

import numpy as np
import cv2
import pytest

from visionmetrics.edge.agent.capture import VideoSource, _is_realtime


def test_source_kind_detection():
    assert _is_realtime(0) is True
    assert _is_realtime("rtsp://10.0.0.1/stream") is True
    assert _is_realtime("https://cam/feed") is True
    assert _is_realtime("fixtures/clip.mp4") is False
    assert _is_realtime("C:/videos/test.avi") is False


def _write_clip(path, n_frames=12, w=64, h=48):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h))
    assert writer.isOpened(), "codec MJPG/.avi not available"
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 5 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_file_source_reads_every_frame_in_order(tmp_path):
    clip = tmp_path / "clip.avi"
    _write_clip(clip, n_frames=12)

    src = VideoSource(str(clip))
    assert src.realtime is False
    assert src.open() is True

    count = 0
    while True:
        ok, frame = src.read()
        if not ok or frame is None:
            break
        assert frame.shape[:2] == (48, 64)
        count += 1
    src.release()

    assert count == 12  # sequential, nothing dropped


def test_open_returns_false_for_bad_source():
    src = VideoSource("does_not_exist_12345.avi")
    assert src.open() is False


# ── macOS phantom-index guards ────────────────────────────────────────────────
# On macOS, asking AVFoundation for a camera index at or beyond the number of
# enumerated devices does NOT fail — it silently returns the Mac's own built-in
# camera. That is the root of the recurring "I picked my phone but the Mac's
# webcam runs" bug, and it is reachable whenever a saved index outlives the
# camera it pointed at (phone sleeps / disconnects between picking and running).

def _darwin_with_cameras(monkeypatch, names):
    """Pretend we're on macOS with exactly `names` enumerated, and make probing
    always fail so the tested code path is the index logic, not real hardware."""
    import sys as _sys

    from visionmetrics.edge.agent import capture as cap
    monkeypatch.setattr(_sys, "platform", "darwin")
    monkeypatch.setattr(cap, "_mac_camera_names", lambda: list(names))
    monkeypatch.setattr(cap, "_probe_camera", lambda *a, **k: (False, False))
    return cap


def test_pick_chosen_honours_visual_pick_even_when_named_like_the_builtin(monkeypatch):
    """The whole point of honouring the click: on some Macs system_profiler's
    order disagrees with cv2's, so index 0 can stream the iPhone while being
    NAMED "FaceTime HD Camera". Name-based rejection here is what used to drag
    a good pick back onto the Mac, so an in-range index must be honoured as-is."""
    cap = _darwin_with_cameras(monkeypatch, ["FaceTime HD Camera", "hector Camera"])
    assert cap.pick_chosen_external("", 0) == 0


def test_pick_chosen_refuses_a_phantom_index_instead_of_opening_the_mac(monkeypatch):
    """Saved index 1, but the phone is gone and only one camera is enumerated:
    index 1 no longer exists. Honouring it would hand back the Mac's built-in,
    so we must refuse (None = don't run) rather than silently film the operator."""
    cap = _darwin_with_cameras(monkeypatch, ["FaceTime HD Camera"])
    assert cap.pick_chosen_external("", 1) is None


def test_pick_chosen_does_not_second_guess_when_camera_list_unreadable(monkeypatch):
    """If system_profiler can't be read we have no COUNT to bound by; degrade to
    honouring the pick rather than refusing every camera."""
    cap = _darwin_with_cameras(monkeypatch, [])
    assert cap.pick_chosen_external("", 1) == 1


def test_video_source_refuses_phantom_index_even_without_avoid_builtin(monkeypatch):
    """Second, independent net. The 'honour the visual pick' path deliberately
    runs with avoid_builtin=False (macOS NAMES are untrustworthy), but the camera
    COUNT still is trustworthy — so a phantom index must be rejected there too.
    None means 'open nothing, retry', letting a sleeping phone come back."""
    cap = _darwin_with_cameras(monkeypatch, ["FaceTime HD Camera"])
    src = cap.VideoSource(1, avoid_builtin=False)
    assert src._resolve_source() is None

    in_range = cap.VideoSource(0, avoid_builtin=False)
    assert in_range._resolve_source() == 0
