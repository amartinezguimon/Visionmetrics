"""Tests for the offline-labeling frame sampler (pure logic, no cv2/models).

The video I/O + detector/head-pose loop in `prep.run()` needs a real video and
models and isn't unit-tested here; that path is exercised manually / via the
edge tests that already cover the detector and head-pose analyzer.
"""

from __future__ import annotations

from visionmetrics.training.prep import FrameSampler


def _people(*ids: int) -> list[dict]:
    return [{"id": i, "box": [0, 0, 10, 10]} for i in ids]


def test_empty_frame_never_kept():
    s = FrameSampler(sample_every=5)
    for i in range(20):
        assert s.should_keep(i, []) is False


def test_first_frame_of_a_new_track_always_kept():
    s = FrameSampler(sample_every=100)  # huge stride so only "new track" can trigger
    assert s.should_keep(0, _people(1)) is True
    # same track, well before the stride would fire again -> not kept
    assert s.should_keep(1, _people(1)) is False
    assert s.should_keep(2, _people(1)) is False
    # a second, different person shows up -> kept immediately regardless of stride
    assert s.should_keep(3, _people(1, 2)) is True


def test_throttles_a_lingering_person_by_sample_every():
    s = FrameSampler(sample_every=5)
    kept = [i for i in range(21) if s.should_keep(i, _people(1))]
    # frame 0 kept (new track), then every 5th frame after that
    assert kept == [0, 5, 10, 15, 20]


def test_brief_visit_is_never_silently_dropped():
    """A person who appears and leaves inside one throttle window must still
    get at least one labeled frame, even if their whole visit never lines up
    with the stride boundary."""
    s = FrameSampler(sample_every=10)
    kept = []
    for i in range(4):  # frames 0-3: person visible for a very short time
        if s.should_keep(i, _people(1)):
            kept.append(i)
    assert kept == [0]  # first-appearance trigger fired even though 10 never elapsed


def test_sample_every_clamped_to_at_least_one():
    s = FrameSampler(sample_every=0)
    kept = [i for i in range(3) if s.should_keep(i, _people(1))]
    assert kept == [0, 1, 2]  # stride of 0 would be a ZeroDivisionError-ish footgun; clamp to 1
