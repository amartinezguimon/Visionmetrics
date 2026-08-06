"""Tests for the One-Euro filter (pure, no numpy/cv2)."""

from __future__ import annotations

import math
import random

from visionmetrics.edge.agent.one_euro import OneEuroFilter


def test_first_sample_passes_through_unchanged():
    f = OneEuroFilter()
    assert f.filter(0.0, 12.3) == 12.3


def test_constant_signal_stays_constant():
    f = OneEuroFilter()
    out = [f.filter(t / 30.0, 5.0) for t in range(60)]
    assert all(math.isclose(v, 5.0, abs_tol=1e-9) for v in out)


def test_smooths_noise_around_a_constant_value():
    # A constant "true" signal with jitter added — the filtered output must
    # vary MUCH less than the raw noisy input (that's the entire point).
    rng = random.Random(7)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)   # beta=0 -> pure jitter-killing, no speed boost
    raw = [5.0 + rng.uniform(-0.5, 0.5) for _ in range(120)]
    out = [f.filter(t / 30.0, x) for t, x in enumerate(raw)]

    def _variance(xs):
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / len(xs)

    tail_raw, tail_out = raw[30:], out[30:]   # skip the warm-up transient
    assert _variance(tail_out) < _variance(tail_raw) * 0.3


def test_tracks_a_real_step_change_eventually():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    for t in range(30):
        f.filter(t / 30.0, 0.0)
    last = None
    for t in range(30, 90):
        last = f.filter(t / 30.0, 40.0)   # a genuine, sustained jump
    assert last is not None and math.isclose(last, 40.0, abs_tol=1.0)


def test_higher_beta_reacts_faster_to_motion():
    # Same step input, two filters differing only in beta — the one with more
    # "let fast motion through" allowance must be closer to the target sooner.
    low_beta = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    high_beta = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    for t in range(30):
        low_beta.filter(t / 30.0, 0.0)
        high_beta.filter(t / 30.0, 0.0)
    low_out = high_out = None
    for t in range(30, 33):   # look shortly after the step, not fully settled
        low_out = low_beta.filter(t / 30.0, 40.0)
        high_out = high_beta.filter(t / 30.0, 40.0)
    assert high_out > low_out


def test_independent_filter_instances_do_not_share_state():
    # Guards the exact bug class this exists to avoid in production: two
    # different tracks (people) must never influence each other's smoothing.
    a = OneEuroFilter()
    b = OneEuroFilter()
    a.filter(0.0, 100.0)
    a.filter(1.0, 100.0)
    assert b.filter(0.0, -5.0) == -5.0   # unaffected by `a`'s history


def test_non_advancing_timestamp_does_not_crash_or_blow_up():
    f = OneEuroFilter()
    f.filter(1.0, 10.0)
    out = f.filter(1.0, 12.0)   # same timestamp twice (dt=0) — must not raise/div-by-zero
    assert math.isfinite(out)
