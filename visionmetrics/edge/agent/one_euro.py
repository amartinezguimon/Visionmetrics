"""One-Euro filter — smooths a noisy signal without adding perceptible lag.

Standard reference algorithm (Casiez, Roussel & Vogel, "1€ Filter: A Simple
Speed-based Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012).
The trick that makes it better than a flat exponential-moving-average for this
use case: the smoothing strength adapts to how fast the signal is currently
changing — heavy smoothing when the head is roughly still (killing jitter),
almost none while it is genuinely turning (so a real look-away isn't smeared
across several frames of lag).

Pure Python/math, no OpenCV/MediaPipe/numpy dependency — trivially unit-
testable, and cheap enough to run per-track, per-frame without it ever
registering on a latency budget.
"""

from __future__ import annotations

import math


def _alpha(cutoff: float, dt: float) -> float:
    """Exponential-smoothing weight for a given cutoff frequency and time step."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _LowPassFilter:
    """A single exponential low-pass step; `_alpha` is recomputed by the
    caller each call, since One-Euro's whole point is a signal-dependent
    cutoff rather than a fixed one."""

    def __init__(self) -> None:
        self._filtered: float | None = None
        self._raw: float | None = None

    def apply(self, value: float, alpha: float) -> float:
        if self._filtered is None:
            self._filtered = value
        else:
            self._filtered = alpha * value + (1.0 - alpha) * self._filtered
        self._raw = value
        return self._filtered

    @property
    def raw(self) -> float | None:
        return self._raw


class OneEuroFilter:
    """One filter instance per SCALAR signal (e.g. one for yaw, one for pitch)
    and per TRACKED PERSON — never shared across two different people's
    tracks, or one person's smoothing state leaks into another's the moment
    ByteTrack reassigns a raw id (the exact bug class flagged before building
    this: see `HeadPoseAnalyzer._filters`, keyed by CANONICAL track id and
    cleared in `forget()`, the same lifecycle already used for the landmark
    cache).

    ``min_cutoff``: baseline smoothing when the signal is roughly still — lower
    means smoother (more jitter killed) but slower to react to a genuine change.
    ``beta``: how much a fast-changing signal is allowed to cut through the
    smoothing — higher means a real head-turn is tracked with less lag, at the
    cost of passing a bit more noise during it. Defaults are the values from
    the original paper's reference implementation, a reasonable starting point
    for a human-motion-speed signal; re-tune from real footage if it under- or
    over-smooths in practice.
    """

    def __init__(self, *, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPassFilter()
        self._dx = _LowPassFilter()
        self._t_prev: float | None = None

    def filter(self, t: float, x: float) -> float:
        """Feed one new (timestamp_seconds, raw_value) sample; returns the
        smoothed value. The FIRST call for a fresh filter returns `x`
        unchanged (nothing to smooth against yet)."""
        if self._t_prev is None:
            self._t_prev = float(t)
            return self._x.apply(x, alpha=1.0)   # alpha=1 -> filtered = x, seeds the state

        dt = float(t) - self._t_prev
        if dt <= 0.0:
            # A non-advancing or out-of-order timestamp (a repeated frame_idx,
            # clock jitter): reuse the smallest sane step instead of a div-by-
            # zero or a runaway alpha.
            dt = 1e-3
        self._t_prev = float(t)

        prev_x = self._x.raw if self._x.raw is not None else x
        dx = (x - prev_x) / dt
        edx = self._dx.apply(dx, alpha=_alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x.apply(x, alpha=_alpha(cutoff, dt))
