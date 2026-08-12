"""Soft engagement-zone confidence.

Returns a multiplier in [0, 1] instead of a hard YES/NO gate, so a person
standing right at the calibrated boundary doesn't flicker between engaged and
away. Distance is still a hard cutoff (someone 6 m away is never a customer).

Extracted verbatim (behavior-preserving) from main.py `zone_confidence`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GazeReference:
    """Head-pose direction recorded when looking at the WINDOW CENTRE at calibration.

    The classifier was trained with subjects facing the camera (yaw≈0, pitch≈0 =
    looking). In a real store the camera sits off to one side / in a corner, so
    looking at the window is NOT looking at the camera — the head is turned by a
    fixed offset. Subtracting that offset (``recenter``) before the classifier maps
    "looking at the window" back onto the model's "straight ahead", so the same
    trained model works from any camera position.

    Defaults to (0, 0) = no shift (camera roughly on the display, or uncalibrated),
    which preserves the prior behaviour exactly.
    """
    yaw_center: float = 0.0
    pitch_center: float = 0.0

    @classmethod
    def from_config(cls, engagement_zone: dict | None) -> "GazeReference":
        """Build from the ``engagement_zone`` block of a calibration config."""
        if not engagement_zone:
            return cls()
        return cls(
            yaw_center=engagement_zone.get("yaw_center", 0.0),
            pitch_center=engagement_zone.get("pitch_center", 0.0),
        )

    def recenter(self, yaw: float, pitch: float) -> tuple[float, float]:
        """Shift live angles so the window direction becomes (0, 0) for the model."""
        return yaw - self.yaw_center, pitch - self.pitch_center


@dataclass(frozen=True)
class CountingRegion:
    """A calibrated polygon (normalised [0..1] image coords) — the operator-drawn
    "counting zone" (the pavement in front of the window).

    When a zone is set, footfall is counted by ENTRY: a person is counted once when
    they were seen OUTSIDE the zone and then cross INSIDE it (feet = bbox
    bottom-centre). This is the anonymous, retail-standard "count crossings, not
    identities" rule — no face/appearance recognition, GDPR/AI-Act friendly. It
    also fixes the field problems: people too far (across the street) never enter;
    a seated/standing person already inside, or a re-acquired track id popping up
    inside, has no 'outside' history and is NOT (re-)counted.

    Draw the polygon with a margin from the frame edges so people are detected
    outside it before they cross in. Image-space, so re-draw if the camera moves.
    ``None``/empty => count everywhere (legacy confirmed-track rule).
    """
    polygon: tuple[tuple[float, float], ...] = ()

    @classmethod
    def from_config(cls, region: dict | None) -> "CountingRegion | None":
        """Build from the ``counting_region`` block of a calibration config."""
        if not region:
            return None
        poly = region.get("polygon") or []
        if len(poly) < 3:                       # a polygon needs >= 3 vertices
            return None
        return cls(polygon=tuple((float(x), float(y)) for x, y in poly))

    def contains(self, x: float, y: float) -> bool:
        """Point-in-polygon (ray casting). x, y are normalised [0..1] image coords."""
        poly = self.polygon
        n = len(poly)
        if n < 3:
            return True                         # degenerate => don't filter
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            ):
                inside = not inside
            j = i
        return inside


@dataclass(frozen=True)
class FarLine:
    """A calibrated "line of farness" (normalised [0..1] image coords, two endpoints).

    A single straight — usually slanted — line drawn across the frame. Everything
    on the FAR side of it (deeper into the scene / higher up the image, e.g. the
    opposite pavement or the far end of the street) is treated as "too far to be a
    customer": those tracks are neither counted nor scored for engagement.

    Simpler and more robust than a full polygon when the only thing the operator
    needs is a depth cutoff: two clicks and done. The "far" side is inferred
    automatically as the side AWAY from the bottom of the frame (where the camera
    stands and the nearest customers walk), so the operator never has to say which
    half is which.

    ``feet`` = bbox bottom-centre, same reference the counting zone uses.
    ``None``/degenerate => no far cutoff (legacy behaviour).
    """
    a: tuple[float, float] = (0.0, 0.5)
    b: tuple[float, float] = (1.0, 0.5)

    @classmethod
    def from_config(cls, block: dict | None) -> "FarLine | None":
        """Build from the ``far_line`` block of a calibration config."""
        if not block:
            return None
        line = block.get("line") or []
        if len(line) != 2:                       # a line needs exactly 2 endpoints
            return None
        a = (float(line[0][0]), float(line[0][1]))
        b = (float(line[1][0]), float(line[1][1]))
        if a == b:                               # degenerate => no cutoff
            return None
        return cls(a=a, b=b)

    def _side(self, x: float, y: float) -> float:
        """Signed z-component of the cross product (b-a)×(p-a): >0 one side, <0 the
        other, 0 exactly on the line. Sign alone tells you which half a point is in."""
        ax, ay = self.a
        bx, by = self.b
        return (bx - ax) * (y - ay) - (by - ay) * (x - ax)

    def is_far(self, feet_x: float, feet_y: float) -> bool:
        """True if the feet fall on the FAR side of the line (away from frame bottom).

        The reference "near" point is the bottom-centre of the frame (0.5, 1.0),
        where the camera stands. A point is 'far' when it sits on the opposite side
        of the line from that reference."""
        ref = self._side(0.5, 1.0)
        if abs(ref) < 1e-9:                       # line passes through the reference
            ref = self._side(0.5, 2.0)            # push further "near" and retry
        return self._side(feet_x, feet_y) * ref < 0.0


@dataclass(frozen=True)
class EngagementZone:
    """Calibrated boundaries for one display, produced by calibration."""
    yaw_min: float
    yaw_max: float
    pitch_min: float
    pitch_max: float
    dist_min: float = 0.0           # normalised face-width proxy (legacy fallback)
    dist_max_m: float | None = None  # real-world far limit in metres (preferred)

    @classmethod
    def from_config(cls, derived: dict | None) -> "EngagementZone | None":
        """Build from the ``derived`` block of a calibration config, or None."""
        if not derived:
            return None
        return cls(
            yaw_min=derived["yaw_min"],
            yaw_max=derived["yaw_max"],
            pitch_min=derived["pitch_min"],
            pitch_max=derived["pitch_max"],
            dist_min=derived.get("dist_min", 0.0),
            dist_max_m=derived.get("dist_max_m"),
        )


def zone_confidence(
    yaw: float,
    pitch: float,
    distance: float,
    zone: EngagementZone | None,
    dist_m: float | None = None,
    *,
    soft_margin: float = 0.30,
    dist_buffer: float = 1.2,
) -> float:
    """Smooth [0, 1] confidence that the gaze falls inside the engagement zone.

    1.0 well inside; decays linearly to 0 across ``soft_margin`` normalised
    units beyond the yaw/pitch boundary. Distance is a hard cutoff with a
    ``dist_buffer`` margin beyond the calibrated far limit.

    With no calibration (``zone is None``) everything passes (returns 1.0), so
    the classifier alone decides — matching the prototype's behavior.
    """
    if zone is None:
        return 1.0

    # Hard distance cutoff.
    if dist_m is not None and zone.dist_max_m is not None:
        if dist_m > zone.dist_max_m * dist_buffer:
            return 0.0
    elif distance < zone.dist_min * 0.8:
        return 0.0

    # Soft angle penalty: how far outside each boundary are we?
    yaw_excess = max(0.0, zone.yaw_min - yaw, yaw - zone.yaw_max)
    pitch_excess = max(0.0, zone.pitch_min - pitch, pitch - zone.pitch_max)

    return max(0.0, 1.0 - (yaw_excess + pitch_excess) / soft_margin)
