"""Optional debug overlay — draws pipeline results onto a frame.

Kept entirely separate from the pipeline so the production service runs with no
GUI. Used only by `service.py --debug` when an installer is on-site verifying a
camera. All drawing logic that used to be tangled into main.py's loop lives here.
"""

from __future__ import annotations

import cv2

from .pipeline import FrameResult

_TIER_COLOR = {"HIGH": (0, 255, 80), "MED": (0, 215, 255), "LOW": (100, 100, 100)}


def draw(frame, result: FrameResult, *, store_name: str, tracker):
    """Return the frame annotated with per-person boxes and a session HUD."""
    for p in result.persons:
        x1, y1, x2, y2 = p.bbox
        color = _TIER_COLOR.get(p.tier, (100, 100, 100))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{p.track_id}"
        if p.yaw is not None:
            label += f" {p.tier} ({p.engage_prob:.0%})"
        if p.total_engage_s > 0:
            label += f" | {p.total_engage_s:.1f}s"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if p.yaw is not None:
            dy = y2 + 18 if y2 + 38 <= frame.shape[0] else y2 - 30
            dist = f"{p.dist_m:.1f}m" if p.dist_m is not None else "?"
            cv2.putText(
                frame,
                f"Yaw:{p.yaw:+.1f}deg Pitch:{p.pitch:+.1f}deg Dist:{dist} "
                f"Torso:{p.torso_conf:.0%} Zone:{p.zone_conf:.0%}",
                (x1, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1,
            )
        if p.nose_px is not None:
            cv2.circle(frame, p.nose_px, 4, (0, 255, 0), -1)

    _hud(frame, store_name, tracker)
    return frame


def _hud(frame, store_name, tracker):
    bg = frame.copy()
    cv2.rectangle(bg, (0, 0), (320, 120), (20, 20, 20), -1)
    cv2.addWeighted(bg, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, store_name, (10, 24), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 200, 255), 1)
    cv2.putText(frame, f"Passersby: {tracker.total_passersby}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, f"Engaged:   {tracker.total_engaged}", (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 130), 1)
    cv2.putText(frame, f"Attention: {tracker.total_attention_s():.0f}s", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
