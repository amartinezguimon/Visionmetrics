"""EngagementPipeline — orchestrates the 7 layers, returns data, draws nothing.

This is the heart of the agent, lifted out of main.py's monolithic while-loop.
Crucially it has NO OpenCV window, NO HUD drawing, and NO file I/O: it consumes
a frame and returns a structured result. Drawing (viewer.py), metric emission
(emitter.py), and the run loop (service.py) are separate concerns built on top.

Vision components are injected, so the whole orchestration is unit-testable with
fakes — no models, no camera, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .camera_model import focal_length_px
from .engagement import EngagementTracker, EngagementParams
from .tracking import ReconcileParams, TrackReconciler
from .zone import CountingRegion, EngagementZone, FarLine, GazeReference, zone_confidence

ENGAGE_THRESHOLD = 0.50


@dataclass
class PersonResult:
    track_id: int
    bbox: tuple[int, int, int, int]
    yaw: float | None = None
    pitch: float | None = None
    dist_m: float | None = None
    distance: float | None = None  # normalised cheekbone width — training.collect.tier_for() input
    engage_prob: float = 0.0
    zone_conf: float = 1.0
    torso_conf: float = 1.0
    rel_yaw: float | None = None
    nose_px: tuple[int, int] | None = None
    is_engaged: bool = False
    total_engage_s: float = 0.0
    tier: str = "LOW"
    newly_counted: bool = False
    # Feet fell on the FAR side of the operator's line: still detected and drawn
    # (so the operator can validate the line) and its raw frame is still recorded
    # for analytics, but it is NEVER counted or scored as a customer.
    is_far: bool = False


@dataclass
class FrameResult:
    persons: list[PersonResult] = field(default_factory=list)
    active_ids: set[int] = field(default_factory=set)
    # EVERY detector box this frame, before any zone/size/debounce/passerby
    # gating below — lets a reviewer audit the raw detector, including boxes
    # that never make it into `persons` (e.g. rejected as furniture/posters).
    raw_boxes: list[dict] = field(default_factory=list)


def _tier(prob: float) -> str:
    if prob >= 0.80:
        return "HIGH"
    if prob >= ENGAGE_THRESHOLD:
        return "MED"
    return "LOW"


class EngagementPipeline:
    def __init__(
        self, *, detector, head_pose, torso, classifier,
        zone: EngagementZone | None,
        engagement_params: EngagementParams,
        fov_h_deg: float,
        passerby_min_frames: int = 8,
        passerby_motion_px: int = 40,
        passerby_min_height_frac: float = 0.0,
        zone_soft_margin: float = 0.30,
        reconcile_params: ReconcileParams | None = None,
        gaze_reference: GazeReference | None = None,
        counting_region: CountingRegion | None = None,
        far_line: FarLine | None = None,
    ):
        self.detector = detector
        self.head_pose = head_pose
        self.torso = torso
        self.classifier = classifier
        self.zone = zone
        # Per-store re-centring of head angles onto the window direction. Defaults
        # to no shift (uncalibrated / camera on the display).
        self.gaze = gaze_reference or GazeReference()
        # Operator-drawn counting zone (feet must fall inside to be counted at all).
        # None => count everywhere (uncalibrated behaviour).
        self.region = counting_region
        # Operator-drawn "line of farness": tracks whose feet fall on the far side
        # are too distant to be customers — never counted, never scored. None => off.
        self.far_line = far_line
        self.fov_h_deg = fov_h_deg
        self.passerby_min_frames = max(1, passerby_min_frames)
        self.passerby_motion_px = passerby_motion_px
        self.min_height_frac = passerby_min_height_frac
        self.zone_soft_margin = zone_soft_margin
        self.tracker = EngagementTracker(engagement_params)
        # Heals ByteTrack id switches: a re-detected person keeps one stable id,
        # so they are not re-counted or their engagement reset.
        self.reconciler = TrackReconciler(reconcile_params)
        self._focal_px: float | None = None
        self._seen: dict[int, int] = {}            # canonical id -> frames seen
        self._first_center: dict[int, tuple[float, float]] = {}  # bbox centre when first seen
        self._last_center: dict[int, tuple[float, float]] = {}   # bbox centre most recently seen
        self._moved: set[int] = set()              # canonical ids that have moved enough
        self._face_seen: set[int] = set()          # canonical ids that have shown a face
        self._passerby: set[int] = set()           # canonical ids confirmed as real people
        self._seen_outside: set[int] = set()       # canonical ids seen OUTSIDE the zone (for entry counting)
        self._engaged_ever: set[int] = set()       # canonical ids that engaged at least once
        # Which way foot traffic flows past the window, resolved once per track
        # when it departs (net horizontal displacement of its trajectory). Purely
        # anonymous: a direction sign, never a path or identity. "left"/"right" is
        # the side the person CAME FROM in camera space; the operator can name the
        # sides in device.yaml. `engaged` counts those who also stopped to look.
        self.flow: dict[str, int] = {
            "from_left": 0, "from_right": 0,
            "from_left_engaged": 0, "from_right_engaged": 0,
            "ambiguous": 0,
        }

    def process_frame(self, frame, frame_idx: int, now: float) -> FrameResult:
        if self._focal_px is None:
            self._focal_px = focal_length_px(frame.shape[1], self.fov_h_deg)

        result = FrameResult()
        dets = self.detector.detect(frame)
        # Resolve raw ByteTrack ids to stable canonical ids before anything else,
        # so a re-detected person is recognised as the same individual.
        canon = self.reconciler.reconcile([(d.track_id, d.bbox) for d in dets], frame_idx)
        frame_h, frame_w = frame.shape[0], frame.shape[1]
        for det, tid in zip(dets, canon):
            x1, y1, x2, y2 = det.bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            result.raw_boxes.append({
                "id": tid, "box": [x1, y1, x2, y2], "conf": round(det.confidence, 3),
            })

            # Feet = bbox bottom-centre, normalised — the reference point both the
            # far-line and the counting zone test against.
            feet_x = cx / frame_w if frame_w else 0.0
            feet_y = y2 / frame_h if frame_h else 0.0

            # Far-line gate: feet past the operator's line of farness are too distant
            # to be a customer. We do NOT drop them — the operator needs to SEE the
            # far people (to validate the line is placed right) and we still record
            # the raw frame for analytics. Instead we mark them is_far, expose them
            # in the frame result (so overlay + review can segment shown-vs-ignored),
            # and skip all counting/scoring below via `continue`.
            if self.far_line is not None and self.far_line.is_far(feet_x, feet_y):
                result.active_ids.add(tid)
                result.persons.append(PersonResult(
                    track_id=tid, bbox=(x1, y1, x2, y2),
                    is_far=True, tier="FAR",
                ))
                continue

            # Inside the counting zone? (feet = bbox bottom-centre). With no zone
            # calibrated, everyone is "inside" (legacy behaviour).
            inside = True
            if self.region is not None:
                inside = self.region.contains(feet_x, feet_y)
                if not inside:
                    # Outside the zone: remember we saw them out here (so a later
                    # cross-in can be detected) but do NOT count or score. This also
                    # ignores people across the street / on a transverse pavement.
                    self._seen_outside.add(tid)
                    continue

            # Size gate: a box too short (relative to the frame) is someone too far
            # to be a customer — reject before counting.
            if self.min_height_frac > 0.0 and frame_h and (y2 - y1) < self.min_height_frac * frame_h:
                continue

            seen = self._seen[tid] = self._seen.get(tid, 0) + 1

            # Track movement since first seen (cheap, every frame): a real person
            # walks; furniture doesn't. Also remember the latest centre so the
            # departure hook can resolve the track's net direction of travel.
            fx, fy = self._first_center.setdefault(tid, (cx, cy))
            self._last_center[tid] = (cx, cy)
            if (cx - fx) ** 2 + (cy - fy) ** 2 >= self.passerby_motion_px ** 2:
                self._moved.add(tid)

            # Anti-flicker debounce: ignore boxes that haven't persisted yet.
            if seen < self.passerby_min_frames:
                continue

            pose = self.head_pose.analyze(frame, det.bbox, tid, frame_idx, self._focal_px, now)
            if pose is not None:
                self._face_seen.add(tid)

            if tid not in self._passerby:
                if self.region is not None:
                    # ZONE-ENTRY counting (anonymous "count crossings, not people"):
                    # count once when a track that was seen OUTSIDE the zone is now
                    # inside it — i.e. it actually crossed in. A track that only ever
                    # appears inside (a seated/standing person, or a re-acquired id
                    # popping up inside) has no 'outside' history and is NOT counted,
                    # which kills both the static over-count and the re-count churn.
                    if tid in self._seen_outside:
                        self._passerby.add(tid)
                        self.tracker.register(tid, now)
                    else:
                        continue
                elif tid in self._moved or tid in self._face_seen:
                    # No zone: legacy rule — a confirmed real person (moved or faced).
                    self._passerby.add(tid)
                    self.tracker.register(tid, now)
                else:
                    continue   # persisted but static & faceless -> likely not a person

            result.active_ids.add(tid)
            torso = self.torso.analyze(frame, det.bbox, tid, frame_idx)
            person = self._score(tid, det.bbox, pose, torso, now)
            if person.is_engaged or person.total_engage_s > 0.0:
                self._engaged_ever.add(tid)
            result.persons.append(person)

        # Forget tracks gone past the grace window: free their per-track state
        # everywhere. drop() banks a real person's attention into the session
        # total first, so counts/attention are preserved while memory is freed.
        for cid in self.reconciler.expire(frame_idx):
            self._tally_flow(cid)
            self.tracker.drop(cid)
            self.head_pose.forget(cid)
            self.torso.forget(cid)
            self._seen.pop(cid, None)
            self._first_center.pop(cid, None)
            self._last_center.pop(cid, None)
            self._moved.discard(cid)
            self._face_seen.discard(cid)
            self._passerby.discard(cid)
            self._seen_outside.discard(cid)
            self._engaged_ever.discard(cid)

        return result

    def _tally_flow(self, cid: int) -> None:
        """When a confirmed passer-by departs, bank which side they came from
        into `self.flow`, from the net horizontal shift of their trajectory.
        Only real people are counted (same gate as `total_passersby`); a track
        that barely moved horizontally is 'ambiguous' rather than mis-attributed."""
        if cid not in self._passerby:
            return
        first = self._first_center.get(cid)
        last = self._last_center.get(cid)
        if first is None or last is None:
            return
        dx = last[0] - first[0]
        if abs(dx) < self.passerby_motion_px:
            self.flow["ambiguous"] += 1
            return
        engaged = cid in self._engaged_ever
        # Moving right (dx>0) means they entered from the left of frame, and vice
        # versa — so the side they CAME FROM is the sign's opposite in space.
        side = "from_left" if dx > 0 else "from_right"
        self.flow[side] += 1
        if engaged:
            self.flow[f"{side}_engaged"] += 1

    # ── scoring ──────────────────────────────────────────────────
    def _score(self, tid, bbox, pose, torso, now) -> PersonResult:
        person = PersonResult(track_id=tid, bbox=bbox)
        person.torso_conf = torso.confidence
        person.rel_yaw = torso.rel_yaw

        raw_engaged = False
        engage_prob = 0.0
        zone_conf = 1.0
        if pose is not None:
            # Re-centre the head angles onto the calibrated window direction before
            # the classifier, which learned "straight ahead = looking". The zone
            # check below stays on RAW angles (its bounds are in camera space).
            yaw_c, pitch_c = self.gaze.recenter(pose.yaw, pose.pitch)
            prob = self.classifier.probability(yaw_c, pitch_c, pose.distance)
            torso_weight = 0.40 + 0.60 * torso.confidence
            prob *= torso_weight
            zone_conf = zone_confidence(
                pose.yaw, pose.pitch, pose.distance, self.zone,
                pose.dist_m, soft_margin=self.zone_soft_margin,
            )
            engage_prob = prob * zone_conf
            raw_engaged = engage_prob >= ENGAGE_THRESHOLD
            person.yaw, person.pitch, person.dist_m = pose.yaw, pose.pitch, pose.dist_m
            person.distance = pose.distance
            person.nose_px = pose.nose_px

        update = self.tracker.update(tid, raw_engaged, now, prob=engage_prob)

        person.engage_prob = engage_prob
        person.zone_conf = zone_conf
        person.is_engaged = update.is_engaged
        person.total_engage_s = update.total_engage_s
        person.newly_counted = update.newly_counted
        person.tier = _tier(engage_prob)
        return person
