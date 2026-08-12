"""Tests for EngagementPipeline orchestration using fake vision components.

No models, no camera, no GPU — we inject fakes and assert the pipeline wires
the layers together correctly (scoring, counting, ghost rejection, re-association).
"""

import numpy as np

from visionmetrics.edge.agent.engagement import EngagementParams
from visionmetrics.edge.agent.pipeline import EngagementPipeline
from visionmetrics.edge.agent.zone import CountingRegion, FarLine, GazeReference
from visionmetrics.edge.agent.vision.detector import Detection
from visionmetrics.edge.agent.vision.face import HeadPose
from visionmetrics.edge.agent.vision.pose import TorsoResult, NEUTRAL


class FakeDetector:
    def __init__(self, detections):
        self._dets = detections

    def detect(self, frame):
        return list(self._dets)


class FakeDetectorSequence:
    """Returns a different detection list per call (one per processed frame)."""
    def __init__(self, per_frame):
        self._per_frame = per_frame
        self._i = 0

    def detect(self, frame):
        dets = self._per_frame[min(self._i, len(self._per_frame) - 1)]
        self._i += 1
        return list(dets)


class FakeHeadPose:
    """Returns a fixed pose, or None to simulate 'no face found'."""
    def __init__(self, pose):
        self._pose = pose
        self.forgotten = []

    def analyze(self, frame, bbox, tid, frame_idx, focal_px, now):
        return self._pose

    def forget(self, tid):
        self.forgotten.append(tid)


class FakeHeadPoseSequence:
    """Returns None for the first `none_for` analyze calls, then a fixed pose.

    Simulates a person who approaches with their back turned (no face) and only
    later turns to look at the display.
    """
    def __init__(self, pose, none_for):
        self._pose = pose
        self._none_for = none_for
        self._calls = 0

    def analyze(self, frame, bbox, tid, frame_idx, focal_px, now):
        self._calls += 1
        return None if self._calls <= self._none_for else self._pose

    def forget(self, tid):
        pass


class FakeTorso:
    def __init__(self, result=NEUTRAL):
        self._result = result

    def analyze(self, frame, bbox, tid, frame_idx):
        return self._result

    def forget(self, tid):
        pass


class FakeClassifier:
    def __init__(self, prob):
        self._prob = prob

    def probability(self, yaw, pitch, distance):
        return self._prob


class YawSensitiveClassifier:
    """Engaged only when the (re-centred) yaw is near 0 — to test gaze recentering."""
    def probability(self, yaw, pitch, distance):
        return 1.0 if abs(yaw) < 0.1 else 0.0


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
STRAIGHT = HeadPose(yaw=0.0, pitch=0.0, distance=0.19, dist_m=1.0, nose_px=(10, 10))


def make_pipeline(detector, head_pose, classifier_prob, *,
                  passerby_min_frames=1, passerby_motion_px=40,
                  passerby_min_height_frac=0.0,
                  classifier=None, gaze_reference=None, counting_region=None,
                  far_line=None, **params):
    return EngagementPipeline(
        detector=detector,
        head_pose=head_pose,
        torso=FakeTorso(),
        classifier=classifier or FakeClassifier(classifier_prob),
        zone=None,
        engagement_params=EngagementParams(frame_buffer_size=1, frame_engage_min=1, **params),
        fov_h_deg=70.0,
        passerby_min_frames=passerby_min_frames,
        passerby_motion_px=passerby_motion_px,
        passerby_min_height_frac=passerby_min_height_frac,
        gaze_reference=gaze_reference,
        counting_region=counting_region,
        far_line=far_line,
    )


def test_engaged_person_is_scored_and_counted():
    det = Detection(track_id=5, bbox=(100, 50, 200, 400), confidence=0.9)
    pipe = make_pipeline(FakeDetector([det]), FakeHeadPose(STRAIGHT), 1.0,
                         count_threshold_s=2.0)
    pipe.process_frame(FRAME, frame_idx=0, now=0.0)
    pipe.process_frame(FRAME, frame_idx=1, now=1.0)
    r = pipe.process_frame(FRAME, frame_idx=2, now=2.0)

    assert len(r.persons) == 1
    p = r.persons[0]
    assert p.track_id == 5
    assert p.is_engaged
    assert p.tier == "HIGH"            # prob 1.0 * torso 1.0 * zone 1.0
    assert p.total_engage_s == 2.0
    assert pipe.tracker.total_engaged == 1


def test_away_person_not_counted():
    det = Detection(track_id=7, bbox=(0, 0, 50, 200), confidence=0.9)
    pipe = make_pipeline(FakeDetector([det]), FakeHeadPose(STRAIGHT), 0.10,
                         count_threshold_s=2.0)
    for i in range(5):
        r = pipe.process_frame(FRAME, frame_idx=i, now=float(i))
    assert not r.persons[0].is_engaged
    assert pipe.tracker.total_engaged == 0


def test_static_faceless_box_never_counted():
    # A static box that never yields a face (a chair / mannequin) is never a person.
    det = Detection(track_id=9, bbox=(0, 0, 50, 200), confidence=0.9)
    pipe = make_pipeline(FakeDetector([det]), FakeHeadPose(None), 1.0,
                         passerby_min_frames=3, passerby_motion_px=30)
    for i in range(30):
        r = pipe.process_frame(FRAME, frame_idx=i, now=float(i))
    assert pipe.tracker.total_passersby == 0
    assert r.persons == []


def test_moving_person_without_face_counts_as_passerby():
    # Foot traffic: someone walks across frame and never shows their face. They
    # are counted as a passerby (traffic) but never as engaged (needs a face).
    per_frame = [[Detection(track_id=1, bbox=(100 + i * 20, 50, 180 + i * 20, 400),
                            confidence=0.9)] for i in range(12)]
    pipe = make_pipeline(FakeDetectorSequence(per_frame), FakeHeadPose(None), 1.0,
                         passerby_min_frames=3, passerby_motion_px=30)
    for i in range(12):
        pipe.process_frame(FRAME, frame_idx=i, now=float(i))
    assert pipe.tracker.total_passersby == 1
    assert pipe.tracker.total_engaged == 0


def test_track_confirmed_when_face_appears_later():
    # The real-store scenario: person approaches with back turned (no face) for a
    # while, THEN looks at the display. Must still be detected, counted, scored.
    det = Detection(track_id=4, bbox=(100, 50, 200, 400), confidence=0.9)
    pipe = make_pipeline(FakeDetector([det]),
                         FakeHeadPoseSequence(STRAIGHT, none_for=15), 1.0,
                         count_threshold_s=2.0)
    for i in range(40):
        r = pipe.process_frame(FRAME, frame_idx=i, now=float(i))
    assert pipe.tracker.total_passersby == 1   # confirmed once the face appeared
    assert pipe.tracker.total_engaged == 1     # engagement scored after confirmation
    assert r.persons and r.persons[0].is_engaged


def test_reassociation_prevents_double_count_after_id_switch():
    # Hector's bug: a person is tracked + counted, the detection is lost for a
    # few frames, then ByteTrack resurrects them under a NEW id. They must NOT be
    # re-counted, and their engagement must continue (not reset).
    box = (100, 50, 200, 400)
    a = Detection(track_id=1, bbox=box, confidence=0.9)
    b = Detection(track_id=2, bbox=box, confidence=0.9)   # new id, same place
    per_frame = [
        [a], [a], [a],        # frames 0-2: id=1, looking -> crosses 2s, counted
        [], [], [],           # frames 3-5: lost (occlusion / conf dip)
        [b], [b], [b],        # frames 6-8: reappears as id=2 at the same spot
    ]
    pipe = make_pipeline(FakeDetectorSequence(per_frame), FakeHeadPose(STRAIGHT), 1.0,
                         count_threshold_s=2.0)
    for i in range(len(per_frame)):
        r = pipe.process_frame(FRAME, frame_idx=i, now=float(i))

    assert pipe.tracker.total_passersby == 1   # one person, not two
    assert pipe.tracker.total_engaged == 1     # counted once, not re-counted
    assert r.persons and r.persons[0].track_id == 1   # id=2 healed back to 1
    assert r.persons[0].is_engaged             # engagement continued through the gap


def test_gaze_recentering_lets_offaxis_camera_detect_looking():
    # Corner-mounted camera: looking AT the window reads as a turned head (raw yaw
    # 0.4). The classifier only fires near 0. Without re-centring it misses it;
    # with the window calibrated at yaw_center=0.4 it maps to ~0 and counts.
    det = Detection(track_id=1, bbox=(100, 50, 200, 400), confidence=0.9)
    looking_offaxis = HeadPose(yaw=0.4, pitch=0.0, distance=0.19, dist_m=1.0, nose_px=(10, 10))

    no_cal = make_pipeline(FakeDetector([det]), FakeHeadPose(looking_offaxis), 0,
                           classifier=YawSensitiveClassifier(), count_threshold_s=2.0)
    for i in range(4):
        no_cal.process_frame(FRAME, frame_idx=i, now=float(i))
    assert no_cal.tracker.total_engaged == 0          # missed without calibration

    calibrated = make_pipeline(FakeDetector([det]), FakeHeadPose(looking_offaxis), 0,
                               classifier=YawSensitiveClassifier(),
                               gaze_reference=GazeReference(yaw_center=0.4),
                               count_threshold_s=2.0)
    for i in range(4):
        r = calibrated.process_frame(FRAME, frame_idx=i, now=float(i))
    assert calibrated.tracker.total_engaged == 1      # re-centred -> detected
    assert r.persons[0].is_engaged


def test_departed_person_is_dropped_but_attention_kept():
    # After the grace window a gone person is freed from memory, but their
    # banked attention stays in the session total (monotonic).
    box = (100, 50, 200, 400)
    a = Detection(track_id=1, bbox=box, confidence=0.9)
    present = [[a]] * 4                  # frames 0-3: looking (banks ~3s attention)
    gone = [[]] * 60                     # long enough to exceed the 45-frame grace
    pipe = make_pipeline(FakeDetectorSequence(present + gone),
                         FakeHeadPose(STRAIGHT), 1.0, count_threshold_s=2.0)
    for i in range(4 + 60):
        pipe.process_frame(FRAME, frame_idx=i, now=float(i))

    assert pipe.tracker.people == {}                 # state freed
    assert pipe.tracker.total_passersby == 1         # still counted
    assert pipe.tracker.total_engaged == 1
    assert pipe.tracker.total_attention_s() >= 3.0   # attention banked, not lost


# FRAME is 480(h) x 640(w); feet = bbox bottom-centre, normalised by frame size.
def test_person_outside_counting_region_is_not_counted():
    # Region = bottom half only. A detection high in the frame (far away, feet at
    # ~y=0.25) is ignored entirely — not a passerby, not scored.
    region = CountingRegion.from_config(
        {"polygon": [[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]]})
    far = Detection(track_id=1, bbox=(560, 40, 620, 120), confidence=0.9)  # feet ~(0.92, 0.25)
    pipe = make_pipeline(FakeDetector([far]), FakeHeadPose(STRAIGHT), 1.0,
                         passerby_min_frames=1, counting_region=region,
                         count_threshold_s=1.0)
    for i in range(5):
        r = pipe.process_frame(FRAME, frame_idx=i, now=float(i))
    assert pipe.tracker.total_passersby == 0
    assert pipe.tracker.total_engaged == 0
    assert r.persons == []


def test_person_too_small_is_not_counted():
    # bbox height 40px / 480 = 8% < 20% threshold -> too far, ignored entirely.
    small = Detection(track_id=1, bbox=(300, 200, 360, 240), confidence=0.9)
    pipe = make_pipeline(FakeDetector([small]), FakeHeadPose(STRAIGHT), 1.0,
                         passerby_min_frames=1, passerby_min_height_frac=0.20,
                         count_threshold_s=1.0)
    for i in range(4):
        r = pipe.process_frame(FRAME, frame_idx=i, now=float(i))
    assert pipe.tracker.total_passersby == 0
    assert r.persons == []


def test_person_tall_enough_passes_size_gate():
    tall = Detection(track_id=1, bbox=(300, 80, 360, 400), confidence=0.9)  # h=320 = 67%
    pipe = make_pipeline(FakeDetector([tall]), FakeHeadPose(STRAIGHT), 1.0,
                         passerby_min_frames=1, passerby_min_height_frac=0.20,
                         count_threshold_s=1.0)
    for i in range(3):
        pipe.process_frame(FRAME, frame_idx=i, now=float(i))
    assert pipe.tracker.total_passersby == 1


# Horizontal line of farness across the middle: feet above y=0.5 (deeper in the
# scene) are "too far"; feet below it (near the camera at the bottom) are customers.
MID_FAR_LINE = FarLine.from_config({"line": [[0.0, 0.5], [1.0, 0.5]]})


def test_person_past_far_line_is_visible_but_not_counted():
    # Feet at y2=120 -> 0.25 normalised, ABOVE the far line -> too far to be a
    # customer. It is NOT counted or scored, but it IS still detected and exposed
    # (is_far=True) so the operator can validate the line and analytics keep it.
    far = Detection(track_id=1, bbox=(300, 40, 360, 120), confidence=0.9)
    pipe = make_pipeline(FakeDetector([far]), FakeHeadPose(STRAIGHT), 1.0,
                         passerby_min_frames=1, far_line=MID_FAR_LINE,
                         count_threshold_s=1.0)
    for i in range(5):
        r = pipe.process_frame(FRAME, frame_idx=i, now=float(i))
    assert pipe.tracker.total_passersby == 0
    assert pipe.tracker.total_engaged == 0
    # Still visible + segmentable: one person, marked far, never engaged.
    assert len(r.persons) == 1
    assert r.persons[0].is_far is True
    assert r.persons[0].tier == "FAR"
    assert r.persons[0].is_engaged is False
    assert 1 in r.active_ids


def test_person_near_side_of_far_line_is_counted():
    # Same tool active, but feet at y2=400 -> 0.83 normalised, on the NEAR side.
    near = Detection(track_id=1, bbox=(300, 80, 360, 400), confidence=0.9)
    pipe = make_pipeline(FakeDetector([near]), FakeHeadPose(STRAIGHT), 1.0,
                         passerby_min_frames=1, far_line=MID_FAR_LINE,
                         count_threshold_s=1.0)
    for i in range(3):
        r = pipe.process_frame(FRAME, frame_idx=i, now=float(i))
    assert pipe.tracker.total_passersby == 1
    assert len(r.persons) == 1


BOTTOM_HALF = CountingRegion.from_config(
    {"polygon": [[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]]})  # inside = feet below y=240


def test_zone_entry_counts_a_cross_in_once():
    # Walks DOWN into the zone: feet y2 goes 150,200,230 (outside) -> 260,300,340 (inside).
    per_frame = [[Detection(track_id=1, bbox=(100, y2 - 160, 200, y2), confidence=0.9)]
                 for y2 in (150, 200, 230, 260, 300, 340)]
    pipe = make_pipeline(FakeDetectorSequence(per_frame), FakeHeadPose(None), 1.0,
                         passerby_min_frames=1, counting_region=BOTTOM_HALF, count_threshold_s=1.0)
    for i in range(len(per_frame)):
        pipe.process_frame(FRAME, frame_idx=i, now=float(i))
    assert pipe.tracker.total_passersby == 1     # one cross-in = one count


def test_static_person_inside_zone_is_not_counted():
    # A seated/standing person always inside the zone, never seen outside -> never
    # counted (kills the bar "passerby rises with nobody entering" problem).
    inside = Detection(track_id=1, bbox=(100, 140, 200, 300), confidence=0.9)  # feet y=300 inside
    pipe = make_pipeline(FakeDetector([inside]), FakeHeadPose(STRAIGHT), 1.0,
                         passerby_min_frames=1, counting_region=BOTTOM_HALF, count_threshold_s=1.0)
    for i in range(10):
        pipe.process_frame(FRAME, frame_idx=i, now=float(i))
    assert pipe.tracker.total_passersby == 0


def test_zone_entry_not_recounted_on_id_switch():
    # Person crosses in (id 1), is lost, reappears inside as id 2 at the same spot.
    # Must NOT be re-counted: either the reconciler heals 2->1, or id 2 has no
    # 'outside' history -> not counted. Either way the count stays 1.
    box_in = (100, 140, 200, 300)
    out = Detection(track_id=1, bbox=(100, 40, 200, 200), confidence=0.9)   # feet y=200 outside
    a = Detection(track_id=1, bbox=box_in, confidence=0.9)                  # id1 inside
    c = Detection(track_id=2, bbox=box_in, confidence=0.9)                  # id2 same spot (churn)
    per_frame = [[out], [a], [a], [], [], [c], [c]]
    pipe = make_pipeline(FakeDetectorSequence(per_frame), FakeHeadPose(STRAIGHT), 1.0,
                         passerby_min_frames=1, counting_region=BOTTOM_HALF, count_threshold_s=1.0)
    for i in range(len(per_frame)):
        pipe.process_frame(FRAME, frame_idx=i, now=float(i))
    assert pipe.tracker.total_passersby == 1
