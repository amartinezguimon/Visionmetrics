"""Pre-compute detections + head-pose for a video so it can be labeled offline.

Runs the SAME production detector + head-pose analyzer as the live agent (so
features match inference exactly, same principle as `collect.py`) over a WHOLE
frame sampled every few seconds, and writes a compact JSON the labeling page
loads with no video file needed.

One full frame is embedded as a JPEG every `sample_seconds` (default 5s),
whether or not anyone was detected in it — an empty-looking frame is exactly
where the labeler might draw a person the detector missed. Each detected person
carries its native-pixel box + head pose, so the page can draw the boxes over
the frame, let you approve/dismiss each one, mark looking / not looking, and
add boxes the detector missed. No dragging video files around, no seeking, no
"wrong camera" surprises.

(`FrameSampler` and `crop_to_jpeg_b64` below are the older per-person-crop
path, still used by the live dashboard in `webserver.py`.)

Run:
    python -m visionmetrics.training.prep clip.mp4
    python -m visionmetrics.training.prep clip.mp4 --out clip.json --sample-seconds 5
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def crop_to_jpeg_b64(frame, box, *, pad: float = 0.3, max_dim: int = 320, quality: int = 82) -> str:
    """Crop a padded region around `box` out of `frame` and return it as a
    base64 JPEG string, ready to embed directly in the labeling page."""
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    sx1 = max(0, int(x1 - bw * pad))
    sy1 = max(0, int(y1 - bh * pad))
    sx2 = min(w, int(x2 + bw * pad))
    sy2 = min(h, int(y2 + bh * pad))
    crop = frame[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        crop = frame
    ch, cw = crop.shape[:2]
    if max(ch, cw) > max_dim:
        scale = max_dim / max(ch, cw)
        crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def full_frame_to_jpeg_b64(frame, *, max_dim: int = 1280, quality: int = 80) -> str:
    """Encode a WHOLE frame (optionally downscaled) as a base64 JPEG so the
    labeling page can show the full scene with detection boxes drawn on top —
    no video file needed. Person boxes stay in the video's native pixel space;
    the page rescales them to this (possibly smaller) embedded image."""
    import cv2

    h, w = frame.shape[:2]
    img = frame
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


class FrameSampler:
    """Decides which frames make it into the labeling JSON.

    Old behaviour sampled 1-in-N frames on a fixed timer, regardless of
    whether anyone was even in frame — a labeler could sit there clicking
    through empty storefront shots. New rule, two triggers, either one keeps
    the frame, and frames with **nobody** detected are never kept at all:

    * a **new** track just appeared — its very first frame is always kept,
      even if it doesn't land on the throttle boundary, so brief/passing
      visits are never silently dropped;
    * the throttle stride (``sample_every``) has elapsed since the last kept
      frame — keeps a steady trickle of frames while someone lingers, instead
      of keeping literally every frame of a stationary person.
    """

    def __init__(self, sample_every: int = 5):
        self.sample_every = max(1, sample_every)
        self._seen_ids: set[int] = set()
        self._last_kept_idx: int | None = None

    def should_keep(self, frame_idx: int, people: list[dict]) -> bool:
        if not people:
            return False
        ids = {p["id"] for p in people}
        is_new = bool(ids - self._seen_ids)
        due = self._last_kept_idx is None or (frame_idx - self._last_kept_idx) >= self.sample_every
        keep = is_new or due
        self._seen_ids |= ids
        if keep:
            self._last_kept_idx = frame_idx
        return keep


def run(video_path: str, out_path: str | None = None, *, sample_seconds: float = 5.0,
        fov_h_deg: float = 70.0, conf: float = 0.25, aspect: float = 0.30,
        config_path: str | None = None, open_browser: bool = True) -> int:
    # Heavy deps imported lazily so --help stays fast and tests can avoid them.
    import cv2

    from ..edge.agent.camera_model import focal_length_px
    from ..edge.agent.classifier import EngagementClassifier
    from ..edge.agent.config import DeviceConfig
    from ..edge.agent.models_bootstrap import ensure_models
    from ..edge.agent.tracking import ReconcileParams, TrackReconciler
    from ..edge.agent.vision.detector import PersonDetector
    from ..edge.agent.vision.face import HeadPoseAnalyzer
    from .collect import tier_for

    config = DeviceConfig.load(config_path) if config_path else DeviceConfig()
    ensure_models(config)
    v = config.vision

    detector = PersonDetector(config.models.yolo, conf_min=conf, aspect_ratio_min=aspect)
    analyzer = HeadPoseAnalyzer(
        config.models.face, face_width_m=v.face_width_m, head_crop_frac=v.head_crop_frac,
        head_upscale=v.head_upscale, skip_frames=1,
    )
    reconciler = TrackReconciler(ReconcileParams(
        grace_frames=v.reassoc_grace_frames, min_iou=v.reassoc_min_iou,
    ))

    # Embed the engagement model's own P(looking) for each person, so the review
    # page can score the model's initial predictions against the human labels
    # (ROC / confusion / etc.). Optional — a missing model just omits `p_look`.
    classifier = None
    if Path(config.models.engagement).exists():
        try:
            classifier = EngagementClassifier.load(config.models.engagement)
        except Exception as e:  # noqa: BLE001 — never let a bad model block labeling
            print(f"[prep] engagement model unusable ({e}); frames will carry no p_look")
    else:
        print("[prep] no engagement model found; frames will carry no model prediction to compare")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open video {video_path!r}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal_px = focal_length_px(width, fov_h_deg)

    if not out_path:
        out_path = str(Path(video_path).with_suffix(".json"))

    # Time-based sampling: keep one WHOLE frame every `sample_seconds`, whether
    # or not anyone was detected in it — an empty-looking frame is exactly where
    # the labeler might draw a person the detector missed. Boxes stay in native
    # pixel coords (faithful export); the embedded JPEG may be downscaled and the
    # page rescales the boxes to match it.
    stride = max(1, int(round(sample_seconds * fps)))
    frames_out: list[dict] = []
    frame_idx = 0
    frames_with_people = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        raw_dets = detector.detect(frame)
        canon_ids = reconciler.reconcile(
            [(d.track_id, d.bbox) for d in raw_dets], frame_idx)
        reconciler.expire(frame_idx)

        people = []
        for det, cid in zip(raw_dets, canon_ids):
            pose = analyzer.analyze(frame, det.bbox, cid, frame_idx, focal_px)
            entry = {"id": int(cid), "box": list(det.bbox), "conf": round(det.confidence, 3)}
            if pose is not None:
                entry.update(yaw=round(pose.yaw, 4), pitch=round(pose.pitch, 4),
                            distance=round(pose.distance, 4), dist_m=round(pose.dist_m, 3),
                            tier=tier_for(pose.distance))
                if classifier is not None:
                    entry["p_look"] = round(
                        classifier.probability(pose.yaw, pose.pitch, pose.distance), 4)
            people.append(entry)

        if people:
            frames_with_people += 1
        frames_out.append({
            "i": frame_idx, "t": round(frame_idx / fps, 3),
            "frame": full_frame_to_jpeg_b64(frame), "people": people,
        })

        if frame_idx % 300 == 0:
            print(f"[prep] frame {frame_idx}...")
        frame_idx += 1

    cap.release()

    out = {
        "video": {
            "filename": Path(video_path).name, "fps": fps, "width": width,
            "height": height, "sample_seconds": sample_seconds, "total_frames": frame_idx,
        },
        "frames": frames_out,
    }
    Path(out_path).write_text(json.dumps(out), encoding="utf-8")
    print(f"[prep] {frame_idx} frames scanned, kept {len(frames_out)} whole frames "
          f"every {sample_seconds:g}s ({frames_with_people} had a person) -> {out_path}")

    html_path = write_review_page(video_path, out)
    if html_path and open_browser:
        import webbrowser
        webbrowser.open(Path(html_path).resolve().as_uri())
    if html_path:
        print(f"[prep] ready to label -> double-click {html_path} "
              f"(photos are baked in, the video isn't needed anymore)")
    else:
        print(f"[prep] now drag '{out_path}' into "
              f"visionmetrics/training/web/review.html")
    return 0


def write_review_page(video_path: str, embedded: dict) -> str | None:
    """Bake the per-person cropped photos + pose data into a copy of
    review.html so a teammate only has to double-click ONE self-contained
    file — no video required, just the extracted frames, one at a time."""
    template_path = Path(__file__).parent / "web" / "review.html"
    if not template_path.exists():
        return None
    template = template_path.read_text(encoding="utf-8")
    embed_script = f"<script>window.__VM_EMBEDDED__ = {json.dumps(embedded)};</script>"
    merged = template.replace("<!--VM_EMBED_PLACEHOLDER-->", embed_script)
    html_path = Path(video_path).with_name(Path(video_path).stem + "_review.html")
    html_path.write_text(merged, encoding="utf-8")
    return str(html_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-compute detections + pose for offline labeling.")
    ap.add_argument("video", help="path to the recorded clip")
    ap.add_argument("--out", default=None, help="output JSON path (default: same name as video)")
    ap.add_argument("--sample-seconds", type=float, default=5.0,
                    help="store one whole frame every N seconds (smaller = finer "
                         "review, bigger file); frames with nobody in them are kept "
                         "too, so you can draw people the detector missed")
    ap.add_argument("--fov", type=float, default=70.0, help="camera horizontal FOV (deg)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO person confidence floor")
    ap.add_argument("--aspect", type=float, default=0.30, help="bbox height/width floor")
    ap.add_argument("--config", default=None, help="optional device.yaml for model paths")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the labeling page")
    a = ap.parse_args()
    return run(a.video, a.out, sample_seconds=a.sample_seconds, fov_h_deg=a.fov,
               conf=a.conf, aspect=a.aspect, config_path=a.config,
               open_browser=not a.no_open)


if __name__ == "__main__":
    raise SystemExit(main())
