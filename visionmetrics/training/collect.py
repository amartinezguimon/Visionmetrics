"""Training-data collector built on the PRODUCTION vision pipeline.

Records `(yaw, pitch, distance, label)` rows for the engagement model using the
EXACT same person detector + head-pose extractor the live agent uses
(`visionmetrics.edge.agent.vision`). Two consequences that matter:

* **Same "eye" as inference** — whatever the running agent can resolve far away,
  this captures too. The old `src/training/data_collector.py` *looked* like it
  missed far subjects: it recorded a single snapshot on each key-press, and at
  range the face only resolves intermittently, so the press often landed on a
  miss. This tool captures CONTINUOUSLY (like the agent), catching the hits as
  they come.
* **No train/serve skew** — the features written here are computed by the very
  same code that computes them in production, so the model trains on exactly what
  it will see live (the legacy collector duplicated the logic and could drift).

Controls (OpenCV window):
    L  record one LOOKING (1)        A  record one AWAY (0)
    T  toggle continuous capture     M  switch the capture label
    Q  quit and save

Run:
    python -m visionmetrics.training.collect --camera 0
    python -m visionmetrics.training.collect --camera 1 --output data/eval_set.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

# One row per captured sample. Features the model trains on (yaw, pitch, distance)
# + the label + provenance/condition metadata (constant per session) used to track
# coverage and evaluate accuracy per condition. See training/dataset.py.
SESSION_COLUMNS = [
    "yaw", "pitch", "distance", "label", "distance_tier",
    "glasses", "headwear", "subject", "collector", "session", "captured_at",
]


def tier_for(face_width: float) -> str:
    """Coarse distance bucket from normalised face width (for live feedback)."""
    if face_width >= 0.25:
        return "near <0.5m"
    if face_width >= 0.10:
        return "mid 0.5-1.5m"
    if face_width >= 0.04:
        return "far 1.5-3.5m"
    return "v-far >3.5m"


class SampleWriter:
    """Buffers labelled samples and appends them to a CSV (header if new).

    glasses/headwear are recorded PER ROW (they can be toggled mid-session as
    different people walk up), so one session file can hold several people and
    conditions. ``meta`` holds only the run-constant fields (subject/collector/
    session/captured_at).
    """

    def __init__(self, path: str, *, meta: dict | None = None):
        self.path = Path(path)
        self.meta = dict(meta or {})
        self.rows: list[tuple] = []
        self.counts = {1: 0, 0: 0}
        self.tier_counts: dict[str, dict[int, int]] = {}

    def add(self, yaw: float, pitch: float, distance: float, label: int, tier: str,
            glasses: str = "unknown", headwear: str = "unknown") -> None:
        self.rows.append((round(yaw, 5), round(pitch, 5), round(distance, 5),
                          int(label), tier, glasses, headwear))
        self.counts[label] += 1
        self.tier_counts.setdefault(tier, {1: 0, 0: 0})[label] += 1

    def save(self) -> int:
        """Append buffered rows to the CSV. Returns how many were written."""
        if not self.rows:
            return 0
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        m = self.meta
        tail = [m.get("subject", "unknown"), m.get("collector", "unknown"),
                m.get("session", ""), m.get("captured_at", "")]
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(SESSION_COLUMNS)
            for yaw, pitch, distance, label, tier, glasses, headwear in self.rows:
                w.writerow([yaw, pitch, distance, label, tier, glasses, headwear, *tail])
        return len(self.rows)


def _largest_box(detections):
    if not detections:
        return None
    d = max(detections, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
    return d.bbox


def run(camera, output=None, *, fov_h_deg=70.0, every=2, conf=0.25, aspect=0.30,
        config_path=None, glasses="unknown", headwear="unknown",
        subject="unknown", collector="unknown") -> int:
    # Heavy deps imported lazily so the pure helpers above stay cheap to import/test.
    import cv2

    from ..edge.agent.camera_model import focal_length_px
    from ..edge.agent.capture import open_capture
    from ..edge.agent.config import DeviceConfig
    from ..edge.agent.models_bootstrap import ensure_models
    from ..edge.agent.vision.detector import PersonDetector
    from ..edge.agent.vision.face import HeadPoseAnalyzer

    config = DeviceConfig.load(config_path) if config_path else DeviceConfig()
    ensure_models(config)
    v = config.vision

    # Permissive detection for collection (find far people); face extraction is the
    # SAME production code, so what gets recorded matches what inference sees.
    detector = PersonDetector(config.models.yolo, conf_min=conf, aspect_ratio_min=aspect)
    analyzer = HeadPoseAnalyzer(
        config.models.face, face_width_m=v.face_width_m, head_crop_frac=v.head_crop_frac,
        head_upscale=v.head_upscale, skip_frames=1,   # analyse EVERY frame (no skipping)
    )

    cap = open_capture(camera)   # DirectShow on Windows so Camo/OBS open
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {camera!r}. Run 'Ver cámaras' to find the number.")
        return 1

    # One file per RUN (a "session"). Record as many people as you like in one run;
    # toggle glasses/headwear with G/H as different people walk up. Default to a
    # timestamped file so sessions never collide and each is easy to send.
    captured_at = dt.datetime.now(dt.timezone.utc).isoformat()
    if not output:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = "".join(c for c in collector if c.isalnum()).lower() or "session"
        output = f"data/raw_sessions/{stamp}_{safe}.csv"
    session_id = Path(output).stem
    meta = {"subject": subject, "collector": collector,
            "session": session_id, "captured_at": captured_at}

    glasses_cycle = ["no", "yes", "unknown"]
    headwear_cycle = ["none", "cap", "hat", "hood", "unknown"]
    if glasses not in glasses_cycle:
        glasses = "unknown"
    if headwear not in headwear_cycle:
        headwear = "unknown"

    writer = SampleWriter(output, meta=meta)
    auto = False
    label = 1
    focal_px = None
    frame_idx = 0
    print(f"[collect] session={session_id}  collector={collector}")
    print("[collect] L=look  A=away  T=toggle auto  M=switch label  "
          "G=gafas  H=gorra  Q=quit+save")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if focal_px is None:
                focal_px = focal_length_px(frame.shape[1], fov_h_deg)

            box = _largest_box(detector.detect(frame))
            pose = analyzer.analyze(frame, box, 0, frame_idx, focal_px) if box else None
            frame_idx += 1

            tier = "—"
            if pose is not None:
                tier = tier_for(pose.distance)
                cv2.circle(frame, pose.nose_px, 5, (0, 255, 0), -1)
                if auto and frame_idx % max(1, every) == 0:
                    writer.add(pose.yaw, pose.pitch, pose.distance, label, tier, glasses, headwear)

            _draw_hud(cv2, frame, pose, tier, auto, label, writer, glasses, headwear)
            cv2.imshow("VisionMetrics — data collector", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("t"):
                auto = not auto
            elif k == ord("m"):
                label = 0 if label == 1 else 1
            elif k == ord("g"):
                glasses = glasses_cycle[(glasses_cycle.index(glasses) + 1) % len(glasses_cycle)]
            elif k == ord("h"):
                headwear = headwear_cycle[(headwear_cycle.index(headwear) + 1) % len(headwear_cycle)]
            elif k == ord("l") and pose is not None:
                writer.add(pose.yaw, pose.pitch, pose.distance, 1, tier, glasses, headwear)
            elif k == ord("a") and pose is not None:
                writer.add(pose.yaw, pose.pitch, pose.distance, 0, tier, glasses, headwear)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    n = writer.save()
    print(f"[collect] saved {n} samples (look={writer.counts[1]} away={writer.counts[0]})")
    for t, c in writer.tier_counts.items():
        print(f"          {t}: look={c[1]} away={c[0]}")
    if n:
        full = Path(output).resolve()
        print("\n[collect] DONE. Send this ONE file to Alvaro (WhatsApp / email):")
        print(f"          {full}")
    return 0


def _draw_hud(cv2, frame, pose, tier, auto, label, writer, glasses="unknown", headwear="unknown"):
    """Self-explanatory on-screen overlay (Spanish) so the collector needs no manual."""
    h, w = frame.shape[:2]

    def put(txt, x, y, col=(255, 255, 255), s=0.6, th=2):
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)

    def band(y0, y1, alpha=0.5):
        ov = frame.copy()
        cv2.rectangle(ov, (0, y0), (w, y1), (0, 0, 0), -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    # ── top info panel ──
    band(0, 116)
    put("MIRA A LA CAMARA = 'mirando'", 12, 28, (0, 220, 255), 0.62)
    if pose is not None:
        put(f"cara OK   [{tier}]", 12, 54, (0, 220, 80), 0.55)
    else:
        put("sin cara — acercate / mas luz", 12, 54, (80, 80, 255), 0.55)
    if auto:
        col = (0, 220, 80) if label == 1 else (80, 80, 255)
        put(f"* GRABANDO: {'MIRA' if label == 1 else 'NO MIRA'}", 12, 80, col, 0.62)
    else:
        put("en pausa — pulsa T para grabar seguido", 12, 80, (200, 200, 200), 0.55)
    put(f"gafas: {glasses}    gorra: {headwear}        mira: {writer.counts[1]}   no mira: {writer.counts[0]}",
        12, 106, (210, 210, 255), 0.52)

    # ── bottom controls bar (auto-shrink so it always fits the frame width) ──
    band(h - 34, h, 0.55)
    legend = "L=mira   A=no mira   T=grabar   M=cambia   G=gafas   H=gorra   Q=guardar y salir"
    fs = 0.5
    while fs > 0.32 and cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] > w - 24:
        fs -= 0.02
    put(legend, 12, h - 12, (255, 255, 255), fs, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="VisionMetrics training-data collector")
    ap.add_argument("--camera", default="0", help="camera index or path/RTSP url")
    ap.add_argument("--output", default=None,
                    help="CSV to write (default: data/raw_sessions/<timestamp>_<collector>.csv)")
    ap.add_argument("--collector", default="unknown", help="who is running the capture (e.g. hector)")
    ap.add_argument("--subject", default=None,
                    help="who is in front of the camera. Leave unset only when you "
                         "genuinely don't know: rows then group by SESSION for the "
                         "train/test split. Do NOT assume it's the collector — that "
                         "would tag every session with one identity and collapse the split.")
    ap.add_argument("--glasses", choices=["yes", "no", "unknown"], default="unknown")
    ap.add_argument("--headwear", choices=["none", "cap", "hat", "hood", "unknown"], default="unknown")
    ap.add_argument("--fov", type=float, default=70.0, help="camera horizontal FOV (deg)")
    ap.add_argument("--every", type=int, default=2, help="auto-capture every Nth frame")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO person confidence floor")
    ap.add_argument("--aspect", type=float, default=0.30, help="bbox height/width floor")
    ap.add_argument("--config", default=None, help="optional device.yaml for model paths")
    a = ap.parse_args()
    return run(a.camera, a.output, fov_h_deg=a.fov, every=a.every, conf=a.conf,
               aspect=a.aspect, config_path=a.config,
               glasses=a.glasses, headwear=a.headwear,
               subject=a.subject or "unknown", collector=a.collector)


if __name__ == "__main__":
    raise SystemExit(main())
