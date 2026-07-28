"""Edge agent — local web dashboard (replaces the OpenCV --debug preview window).

Runs the exact same detection pipeline as service.py, but instead of opening a
cv2.imshow() window it starts a small local HTTP server (Python stdlib only —
no new dependency, no extra `pip install`) and auto-opens a browser tab: live
camera feed on one side, key stats (passersby, engaged, engagement rate,
attention time) on the other, updated ~twice a second.

Clicking "Stop session" in the browser ends things exactly like pressing Q used
to on the old window: the report (if --report was given) gets written and the
process exits, so whatever launched this (run.py) can carry on and tell the
user where the file is.

Run:
    python -m visionmetrics.edge.agent.webserver --config configs/demo.yaml --debug --report results/demo.json
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from .build import build_pipeline
from .capture import VideoSource, pick_working_camera
from .config import DeviceConfig
from .emitter import MetricEmitter, SessionCounters
from . import viewer
from ...training.collect import SESSION_COLUMNS, tier_for
from ...training.prep import full_frame_to_jpeg_b64

AGENT_VERSION = "0.2.0-web"
DEFAULT_PORT = 8642
_PERF_INTERVAL_S = 5.0
_REVIEW_QUEUE_CAP = 300  # drop oldest if the browser tab stops polling for a while

# One JSON line per finished session — the durable record the dashboard reads to
# show trends across days (footfall, engagement rate, attention) without keeping
# any video or per-person data. This is the ONLY thing that outlives a session
# besides the training CSVs, so it stays deliberately small and append-only.
_HISTORY_PATH = Path("data/metrics_history.jsonl")

# One JSON line per model-performance snapshot (how the model scored against the
# operator's own labels, and every retrain's candidate metrics). Append-only, so
# the analysis screen can plot the model getting better over time — the durable
# backbone behind both the "improvements dashboard" and the downloadable report.
_PERF_PATH = Path("data/model_performance.jsonl")

# Detection-verification rows: is this raw detector box actually a person?
# A parallel, simpler schema to SESSION_COLUMNS — used to audit/improve the
# YOLO detector itself, not the engagement classifier.
DETECTION_COLUMNS = [
    "frame_idx", "track_id", "x1", "y1", "x2", "y2", "conf", "verdict",
    "collector", "session", "captured_at",
]


class _SharedState:
    """Everything the HTTP handlers read/write, guarded by one lock.

    The camera/pipeline loop (main thread) writes; the HTTP server (background
    thread, one worker per request) reads — and sets `stopping` when the
    browser's Stop button is clicked.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.stats: dict = {
            "running": True, "passersby": 0, "engaged": 0, "attention_s": 0.0,
            "people_now": 0, "fps": 0.0, "elapsed_s": 0.0, "store_name": "",
            "recording": False,
        }
        self.stopping = threading.Event()
        # Set by the main screen's ▶ Start button: the camera loop idles until
        # this fires, so the operator can set the capture interval first.
        self.start = threading.Event()
        # Set by Save/Finish: loop the server back to the main screen for a
        # fresh session (instead of quitting), so the app is reusable.
        self.restart = threading.Event()
        # How often (seconds of session time) a whole frame is kept for review.
        # The main screen can change this before each session starts.
        self.review_seconds: float = 10.0
        # Set only when the operator is fully finished (closes the review, or
        # Ctrl-C): the HTTP server keeps serving /api/frames + label endpoints
        # AFTER the camera stops so the post-session review can run, then exits.
        self.done = threading.Event()
        self.report_path: str | None = None
        # Whole-frame samples (one every N seconds) for the post-stop review:
        # {i, t, frame:<jpeg b64>, w, h, people:[{id,box,yaw,pitch,distance,tier,conf}]}.
        # The reviewer scrubs these one at a time AFTER the session is stopped —
        # unlike review_new below, which fed the old live rotating crop panel.
        self.review_frames: list[dict] = []
        self.review_new: list[dict] = []  # crops queued since the last /api/stats poll
        # Durable training-data persistence: every labeled (look/away) photo is
        # kept here keyed by "<frameIdx>_<personId>" (so undo can cleanly
        # replace/remove a row) and the CSV is rewritten on each change.
        self.label_rows: dict[str, dict] = {}
        self.session_id: str = ""
        self.csv_path: Path | None = None
        self.images_dir: Path | None = None
        # Same idea, for detection-verification (is this box a real person?).
        self.detect_new: list[dict] = []
        self.detect_rows: dict[str, dict] = {}
        self.detect_csv_path: Path | None = None
        self.detect_images_dir: Path | None = None
        # Retrain-and-compare loop. The live engagement model the pipeline is
        # running (`engagement_model_path`) is NEVER touched by a retrain until
        # the operator explicitly promotes; retraining writes a *candidate* next
        # to it, which the analysis screen scores side-by-side with the current
        # one. `retrain` is the job status the browser polls.
        self.engagement_model_path: str = ""
        self.candidate_model_path: str = ""
        self.retrain: dict = {
            "running": False, "done": False, "ok": False, "msg": "", "log": "",
        }
        # YOLO detector fine-tune job status the browser polls (separate from the
        # engagement retrain above — this trains the *person detector* on the R
        # "not a person" rejects and hand-drawn missed people from this session).
        self.yolo_ft: dict = {
            "running": False, "done": False, "ok": False, "msg": "", "log": "",
        }

    def reset_for_session(self, session_id: str, csv_path: Path, images_dir: Path,
                          detect_csv_path: Path, detect_images_dir: Path) -> None:
        """Wipe all per-session data so the server can run a fresh session in the
        same process after Save (main screen -> live -> review -> Save -> main).
        `stopping` is cleared here for the new capture loop. `start` is NOT
        touched: it's cleared in the /api/restart handler (synchronously, before
        the browser shows the main screen), so a fast ▶ Start click can't be
        wiped by this reset running a moment later."""
        with self.lock:
            self.stopping.clear()
            self.jpeg = None
            self.review_frames = []
            self.review_new = []
            self.label_rows = {}
            self.detect_new = []
            self.detect_rows = {}
            self.session_id = session_id
            self.csv_path = csv_path
            self.images_dir = images_dir
            self.detect_csv_path = detect_csv_path
            self.detect_images_dir = detect_images_dir
            self.stats.update({
                "running": True, "passersby": 0, "engaged": 0, "attention_s": 0.0,
                "people_now": 0, "fps": 0.0, "elapsed_s": 0.0, "boxes": [], "flow": {},
                "session_csv": str(csv_path),
            })


def _save_labels(state: "_SharedState") -> None:
    """Rewrite the live-session CSV from `state.label_rows` (call with the lock
    held). Session row counts are small (one live dashboard, one operator), so
    a full rewrite on every label/undo is simplest and keeps the file always
    consistent with the in-memory queue — no separate append/undo bookkeeping."""
    if state.csv_path is None:
        return
    state.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state.csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SESSION_COLUMNS)
        for row in state.label_rows.values():
            w.writerow([row.get(c, "") for c in SESSION_COLUMNS])


def _save_detections(state: "_SharedState") -> None:
    """Same idea as `_save_labels`, for the detection-verification CSV."""
    if state.detect_csv_path is None:
        return
    state.detect_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state.detect_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(DETECTION_COLUMNS)
        for row in state.detect_rows.values():
            w.writerow([row.get(c, "") for c in DETECTION_COLUMNS])


def _append_history(store_name: str, totals: dict, started_at: str,
                    ended_at: str, duration_s: float, flow: dict | None = None) -> None:
    """Append one finished session's aggregate metrics to the durable history
    file. Called once at shutdown, always (independent of --report), so every
    run leaves a trail the dashboard can plot across days. Best-effort: a disk
    error here must never crash the shutdown path."""
    pax = int(totals.get("passersby", 0))
    engaged = int(totals.get("engaged", 0))
    row = {
        "ended_at": ended_at,
        "date": ended_at[:10],
        "store_name": store_name,
        "passersby": pax,
        "engaged": engaged,
        "engagement_rate": round(100.0 * engaged / pax, 1) if pax else 0.0,
        "attention_s": round(float(totals.get("attention_s", 0.0)), 1),
        "duration_s": round(float(duration_s), 1),
        "started_at": started_at,
    }
    if flow:
        row["flow"] = {k: int(v) for k, v in flow.items()}
    try:
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as e:  # noqa: BLE001 — history is nice-to-have, never fatal
        print(f"[web] WARNING: could not append session history: {e}")


def _read_history(limit: int = 60) -> list[dict]:
    """Return the last `limit` finished sessions (oldest→newest) from the
    history file, skipping any corrupt line. Empty list if there's no history
    yet — a brand-new store simply shows 'no sessions recorded yet'."""
    if not _HISTORY_PATH.exists():
        return []
    rows: list[dict] = []
    try:
        for line in _HISTORY_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        return []
    return rows[-limit:]


def _append_perf(row: dict) -> None:
    """Append one model-performance snapshot. Best-effort — a disk error here
    must never break the analysis screen or the review flow."""
    try:
        _PERF_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_PERF_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as e:  # noqa: BLE001
        print(f"[web] WARNING: could not append model-performance snapshot: {e}")


def _write_perf(rows: list[dict]) -> None:
    """Rewrite the whole performance log (used to dedupe a session's live-model
    evaluation to a single latest row). Volumes are tiny — one row per session
    plus per-retrain events — so a full rewrite is simplest and always consistent."""
    try:
        _PERF_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_PERF_PATH, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except OSError as e:  # noqa: BLE001
        print(f"[web] WARNING: could not rewrite model-performance log: {e}")


def _read_perf(limit: int = 200) -> list[dict]:
    """Return the last `limit` performance snapshots (oldest→newest), skipping
    any corrupt line. Empty when no evaluation has been logged yet."""
    if not _PERF_PATH.exists():
        return []
    rows: list[dict] = []
    try:
        for line in _PERF_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        return []
    return rows[-limit:]


# Cache of loaded classifiers keyed by (path, mtime), so /api/rescore doesn't
# re-read weights from disk on every threshold tweak. mtime in the key means a
# freshly-retrained candidate is picked up automatically.
_clf_cache: dict[tuple, object] = {}


def _load_classifier(path: str):
    """Load an EngagementClassifier from `path`, memoised by file mtime. Returns
    None if the file is missing or unreadable (so rescore degrades gracefully to
    'only the current model')."""
    from .classifier import EngagementClassifier

    p = Path(path)
    if not p.exists():
        return None
    key = (str(p.resolve()), p.stat().st_mtime)
    clf = _clf_cache.get(key)
    if clf is None:
        try:
            clf = EngagementClassifier.load(path)
        except Exception:  # noqa: BLE001 — a bad/partial weights file just means "no score"
            return None
        _clf_cache[key] = clf
    return clf


def _run_retrain(state: "_SharedState") -> None:
    """Rebuild the master dataset from every session CSV (including the labels
    just made live this session) and train a fresh engagement model into the
    *candidate* path — never the live one. Runs as a background thread; the
    browser polls `/api/retrain_status`. Uses the module CLIs as subprocesses so
    the heavy torch training is isolated from the running server."""
    def _set(**kw):
        with state.lock:
            state.retrain.update(kw)

    _set(running=True, done=False, ok=False, msg="Rebuilding dataset…", log="")
    log_parts: list[str] = []

    def _run(step_name: str, cmd: list[str]) -> bool:
        _set(msg=step_name)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except (subprocess.SubprocessError, OSError) as e:
            log_parts.append(f"$ {' '.join(cmd)}\n{e}")
            _set(running=False, done=True, ok=False,
                 msg=f"{step_name} failed to start", log="\n".join(log_parts)[-4000:])
            return False
        tail = (proc.stdout or "") + (proc.stderr or "")
        log_parts.append(f"$ {' '.join(cmd)}\n{tail.strip()}")
        _set(log="\n".join(log_parts)[-4000:])
        if proc.returncode != 0:
            _set(running=False, done=True, ok=False,
                 msg=f"{step_name} failed (exit {proc.returncode})",
                 log="\n".join(log_parts)[-4000:])
            return False
        return True

    py = sys.executable
    if not _run("Rebuilding dataset…",
                [py, "-m", "visionmetrics.training.build_dataset"]):
        return
    if not _run("Training candidate model…",
                [py, "-m", "visionmetrics.training.train",
                 "--out", state.candidate_model_path]):
        return

    if not Path(state.candidate_model_path).exists():
        _set(running=False, done=True, ok=False,
             msg="Training finished but no candidate weights were written.",
             log="\n".join(log_parts)[-4000:])
        return

    # Surface the candidate's held-out accuracy from the metrics json train.py
    # writes next to the weights, so the operator sees a headline number even
    # before the side-by-side comparison renders.
    acc = None
    metrics_path = Path(state.candidate_model_path).with_name("engagement_metrics.json")
    try:
        acc = json.loads(metrics_path.read_text(encoding="utf-8")).get("overall", {}).get("accuracy")
    except (OSError, ValueError, AttributeError):
        pass
    msg = "Candidate ready — compare below."
    if acc is not None:
        msg = f"Candidate ready (held-out acc {acc}%) — compare below."
    _set(running=False, done=True, ok=True, msg=msg, log="\n".join(log_parts)[-4000:])


def _run_finetune_yolo(state: "_SharedState", epochs: int = 20, imgsz: int = 640) -> None:
    """Fine-tune the PERSON DETECTOR (YOLO) on this session's detection verdicts.

    Every box the operator judged in review is already saved to the session's
    `*_detections.csv` (verdict 1 = real person / hand-drawn missed person,
    verdict 0 = false positive). Here we pair that CSV with the review frames the
    pipeline kept (they share the same frame index + pixel space), stage them as a
    prep JSON, then run the two detector CLIs as subprocesses:
      build_yolo_dataset  -> Ultralytics dataset (rejects become background)
      finetune_detector   -> best.pt copied to models/yolo_finetuned.pt
    Runs on a background thread; the browser polls /api/finetune_yolo_status. The
    live model is NOT swapped — the operator points device.yaml at the new weights
    and restarts, exactly like the CLI flow."""
    def _set(**kw):
        with state.lock:
            state.yolo_ft.update(kw)

    _set(running=True, done=False, ok=False, msg="Preparing frames…", log="")
    log_parts: list[str] = []

    with state.lock:
        frames = list(state.review_frames)
        w = state.stats.get("frame_w") or (frames[0]["w"] if frames else 0)
        h = state.stats.get("frame_h") or (frames[0]["h"] if frames else 0)
        detect_csv = state.detect_csv_path
        n_det = len(state.detect_rows)

    if not detect_csv or not Path(detect_csv).exists() or n_det == 0:
        _set(running=False, done=True, ok=False,
             msg="No detection verdicts yet — in review, press R on a false positive "
                 "or drag a box over a missed person, then try again.")
        return
    if not frames:
        _set(running=False, done=True, ok=False, msg="No captured frames to train on.")
        return

    out_dir = Path("data/detector_dataset")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        prep_json = out_dir / "_live_session.json"
        prep = {"video": {"width": int(w), "height": int(h)},
                "frames": [{"i": f["i"], "frame": f["frame"]} for f in frames]}
        prep_json.write_text(json.dumps(prep), encoding="utf-8")
    except (OSError, KeyError, TypeError) as e:
        _set(running=False, done=True, ok=False, msg=f"Could not stage frames: {e}")
        return

    def _run(step_name: str, cmd: list[str], timeout: int) -> bool:
        _set(msg=step_name)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (subprocess.SubprocessError, OSError) as e:
            log_parts.append(f"$ {' '.join(cmd)}\n{e}")
            _set(running=False, done=True, ok=False,
                 msg=f"{step_name} failed to start", log="\n".join(log_parts)[-4000:])
            return False
        tail = (proc.stdout or "") + (proc.stderr or "")
        log_parts.append(f"$ {' '.join(cmd)}\n{tail.strip()}")
        _set(log="\n".join(log_parts)[-4000:])
        if proc.returncode != 0:
            _set(running=False, done=True, ok=False,
                 msg=f"{step_name} failed (exit {proc.returncode})",
                 log="\n".join(log_parts)[-4000:])
            return False
        return True

    py = sys.executable
    if not _run("Building detector dataset…",
                [py, "-m", "visionmetrics.training.detector.build_yolo_dataset",
                 "--pair", str(detect_csv), str(prep_json), "--out", str(out_dir)],
                timeout=600):
        return
    out_weights = "models/yolo_finetuned.pt"
    if not _run(f"Fine-tuning YOLO ({epochs} epochs — this can take a while)…",
                [py, "-m", "visionmetrics.training.detector.finetune_detector",
                 "--data", str(out_dir / "data.yaml"),
                 "--epochs", str(epochs), "--imgsz", str(imgsz), "--out", out_weights],
                timeout=5400):
        return
    if not Path(out_weights).exists():
        _set(running=False, done=True, ok=False,
             msg="Training finished but no weights were written.",
             log="\n".join(log_parts)[-4000:])
        return
    _set(running=False, done=True, ok=True,
         msg=f"Done — new detector saved to {out_weights}. To use it, set "
             "models.yolo to that path in your config (demo: configs/demo.yaml) and "
             "restart the session.",
         log="\n".join(log_parts)[-4000:])


class BoxSmoother:
    """Exponential moving average of each track's box, for the DRAWN overlay only.

    YOLO+ByteTrack give a box that jitters a few pixels every frame even when the
    person is standing still, and in a busy scene boxes visibly twitch and swim.
    That looks laggy/chaotic to the viewer even when tracking is actually fine.
    We keep a per-track_id EMA of the four corner coords and draw the smoothed
    box instead of the raw one: new = a*raw + (1-a)*prev. `alpha` ~0.4 removes
    the twitch while still following real movement within a couple of frames.

    IMPORTANT: this touches ONLY the pixels we render. Counting, zone tests,
    engagement, and the clickable overlay data all still use the raw p.bbox, so
    smoothing can never change a metric — it is purely cosmetic. Cost is a few
    floats per person, so it stays free on the CPU budget even in a crowd.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._boxes: dict[int, tuple[float, float, float, float]] = {}
        self._seen: dict[int, int] = {}
        self._tick = 0

    def smooth(self, track_id: int, bbox) -> tuple[int, int, int, int]:
        a = self.alpha
        prev = self._boxes.get(track_id)
        if prev is None:
            cur = tuple(float(v) for v in bbox)          # first sighting: adopt as-is
        else:
            cur = tuple(a * float(n) + (1 - a) * p for n, p in zip(bbox, prev))
        self._boxes[track_id] = cur
        self._seen[track_id] = self._tick
        return tuple(int(round(v)) for v in cur)

    def end_frame(self) -> None:
        """Advance time and forget tracks not seen for a while, so the dict can't
        grow without bound over a long session."""
        self._tick += 1
        stale = [tid for tid, t in self._seen.items() if self._tick - t > 30]
        for tid in stale:
            self._boxes.pop(tid, None)
            self._seen.pop(tid, None)


def _draw_boxes(frame, result, store_name: str, smoother: "BoxSmoother | None" = None):
    """Per-person boxes/labels only — no burned-in HUD block, since the
    browser sidebar (not the video) shows the running totals. When a `smoother`
    is given, the DRAWN box is its EMA (less twitch in crowds); the label anchor
    and everything else still key off the smoothed rectangle."""
    for p in result.persons:
        if smoother is not None:
            x1, y1, x2, y2 = smoother.smooth(p.track_id, p.bbox)
        else:
            x1, y1, x2, y2 = p.bbox
        color = viewer._TIER_COLOR.get(p.tier, (100, 100, 100))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{p.track_id}"
        if p.yaw is not None:
            label += f" {p.tier} ({p.engage_prob:.0%})"
        cv2.putText(frame, label, (x1, max(16, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if p.nose_px is not None:
            cv2.circle(frame, p.nose_px, 4, (0, 255, 0), -1)
    if smoother is not None:
        smoother.end_frame()
    return frame


# The three people collecting training data. The UI offers exactly these as the
# "who is labelling" choice, and the analysis bar chart counts rows per name.
KNOWN_COLLECTORS = ["Alvaro", "Hector", "Cristian"]
_RAW_SESSIONS_DIR = Path("data/raw_sessions")


def _iter_engagement_csvs():
    """Yield every engagement (look/away) session CSV that build_dataset.py would
    feed into a retrain — i.e. the real training data. Skips the parallel
    *_detections.csv files, which are detection-audit rows, not training rows."""
    if not _RAW_SESSIONS_DIR.exists():
        return
    for p in sorted(_RAW_SESSIONS_DIR.glob("*.csv")):
        if p.name.endswith("_detections.csv"):
            continue
        yield p


def _training_stats() -> dict:
    """Count the training rows that would go into a retrain, grouped by the
    collector name — so the operator can VERIFY exactly what each person has
    contributed (this is the same data the model trains on)."""
    per: dict[str, int] = {}
    total, files = 0, 0
    for p in _iter_engagement_csvs():
        try:
            with open(p, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "collector" not in reader.fieldnames:
                    continue
                files += 1
                for row in reader:
                    name = (row.get("collector") or "unknown").strip() or "unknown"
                    per[name] = per.get(name, 0) + 1
                    total += 1
        except OSError:
            continue
    return {"total": total, "files": files, "per_collector": per,
            "collectors": KNOWN_COLLECTORS, "columns": SESSION_COLUMNS}


def _training_csv_bytes() -> bytes:
    """Concatenate every engagement session CSV into one downloadable file (one
    header) — the operator's backup / proof of the exact training set."""
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(SESSION_COLUMNS)
    for p in _iter_engagement_csvs():
        try:
            with open(p, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "collector" not in reader.fieldnames:
                    continue
                for row in reader:
                    w.writerow([row.get(c, "") for c in SESSION_COLUMNS])
        except OSError:
            continue
    return buf.getvalue().encode("utf-8")


def _make_handler(state: _SharedState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # keep the console clean — run.py already prints [web] progress lines

        def do_GET(self):
            if self.path == "/":
                body = DASHBOARD_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/stats":
                with state.lock:
                    payload = dict(state.stats)
                    payload["review_new"] = state.review_new
                    state.review_new = []
                    payload["detect_new"] = state.detect_new
                    state.detect_new = []
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/frames":
                with state.lock:
                    payload = {
                        "video": {"width": state.stats.get("frame_w") or 0,
                                  "height": state.stats.get("frame_h") or 0},
                        "session": state.session_id,
                        "running": state.stats.get("running", False),
                        "frames": state.review_frames,
                    }
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/history":
                payload = {"sessions": _read_history()}
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/perf_history":
                payload = {"evals": _read_perf()}
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/retrain_status":
                with state.lock:
                    payload = dict(state.retrain)
                    payload["has_candidate"] = bool(
                        state.candidate_model_path
                        and Path(state.candidate_model_path).exists())
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/finetune_yolo_status":
                with state.lock:
                    payload = dict(state.yolo_ft)
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/training_stats":
                body = json.dumps(_training_stats()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/training_csv":
                body = _training_csv_bytes()
                stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="visionmetrics_training_{stamp}.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream":
                self._stream_mjpeg()
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/api/stop":
                state.stopping.set()
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/quit":
                state.stopping.set()
                state.done.set()
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/start":
                # Operator pressed ▶ Start on the main screen. They can set how
                # often a whole frame is kept for review here, BEFORE capture
                # begins (default stays whatever the CLI passed).
                payload = self._read_json()
                secs = payload.get("review_seconds")
                try:
                    if secs is not None:
                        state.review_seconds = max(1.0, float(secs))
                except (TypeError, ValueError):
                    pass
                state.start.set()
                self._reply_json({"ok": True, "review_seconds": state.review_seconds})
            elif self.path == "/api/restart":
                # Save/Finish on the review or analysis screen: end this review
                # and loop the server back to the main screen for a fresh
                # session. Clear `start` here (before the browser re-shows the
                # main screen) so the next ▶ Start click lands on a clean gate.
                state.start.clear()
                state.stopping.set()
                state.restart.set()
                self._reply_json({"ok": True})
            elif self.path == "/api/label":
                self._handle_label()
            elif self.path == "/api/unlabel":
                self._handle_unlabel()
            elif self.path == "/api/detect_label":
                self._handle_detect_label()
            elif self.path == "/api/detect_unlabel":
                self._handle_detect_unlabel()
            elif self.path == "/api/retrain":
                self._handle_retrain()
            elif self.path == "/api/finetune_yolo":
                self._handle_finetune_yolo()
            elif self.path == "/api/rescore":
                self._handle_rescore()
            elif self.path == "/api/promote":
                self._handle_promote()
            elif self.path == "/api/perf_log":
                self._handle_perf_log()
            else:
                self.send_response(404)
                self.end_headers()

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                return json.loads(raw) if raw else {}
            except ValueError:
                return {}

        def _reply_json(self, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_label(self) -> None:
            """Persist one labeled photo (look/away) as a training row + crop —
            fires every time the operator labels a person in Quick review, so
            live sessions build the CSV/images live instead of only being
            exportable by hand at the end."""
            payload = self._read_json()
            key = str(payload.get("key") or "")
            yaw = payload.get("yaw")
            if not key or yaw is None:
                self._reply_json({"ok": False})
                return
            row = {
                "yaw": yaw, "pitch": payload.get("pitch"), "distance": payload.get("distance"),
                "label": int(payload.get("label", 0)),
                "distance_tier": payload.get("tier") or "",
                "glasses": "unknown", "headwear": "unknown", "subject": "unknown",
                "collector": payload.get("collector") or "unknown",
                "session": state.session_id,
                "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            with state.lock:
                state.label_rows[key] = row
                count = len(state.label_rows)
                _save_labels(state)
            crop_b64 = payload.get("crop")
            if crop_b64 and state.images_dir is not None:
                try:
                    state.images_dir.mkdir(parents=True, exist_ok=True)
                    (state.images_dir / f"{key}.jpg").write_bytes(base64.b64decode(crop_b64))
                except (ValueError, OSError):
                    pass  # bad/short base64 or disk issue — the CSV row is still saved
            self._reply_json({"ok": True, "count": count, "path": str(state.csv_path)})

        def _handle_unlabel(self) -> None:
            payload = self._read_json()
            key = str(payload.get("key") or "")
            with state.lock:
                state.label_rows.pop(key, None)
                count = len(state.label_rows)
                _save_labels(state)
            self._reply_json({"ok": True, "count": count})

        def _handle_detect_label(self) -> None:
            """Persist one detection-box verdict (real person / not a person) —
            fires from the Detections review panel, covering EVERY raw box the
            detector drew (not just ones the engagement pipeline kept), so
            false positives can be caught and used to improve the detector."""
            payload = self._read_json()
            key = str(payload.get("key") or "")
            box = payload.get("box") or [0, 0, 0, 0]
            if not key:
                self._reply_json({"ok": False})
                return
            row = {
                "frame_idx": payload.get("frameIdx"), "track_id": payload.get("id"),
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "conf": payload.get("conf"),
                "verdict": int(payload.get("verdict", 0)),
                "collector": payload.get("collector") or "unknown",
                "session": state.session_id,
                "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            with state.lock:
                state.detect_rows[key] = row
                count = len(state.detect_rows)
                _save_detections(state)
            crop_b64 = payload.get("crop")
            if crop_b64 and state.detect_images_dir is not None:
                try:
                    state.detect_images_dir.mkdir(parents=True, exist_ok=True)
                    (state.detect_images_dir / f"{key}.jpg").write_bytes(base64.b64decode(crop_b64))
                except (ValueError, OSError):
                    pass
            self._reply_json({"ok": True, "count": count, "path": str(state.detect_csv_path)})

        def _handle_detect_unlabel(self) -> None:
            payload = self._read_json()
            key = str(payload.get("key") or "")
            with state.lock:
                state.detect_rows.pop(key, None)
                count = len(state.detect_rows)
                _save_detections(state)
            self._reply_json({"ok": True, "count": count})

        def _handle_retrain(self) -> None:
            """Kick off (or refuse to double-start) a background retrain that
            folds this session's labels into the dataset and trains a candidate
            model. Returns immediately; the browser polls /api/retrain_status."""
            with state.lock:
                if state.retrain.get("running"):
                    self._reply_json({"ok": False, "running": True,
                                      "msg": "A retrain is already in progress."})
                    return
                if not state.candidate_model_path:
                    self._reply_json({"ok": False,
                                      "msg": "Retrain is not available in this run."})
                    return
                state.retrain.update(running=True, done=False, ok=False,
                                     msg="Starting…", log="")
            threading.Thread(target=_run_retrain, args=(state,), daemon=True).start()
            self._reply_json({"ok": True, "started": True})

        def _handle_finetune_yolo(self) -> None:
            """Kick off (or refuse to double-start) a background YOLO detector
            fine-tune on this session's detection verdicts. Returns immediately;
            the browser polls /api/finetune_yolo_status."""
            payload = self._read_json() or {}
            try:
                epochs = max(1, min(200, int(payload.get("epochs") or 20)))
            except (TypeError, ValueError):
                epochs = 20
            try:
                imgsz = max(320, min(1280, int(payload.get("imgsz") or 640)))
            except (TypeError, ValueError):
                imgsz = 640
            with state.lock:
                if state.yolo_ft.get("running"):
                    self._reply_json({"ok": False, "running": True,
                                      "msg": "A detector fine-tune is already running."})
                    return
                state.yolo_ft.update(running=True, done=False, ok=False,
                                     msg="Starting…", log="")
            threading.Thread(target=_run_finetune_yolo, args=(state, epochs, imgsz),
                             daemon=True).start()
            self._reply_json({"ok": True, "started": True})

        def _handle_rescore(self) -> None:
            """Score a batch of head-pose feature triples with BOTH the live
            model and the freshly-trained candidate, so the analysis screen can
            show old-vs-new curves on the operator's own labeled examples. The
            browser stays the source of truth for the labels; we only return
            probabilities."""
            payload = self._read_json()
            items = payload.get("items") or []
            old_clf = _load_classifier(state.engagement_model_path)
            new_clf = _load_classifier(state.candidate_model_path)
            old_scores: list[float | None] = []
            new_scores: list[float | None] = []
            for it in items:
                try:
                    yaw = float(it["yaw"]); pitch = float(it["pitch"]); dist = float(it["distance"])
                except (KeyError, TypeError, ValueError):
                    old_scores.append(None); new_scores.append(None)
                    continue
                old_scores.append(round(old_clf.probability(yaw, pitch, dist), 4) if old_clf else None)
                new_scores.append(round(new_clf.probability(yaw, pitch, dist), 4) if new_clf else None)
            self._reply_json({
                "ok": True, "old": old_scores, "new": new_scores,
                "has_old": old_clf is not None, "has_new": new_clf is not None,
            })

        def _handle_promote(self) -> None:
            """Adopt the candidate as the live engagement model by copying it
            over `models.engagement`. The running pipeline keeps its in-memory
            weights for the rest of this session; the new file is used on the
            next launch. A backup of the previous weights is kept alongside."""
            src = Path(state.candidate_model_path)
            dst = Path(state.engagement_model_path)
            if not src.exists():
                self._reply_json({"ok": False, "msg": "No candidate model to promote."})
                return
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    shutil.copyfile(dst, dst.with_name(dst.stem + "_prev" + dst.suffix))
                shutil.copyfile(src, dst)
            except OSError as e:
                self._reply_json({"ok": False, "msg": f"Could not promote: {e}"})
                return
            self._reply_json({"ok": True, "path": str(dst)})

        def _handle_perf_log(self) -> None:
            """Persist one model-performance snapshot the browser computed on the
            operator's own labels. The browser owns the labels/metrics; here we
            just stamp it with the server's session + store + time and append."""
            payload = self._read_json()
            snap = payload.get("snapshot") or {}
            if not isinstance(snap, dict) or not snap:
                self._reply_json({"ok": False})
                return
            snap.setdefault("at", dt.datetime.now().isoformat(timespec="seconds"))
            snap.setdefault("session", state.session_id)
            with state.lock:
                snap.setdefault("store_name", state.stats.get("store_name", ""))
            # A session's live-model evaluation is a single data point: keep only
            # the latest for that session (re-analyzing after more labels should
            # refine, not duplicate). Retrain/promote events are distinct — append.
            if snap.get("kind") == "session_eval" and snap.get("session"):
                rows = [r for r in _read_perf(limit=10000)
                        if not (r.get("kind") == "session_eval"
                                and r.get("session") == snap["session"])]
                rows.append(snap)
                _write_perf(rows)
            else:
                _append_perf(snap)
            self._reply_json({"ok": True, "count": len(_read_perf())})

        def _stream_mjpeg(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not state.stopping.is_set():
                    with state.lock:
                        jpeg = state.jpeg
                    if jpeg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)   # ~30fps cap on the stream itself
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser tab closed / navigated away — not an error

    return Handler


def _run_one_session(config, state: "_SharedState", *, report_path: str | None,
                     record: bool, record_path: str | None, max_width: int,
                     loop: bool, open_browser: bool, port: int) -> int:
    """Run ONE capture -> review cycle in the already-running HTTP server:
    (re)build the pipeline, open the camera, wait for the browser's ▶ Start,
    capture until Stop, then keep serving the post-stop review until the
    operator Saves (state.restart) or quits (state.done). Rebuilding the
    pipeline per session guarantees the passerby/engaged counters start clean.
    Returns 1 only if the camera can't be opened (fatal for the whole process)."""
    print("[web] loading models and building pipeline...")
    pipeline = build_pipeline(config)

    vsource = VideoSource.from_config(config.camera, loop=loop)
    if not vsource.open():
        print(f"[web] ERROR: cannot open camera source {config.camera.source!r}")
        return 1
    print(f"[web] camera open: {vsource.width}x{vsource.height} @ {vsource.fps:.0f}fps "
          f"(realtime={vsource.realtime})")

    # Downscale before running the pipeline: at native webcam resolution (often
    # 1920x1080) the detector + head-pose + torso models can't keep up with the
    # camera's frame rate on CPU, so the dashboard looks choppy. A person is
    # still plenty resolvable at webcam distance after this, and every stage
    # (detection, drawing, recording, streaming) gets cheaper together since
    # they all run on the same resized frame.
    scale = min(1.0, max_width / vsource.width) if vsource.width > max_width else 1.0
    out_w, out_h = int(vsource.width * scale), int(vsource.height * scale)
    if scale < 1.0:
        print(f"[web] downscaling to {out_w}x{out_h} for realtime processing")

    writer = None
    resolved_record_path: Path | None = None
    if record:
        if record_path:
            resolved_record_path = Path(record_path)
        elif report_path:
            resolved_record_path = Path(report_path).with_suffix(".mp4")
        else:
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            resolved_record_path = Path("recordings") / f"session_{stamp}.mp4"
        resolved_record_path.parent.mkdir(parents=True, exist_ok=True)
        rec_fps = vsource.fps if vsource.fps and vsource.fps > 1 else 15.0
        writer = cv2.VideoWriter(
            str(resolved_record_path), cv2.VideoWriter_fourcc(*"avc1"),
            rec_fps, (out_w, out_h),
        )
        if not writer.isOpened():
            print(f"[web] WARNING: could not open recording file {resolved_record_path} "
                  f"— session will run without recording.")
            writer = None
        else:
            print(f"[web] recording session to -> {resolved_record_path}")

    if vsource.realtime:
        print("[web] esperando imagen de la cámara…")
        t_wait = time.time()
        got_image = False
        # 20s, not 12s: a Continuity Camera (iPhone) can take many seconds to
        # physically wake the first time it's used after a cold launch.
        while time.time() - t_wait < 20.0:
            ok, f = vsource.read()
            if ok and f is not None and float(f.mean()) >= 8:
                print("[web] imagen recibida.")
                got_image = True
                break
            time.sleep(0.3)
        if not got_image:
            print(
                "[web] AVISO: la cámara abrió pero solo entrega imagen negra.\n"
                "      Causas habituales (macOS):\n"
                "        1) Permiso de cámara: System Settings > Privacy & Security >\n"
                "           Camera → activa Terminal (y reinicia esta ventana).\n"
                "        2) Cámara de Continuidad (iPhone) dormida: desbloquea el\n"
                "           iPhone y déjalo cerca; o desactívala para usar la cámara\n"
                "           integrada del Mac.\n"
                "      El programa seguirá; la imagen aparecerá en cuanto la cámara despierte."
            )

    # ---- fresh per-session state (wipes the previous session cleanly) ----
    # Every look/away label the operator makes in review is persisted live, in
    # the same data/raw_sessions/ folder build_dataset.py scans — no manual
    # "download CSV + move file" step needed after the session.
    session_stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"live_{session_stamp}"
    state.reset_for_session(
        session_id,
        Path("data/raw_sessions") / f"{session_id}.csv",
        Path("data/raw_sessions") / f"{session_id}_images",
        Path("data/raw_sessions") / f"{session_id}_detections.csv",
        Path("data/raw_sessions") / f"{session_id}_detimages",
    )
    with state.lock:
        state.stats["recording"] = writer is not None
    print(f"[web] live-labeled samples will be saved to -> {state.csv_path}")

    # ---- wait for the operator to press ▶ Start on the main screen ----
    # They set the capture interval there first; an unattended file/preview run
    # (no browser) just auto-starts so it can't hang waiting for a click.
    if open_browser:
        print("[web] esperando a que pulses ▶ Start en el navegador…")
        while not state.start.is_set():
            if state.done.is_set():
                vsource.release()
                if writer is not None:
                    writer.release()
                return 0
            time.sleep(0.05)
    review_seconds = float(state.review_seconds)
    print(f"[web] repaso: se guardará un frame completo cada {review_seconds:.0f}s")

    emitter = MetricEmitter(config.device.device_id, config.device.store_id,
                            window_s=config.uplink.window_s)
    report_buckets: list[dict] = []
    started_at = dt.datetime.now().isoformat(timespec="seconds")

    def _counters() -> SessionCounters:
        t = pipeline.tracker
        return SessionCounters(passersby=t.total_passersby, engaged=t.total_engaged,
                                total_attention_s=t.total_attention_s())

    def _dispatch(bucket) -> None:
        if report_path:
            report_buckets.append({
                "window_start": bucket.window_start, "window_end": bucket.window_end,
                "passersby": bucket.passersby, "engaged": bucket.engaged,
                "engagement_rate": bucket.engagement_rate,
                "total_attention_s": bucket.total_attention_s,
            })
        print(f"[web] bucket {bucket.window_start} pax={bucket.passersby} "
              f"engaged={bucket.engaged} rate={bucket.engagement_rate}% "
              f"attention={bucket.total_attention_s}s")

    frame_idx = 0
    t0 = time.time()
    session_t0 = t0
    frames_since = 0
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 30
    last_now = 0.0
    file_fps = vsource.fps or 30.0
    # Same person-gating rule as offline prep.py: a passerby's first sighting is
    # always queued for review, then throttled while they linger — so the live
    # panel gets a steady trickle instead of a flood from a stationary customer.
    # Post-stop review keeps one WHOLE frame every `review_seconds` of session
    # time (not per-frame), so the reviewer scrubs a manageable timeline of the
    # whole scene rather than a flood of per-person crops.
    last_review_t = -1e9
    # Cosmetic-only EMA so drawn boxes don't twitch/swim in busy scenes.
    box_smoother = BoxSmoother(alpha=0.4)
    print("[web] running. Click 'Stop session' in the browser (or Ctrl-C here) to end.")
    try:
        while not state.stopping.is_set():
            ok, frame = vsource.read()
            if not ok or frame is None:
                if not vsource.realtime:
                    break
                time.sleep(0.01)
                continue
            if scale < 1.0:
                frame = cv2.resize(frame, (out_w, out_h))

            now = time.time() if vsource.realtime else (frame_idx / file_fps)
            last_now = now
            try:
                result = pipeline.process_frame(frame, frame_idx, now)
                consecutive_errors = 0
            except Exception as e:                       # noqa: BLE001
                consecutive_errors += 1
                print(f"[web] frame {frame_idx} failed ({consecutive_errors}): {e}")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print("[web] too many consecutive frame errors; exiting.")
                    break
                frame_idx += 1
                continue
            frame_idx += 1
            frames_since += 1

            if writer is not None:
                # Raw frame, no boxes — matches what training/prep.py expects
                # to re-run detection on later.
                writer.write(frame)

            for bucket in emitter.sample(_counters(), now):
                _dispatch(bucket)

            # Queue crops for the live "Quick review" panel — must run on the
            # RAW frame, before _draw_boxes() burns boxes into it below.
            review_people = [
                {"id": p.track_id, "box": list(p.bbox), "yaw": p.yaw, "pitch": p.pitch,
                 "dist_m": p.dist_m, "distance": p.distance,
                 "tier": tier_for(p.distance) if p.distance is not None else None,
                 "engaged": bool(p.is_engaged), "p_look": round(float(p.engage_prob), 4)}
                for p in result.persons
            ]
            # Every-frame (ungated) box positions, so the dashboard can draw
            # clickable overlays on the live stream — separate from the
            # throttled review-photo queue below, which only fires periodically.
            current_boxes = [
                {"id": p["id"], "box": p["box"], "tier": p["tier"], "engaged": p["engaged"]}
                for p in review_people
            ]
            t_session = time.time() - session_t0
            if t_session - last_review_t >= review_seconds:
                last_review_t = t_session
                # Full-fidelity JPEG of the exact frame just processed (no extra
                # downscale, near-lossless) so the review crops are the true pixels
                # the camera saw at that instant, faithful as training data.
                frame_b64 = full_frame_to_jpeg_b64(frame, max_dim=max(out_w, out_h), quality=95)
                people = [
                    {"id": p["id"], "box": p["box"], "yaw": p["yaw"], "pitch": p["pitch"],
                     "distance": p["distance"], "tier": p["tier"], "conf": p.get("conf"),
                     "engaged": p["engaged"], "p_look": p.get("p_look")}
                    for p in review_people
                ]
                with state.lock:
                    state.review_frames.append({
                        # frame_idx was already ++'d above, so the pixels in `frame`
                        # are index frame_idx-1 — the same frame the recorder wrote.
                        # Store that so the detection CSV seeks the right frame later.
                        "i": frame_idx - 1, "t": round(t_session, 1),
                        "frame": frame_b64, "w": out_w, "h": out_h, "people": people,
                    })
                    if len(state.review_frames) > _REVIEW_QUEUE_CAP:
                        state.review_frames = state.review_frames[-_REVIEW_QUEUE_CAP:]

            annotated = _draw_boxes(frame, result, config.device.store_name, box_smoother)
            ok_enc, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            with state.lock:
                if ok_enc:
                    state.jpeg = buf.tobytes()
                state.stats.update({
                    "running": True,
                    "passersby": pipeline.tracker.total_passersby,
                    "engaged": pipeline.tracker.total_engaged,
                    "attention_s": round(pipeline.tracker.total_attention_s(), 1),
                    "people_now": len(result.active_ids),
                    "elapsed_s": round(time.time() - session_t0, 1),
                    "boxes": current_boxes,
                    "frame_w": out_w,
                    "frame_h": out_h,
                    "flow": dict(pipeline.flow),
                })

            elapsed = time.time() - t0
            if elapsed >= _PERF_INTERVAL_S:
                fps = frames_since / elapsed
                with state.lock:
                    state.stats["fps"] = round(fps, 1)
                print(f"[web] {fps:.1f} fps | passersby={pipeline.tracker.total_passersby} "
                      f"engaged={pipeline.tracker.total_engaged}")
                t0, frames_since = time.time(), 0
    finally:
        final = emitter.flush(_counters(), last_now)
        if final is not None:
            _dispatch(final)
        vsource.release()
        if writer is not None:
            writer.release()
        with state.lock:
            state.stats["running"] = False

    t = pipeline.tracker
    print(f"[web] stopped. passersby={t.total_passersby} engaged={t.total_engaged} "
          f"attention={t.total_attention_s():.0f}s")

    # Always leave a durable trail so the dashboard can plot trends across days,
    # even when this run had no --report. One tiny JSON line, aggregate only.
    ended_at = dt.datetime.now().isoformat(timespec="seconds")
    _append_history(
        config.device.store_name,
        {"passersby": t.total_passersby, "engaged": t.total_engaged,
         "attention_s": round(t.total_attention_s(), 1)},
        started_at, ended_at, duration_s=time.time() - session_t0,
        flow=dict(pipeline.flow),
    )

    if writer is not None and resolved_record_path is not None:
        print(f"[web] session recording saved -> {resolved_record_path.resolve()}")
        print("[web] to use it for training: menu option 6 (Importar y etiquetar) "
              "-> give it that video path.")

    if report_path:
        report = {
            "device_id": config.device.device_id,
            "store_name": config.device.store_name,
            "agent_version": AGENT_VERSION,
            "started_at": started_at,
            "ended_at": dt.datetime.now().isoformat(timespec="seconds"),
            "frames_processed": frame_idx,
            "totals": {
                "passersby": t.total_passersby,
                "engaged": t.total_engaged,
                "attention_s": round(t.total_attention_s(), 1),
            },
            "buckets": report_buckets,
        }
        p = Path(report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[web] report saved -> {p.resolve()}")

    # The camera is stopped, but the HTTP server keeps running so the browser
    # can do the post-session review (fetch /api/frames, POST labels). Block
    # here until the operator Saves (state.restart -> back to the main screen)
    # or quits (state.done). An unattended file/preview run (no browser) returns
    # immediately so the caller's loop can exit after one pass.
    if state.done.is_set() or not open_browser:
        return 0
    with state.lock:
        has_frames = bool(state.review_frames)
    if has_frames:
        print(f"[web] sesión parada — repaso abierto en http://localhost:{port}/  "
              f"(Guarda para volver al inicio, o Ctrl-C para salir)")
    else:
        print("[web] no se capturó ningún frame para repasar.")
    while not state.done.is_set() and not state.restart.is_set():
        time.sleep(0.1)
    return 0


def run(config_path: str, *, debug: bool = False, report_path: str | None = None,
        source: str | None = None, port: int = DEFAULT_PORT,
        open_browser: bool = True, record: bool = True,
        record_path: str | None = None, review_sample_every: int = 30,
        review_seconds: float = 10.0,
        max_width: int = 640, loop: bool = False) -> int:
    config = DeviceConfig.load(config_path)
    if source is not None:
        config.camera.source = int(source) if str(source).isdigit() else source
    # Make the camera "just work" on a machine we've never seen. When the source
    # is a webcam index, pick one that actually delivers an image: on a Mac index 0
    # is often an absent/black Continuity Camera while the real webcam is 1 or 2,
    # so the hard-coded 0 the double-click launcher passes would show a black feed
    # on a friend's laptop. File/RTSP sources are left untouched.
    if isinstance(config.camera.source, int):
        picked = pick_working_camera(config.camera.source)
        if picked is None:
            print("[web] AVISO: ninguna cámara entregó imagen en los índices 0-2.\n"
                  "      1) Permiso de cámara: System Settings > Privacy & Security >\n"
                  "         Camera → activa Terminal, luego cierra y reabre esta ventana.\n"
                  "      2) Si usas la cámara de Continuidad (iPhone), desbloquéalo y\n"
                  "         déjalo cerca; o desactívala para usar la cámara del Mac.")
        elif picked != config.camera.source:
            print(f"[web] camera index {config.camera.source} produced no image; "
                  f"using the working camera at index {picked} instead.")
            config.camera.source = picked
    print(f"[web] device={config.device.device_id} store='{config.device.store_name}'")

    # ---- one HTTP server + one shared state for the whole process ----
    state = _SharedState()
    state.stats["store_name"] = config.device.store_name
    state.review_seconds = float(review_seconds)
    # Retrain-and-compare wiring (constant across sessions): the live engagement
    # model the pipeline serves, and where a retrain drops its candidate.
    state.engagement_model_path = config.models.engagement
    state.candidate_model_path = str(
        Path(config.models.engagement).with_name("engagement_candidate.pth"))

    httpd = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(state))
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    url = f"http://localhost:{port}/"
    print(f"[web] dashboard -> {url}")
    if open_browser:
        webbrowser.open(url)

    def _stop(*_):
        state.stopping.set()
        state.done.set()  # Ctrl-C / SIGTERM: end the camera AND quit the review server
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # ---- session loop: main screen -> live -> review -> Save loops back here ----
    try:
        while not state.done.is_set():
            rc = _run_one_session(
                config, state, report_path=report_path, record=record,
                record_path=record_path, max_width=max_width, loop=loop,
                open_browser=open_browser, port=port)
            if rc != 0:
                break  # camera failed to open — fatal
            if state.restart.is_set():
                state.restart.clear()
                print("[web] nueva sesión — volviendo a la pantalla principal…")
                continue
            break  # done (quit) or an unattended one-shot run
    finally:
        time.sleep(0.2)
        httpd.shutdown()
        httpd.server_close()
    return 0


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VisionMetrics — Live</title>
<style>
  :root {
    --look:#3fae74; --away:#e15a4d; --new:#0bb3a6; --reject:#ef8f3c;
    --bg:#f2efe8; --card:#ffffff; --card2:#faf8f4; --border:#e9e5dc;
    --ink:#1c1b22; --muted:#9a97a2; --accent:#4a2a86; --accent-soft:#efe9f8;
    --purple-deep:#3a1d6e;
  }
  * { box-sizing: border-box; }
  html, body { height:100%; }
  html { scroll-behavior:smooth; }
  /* App is a fixed-viewport shell: the header is a thin fixed strip and exactly
     one screen (live / label / analysis) fills the rest. Nothing scrolls the
     page — each screen owns the whole viewport. */
  body {
    margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background:var(--bg); color:var(--ink); -webkit-font-smoothing:antialiased;
    display:flex; flex-direction:column; height:100vh; overflow:hidden;
  }
  header { flex:none; }
  #startScreen, #layout, #analysis { flex:1 1 auto; min-height:0; }
  /* Main / start screen: one centered card to name yourself, set the capture
     interval, and start. Owns the whole viewport like the other screens. */
  #startScreen { display:flex; align-items:center; justify-content:center; padding:24px; }
  .startCard {
    width:min(440px, 92vw); background:var(--card); border:1px solid var(--border);
    border-radius:18px; padding:28px 30px; box-shadow:0 10px 30px rgba(40,25,80,.08);
  }
  .startCard h2 { margin:0 0 6px; font-size:22px; color:var(--ink); }
  .startCard .startSub { margin:0 0 20px; font-size:13px; color:var(--muted); line-height:1.5; }
  .startCard label { display:block; font-size:12px; font-weight:700; color:var(--ink); margin:14px 0 6px; }
  .startCard input[type=text], .startCard input[type=number], .startCard select {
    width:100%; padding:10px 12px; font-size:15px; border:1px solid var(--border);
    border-radius:10px; background:var(--card2); color:var(--ink);
  }
  .startCard .secRow { display:flex; align-items:center; gap:10px; }
  .startCard .secRow input { width:110px; }
  .startCard .secRow span { font-size:14px; color:var(--muted); }
  #btnStart {
    width:100%; margin-top:22px; padding:14px; font-size:16px; font-weight:800;
    color:#fff; background:var(--look); border:1px solid var(--look);
    border-radius:12px; cursor:pointer;
  }
  #btnStart:hover { filter:brightness(1.05); }
  #btnStart:disabled { opacity:.6; cursor:default; }
  #startStatus { margin-top:12px; font-size:12px; color:var(--muted); min-height:16px; }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { transition:none !important; animation:none !important; }
  }
  header {
    display:flex; align-items:center; gap:16px; padding:12px 32px; background:var(--bg);
  }
  #logo {
    display:flex; align-items:center; gap:9px; flex:none;
    background:var(--purple-deep); color:#fff; border-radius:12px;
    padding:9px 16px; font-weight:800; font-size:16px; letter-spacing:0.2px;
    box-shadow:0 4px 14px rgba(58,29,110,0.28);
    transition:transform .2s ease, box-shadow .2s ease;
  }
  #logo:hover { transform:translateY(-1px); box-shadow:0 8px 22px rgba(58,29,110,0.34); }
  #logo .dot { width:8px; height:8px; border-radius:50%; background:#c9b8e8; }
  .head-text { flex:1; min-width:0; }
  header h1 { font-size:18px; margin:0; font-weight:600; letter-spacing:-0.2px; color:var(--ink); }
  #store { color:var(--ink); }
  .live {
    display:flex; align-items:center; gap:7px; flex:none; font-size:12.5px; font-weight:600;
    color:#4a4752; background:var(--card); border:1px solid var(--border);
    padding:8px 14px; border-radius:999px; box-shadow:0 1px 4px rgba(0,0,0,0.03);
    transition:box-shadow .2s ease, transform .2s ease;
  }
  .live:hover { box-shadow:0 4px 12px rgba(0,0,0,0.06); transform:translateY(-1px); }
  .live .dot { width:8px; height:8px; border-radius:50%; background:var(--look);
    box-shadow:0 0 0 0 rgba(63,174,116,0.5); animation:livepulse 2s ease-out infinite; }
  @keyframes livepulse {
    0% { box-shadow:0 0 0 0 rgba(63,174,116,0.45); }
    70% { box-shadow:0 0 0 6px rgba(63,174,116,0); }
    100% { box-shadow:0 0 0 0 rgba(63,174,116,0); }
  }
  .avatar {
    width:34px; height:34px; border-radius:50%; flex:none; background:var(--purple-deep);
    color:#fff; display:flex; align-items:center; justify-content:center;
    font-size:12px; font-weight:700; letter-spacing:0.3px;
  }

  #layout { display:flex; gap:20px; padding:12px 32px 20px; align-items:stretch; min-height:0; overflow:hidden; }
  #videoCol { flex:4; min-width:460px; display:flex; flex-direction:column; min-height:0; }
  #statsCol { flex:1; min-width:280px; min-height:0; overflow-y:auto; padding-right:4px;
              display:flex; flex-direction:column; gap:9px; justify-content:space-between; }
  /* Cards live in a flex column that spreads to the camera's height: with room to
     spare they space out evenly so the stack lines up with the video; when tight
     the column just scrolls. gap replaces the per-panel margin so spacing stays even. */
  #statsCol .panel { margin-bottom:0; }

  /* The review photo fills its whole box edge-to-edge (object-fit:cover), matching
     the live feed. toCanvas() inverts this cover transform so clicks stay aligned. */
  #rcanvas { display:block; width:100%; height:100%; object-fit:cover;
             margin:0; cursor:crosshair; }
  #reviewBar { margin-top:10px; flex:none; }
  .revNav { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
  .revNav #rCounter {
    flex:1; text-align:center; font-weight:700; font-size:13px; color:#4a4752;
    font-variant-numeric:tabular-nums;
  }
  .revNav .rbtn { flex:none; width:auto; min-width:96px; }

  #stage {
    position:relative; border-radius:16px; overflow:hidden; border:1px solid var(--border);
    box-shadow:0 10px 30px rgba(28,27,34,0.10); background:#1c1b22;
    flex:1 1 auto; min-height:0;
    display:flex; align-items:center; justify-content:center;
  }
  /* Live feed fills the whole stage edge-to-edge (no letterbox bars). Only the
     live <img> is covered — the review <canvas> keeps its own contain sizing so
     box coordinates stay aligned. */
  #stage img { display:block; width:100%; height:100%; object-fit:cover; }
  #stage.stopped::after {
    content:"Session stopped"; position:absolute; inset:0; display:flex; align-items:center;
    justify-content:center; background:rgba(28,27,34,0.6); font-size:20px; font-weight:600; color:#fff;
  }
  #stageInner { position:relative; width:100%; height:100%; min-height:0; line-height:0;
                display:flex; align-items:center; justify-content:center; }
  #boxOverlay { position:absolute; inset:0; pointer-events:none; }
  #detectOverlay { position:absolute; inset:0; pointer-events:none; display:none; }
  .pbox {
    position:absolute; border:2px solid var(--accent); border-radius:4px; box-sizing:border-box;
    cursor:pointer; pointer-events:auto; transition:border-color 0.1s;
  }
  .pbox:hover { border-color:#fff; }
  .pbox.engaged { border-color:var(--look); }
  .pbox.selected { border-color:#fff; box-shadow:0 0 0 2px var(--accent); }
  .pbox .tag {
    position:absolute; top:-20px; left:-2px; background:var(--accent); color:#fff; font-size:11px;
    font-weight:700; padding:2px 7px; border-radius:5px 5px 0 0; white-space:nowrap; line-height:1.6;
  }
  .pbox.engaged .tag { background:var(--look); }
  .pbox.dbox { border-color:#f5c518; border-style:dashed; }
  .pbox.dbox .tag { background:#f5c518; color:#111; }
  .pbox.dbox.verified-yes { border-color:var(--look); border-style:solid; }
  .pbox.dbox.verified-no { border-color:var(--away); border-style:solid; }

  .modeRow {
    display:flex; gap:4px; margin-bottom:16px; background:var(--card2);
    border:1px solid var(--border); border-radius:999px; padding:4px;
  }
  .modebtn {
    background:transparent; color:var(--muted); border:none; border-radius:999px;
    padding:8px 14px; font-size:12.5px; font-weight:700; cursor:pointer; width:auto; flex:1;
    transition:background .2s ease, color .2s ease, box-shadow .2s ease;
  }
  .modebtn:hover { color:var(--ink); background:rgba(74,42,134,0.06); }
  .modebtn.active { background:var(--purple-deep); color:#fff; box-shadow:0 2px 8px rgba(58,29,110,0.25); }
  .modebtn.active:hover { background:var(--purple-deep); color:#fff; }

  .panel {
    background:var(--card); border:1px solid var(--border); border-radius:14px;
    padding:10px 14px; box-shadow:0 2px 12px rgba(28,27,34,0.05); margin-bottom:9px;
    transition:box-shadow .25s ease, transform .25s ease, border-color .25s ease;
  }
  .panel:hover {
    box-shadow:0 10px 28px rgba(28,27,34,0.09); transform:translateY(-2px); border-color:#e0dace;
  }
  .panel h3 {
    margin:0 0 6px; font-size:11px; color:var(--muted); text-transform:uppercase;
    letter-spacing:0.8px; font-weight:700;
  }
  .stat { display:flex; justify-content:space-between; align-items:baseline; padding:4px 0;
          border-bottom:1px solid var(--border); }
  .stat:last-child { border-bottom:none; }
  .stat .label { font-size:13px; color:var(--muted); }
  .stat .value { font-size:20px; font-weight:700; font-variant-numeric:tabular-nums; color:var(--ink); }
  .stat .value.look { color:var(--look); }
  .stat .value.accent { color:var(--accent); }

  button {
    background:var(--away); color:#fff; border:1px solid var(--away); border-radius:10px;
    padding:12px 16px; cursor:pointer; font-size:14px; font-weight:700; width:100%;
    transition:transform .16s ease, box-shadow .16s ease, filter .16s ease, opacity .16s ease;
  }
  button:hover { filter:brightness(1.04); transform:translateY(-1px); box-shadow:0 8px 20px rgba(28,27,34,0.16); }
  button:active { transform:translateY(0); box-shadow:0 2px 8px rgba(28,27,34,0.12); }
  button:disabled { opacity:0.5; cursor:default; transform:none; box-shadow:none; }
  #status { font-size:12px; color:var(--muted); margin-top:10px; text-align:center; }

  #recBadge {
    display:none; align-items:center; gap:6px; padding:7px 13px;
    border-radius:999px; background:rgba(225,90,77,0.10); border:1px solid rgba(225,90,77,0.35);
    color:#c0463a; font-size:12px; font-weight:700; letter-spacing:0.4px; flex:none;
  }
  #recBadge.on { display:flex; }
  #recBadge .dot {
    width:8px; height:8px; border-radius:50%; background:var(--away);
    animation:pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.25; } }

  .panel input[type=text], .panel select {
    width:100%; padding:9px 11px; background:var(--card2); border:1px solid var(--border);
    color:var(--ink); border-radius:9px; font-size:13px; margin-top:6px;
    transition:border-color .18s ease, background .18s ease, box-shadow .18s ease;
  }
  .panel input[type=text]:focus, .panel select:focus { outline:none; border-color:var(--accent); background:#fff;
    box-shadow:0 0 0 3px rgba(74,42,134,0.12); }
  .panel label { display:block; font-size:12px; color:var(--muted); }

  #reviewStage, #detectStage {
    position:relative; border-radius:12px; overflow:hidden; border:1px solid var(--border);
    background:#1c1b22; min-height:200px; display:flex; align-items:center; justify-content:center;
  }
  #reviewPhoto, #detectPhoto { display:block; max-width:100%; max-height:280px; }
  #reviewPlaceholder, #detectPlaceholder { color:var(--muted); font-size:13px; padding:20px; text-align:center; }
  #reviewBadge, #detectBadge {
    position:absolute; top:10px; left:10px; padding:5px 10px; border-radius:6px;
    font-weight:700; font-size:12px; color:#fff; display:none;
  }

  #reviewProgressBar {
    margin-top:12px; height:7px; border-radius:5px; background:var(--card2);
    border:1px solid var(--border); overflow:hidden;
  }
  #reviewProgressFill { height:100%; width:0%; background:var(--accent); transition:width 0.15s; }
  #reviewProgress, #detectProgress { margin-top:8px; font-size:12.5px; color:#4a4752; }
  #reviewStats, #detectStats { margin-top:4px; font-size:11.5px; color:var(--muted); }

  .rbtnrow { margin-top:12px; display:flex; gap:8px; flex-wrap:wrap; }
  .rbtn {
    flex:1; min-width:90px; background:var(--card); border:1px solid var(--border);
    color:var(--ink); border-radius:10px; padding:9px 10px; font-size:12.5px; font-weight:600;
    transition:transform .16s ease, box-shadow .16s ease, background .16s ease, border-color .16s ease;
  }
  .rbtn:hover { filter:none; background:var(--card2); border-color:#dcd6ca; transform:translateY(-1px); box-shadow:0 6px 16px rgba(28,27,34,0.10); }
  .rbtn:active { transform:translateY(0); box-shadow:0 1px 4px rgba(28,27,34,0.08); }
  .rbtn.look { border-color:var(--look); color:#2f7d55; }
  .rbtn.look:hover { background:rgba(63,174,116,0.10); }
  .rbtn.away { border-color:var(--away); color:#b8433a; }
  .rbtn.away:hover { background:rgba(225,90,77,0.10); }
  .rbtn.reject { border-color:var(--reject); color:#c26e1e; }
  .rbtn.reject:hover { background:rgba(239,143,60,0.10); }

  .rhint { margin-top:12px; font-size:11.5px; color:#4a4752; line-height:1.7; background:var(--card2);
           border:1px solid var(--border); border-radius:12px; padding:10px 12px; }
  .kbd {
    background:#fff; border:1px solid var(--border); border-radius:6px; padding:1px 6px;
    font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:11px; color:var(--accent);
    box-shadow:0 1px 2px rgba(28,27,34,0.04);
  }

  /* ---- model-check (analysis) phase ---- */
  #analysis { padding:12px 32px 24px; overflow-y:auto; }
  #anNav { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px; }
  #anNav .an-title { font-size:14px; font-weight:700; color:var(--ink); }
  #anNav button { flex:none; width:auto; }
  #analysis .panel {
    background:var(--card); border:1px solid var(--border); border-radius:16px;
    padding:16px 18px; box-shadow:0 2px 10px rgba(28,27,34,0.05);
  }
  #analysis h3 { margin:0 0 6px; font-size:14px; color:var(--ink); }
  .an-grid { display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start; margin-top:8px; }
  .an-col { flex:1; min-width:280px; }
  .an-plot { width:100%; height:auto; background:var(--card2); border:1px solid var(--border); border-radius:12px; }
  .an-cap { font-size:12px; color:#4a4752; margin-top:6px; text-align:center; font-weight:600; }
  .cm { display:grid; grid-template-columns:auto 1fr 1fr; gap:6px; margin-top:12px; }
  .cm .hd { color:var(--muted); font-size:11.5px; font-weight:600; align-self:center; text-align:center; }
  .cm .cell { padding:10px 8px; border-radius:9px; text-align:center; font-size:11px; color:var(--muted);
              border:1px solid var(--border); letter-spacing:0.4px; }
  .cm .cell b { display:block; font-size:19px; color:var(--ink); margin-top:2px; }
  .cm .tp, .cm .tn { background:rgba(63,174,116,0.13); }
  .cm .fp { background:rgba(225,90,77,0.13); }
  .cm .fn { background:rgba(239,143,60,0.13); }
  .metric-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
  .metric { flex:1; min-width:84px; background:var(--card2); border:1px solid var(--border);
            border-radius:10px; padding:10px 8px; text-align:center;
            transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease; }
  .metric:hover { transform:translateY(-2px); box-shadow:0 6px 16px rgba(28,27,34,0.08); border-color:#e0dace; }
  .metric .v { font-size:20px; font-weight:700; color:var(--accent); }
  .metric .l { font-size:10.5px; color:var(--muted); text-transform:uppercase; letter-spacing:0.6px; margin-top:2px; }
  .anhint { font-size:11.5px; color:#4a4752; line-height:1.7; background:var(--card2);
            border:1px solid var(--border); border-radius:12px; padding:10px 12px; }
  #retrainBox h3, #perfBox h3 { font-size:14px; color:var(--ink); text-transform:none; letter-spacing:0; }
  .perf { width:100%; border-collapse:collapse; font-size:12px; }
  .perf th, .perf td { padding:6px 8px; text-align:right; border-bottom:1px solid var(--border); white-space:nowrap; }
  .perf th:first-child, .perf td:first-child { text-align:left; color:var(--muted); }
  .perf th { font-size:10.5px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; }
  .perf td { font-variant-numeric:tabular-nums; color:#4a4752; }
  .perf tbody tr { transition:background .15s ease; }
  .perf tbody tr:hover { background:var(--card2); }
  .perf tr.cand td:first-child { color:var(--accent); font-weight:700; }
  .perf .best { color:var(--look); font-weight:700; }
  .cmp { width:100%; border-collapse:collapse; font-size:13px; }
  .cmp th, .cmp td { padding:8px 10px; text-align:right; border-bottom:1px solid var(--border); }
  .cmp th:first-child, .cmp td:first-child { text-align:left; color:var(--muted); font-weight:600; }
  .cmp th { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; }
  .cmp td.now { color:#4a4752; font-variant-numeric:tabular-nums; }
  .cmp td.cand { color:var(--accent); font-weight:700; font-variant-numeric:tabular-nums; }
  .cmp .delta { font-size:11px; font-weight:700; }
  .cmp .up { color:var(--look); }
  .cmp .down { color:var(--away); }
  /* history / trends mini-list */
  #trendList .trow { display:flex; justify-content:space-between; gap:8px; padding:5px 0;
                     border-bottom:1px solid var(--border); }
  #trendList .trow:last-child { border-bottom:none; }
  #trendList .tdate { color:var(--muted); }
  #trendList .tnums { font-variant-numeric:tabular-nums; }
  #trendList .tnums b { color:var(--look); }
</style>
</head>
<body>
<header>
  <div id="logo"><span class="dot"></span>VisionMetrics</div>
  <div class="head-text">
    <h1 id="store">—</h1>
  </div>
  <div id="recBadge"><span class="dot"></span>REC</div>
  <div class="live"><span class="dot"></span>En directo</div>
  <div class="avatar">VM</div>
</header>

<div id="startScreen">
  <div class="startCard">
    <h2>Start a new session</h2>
    <p class="startSub">The live camera fills the screen. When you stop, you'll review the captured frames and check the model.</p>
    <label for="startCollector">Who is labelling? (collector)</label>
    <select id="startCollector">
      <option value="Alvaro">Alvaro</option>
      <option value="Hector">Hector</option>
      <option value="Cristian">Cristian</option>
    </select>
    <label for="startSeconds">Save a frame for training every…</label>
    <div class="secRow">
      <input type="number" id="startSeconds" min="1" max="120" step="1" value="10">
      <span>seconds</span>
    </div>
    <button id="btnStart">▶ Start session</button>
    <div id="startStatus"></div>
  </div>
</div>

<div id="layout" style="display:none;">
  <div id="videoCol">
    <div id="stage">
      <div id="stageInner">
        <img id="stream" src="/stream" alt="live camera">
        <canvas id="rcanvas" style="display:none;"></canvas>
      </div>
    </div>

    <div id="reviewBar" style="display:none;">
      <div class="revNav">
        <button class="rbtn" id="rPrev">← Prev</button>
        <span id="rCounter">Frame 0 / 0</span>
        <button class="rbtn" id="rNext">Next →</button>
      </div>
      <div class="rbtnrow">
        <button class="rbtn look" id="rLook">L · Looking</button>
        <button class="rbtn away" id="rAway">A · Not looking</button>
        <button class="rbtn reject" id="rReject">R · Not a person</button>
        <button class="rbtn" id="rDelete">Delete box</button>
      </div>
      <div class="rhint">
        Just tap the keys — the next box is highlighted for you. Drag on empty space to add a missed person.<br>
        <span class="kbd">L</span> looking &nbsp; <span class="kbd">A</span> not looking &nbsp;
        <span class="kbd">R</span> not a person &nbsp; <span class="kbd">←</span>/<span class="kbd">→</span> frames
      </div>
      <div id="reviewStatus" style="font-size:11.5px;color:#4a4752;margin-top:8px;"></div>
    </div>
  </div>

  <div id="statsCol">
    <div class="panel">
      <h3>Right now</h3>
      <div class="stat"><span class="label">People in frame</span><span class="value accent" id="peopleNow">0</span></div>
      <div class="stat"><span class="label">Session time</span><span class="value" id="elapsed">0:00</span></div>
    </div>
    <div class="panel">
      <h3>Session totals</h3>
      <div class="stat"><span class="label">Passersby</span><span class="value" id="passersby">0</span></div>
      <div class="stat"><span class="label">Looked (engaged)</span><span class="value look" id="engaged">0</span></div>
      <div class="stat"><span class="label">Engagement rate</span><span class="value look" id="rate">0%</span></div>
      <div class="stat"><span class="label">Total attention</span><span class="value" id="attention">0s</span></div>
    </div>
    <div class="panel" id="flowPanel">
      <h3>Foot-traffic flow</h3>
      <div id="flowSplit" style="display:flex; height:24px; border-radius:6px; overflow:hidden; border:1px solid var(--border); margin-bottom:10px;">
        <div id="flowLeftSeg" style="width:50%; background:var(--accent);"></div>
        <div id="flowRightSeg" style="width:50%; background:#0bb3a6;"></div>
      </div>
      <div class="stat"><span class="label"><span style="color:var(--accent);">●</span> Came from the left →</span><span class="value" id="flowLeft">0</span></div>
      <div class="stat"><span class="label">← <span style="color:#0bb3a6;">Came from the right</span></span><span class="value" id="flowRight">0</span></div>
      <div id="flowNote" style="font-size:11px; color:var(--muted); margin-top:8px; line-height:1.5;"></div>
      <div id="flowEmpty" style="font-size:11.5px; color:var(--muted);">Not enough movement yet.</div>
    </div>
    <div class="panel" id="stopPanel">
      <label>Who is labelling? (collector)</label>
      <select id="collector" style="margin-bottom:10px;">
        <option value="Alvaro">Alvaro</option>
        <option value="Hector">Hector</option>
        <option value="Cristian">Cristian</option>
      </select>
      <button id="btnStop">Stop session</button>
      <div id="status">When you stop, you'll review the captured frames.</div>
    </div>

    <div class="panel" id="reviewSummary" style="display:none;">
      <h3>Review progress</h3>
      <div class="stat"><span class="label">Frames</span><span class="value" id="rFrames">0</span></div>
      <div class="stat"><span class="label">Looking</span><span class="value look" id="rLookN">0</span></div>
      <div class="stat"><span class="label">Not looking</span><span class="value" id="rAwayN">0</span></div>
      <div class="stat"><span class="label">Rejected boxes</span><span class="value" id="rRejN">0</span></div>
      <div class="stat"><span class="label">Added (missed)</span><span class="value accent" id="rAddN">0</span></div>
      <button class="rbtn" id="rExportEng" style="margin-top:12px;">Engagement CSV</button>
      <button class="rbtn" id="rExportDet" style="margin-top:8px;">Detection CSV</button>
      <button id="bAnalyze" style="margin-top:12px;">Analyze model</button>
      <button id="rFinish" style="margin-top:8px;">Finish</button>
      <div id="reviewSaveStatus" style="font-size:11px;color:var(--muted);margin-top:8px;"></div>
    </div>
  </div>
</div>

<div id="analysis" style="display:none;">
  <div id="anNav">
    <button class="rbtn" id="anBack" style="width:auto;">← Back to labels</button>
    <div class="an-title">Model check</div>
    <button id="anFinish" style="width:auto; background:var(--look); border-color:var(--look);">Finish</button>
  </div>
  <div class="panel">
    <h3>Model check</h3>
    <div id="anSummary" style="font-size:13px; color:#4a4752; line-height:1.6; margin-bottom:4px;"></div>
    <div class="an-grid">
      <div class="an-col">
        <canvas id="rocCanvas" class="an-plot" width="360" height="360"></canvas>
        <div class="an-cap" id="rocCap">ROC curve</div>
      </div>
      <div class="an-col">
        <canvas id="prCanvas" class="an-plot" width="360" height="360"></canvas>
        <div class="an-cap" id="prCap">Precision–Recall curve</div>
      </div>
      <div class="an-col">
        <label>Decision threshold: <b id="thVal">0.50</b></label>
        <input type="range" id="thSlider" min="0" max="1" step="0.01" value="0.5" style="width:100%; accent-color:var(--accent); margin-top:6px;">
        <div class="cm">
          <div class="hd"></div><div class="hd">pred: looking</div><div class="hd">pred: not</div>
          <div class="hd">actual:<br>looking</div><div class="cell tp">TP<b id="cmTP">0</b></div><div class="cell fn">FN<b id="cmFN">0</b></div>
          <div class="hd">actual:<br>not</div><div class="cell fp">FP<b id="cmFP">0</b></div><div class="cell tn">TN<b id="cmTN">0</b></div>
        </div>
        <div class="metric-row">
          <div class="metric"><div class="v" id="mPrec">—</div><div class="l">Precision</div></div>
          <div class="metric"><div class="v" id="mRec">—</div><div class="l">Recall</div></div>
          <div class="metric"><div class="v" id="mF1">—</div><div class="l">F1</div></div>
        </div>
        <div class="metric-row">
          <div class="metric"><div class="v" id="mAcc">—</div><div class="l">Accuracy</div></div>
          <div class="metric"><div class="v" id="mAuc">—</div><div class="l">ROC AUC</div></div>
          <div class="metric"><div class="v" id="mBrier">—</div><div class="l">Brier</div></div>
        </div>
        <div class="metric-row">
          <div class="metric"><div class="v" id="mMcc">—</div><div class="l" title="Matthews correlation: -1 to 1, robust when the classes are imbalanced">MCC</div></div>
          <div class="metric"><div class="v" id="mBalAcc">—</div><div class="l" title="Average of the looking- and not-looking accuracies — fair when few people look">Balanced acc.</div></div>
          <div class="metric"><div class="v" id="mBase">—</div><div class="l" title="Share of labelled people who actually looked (the class balance)">Base rate</div></div>
        </div>
        <div class="anhint" id="anNote" style="margin-top:12px;"></div>
      </div>
    </div>

    <div id="perfBox" style="margin-top:18px; border-top:1px solid var(--border); padding-top:16px;">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
        <h3 style="margin:0;">Model performance history</h3>
        <button id="bPerfReport" class="rbtn" style="width:auto;">Report</button>
      </div>
      <div class="anhint" style="margin:8px 0 12px;">
        Saved after each session — the model's score against your labels over time. AUC higher is better, Brier lower.
      </div>
      <div class="an-grid">
        <div class="an-col">
          <canvas id="perfCanvas" class="an-plot" width="360" height="220"></canvas>
          <div class="an-cap">
            <span style="color:var(--accent);">●</span> ROC AUC &nbsp;
            <span style="color:var(--reject);">●</span> Brier &nbsp;
            <span style="color:var(--muted);">▲ retrain candidate</span>
          </div>
        </div>
        <div class="an-col">
          <div id="perfTable"></div>
          <div id="perfEmpty" class="anhint" style="margin-top:8px;">No evaluations logged yet — run <b>Analyze</b> above and this fills in.</div>
        </div>
      </div>
    </div>

    <div id="retrainBox" style="margin-top:18px; border-top:1px solid var(--border); padding-top:16px;">
      <h3 style="margin:0 0 6px;">Retrain on your labels</h3>
      <div class="anhint" style="margin-bottom:12px;">
        Train a candidate model on this session's labels. Your live model stays untouched — compare below and promote only if it's better.
      </div>
      <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
        <button id="bRetrain" style="width:auto; background:var(--accent); border-color:var(--accent);">Retrain &amp; compare</button>
        <button id="bPromote" style="width:auto; display:none; background:var(--look); border-color:var(--look);">Use new model</button>
        <button id="bDiscard" class="rbtn" style="width:auto; display:none;">Discard candidate</button>
        <span id="retrainStatus" style="font-size:12px; color:#4a4752;"></span>
      </div>
      <div id="compareWrap" style="display:none; margin-top:16px;">
        <div class="an-grid">
          <div class="an-col">
            <canvas id="rocCmpCanvas" class="an-plot" width="360" height="360"></canvas>
            <div class="an-cap" id="rocCmpCap">ROC — current vs candidate</div>
          </div>
          <div class="an-col">
            <div id="cmpTable"></div>
            <div class="anhint" id="cmpNote" style="margin-top:12px;"></div>
          </div>
        </div>
      </div>
      <details style="margin-top:12px;">
        <summary style="font-size:11.5px; color:var(--muted); cursor:pointer;">Training log</summary>
        <pre id="retrainLog" style="font-size:10.5px; color:#4a4752; white-space:pre-wrap; max-height:200px; overflow:auto; background:var(--card2); border:1px solid var(--border); border-radius:8px; padding:8px; margin-top:6px;"></pre>
      </details>
    </div>

    <div id="yoloBox" style="margin-top:18px; border-top:1px solid var(--border); padding-top:16px;">
      <h3 style="margin:0 0 6px;">Improve person detection (YOLO)</h3>
      <div class="anhint" style="margin-bottom:12px;">
        Different model from the one above. This fine-tunes the <b>person detector</b> on
        this session's boxes: every <b>R · Not a person</b> becomes background (fewer false
        positives) and every box you <b>drew over a missed person</b> becomes a new positive —
        so YOLO learns <i>your</i> camera, lighting and mistakes. Your live detector stays
        untouched; the new weights are saved to a file you switch on when ready.
      </div>
      <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
        <button id="bYolo" style="width:auto; background:var(--accent); border-color:var(--accent);">Fine-tune YOLO</button>
        <label style="font-size:12px; color:#4a4752;">epochs
          <input id="yoloEpochs" type="number" min="1" max="200" value="20"
                 style="width:64px; margin-left:4px;">
        </label>
        <span id="yoloStatus" style="font-size:12px; color:#4a4752;"></span>
      </div>
      <div class="anhint" style="margin-top:8px;">
        Heads-up: training the detector is heavy — it can take several minutes (longer on a
        laptop CPU). You can keep using the page; the status updates here when it finishes.
      </div>
      <details style="margin-top:12px;">
        <summary style="font-size:11.5px; color:var(--muted); cursor:pointer;">Training log</summary>
        <pre id="yoloLog" style="font-size:10.5px; color:#4a4752; white-space:pre-wrap; max-height:200px; overflow:auto; background:var(--card2); border:1px solid var(--border); border-radius:8px; padding:8px; margin-top:6px;"></pre>
      </details>
    </div>
  </div>

  <div class="panel" id="trainDataPanel">
    <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
      <h3 style="margin:0;">Training data — what the model learns from</h3>
      <a id="btnTrainCsv" class="rbtn" href="/api/training_csv" download
         style="width:auto; text-decoration:none; text-align:center; line-height:1.2;">⬇ Download training CSV</a>
    </div>
    <div class="anhint" style="margin:8px 0 12px;">
      Every row you label is appended to <code>data/raw_sessions/*.csv</code>. That folder is
      <b>exactly</b> what <code>build_dataset.py</code> reads to retrain the engagement model —
      nothing else. The download above is those files merged into one CSV so you can verify it
      yourself. The bar chart counts how many labelled rows each of us has contributed.
    </div>
    <div id="trainSummary" style="font-size:13px; color:#4a4752; margin-bottom:10px;"></div>
    <div class="an-grid">
      <div class="an-col">
        <canvas id="trainBarCanvas" class="an-plot"></canvas>
        <div class="an-cap" id="trainBarCap">Labelled rows per collector</div>
      </div>
      <div class="an-col">
        <div id="trainTable"></div>
        <div id="trainEmpty" class="anhint" style="margin-top:8px;">No training rows yet — label a session and they show up here.</div>
      </div>
    </div>
  </div>

  <div class="panel" id="trendsPanel">
    <h3>Trends across sessions</h3>
    <canvas id="trendCanvas" width="300" height="110" style="width:100%; height:auto; background:var(--card2); border:1px solid var(--border); border-radius:10px;"></canvas>
    <div id="trendLegend" style="font-size:10.5px; color:var(--muted); margin:6px 0 10px; text-align:center;">
      <span style="color:var(--accent);">●</span> passersby &nbsp;
      <span style="color:var(--look);">●</span> engagement rate
    </div>
    <div id="trendList" style="font-size:11.5px; color:#4a4752;"></div>
    <div id="trendEmpty" style="font-size:11.5px; color:var(--muted);">No sessions recorded yet.</div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
let stopped = false;
let started = false;     // false until the operator presses ▶ Start (main screen)
let sawRunning = false;  // guards against flipping to review before capture began

function fmtTime(s) {
  s = Math.round(s);
  const m = Math.floor(s / 60), r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

function tierFor(d) {
  if (d == null) return "";
  if (d >= 0.25) return "near <0.5m";
  if (d >= 0.10) return "mid 0.5-1.5m";
  if (d >= 0.04) return "far 1.5-3.5m";
  return "v-far >3.5m";
}

// Which way foot traffic flows past the window, from the pipeline's per-track
// net horizontal displacement. "left"/"right" = the side people CAME FROM, in
// camera space (the operator can name the sides in device.yaml). Also shows what
// share of each side actually stopped to look — often the more useful signal.
function renderFlow(flow) {
  const f = flow || {};
  const L = f.from_left || 0, R = f.from_right || 0;
  const Le = f.from_left_engaged || 0, Re = f.from_right_engaged || 0;
  const total = L + R;
  $("flowLeft").textContent = L;
  $("flowRight").textContent = R;
  const hasData = total >= 3;
  $("flowEmpty").style.display = hasData ? "none" : "";
  $("flowNote").style.display = hasData ? "" : "none";
  $("flowSplit").style.display = hasData ? "flex" : "none";
  if (!hasData) return;
  const lPct = Math.round((L / total) * 100);
  $("flowLeftSeg").style.width = lPct + "%";
  $("flowRightSeg").style.width = (100 - lPct) + "%";
  const lStop = L ? Math.round((Le / L) * 100) : 0;
  const rStop = R ? Math.round((Re / R) * 100) : 0;
  const dom = L === R ? "Balanced both ways"
    : (L > R ? `Most arrive from the left (${lPct}%)` : `Most arrive from the right (${100 - lPct}%)`);
  $("flowNote").innerHTML =
    `${dom}.<br>Stopped to look — from left <b>${lStop}%</b> · from right <b>${rStop}%</b>`;
}

// ================= PHASE 1 — live stats poll =================
async function poll() {
  if (stopped || !started) return;   // idle on the main screen until ▶ Start
  try {
    const res = await fetch("/api/stats", { cache: "no-store" });
    const d = await res.json();
    $("store").textContent = d.store_name || "Demo";
    $("peopleNow").textContent = d.people_now;
    $("elapsed").textContent = fmtTime(d.elapsed_s);
    $("passersby").textContent = d.passersby;
    $("engaged").textContent = d.engaged;
    const rate = d.passersby > 0 ? Math.round((d.engaged / d.passersby) * 100) : 0;
    $("rate").textContent = rate + "%";
    $("attention").textContent = Math.round(d.attention_s) + "s";
    renderFlow(d.flow);
    $("recBadge").classList.toggle("on", !!d.recording);
    // Only flip to review once the session has actually been running — this
    // avoids a race right after Start where the server briefly reports
    // running:false while it (re)builds the pipeline for the new session.
    if (d.running) sawRunning = true;
    if (!d.running && sawRunning && !stopped) enterReview();
  } catch (e) {
    if (sawRunning && !stopped) enterReview();
  }
}

// ▶ Start on the main screen: tell the server the capture interval, then show
// the live layout and begin polling.
async function startSession() {
  const secs = Math.max(1, Math.min(120, parseInt($("startSeconds").value, 10) || 10));
  const name = ($("startCollector").value || "").trim();
  if (name) $("collector").value = name;   // carry the name into the review labeler
  $("btnStart").disabled = true;
  $("startStatus").textContent = "Starting the camera…";
  try { await fetch("/api/start", { method: "POST", body: JSON.stringify({ review_seconds: secs }) }); }
  catch (e) {}
  stopped = false; sawRunning = false; started = true;
  $("status").textContent = `Saving a training frame every ${secs}s. When you stop, you'll review them.`;
  showScreen("layout");
}
$("btnStart").onclick = () => startSession();

$("btnStop").addEventListener("click", async () => {
  $("btnStop").disabled = true;
  $("btnStop").textContent = "Stopping…";
  $("status").textContent = "Stopping — capturing the session frames for review…";
  try { await fetch("/api/stop", { method: "POST" }); } catch (e) {}
  setTimeout(enterReview, 700);
});

// ================= PHASE 2 — post-stop whole-frame review =================
// One frame at a time (next/prev). On each frame you click a detected box to
// select a person and mark Looking / Not looking / Not-a-person, or drag on
// empty space to add a person the detector missed. Every action is persisted
// to the server immediately (engagement CSV + detection CSV).

let frames = [];       // [{i, t, img, w, h, people:[{id,box,yaw,pitch,distance,tier,conf}]}]
let fIdx = 0;
let gaze = {};         // "<i>_<id>" -> "look" | "away"
let det = {};          // "<i>_<id>" -> "real" | "notperson"
let manual = {};       // i -> [{mid, box}]  (missed people the reviewer drew; mid < 0)
let nextMid = -1;      // running negative id for drawn boxes
let sel = null;        // {kind:"person", id} | {kind:"manual", mid}  on the current frame
const canvas = $("rcanvas");
const ctx = canvas.getContext("2d");

const COL = { unset:"#8a5be0", look:"#3fae74", away:"#e15a4d", notperson:"#ef8f3c", manual:"#0bb3a6" };

function curFrame() { return frames[fIdx]; }
function pKey(fr, id) { return fr.i + "_" + id; }
function manualList(fr) { return manual[fr.i] || (manual[fr.i] = []); }

async function enterReview() {
  if (stopped) return;
  stopped = true;
  let data;
  try {
    const res = await fetch("/api/frames", { cache: "no-store" });
    data = await res.json();
  } catch (e) { data = { frames: [] }; }

  $("stream").style.display = "none";
  $("stopPanel").style.display = "none";
  $("reviewSummary").style.display = "block";
  $("reviewBar").style.display = "block";
  canvas.style.display = "block";

  const raw = data.frames || [];
  if (!raw.length) {
    $("rCounter").textContent = "No frames captured";
    $("reviewStatus").textContent =
      "The session was too short to capture any frames (one is kept every 10 s). Nothing to review.";
    return;
  }
  // Kick off image decode for every frame; render as soon as the first is ready.
  let ready = 0;
  frames = raw.map((f) => {
    const img = new Image();
    const obj = { i: f.i, t: f.t, w: f.w, h: f.h, people: f.people || [], img };
    img.onload = () => { if (++ready === 1) { gotoFrame(0); } };
    img.src = "data:image/jpeg;base64," + f.frame;
    return obj;
  });
  updateSummary();
  setTimeout(loadHistory, 500);  // the session that just ended is now in history
}

function gotoFrame(n) {
  if (!frames.length) return;
  fIdx = Math.max(0, Math.min(n, frames.length - 1));
  sel = firstUnlabeled();   // land with the first box needing a verdict already picked
  drawFrame();
}

function drawFrame() {
  const fr = curFrame();
  if (!fr || !fr.img.complete || !fr.img.naturalWidth) return;
  canvas.width = fr.w || fr.img.naturalWidth;
  canvas.height = fr.h || fr.img.naturalHeight;
  ctx.drawImage(fr.img, 0, 0, canvas.width, canvas.height);

  for (const p of fr.people) {
    const k = pKey(fr, p.id);
    // OUTER ring = how the MODEL originally framed this person when it captured
    // the frame: green if it thought "looking", red if "not looking".
    const origLook = p.engaged === true ||
      (typeof p.p_look === "number" && p.p_look >= 0.5);
    let outer = origLook ? COL.look : COL.away;
    if (det[k] === "notperson") outer = COL.notperson; // you rejected the box entirely
    // INNER thick frame = YOUR verdict; white until you label it.
    let inner = null;
    if (det[k] === "notperson") inner = COL.notperson;
    else if (gaze[k] === "look") inner = COL.look;
    else if (gaze[k] === "away") inner = COL.away;
    const isSel = sel && sel.kind === "person" && sel.id === p.id;
    drawBox(p.box, outer, inner, isSel, boxTag(p, k));
  }
  for (const m of manualList(fr)) {
    const mk = pKey(fr, m.mid);
    // Boxes YOU added weren't seen by the model, so there is no "original
    // framing" — the outer ring stays teal (added) and the inner frame follows
    // your verdict.
    let inner = null;
    if (gaze[mk] === "look") inner = COL.look;
    else if (gaze[mk] === "away") inner = COL.away;
    const isSel = sel && sel.kind === "manual" && sel.mid === m.mid;
    drawBox(m.box, COL.manual, inner, isSel, manualTag(mk), true);  // thin: hand-drawn
  }
  $("rCounter").textContent = `Frame ${fIdx + 1} / ${frames.length}  ·  t=${fr.t}s`;
  $("rNext").textContent = fIdx >= frames.length - 1 ? "Analyze →" : "Next →";
  updateSummary();
}

function boxTag(p, k) {
  if (det[k] === "notperson") return "not a person";
  if (gaze[k] === "look") return "looking";
  if (gaze[k] === "away") return "not looking";
  return "#" + p.id + (p.tier ? " · " + p.tier.split(" ")[0] : "");
}
function manualTag(mk) {
  if (gaze[mk] === "look") return "added · looking";
  if (gaze[mk] === "away") return "added · not looking";
  return "added";
}

// Two concentric rings tell two different stories:
//   outerColor — how the MODEL originally framed this person (green/red/orange)
//   innerColor — YOUR own verdict; null means "not labeled yet" → drawn white
function drawBox(box, outerColor, innerColor, selected, tag, thin) {
  const [x1, y1, x2, y2] = box;
  const w = x2 - x1, h = y2 - y1;
  // Selection halo — a gold offset ring so the active box stands out without
  // clashing with the green/red/white rings that carry meaning.
  if (selected) {
    ctx.lineWidth = thin ? 1.25 : 2;
    ctx.strokeStyle = "#ffcf3f";
    ctx.strokeRect(x1 - 3, y1 - 3, w + 6, h + 6);
  }
  // OUTER ring — the model's original framing. `thin` (boxes YOU drew) uses much
  // finer lines so a hand-added box reads as lightweight next to the model's own.
  ctx.lineWidth = thin ? (selected ? 1.25 : 1) : (selected ? 3.5 : 2.5);
  ctx.strokeStyle = outerColor;
  ctx.strokeRect(x1, y1, w, h);
  // INNER frame — your verdict; white until you decide (thin for hand-drawn boxes).
  const pad = thin ? 2.5 : 4;
  ctx.lineWidth = thin ? (selected ? 1.5 : 1.25) : (selected ? 4 : 3);
  ctx.strokeStyle = innerColor || "#ffffff";
  ctx.strokeRect(x1 + pad, y1 + pad, Math.max(1, w - 2 * pad), Math.max(1, h - 2 * pad));
  // Label chip uses the outer (model) color so the tag matches the outer ring.
  const label = " " + tag + " ";
  ctx.font = "600 13px -apple-system, Segoe UI, sans-serif";
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = outerColor;
  ctx.fillRect(x1, Math.max(0, y1 - 20), tw, 18);
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "top";
  ctx.fillText(label, x1, Math.max(1, y1 - 19));
}

// ---- pointer: click to select a box, drag on empty space to add a person ----
let dragStart = null;
canvas.addEventListener("mousedown", (e) => { dragStart = toCanvas(e); });
canvas.addEventListener("mousemove", (e) => {
  if (!dragStart) return;
  const p = toCanvas(e);
  if (Math.hypot(p.x - dragStart.x, p.y - dragStart.y) < 5) return;
  drawFrame();
  ctx.setLineDash([6, 4]); ctx.lineWidth = 2; ctx.strokeStyle = COL.manual;
  ctx.strokeRect(dragStart.x, dragStart.y, p.x - dragStart.x, p.y - dragStart.y);
  ctx.setLineDash([]);
});
canvas.addEventListener("mouseup", (e) => {
  if (!dragStart) return;
  const p = toCanvas(e);
  const moved = Math.hypot(p.x - dragStart.x, p.y - dragStart.y);
  if (moved >= 8) {
    const box = [Math.round(Math.min(dragStart.x, p.x)), Math.round(Math.min(dragStart.y, p.y)),
                 Math.round(Math.max(dragStart.x, p.x)), Math.round(Math.max(dragStart.y, p.y))];
    addManual(box);
  } else {
    selectAt(p.x, p.y);
  }
  dragStart = null;
});

function toCanvas(e) {
  const r = canvas.getBoundingClientRect();
  const cw = canvas.width, ch = canvas.height;
  // Canvas is shown with object-fit:cover (fills the box, centre-cropped). Invert
  // that transform so a click maps back to the right pixel in canvas space.
  const scale = Math.max(r.width / cw, r.height / ch);
  const offX = (r.width - cw * scale) / 2, offY = (r.height - ch * scale) / 2;
  return { x: (e.clientX - r.left - offX) / scale,
           y: (e.clientY - r.top - offY) / scale };
}

function inBox(x, y, b) { return x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]; }

function selectAt(x, y) {
  const fr = curFrame();
  for (const m of manualList(fr)) {
    if (inBox(x, y, m.box)) { sel = { kind: "manual", mid: m.mid }; drawFrame(); return; }
  }
  // Smallest containing detected box wins, so overlapping far/near boxes are both reachable.
  let best = null, bestArea = Infinity;
  for (const p of fr.people) {
    if (!inBox(x, y, p.box)) continue;
    const a = (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]);
    if (a < bestArea) { bestArea = a; best = p; }
  }
  sel = best ? { kind: "person", id: best.id } : null;
  drawFrame();
}

function selectedPerson() {
  if (!sel || sel.kind !== "person") return null;
  return curFrame().people.find((p) => p.id === sel.id) || null;
}
function selectedManual() {
  if (!sel || sel.kind !== "manual") return null;
  return manualList(curFrame()).find((m) => m.mid === sel.mid) || null;
}

// Fluid labeling: the first box on a frame that still needs a verdict. Used to
// auto-select as you land on a frame and to hop to the next box after each
// keypress, so a whole frame is labeled with just L / A / R — no clicking.
function firstUnlabeled() {
  const fr = curFrame();
  if (!fr) return null;
  for (const p of fr.people) {
    const k = pKey(fr, p.id);
    if (!gaze[k] && det[k] !== "notperson") return { kind: "person", id: p.id };
  }
  for (const m of manualList(fr)) {
    if (!gaze[pKey(fr, m.mid)]) return { kind: "manual", mid: m.mid };
  }
  return null;
}
// After any verdict: move the highlight to the next box that needs one (or, if
// the frame is fully labeled, let maybeAdvance() carry us to the next frame).
function afterLabel() { sel = firstUnlabeled(); drawFrame(); maybeAdvance(); }

// ---- labeling actions on the selected box ----
// Works on BOTH a detected person and a box the reviewer drew. Detected boxes
// carry head pose, so they feed the engagement model; drawn boxes have no pose,
// so marking them looking/not-looking is recorded as a detection + counted, but
// can't enter the engagement curves.
function markGaze(label) {
  const fr = curFrame();
  const p = selectedPerson();
  if (p) {
    const k = pKey(fr, p.id);
    gaze[k] = label; det[k] = "real";
    saveGaze(fr, p, label);
    afterLabel();
    return;
  }
  const m = selectedManual();
  if (m) {
    gaze[pKey(fr, m.mid)] = label;
    saveManualGaze(fr, m, label);
    afterLabel();
    return;
  }
  flash("Nothing selected — press L / A / R once a box is highlighted.");
}

// Once every detected box in this frame has a verdict (looking / not-looking /
// not-a-person), jump to the next frame automatically, so labeling flows one
// image straight into the next. A brief pause lets the box's colour register.
function maybeAdvance() {
  const fr = curFrame();
  if (!fr) return;
  const ml = manualList(fr);
  if (!fr.people.length && !ml.length) return;
  const peopleDone = fr.people.every((p) => {
    const k = pKey(fr, p.id);
    return gaze[k] || det[k] === "notperson";
  });
  const manualDone = ml.every((m) => gaze[pKey(fr, m.mid)]);
  if (peopleDone && manualDone) {
    const from = fIdx;
    if (fIdx < frames.length - 1) {
      setTimeout(() => { if (fIdx === from) gotoFrame(from + 1); }, 200);
    } else {
      // Last frame just got its final verdict: flow straight into the analysis
      // page instead of stalling on the last image.
      setTimeout(() => {
        if (fIdx === from && stopped && $("analysis").style.display === "none") goAnalysis();
      }, 350);
    }
  }
}

function markReject() {
  const fr = curFrame();
  if (sel && sel.kind === "manual") { deleteManual(); return; }
  const p = selectedPerson();
  if (!p) {
    // R with nothing to reject: if the model detected no one on this frame (and you
    // haven't drawn a box), treat R as "nothing here → skip to the next photo", so
    // empty frames don't stall the flow — no mouse, no Next button needed.
    if (!fr.people.length && !manualList(fr).length) {
      flash("Nothing here \u2713 — next photo.");
      nextFrameOrAnalyze();
      return;
    }
    flash("Click a box first.");
    return;
  }
  const k = pKey(fr, p.id);
  det[k] = "notperson"; delete gaze[k];
  saveReject(fr, p);
  flash("Not a person \u2713 — got it, moving on.");
  afterLabel();
}

function addManual(box) {
  const fr = curFrame();
  const mid = nextMid--;
  manualList(fr).push({ mid, box });
  sel = { kind: "manual", mid };
  saveManual(fr, mid, box);
  drawFrame();
}

function deleteManual() {
  if (!sel || sel.kind !== "manual") { flash("Select a drawn box to delete it."); return; }
  const fr = curFrame();
  const arr = manualList(fr);
  const i = arr.findIndex((m) => m.mid === sel.mid);
  if (i !== -1) { const [rm] = arr.splice(i, 1); unsaveManual(fr, rm.mid); }
  sel = null;
  // Route through afterLabel so removing the last pending box still carries us to
  // the next frame — otherwise R / Delete on a drawn box would stall the flow.
  afterLabel();
}

// ---- crop the selected box out of the frame for the saved image ----
function cropB64(fr, box) {
  const [x1, y1, x2, y2] = box;
  const w = Math.max(1, x2 - x1), h = Math.max(1, y2 - y1);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").drawImage(fr.img, x1, y1, w, h, 0, 0, w, h);
  return c.toDataURL("image/jpeg", 0.82).split(",")[1];
}

async function post(url, body) {
  try {
    const res = await fetch(url, { method: "POST", body: JSON.stringify(body) });
    return await res.json();
  } catch (e) { return { ok: false }; }
}

// Looking/Not-looking → engagement row (if pose known) + detection true positive.
async function saveGaze(fr, p, label) {
  const key = pKey(fr, p.id);
  const collector = $("collector").value || "unknown";
  const crop = cropB64(fr, p.box);
  post("/api/detect_label", { key, frameIdx: fr.i, id: p.id, box: p.box, conf: p.conf, verdict: 1, collector, crop });
  if (p.yaw == null) { flash("Saved (no head pose — detection only, not engagement)."); updateSummary(); return; }
  const d = await post("/api/label", {
    key, yaw: p.yaw, pitch: p.pitch, distance: p.distance, label: label === "look" ? 1 : 0,
    tier: p.tier || tierFor(p.distance), collector, crop,
  });
  if (d && d.ok) flash(`Saved → ${d.path}`);
  updateSummary();
}

// Not-a-person → detection false positive; drop any engagement row.
async function saveReject(fr, p) {
  const key = pKey(fr, p.id);
  const collector = $("collector").value || "unknown";
  post("/api/unlabel", { key });
  const d = await post("/api/detect_label", {
    key, frameIdx: fr.i, id: p.id, box: p.box, conf: p.conf, verdict: 0, collector, crop: cropB64(fr, p.box),
  });
  if (d && d.ok) flash(`Rejected → ${d.path}`);
  updateSummary();
}

// Drawn missed person → detection true positive under a negative track id.
async function saveManual(fr, mid, box) {
  const key = pKey(fr, mid);
  const collector = $("collector").value || "unknown";
  const d = await post("/api/detect_label", {
    key, frameIdx: fr.i, id: mid, box, conf: null, verdict: 1, collector, crop: cropB64(fr, box),
  });
  if (d && d.ok) flash(`Added missed person → ${d.path}`);
  updateSummary();
}
function unsaveManual(fr, mid) { post("/api/detect_unlabel", { key: pKey(fr, mid) }); updateSummary(); }

// Looking/Not-looking on a DRAWN box. No head pose exists for a hand-drawn box,
// so there's no engagement row — we keep it as a confirmed detection (verdict 1)
// and record the verdict locally so it colours + counts in the summary.
async function saveManualGaze(fr, m, label) {
  const collector = $("collector").value || "unknown";
  await post("/api/detect_label", {
    key: pKey(fr, m.mid), frameIdx: fr.i, id: m.mid, box: m.box,
    conf: null, verdict: 1, collector, crop: cropB64(fr, m.box),
  });
  flash(label === "look"
    ? "Drawn box → looking (detection only, no head pose)."
    : "Drawn box → not looking (detection only, no head pose).");
  updateSummary();
}

function flash(msg) { $("reviewSaveStatus").textContent = msg; }

function updateSummary() {
  let look = 0, away = 0, rej = 0;
  for (const k in gaze) { if (gaze[k] === "look") look++; else if (gaze[k] === "away") away++; }
  for (const k in det) { if (det[k] === "notperson") rej++; }
  let added = 0; for (const i in manual) added += manual[i].length;
  $("rFrames").textContent = frames.length;
  $("rLookN").textContent = look;
  $("rAwayN").textContent = away;
  $("rRejN").textContent = rej;
  $("rAddN").textContent = added;
  $("reviewStatus").textContent = frames.length
    ? `Frame ${fIdx + 1} of ${frames.length}. Tap L / A / R — next box auto-selected. Drag empty space to add a missed person.`
    : "";
}

// ---- screen switching: start -> layout (live/label) -> analysis ----
// The four phases (start -> live -> label -> analysis) each own the whole
// viewport. start -> live happens in startSession(); live -> label in
// enterReview(); label -> analysis when the reviewer finishes the last frame.
function showScreen(name) {
  const onStart = name === "start";
  const onAnalysis = name === "analysis";
  const onLayout = !onStart && !onAnalysis;
  $("startScreen").style.display = onStart ? "flex" : "none";
  $("layout").style.display = onLayout ? "flex" : "none";
  $("analysis").style.display = onAnalysis ? "block" : "none";
  if (onLayout) drawFrame();
}
function goAnalysis() { runAnalysis(); }

// Advancing past the last frame moves the whole flow into the analysis screen.
function nextFrameOrAnalyze() {
  if (fIdx >= frames.length - 1) { goAnalysis(); return; }
  gotoFrame(fIdx + 1);
}

// ---- navigation + buttons + keys ----
$("rPrev").onclick = () => gotoFrame(fIdx - 1);
$("rNext").onclick = () => nextFrameOrAnalyze();
$("rLook").onclick = () => markGaze("look");
$("rAway").onclick = () => markGaze("away");
$("rReject").onclick = () => markReject();
$("rDelete").onclick = () => deleteManual();

window.addEventListener("keydown", (e) => {
  if (!stopped) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  const k = e.key.toLowerCase();
  if (k === "l") markGaze("look");
  else if (k === "a") markGaze("away");
  else if (k === "r") markReject();
  else if (e.key === "ArrowRight") nextFrameOrAnalyze();
  else if (e.key === "ArrowLeft") gotoFrame(fIdx - 1);
  else if (e.key === "Backspace" || e.key === "Delete") { deleteManual(); e.preventDefault(); }
});
window.addEventListener("resize", () => { if (stopped) drawFrame(); redrawAnalysisCharts(); });

// ---- CSV exports (server already persists live; these are a local copy) ----
$("rExportEng").onclick = () => {
  const cols = ["yaw", "pitch", "distance", "label", "distance_tier", "glasses", "headwear",
                "subject", "collector", "session", "captured_at"];
  const rows = [cols.join(",")];
  const collector = $("collector").value || "unknown";
  const session = "live_" + Date.now();
  const at = new Date().toISOString();
  for (const fr of frames) {
    for (const p of fr.people) {
      const lab = gaze[pKey(fr, p.id)];
      if (!lab || p.yaw == null) continue;
      rows.push([p.yaw, p.pitch, p.distance, lab === "look" ? 1 : 0, p.tier || tierFor(p.distance),
                 "unknown", "unknown", "unknown", collector, session, at].join(","));
    }
  }
  download(rows.join("\\n"), "live_session_reviewed.csv");
};

$("rExportDet").onclick = () => {
  const cols = ["frame_idx", "track_id", "x1", "y1", "x2", "y2", "conf", "verdict",
                "collector", "session", "captured_at"];
  const rows = [cols.join(",")];
  const collector = $("collector").value || "unknown";
  const session = "live_" + Date.now();
  const at = new Date().toISOString();
  for (const fr of frames) {
    for (const p of fr.people) {
      const k = pKey(fr, p.id);
      let verdict = null;
      if (det[k] === "notperson") verdict = 0;
      else if (det[k] === "real" || gaze[k]) verdict = 1;
      if (verdict === null) continue;
      rows.push([fr.i, p.id, p.box[0], p.box[1], p.box[2], p.box[3], p.conf ?? "", verdict,
                 collector, session, at].join(","));
    }
    for (const m of manualList(fr)) {
      rows.push([fr.i, m.mid, m.box[0], m.box[1], m.box[2], m.box[3], "", 1,
                 collector, session, at].join(","));
    }
  }
  download(rows.join("\\n"), "live_session_detections.csv");
};

function download(text, name) {
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
}

// Save/Finish: labels are already persisted live, so this just tells the
// server to loop back to a fresh session and returns the browser to the main
// screen (instead of quitting) — the app is reusable end to end.
async function finishSession(btn) {
  const label = btn ? btn.textContent : "";
  if (btn) { btn.disabled = true; btn.textContent = "Saving…"; }
  try { await fetch("/api/restart", { method: "POST" }); } catch (e) {}
  resetToMain();
  if (btn) { btn.disabled = false; btn.textContent = label; }
}

// Wipe the front-end session so a new run starts clean on the main screen.
function resetToMain() {
  stopped = false; started = false; sawRunning = false;
  frames = []; fIdx = 0; gaze = {}; det = {}; manual = {}; nextMid = -1; sel = null;
  // restore the live layout's default panels for next time
  $("stream").style.display = "";
  $("stopPanel").style.display = "";
  $("reviewSummary").style.display = "none";
  $("reviewBar").style.display = "none";
  canvas.style.display = "none";
  $("btnStop").disabled = false;
  $("btnStop").textContent = "Stop session";
  $("reviewSaveStatus").textContent = "";
  $("btnStart").disabled = false;
  $("startStatus").textContent = "Saved. Ready for another session.";
  showScreen("start");
  loadHistory();
}
$("rFinish").onclick = () => finishSession($("rFinish"));
$("anFinish").onclick = () => finishSession($("anFinish"));
$("anBack").onclick = () => showScreen("layout");

// ================= PHASE 3 — model check (analysis) =================
// Eval set = every DETECTED person that both carries a model prediction
// (p_look, the live pipeline's engage_prob) AND that you labeled looking /
// not looking. That pairing (predicted probability vs your ground truth) is
// what every metric below reads.
let anPairs = [];
function evalPairs() {
  const pairs = [];
  for (const f of frames) for (const p of f.people) {
    if (p.p_look === undefined || p.p_look === null) continue;
    const g = gaze[pKey(f, p.id)];
    if (g !== "look" && g !== "away") continue;
    pairs.push({ s: p.p_look, y: g === "look" ? 1 : 0 });
  }
  return pairs;
}

// ROC by sweeping every score as a threshold; AUC by trapezoid over (FPR, TPR).
function rocData(pairs) {
  const P = pairs.reduce((a, p) => a + p.y, 0), N = pairs.length - P;
  const pts = [[0, 0]];
  if (!P || !N) return { pts, auc: NaN };
  const s = [...pairs].sort((a, b) => b.s - a.s);
  let tp = 0, fp = 0;
  for (const p of s) { if (p.y) tp++; else fp++; pts.push([fp / N, tp / P]); }
  let auc = 0;
  for (let i = 1; i < pts.length; i++) auc += (pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1]) / 2;
  return { pts, auc };
}

// Precision–Recall curve + average precision (area under it).
function prData(pairs) {
  const P = pairs.reduce((a, p) => a + p.y, 0);
  const pts = [];
  if (!P) return { pts, ap: NaN };
  const s = [...pairs].sort((a, b) => b.s - a.s);
  let tp = 0, fp = 0, ap = 0, prevRec = 0;
  for (const p of s) {
    if (p.y) tp++; else fp++;
    const prec = tp / (tp + fp), rec = tp / P;
    ap += (rec - prevRec) * prec; prevRec = rec;
    pts.push([rec, prec]);
  }
  return { pts, ap };
}

function confAt(pairs, t) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (const { s, y } of pairs) {
    const pred = s >= t ? 1 : 0;
    if (pred && y) tp++; else if (pred && !y) fp++; else if (!pred && y) fn++; else tn++;
  }
  return { tp, fp, fn, tn };
}

// ---- crisp, interactive charts ------------------------------------------
// Every <canvas> here is drawn at devicePixelRatio so lines stay sharp on
// Retina screens (the old code drew a 360px bitmap and let CSS stretch it —
// that's the "blurry image" the operator saw). hidpi() sizes the backing store
// to cssWidth*dpr, keeps CSS width responsive, and scales the context so all the
// drawing code can keep thinking in plain CSS pixels.
const DPR = () => window.devicePixelRatio || 1;
function hidpi(cv, aspect) {
  const dpr = DPR();
  const cssW = cv.clientWidth || parseInt(cv.getAttribute("width")) || 360;
  const cssH = Math.round(cssW * aspect);
  cv.style.height = cssH + "px";
  cv.width = Math.round(cssW * dpr);
  cv.height = Math.round(cssH * dpr);
  const c = cv.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, cssW, cssH);
  return { c, W: cssW, H: cssH };
}

// One floating tooltip shared by every chart's hover handler.
function chartTip() {
  let t = document.getElementById("chartTip");
  if (!t) {
    t = document.createElement("div"); t.id = "chartTip";
    t.style.cssText = "position:fixed; z-index:60; pointer-events:none; display:none;" +
      "background:rgba(28,27,34,.94); color:#fff; font-size:11px; padding:5px 8px;" +
      "border-radius:7px; white-space:nowrap; box-shadow:0 6px 18px rgba(0,0,0,.28);";
    document.body.appendChild(t);
  }
  return t;
}
function showTip(px, py, html) {
  const t = chartTip(); t.innerHTML = html; t.style.display = "block";
  t.style.left = (px + 14) + "px"; t.style.top = (py + 14) + "px";
}
function hideTip() { chartTip().style.display = "none"; }

// Redraw whatever a canvas last drew (each chart stashes a __redraw closure).
function redrawChart(cv) { if (cv && cv.__redraw) cv.__redraw(); }
function redrawAnalysisCharts() {
  if ($("analysis").style.display === "none") return;
  ["rocCanvas", "prCanvas", "perfCanvas", "trendCanvas", "rocCmpCanvas", "trainBarCanvas"]
    .forEach((id) => redrawChart($(id)));
}

function plotCurve(cv, pts, diag, pts2) {
  cv.__redraw = () => plotCurveCore(cv, pts, diag, pts2);
  plotCurveCore(cv, pts, diag, pts2);
  bindCurveHover(cv);
}
function plotCurveCore(cv, pts, diag, pts2) {
  const { c, W, H } = hidpi(cv, 1), pad = 34;
  const X = (x) => pad + x * (W - 2 * pad), Y = (y) => (H - pad) - y * (H - 2 * pad);
  cv.__geo = { pad, W, H, X, Y };
  // gridlines + 0..1 tick labels on both axes
  c.font = "10px system-ui, -apple-system, sans-serif";
  for (let g = 0; g <= 1.0001; g += 0.25) {
    c.strokeStyle = "#efeae0"; c.lineWidth = 1;
    c.beginPath(); c.moveTo(X(g), Y(0)); c.lineTo(X(g), Y(1)); c.stroke();
    c.beginPath(); c.moveTo(X(0), Y(g)); c.lineTo(X(1), Y(g)); c.stroke();
    c.fillStyle = "#9a9488";
    c.textAlign = "center"; c.textBaseline = "top"; c.fillText(g.toFixed(2), X(g), H - pad + 4);
    c.textAlign = "right"; c.textBaseline = "middle"; c.fillText(g.toFixed(2), pad - 5, Y(g));
  }
  c.strokeStyle = "#c9c2b4"; c.lineWidth = 1.2; c.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad);
  if (diag) {  // chance line for ROC
    c.strokeStyle = "#c9c2b4"; c.setLineDash([4, 4]);
    c.beginPath(); c.moveTo(X(0), Y(0)); c.lineTo(X(1), Y(1)); c.stroke(); c.setLineDash([]);
  }
  if (pts && pts.length) {   // soft fill under the primary curve — reads better
    c.fillStyle = "rgba(74,42,134,.08)"; c.beginPath(); c.moveTo(X(pts[0][0]), Y(0));
    pts.forEach(([x, y]) => c.lineTo(X(x), Y(y)));
    c.lineTo(X(pts[pts.length - 1][0]), Y(0)); c.closePath(); c.fill();
  }
  const line = (p, color, w) => {
    if (!p || !p.length) return;
    c.strokeStyle = color; c.lineWidth = w; c.lineJoin = "round"; c.beginPath();
    p.forEach(([x, y], i) => { const px = X(x), py = Y(y); i ? c.lineTo(px, py) : c.moveTo(px, py); });
    c.stroke();
  };
  line(pts, "#4a2a86", 2.2);     // current / primary model
  line(pts2, "#0bb3a6", 2.2);    // candidate (only when comparing)
}
function bindCurveHover(cv) {
  if (cv.__hoverBound) return; cv.__hoverBound = true;
  cv.style.cursor = "crosshair";
  cv.addEventListener("mousemove", (e) => {
    const g = cv.__geo; if (!g) return;
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const { pad, W, H } = g;
    if (mx < pad || mx > W - pad || my < pad || my > H - pad) { hideTip(); redrawChart(cv); return; }
    redrawChart(cv);
    const c = cv.getContext("2d");
    c.strokeStyle = "rgba(28,27,34,.22)"; c.setLineDash([3, 3]); c.lineWidth = 1;
    c.beginPath(); c.moveTo(mx, pad); c.lineTo(mx, H - pad); c.moveTo(pad, my); c.lineTo(W - pad, my); c.stroke(); c.setLineDash([]);
    const x = (mx - pad) / (W - 2 * pad), y = (H - pad - my) / (H - 2 * pad);
    showTip(e.clientX, e.clientY, `x ${x.toFixed(2)} &middot; y ${y.toFixed(2)}`);
  });
  cv.addEventListener("mouseleave", () => { hideTip(); redrawChart(cv); });
}

function renderConfusion() {
  const t = Number($("thSlider").value);
  $("thVal").textContent = t.toFixed(2);
  const { tp, fp, fn, tn } = confAt(anPairs, t);
  $("cmTP").textContent = tp; $("cmFP").textContent = fp; $("cmFN").textContent = fn; $("cmTN").textContent = tn;
  const prec = (tp + fp) ? tp / (tp + fp) : NaN;
  const rec = (tp + fn) ? tp / (tp + fn) : NaN;
  const f1 = (prec + rec) ? 2 * prec * rec / (prec + rec) : NaN;
  const acc = anPairs.length ? (tp + tn) / anPairs.length : NaN;
  const fmt = (v) => isNaN(v) ? "—" : v.toFixed(2);
  $("mPrec").textContent = fmt(prec); $("mRec").textContent = fmt(rec);
  $("mF1").textContent = fmt(f1); $("mAcc").textContent = fmt(acc);
  // Extra reads that hold up when the classes are imbalanced (few people look):
  //  · MCC — a single -1..1 score that only goes high when BOTH classes are right.
  //  · Balanced accuracy — the mean of the two per-class hit rates.
  //  · Base rate — what fraction of the labelled people actually looked.
  const tnr = (tn + fp) ? tn / (tn + fp) : NaN;
  const balAcc = (isNaN(rec) || isNaN(tnr)) ? NaN : (rec + tnr) / 2;
  const mccDen = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  const mcc = mccDen ? (tp * tn - fp * fn) / mccDen : NaN;
  const base = anPairs.length ? (tp + fn) / anPairs.length : NaN;
  $("mMcc").textContent = fmt(mcc); $("mBalAcc").textContent = fmt(balAcc);
  $("mBase").textContent = isNaN(base) ? "—" : Math.round(base * 100) + "%";
}

function runAnalysis() {
  anPairs = evalPairs();
  showScreen("analysis");
  const roc = rocData(anPairs), pr = prData(anPairs);
  const pos = anPairs.reduce((a, p) => a + p.y, 0), neg = anPairs.length - pos;
  const brier = anPairs.length ? anPairs.reduce((a, p) => a + (p.s - p.y) ** 2, 0) / anPairs.length : NaN;
  const anyPred = frames.some((f) => f.people.some((p) => p.p_look !== undefined && p.p_look !== null));
  $("anSummary").innerHTML = `<b>${anPairs.length}</b> labeled detections carry a model prediction ` +
    `(<b>${pos}</b> looking / <b>${neg}</b> not looking).`;
  plotCurve($("rocCanvas"), roc.pts, true);
  plotCurve($("prCanvas"), pr.pts, false);
  $("rocCap").textContent = `ROC — AUC ${isNaN(roc.auc) ? "n/a" : roc.auc.toFixed(3)}  ·  x=FPR, y=TPR`;
  $("prCap").textContent = `Precision–Recall — AP ${isNaN(pr.ap) ? "n/a" : pr.ap.toFixed(3)}  ·  x=recall, y=precision`;
  $("mAuc").textContent = isNaN(roc.auc) ? "—" : roc.auc.toFixed(2);
  $("mBrier").textContent = isNaN(brier) ? "—" : brier.toFixed(3);
  let note;
  if (!anyPred) note = "No model predictions in these frames — nothing to score.";
  else if (anPairs.length < 20) note = "Very few samples — label more for a trustworthy read.";
  else if (!pos || !neg) note = "Need both looking and not-looking labels for ROC / AUC.";
  else note = "AUC ≈ 0.5 is guessing, 0.9+ is strong. Brier: lower is better. Slide the threshold to trade precision against recall.";
  $("anNote").innerHTML = note;
  renderConfusion();
  // Auto-save this session's live-model performance against your labels, so the
  // history below builds itself with zero effort. Only log a meaningful read:
  // needs a prediction AND both classes present.
  if (anyPred && pos && neg) logPerf(snapFrom(anPairs, "session_eval"));
  loadPerf();
  loadTrainingData();   // per-person bar chart + downloadable proof of the training set
  loadHistory();        // redraw the trend chart now that #analysis is visible (crisp sizing)
}
$("bAnalyze").onclick = runAnalysis;
$("thSlider").addEventListener("input", renderConfusion);

// ---- retrain-and-compare: close the loop from labels to a better model ----
// Every labeled detection that has a real head-pose triple becomes a training-
// era example we can re-score with BOTH models. y = your ground truth.
function labeledItems() {
  const out = [];
  for (const f of frames) for (const p of f.people) {
    const g = gaze[pKey(f, p.id)];
    if (g !== "look" && g !== "away") continue;
    if (typeof p.yaw !== "number" || typeof p.pitch !== "number" || typeof p.distance !== "number") continue;
    out.push({ yaw: p.yaw, pitch: p.pitch, distance: p.distance, y: g === "look" ? 1 : 0 });
  }
  return out;
}

function metricsFor(pairs) {
  const roc = rocData(pairs), pr = prData(pairs);
  const brier = pairs.length ? pairs.reduce((a, p) => a + (p.s - p.y) ** 2, 0) / pairs.length : NaN;
  const { tp, fp, fn } = confAt(pairs, 0.5);
  const prec = (tp + fp) ? tp / (tp + fp) : NaN, rec = (tp + fn) ? tp / (tp + fn) : NaN;
  const f1 = (prec + rec) ? 2 * prec * rec / (prec + rec) : NaN;
  return { auc: roc.auc, ap: pr.ap, brier, f1, roc: roc.pts };
}

function deltaCell(oldV, newV, higherBetter) {
  if (isNaN(oldV) || isNaN(newV)) return "";
  const d = newV - oldV, better = higherBetter ? d > 0 : d < 0;
  if (Math.abs(d) < 1e-4) return `<span class="delta">±0</span>`;
  const sign = d > 0 ? "+" : "";
  return `<span class="delta ${better ? "up" : "down"}">${sign}${d.toFixed(3)}</span>`;
}

let retrainTimer = null;
async function runRetrain() {
  $("bRetrain").disabled = true;
  $("compareWrap").style.display = "none";
  $("bPromote").style.display = "none";
  $("bDiscard").style.display = "none";
  $("retrainStatus").textContent = "Starting…";
  try {
    const r = await (await fetch("/api/retrain", { method: "POST" })).json();
    if (!r.ok) { $("retrainStatus").textContent = r.msg || "Could not start."; $("bRetrain").disabled = false; return; }
  } catch (e) { $("retrainStatus").textContent = "Could not reach the server."; $("bRetrain").disabled = false; return; }
  pollRetrain();
}

function pollRetrain() {
  clearTimeout(retrainTimer);
  retrainTimer = setTimeout(async () => {
    let s;
    try { s = await (await fetch("/api/retrain_status", { cache: "no-store" })).json(); }
    catch (e) { $("retrainStatus").textContent = "Lost contact with the trainer."; $("bRetrain").disabled = false; return; }
    $("retrainStatus").textContent = s.msg || "";
    if (s.log) $("retrainLog").textContent = s.log;
    if (s.running) { pollRetrain(); return; }
    $("bRetrain").disabled = false;
    if (s.done && s.ok) await onRetrainDone();
  }, 1200);
}

async function onRetrainDone() {
  const items = labeledItems();
  if (!items.length) { $("retrainStatus").textContent = "Trained, but you have no pose-carrying labels to compare on."; return; }
  let res;
  try {
    res = await (await fetch("/api/rescore", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items: items.map(({ yaw, pitch, distance }) => ({ yaw, pitch, distance })) }),
    })).json();
  } catch (e) { $("retrainStatus").textContent = "Could not score the models."; return; }
  if (!res.ok || !res.has_new) { $("retrainStatus").textContent = "The candidate model could not be scored."; return; }
  renderCompare(items, res.old, res.new);
}

function renderCompare(items, oldS, newS) {
  const oldPairs = [], newPairs = [];
  items.forEach((it, i) => {
    if (typeof oldS[i] === "number") oldPairs.push({ s: oldS[i], y: it.y });
    if (typeof newS[i] === "number") newPairs.push({ s: newS[i], y: it.y });
  });
  const mo = metricsFor(oldPairs), mn = metricsFor(newPairs);
  plotCurve($("rocCmpCanvas"), mo.roc, true, mn.roc);
  const f = (v) => isNaN(v) ? "—" : v.toFixed(3);
  $("cmpTable").innerHTML =
    `<table class="cmp"><thead><tr><th>Metric</th><th>Current</th><th>Candidate</th><th>Δ</th></tr></thead><tbody>` +
    `<tr><td>ROC AUC</td><td class="now">${f(mo.auc)}</td><td class="cand">${f(mn.auc)}</td><td>${deltaCell(mo.auc, mn.auc, true)}</td></tr>` +
    `<tr><td>Avg precision</td><td class="now">${f(mo.ap)}</td><td class="cand">${f(mn.ap)}</td><td>${deltaCell(mo.ap, mn.ap, true)}</td></tr>` +
    `<tr><td>F1 @ 0.50</td><td class="now">${f(mo.f1)}</td><td class="cand">${f(mn.f1)}</td><td>${deltaCell(mo.f1, mn.f1, true)}</td></tr>` +
    `<tr><td>Brier</td><td class="now">${f(mo.brier)}</td><td class="cand">${f(mn.brier)}</td><td>${deltaCell(mo.brier, mn.brier, false)}</td></tr>` +
    `</tbody></table>`;
  const better = (!isNaN(mn.auc) && !isNaN(mo.auc) && mn.auc >= mo.auc);
  $("cmpNote").innerHTML = better
    ? "Candidate matches or beats the current model on your labels (purple = current, teal = candidate). Promote to use it next launch."
    : "Candidate isn't clearly better — collect more labels before promoting.";
  $("compareWrap").style.display = "block";
  $("bPromote").style.display = "inline-block";
  $("bDiscard").style.display = "inline-block";
  // Record the candidate's performance on your labels as its own history point
  // (marked as a retrain event), so the improvement chart shows the jump.
  if (newPairs.length) { logPerf(snapFrom(newPairs, "retrain")).then(loadPerf); }
}

$("bRetrain").onclick = runRetrain;

// ---- fine-tune the YOLO person detector on this session's detection verdicts ----
let yoloTimer = null;
async function runYolo() {
  const epochs = Math.max(1, Math.min(200, parseInt($("yoloEpochs").value, 10) || 20));
  $("bYolo").disabled = true;
  $("yoloStatus").textContent = "Starting…";
  try {
    const r = await (await fetch("/api/finetune_yolo", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs }),
    })).json();
    if (!r.ok) { $("yoloStatus").textContent = r.msg || "Could not start."; $("bYolo").disabled = false; return; }
  } catch (e) { $("yoloStatus").textContent = "Could not reach the server."; $("bYolo").disabled = false; return; }
  pollYolo();
}

function pollYolo() {
  clearTimeout(yoloTimer);
  yoloTimer = setTimeout(async () => {
    let s;
    try { s = await (await fetch("/api/finetune_yolo_status", { cache: "no-store" })).json(); }
    catch (e) { $("yoloStatus").textContent = "Lost contact with the trainer."; $("bYolo").disabled = false; return; }
    $("yoloStatus").textContent = s.msg || "";
    if (s.log) $("yoloLog").textContent = s.log;
    if (s.running) { pollYolo(); return; }
    $("bYolo").disabled = false;
  }, 1500);
}
$("bYolo").onclick = runYolo;

$("bPromote").onclick = async () => {
  $("bPromote").disabled = true;
  try {
    const r = await (await fetch("/api/promote", { method: "POST" })).json();
    if (r.ok) {
      $("retrainStatus").textContent = "New model promoted — used next launch.";
      const items = labeledItems();
      const pairs = [];
      if (items.length) {
        try {
          const res = await (await fetch("/api/rescore", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ items: items.map(({ yaw, pitch, distance }) => ({ yaw, pitch, distance })) }),
          })).json();
          items.forEach((it, i) => { if (typeof res.new[i] === "number") pairs.push({ s: res.new[i], y: it.y }); });
        } catch (e) { /* fall through — promotion still succeeded */ }
      }
      if (pairs.length) { await logPerf(snapFrom(pairs, "promote")); await loadPerf(); }
    } else {
      $("retrainStatus").textContent = r.msg || "Could not promote.";
    }
  } catch (e) { $("retrainStatus").textContent = "Could not promote."; }
  $("bPromote").disabled = false;
};
$("bDiscard").onclick = () => {
  $("compareWrap").style.display = "none";
  $("bPromote").style.display = "none";
  $("bDiscard").style.display = "none";
  $("retrainStatus").textContent = "Candidate discarded (the file is kept but not used).";
};

// ---- trends: past sessions across days, loaded from durable history ----
function drawTrend(sessions) {
  const cv = $("trendCanvas");
  cv.__redraw = () => drawTrend(sessions);
  const { c, W, H } = hidpi(cv, 0.34), pad = 22;
  if (sessions.length < 2) return;
  const maxPax = Math.max(1, ...sessions.map((s) => s.passersby || 0));
  const X = (i) => pad + i * (W - 2 * pad) / (sessions.length - 1);
  const Yp = (v) => (H - pad) - (v / maxPax) * (H - 2 * pad);
  const Yr = (v) => (H - pad) - (v / 100) * (H - 2 * pad);
  c.strokeStyle = "#efeae0"; c.lineWidth = 1;   // horizontal gridlines
  for (let g = 0; g <= 1.0001; g += 0.25) {
    const y = (H - pad) - g * (H - 2 * pad);
    c.beginPath(); c.moveTo(pad, y); c.lineTo(W - pad, y); c.stroke();
  }
  const draw = (accessor, color) => {
    c.strokeStyle = color; c.lineWidth = 2; c.lineJoin = "round"; c.beginPath();
    sessions.forEach((s, i) => { const x = X(i), y = accessor(s); i ? c.lineTo(x, y) : c.moveTo(x, y); });
    c.stroke();
    c.fillStyle = color;
    sessions.forEach((s, i) => { c.beginPath(); c.arc(X(i), accessor(s), 2.5, 0, 7); c.fill(); });
  };
  draw((s) => Yp(s.passersby || 0), "#4a2a86");
  draw((s) => Yr(s.engagement_rate || 0), "#3fae74");
}

async function loadHistory() {
  let data;
  try { data = await (await fetch("/api/history", { cache: "no-store" })).json(); }
  catch (e) { return; }
  const sessions = data.sessions || [];
  if (!sessions.length) { $("trendEmpty").style.display = "block"; $("trendList").innerHTML = ""; drawTrend([]); return; }
  $("trendEmpty").style.display = "none";
  drawTrend(sessions);
  const recent = sessions.slice(-6).reverse();
  $("trendList").innerHTML = recent.map((s) => {
    const when = (s.ended_at || "").replace("T", " ").slice(5, 16);
    return `<div class="trow"><span class="tdate">${when}</span>` +
      `<span class="tnums">${s.passersby} pax · <b>${s.engagement_rate}%</b> · ${Math.round(s.attention_s)}s</span></div>`;
  }).join("");
}

// ---- model performance history: watch the model improve over time ----
// A snapshot is the metric set on a given set of labeled pairs. metricsFor()
// (defined above) gives auc/ap/brier/f1; we add accuracy + class counts.
function snapFrom(pairs, kind) {
  const m = metricsFor(pairs);
  const pos = pairs.reduce((a, p) => a + p.y, 0), neg = pairs.length - pos;
  const { tp, tn } = confAt(pairs, 0.5);
  const acc = pairs.length ? (tp + tn) / pairs.length : NaN;
  const num = (v) => (typeof v === "number" && !isNaN(v)) ? Number(v.toFixed(4)) : null;
  return { kind, n: pairs.length, pos, neg,
           auc: num(m.auc), ap: num(m.ap), brier: num(m.brier), f1: num(m.f1), accuracy: num(acc) };
}

async function logPerf(snap) {
  try {
    await fetch("/api/perf_log", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ snapshot: snap }),
    });
  } catch (e) { /* logging is best-effort; never block the UI */ }
}

let perfEvals = [];
async function loadPerf() {
  try { perfEvals = (await (await fetch("/api/perf_history", { cache: "no-store" })).json()).evals || []; }
  catch (e) { return; }
  drawPerf(perfEvals);
  renderPerfTable(perfEvals);
}

function drawPerf(evals) {
  const cv = $("perfCanvas");
  cv.__redraw = () => drawPerf(evals);
  const { c, W, H } = hidpi(cv, 0.6), pad = 30;
  cv.__geo = { pad, W, H };
  const pts = evals.filter((e) => e.auc != null || e.brier != null);
  c.font = "10px system-ui, -apple-system, sans-serif";
  for (let g = 0; g <= 1.0001; g += 0.25) {   // gridlines + y ticks (metrics in [0,1])
    const y = (H - pad) - g * (H - 2 * pad);
    c.strokeStyle = "#efeae0"; c.lineWidth = 1;
    c.beginPath(); c.moveTo(pad, y); c.lineTo(W - pad, y); c.stroke();
    c.fillStyle = "#9a9488"; c.textAlign = "right"; c.textBaseline = "middle";
    c.fillText(g.toFixed(2), pad - 5, y);
  }
  c.strokeStyle = "#c9c2b4"; c.lineWidth = 1.2; c.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad);
  cv.__pts = pts;
  if (pts.length < 1) return;
  const X = (i) => pts.length === 1 ? W / 2 : pad + i * (W - 2 * pad) / (pts.length - 1);
  const Y = (v) => (H - pad) - v * (H - 2 * pad);   // metrics live in [0,1]
  cv.__map = { X, Y, pad, W, H };
  const series = (key, color) => {
    c.strokeStyle = color; c.fillStyle = color; c.lineWidth = 2; c.beginPath();
    let started = false;
    pts.forEach((e, i) => {
      const v = e[key]; if (v == null) return;
      const x = X(i), y = Y(v);
      started ? c.lineTo(x, y) : c.moveTo(x, y); started = true;
    });
    c.stroke();
    pts.forEach((e, i) => {
      const v = e[key]; if (v == null) return;
      const x = X(i), y = Y(v);
      c.beginPath();
      if (e.kind === "retrain") { c.moveTo(x, y - 4); c.lineTo(x + 4, y + 3); c.lineTo(x - 4, y + 3); c.closePath(); }
      else c.arc(x, y, 3, 0, 7);
      c.fill();
    });
  };
  series("auc", "#4a2a86");
  series("brier", "#ef8f3c");
  bindPerfHover(cv);
}
function bindPerfHover(cv) {
  if (cv.__hoverBound) return; cv.__hoverBound = true;
  cv.style.cursor = "crosshair";
  cv.addEventListener("mousemove", (e) => {
    const m = cv.__map, pts = cv.__pts; if (!m || !pts || !pts.length) { return; }
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left;
    let bi = 0, best = 1e9;
    pts.forEach((_, i) => { const d = Math.abs(m.X(i) - mx); if (d < best) { best = d; bi = i; } });
    if (best > 24) { hideTip(); redrawChart(cv); return; }
    redrawChart(cv);
    const c = cv.getContext("2d"), x = m.X(bi);
    c.strokeStyle = "rgba(28,27,34,.22)"; c.setLineDash([3, 3]); c.lineWidth = 1;
    c.beginPath(); c.moveTo(x, m.pad); c.lineTo(x, m.H - m.pad); c.stroke(); c.setLineDash([]);
    const ev = pts[bi], when = (ev.at || "").replace("T", " ").slice(5, 16);
    const tag = ev.kind === "retrain" ? "candidate" : (ev.kind === "promote" ? "promoted" : "live");
    const f = (v) => v == null ? "—" : v.toFixed(3);
    showTip(e.clientX, e.clientY,
      `<b>${when}</b> &middot; ${tag}<br>AUC ${f(ev.auc)} &middot; Brier ${f(ev.brier)} &middot; n ${ev.n ?? "—"}`);
  });
  cv.addEventListener("mouseleave", () => { hideTip(); redrawChart(cv); });
}

function renderPerfTable(evals) {
  if (!evals.length) { $("perfEmpty").style.display = "block"; $("perfTable").innerHTML = ""; return; }
  $("perfEmpty").style.display = "none";
  const bestAuc = Math.max(...evals.map((e) => e.auc == null ? -1 : e.auc));
  const rows = evals.slice(-8).reverse().map((e) => {
    const when = (e.at || "").replace("T", " ").slice(5, 16);
    const tag = e.kind === "retrain" ? "candidate" : (e.kind === "promote" ? "promoted" : "live");
    const f = (v) => v == null ? "—" : v.toFixed(3);
    const aucCls = (e.auc != null && e.auc === bestAuc) ? "best" : "";
    return `<tr class="${e.kind === 'retrain' ? 'cand' : ''}"><td>${when} · ${tag}</td>` +
      `<td class="${aucCls}">${f(e.auc)}</td><td>${f(e.brier)}</td><td>${f(e.f1)}</td><td>${e.n ?? "—"}</td></tr>`;
  }).join("");
  $("perfTable").innerHTML =
    `<table class="perf"><thead><tr><th>When</th><th>AUC</th><th>Brier</th><th>F1</th><th>n</th></tr></thead><tbody>${rows}</tbody></table>`;
}

$("bPerfReport").onclick = () => {
  const report = {
    generated_at: new Date().toISOString(),
    store: ($("store").textContent || "").trim(),
    current_session_labeled: anPairs.length,
    evaluations: perfEvals,
  };
  download(JSON.stringify(report, null, 2), "model_performance_report.json");
};

// ---- training data: proof of what the model retrains on + per-person bars ----
// Feeds off /api/training_stats, which counts rows in data/raw_sessions/*.csv —
// the exact files build_dataset.py consumes. So the bars ARE the training set.
const BAR_COLORS = { Alvaro: "#4a2a86", Hector: "#0bb3a6", Cristian: "#ef8f3c" };
async function loadTrainingData() {
  let s;
  try { s = await (await fetch("/api/training_stats", { cache: "no-store" })).json(); }
  catch (e) { return; }
  const known = s.collectors || ["Alvaro", "Hector", "Cristian"];
  const per = s.per_collector || {};
  // Show the known three first, then any other names that slipped into the data.
  const names = known.slice();
  Object.keys(per).forEach((n) => { if (!names.includes(n)) names.push(n); });
  const bars = names.map((n) => ({ name: n, n: per[n] || 0 }));
  $("trainSummary").innerHTML =
    `<b>${s.total || 0}</b> labelled rows across <b>${s.files || 0}</b> session file(s) — ` +
    `this is the full training set the engagement model would retrain on.`;
  if (!s.total) {
    $("trainEmpty").style.display = "block"; $("trainTable").innerHTML = "";
  } else {
    $("trainEmpty").style.display = "none";
    const rows = bars.map((b) => {
      const pct = s.total ? Math.round(100 * b.n / s.total) : 0;
      return `<tr><td><span style="color:${BAR_COLORS[b.name] || '#777'};">●</span> ${b.name}</td>` +
             `<td style="text-align:right;font-variant-numeric:tabular-nums;">${b.n}</td>` +
             `<td style="text-align:right;color:var(--muted);">${pct}%</td></tr>`;
    }).join("");
    $("trainTable").innerHTML =
      `<table class="perf"><thead><tr><th>Collector</th><th style="text-align:right;">Rows</th>` +
      `<th style="text-align:right;">Share</th></tr></thead><tbody>${rows}</tbody></table>`;
  }
  drawTrainBars(bars);
}

function drawTrainBars(bars) {
  const cv = $("trainBarCanvas");
  cv.__redraw = () => drawTrainBars(bars);
  const { c, W, H } = hidpi(cv, 0.6), padL = 34, padR = 14, padT = 14, padB = 26;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const maxN = Math.max(1, ...bars.map((b) => b.n));
  // "nice" top of axis so the gridlines land on round numbers
  const top = niceCeil(maxN);
  const Y = (v) => (H - padB) - (v / top) * plotH;
  c.font = "10px system-ui, -apple-system, sans-serif";
  for (let g = 0; g <= 4; g++) {   // 5 horizontal gridlines with value labels
    const v = top * g / 4, y = Y(v);
    c.strokeStyle = "#efeae0"; c.lineWidth = 1;
    c.beginPath(); c.moveTo(padL, y); c.lineTo(W - padR, y); c.stroke();
    c.fillStyle = "#9a9488"; c.textAlign = "right"; c.textBaseline = "middle";
    c.fillText(String(Math.round(v)), padL - 5, y);
  }
  const slot = plotW / Math.max(1, bars.length);
  const bw = Math.min(70, slot * 0.6);
  const rects = [];
  bars.forEach((b, i) => {
    const cx = padL + slot * (i + 0.5), x = cx - bw / 2, y = Y(b.n), h = (H - padB) - y;
    rects.push({ x, y, w: bw, h, b, cx });
    c.fillStyle = BAR_COLORS[b.name] || "#8a84d6";
    roundRect(c, x, y, bw, Math.max(0, h), 5); c.fill();
    c.fillStyle = "#4a4752"; c.textAlign = "center"; c.textBaseline = "bottom";
    if (b.n > 0) c.fillText(String(b.n), cx, y - 3);
    c.fillStyle = "#6a6560"; c.textBaseline = "top";
    c.fillText(b.name, cx, H - padB + 5);
  });
  cv.__rects = rects;
  bindBarHover(cv);
}
function bindBarHover(cv) {
  if (cv.__hoverBound) return; cv.__hoverBound = true;
  cv.style.cursor = "default";
  cv.addEventListener("mousemove", (e) => {
    const rects = cv.__rects; if (!rects) return;
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const hit = rects.find((q) => mx >= q.x - 6 && mx <= q.x + q.w + 6 && my >= q.y && my <= q.y + q.h + 20);
    if (!hit) { hideTip(); cv.style.cursor = "default"; return; }
    cv.style.cursor = "pointer";
    showTip(e.clientX, e.clientY, `<b>${hit.b.name}</b>: ${hit.b.n} labelled row${hit.b.n === 1 ? "" : "s"}`);
  });
  cv.addEventListener("mouseleave", () => { hideTip(); cv.style.cursor = "default"; });
}
function niceCeil(v) {
  if (v <= 5) return 5;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  return Math.ceil(v / mag) * mag;
}
function roundRect(c, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2 || r);
  c.beginPath();
  c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
}

// Land on the main screen. Fetch the store name once for the header (poll is
// idle until ▶ Start), then keep the poll loop ready for when a session begins.
async function initMain() {
  try {
    const res = await fetch("/api/stats", { cache: "no-store" });
    const d = await res.json();
    $("store").textContent = d.store_name || "Demo";
  } catch (e) {}
  showScreen("start");
}
setInterval(poll, 200);
initMain();
loadHistory();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="VisionMetrics edge agent — local web dashboard")
    ap.add_argument("--config", required=True, help="path to device.yaml")
    ap.add_argument("--debug", action="store_true", help="kept for CLI parity with service.py (unused: the web dashboard IS the debug view)")
    ap.add_argument("--report", default=None, help="write a session report (JSON) here on stop")
    ap.add_argument("--source", default=None, help="override camera source (e.g. 1 for Camo)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="local port for the dashboard")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the browser tab")
    ap.add_argument("--no-record", action="store_true",
                     help="don't save the session video (recording is on by default, "
                          "so sessions can later be labeled for training)")
    ap.add_argument("--record-path", default=None, help="override where the recording is saved")
    ap.add_argument("--review-sample-every", type=int, default=30,
                     help="live 'Quick review' panel: while someone lingers, queue at most "
                          "1 crop every N frames (a new passerby's first frame always queues)")
    ap.add_argument("--review-seconds", type=float, default=10.0,
                     help="post-session review: keep one full frame for labeling every N "
                          "seconds of the session (default: 10)")
    ap.add_argument("--max-width", type=int, default=640,
                     help="downscale camera frames to at most this width before running the "
                          "pipeline, so processing can keep up with the camera in realtime")
    ap.add_argument("--loop", action="store_true",
                     help="replay a file --source from the start when it ends (for demos/previews); "
                          "no effect on a live camera")
    args = ap.parse_args()
    return run(args.config, debug=args.debug, report_path=args.report, source=args.source,
               port=args.port, open_browser=not args.no_open,
               record=not args.no_record, record_path=args.record_path,
               review_sample_every=args.review_sample_every,
               review_seconds=args.review_seconds, max_width=args.max_width,
               loop=args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
