"""Turn human detection verdicts into a YOLO fine-tuning dataset.

The labeling page (`review.html`) and the live dashboard (`webserver.py`) both
write a `*_detections.csv` in the shared DETECTION_COLUMNS schema — one row per
box the human judged: verdict 1 = real person, 0 = not a person, and hand-drawn
"missed person" boxes come through as verdict 1 with a negative track_id. This
script joins those verdicts back to the actual frame pixels and writes an
Ultralytics-format dataset (images/ + labels/ + data.yaml) you can fine-tune
YOLOv8 on, so the detector learns *your* camera, lighting and false positives.

Two frame sources, auto-detected per session:
  * a prep `<stem>.json` (from prep.py) — pixels are embedded, no video needed;
  * a video file — frames are seeked by index (for live-recorded sessions).

Only frames a human actually reviewed become images. Within a reviewed frame,
verdict-1 boxes become `person` labels; verdict-0 boxes are simply left out, so
that region trains the model as background (this is how you *reduce* false
positives). A reviewed frame with no real people becomes an empty-label
background image on purpose.

Run:
    python -m visionmetrics.training.detector.build_yolo_dataset --scan .
    python -m visionmetrics.training.detector.build_yolo_dataset \
        --pair clip_detections.csv clip.json --out data/detector_dataset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
DET_SUFFIX = "_detections.csv"


class FrameSource:
    """Yields native-resolution frames + their (width, height) by frame index,
    from either a prep JSON (embedded JPEGs) or a video file."""

    def __init__(self, path: Path):
        import cv2
        import numpy as np

        self._cv2 = cv2
        self._np = np
        self.path = path
        self._cap = None
        self._embedded: dict[int, str] = {}
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = data.get("video", {})
            self.width = int(meta.get("width") or 0)
            self.height = int(meta.get("height") or 0)
            for f in data.get("frames", []):
                if f.get("frame"):
                    self._embedded[int(f["i"])] = f["frame"]
        else:
            self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                raise SystemExit(f"cannot open frame source {path!r}")
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get(self, frame_idx: int):
        import base64

        if self._cap is not None:
            self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self._cap.read()
            return frame if ok else None
        b64 = self._embedded.get(frame_idx)
        if not b64:
            return None
        buf = self._np.frombuffer(base64.b64decode(b64), self._np.uint8)
        return self._cv2.imdecode(buf, self._cv2.IMREAD_COLOR)

    def release(self):
        if self._cap is not None:
            self._cap.release()


def find_source(csv_path: Path) -> Path | None:
    """Locate the frame source that matches a `<stem>_detections.csv`, preferring
    an embedded prep JSON, then a same-named video, both next to the CSV."""
    stem = csv_path.name[: -len(DET_SUFFIX)] if csv_path.name.endswith(DET_SUFFIX) else csv_path.stem
    folder = csv_path.parent
    js = folder / f"{stem}.json"
    if js.exists():
        return js
    for suf in VIDEO_SUFFIXES:
        vid = folder / f"{stem}{suf}"
        if vid.exists():
            return vid
    return None


def rows_by_frame(csv_path: Path) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                fi = int(row["frame_idx"])
            except (KeyError, ValueError):
                continue
            grouped.setdefault(fi, []).append(row)
    return grouped


def yolo_line(row: dict, w: int, h: int) -> str | None:
    """One YOLO label line (class 0 = person) from a native-pixel box, or None
    if the box is degenerate. Coordinates are clamped into [0, 1]."""
    try:
        x1, y1, x2, y2 = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
    except (KeyError, ValueError):
        return None
    if w <= 0 or h <= 0 or x2 <= x1 or y2 <= y1:
        return None
    cx = min(max((x1 + x2) / 2 / w, 0.0), 1.0)
    cy = min(max((y1 + y2) / 2 / h, 0.0), 1.0)
    bw = min(max((x2 - x1) / w, 0.0), 1.0)
    bh = min(max((y2 - y1) / h, 0.0), 1.0)
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _to_px(row: dict, nw: int, nh: int, fw: int, fh: int) -> list[float] | None:
    """Detection box (stored in native nw×nh coords) mapped into the actual frame
    pixel space (fw×fh differ when a prep JSON embedded a downscaled JPEG)."""
    try:
        x1, y1, x2, y2 = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
    except (KeyError, ValueError):
        return None
    if nw <= 0 or nh <= 0 or x2 <= x1 or y2 <= y1:
        return None
    sx, sy = fw / nw, fh / nh
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def _window_around(box_px, fw: int, fh: int, context: float, tile_min: int) -> list[int]:
    """Crop window centred on `box_px`, padded by `context`× the box on each side
    and never smaller than `tile_min`, clamped to the frame."""
    x1, y1, x2, y2 = box_px
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max((x2 - x1) * (1 + context) / 2, tile_min / 2)
    half_h = max((y2 - y1) * (1 + context) / 2, tile_min / 2)
    return [int(max(0, round(cx - half_w))), int(max(0, round(cy - half_h))),
            int(min(fw, round(cx + half_w))), int(min(fh, round(cy + half_h)))]


def _line_in_window(box_px, win, min_keep: float = 0.35) -> str | None:
    """YOLO line (class 0) for `box_px` clipped into crop `win`, or None if too
    little of the box survives — a person sliced off at the tile edge is dropped
    rather than mislabeled as a whole one."""
    x1, y1, x2, y2 = box_px
    wx1, wy1, wx2, wy2 = win
    ix1, iy1, ix2, iy2 = max(x1, wx1), max(y1, wy1), min(x2, wx2), min(y2, wy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    orig, inter = (x2 - x1) * (y2 - y1), (ix2 - ix1) * (iy2 - iy1)
    W, H = (wx2 - wx1), (wy2 - wy1)
    if orig <= 0 or W <= 0 or H <= 0 or inter / orig < min_keep:
        return None
    cx = min(max(((ix1 + ix2) / 2 - wx1) / W, 0.0), 1.0)
    cy = min(max(((iy1 + iy2) / 2 - wy1) / H, 0.0), 1.0)
    bw = min(max((ix2 - ix1) / W, 0.0), 1.0)
    bh = min(max((iy2 - iy1) / H, 0.0), 1.0)
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _tile_reps(is_miss: bool, box_px, conf, fw: int, fh: int, max_reps: int) -> int:
    """Copies of one hard-example tile, weighted by how *surprising* the error is:
    confident false positives and small/far missed people are the most valuable,
    so they repeat more. Deliberately gentle — capped at `max_reps`."""
    if max_reps <= 1:
        return 1
    if is_miss:
        area = max(0.0, box_px[2] - box_px[0]) * max(0.0, box_px[3] - box_px[1])
        weight = 1.0 - min(1.0, (area / max(1.0, float(fw * fh))) / 0.15)  # tiny/far→~1
    else:
        try:
            weight = min(max(float(conf), 0.0), 1.0)  # confident FP → more copies
        except (TypeError, ValueError):
            weight = 0.5
    return 1 + round(weight * (max_reps - 1))


def is_val(key: str, val_frac: float, seed: int) -> bool:
    """Deterministic per-image train/val assignment (stable across re-runs, so
    the same frame never hops between splits and leaks)."""
    digest = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1000.0 < val_frac


def build(pairs: list[tuple[Path, Path]], out_dir: Path, *, val_frac: float, seed: int,
          tiles: bool = True, tile_context: float = 1.0, tile_min: int = 128,
          max_reps: int = 3) -> int:
    import cv2

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    positives = 0
    backgrounds = 0
    tile_imgs = 0
    tile_neg = 0
    for csv_path, source_path in pairs:
        grouped = rows_by_frame(csv_path)
        if not grouped:
            print(f"[dataset] {csv_path.name}: no rows, skipping")
            continue
        src = FrameSource(source_path)
        w, h = src.width, src.height
        stem = csv_path.name[: -len(DET_SUFFIX)] if csv_path.name.endswith(DET_SUFFIX) else csv_path.stem
        kept = 0
        for fi, rows in sorted(grouped.items()):
            frame = src.get(fi)
            if frame is None:
                continue
            fh, fw = frame.shape[:2]
            nw, nh = (w or fw), (h or fh)   # native dims the CSV boxes are in
            lines = [ln for r in rows if str(r.get("verdict", "")).strip() == "1"
                     for ln in [yolo_line(r, nw, nh)] if ln]
            key = f"{stem}_{fi}"
            split = "val" if is_val(key, val_frac, seed) else "train"
            cv2.imwrite(str(out_dir / "images" / split / f"{key}.jpg"), frame)
            (out_dir / "labels" / split / f"{key}.txt").write_text("\n".join(lines), encoding="utf-8")
            counts[split] += 1
            if lines:
                positives += len(lines)
            else:
                backgrounds += 1
            kept += 1

            # Hard-example tiles: crops centred on the two things the model got
            # WRONG — rejected boxes (false positives → background here) and drawn
            # missed people (→ positives) — so their gradient isn't diluted among a
            # whole frame of easy background. Train split only, to keep val a clean,
            # realistic full-frame measure. Every verdict-1 box that survives the
            # clip is labeled, so a tile stays correct even with bystanders in view.
            if tiles and split == "train":
                verdict1_px = [pb for r in rows if str(r.get("verdict", "")).strip() == "1"
                               for pb in [_to_px(r, nw, nh, fw, fh)] if pb]
                foci = []
                for r in rows:
                    v = str(r.get("verdict", "")).strip()
                    try:
                        tid = int(float(r.get("track_id")))
                    except (TypeError, ValueError):
                        tid = 0
                    if v == "0":
                        foci.append((r, False))      # rejected false positive
                    elif v == "1" and tid < 0:
                        foci.append((r, True))       # hand-drawn missed person
                for ti, (r, is_miss) in enumerate(foci):
                    bpx = _to_px(r, nw, nh, fw, fh)
                    if bpx is None:
                        continue
                    win = _window_around(bpx, fw, fh, tile_context, tile_min)
                    wx1, wy1, wx2, wy2 = win
                    if (wx2 - wx1) < 24 or (wy2 - wy1) < 24:
                        continue
                    crop = frame[wy1:wy2, wx1:wx2]
                    if crop.size == 0:
                        continue
                    tlines = [ln for pb in verdict1_px for ln in [_line_in_window(pb, win)] if ln]
                    reps = _tile_reps(is_miss, bpx, r.get("conf"), fw, fh, max_reps)
                    for k in range(reps):
                        tkey = f"{stem}_{fi}_t{ti}" + (f"_r{k}" if k else "")
                        cv2.imwrite(str(out_dir / "images" / "train" / f"{tkey}.jpg"), crop)
                        (out_dir / "labels" / "train" / f"{tkey}.txt").write_text(
                            "\n".join(tlines), encoding="utf-8")
                        counts["train"] += 1
                        tile_imgs += 1
                        if not tlines:
                            tile_neg += 1
        src.release()
        print(f"[dataset] {csv_path.name} + {source_path.name}: {kept} reviewed frame(s)")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: person\n",
        encoding="utf-8",
    )
    total = counts["train"] + counts["val"]
    print(f"\n[dataset] {total} images (train {counts['train']} / val {counts['val']}), "
          f"{positives} person box(es), {backgrounds} background frame(s)")
    if tiles and tile_imgs:
        print(f"[dataset] + {tile_imgs} hard-example tile(s) "
              f"({tile_imgs - tile_neg} positive / {tile_neg} negative), "
              f"oversampled up to {max_reps}× by surprise (train only)")
    print(f"[dataset] wrote {data_yaml}")
    if counts["val"] == 0:
        print("[dataset] WARNING: val split is empty — add more sessions or raise --val-frac.")
    print("[dataset] next: python -m visionmetrics.training.detector.finetune_detector "
          f"--data {data_yaml}")
    return 0


def resolve_pairs(scan: str | None, explicit: list[list[str]]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for csv_str, src_str in explicit:
        pairs.append((Path(csv_str), Path(src_str)))
    if scan:
        for csv_path in sorted(Path(scan).rglob(f"*{DET_SUFFIX}")):
            source = find_source(csv_path)
            if source is None:
                print(f"[dataset] {csv_path}: no matching .json/video next to it, skipping")
                continue
            pairs.append((csv_path, source))
    # de-dupe while preserving order
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[Path, Path]] = []
    for c, s in pairs:
        k = (str(c.resolve()), str(s.resolve()))
        if k not in seen:
            seen.add(k)
            unique.append((c, s))
    return unique


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a YOLO dataset from detection verdicts.")
    ap.add_argument("--scan", default=None,
                    help="folder to search recursively for *_detections.csv (each matched "
                         "to a <stem>.json or <stem>.<video> next to it)")
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("CSV", "SOURCE"),
                    help="explicit detections CSV + frame source (prep .json or a video); repeatable")
    ap.add_argument("--out", default="data/detector_dataset", help="dataset output directory")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of frames held out for val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tiles", action=argparse.BooleanOptionalAction, default=True,
                    help="also emit hard-example crop tiles centred on each rejected box "
                         "(false positive → background) and each hand-drawn missed person "
                         "(positive), so the gradient concentrates on your actual errors; "
                         "full frames are always kept too (--no-tiles to A/B without them)")
    ap.add_argument("--tile-context", type=float, default=1.0,
                    help="context padding around a tile's focus box, in multiples of the box "
                         "per side (bigger = more scene context, less concentrated signal)")
    ap.add_argument("--tile-min", type=int, default=128,
                    help="minimum tile side in px, so tiny far boxes still crop to a usable size")
    ap.add_argument("--max-reps", type=int, default=3,
                    help="max copies of one hard-example tile; confident false positives and "
                         "small/far misses repeat up to this. 1 disables oversampling")
    a = ap.parse_args()

    if not a.scan and not a.pair:
        a.scan = "."   # sensible default: look in the current folder
    pairs = resolve_pairs(a.scan, a.pair)
    if not pairs:
        print("[dataset] no (CSV, frame-source) pairs found. Label a session first, or pass --pair.")
        return 1
    return build(pairs, Path(a.out), val_frac=a.val_frac, seed=a.seed,
                 tiles=a.tiles, tile_context=a.tile_context,
                 tile_min=a.tile_min, max_reps=a.max_reps)


if __name__ == "__main__":
    raise SystemExit(main())
