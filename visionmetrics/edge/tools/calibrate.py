"""calibrate.py — Store-Specific Engagement Zone Calibration
----------------------------------------------------------
PURPOSE:
    Run ONCE per store installation. Generates configs/store_config.json,
    which defines the calibrated boundaries of a specific display (vitrina,
    poster, shelf, etc.) relative to where the camera is mounted.

    The engagement model (trained by training/train.py) does NOT change
    between stores. Only this config file changes — that's what makes the
    system scalable to any store without retraining.

WHY THIS REUSES THE LIVE AGENT'S OWN CLASSES (PersonDetector, HeadPoseAnalyzer,
TorsoAnalyzer), INSTEAD OF ITS OWN COPY OF THE ANGLE MATH:
    This file used to carry its own MediaPipe setup and its own `get_angles()`
    — a straight duplicate of the pipeline's angle math, with its own hardcoded
    FOV/face-width constants. `geometry.py`'s own docstring records why that is
    dangerous: "This logic was previously DUPLICATED... Having two copies
    caused a real bug once." Calibration now runs through the EXACT SAME code
    path as the live agent (same `DeviceConfig`, same classes), so the angles
    you capture here are, by construction, computed identically to what the
    live agent will see later — there is no second implementation to drift.

HOW TO RUN:
    python -m visionmetrics.edge.tools.calibrate --device-config configs/demo.yaml

SETUP BEFORE RUNNING:
    1. Mount your camera exactly where it will stay permanently. Do NOT move
       it after calibration.
    2. Make sure the display/vitrina you want to track is in the camera's view.
    3. Run this script and follow the on-screen instructions.

OUTPUT:
    configs/store_config.json (merged with any existing counting_region —
    this tool and draw_zone.py write different parts of the same file).

CONTROLS DURING CALIBRATION:
    '1'  ->  Capture LEFT edge of the display (look at the far left)
    '2'  ->  Capture CENTER of the display (look straight at the center)
    '3'  ->  Capture RIGHT edge of the display (look at the far right)
    '4'  ->  Capture the closest customer distance (stand 0.3-0.5m away)
    '5'  ->  Capture the farthest useful customer distance (stand 2-3m away)
    'S'  ->  Save config and exit
    'Q'  ->  Quit without saving
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# ── pure helpers (unit-tested without a camera/model) ──────────────────────

DEFAULT_TOLERANCE_YAW_DEG = 15.0    # buffer on each side of the captured left/right
DEFAULT_TOLERANCE_PITCH_DEG = 10.0  # buffer above/below the captured centre
# Both in DEGREES: yaw/pitch are real solvePnP angles (see geometry.py), not
# the old unitless 2D ratio — a completely different numeric scale, so these
# replace what used to be 0.15/0.10 "ratio units". Reasoned starting values,
# not yet validated against real calibration data; re-tune from a real store.


def compute_derived_zone(
    yaw_left: float, yaw_right: float, pitch_center: float, *,
    tolerance_yaw_deg: float = DEFAULT_TOLERANCE_YAW_DEG,
    tolerance_pitch_deg: float = DEFAULT_TOLERANCE_PITCH_DEG,
) -> dict:
    """Turn 3 captured angle samples (degrees) into the yaw/pitch tolerance
    band `zone.zone_confidence` reads. Pure — no camera, no I/O."""
    return {
        "yaw_min": round(min(yaw_left, yaw_right) - tolerance_yaw_deg, 3),
        "yaw_max": round(max(yaw_left, yaw_right) + tolerance_yaw_deg, 3),
        "pitch_min": round(pitch_center - tolerance_pitch_deg, 3),
        "pitch_max": round(pitch_center + tolerance_pitch_deg, 3),
    }


def load_config_dict(path: str | Path) -> dict:
    """Read a store-config JSON, or an empty dict if it doesn't exist yet."""
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def merge_calibration(existing: dict, calibration: dict) -> dict:
    """`calibration` wins on any key it sets; anything ELSE already in
    `existing` (notably `counting_region`, written by draw_zone.py) survives
    untouched — the two tools share one file without clobbering each other."""
    merged = dict(calibration)
    for k, v in existing.items():
        merged.setdefault(k, v)
    return merged


def save_calibration(config_path: str | Path, calibration: dict) -> None:
    merged = merge_calibration(load_config_dict(config_path), calibration)
    p = Path(config_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")


# ── interactive tool (needs a display + camera + models; not unit-tested) ──

def main() -> int:
    import cv2

    from ..agent import camera_model
    from ..agent.capture import open_capture
    from ..agent.config import DeviceConfig
    from ..agent.vision.detector import PersonDetector
    from ..agent.vision.face import HeadPoseAnalyzer
    from ..agent.vision.pose import TorsoAnalyzer

    ap = argparse.ArgumentParser(description="Calibrate a store's engagement zone.")
    ap.add_argument("--device-config", default="configs/demo.yaml",
                    help="device/demo yaml with this camera's real FOV, face-size "
                         "assumption and model paths (the SAME file the live agent uses)")
    ap.add_argument("--config", default="configs/store_config.json",
                    help="store calibration JSON to update (created if missing)")
    ap.add_argument("--tolerance-yaw", type=float, default=DEFAULT_TOLERANCE_YAW_DEG)
    ap.add_argument("--tolerance-pitch", type=float, default=DEFAULT_TOLERANCE_PITCH_DEG)
    args = ap.parse_args()

    device = DeviceConfig.load(args.device_config)
    # VM_CAMERA env var (set by run.py's menu) overrides the config's camera
    # source, so an operator picking "camera 2" in the menu doesn't have to
    # also go edit a yaml file.
    camera_source = os.environ.get("VM_CAMERA", str(device.camera.source))
    camera_index = int(camera_source) if str(camera_source).isdigit() else camera_source

    print(f"[calibrate] cámara={camera_index}  fov_h_deg={device.camera.fov_h_deg}  "
          f"face_width_m={device.vision.face_width_m}")
    print("[calibrate] cargando modelos (YOLO + FaceLandmarker + PoseLandmarker)...")
    detector = PersonDetector(device.models.yolo, conf_min=device.vision.yolo_conf_min,
                               aspect_ratio_min=device.vision.aspect_ratio_min)
    head_pose = HeadPoseAnalyzer(
        device.models.face, face_width_m=device.vision.face_width_m,
        head_crop_frac=device.vision.head_crop_frac, head_upscale=device.vision.head_upscale,
        skip_frames=1,   # this is an interactive tool — analyse every frame, no stale cache
    )
    torso = TorsoAnalyzer(
        device.models.pose, neutral_span=device.vision.torso_neutral_span,
        min_visibility=device.vision.torso_min_visibility, skip_frames=1,
    )

    cap = open_capture(camera_index)   # DirectShow-on-Windows + backend fallback, same as the live agent
    if not cap.isOpened():
        print(f"ERROR: no pude abrir la cámara {camera_index}. Usa la opción 'Ver cámaras'.")
        return 1

    store_name = input("Nombre de esta tienda/ubicación: ").strip() or "default_store"
    captured: dict[str, tuple[float, float, float]] = {}   # key -> (yaw, pitch, distance)
    calibration = {"store_name": store_name, "camera_index": camera_index, "engagement_zone": {}}

    INSTRUCTIONS = {
        "1": "Mira al borde IZQUIERDO del escaparate",
        "2": "Mira al CENTRO del escaparate",
        "3": "Mira al borde DERECHO del escaparate",
        "4": "Ponte tan CERCA como llegaría un cliente (0.3-0.5m), mira al centro",
        "5": "Ponte tan LEJOS como llegaría un cliente (2-3m), mira al centro",
    }
    print("\n" + "=" * 55)
    print("  MODO CALIBRACIÓN")
    print("=" * 55)
    for k, v in INSTRUCTIONS.items():
        print(f"  Pulsa '{k}'  ->  {v}")
    print("  Pulsa 'S'  ->  Guardar y salir")
    print("  Pulsa 'Q'  ->  Salir sin guardar")
    print("=" * 55 + "\n")

    frame_idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_idx += 1
        now = time.time() - t0
        h, w = frame.shape[:2]
        focal_px = camera_model.focal_length_px(w, device.camera.fov_h_deg)

        dets = detector.detect(frame)
        best = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                   default=None)

        yaw = pitch = distance = dist_m = rel_yaw = None
        if best is not None:
            x1, y1, x2, y2 = best.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
            pose = head_pose.analyze(frame, best.bbox, best.track_id, frame_idx, focal_px, now)
            if pose is not None:
                yaw, pitch, distance, dist_m = pose.yaw, pose.pitch, pose.distance, pose.dist_m
                cv2.circle(frame, pose.nose_px, 6, (0, 255, 0), -1)
                t_result = torso.analyze(frame, best.bbox, best.track_id, frame_idx)
                rel_yaw = t_result.rel_yaw

        # ── HUD ──────────────────────────────────────────────────────
        if yaw is not None:
            cv2.putText(frame, f"Yaw:   {yaw:+.1f} deg", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch:+.1f} deg", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Dist:  ~{dist_m:.2f}m", (10, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            if rel_yaw is not None:
                cv2.putText(frame, f"R-Yaw: {rel_yaw:+.3f}", (10, 114),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 100), 2)
        else:
            msg = "Persona no detectada — ponte delante de la cámara" if best is None \
                else "Cara no detectada — mira hacia la cámara"
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        status_y = 145
        for k in "12345":
            color = (0, 255, 0) if k in captured else (100, 100, 100)
            mark = "OK" if k in captured else "  "
            cv2.putText(frame, f"[{mark}] {k}: {INSTRUCTIONS[k][:42]}", (10, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            status_y += 22

        cv2.imshow("VisionMetrics - Calibracion  [1-5=Capturar | S=Guardar | Q=Salir]", frame)
        key = cv2.waitKey(1) & 0xFF
        k = chr(key) if 0 <= key < 256 else ""

        if k in "12345" and yaw is not None:
            captured[k] = (yaw, pitch, distance)
            print(f"  [{k}] capturado -> yaw={yaw:+.1f} pitch={pitch:+.1f} dist=~{dist_m:.2f}m")
            z = calibration["engagement_zone"]
            if k == "2":
                z["yaw_center"], z["pitch_center"] = round(yaw, 3), round(pitch, 3)
            elif k == "1":
                z["yaw_left"] = round(yaw, 3)
            elif k == "3":
                z["yaw_right"] = round(yaw, 3)
            elif k == "4":
                z["dist_close"], z["dist_close_m"] = round(distance, 5), round(dist_m, 2)
            elif k == "5":
                z["dist_far"], z["dist_far_m"] = round(distance, 5), round(dist_m, 2)

        elif key == ord("s"):
            z = calibration["engagement_zone"]
            if None in (z.get("yaw_left"), z.get("yaw_right"), z.get("pitch_center")):
                print("  Captura al menos IZQUIERDA (1), CENTRO (2) y DERECHA (3) antes de guardar.")
                continue
            calibration["derived"] = compute_derived_zone(
                z["yaw_left"], z["yaw_right"], z["pitch_center"],
                tolerance_yaw_deg=args.tolerance_yaw, tolerance_pitch_deg=args.tolerance_pitch,
            )
            calibration["derived"]["dist_min"] = z.get("dist_far") or 0.03
            if z.get("dist_far_m") is not None:
                calibration["derived"]["dist_max_m"] = z["dist_far_m"]

            save_calibration(args.config, calibration)
            d = calibration["derived"]
            print(f"\nCalibracion guardada -> {args.config}")
            print(f"  Yaw:   [{d['yaw_min']:+.1f}, {d['yaw_max']:+.1f}] deg")
            print(f"  Pitch: [{d['pitch_min']:+.1f}, {d['pitch_max']:+.1f}] deg")
            if d.get("dist_max_m"):
                print(f"  Distancia max: {d['dist_max_m']:.1f}m")
            break

        elif key == ord("q"):
            print("Cancelado - no se guardo nada.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
