"""Draw the "line of farness" for a store, on a frame from its camera.

This is the operator step that fixes "people too far away get counted": instead of
tracing a whole polygon, you drop just TWO points to draw a single (usually slanted)
line across the scene. Everyone whose feet land on the FAR side of it — the opposite
pavement, the far end of the street — is ignored: never counted, never scored for
engagement. The near side (toward the bottom of the frame, where the camera stands)
is where your customers walk.

    python -m visionmetrics.edge.tools.draw_line --source 0 --config configs/store_config.json

Controls (in the window):
    SPACE       freeze the current live frame to draw on (wait until you see the image)
    left click  drop an endpoint (two clicks = one line; a 3rd click starts over)
    u           undo last point        c   clear both points
    r           back to live video (re-freeze)
    s / Enter   save the line into the store config and quit
    q / Esc     quit without saving

Showing the LIVE feed first (instead of grabbing one still) is deliberate: phone
webcams like Camo Studio stream intermittently / start black, so you freeze the
moment the picture looks right.

The line is stored in NORMALISED [0..1] image coordinates (two endpoints), so it
survives a camera-resolution change (but must be re-drawn if the camera is moved).

The pure helpers (normalise / merge / load) are unit-tested; the OpenCV window in
``main`` is not (it needs a display + a camera).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_line(points_px: list[tuple[int, int]], frame_w: int, frame_h: int
                   ) -> list[list[float]]:
    """Two pixel endpoints -> normalised [0..1] coords, clamped to the frame."""
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("frame_w and frame_h must be positive")
    if len(points_px) != 2:
        raise ValueError("a line needs exactly 2 endpoints")
    out: list[list[float]] = []
    for x, y in points_px:
        nx = min(1.0, max(0.0, x / frame_w))
        ny = min(1.0, max(0.0, y / frame_h))
        out.append([round(nx, 4), round(ny, 4)])
    return out


def load_config_dict(path: str | Path) -> dict:
    """Read a store-config JSON, or an empty dict if it doesn't exist yet."""
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def merge_far_line(config: dict, line_norm: list[list[float]]) -> dict:
    """Return a copy of ``config`` with the far_line set/replaced.

    Every other key (engagement_zone, derived, counting_region, …) is preserved
    untouched, so this can be run alongside the other calibration steps.
    """
    updated = dict(config)
    updated["far_line"] = {
        "_comment": "Normalised [0..1] image coords, two endpoints. Feet on the FAR "
                    "side (away from the frame bottom) are ignored: not counted, "
                    "not scored for engagement.",
        "line": line_norm,
    }
    return updated


def save_far_line(config_path: str | Path, line_norm: list[list[float]]) -> None:
    merged = merge_far_line(load_config_dict(config_path), line_norm)
    Path(config_path).write_text(json.dumps(merged, indent=2, ensure_ascii=False),
                                 encoding="utf-8")


def main() -> int:
    import cv2
    import numpy as np

    from ..agent.capture import open_capture  # DirectShow on Windows so Camo/OBS open
    from ..agent.zone import FarLine

    ap = argparse.ArgumentParser(description="Draw the per-store line of farness.")
    ap.add_argument("--source", default="0", help="camera index, RTSP url, video, or image path")
    ap.add_argument("--config", default="configs/store_config.json",
                    help="store config JSON to update (created if missing)")
    args = ap.parse_args()

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    win = "VisionMetrics - linea de lejania"
    points: list[tuple[int, int]] = []
    frozen = {"img": None}     # the still we draw on (None = showing live video)

    # An image-file source is frozen immediately; a camera streams until you freeze it.
    cap = None
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if Path(args.source).suffix.lower() in img_exts and Path(args.source).exists():
        img = cv2.imread(args.source)
        if img is None:
            raise SystemExit(f"No pude leer la imagen: {args.source}")
        frozen["img"] = img
    else:
        cap = open_capture(args.source)
        if not cap.isOpened():
            raise SystemExit(f"No pude abrir la cámara {args.source}. Usa 'Ver cámaras'.")

    def on_mouse(event, x, y, _flags, _param):
        if frozen["img"] is not None and event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= 2:          # a 3rd click starts a fresh line
                points.clear()
            points.append((x, y))

    def shade_far_side(canvas):
        """Tint the far half so the operator sees exactly what gets ignored."""
        h, w = canvas.shape[:2]
        fl = FarLine.from_config({"line": normalize_line(points, w, h)})
        if fl is None:
            return
        overlay = canvas.copy()
        # Cheap raster: mark the far side per pixel via a coarse grid of blocks.
        step = 12
        for gy in range(0, h, step):
            fy = (gy + step / 2) / h
            for gx in range(0, w, step):
                fx = (gx + step / 2) / w
                if fl.is_far(fx, fy):
                    cv2.rectangle(overlay, (gx, gy), (gx + step, gy + step),
                                  (40, 40, 200), -1)
        cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("[linea] ESPACIO=congelar  clic=extremo (2)  S=guardar  U=deshacer  C=limpiar  R=en vivo  Q=salir")

    live = None
    while True:
        if frozen["img"] is None:                       # ── LIVE preview ──
            ok, f = cap.read()
            if ok and f is not None:
                live = f
            canvas = live.copy() if live is not None else np.zeros((480, 640, 3), np.uint8)
            if live is None or float(live.mean()) < 8:
                cv2.putText(canvas, "Imagen NEGRA: abre Camo y conecta el movil",
                            (12, 30), FONT, 0.6, (60, 60, 255), 2)
            else:
                cv2.putText(canvas, "Cuando se vea bien, pulsa ESPACIO para congelar",
                            (12, 30), FONT, 0.6, (0, 215, 255), 2)
        else:                                           # ── FROZEN: draw line ──
            canvas = frozen["img"].copy()
            if len(points) == 2:
                shade_far_side(canvas)
                cv2.line(canvas, points[0], points[1], (0, 215, 240), 2)
            for p in points:
                cv2.circle(canvas, p, 6, (0, 215, 240), -1)
            hint = ("2 extremos | S=guardar  U=deshacer  R=en vivo" if len(points) == 2
                    else f"{len(points)}/2 extremos | clic para marcar")
            cv2.putText(canvas, hint, (12, 30), FONT, 0.52, (255, 255, 255), 2)
            if len(points) == 2:
                cv2.putText(canvas, "zona roja = IGNORADA (demasiado lejos)",
                            (12, canvas.shape[0] - 16), FONT, 0.52, (60, 60, 255), 2)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            print("Cancelado — no se guardó nada.")
            break
        elif key == 32 and cap is not None:             # SPACE = freeze the current frame
            if live is not None and float(live.mean()) >= 8:
                frozen["img"] = live.copy()
                points.clear()
            else:
                print("  Aún no hay imagen (negra). Abre Camo / conecta el móvil.")
        elif key == ord("r") and cap is not None:       # back to live video
            frozen["img"] = None
            points.clear()
        elif key == ord("u") and points:
            points.pop()
        elif key == ord("c"):
            points.clear()
        elif key in (ord("s"), 13):
            if frozen["img"] is None:
                print("  Primero congela la imagen con ESPACIO.")
                continue
            if len(points) != 2:
                print("  Necesitas exactamente 2 extremos.")
                continue
            h, w = frozen["img"].shape[:2]
            line = normalize_line(points, w, h)
            save_far_line(args.config, line)
            print(f"Línea de lejanía guardada en {args.config}")
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
