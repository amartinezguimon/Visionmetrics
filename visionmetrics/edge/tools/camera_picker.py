"""Visual camera picker — see a snapshot of every camera before choosing one.

`check_cameras.py` prints index numbers to a terminal, which the user then has
to type back into `configs/camera_pref.txt` (or answer a prompt) blind — and
on macOS the index AVFoundation assigns to each device isn't stable: it can
shift depending on what's connected/streaming (built-in FaceTime camera,
Continuity Camera, Camo Studio...), so "index 1" one session can be a totally
different physical camera the next.

This tool removes the guesswork: it snapshots every camera index, opens a
browser page showing all of them side by side, and a single click saves the
chosen index to `configs/camera_pref.txt` — the same file every launcher
(`run.py`, `VisionMetrics Live.command`) already reads. No typing required.

Run:
    python -m visionmetrics.edge.tools.camera_picker
"""

from __future__ import annotations

import base64
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
PREF_FILE = ROOT / "configs" / "camera_pref.txt"
MAX_INDEX = 6
DEFAULT_PORT = 8850


def _snapshot(index: int, warm_secs: float = 1.5) -> tuple[bytes, tuple[int, int]] | None:
    """Open `index`, warm it up briefly, and return (jpeg_bytes, (w, h)) for the
    brightest frame seen — or None if the camera doesn't open / never yields a
    real (non-black) frame."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None

    best_frame, best_brightness, t0 = None, -1.0, time.time()
    while time.time() - t0 < warm_secs:
        ok, frame = cap.read()
        if ok and frame is not None:
            b = float(frame.mean())
            if b > best_brightness:
                best_frame, best_brightness = frame, b
            if b >= 30:  # clearly a real, lit image — no need to keep waiting
                break
        time.sleep(0.03)
    cap.release()

    if best_frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", best_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return None
    h, w = best_frame.shape[:2]
    return buf.tobytes(), (w, h)


def _indices_to_scan(max_index: int) -> list[int]:
    """Which camera indices to snapshot. On macOS, ONLY the real enumerated devices:
    asking OpenCV for an index beyond the camera count silently opens the DEFAULT
    (built-in) camera, so probing 0..max_index would (a) show duplicate Mac cards for
    every phantom index and (b) churn the iPhone's Continuity Camera enough to make it
    drop after a couple of seconds. Off-macOS we can't name devices, so scan the range."""
    import sys
    if sys.platform == "darwin":
        try:
            from visionmetrics.edge.agent.capture import _mac_camera_names
            n = len(_mac_camera_names())
            if n:
                return list(range(n))
        except Exception:
            pass
    return list(range(max_index))


def scan_cameras(max_index: int = MAX_INDEX) -> list[dict]:
    found = []
    for i in _indices_to_scan(max_index):
        print(f"[camera-picker] probando cámara {i}...")
        result = _snapshot(i)
        if result is None:
            continue
        jpg, (w, h) = result
        name = ""
        try:
            from visionmetrics.edge.agent.capture import camera_name_at
            name = camera_name_at(i) or ""
        except Exception:
            name = ""
        found.append({
            "index": i, "width": w, "height": h, "name": name,
            "b64": base64.b64encode(jpg).decode("ascii"),
        })
    return found


_PAGE_TEMPLATE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>VisionMetrics — elegir cámara</title>
<style>
  :root {{ --bg:#0e1117; --panel:#161b22; --border:#30363d; --accent:#3b82f6; --text:#e6edf3; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
          font:15px/1.5 -apple-system, system-ui, sans-serif; padding:28px; }}
  h1 {{ margin:0 0 6px; font-size:22px; }}
  p.hint {{ color:#9aa4af; margin:0 0 24px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr));
           gap:20px; }}
  .card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px;
           overflow:hidden; display:flex; flex-direction:column; }}
  .card img {{ width:100%; aspect-ratio:4/3; object-fit:cover; display:block;
               background:#000; }}
  .card .body {{ padding:12px 14px 14px; display:flex; flex-direction:column; gap:8px; }}
  .card .meta {{ color:#9aa4af; font-size:13px; }}
  button {{ background:var(--accent); color:#fff; border:none; border-radius:7px;
            padding:10px 14px; font-size:14px; font-weight:600; cursor:pointer; }}
  button:hover {{ filter:brightness(1.1); }}
  button:disabled {{ background:#2e7d32; cursor:default; }}
  .empty {{ color:#9aa4af; padding:40px 0; text-align:center; }}
  #status {{ margin-top:22px; font-size:14px; color:#9aa4af; }}
</style>
</head>
<body>
  <h1>¿Cuál es tu cámara?</h1>
  <p class="hint">Se probaron los índices 0-{max_index_m1}. Haz clic en la que muestre tu
     móvil (o la cámara correcta) — se guarda al momento, sin escribir nada.<br>
     <b>¿Tu móvil aparece un segundo y desaparece?</b> Ábrelo/reconéctalo, espera a que el
     vídeo se vea estable en su propia app, y pulsa "Volver a escanear" — puedes pulsarlo
     tantas veces como haga falta, no hace falta cerrar esta ventana.</p>
  <button id="rescanBtn" onclick="rescan()" style="margin-bottom:20px;">🔄 Volver a escanear</button>
  <div class="grid" id="grid">{cards}</div>
  <div id="status"></div>

<script>
function pick(index, btn) {{
  document.querySelectorAll('button').forEach(b => b.disabled = true);
  btn.textContent = "Guardando...";
  fetch('/api/select', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{index: index}})
  }}).then(r => r.json()).then(d => {{
    btn.textContent = "✓ Guardada";
    if (d.redirect) {{
      // Esta MISMA pestaña se convierte en el panel: esperamos a que el servidor
      // del panel esté arriba (lo lanza el programa justo después) y navegamos.
      document.getElementById('status').textContent =
        "Cámara " + index + " seleccionada. Abriendo el panel VisionMetrics…";
      goTo(d.redirect);
    }} else {{
      document.getElementById('status').textContent =
        "Cámara " + index + " seleccionada. Continúa en la ventana del programa. " +
        "Puedes cerrar esta pestaña.";
    }}
  }}).catch(() => {{
    document.getElementById('status').textContent = "Error guardando. Inténtalo de nuevo.";
    document.querySelectorAll('button').forEach(b => b.disabled = false);
    btn.textContent = "Usar esta cámara";
  }});
}}
function goTo(url) {{
  // Sondea el panel hasta que responde y entonces redirige esta pestaña a él.
  fetch(url, {{mode: 'no-cors', cache: 'no-store'}})
    .then(() => {{ window.location.href = url; }})
    .catch(() => setTimeout(() => goTo(url), 500));
}}
function rescan() {{
  document.getElementById('rescanBtn').disabled = true;
  document.getElementById('rescanBtn').textContent = "Escaneando...";
  location.reload();
}}
</script>
</body>
</html>"""

_CARD_TEMPLATE = """<div class="card">
  <img src="data:image/jpeg;base64,{b64}" alt="cámara {index}">
  <div class="body">
    <div class="meta">Cámara {index} · {width}x{height}</div>
    <div class="meta" style="color:#e6edf3;font-weight:600;">{name}</div>
    <button onclick="pick({index}, this)">Usar esta cámara</button>
  </div>
</div>"""


def _render_page(cams: list[dict]) -> str:
    if not cams:
        cards = '<div class="empty">No se encontró ninguna cámara. Conecta tu móvil ' \
                '(Camo / Continuity Camera) y vuelve a intentarlo.</div>'
    else:
        cards = "".join(_CARD_TEMPLATE.format(**c) for c in cams)
    return _PAGE_TEMPLATE.format(cards=cards, max_index_m1=MAX_INDEX - 1)


def _make_handler(scan: "callable[[], list[dict]]", on_selected, after_url: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path != "/":
                self.send_response(404)
                self.end_headers()
                return
            cams = scan()  # rescans live on every load, so a just-reconnected
                            # phone camera (or a dropped one) is picked up fresh
            page = _render_page(cams).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self):
            if self.path != "/api/select":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
                index = int(data["index"])
            except Exception:
                self.send_response(400)
                self.end_headers()
                return
            on_selected(index)
            resp = {"ok": True, "index": index}
            if after_url:
                resp["redirect"] = after_url
            body = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    return Handler


def run(port: int = DEFAULT_PORT, open_browser: bool = True, max_index: int = MAX_INDEX,
        after_url: str | None = None) -> int:
    def scan() -> list[dict]:
        print("[camera-picker] escaneando cámaras (incluye Camo Studio / Continuity Camera)...")
        cams = scan_cameras(max_index)
        print(f"[camera-picker] {len(cams)} cámara(s) con imagen encontradas.")
        return cams

    selected: dict = {}

    def on_selected(index: int) -> None:
        selected["index"] = index
        PREF_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Save BOTH the index AND the camera's NAME (its stable identity). macOS
        # reshuffles indices between now and launch, so at launch we re-resolve
        # the name to whatever index it currently holds — this is what guarantees
        # "the camera you picked is the camera that actually runs".
        name = None
        try:
            from visionmetrics.edge.agent.capture import camera_name_at
            name = camera_name_at(index)
        except Exception:
            name = None
        payload = json.dumps({"index": index, "name": name or ""}, ensure_ascii=False)
        PREF_FILE.write_text(payload, encoding="utf-8")
        print(f"[camera-picker] guardado -> {PREF_FILE} = {payload}")

    handler = _make_handler(scan, on_selected, after_url)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"[camera-picker] abriendo {url}")
    print("[camera-picker] si tu móvil aparece y desaparece, reconéctalo y pulsa "
          "'Volver a escanear' en la página — no hace falta reiniciar esto.")
    if open_browser:
        webbrowser.open(url)
    httpd.serve_forever()

    if "index" in selected:
        print(f"[camera-picker] listo — la demo usará la cámara {selected['index']} a partir de ahora.")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Pick a camera visually and save it as the default.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    ap.add_argument("--max-index", type=int, default=MAX_INDEX)
    ap.add_argument("--after-url", default=None,
                    help="on pick, redirect the SAME browser tab here (e.g. the live "
                         "dashboard) once it's reachable, instead of showing a dead-end")
    a = ap.parse_args()
    return run(port=a.port, open_browser=not a.no_open, max_index=a.max_index,
               after_url=a.after_url)


if __name__ == "__main__":
    raise SystemExit(main())
