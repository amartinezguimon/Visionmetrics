#!/usr/bin/env python3
"""VisionMetrics — lanzador único (menú) para probar y grabar SIN LIARSE.

    python run.py        (o, en Windows, doble clic en DEMO.bat)

Un solo sitio, paso a paso: probar el modelo en vivo, grabar datos para entrenar,
dibujar la zona de conteo. Cada opción te dice qué archivo enviar a Álvaro.
Pensado para una demo local (portátil + webcam); la versión por tienda vendrá luego.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# Show accents correctly even on a plain Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


def _check_deps() -> bool:
    missing = []
    for mod in ("cv2", "ultralytics", "mediapipe", "torch", "yaml"):
        try:
            __import__(mod)
        except Exception:
            missing.append("opencv-python" if mod == "cv2" else
                           "pyyaml" if mod == "yaml" else mod)
    if missing:
        print("\n  Faltan dependencias:", ", ".join(missing))
        print("  Instálalas UNA vez (desde esta carpeta):\n")
        print("    python -m venv venv")
        print("    venv\\Scripts\\activate      (Windows)")
        print("    source venv/bin/activate    (Mac/Linux)")
        print("    pip install -r requirements.txt\n")
        return False
    return True


def _run(args: list[str], env: dict | None = None) -> None:
    print("\n  > " + " ".join([Path(PY).name, *args]) + "\n")
    # Force UTF-8 in the child so accents/emojis never crash on a Windows console.
    full_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", **(env or {})}
    subprocess.run([PY, *args], cwd=ROOT, env=full_env)


_cam = {"idx": None, "name": None}
CAMERA_PREF_FILE = ROOT / "configs" / "camera_pref.txt"


def _load_camera_pref() -> dict | None:
    """Read the camera chosen in the visual picker as {"index": str, "name": str},
    or None if nothing saved. Accepts BOTH the new JSON format
    (``{"index": 1, "name": "hector gregori Camera"}``) and the legacy plain-int
    format (``1``) so an old pref file still works."""
    if not CAMERA_PREF_FILE.exists():
        return None
    saved = CAMERA_PREF_FILE.read_text(encoding="utf-8").strip()
    if not saved:
        return None
    try:
        data = json.loads(saved)
        if isinstance(data, dict) and "index" in data:
            return {"index": str(data["index"]), "name": str(data.get("name") or "")}
    except (ValueError, TypeError):
        pass
    # Legacy: the file is just a bare index number.
    return {"index": saved, "name": ""}


def _resolve_current_index(pref: dict) -> str:
    """Turn a saved pref into the index to open RIGHT NOW. On macOS the index the
    picker saw can shift before launch, so we re-resolve the saved NAME to whatever
    index currently holds that exact camera — this is what makes "the camera you
    picked is the one that runs" hold. Falls back to the saved index if the name
    can't be resolved (e.g. the phone was unplugged)."""
    name = (pref.get("name") or "").strip()
    if name:
        try:
            from visionmetrics.edge.agent.capture import resolve_camera_by_name
            found = resolve_camera_by_name(name)
            if found is not None:
                if str(found) != str(pref.get("index")):
                    print(f"  (macOS reordenó las cámaras: '{name}' está ahora en el "
                          f"índice {found}, no {pref.get('index')} — uso {found})")
                return str(found)
            print(f"  AVISO: no encuentro la cámara elegida ('{name}') ahora mismo. "
                  f"Conéctala/desbloquéala. Uso el índice guardado {pref.get('index')}.")
        except Exception:
            pass
    return str(pref.get("index"))


def _seleccionar_camara(after_url: str | None = None) -> None:
    """PRIMERA PANTALLA, SIEMPRE (Mac y Windows): abre el selector visual con TODAS
    las cámaras disponibles y OBLIGA a elegir una antes de seguir. Se ejecuta en cada
    arranque; nunca se teclea un número a mano. La elección se guarda en
    configs/camera_pref.txt (la misma que leen todos los lanzadores).

    Si ``after_url`` se pasa, al elegir la cámara la MISMA pestaña se redirige a esa
    URL (el panel web) en cuanto está disponible — así 'elegir cámara' encadena
    directo al sitio, sin callejón de 'cierra la pestaña'."""
    print("\n  === ELIGE TU CÁMARA ===  (obligatorio, en cada arranque)")
    print("  Se abrirá el navegador con TODAS las cámaras disponibles (móvil, webcam…).")
    print("  Haz clic en la que quieras usar — se guarda sola, no hay que escribir nada.")
    print("  Si tu móvil no aparece: conéctalo y pulsa 'Volver a escanear' en la página.")
    picker_args = ["-m", "visionmetrics.edge.tools.camera_picker"]
    if after_url:
        picker_args += ["--after-url", after_url]
    while True:
        try:
            _run(picker_args)
        except KeyboardInterrupt:
            print("\n  (selección de cámara interrumpida)")
        pref = _load_camera_pref()
        if pref is not None:
            _cam["idx"] = pref["index"]
            _cam["name"] = pref["name"]
            label = f"{pref['index']}" + (f" ({pref['name']})" if pref['name'] else "")
            print(f"\n  Cámara seleccionada: {label}")
            return
        if not _yes("No se eligió ninguna cámara. ¿Volver a abrir el selector?"):
            print("  (sin cámara elegida; deberás elegir una en la opción 'Elegir cámara')")
            return


def _camera() -> str:
    """El índice de cámara que se abrirá AHORA, honrando la elección del selector.
    Nunca se teclea a mano. En macOS re-resuelve el NOMBRE guardado al índice que
    esa cámara tenga en este momento (los índices se reordenan), así siempre se
    abre la cámara que de verdad elegiste — no otra."""
    pref = _load_camera_pref()
    if pref is None:
        _seleccionar_camara()
        pref = _load_camera_pref()
    if pref is None:
        return "0"
    _cam["idx"] = pref["index"]
    _cam["name"] = pref["name"]
    return _resolve_current_index(pref)


STORE_CFG = "configs/store_config.json"   # calibrate + draw_zone + live demo all share this
SITE_PORT = 8642                          # puerto del panel web (webserver DEFAULT_PORT)


def _yes(question: str) -> bool:
    """Ask a yes/no question. Enter or 's' = yes; 'n' = skip."""
    return input(f"  {question} [S/n]: ").strip().lower() not in ("n", "no")


def _calibrar() -> None:
    print("\n  Calibrar el escaparate: pon la cámara donde irá fija, mira al centro del")
    print("  escaparate y captura con las teclas 1-5. S = guardar, Q = salir.")
    _run(["-m", "visionmetrics.edge.tools.calibrate", "--device-config", "configs/demo.yaml"],
         env={"VM_CAMERA": _camera()})


def _dibujar_linea() -> None:
    print("\n  Dibuja la LÍNEA DE LEJANÍA con 2 clics (suele ir inclinada, cruzando la calle).")
    print("  Todo lo que quede al OTRO lado (más lejos) NO se cuenta ni se puntúa.")
    print("  Clic en los 2 extremos.  S = guardar   U = deshacer   C = limpiar   Q = salir.")
    _run(["-m", "visionmetrics.edge.tools.draw_line", "--source", _camera(), "--config", STORE_CFG])


def _probar_en_vivo() -> None:
    (ROOT / "results").mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    report = f"results/demo_{stamp}.json"
    print("\n  Se abrirá tu navegador con la cámara y los datos en directo.")
    print("  Pulsa 'Detener sesión' en la página cuando termines.")
    _run(["-m", "visionmetrics.edge.agent.webserver",
          "--config", "configs/demo.yaml", "--report", report,
          "--source", "chosen"])  # el webserver resuelve la cámara elegida, nunca la del Mac
    print("\n  *** LISTO. Envíale a Álvaro este archivo (WhatsApp / email): ***")
    print(f"      {ROOT / report}")

    video = (ROOT / report).with_suffix(".mp4")
    if video.exists() and _yes("\n  La sesión quedó grabada. ¿Quieres etiquetarla ahora?"):
        print("\n  Analizando el vídeo (detectando personas + mirada)... puede tardar varios minutos.")
        _run(["-m", "visionmetrics.training.prep", str(video)])
        print("\n  *** Se habrá abierto el navegador para etiquetar/corregir. ***")
        print("  *** Cuando acabes, pulsa 'Descargar CSV etiquetado' y mándamelo (WhatsApp / email). ***")


def demo_guiada() -> None:
    """Todo en uno: calibrar → dibujar zona → probar. Cada paso se puede saltar."""
    print("\n  === DEMO GUIADA ===  (puedes saltar cualquier paso escribiendo 'n')")
    print("\n  Paso 1/3 — Calibrar el escaparate (hacia dónde mira la gente).")
    if _yes("¿Calibrar ahora?"):
        _calibrar()
    else:
        print("  (saltado — se usará lo que ya hubiera, o nada)")
    print("\n  Paso 2/3 — Dibujar la línea de lejanía (a partir de ahí no se cuenta).")
    if _yes("¿Dibujar la línea ahora?"):
        _dibujar_linea()
    else:
        print("  (saltado)")
    print("\n  Paso 3/3 — Probar el modelo en vivo.")
    _probar_en_vivo()


def _recolectar(nombre: str) -> None:
    """Abre la pantalla de recolección (ventana de la cámara) con la cámara ELEGIDA."""
    print("\n  Se abrirá la cámara. MIRA A LA CÁMARA = 'mirando'.")
    print("  L=mira  A=no mira  T=grabar seguido  M=cambia  G=gafas  H=gorra  Q=guardar.")
    _run(["-m", "visionmetrics.training.collect", "--collector", nombre, "--camera", _camera()])
    print("\n  *** Al terminar te dijo la ruta de UN archivo .csv: envíaselo a Álvaro. ***")


def grabar_datos() -> None:
    nombre = input("\n  Tu nombre (p. ej. hector): ").strip() or "hector"
    _recolectar(nombre)


def abrir_sitio() -> None:
    """Flujo de UN CLIC: primera pantalla = selector de cámara; al elegir, la MISMA
    pestaña se redirige AL SITIO (el panel web VisionMetrics — vista en vivo +
    etiquetado + reentreno). El selector se abre con --after-url apuntando al panel,
    y el webserver se lanza con --no-open para no abrir una segunda pestaña."""
    url = f"http://localhost:{SITE_PORT}/"
    _seleccionar_camara(after_url=url)   # primera pantalla; al clicar redirige al sitio
    print(f"\n  Abriendo el panel VisionMetrics… ({url})")
    print("  La pestaña del selector se convertirá en el panel en unos segundos.")
    # 'chosen' = el webserver resuelve la cámara elegida EN SU PROPIO proceso (mismo
    # que captura) y JAMÁS usa la del Mac — arregla 'elijo el móvil pero corre el Mac'.
    _run(["-m", "visionmetrics.edge.agent.webserver",
          "--config", "configs/demo.yaml", "--source", "chosen",
          "--no-open", "--port", str(SITE_PORT)])


def ver_camaras() -> None:
    """Reabrir el selector de cámara para cambiarla en mitad de la sesión."""
    _seleccionar_camara()


def importar_y_etiquetar() -> None:
    """Import a recorded video, run the real detector over it, and open the
    browser labeler with everything already loaded — no extra Python needed."""
    ruta = input("\n  Ruta del vídeo a importar (arrástralo aquí y pulsa Enter): ").strip().strip('"')
    if not ruta:
        print("  (cancelado, no se dio ninguna ruta)")
        return
    if not Path(ruta).exists():
        print(f"  No encuentro el archivo: {ruta}")
        return
    print("\n  Analizando el vídeo (detectando personas + mirada)... puede tardar varios minutos.")
    _run(["-m", "visionmetrics.training.prep", ruta])
    print("\n  *** Se habrá abierto el navegador para etiquetar/corregir. ***")
    print("  *** Cuando acabes, pulsa 'Descargar CSV etiquetado' y mándamelo (WhatsApp / email). ***")


def etiquetar_ultima_grabacion() -> None:
    """No path to type: grabs whichever recording (results/ or recordings/) is
    newest and opens it straight in the labeler — the direct follow-up to
    stopping a live session without going through the menu that ran it."""
    print("\n  Buscando tu grabación más reciente...")
    _run(["-m", "visionmetrics.training.label_latest"])
    print("\n  *** Se habrá abierto el navegador para etiquetar/corregir. ***")
    print("  *** Cuando acabes, pulsa 'Descargar CSV etiquetado' y mándamelo (WhatsApp / email). ***")


MENU = {
    "1": ("Probar en vivo", demo_guiada),
    "2": ("Grabar datos", grabar_datos),
    "3": ("Dibujar línea de lejanía", _dibujar_linea),
    "4": ("Calibrar escaparate", _calibrar),
    "5": ("Elegir cámara", ver_camaras),
    "6": ("Importar vídeo", importar_y_etiquetar),
    "7": ("Etiquetar última grabación", etiquetar_ultima_grabacion),
}


def main() -> int:
    if not _check_deps():
        input("\n  Enter para salir  ")
        return 1
    # Modo "un clic → al sitio": la primera pantalla es el selector de cámara y, al
    # elegir, la MISMA pestaña se redirige al panel web. Sin menú. Lo usan
    # Recolectar.command/.bat y VisionMetrics.command/.bat.
    if any(f in sys.argv for f in ("--sitio", "--recolectar", "--collect")):
        abrir_sitio()
        return 0

    _seleccionar_camara()   # PRIMERA PANTALLA obligatoria: elegir cámara, siempre
    while True:
        print("\n  VisionMetrics\n")
        for key, (label, _) in MENU.items():
            print(f"    {key}   {label}")
        print(f"    0   Salir")
        choice = input("\n  ›  ").strip()
        if choice == "0":
            print("\n  Hasta luego.\n")
            return 0
        item = MENU.get(choice)
        if not item:
            print("  Opción no válida.")
            continue
        try:
            item[1]()
        except KeyboardInterrupt:
            print("\n  (interrumpido)")
        except Exception as exc:  # noqa: BLE001 - keep the menu alive for a non-tech user
            print(f"  Error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
