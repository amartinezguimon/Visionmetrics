#!/usr/bin/env python3
"""VisionMetrics — lanzador único (menú) para probar y grabar SIN LIARSE.

    python run.py        (o, en Windows, doble clic en DEMO.bat)

Un solo sitio, paso a paso: probar el modelo en vivo, grabar datos para entrenar,
dibujar la zona de conteo. Cada opción te dice qué archivo enviar a Álvaro.
Pensado para una demo local (portátil + webcam); la versión por tienda vendrá luego.
"""

from __future__ import annotations

import datetime as dt
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


_cam = {"idx": None}
CAMERA_PREF_FILE = ROOT / "configs" / "camera_pref.txt"


def _camera() -> str:
    """Which camera number to use. Remembered across runs once you pick one,
    so you don't have to type it in every time (e.g. your phone via Camo)."""
    if _cam["idx"] is not None:
        return _cam["idx"]

    if CAMERA_PREF_FILE.exists():
        saved = CAMERA_PREF_FILE.read_text(encoding="utf-8").strip()
        if saved:
            print(f"  Usando la cámara guardada: {saved}  "
                  f"(para cambiarla, borra {CAMERA_PREF_FILE.relative_to(ROOT)})")
            _cam["idx"] = saved
            return saved

    v = input("  ¿Número de cámara? (Enter = 0; usa 'Ver cámaras' si no sabes): ").strip()
    _cam["idx"] = v if v else "0"
    try:
        CAMERA_PREF_FILE.parent.mkdir(parents=True, exist_ok=True)
        CAMERA_PREF_FILE.write_text(_cam["idx"], encoding="utf-8")
    except Exception:
        pass  # not critical if we can't save the preference
    return _cam["idx"]


STORE_CFG = "configs/store_config.json"   # calibrate + draw_zone + live demo all share this


def _yes(question: str) -> bool:
    """Ask a yes/no question. Enter or 's' = yes; 'n' = skip."""
    return input(f"  {question} [S/n]: ").strip().lower() not in ("n", "no")


def _calibrar() -> None:
    print("\n  Calibrar el escaparate: pon la cámara donde irá fija, mira al centro del")
    print("  escaparate y captura con las teclas 1-5. S = guardar, Q = salir.")
    _run(["visionmetrics/edge/tools/calibrate.py"], env={"VM_CAMERA": _camera()})


def _dibujar_zona() -> None:
    print("\n  Dibuja la zona que SÍ cuenta (la acera de delante; deja margen en los bordes).")
    print("  Clic en cada esquina.  S = guardar   U = deshacer   C = limpiar   Q = salir.")
    _run(["-m", "visionmetrics.edge.tools.draw_zone", "--source", _camera(), "--config", STORE_CFG])


def _probar_en_vivo() -> None:
    (ROOT / "results").mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    report = f"results/demo_{stamp}.json"
    print("\n  Se abrirá tu navegador con la cámara y los datos en directo.")
    print("  Pulsa 'Detener sesión' en la página cuando termines.")
    _run(["-m", "visionmetrics.edge.agent.webserver",
          "--config", "configs/demo.yaml", "--report", report, "--source", _camera()])
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
    print("\n  Paso 2/3 — Dibujar la zona de conteo (la acera que cuenta).")
    if _yes("¿Dibujar la zona ahora?"):
        _dibujar_zona()
    else:
        print("  (saltado)")
    print("\n  Paso 3/3 — Probar el modelo en vivo.")
    _probar_en_vivo()


def grabar_datos() -> None:
    nombre = input("\n  Tu nombre (p. ej. hector): ").strip() or "hector"
    print("\n  Se abrirá la cámara. MIRA A LA CÁMARA = 'mirando'.")
    print("  L=mira  A=no mira  T=grabar seguido  M=cambia  G=gafas  H=gorra  Q=guardar.")
    _run(["-m", "visionmetrics.training.collect", "--collector", nombre, "--camera", _camera()])
    print("\n  *** Al terminar te dijo la ruta de UN archivo .csv: envíaselo a Álvaro. ***")


def ver_camaras() -> None:
    print("\n  Se abrirá el navegador con una foto de cada cámara conectada.")
    print("  Haz clic en la que sea tu móvil — se guarda sola, no hay que escribir nada.")
    _run(["-m", "visionmetrics.edge.tools.camera_picker"])
    _cam["idx"] = None  # forzar a releer configs/camera_pref.txt (se acaba de actualizar)


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
    "3": ("Dibujar zona", _dibujar_zona),
    "4": ("Calibrar escaparate", _calibrar),
    "5": ("Elegir cámara", ver_camaras),
    "6": ("Importar vídeo", importar_y_etiquetar),
    "7": ("Etiquetar última grabación", etiquetar_ultima_grabacion),
}


def main() -> int:
    if not _check_deps():
        input("\n  Enter para salir  ")
        return 1
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
