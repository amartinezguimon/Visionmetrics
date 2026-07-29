#!/bin/bash
# ============================================================
#  VisionMetrics — lanzador de doble clic (vista en vivo)
#  Portátil: funciona desde cualquier carpeta y en cualquier Mac.
#  La PRIMERA vez crea el entorno e instala las dependencias solo.
#  Al lanzarse desde tu Terminal, macOS le concede permiso de cámara.
# ============================================================

# Ir SIEMPRE a la carpeta de este archivo (no a una ruta fija) → portátil.
cd "$(dirname "$0")" || {
  echo "  No pude entrar en la carpeta del programa."
  read -n 1 -s -r -p "  Pulsa una tecla para cerrar…"
  exit 1
}

clear
echo ""
echo "  VisionMetrics — arrancando la vista en vivo…"
echo "  (se abrirá solo en el navegador; deja esta ventana abierta)"
echo ""

# --- ¿Hay Python 3? ---
if ! command -v python3 >/dev/null 2>&1; then
  echo "  No encuentro Python 3 en este ordenador."
  echo "  Instálalo desde https://www.python.org/downloads/ y vuelve a abrir este archivo."
  echo ""
  read -n 1 -s -r -p "  Pulsa una tecla para cerrar…"
  exit 1
fi

# --- Primera vez: crear el entorno virtual e instalar dependencias ---
if [ ! -x "venv/bin/python" ]; then
  echo "  Primera vez en este ordenador: preparando el entorno."
  echo "  (esto descarga las librerías; puede tardar varios minutos)"
  echo ""
  python3 -m venv venv || {
    echo "  No pude crear el entorno virtual (venv)."
    read -n 1 -s -r -p "  Pulsa una tecla para cerrar…"
    exit 1
  }
  venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1
  if ! venv/bin/python -m pip install -r requirements.txt; then
    echo ""
    echo "  Fallo instalando las dependencias. Revisa tu conexión e inténtalo de nuevo."
    read -n 1 -s -r -p "  Pulsa una tecla para cerrar…"
    exit 1
  fi
  echo ""
  echo "  Entorno listo."
  echo ""
fi

# Al arrancar, el programa descarga los modelos que falten (MediaPipe + YOLO)
# automáticamente. --source external => SIEMPRE usa la cámara EXTERNA
# (móvil / Camo / USB), nunca la del Mac; si no hay externa, avisa y no arranca.
export PYTHONUTF8=1
venv/bin/python -m visionmetrics.edge.agent.webserver --config configs/demo.yaml --source external

echo ""
echo "  Sesión terminada. Puedes cerrar esta ventana."
read -n 1 -s -r -p "  Pulsa una tecla para cerrar…"
