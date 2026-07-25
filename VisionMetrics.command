#!/bin/bash
# ============================================================
#  VisionMetrics — lanzador de doble clic (vista en vivo)
#  Doble clic sobre este archivo para abrir la cámara en directo.
#  Al lanzarse desde tu Terminal, macOS le concede permiso de cámara.
# ============================================================

cd "$HOME/Desktop/AI-AD-main" || {
  echo "  No encuentro la carpeta ~/Desktop/AI-AD-main"
  read -n 1 -s -r -p "  Pulsa una tecla para cerrar…"
  exit 1
}

clear
echo ""
echo "  VisionMetrics — arrancando la vista en vivo…"
echo "  (se abrirá solo en el navegador; deja esta ventana abierta)"
echo ""

if [ ! -x "venv/bin/python" ]; then
  echo "  No encuentro venv/bin/python en $(pwd)."
  echo "  Falta el entorno virtual (venv). Créalo antes de usar este lanzador."
  echo ""
  read -n 1 -s -r -p "  Pulsa una tecla para cerrar…"
  exit 1
fi

export PYTHONUTF8=1
venv/bin/python -m visionmetrics.edge.agent.webserver --config configs/demo.yaml --source 0

echo ""
echo "  Sesión terminada. Puedes cerrar esta ventana."
read -n 1 -s -r -p "  Pulsa una tecla para cerrar…"
