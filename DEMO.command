#!/bin/bash
# Doble clic para abrir el menu de VisionMetrics (probar / grabar / zona).
# Equivalente en Mac de DEMO.bat (Windows). Portátil + auto-instala la 1ª vez.
cd "$(dirname "$0")" || exit 1

if ! command -v python3 >/dev/null 2>&1; then
    echo "  No encuentro Python 3. Instálalo desde https://www.python.org/downloads/"
    read -p "Pulsa Enter para cerrar esta ventana... " _
    exit 1
fi

# Primera vez: crear el entorno virtual e instalar dependencias.
if [ ! -x "venv/bin/python" ]; then
    echo "  Primera vez: preparando el entorno (puede tardar varios minutos)…"
    python3 -m venv venv || { echo "  No pude crear el venv."; read -p "Enter para cerrar... " _; exit 1; }
    venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1
    venv/bin/python -m pip install -r requirements.txt || { echo "  Fallo instalando dependencias."; read -p "Enter para cerrar... " _; exit 1; }
fi

venv/bin/python run.py

echo
read -p "Pulsa Enter para cerrar esta ventana... " _
