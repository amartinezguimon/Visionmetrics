#!/bin/bash
# Doble clic para abrir el menu de VisionMetrics (probar / grabar / zona).
# Equivalente en Mac de DEMO.bat (Windows).
cd "$(dirname "$0")"

if [ -x "venv/bin/python3" ]; then
    venv/bin/python3 run.py
else
    python3 run.py
fi

echo
read -p "Pulsa Enter para cerrar esta ventana... " _
