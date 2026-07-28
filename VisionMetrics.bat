@echo off
REM ============================================================
REM  VisionMetrics - lanzador de doble clic (vista en vivo) - Windows
REM  Portatil: funciona desde cualquier carpeta y en cualquier PC.
REM  La PRIMERA vez crea el entorno e instala las dependencias solo.
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"

cls
echo.
echo   VisionMetrics - arrancando la vista en vivo...
echo   (se abrira solo en el navegador; deja esta ventana abierta)
echo.

REM --- Si el entorno ya existe, arrancar directamente ---
if exist "venv\Scripts\python.exe" goto run

REM --- Buscar un Python 3 para preparar el entorno la 1a vez ---
set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY ( python --version >nul 2>nul && set "PY=python" )
if not defined PY (
  echo   No encuentro Python 3 en este ordenador.
  echo   Instalalo desde https://www.python.org/downloads/
  echo   IMPORTANTE: en el instalador marca "Add Python to PATH".
  echo   Usa Python 3.11 o 3.12 ^(mediapipe aun NO soporta 3.13^).
  echo.
  pause
  exit /b 1
)

echo   Primera vez en este ordenador: preparando el entorno.
echo   (esto descarga las librerias; puede tardar varios minutos)
echo.
%PY% -m venv venv
if not exist "venv\Scripts\python.exe" (
  echo   No pude crear el entorno virtual ^(venv^).
  pause
  exit /b 1
)
"venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo   Fallo instalando las dependencias.
  echo   Causa habitual: Python 3.13 ^(mediapipe no tiene version compatible aun^).
  echo   Solucion: instala Python 3.11 o 3.12, borra la carpeta "venv" y reabre este archivo.
  pause
  exit /b 1
)
echo.
echo   Entorno listo.
echo.

:run
set PYTHONUTF8=1
"venv\Scripts\python.exe" -m visionmetrics.edge.agent.webserver --config configs/demo.yaml --source 0
echo.
echo   Sesion terminada. Puedes cerrar esta ventana.
pause
