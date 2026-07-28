@echo off
REM Doble clic para abrir el menu de VisionMetrics (probar / grabar / zona) - Windows.
REM Portatil + auto-instala el entorno la primera vez.
chcp 65001 >nul
cd /d "%~dp0"

REM --- Si el entorno ya existe, ir directo al menu ---
if exist "venv\Scripts\python.exe" goto run

REM --- Buscar un Python 3 para preparar el entorno la 1a vez ---
set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY ( python --version >nul 2>nul && set "PY=python" )
if not defined PY (
  echo   No encuentro Python 3. Instalalo desde https://www.python.org/downloads/
  echo   Marca "Add Python to PATH" y usa Python 3.11 o 3.12 ^(no 3.13^).
  pause
  exit /b 1
)

echo   Primera vez: preparando el entorno ^(puede tardar varios minutos^)...
%PY% -m venv venv
if not exist "venv\Scripts\python.exe" ( echo   No pude crear el venv. & pause & exit /b 1 )
"venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo   Fallo instalando dependencias. Si usas Python 3.13, instala 3.11/3.12, borra "venv" y reabre.
  pause
  exit /b 1
)

:run
set PYTHONUTF8=1
"venv\Scripts\python.exe" run.py
echo.
pause
