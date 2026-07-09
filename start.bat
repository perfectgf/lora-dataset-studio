@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (set "PY=py -3") || (where python >nul 2>nul && (set "PY=python") || (
  echo Python 3.11+ is required but was not found.
  echo Install it from https://www.python.org/downloads/ then re-run start.bat
  exit /b 1
))
if not exist .venv ( %PY% -m venv .venv || exit /b 1 )
call .venv\Scripts\activate.bat
python -m pip install -q -r backend\requirements.txt || exit /b 1
if not exist frontend\dist\index.html (
  echo frontend\dist is missing — this repo ships it prebuilt. Run: cd frontend ^&^& npm install ^&^& npm run build
  exit /b 1
)
rem Port 5000 is a frequent collision (macOS AirPlay, another local Flask app).
rem Use 5050 by default; override by setting LDS_PORT before running start.bat.
if not defined LDS_PORT set "LDS_PORT=5050"
start "" http://127.0.0.1:%LDS_PORT%/
python backend\run.py
