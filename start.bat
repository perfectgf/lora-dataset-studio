@echo off
setlocal
cd /d "%~dp0"

REM --- Pick a Python the OPTIONAL ML extras have wheels for (CPython 3.10-3.12).
REM     insightface / numpy<2 / onnxruntime publish NO wheels for 3.13+, so a venv
REM     built on e.g. 3.14 cannot install requirements-ml.txt (pip falls back to
REM     source builds that fail). `py -3.x` selects an EXACT version; bare `py -3`
REM     or `python` grab the NEWEST installed one -- which is exactly the trap.
set "PY="
for %%V in (3.12 3.11 3.10) do (
  if not defined PY ( py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V" )
)
if not defined PY (
  echo [!] No CPython 3.10-3.12 found. The app itself will still run, but the
  echo     OPTIONAL ML extras -- face-similarity scoring and background masks --
  echo     cannot be installed on a newer Python. To use them, install Python 3.12
  echo     from https://www.python.org/downloads/ and re-run start.bat.
  echo.
  where py >nul 2>nul && set "PY=py -3"
  if not defined PY ( where python >nul 2>nul && set "PY=python" )
)
if not defined PY (
  echo Python 3.10-3.12 is required but no Python was found.
  echo Install it from https://www.python.org/downloads/ then re-run start.bat
  exit /b 1
)

if not exist .venv ( %PY% -m venv .venv || exit /b 1 )
call .venv\Scripts\activate.bat

REM Warn if a PRE-EXISTING .venv is on an ML-incompatible Python (made before this
REM fix, or by another tool). Core app still runs; only the in-app "Install (pip)"
REM for ML extras would fail. Probes computed at top level (no parens inside the
REM if-block, which cmd would mis-parse).
for /f "delims=" %%v in ('python -c "import sys;print('.'.join(map(str,sys.version_info[:2])))"') do set "PYVER=%%v"
for /f %%i in ('python -c "import sys;print(1 if (3,10)<=sys.version_info[:2]<=(3,12) else 0)"') do set "ML_OK=%%i"
if "%ML_OK%"=="0" (
  echo [!] This .venv runs Python %PYVER%, outside the ML-extras range 3.10 to 3.12.
  echo     Core features work. For face scoring / masks, delete the .venv folder
  echo     and re-run start.bat with Python 3.12 installed.
)

python -m pip install -q -r backend\requirements.txt || exit /b 1
if not exist frontend\dist\index.html (
  echo frontend\dist is missing -- this repo ships it prebuilt. Run: cd frontend ^&^& npm install ^&^& npm run build
  exit /b 1
)
rem Port 5000 is a frequent collision (macOS AirPlay, another local Flask app).
rem Use 5050 by default; override by setting LDS_PORT before running start.bat.
if not defined LDS_PORT set "LDS_PORT=5050"
start "" http://127.0.0.1:%LDS_PORT%/
python backend\run.py
