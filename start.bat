@echo off
setlocal
cd /d "%~dp0"

REM --- Pick a Python the OPTIONAL ML extras have wheels for (CPython 3.10-3.12).
REM     insightface / numpy<2 / onnxruntime publish NO wheels for 3.13+, so a venv
REM     built on e.g. 3.14 cannot install requirements-ml.txt (pip falls back to
REM     source builds that fail). `py -3.x` selects an EXACT version; bare `py -3`
REM     or `python` grab the NEWEST installed one -- which is exactly the trap.
set "PY="
set "PY_SUPPORTED=1"
for %%V in (3.12 3.11 3.10) do (
  if not defined PY ( py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V" )
)
if not defined PY (
  set "PY_SUPPORTED=0"
  echo [!] No CPython 3.10-3.12 found. The core app still runs, but the OPTIONAL
  echo     ML extras -- face-similarity scoring and background masks -- cannot be
  echo     installed on a newer Python. Install Python 3.12 from
  echo     https://www.python.org/downloads/ and re-run start.bat to enable them.
  echo.
  where py >nul 2>nul && set "PY=py -3"
  if not defined PY ( where python >nul 2>nul && set "PY=python" )
)
if not defined PY (
  echo Python 3.10-3.12 is required but no Python was found.
  echo Install it from https://www.python.org/downloads/ then re-run start.bat
  exit /b 1
)

REM --- Call the venv's python DIRECTLY everywhere below -- never `activate.bat`.
REM     A venv carries hardcoded paths; if it was ever moved or copied (e.g. a
REM     provisioning step that builds it elsewhere then relocates it), its
REM     activate.bat no longer shadows the system python, so `call activate` +
REM     `python` silently falls back to the machine default (the 3.14 trap). The
REM     python.exe stub, by contrast, always resolves to the venv interpreter.
set "VPY=.venv\Scripts\python.exe"

REM --- Ensure .venv exists AND runs a supported Python; rebuild it otherwise
REM     (a stale venv on 3.13+ can't install the ML extras). Only rebuild when a
REM     supported Python is actually available -- rebuilding on 3.14 gains nothing.
set "REBUILD=0"
set "VENV_BAD=0"
if not exist "%VPY%" set "REBUILD=1"
if "%REBUILD%"=="0" (
  "%VPY%" -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>nul
  if errorlevel 1 set "VENV_BAD=1"
)
if "%VENV_BAD%"=="1" if "%PY_SUPPORTED%"=="1" (
  echo [i] Existing .venv is not on Python 3.10-3.12 -- rebuilding it so the ML extras can install.
  set "REBUILD=1"
)
if "%VENV_BAD%"=="1" if "%PY_SUPPORTED%"=="0" (
  echo [!] .venv runs an unsupported Python and no CPython 3.10-3.12 is installed.
  echo     Core features work; ML extras stay unavailable until you install 3.12.
)
if "%REBUILD%"=="1" (
  if exist .venv rmdir /s /q .venv
  %PY% -m venv .venv || exit /b 1
)

"%VPY%" -m pip install -q -r backend\requirements.txt || exit /b 1
if not exist frontend\dist\index.html (
  echo frontend\dist is missing -- this repo ships it prebuilt. Run: cd frontend ^&^& npm install ^&^& npm run build
  exit /b 1
)
rem Port 5000 is a frequent collision (macOS AirPlay, another local Flask app).
rem Use 5050 by default; override by setting LDS_PORT before running start.bat.
if not defined LDS_PORT set "LDS_PORT=5050"
start "" http://127.0.0.1:%LDS_PORT%/
"%VPY%" backend\run.py
