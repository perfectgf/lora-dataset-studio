@echo off
setlocal
cd /d "%~dp0"
REM Force UTF-8 so pip reads requirements*.txt as UTF-8 regardless of the system
REM locale. On a GBK / other non-UTF-8 Windows locale, older pip decodes the file
REM with the ANSI codepage and a non-ASCII comment byte crashes the install
REM (UnicodeDecodeError). Reported by the community. Requirements comments are
REM also kept ASCII-only as a second line of defence.
set "PYTHONUTF8=1"

REM --- Pick a Python the OPTIONAL ML extras have wheels for (CPython 3.10-3.12).
REM     insightface / numpy<2 / onnxruntime publish NO wheels for 3.13+, so a venv
REM     built on e.g. 3.14 cannot install requirements-ml.txt (pip falls back to
REM     source builds that fail). `py -3.x` selects an EXACT version; bare `py -3`
REM     or `python` grab the NEWEST installed one -- which is exactly the trap.
set "PY="
set "PY_SUPPORTED=1"

REM 1) A CPython 3.10-3.12 already installed (via the Windows py launcher)?
for %%V in (3.12 3.11 3.10) do (
  if not defined PY ( py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V" )
)

REM 2) A self-contained Python we fetched on a previous run? (skip the PowerShell hop)
REM    Relative path on purpose: cwd is pinned to the repo root above, so this stays
REM    space-safe even if the folder was unzipped under a path that has spaces.
if not defined PY if exist ".python\python.exe" set "PY=.python\python.exe"

REM 3) Nothing suitable -> fetch a self-contained CPython 3.12 automatically: no
REM    system install, no admin, nothing added to PATH. This is what lets start.bat
REM    be a true one-click launcher even on a machine with no Python at all. Needs
REM    PowerShell (ships with Windows) + an internet connection.
if not defined PY (
  echo [i] No CPython 3.10-3.12 found -- downloading a self-contained one ^(~44 MB, one time^)...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap_python.ps1" -Dest "%~dp0.python" -PyVersion 3.12
  if exist ".python\python.exe" set "PY=.python\python.exe"
)

REM 4) Download failed (offline?) -> fall back to ANY Python so the CORE app can
REM    still run; the optional ML extras just won't install on 3.13+.
if not defined PY (
  set "PY_SUPPORTED=0"
  echo [!] Could not obtain CPython 3.10-3.12 automatically ^(offline?^). The core app
  echo     can still run on any Python, but the OPTIONAL ML extras -- face-similarity
  echo     scoring and background masks -- won't install. Get 3.12 at https://www.python.org/downloads/
  echo.
  where py >nul 2>nul && set "PY=py -3"
  if not defined PY ( where python >nul 2>nul && set "PY=python" )
)

REM 5) Truly no Python anywhere and no download -> can't build a venv, stop.
if not defined PY (
  echo Python 3.10-3.12 is required and the automatic download failed.
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

REM --- Health check: a .venv can EXIST and be on a supported Python yet still be
REM     broken. An "unzip the new release over the same folder" update leaves .venv
REM     untouched, so a half-written site-packages (interrupted install, an extra
REM     that clobbered a core dep) survives every update -- and `pip -r` below can't
REM     repair a package it still thinks is satisfied. If an EXISTING venv can't
REM     import the core stack, rebuild it from scratch. Only worthwhile when a
REM     supported Python is available. NOTE: a Pillow-only MIX (Image.py 12 + old
REM     plugin) still imports here and is repaired at app boot by
REM     bootstrap_dependencies.ensure_pillow_consistent -- no need to nuke for that.
if "%REBUILD%"=="0" if "%PY_SUPPORTED%"=="1" (
  "%VPY%" -c "import flask, werkzeug, sqlalchemy, PIL.Image" >nul 2>nul
  if errorlevel 1 (
    echo [i] Existing .venv is present but its core packages won't import -- rebuilding it.
    set "REBUILD=1"
  )
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
