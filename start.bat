@echo off
setlocal
cd /d "%~dp0"
REM Force UTF-8 so pip reads requirements*.txt as UTF-8 regardless of the system
REM locale. On a GBK / other non-UTF-8 Windows locale, older pip decodes the file
REM with the ANSI codepage and a non-ASCII comment byte crashes the install
REM (UnicodeDecodeError). Reported by the community. Requirements comments are
REM also kept ASCII-only as a second line of defence.
set "PYTHONUTF8=1"

REM --- Pick a CPython 3.11-3.12. The core needs 3.11 or newer (Pillow 13 and
REM     filelock 4.2), and the OPTIONAL ML extras -- insightface / numpy<2 /
REM     onnxruntime -- publish NO wheels for 3.13+, so a venv built on e.g. 3.14
REM     cannot install requirements-ml.txt (pip falls back to source builds that
REM     fail). `py -3.x` selects an EXACT version; bare `py -3` or `python` grab
REM     the NEWEST installed one -- which is exactly the trap.
set "PY="
set "PY_SUPPORTED=1"

REM     Every interpreter started below, except the venv's own python.exe, goes
REM     through CALL: `py` or `python` can be a batch-file shim (pyenv-win writes
REM     python.bat, other version managers python.cmd), and cmd never comes back
REM     from a batch file started without CALL. Issue #77: start.bat ended inside
REM     the check in 1b, with no message and no app.

REM 1) A CPython 3.11-3.12 already installed (via the Windows py launcher)?
for %%V in (3.12 3.11) do (
  if not defined PY ( call py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V" )
)

REM 1b) ... or a `python` on PATH that IS a 3.11-3.12 itself: python.org installs
REM     made without the py launcher, scoop, pyenv-win, conda. Issue #77: such a 3.11
REM     was skipped for a download, then used anyway once the download failed. Its
REM     version is checked, so a newer default `python` -- the 3.14 trap above --
REM     still can't slip through.
if not defined PY (
  call python -c "import sys; sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>nul && set "PY=python"
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
  echo [i] No CPython 3.11-3.12 found -- downloading a self-contained one ^(~44 MB, one time^)...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap_python.ps1" -Dest "%~dp0.python" -PyVersion 3.12
  if exist ".python\python.exe" set "PY=.python\python.exe"
)

REM 4) Download failed (offline?) -> fall back to any Python 3.11 or newer so the
REM    CORE app can still run; the optional ML extras just won't install on 3.13+.
REM    Not 3.10 any more: this version's requirements cannot install on it.
if not defined PY (
  set "PY_SUPPORTED=0"
  echo [!] Could not obtain CPython 3.11-3.12 automatically ^(offline?^). The core app
  echo     can still run on a newer Python, but the OPTIONAL ML extras -- face-similarity
  echo     scoring and background masks -- won't install. Get 3.12 at https://www.python.org/downloads/
  echo.
  call py -3 -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,11) else 1)" >nul 2>nul && set "PY=py -3"
  if not defined PY ( call python -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,11) else 1)" >nul 2>nul && set "PY=python" )
)

REM 5) No Python 3.11+ anywhere and no download -> can't build a venv, stop.
REM    A double-clicked window closes the moment this script ends: from here on,
REM    every stop says why and waits for a key, so the reason stays readable.
if not defined PY (
  echo Python 3.11 or newer is required ^(3.12 recommended^) and the automatic download failed.
  echo Install Python 3.12 from https://www.python.org/downloads/ then re-run start.bat
  pause
  exit /b 1
)

REM --- Call the venv's python DIRECTLY everywhere below -- never `activate.bat`.
REM     A venv carries hardcoded paths; if it was ever moved or copied (e.g. a
REM     provisioning step that builds it elsewhere then relocates it), its
REM     activate.bat no longer shadows the system python, so `call activate` +
REM     `python` silently falls back to the machine default (the 3.14 trap). The
REM     python.exe stub, by contrast, always resolves to the venv interpreter.
set "VPY=.venv\Scripts\python.exe"

REM --- Ensure .venv exists AND runs a supported Python; rebuild it otherwise.
REM     A venv older than 3.11 (one built on 3.10 by an earlier version) cannot run
REM     this version at all; a stale venv on 3.13+ can't install the ML extras.
REM     Rebuild when that helps: onto 3.11-3.12, or off a pre-3.11 venv onto the
REM     3.11+ fallback found above -- rebuilding a 3.14 venv on 3.14 gains nothing.
set "REBUILD=0"
set "VENV_BAD=0"
set "VENV_OLD=0"
if not exist "%VPY%" set "REBUILD=1"
if "%REBUILD%"=="0" (
  "%VPY%" -c "import sys; sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>nul
  if errorlevel 1 set "VENV_BAD=1"
  "%VPY%" -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,11) else 1)" >nul 2>nul
  if errorlevel 1 set "VENV_OLD=1"
)
if "%VENV_BAD%"=="1" if "%PY_SUPPORTED%"=="1" (
  echo [i] Existing .venv is not on Python 3.11-3.12 -- rebuilding it.
  set "REBUILD=1"
)
if "%VENV_OLD%"=="1" if "%PY_SUPPORTED%"=="0" (
  echo [i] Existing .venv runs a Python older than 3.11, which this version needs -- rebuilding it on %PY%.
  set "REBUILD=1"
)
if "%VENV_BAD%"=="1" if "%VENV_OLD%"=="0" if "%PY_SUPPORTED%"=="0" (
  echo [!] .venv runs an unsupported Python and no CPython 3.11-3.12 is installed.
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

REM --- Replace a venv without ever leaving a half-built one. The old folder is set
REM     aside whole: scripts\retire_venv.ps1 waits for the app that used it to exit
REM     (an in-app update restarts it from that venv), then renames it. The new one
REM     is built in its place; if creating it or installing into it fails, the old
REM     one comes back unchanged. It is deleted only once the new one is installed.
set "SET_ASIDE=0"
if "%REBUILD%"=="1" (
  if exist .venv (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\retire_venv.ps1" -Venv "%~dp0.venv" -Backup "%~dp0.venv-previous"
    if errorlevel 1 (
      echo.
      echo [!] The old .venv is still in use, so it was left as it is. Close every
      echo     LoRA Dataset Studio window, then run start.bat again.
      pause
      exit /b 1
    )
    set "SET_ASIDE=1"
  )
  call %PY% -m venv .venv || (
    echo.
    echo [!] Could not create .venv with %PY% -- see the messages above.
    if exist .venv-previous (
      if exist .venv rmdir /s /q .venv
      ren .venv-previous .venv && echo     The previous .venv was put back unchanged.
    )
    pause
    exit /b 1
  )
)

"%VPY%" -m pip install -q -r backend\requirements.txt || (
  echo.
  echo [!] Installing the backend requirements failed -- see the pip messages above.
  if "%SET_ASIDE%"=="1" if exist .venv-previous (
    if exist .venv rmdir /s /q .venv
    ren .venv-previous .venv && echo     The previous .venv was put back unchanged.
  )
  pause
  exit /b 1
)
if exist .venv-previous rmdir /s /q .venv-previous
if not exist frontend\dist\index.html (
  echo frontend\dist is missing -- this repo ships it prebuilt. Run: cd frontend ^&^& npm install ^&^& npm run build
  pause
  exit /b 1
)
rem Port 5000 is a frequent collision (macOS AirPlay, another local Flask app).
rem Use 5050 by default; backend/run.py advances automatically if it is occupied.
if not defined LDS_PORT set "LDS_PORT=5050"
set "LDS_OPEN_BROWSER=1"
rem Launched through the supervisor, not directly: a native crash (an access
rem violation in a C extension, or an antivirus hook faulting inside one) kills
rem the interpreter outright, and the app used to stay down until someone
rem double-clicked this file again. The supervisor brings it back, says so, and
rem gives up after a few deaths in a row so a backend broken at boot cannot
rem become an endless respawn loop. LDS_SUPERVISE=0 opts out.
if "%LDS_SUPERVISE%"=="0" (
  "%VPY%" backend\run.py
) else (
  "%VPY%" backend\supervise.py
)
rem Compared as text: a crash code such as 0xC0000005 is negative to `if errorlevel`.
if not "%errorlevel%"=="0" (
  echo.
  echo [!] LoRA Dataset Studio stopped with an error -- see the messages above.
  pause
  exit /b %errorlevel%
)
