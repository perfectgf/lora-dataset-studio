@echo off
setlocal
REM ============================================================================
REM  One-click build of the portable Windows bundle (launcher .exe + .zip).
REM
REM  Just double-click this file. It wraps build_portable.ps1 so you don't have
REM  to open PowerShell or fiddle with the execution policy.
REM
REM  Requires on THIS (build) machine: a host Python 3.9-3.12 on PATH -- used
REM  only to run PyInstaller. The end user of the resulting bundle needs nothing.
REM
REM  Output: packaging\dist\LoRA-Dataset-Studio-win64.zip
REM  Any arguments you pass are forwarded to the PowerShell script, e.g.
REM      build.bat -PyVersion 3.12
REM ============================================================================

cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo [X] Python was not found on PATH.
  echo     Install Python 3.11 from https://www.python.org/downloads/ ^(tick
  echo     "Add python.exe to PATH" during setup^), then run this again.
  echo.
  pause
  exit /b 1
)

echo ==^> Building the portable bundle. The first run downloads a standalone
echo     Python and may take a few minutes...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_portable.ps1" %*
if errorlevel 1 (
  echo.
  echo [X] Build FAILED -- read the messages above for the cause.
  echo.
  pause
  exit /b 1
)

echo.
echo [OK] Done. Your bundle is here:
echo     %~dp0dist\LoRA-Dataset-Studio-win64.zip
echo.
echo     Opening the output folder...
start "" "%~dp0dist"
pause
