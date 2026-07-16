@echo off
setlocal
REM ============================================================================
REM  One-click build of the supported Windows release ZIP (start.bat + source).
REM
REM  Just double-click this file. It wraps build_release_zip.ps1 so you don't have
REM  to open PowerShell or fiddle with the execution policy.
REM
REM  Output: packaging\dist\LoRA-Dataset-Studio-windows.zip
REM  Any arguments you pass are forwarded to the PowerShell script.
REM ============================================================================

cd /d "%~dp0"

echo ==^> Building the Windows release ZIP...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_release_zip.ps1" %*
if errorlevel 1 (
  echo.
  echo [X] Build FAILED -- read the messages above for the cause.
  echo.
  pause
  exit /b 1
)

echo.
echo [OK] Done. Your release archive is here:
echo     %~dp0dist\LoRA-Dataset-Studio-windows.zip
echo.
echo     Opening the output folder...
start "" "%~dp0dist"
pause
