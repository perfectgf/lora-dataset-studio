@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Portable one-click updater for the Docker installation.
rem Double-click (or no argument) follows the latest published GitHub Release.
rem Pass "v2" to follow the maintained branch; "main" is a compatibility alias.
rem Pass "patreon" to install the latest Patreon release (active membership).
rem After a Patreon release the public channels refuse; a second argument
rem "force" installs the public release anyway: update-docker-gpu.bat stable force
set "LDS_UPDATE_REQUEST=%~1"
set "LDS_UPDATE_EXTRA=%~2"
set "LDS_UPDATE_FORCE="
setlocal EnableDelayedExpansion
if /I "!LDS_UPDATE_EXTRA!"=="force" (
    endlocal
    set "LDS_UPDATE_FORCE=-ForcePublic"
    setlocal EnableDelayedExpansion
) else if not "!LDS_UPDATE_EXTRA!"=="" (
    endlocal
    echo [ERROR] Too many arguments.
    echo Usage : update-docker-gpu.bat [stable^|v2^|main^|patreon] [force]
    pause
    exit /b 2
)
if "!LDS_UPDATE_REQUEST!"=="" (
    endlocal
    goto :channel_stable
)
if /I "!LDS_UPDATE_REQUEST!"=="stable" (
    endlocal
    goto :channel_stable
)
if /I "!LDS_UPDATE_REQUEST!"=="main" (
    endlocal
    goto :channel_main
)
if /I "!LDS_UPDATE_REQUEST!"=="v2" (
    endlocal
    goto :channel_v2
)
if /I "!LDS_UPDATE_REQUEST!"=="patreon" (
    endlocal
    goto :channel_patreon
)
endlocal
echo [ERROR] Unknown channel.
echo Usage : update-docker-gpu.bat [stable^|v2^|main^|patreon] [force]
pause
exit /b 2

:channel_main
set "LDS_UPDATE_CHANNEL=main"
goto :channel_ready

:channel_v2
set "LDS_UPDATE_CHANNEL=v2"
goto :channel_ready

:channel_patreon
set "LDS_UPDATE_CHANNEL=patreon"
goto :channel_ready

:channel_stable
set "LDS_UPDATE_CHANNEL=stable"

:channel_ready

pushd "%~dp0" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Cannot open the installation folder.
    pause
    exit /b 1
)

rem Keep the whole tail in one parsed block. The updater is then free to replace
rem this BAT while it is running without making cmd.exe read half of a new file.
rem Only the last command of the block may exit: an exit /b in a nested block
rem that further commands follow left the process exit code at 0, so a caller
rem saw a refused update as a success (2026-10-18). Both flags start empty, so
rem only the code PowerShell returns inside the block can set them.
set "LDS_UPDATE_CURRENT="
set "LDS_UPDATE_STARTED="
(
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\update-docker-gpu.ps1" -Channel "%LDS_UPDATE_CHANNEL%" -InstallRoot "%~dp0." %LDS_UPDATE_FORCE%
    if errorlevel 3 if not errorlevel 4 set "LDS_UPDATE_CURRENT=1"
    if errorlevel 0 if not errorlevel 1 set "LDS_UPDATE_STARTED=1"
    popd
    echo.
    if defined LDS_UPDATE_CURRENT (
        echo LDS is already up to date. Nothing was changed.
        pause
        exit /b 0
    ) else if defined LDS_UPDATE_STARTED (
        echo Rebuild started. Check that the application becomes reachable.
        pause
        exit /b 0
    ) else (
        echo The update did not complete. Your local data was preserved.
        pause
        exit /b 1
    )
)
