#requires -Version 5.1
<#
.SYNOPSIS
  Assemble the portable Windows bundle of LoRA Dataset Studio.

.DESCRIPTION
  Produces packaging/dist/LoRA-Dataset-Studio-win64.zip: a self-contained folder the
  end user extracts and runs by double-clicking "LoRA Dataset Studio.exe" — no Python
  install, no terminal. The heavy externals (ComfyUI, ai-toolkit, Ollama, ML extras)
  stay OUT of the bundle: the in-app Setup wizard installs them. This is why we ship a
  real standalone CPython (which HAS pip) instead of a single frozen exe — the wizard's
  `pip install -r backend/requirements-ml.txt` runs against the bundled interpreter.

  Bundle layout (mirrors the repo so backend/config.py resolves REPO_ROOT/FRONTEND_DIST
  unchanged):
      LoRA Dataset Studio.exe   python\   backend\   frontend\dist\   icon.ico   README.md

.NOTES
  Prereqs on the BUILD machine: PowerShell 5.1+, a host `python` (3.9-3.12) on PATH
  (only used to run PyInstaller for the launcher), tar.exe (built into Windows 10+),
  and internet access. The end user needs none of this.

  Distribute the resulting .zip as a GitHub Release asset. Unsigned: SmartScreen will
  warn "unknown publisher" (More info -> Run anyway). Code-signing is a later add-on.
#>
[CmdletBinding()]
param(
  [string]$PyVersion = '3.11',
  [string]$OutName   = 'LoRA-Dataset-Studio'
)
$ErrorActionPreference = 'Stop'
$Here  = $PSScriptRoot
$Root  = Split-Path -Parent $Here
$Build = Join-Path $Here 'build'
$Stage = Join-Path $Here "dist\$OutName"
$Zip   = Join-Path $Here "dist\$OutName-win64.zip"
$UA    = @{ 'User-Agent' = 'lds-build' }

function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }

Step 'Clean workspace'
Remove-Item -Recurse -Force $Build, $Stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $Build, $Stage, (Split-Path $Zip) | Out-Null

# 1) Resolve a standalone CPython asset from the LATEST release (no hardcoded tag, so
#    it never goes stale). install_only = a normal, relocatable python\ layout with pip.
Step "Resolving python-build-standalone (CPython $PyVersion, x86_64 msvc)"
$rel = Invoke-RestMethod -Headers $UA 'https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest'
$pattern = "cpython-$([regex]::Escape($PyVersion))\.\d+\+.*-x86_64-pc-windows-msvc-install_only\.tar\.gz$"
$asset = $rel.assets | Where-Object { $_.name -match $pattern } | Select-Object -First 1
if (-not $asset) { throw "No CPython $PyVersion install_only msvc asset in release $($rel.tag_name)." }
$tar = Join-Path $Build $asset.name
Write-Host "    $($asset.name)"
Invoke-WebRequest -Headers $UA -Uri $asset.browser_download_url -OutFile $tar

Step 'Extracting Python into the bundle'
tar -xzf $tar -C $Build                       # -> $Build\python\...
$Py = Join-Path $Stage 'python'
Move-Item (Join-Path $Build 'python') $Py
$PyExe = Join-Path $Py 'python.exe'

# 2) Runtime deps (core requirements minus pytest) into the SHIPPED interpreter.
Step 'Installing runtime deps into the bundle'
& $PyExe -m pip --version *> $null
if ($LASTEXITCODE -ne 0) { & $PyExe -m ensurepip --default-pip }
$reqRun = Join-Path $Build 'requirements-runtime.txt'
Get-Content (Join-Path $Root 'backend\requirements.txt') |
  Where-Object { $_ -notmatch '^\s*pytest' } | Set-Content $reqRun
& $PyExe -m pip install --no-warn-script-location --disable-pip-version-check -r $reqRun
if ($LASTEXITCODE -ne 0) { throw 'pip install of runtime deps failed.' }

# 3) App files — mirror the repo so REPO_ROOT/FRONTEND_DIST resolve unchanged.
Step 'Staging app files'
robocopy (Join-Path $Root 'backend') (Join-Path $Stage 'backend') /E `
  /XD __pycache__ tests .venv /XF *.pyc | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy backend failed ($LASTEXITCODE)." }
$global:LASTEXITCODE = 0
New-Item -ItemType Directory -Force (Join-Path $Stage 'frontend') | Out-Null
Copy-Item -Recurse -Force (Join-Path $Root 'frontend\dist') (Join-Path $Stage 'frontend\dist')
Copy-Item -Force (Join-Path $Here 'icon.ico') $Stage
Copy-Item -Force (Join-Path $Root 'README.md') $Stage -ErrorAction SilentlyContinue
Copy-Item -Force (Join-Path $Root 'LICENSE')   $Stage -ErrorAction SilentlyContinue

# 4) Launcher exe (host python + PyInstaller; tkinter is bundled automatically).
#    PyInstaller needs CPython 3.9-3.12 — bare `python` may resolve to a newer one
#    (the exact trap start.bat dodges for the ML extras), so resolve a compatible
#    host interpreter through the py launcher first, newest supported first.
Step 'Building the launcher exe (PyInstaller)'
$HostPy = $null
foreach ($v in '3.12', '3.11', '3.10', '3.9') {
  $exe = & py "-$v" -c 'import sys; print(sys.executable)' 2>$null
  if ($LASTEXITCODE -eq 0 -and $exe) { $HostPy = $exe.Trim(); break }
}
if (-not $HostPy) { $HostPy = 'python' }   # last resort — may fail on 3.13+
Write-Host "    host python for PyInstaller: $HostPy"
& $HostPy -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) { & $HostPy -m pip install pyinstaller }
& $HostPy -m PyInstaller --noconfirm --onefile --noconsole `
  --name 'LoRA Dataset Studio' --icon (Join-Path $Here 'icon.ico') `
  --distpath (Join-Path $Build 'launcher') --workpath (Join-Path $Build 'pyi') `
  --specpath $Build (Join-Path $Here 'launcher.py')
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
Copy-Item -Force (Join-Path $Build 'launcher\LoRA Dataset Studio.exe') $Stage

# 5) Zip the folder (extraction yields the LoRA-Dataset-Studio\ folder).
Step 'Zipping the bundle'
Remove-Item -Force $Zip -ErrorAction SilentlyContinue
Compress-Archive -Path $Stage -DestinationPath $Zip
$mb = [math]::Round((Get-Item $Zip).Length / 1MB, 1)
Step "Done -> $Zip ($mb MB)"
Write-Host '    Test it: extract the zip and double-click "LoRA Dataset Studio.exe".'
