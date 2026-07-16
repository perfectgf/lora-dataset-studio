#requires -Version 5.1
<#
.SYNOPSIS
  Build the supported Windows release archive.

.DESCRIPTION
  Produces packaging/dist/LoRA-Dataset-Studio-windows.zip. The archive contains
  the application source, the tracked frontend build and start.bat. It contains
  no executable launcher and no embedded Python runtime: start.bat discovers or
  downloads a compatible CPython on first launch, then creates a local .venv.
#>
[CmdletBinding()]
param(
  [string]$OutName = 'LoRA-Dataset-Studio-windows'
)

$ErrorActionPreference = 'Stop'
$Here = $PSScriptRoot
$Root = Split-Path -Parent $Here
$Build = Join-Path $Here 'build\release'
$Stage = Join-Path $Build $OutName
$Dist = Join-Path $Here 'dist'
$Zip = Join-Path $Dist "$OutName.zip"

function Step($message) { Write-Host "==> $message" -ForegroundColor Cyan }

Step 'Preparing the Windows source bundle'
Remove-Item -Recurse -Force $Build -ErrorAction SilentlyContinue
Remove-Item -Force $Zip -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $Stage, $Dist | Out-Null

# Mirror only the runtime application. In particular, do not copy a developer
# venv, node_modules, tests, packaging tools, or any locally-built launcher.
robocopy (Join-Path $Root 'backend') (Join-Path $Stage 'backend') /E `
  /XD __pycache__ tests .venv venv /XF *.pyc *.pyo | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy backend failed ($LASTEXITCODE)." }
$global:LASTEXITCODE = 0

New-Item -ItemType Directory -Force (Join-Path $Stage 'frontend') | Out-Null
Copy-Item -Recurse -Force (Join-Path $Root 'frontend\dist') (Join-Path $Stage 'frontend\dist')

New-Item -ItemType Directory -Force (Join-Path $Stage 'scripts') | Out-Null
Copy-Item -Force (Join-Path $Root 'scripts\bootstrap_python.ps1') (Join-Path $Stage 'scripts')

foreach ($file in @('start.bat', 'README.md', 'LICENSE', 'config.example.json', '.env.example')) {
  Copy-Item -Force (Join-Path $Root $file) $Stage
}

# This is both a local safety check and a release invariant. The CI policy guard
# independently inspects the final ZIP before GitHub receives it.
$executables = @(Get-ChildItem -Path $Stage -Recurse -File -Filter '*.exe')
if ($executables.Count -gt 0) {
  $executables | ForEach-Object { Write-Host $_.FullName }
  throw 'Windows release bundle contains a forbidden executable.'
}

Step 'Creating the release ZIP'
Compress-Archive -Path $Stage -DestinationPath $Zip -CompressionLevel Optimal
$sizeMb = [math]::Round((Get-Item $Zip).Length / 1MB, 1)
Step "Done -> $Zip ($sizeMb MB)"
Write-Host '    Extract the ZIP, then double-click start.bat.'
