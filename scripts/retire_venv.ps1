<#
.SYNOPSIS
  Set an existing virtual environment aside before start.bat rebuilds it.
.DESCRIPTION
  Rechecks first, whatever led start.bat here: a venv that runs Python 3.11-3.12
  and imports the core packages is kept as it is (a start.bat that an update
  replaced while cmd was running it can resume in the middle of the new one).

  Otherwise waits up to -TimeoutSec until no running process executes from the
  venv (the app that was just stopped, a restart helper), then renames the folder
  to -Backup. A rename moves the whole folder or nothing, so the environment is
  never left half deleted: start.bat builds the new one in its place and puts this
  one back if that fails, and deletes it once the new one is installed.

  The last usable environment is never deleted. A -Backup left by an earlier run is
  removed only when the venv being set aside is itself usable (pyvenv.cfg and
  Scripts\python.exe). When the venv is the leftover of an interrupted rebuild, the
  leftover is removed instead and the -Backup stays the environment set aside.

  Exit 0: set aside, or nothing to set aside.
  Exit 2: still in use, or a rename/removal was refused. Nothing usable was changed.
  Exit 3: healthy, kept as it is. Nothing to rebuild.
#>
param(
  [Parameter(Mandatory = $true)][string]$Venv,
  [Parameter(Mandatory = $true)][string]$Backup,
  [int]$TimeoutSec = 60
)
$ErrorActionPreference = 'Stop'

function Test-Usable([string]$Path) {
  (Test-Path -LiteralPath (Join-Path $Path 'pyvenv.cfg')) -and
    (Test-Path -LiteralPath (Join-Path $Path 'Scripts\python.exe'))
}

function Test-Healthy([string]$Path) {
  if (-not (Test-Usable $Path)) { return $false }
  $python = Join-Path $Path 'Scripts\python.exe'
  $previous = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    & $python -c "import sys; sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,12) else 1)" *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $python -c "import flask, werkzeug, sqlalchemy, PIL.Image" *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  } finally {
    $ErrorActionPreference = $previous
  }
}

if (-not (Test-Path -LiteralPath $Venv)) { exit 0 }
if (Test-Healthy $Venv) {
  Write-Host '[i] .venv runs Python 3.11-3.12 and imports the core packages: kept as it is, not rebuilt.'
  exit 3
}
$prefix = (Resolve-Path -LiteralPath $Venv).Path.TrimEnd('\') + '\'

function Get-VenvProcess {
  @(Get-CimInstance Win32_Process | Where-Object {
      $_.ExecutablePath -and $_.ExecutablePath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
    })
}

$deadline = (Get-Date).AddSeconds($TimeoutSec)
$announced = $false
while ($true) {
  $running = Get-VenvProcess
  if ($running.Count -eq 0) { break }
  if ((Get-Date) -gt $deadline) {
    Write-Host "[!] Still running from .venv: process $(($running | ForEach-Object { $_.ProcessId }) -join ', ')."
    exit 2
  }
  if (-not $announced) {
    Write-Host '[i] Waiting for the previous LoRA Dataset Studio process to exit...'
    $announced = $true
  }
  Start-Sleep -Seconds 1
}

try {
  if (Test-Path -LiteralPath $Backup) {
    if (-not (Test-Usable $Venv)) {
      # What an interrupted rebuild left behind; -Backup is the last usable one.
      Remove-Item -LiteralPath $Venv -Recurse -Force
      Write-Host '[i] .venv was left incomplete by an interrupted rebuild; the previous environment stays set aside in .venv-previous.'
      exit 0
    }
    Remove-Item -LiteralPath $Backup -Recurse -Force    # older leftover; .venv is usable
  }
  Rename-Item -LiteralPath $Venv -NewName (Split-Path -Leaf $Backup)
} catch {
  Write-Host "[!] Could not set .venv aside: $($_.Exception.Message)"
  exit 2
}
exit 0
