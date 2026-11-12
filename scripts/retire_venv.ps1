<#
.SYNOPSIS
  Set an existing virtual environment aside before start.bat rebuilds it.
.DESCRIPTION
  Waits up to -TimeoutSec until no running process executes from the venv (the
  app that was just stopped, a restart helper), then renames the folder to
  -Backup. A rename moves the whole folder or nothing, so the environment is
  never left half deleted: start.bat builds the new one in its place and puts
  this one back if that fails, and deletes it once the new one is installed.
  A leftover -Backup from an earlier interrupted rebuild is removed first.

  Exit 0: set aside, or there was nothing to set aside.
  Exit 2: still in use, or the rename was refused. Nothing was changed.
#>
param(
  [Parameter(Mandatory = $true)][string]$Venv,
  [Parameter(Mandatory = $true)][string]$Backup,
  [int]$TimeoutSec = 60
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Venv)) { exit 0 }
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
  if (Test-Path -LiteralPath $Backup) { Remove-Item -LiteralPath $Backup -Recurse -Force }
  Rename-Item -LiteralPath $Venv -NewName (Split-Path -Leaf $Backup)
} catch {
  Write-Host "[!] Could not set .venv aside: $($_.Exception.Message)"
  exit 2
}
exit 0
