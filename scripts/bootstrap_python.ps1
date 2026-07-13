<#
  bootstrap_python.ps1 — fetch a self-contained CPython when the machine has no
  CPython 3.10-3.12 that start.bat can build its .venv on.

  Downloads the official `install_only` build from astral-sh/python-build-standalone
  (the same source the old portable bundle embedded), verifies its SHA-256 against
  the GitHub asset digest, and extracts it to <Dest> so <Dest>\python.exe is a ready
  3.12 interpreter — no system install, no admin rights, nothing added to PATH.

  Idempotent: a valid existing <Dest>\python.exe (already 3.10-3.12) is reused, so
  a second run is instant. Windows x86_64 only; needs PowerShell 5.1+ and the
  built-in tar.exe (Windows 10 1803+). Exits non-zero on any failure so the caller
  (start.bat) can fall back cleanly.
#>
[CmdletBinding()]
param(
  [string]$Dest = ".python",
  [string]$PyVersion = "3.12"
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'                 # the IWR progress bar is very slow
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$UA = @{ 'User-Agent' = 'lora-dataset-studio-bootstrap' }

$exe = Join-Path $Dest 'python.exe'
if (Test-Path $exe) {
  $ok = & $exe -c "import sys; print(1 if (3,10)<=sys.version_info[:2]<=(3,12) else 0)" 2>$null
  if ($ok -eq '1') { Write-Host "Reusing the standalone Python already at $exe"; exit 0 }
  Remove-Item -Recurse -Force $Dest                      # stale/unsupported -> refetch
}

Write-Host "Resolving a self-contained CPython $PyVersion (python-build-standalone)..."
$rel = Invoke-RestMethod -Headers $UA 'https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest'
$asset = $rel.assets |
  Where-Object { $_.name -like "cpython-$PyVersion.*-x86_64-pc-windows-msvc-install_only.tar.gz" } |
  Select-Object -First 1
if (-not $asset) { throw "No CPython $PyVersion install_only build in the latest python-build-standalone release." }

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) $asset.name
Write-Host "Downloading $($asset.name) (~$([math]::Round($asset.size / 1MB)) MB)..."
Invoke-WebRequest -Headers $UA -Uri $asset.browser_download_url -OutFile $tmp

if ($asset.digest -and $asset.digest -match '^sha256:(.+)$') {
  $want = $Matches[1].ToLower()
  $got = (Get-FileHash -Algorithm SHA256 $tmp).Hash.ToLower()
  if ($got -ne $want) { Remove-Item -Force $tmp -ErrorAction SilentlyContinue; throw "SHA-256 mismatch for $($asset.name)." }
  Write-Host "Checksum OK."
}

New-Item -ItemType Directory -Force $Dest | Out-Null
Write-Host "Extracting..."
# The install_only tarball nests everything under a leading `python/` dir; strip it
# so the interpreter lands exactly at <Dest>\python.exe.
tar -xzf $tmp --strip-components=1 -C $Dest
if ($LASTEXITCODE -ne 0) { throw "tar extraction failed (exit $LASTEXITCODE)." }
Remove-Item -Force $tmp -ErrorAction SilentlyContinue

if (-not (Test-Path $exe)) { throw "Extraction did not produce $exe." }
& $exe --version
Write-Host "Ready: $exe"
