<#
  bootstrap_python.ps1 — fetch a self-contained CPython when the machine has no
  CPython 3.11-3.12 that start.bat can build its .venv on.

  Downloads the official `install_only` build from astral-sh/python-build-standalone
  (the runtime source used by the Windows start.bat flow), verifies its SHA-256 against
  the GitHub asset digest, and extracts it to <Dest> so <Dest>\python.exe is a ready
  3.12 interpreter — no system install, no admin rights, nothing added to PATH.

  Idempotent: a valid existing <Dest>\python.exe (already 3.11-3.12) is reused, so
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
  $ok = & $exe -c "import sys; print(1 if (3,11)<=sys.version_info[:2]<=(3,12) else 0)" 2>$null
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
  # Hashed through .NET, not Get-FileHash. In Windows PowerShell 5.1 Get-FileHash is a
  # function of the Microsoft.PowerShell.Utility module, loaded from PSModulePath on
  # first use. Run from a PowerShell 7 prompt, cmd hands this script PowerShell 7's
  # PSModulePath: 5.1 finds PowerShell 7's copy of that module first, cannot load it,
  # and calls Get-FileHash "not recognized" (issue #77). Any other module function
  # would fail the same way, so keep this script to compiled cmdlets and .NET calls.
  $sha = [System.Security.Cryptography.SHA256]::Create()
  $stream = [System.IO.File]::OpenRead($tmp)
  try { $got = [System.BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '').ToLower() }
  finally { $stream.Dispose(); $sha.Dispose() }
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
