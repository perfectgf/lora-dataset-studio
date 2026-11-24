# Prepare the official NVIDIA portable in a new folder on Windows.
[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PWD 'ComfyUI-portable'),
    [ValidateSet('Auto', 'Cuda13', 'Cuda126', 'Arm64')][string]$Build = 'Auto'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = 'https://lds.perfectgf.fr/models/tools'
$packages = @{
    Cuda13 = @('ComfyUI_windows_portable_nvidia.7z', '6a5255c7ec01fdc54bdbd8e4507073cf6acda2fbeea74ae1829a50681af5fac6', 2085169944)
    Cuda126 = @('ComfyUI_windows_portable_nvidia_cu126.7z', '06ef2069702a7d7db7262e29a70ec524207f86c7bc4212f9dfcb7c977bf3e9ed', 1976423950)
    Arm64 = @('ComfyUI_windows_portable_nvidia_arm64.7z', '952b8fd26fc15fbe9a9447aea85d11562d857b0a1b0aee5ebd74d86fa927ebb4', 1926914384)
}
function Get-Tool($specification, $directory) {
    $name, $hash, $length = $specification
    $path = Join-Path $directory $name
    Write-Host "Downloading $name from LDS..."
    Invoke-WebRequest -Uri "$root/$hash/$name" -OutFile $path -UseBasicParsing -MaximumRedirection 0
    if ((Get-Item -LiteralPath $path).Length -ne $length -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $hash) {
        throw "The downloaded $name does not match the official release."
    }
    return $path
}
if ($env:OS -ne 'Windows_NT') { throw 'This portable is for Windows with an NVIDIA GPU.' }
if ($Build -eq 'Auto') {
    $status = (& nvidia-smi | Out-String)
    if ($LASTEXITCODE -ne 0 -or $status -notmatch 'CUDA Version:\s*(\d+\.\d+)') {
        throw 'An NVIDIA driver that supports CUDA 12.6 or later is required.'
    }
    $cuda = [version]$Matches[1]
    if ($cuda -lt [version]'12.6') { throw 'Update the NVIDIA driver to support CUDA 12.6 or later.' }
    $caps = @(& nvidia-smi --query-gpu=compute_cap --format=csv,noheader | ForEach-Object { [version]$_.Trim() })
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') {
        if ($cuda -lt [version]'13.0') { throw 'Windows ARM64 requires an NVIDIA driver supporting CUDA 13.' }
        $Build = 'Arm64'
    } elseif ($cuda -ge [version]'13.0' -and -not ($caps | Where-Object { $_ -lt [version]'7.5' })) {
        $Build = 'Cuda13'
    } else {
        if ($caps | Where-Object { $_ -ge [version]'12.0' }) { throw 'This GPU requires an NVIDIA driver supporting CUDA 13.' }
        $Build = 'Cuda126'
    }
}
$dest = [IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $dest) { throw "Choose a new destination. $dest already exists." }
$parent = (Resolve-Path -LiteralPath (Split-Path -Parent $dest)).ProviderPath
$stage = Join-Path $parent ('.lds-comfyui-' + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path $stage
try {
    $exe = Get-Tool @('7zr.exe', 'e80db2dad715a6c5adabe292d31ed54e11f4e9b99ca349827b50573b175891bc', 600064) $stage
    $archive = Get-Tool $packages[$Build] $stage
    $extract = Join-Path $stage 'extract'
    $null = New-Item -ItemType Directory -Path $extract
    $listing = (& $exe l -slt -ba -sccUTF-8 $archive | Out-String)
    if ($LASTEXITCODE -ne 0) { throw 'The verified archive could not be listed.' }
    $excluded = @()
    [long]$size = 0
    foreach ($entry in [regex]::Split($listing.Trim(), '\r?\n\s*\r?\n')) {
        if ($entry -notmatch '(?m)^Path = (.+?)\r?$') { throw 'Unexpected archive entry.' }
        $member = $Matches[1]
        if ($member -match '(^[\\/]|:|(^|[\\/])\.\.([\\/]|$))' -or
            $entry -match '(?m)^(Symbolic Link|Hard Link|Reparse Point|Alternate Stream) = \S' -or
            $entry -match '(?m)^Attributes = \S*L') { throw 'Unexpected path or link in the portable archive.' }
        if ($member -match '(?i)(^|[\\/])ComfyUI[\\/]models[\\/].*\.(safetensors|ckpt|pt|pth|gguf|onnx|bin)$') {
            $excluded += "-x!$member"
        } elseif ($entry -match '(?m)^Size = (\d+)') { $size += [long]$Matches[1] }
    }
    if ($excluded.Count -ne 6 -or $size -le 0 -or $size -gt 24GB) { throw 'The archive differs from the reviewed portable layout.' }
    $volume = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($stage))
    if ($volume.AvailableFreeSpace -lt ($size + 256MB)) { throw 'Insufficient free space to extract ComfyUI.' }
    & $exe x -y "-o$extract" @excluded $archive | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Extraction failed.' }
    $folders = @((Get-Item -LiteralPath $extract); (Get-ChildItem -LiteralPath $extract -Directory))
    $portable = @($folders | Where-Object {
        (Test-Path -LiteralPath (Join-Path $_.FullName 'ComfyUI/main.py') -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $_.FullName 'python_embeded/python.exe') -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $_.FullName 'run_nvidia_gpu.bat') -PathType Leaf)
    })
    if ($portable.Count -ne 1) { throw 'Missing portable launcher, ComfyUI or embedded Python.' }
    [IO.Directory]::Move($portable[0].FullName, $dest)
    Write-Host "Prepared $dest. Run run_nvidia_gpu.bat there. In LDS Setup select its ComfyUI folder and http://127.0.0.1:8188."
} finally {
    $scratch = (Resolve-Path -LiteralPath $stage).ProviderPath
    if ([IO.Path]::GetDirectoryName($scratch) -eq $parent -and [IO.Path]::GetFileName($scratch) -match '^\.lds-comfyui-[0-9a-f]{32}$') {
        Remove-Item -LiteralPath $scratch -Recurse -Force
    }
}
