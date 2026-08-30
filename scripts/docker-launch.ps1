[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('studio', 'gpu')]
    [string] $Stack,

    [switch] $Configure,
    [switch] $Rebuild,
    [switch] $UpdateRebuild
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8NoBom

$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$NonInteractive = [bool] $UpdateRebuild
# Test mode skips Open-Studio, so the browser that carries the Ollama choice
# never opens and the interactive timeout can only expire in full. The wait
# still runs -- a harness or a second window may answer -- but bounded to
# seconds, so a headless relaunch cannot stall for a quarter of an hour.
$OllamaWaitSeconds = if ($env:LDS_TEST_MODE) { 10 } else { 900 }
$OllamaWaitLabel = if ($env:LDS_TEST_MODE) { "$OllamaWaitSeconds seconds" } else { '15 minutes' }
$DockerReady = $false
$BrowserOpened = $false
$DockerExe = $null
$PowerShellExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
if (-not (Test-Path -LiteralPath $PowerShellExe -PathType Leaf)) {
    $PowerShellExe = (Get-Command powershell.exe -ErrorAction Stop).Source
}

$BaseCompose = if ($Stack -eq 'gpu') {
    Join-Path $RepoRoot 'docker-compose.gpu.yml'
}
else {
    Join-Path $RepoRoot 'docker-compose.yml'
}
$StaticExternalOverlay = Join-Path $RepoRoot 'docker-compose.external-comfy.yml'
$LocalExternalOverlay = Join-Path $RepoRoot '.docker-compose.external-comfy.override.yml'
$HostOverlay = Join-Path $RepoRoot 'docker-compose.ollama-host.yml'
$SidecarOverlay = Join-Path $RepoRoot 'docker-compose.ollama-sidecar.yml'
$SidecarGpuOverlay = Join-Path $RepoRoot 'docker-compose.ollama-gpu.yml'
$InspectHelper = Join-Path $PSScriptRoot 'docker-launch-inspect.ps1'
$ComfyHelper = Join-Path $PSScriptRoot 'configure-external-comfy.ps1'
$OllamaModeHelper = Join-Path $PSScriptRoot 'docker-ollama-mode.ps1'

if ($Stack -eq 'gpu') {
    $Project = 'lora-dataset-studio-gpu'
    $Container = 'lora-dataset-studio-gpu'
    $ExpectedMode = 'gpu'
    $DataDirectory = Join-Path $RepoRoot 'data-docker-gpu'
}
else {
    $Project = 'lora-dataset-studio'
    $Container = 'lora-dataset-studio'
    $ExpectedMode = 'external'
    $DataDirectory = Join-Path $RepoRoot 'data-docker'
}
$SidecarContainer = $Project + '-ollama'
$ConfigPath = Join-Path $DataDirectory 'config.json'
$BindAddress = '127.0.0.1'

function Invoke-NativeCapture {
    param(
        [Parameter(Mandatory = $true)]
        [string] $FilePath,
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = @(& $FilePath @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = $output
        Text = (($output | ForEach-Object { [string] $_ }) -join [Environment]::NewLine)
    }
}

function Invoke-Docker {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments,
        [switch] $Quiet
    )

    if ($Quiet) {
        return Invoke-NativeCapture -FilePath $script:DockerExe -Arguments $Arguments
    }
    # Out-Host, not a bare call: anything Docker prints on stdout would otherwise
    # join this function's OUTPUT, making the caller receive an array instead of
    # the status object. Under Set-StrictMode 2.0 the .ExitCode read below then
    # throws "property not found" and hides the real Docker error.
    & $script:DockerExe @Arguments | Out-Host
    return [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Output = @()
        Text = ''
    }
}

function Get-DockerExecutable {
    if ($env:LDS_DOCKER_EXE) {
        $candidate = [System.IO.Path]::GetFullPath($env:LDS_DOCKER_EXE)
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Docker\Docker\resources\bin\docker.exe'),
        (Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker.exe')
    )
    if ($env:ProgramW6432) {
        $candidates += (Join-Path $env:ProgramW6432 'Docker\Docker\resources\bin\docker.exe')
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    $command = Get-Command docker.exe -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }
    throw 'Docker was not found. Install Docker Desktop, then try again.'
}

function Get-DockerDesktopExecutable {
    if ($env:LDS_DOCKER_DESKTOP_EXE) {
        $candidate = [System.IO.Path]::GetFullPath($env:LDS_DOCKER_DESKTOP_EXE)
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Docker\Docker\Docker Desktop.exe'),
        (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe')
    )
    if ($env:ProgramW6432) {
        $candidates += (Join-Path $env:ProgramW6432 'Docker\Docker\Docker Desktop.exe')
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    return $null
}

function Wait-ForDocker {
    $version = Invoke-Docker -Arguments @('compose', 'version') -Quiet
    if ($version.ExitCode -ne 0) {
        throw 'Docker Compose is unavailable. Update Docker Desktop.'
    }

    $info = Invoke-Docker -Arguments @('info') -Quiet
    if ($info.ExitCode -eq 0) {
        $script:DockerReady = $true
        return
    }

    $desktop = Get-DockerDesktopExecutable
    if (-not $desktop) {
        throw 'Open Docker Desktop manually, wait until it is ready, then try again.'
    }
    Write-Host 'Docker is installed but is not running. Starting Docker Desktop...'
    Start-Process -FilePath $desktop -WindowStyle Hidden
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        Start-Sleep -Seconds 2
        $info = Invoke-Docker -Arguments @('info') -Quiet
        if ($info.ExitCode -eq 0) {
            $script:DockerReady = $true
            return
        }
    }
    throw 'Docker Desktop did not become ready within three minutes.'
}

function Convert-HelperOutput {
    param([object[]] $Lines)

    $values = @{}
    foreach ($lineObject in $Lines) {
        $line = [string] $lineObject
        $separator = $line.IndexOf('=')
        if ($separator -le 0) {
            continue
        }
        $values[$line.Substring(0, $separator)] = $line.Substring($separator + 1)
    }
    return $values
}

function Invoke-PowerShellHelper {
    param(
        [Parameter(Mandatory = $true)]
        [string] $ScriptPath,
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments,
        [switch] $Sta
    )

    $nativeArgs = @('-NoLogo', '-NoProfile', '-NonInteractive')
    if ($Sta) {
        $nativeArgs += '-STA'
    }
    $nativeArgs += @('-ExecutionPolicy', 'Bypass', '-File', $ScriptPath)
    $nativeArgs += $Arguments
    $result = Invoke-NativeCapture -FilePath $script:PowerShellExe -Arguments $nativeArgs
    if ($result.ExitCode -ne 0) {
        throw ('A launcher helper failed: ' + [System.IO.Path]::GetFileName($ScriptPath))
    }
    return Convert-HelperOutput -Lines $result.Output
}

function Configure-Or-ValidateExternalComfy {
    if ($Stack -ne 'studio') {
        if ($Configure) {
            throw '--configure is only supported by start-docker.bat.'
        }
        return
    }
    if (-not (Test-Path -LiteralPath $ComfyHelper -PathType Leaf)) {
        throw 'The external ComfyUI helper is missing.'
    }

    if ($Configure) {
        Write-Host ''
        Write-Host 'Select your existing ComfyUI folder (or its portable parent).'
        Write-Host 'Studio mounts it read-write so it can use or install models and write input/output.'
        Write-Host 'Nothing is copied and your usual ComfyUI is never started by this launcher.'
        $result = Invoke-PowerShellHelper -ScriptPath $ComfyHelper -Sta -Arguments @(
            '-Configure', '-OverridePath', $LocalExternalOverlay)
        if ($result['STATE'] -eq 'CANCELLED') {
            throw 'Folder selection was cancelled. Docker was not changed.'
        }
        if ($result['STATE'] -ne 'SAVED') {
            throw 'Select a folder containing main.py and models, or its portable parent.'
        }
    }
    else {
        $result = Invoke-PowerShellHelper -ScriptPath $ComfyHelper -Arguments @(
            '-OverridePath', $LocalExternalOverlay)
        if ($result['STATE'] -eq 'ABSENT') {
            if ($NonInteractive) {
                throw 'No external ComfyUI folder is configured. Run start-docker.bat once.'
            }
            Write-Host ''
            Write-Host 'LoRA Dataset Studio needs the folder of your existing ComfyUI.'
            Write-Host 'It will be mounted read-write; nothing is copied or started.'
            $result = Invoke-PowerShellHelper -ScriptPath $ComfyHelper -Sta -Arguments @(
                '-Configure', '-OverridePath', $LocalExternalOverlay)
            if ($result['STATE'] -eq 'CANCELLED') {
                throw 'Folder selection was cancelled. Docker was not changed.'
            }
        }
        if ($result['STATE'] -notin @('VALID', 'SAVED')) {
            throw 'The saved ComfyUI folder is missing or its generated override was modified. Run start-docker.bat --configure.'
        }
    }
}

function Write-LauncherMarker {
    $markerPath = Join-Path $RepoRoot '.docker-launch-settings'
    $tempPath = $markerPath + '.tmp'
    $contents = 'LAST_LAUNCHER=' + $Stack + [Environment]::NewLine
    try {
        [System.IO.File]::WriteAllText($tempPath, $contents, $utf8NoBom)
        # One call for both the fresh and the overwrite case. [File]::Replace was
        # used here with $null as the backup name: PowerShell binds that to the
        # empty string, so .NET threw "The path is empty" on EVERY launch after
        # the first one -- before Docker was ever contacted.
        Move-Item -LiteralPath $tempPath -Destination $markerPath -Force
    }
    finally {
        if (Test-Path -LiteralPath $tempPath -PathType Leaf) {
            Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Get-OllamaMode {
    param([switch] $Wait)

    if (-not (Test-Path -LiteralPath $OllamaModeHelper -PathType Leaf)) {
        throw 'The Ollama mode helper is missing.'
    }
    $arguments = @('-Action', 'Read', '-ConfigPath', $ConfigPath)
    if ($Wait) {
        $arguments = @(
            '-Action', 'Wait',
            '-ConfigPath', $ConfigPath,
            '-TimeoutSeconds', [string] $OllamaWaitSeconds,
            '-PollSeconds', '2')
    }
    return Invoke-PowerShellHelper -ScriptPath $OllamaModeHelper -Arguments $arguments
}

function Ensure-LocalFiles {
    $envPath = Join-Path $RepoRoot '.env'
    if ((Test-Path -LiteralPath $envPath) -and
        -not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
        throw '.env exists but is not a file.'
    }
    if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
        $examplePath = Join-Path $RepoRoot '.env.example'
        if (-not (Test-Path -LiteralPath $examplePath -PathType Leaf)) {
            throw '.env.example is missing from the extracted project.'
        }
        Copy-Item -LiteralPath $examplePath -Destination $envPath
    }

    $directories = @($DataDirectory)
    if ($Stack -eq 'gpu') {
        $directories += @(
            (Join-Path $RepoRoot 'run'),
            (Join-Path $RepoRoot 'basedir'),
            (Join-Path $RepoRoot 'bank-images'))
    }
    foreach ($directory in $directories) {
        [System.IO.Directory]::CreateDirectory($directory) | Out-Null
    }
}

function Get-StudioComposeFiles {
    $files = @($BaseCompose)
    if ($Stack -eq 'studio') {
        $files += @($StaticExternalOverlay, $LocalExternalOverlay)
    }
    return $files
}

function Get-ComposeArguments {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Files,
        [switch] $OllamaProfile
    )

    $arguments = @('compose', '-p', $Project)
    foreach ($file in $Files) {
        $arguments += @('-f', $file)
    }
    if ($OllamaProfile) {
        $arguments += @('--profile', 'ollama')
    }
    return $arguments
}

function Inspect-Container {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name,
        [Parameter(Mandatory = $true)]
        [string] $Service,
        [string] $Mode = '',
        [string] $Role = '',
        [switch] $AppOnly,
        [switch] $NoPublishedPorts
    )

    $arguments = @(
        '-DockerExe', $DockerExe,
        '-ContainerName', $Name,
        '-ExpectedProject', $Project,
        '-ExpectedService', $Service,
        '-ExpectedHostIp', $BindAddress,
        '-ExpectedWorkingDir', $RepoRoot)
    if ($Mode) {
        $arguments += @('-ExpectedMode', $Mode)
    }
    if ($Role) {
        $arguments += @('-ExpectedRole', $Role)
    }
    if ($AppOnly) {
        $arguments += '-AppOnly'
    }
    if ($NoPublishedPorts) {
        $arguments += '-NoPublishedPorts'
    }
    return Invoke-PowerShellHelper -ScriptPath $InspectHelper -Arguments $arguments
}

function Show-StudioLogs {
    if (-not $DockerReady) {
        return
    }
    $arguments = Get-ComposeArguments -Files (Get-StudioComposeFiles)
    $arguments += @('logs', '--tail', '120', 'studio')
    [void] (Invoke-Docker -Arguments $arguments)
}

function Test-HostPortFree {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Address,
        [Parameter(Mandatory = $true)]
        [int] $Port
    )

    # Both the wildcard and the address we are about to publish on. Windows lets
    # 127.0.0.1:P coexist with another program's 0.0.0.0:P, and then routes
    # localhost to the MORE SPECIFIC socket -- so binding a port that looks free
    # would quietly steal an already-running app's own address. Refusing the port
    # whenever anything at all holds it keeps this launcher from shadowing it.
    foreach ($candidate in @([System.Net.IPAddress]::Any,
                             [System.Net.IPAddress]::Parse($Address))) {
        $listener = $null
        try {
            $listener = New-Object System.Net.Sockets.TcpListener($candidate, $Port)
            $listener.ExclusiveAddressUse = $true
            $listener.Start()
        }
        catch {
            return $false
        }
        finally {
            if ($null -ne $listener) {
                try { $listener.Stop() } catch { }
            }
        }
    }
    return $true
}

function Get-FreeHostPort {
    param(
        [Parameter(Mandatory = $true)]
        [int] $First,
        [Parameter(Mandatory = $true)]
        [int] $Last,
        [Parameter(Mandatory = $true)]
        [string] $Purpose
    )

    # Docker's own allocator only knows the ports IT handed out, so a published
    # RANGE does not skip a port held by a non-Docker process: it picks the first
    # one its table calls free and then dies at bind time. Anything already bound
    # to this exact address -- a ComfyUI on 8188, another app on 5055 -- has to be
    # found here instead, and a single resolved port is published.
    for ($port = $First; $port -le $Last; $port++) {
        if (Test-HostPortFree -Address $BindAddress -Port $port) {
            return $port
        }
    }
    throw ('No free ' + $Purpose + ' port between ' + $First + ' and ' + $Last +
        ' on ' + $BindAddress + '. Close an application using that range, then try again.')
}

function Test-HttpEndpoint {
    param([string] $Uri)

    try {
        $request = [System.Net.HttpWebRequest]::Create($Uri)
        $request.Timeout = 3000
        $request.ReadWriteTimeout = 3000
        $response = $request.GetResponse()
        $response.Close()
        return $true
    }
    catch {
        return $false
    }
}

function Open-Studio {
    param([string] $Uri)

    if ($NonInteractive -or $env:LDS_TEST_MODE -or $script:BrowserOpened) {
        return
    }
    try {
        Start-Process $Uri
        $script:BrowserOpened = $true
        Write-Host ('Opening ' + $Uri)
    }
    catch {
        Write-Warning ('Windows could not open the browser. Open ' + $Uri + ' manually.')
    }
}

function Get-ContainerHealth {
    $result = Invoke-Docker -Arguments @(
        'inspect', '--format',
        '{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}',
        $Container) -Quiet
    if ($result.ExitCode -ne 0) {
        return @{ State = 'missing'; Health = '' }
    }
    $parts = $result.Text.Trim().Split('|')
    $state = $parts[0].Trim().ToLowerInvariant()
    $health = ''
    if ($parts.Count -gt 1) {
        $health = $parts[1].Trim().ToLowerInvariant()
    }
    return @{ State = $state; Health = $health }
}

function Wait-ForStudio {
    param([string] $AppUrl)

    $attemptLimit = if ($Stack -eq 'gpu') { 300 } else { 120 }
    Write-Host 'Waiting for LoRA Dataset Studio to become healthy...'
    for ($attempt = 0; $attempt -lt $attemptLimit; $attempt++) {
        $state = Get-ContainerHealth
        if ($state.State -in @('exited', 'dead', 'restarting', 'missing') -or
            $state.Health -eq 'unhealthy') {
            throw 'The Studio container exited or became unhealthy during startup.'
        }
        if ($state.Health -eq 'healthy') {
            return
        }
        if (-not $BrowserOpened -and (Test-HttpEndpoint -Uri ($AppUrl + 'api/health'))) {
            Open-Studio -Uri $AppUrl
        }
        Start-Sleep -Seconds 5
    }
    throw 'LoRA Dataset Studio did not become healthy before the startup timeout.'
}

function Stop-OwnedSidecar {
    $inspection = Inspect-Container -Name $SidecarContainer -Service 'ollama' -Role 'ollama' -NoPublishedPorts
    if ($inspection['STATE'] -in @('ABSENT', 'STOPPED')) {
        return
    }
    if ($inspection['STATE'] -ne 'RUNNING') {
        Write-Warning 'An Ollama-named container is not owned by this LDS stack; it was not touched.'
        return
    }

    $files = @($BaseCompose, $SidecarOverlay)
    if ($Stack -eq 'studio') {
        $files = @($BaseCompose, $StaticExternalOverlay, $LocalExternalOverlay, $SidecarOverlay)
    }
    $arguments = Get-ComposeArguments -Files $files -OllamaProfile
    $arguments += @('stop', 'ollama')
    $result = Invoke-Docker -Arguments $arguments
    if ($result.ExitCode -ne 0) {
        Write-Warning 'The LDS Ollama sidecar could not be stopped; Studio remains available.'
    }
    else {
        Write-Host 'Stopped the LDS Ollama sidecar. ollama-data was preserved.'
    }
}

function Wait-ForOptionalSidecar {
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        $result = Invoke-Docker -Arguments @(
            'inspect', '--format',
            '{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}',
            $SidecarContainer) -Quiet
        if ($result.ExitCode -ne 0) {
            return $false
        }
        $parts = $result.Text.Trim().Split('|')
        $status = $parts[0].Trim().ToLowerInvariant()
        $health = ''
        if ($parts.Count -gt 1) {
            $health = $parts[1].Trim().ToLowerInvariant()
        }
        if ($status -eq 'running' -and
            ($health -eq 'healthy' -or -not $health)) {
            return $true
        }
        if ($status -in @('exited', 'dead', 'restarting') -or
            $health -eq 'unhealthy') {
            return $false
        }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Start-OptionalSidecar {
    [System.IO.Directory]::CreateDirectory((Join-Path $RepoRoot 'ollama-data')) | Out-Null
    $env:LDS_OLLAMA_DATA = './ollama-data'

    $owned = Inspect-Container -Name $SidecarContainer -Service 'ollama' -Role 'ollama' -NoPublishedPorts
    if ($owned['STATE'] -eq 'COLLISION' -or $owned['STATE'] -eq 'INVALID') {
        Write-Warning 'The Ollama container name is not safely owned by this LDS stack. Studio remains ready.'
        return
    }

    $files = @($BaseCompose, $SidecarOverlay)
    if ($Stack -eq 'studio') {
        $files = @($BaseCompose, $StaticExternalOverlay, $LocalExternalOverlay, $SidecarOverlay)
    }

    if ($Stack -eq 'gpu') {
        Write-Host 'Starting optional Ollama Docker with NVIDIA acceleration...'
        $gpuFiles = $files + $SidecarGpuOverlay
        $arguments = Get-ComposeArguments -Files $gpuFiles -OllamaProfile
        $arguments += @('up', '-d', 'ollama')
        $result = Invoke-Docker -Arguments $arguments
        if ($result.ExitCode -eq 0 -and (Wait-ForOptionalSidecar)) {
            Write-Host 'Ollama Docker requested NVIDIA acceleration. No model was downloaded.'
            return
        }
        Write-Warning 'NVIDIA Ollama could not start; retrying portable CPU mode.'
        $arguments = Get-ComposeArguments -Files $files -OllamaProfile
        $arguments += @('up', '-d', '--force-recreate', 'ollama')
        $result = Invoke-Docker -Arguments $arguments
        if ($result.ExitCode -eq 0 -and (Wait-ForOptionalSidecar)) {
            Write-Host 'Ollama Docker is running in portable CPU mode. No model was downloaded.'
            return
        }
    }
    else {
        Write-Host 'Starting optional Ollama Docker in portable CPU mode...'
        $arguments = Get-ComposeArguments -Files $files -OllamaProfile
        $arguments += @('up', '-d', 'ollama')
        $result = Invoke-Docker -Arguments $arguments
        if ($result.ExitCode -eq 0 -and (Wait-ForOptionalSidecar)) {
            Write-Host 'Ollama Docker is running in CPU mode. No model was downloaded.'
            return
        }
    }
    Write-Warning 'The optional Ollama sidecar could not start. Studio remains ready.'
}

function Diagnose-HostOllama {
    if ($env:LDS_TEST_MODE) {
        return
    }
    if (-not (Test-HttpEndpoint -Uri 'http://127.0.0.1:11434/api/tags')) {
        Write-Warning 'Ollama is not reachable on Windows at http://127.0.0.1:11434.'
        return
    }
    $result = Invoke-Docker -Arguments @(
        'exec', $Container, 'python3', '-c',
        "import urllib.request; urllib.request.urlopen('http://host.docker.internal:11434/api/tags', timeout=3).read()") -Quiet
    if ($result.ExitCode -ne 0) {
        Write-Warning 'Ollama works on Windows but Docker cannot reach it.'
        Write-Host 'Set OLLAMA_HOST=0.0.0.0:11434, restart Ollama, and allow port 11434 on private/trusted networks only.'
        return
    }
    Write-Host 'Ollama on Windows is reachable from Studio.'
}

function Diagnose-ExternalComfy {
    if ($env:LDS_TEST_MODE) {
        return
    }
    if ($Stack -ne 'studio') {
        return
    }
    if (-not (Test-HttpEndpoint -Uri 'http://127.0.0.1:8188/system_stats')) {
        Write-Warning 'Your usual ComfyUI is not reachable at http://127.0.0.1:8188.'
        Write-Host 'Start it normally with --listen 0.0.0.0.'
        Write-Host 'Allow port 8188 through Windows Firewall on private/trusted networks only.'
        return
    }
    $result = Invoke-Docker -Arguments @(
        'exec', $Container, 'python3', '-c',
        "import urllib.request; urllib.request.urlopen('http://host.docker.internal:8188/system_stats', timeout=3).read()") -Quiet
    if ($result.ExitCode -ne 0) {
        Write-Warning 'ComfyUI works on Windows but the Studio container cannot reach it.'
        Write-Host 'Restart your usual ComfyUI with --listen 0.0.0.0.'
        Write-Host 'Allow port 8188 through Windows Firewall on private/trusted networks only.'
        return
    }
    Write-Host 'Your existing ComfyUI is reachable from Studio.'
}

function Apply-OllamaMode {
    param([string] $Mode)

    if ($Mode -eq 'docker') {
        Start-OptionalSidecar
        return
    }
    if ($Mode -in @('none', 'host')) {
        Stop-OwnedSidecar
    }
    if ($Mode -eq 'host') {
        Diagnose-HostOllama
    }
    elseif ($Mode -eq 'none') {
        Write-Host 'Ollama is disabled. No Ollama image or model was downloaded.'
    }
}

$previousLocation = Get-Location
try {
    Set-Location -LiteralPath $RepoRoot

    foreach ($required in @($BaseCompose, $InspectHelper, $OllamaModeHelper)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw ('A required Docker file is missing: ' + [System.IO.Path]::GetFileName($required))
        }
    }
    if ($Stack -eq 'studio') {
        foreach ($required in @($StaticExternalOverlay, $ComfyHelper)) {
            if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
                throw ('A required external-ComfyUI file is missing: ' + [System.IO.Path]::GetFileName($required))
            }
        }
    }
    foreach ($required in @($HostOverlay, $SidecarOverlay, $SidecarGpuOverlay)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw ('A required Ollama overlay is missing: ' + [System.IO.Path]::GetFileName($required))
        }
    }

    Configure-Or-ValidateExternalComfy
    Write-LauncherMarker
    Ensure-LocalFiles
    $requestedBindAddress = ([string] $env:LDS_BIND_ADDRESS).Trim()
    if ($requestedBindAddress) {
        if ($requestedBindAddress -notin @('127.0.0.1', '0.0.0.0')) {
            throw 'LDS_BIND_ADDRESS must be 127.0.0.1 or 0.0.0.0.'
        }
        $BindAddress = $requestedBindAddress
    }
    $env:LDS_BIND_ADDRESS = $BindAddress

    if ($Stack -eq 'gpu') {
        $env:LDS_DATA = './data-docker-gpu'
        $env:LDS_COMFY_RUN = './run'
        $env:LDS_COMFY_BASEDIR = './basedir'
        $env:LDS_BANK_SOURCES = './bank-images'
        # Docker Desktop exposes every Windows bind mount as root:root and no
        # chown inside the container can change that, so upstream's init script
        # aborted on /comfy/mnt ("expected 1000:1000, actual 0:0") and the
        # container restart-looped. This launcher only ever runs on Windows, so
        # the owner it must declare is 0:0. The Compose default stays 1000:1000
        # for a Linux host, where the mounts really are owned by the user.
        if (-not ([string] $env:LDS_UID).Trim()) { $env:LDS_UID = '0' }
        if (-not ([string] $env:LDS_GID).Trim()) { $env:LDS_GID = '0' }
    }
    else {
        $env:LDS_DATA = './data-docker'
    }

    $modeResult = Get-OllamaMode
    if ($modeResult['STATE'] -eq 'INVALID') {
        throw 'config.json contains an invalid Ollama deployment mode. No secret was displayed.'
    }
    $ollamaMode = $modeResult['MODE']
    # An older helper answers without this line; treat its silence as Ollama, the
    # only thing that existed before the setting.
    $llmProvider = $modeResult['PROVIDER']
    if (-not $llmProvider) { $llmProvider = 'ollama' }

    $DockerExe = Get-DockerExecutable
    Wait-ForDocker

    if ($NonInteractive -and ($llmProvider -eq 'lmstudio' -or $ollamaMode -in @('none', 'host'))) {
        Stop-OwnedSidecar
    }

    # Resolved now, and published as ONE port rather than a range: a fixed
    # publish also survives a plain `docker start`, which re-picks a port out of
    # a range and silently moves the URL the user bookmarked.
    $env:LDS_HOST_PORT = [string] (Get-FreeHostPort -First 5050 -Last 5149 -Purpose 'Studio')
    if ($Stack -eq 'gpu') {
        $env:LDS_COMFY_HOST_PORT = [string] (
            Get-FreeHostPort -First 8188 -Last 8287 -Purpose 'private ComfyUI')
    }

    $inspectionParameters = @{
        Name = $Container
        Service = 'studio'
        Mode = $ExpectedMode
    }
    if ($Stack -eq 'studio') {
        $inspectionParameters['AppOnly'] = $true
    }
    $inspection = Inspect-Container @inspectionParameters
    if ($inspection['STATE'] -eq 'COLLISION' -or $inspection['STATE'] -eq 'INVALID') {
        throw 'The expected Studio container name belongs to another project or folder. Nothing was changed.'
    }

    $forceRecreate = $false
    $appPort = $inspection['APP_PORT']
    $comfyPort = $inspection['COMFY_PORT']
    if ($inspection['STATE'] -eq 'RUNNING' -and -not $Configure -and -not $Rebuild -and -not $UpdateRebuild) {
        if (-not $appPort) {
            throw 'The running Studio port could not be read safely.'
        }
    }
    else {
        if ($inspection['STATE'] -in @('RUNNING', 'STOPPED', 'SWITCH')) {
            $forceRecreate = $true
            if ($appPort) {
                $env:LDS_HOST_PORT = $appPort
            }
            if ($Stack -eq 'gpu' -and $comfyPort) {
                $env:LDS_COMFY_HOST_PORT = $comfyPort
            }
        }

        $studioFiles = Get-StudioComposeFiles
        $configArguments = Get-ComposeArguments -Files $studioFiles
        $configArguments += @('config', '--quiet')
        $configResult = Invoke-Docker -Arguments $configArguments
        if ($configResult.ExitCode -ne 0) {
            throw 'Docker Compose rejected the Studio configuration.'
        }

        Write-Host 'Building and starting LoRA Dataset Studio...'
        $upArguments = Get-ComposeArguments -Files $studioFiles
        $upArguments += @('up', '-d', '--build')
        if ($forceRecreate) {
            $upArguments += '--force-recreate'
        }
        $upArguments += 'studio'
        $upResult = Invoke-Docker -Arguments $upArguments
        if ($upResult.ExitCode -ne 0) {
            throw 'Docker could not build or start LoRA Dataset Studio.'
        }

        $inspection = Inspect-Container @inspectionParameters
        if ($inspection['STATE'] -ne 'RUNNING') {
            throw 'Docker started Studio but its published ports could not be read safely.'
        }
        $appPort = $inspection['APP_PORT']
        $comfyPort = $inspection['COMFY_PORT']
    }

    if (-not $appPort) {
        throw 'The Studio port is missing.'
    }
    if ($Stack -eq 'gpu' -and -not $comfyPort) {
        throw 'The private ComfyUI port is missing.'
    }

    $appUrl = 'http://127.0.0.1:' + $appPort + '/'
    Wait-ForStudio -AppUrl $appUrl
    if ($NonInteractive) {
        # Reaching here means Wait-ForStudio saw Docker report healthy, so this
        # zero exit is what the updater commits its transaction on. Updater mode
        # will not wait for an Ollama deployment choice: that prompt needs a
        # person, and the update must not hang for 15 minutes without one.
        Write-Host ('LoRA Dataset Studio is healthy at ' + $appUrl)
        Write-Host 'Updater mode will not wait for the browser or the Setup choices.'
        exit 0
    }

    Write-Host ''
    Write-Host ('LoRA Dataset Studio is ready at ' + $appUrl)
    if ($Stack -eq 'gpu') {
        Write-Host ('Private ComfyUI is ready at http://127.0.0.1:' + $comfyPort + '/')
    }
    Open-Studio -Uri $appUrl
    Diagnose-ExternalComfy

    if ($llmProvider -eq 'lmstudio') {
        # The sidecar exists to provide Ollama. The user has said they run LM
        # Studio, so there is nothing for it to provide -- and `deployment_mode`
        # may still read 'docker' from an earlier session, which would otherwise
        # start a container that nothing in the app ever calls.
        Write-Host ''
        Stop-OwnedSidecar
        Write-Host 'Local LLM provider is LM Studio, so no Ollama sidecar was started.'
        Write-Host 'Studio reaches LM Studio on this machine at http://host.docker.internal:1234 --'
        Write-Host 'start its server from LM Studio > Developer > Start Server.'
    }
    elseif ($ollamaMode -eq 'unset') {
        Write-Host ''
        Write-Host 'Choose the Ollama deployment mode in the Studio Setup page.'
        if ($env:LDS_TEST_MODE) {
            Write-Host ('Test mode waits ' + $OllamaWaitLabel +
                ' for that choice, not the interactive timeout.')
        }
        else {
            Write-Host 'This window will wait up to 15 minutes; closing it is safe, and rerunning the BAT resumes.'
        }
        $modeResult = Get-OllamaMode -Wait
        if ($modeResult['STATE'] -eq 'VALID') {
            $ollamaMode = $modeResult['MODE']
            Apply-OllamaMode -Mode $ollamaMode
        }
        elseif ($modeResult['STATE'] -eq 'TIMEOUT') {
            Write-Warning ('Ollama setup was not completed within ' + $OllamaWaitLabel +
                '. Studio remains ready; rerun the BAT later.')
        }
        else {
            Write-Warning 'config.json became invalid while waiting. Studio remains ready; no sidecar was changed.'
        }
    }
    else {
        Apply-OllamaMode -Mode $ollamaMode
    }

    exit 0
}
catch {
    Write-Host ''
    Write-Error $_.Exception.Message -ErrorAction Continue
    Show-StudioLogs
    if (-not $NonInteractive -and -not $env:LDS_TEST_MODE) {
        Write-Host ''
        Write-Host 'No data was removed. Press any key to close this window.'
        try {
            [void] [Console]::ReadKey($true)
        }
        catch {
        }
    }
    exit 1
}
finally {
    Set-Location -LiteralPath $previousLocation
}
