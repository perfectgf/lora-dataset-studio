[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Read', 'Wait')]
    [string] $Action,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ConfigPath,

    [ValidateRange(1, 3600)]
    [int] $TimeoutSeconds = 900,

    [ValidateRange(1, 30)]
    [int] $PollSeconds = 2
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8NoBom

function Get-LocalLlmProvider {
    param($Root)

    # The Ollama sidecar answers ONE question: "where does Ollama run?". Someone
    # who picked LM Studio has already answered it -- nowhere -- so the mode is
    # moot for them. Unset, or anything unknown, reads as Ollama: that is what
    # every install that predates the setting has, and it must not change.
    if ($null -eq $Root) { return 'ollama' }
    $localLlm = $Root.PSObject.Properties['local_llm']
    if ($null -eq $localLlm -or $null -eq $localLlm.Value) { return 'ollama' }
    if ($localLlm.Value -isnot [System.Management.Automation.PSCustomObject]) { return 'ollama' }
    $provider = $localLlm.Value.PSObject.Properties['provider']
    if ($null -eq $provider -or $provider.Value -isnot [string]) { return 'ollama' }
    $name = $provider.Value.Trim().ToLowerInvariant()
    if ($name -eq 'lmstudio') { return 'lmstudio' }
    return 'ollama'
}

function Get-DeploymentMode {
    param([string] $LiteralPath)

    if (-not (Test-Path -LiteralPath $LiteralPath -PathType Leaf)) {
        return @{ State = 'VALID'; Mode = 'unset'; Provider = 'ollama' }
    }

    try {
        $root = Get-Content -LiteralPath $LiteralPath -Raw -Encoding UTF8 |
            ConvertFrom-Json
    }
    catch {
        return @{ State = 'INVALID'; Mode = 'unset'; Provider = 'ollama' }
    }

    if ($null -eq $root -or
        $root.GetType() -ne [System.Management.Automation.PSCustomObject]) {
        return @{ State = 'INVALID'; Mode = 'unset'; Provider = 'ollama' }
    }

    $provider = Get-LocalLlmProvider -Root $root

    $ollamaProperty = $root.PSObject.Properties['ollama']
    if ($null -eq $ollamaProperty -or $null -eq $ollamaProperty.Value) {
        return @{ State = 'VALID'; Mode = 'unset'; Provider = $provider }
    }
    $ollama = $ollamaProperty.Value
    if ($ollama.GetType() -ne [System.Management.Automation.PSCustomObject]) {
        return @{ State = 'INVALID'; Mode = 'unset'; Provider = $provider }
    }

    $modeProperty = $ollama.PSObject.Properties['deployment_mode']
    if ($null -eq $modeProperty -or $null -eq $modeProperty.Value) {
        return @{ State = 'VALID'; Mode = 'unset'; Provider = $provider }
    }
    if ($modeProperty.Value -isnot [string]) {
        return @{ State = 'INVALID'; Mode = 'unset'; Provider = $provider }
    }

    $mode = $modeProperty.Value.Trim().ToLowerInvariant()
    if (-not $mode -or $mode -eq 'unconfigured') {
        return @{ State = 'VALID'; Mode = 'unset'; Provider = $provider }
    }
    if ($mode -notin @('none', 'host', 'docker')) {
        return @{ State = 'INVALID'; Mode = 'unset'; Provider = $provider }
    }
    return @{ State = 'VALID'; Mode = $mode; Provider = $provider }
}

function Write-Result {
    param([string] $State, [string] $Mode, [string] $Provider = 'ollama')

    [Console]::Out.WriteLine(('STATE={0}' -f $State))
    [Console]::Out.WriteLine(('MODE={0}' -f $Mode))
    [Console]::Out.WriteLine(('PROVIDER={0}' -f $Provider))
}

$fullPath = [System.IO.Path]::GetFullPath($ConfigPath)
if ($Action -eq 'Read') {
    $result = Get-DeploymentMode -LiteralPath $fullPath
    Write-Result -State $result.State -Mode $result.Mode -Provider $result.Provider
    exit 0
}

$deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
do {
    $result = Get-DeploymentMode -LiteralPath $fullPath
    if ($result.State -eq 'INVALID') {
        Write-Result -State 'INVALID' -Mode 'unset' -Provider $result.Provider
        exit 0
    }
    # Waiting for an Ollama deployment choice is over the moment the user says
    # they run LM Studio: that choice will never arrive, because the Setup page
    # stops offering the cards. Without this the window blocks for its full
    # timeout on a question nobody is being asked.
    if ($result.Provider -eq 'lmstudio' -or $result.Mode -ne 'unset') {
        Write-Result -State 'VALID' -Mode $result.Mode -Provider $result.Provider
        exit 0
    }
    Start-Sleep -Seconds $PollSeconds
} while ([DateTime]::UtcNow -lt $deadline)

Write-Result -State 'TIMEOUT' -Mode 'unset' -Provider $result.Provider
exit 0
