# Logged setup wrapper. Runs scripts\setup.ps1 in a CHILD PowerShell process so that
# native pip/venv stdout+stderr are merged only by this wrapper. Running setup.ps1
# in-process with *>&1 would turn pip's stderr into terminating NativeCommandError
# under setup.ps1's $ErrorActionPreference = 'Stop'.
# Start-Transcript is NOT used: Windows PowerShell 5.1 transcripts omit native pip output.
param(
    [string]$BaseProject = '',
    [switch]$SkipCloneRequirements
)
$ErrorActionPreference = 'Continue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$LogDir = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ('setup_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
$Setup = Join-Path $PSScriptRoot 'setup.ps1'

$PowerShellArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Setup)
if (-not [string]::IsNullOrWhiteSpace($BaseProject)) {
    $PowerShellArgs += @('-BaseProject', $BaseProject)
}
if ($SkipCloneRequirements) {
    $PowerShellArgs += '-SkipCloneRequirements'
}

Write-Host "[BENCHMARK2] Setup log: $LogPath"
$ExitCode = 1
try {
    & powershell.exe @PowerShellArgs *>&1 | Tee-Object -FilePath $LogPath
    if ($null -ne $LASTEXITCODE) { $ExitCode = $LASTEXITCODE } else { $ExitCode = 0 }
} catch {
    $_ | Out-String | Tee-Object -FilePath $LogPath -Append | Write-Host
    $ExitCode = 1
} finally {
    ("[BENCHMARK2] Setup exit code: " + $ExitCode) | Out-File -LiteralPath $LogPath -Append -Encoding Unicode
    Write-Host "[BENCHMARK2] Setup log saved: $LogPath"
}
exit $ExitCode
