# Install Benchmark2 on ANY Windows drive. No write operations on the source project.
# Rebuilds an isolated venv using a pip snapshot from the source project's venv.
param(
    [string]$BaseProject = '',
    [switch]$SkipCloneRequirements
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($BaseProject)) {
    if (-not [string]::IsNullOrWhiteSpace($env:BENCHMARK2_BASE_PROJECT)) {
        $BaseProject = $env:BENCHMARK2_BASE_PROJECT
    } else {
        $BaseProject = 'C:\Work\Projects\Prj_18_OCR_2'
    }
}
$BaseProject = $BaseProject.TrimEnd('\')
$SourcePy = Join-Path $BaseProject '.venv\Scripts\python.exe'
$Target = Join-Path $Root '.venv'
$TargetPy = Join-Path $Target 'Scripts\python.exe'
$Marker = Join-Path $Target '.benchmark2-root.txt'
$env:BENCHMARK2_BASE_PROJECT = $BaseProject
Write-Host "[BENCHMARK2] Installation directory: $Root"
Write-Host "[BENCHMARK2] Source project: $BaseProject"

if (-not (Test-Path -LiteralPath $SourcePy -PathType Leaf)) {
    throw "Main project Python missing at $SourcePy. Confirm the original project's .venv exists."
}
# Avoid embedded double quotes: Windows PowerShell 5.1 mangles them for native -c arguments.
$Version = & $SourcePy -c 'import sys; print(sys.version_info.major, sys.version_info.minor, sep=chr(46))'
if ($LASTEXITCODE -ne 0 -or $Version.Trim() -ne '3.14') {
    throw "Expected Python 3.14 in main project venv, got: $Version"
}
if (Test-Path -LiteralPath $Marker) {
    $PreviousRoot = (Get-Content -LiteralPath $Marker -Encoding UTF8 -Raw).Trim()
    if (-not [string]::Equals($PreviousRoot, $Root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Benchmark2 was moved after installation from '$PreviousRoot' to '$Root'. Delete only '$Target' and rerun setup."
    }
}
if (-not (Test-Path -LiteralPath $TargetPy -PathType Leaf)) {
    Write-Host '[1/5] Create isolated Python 3.14 venv at current location (do not copy binary venv)'
    & $SourcePy -m venv $Target
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create isolated venv.' }
}
# Remove completion marker during updates so failed installs are not considered ready.
if (Test-Path -LiteralPath $Marker) { Remove-Item -LiteralPath $Marker -Force }

$Frozen = Join-Path $Root 'config\base_requirements_snapshot.txt'
if (-not $SkipCloneRequirements) {
    Write-Host '[2/5] Export dependency versions from original project (read-only)'
    $FreezeLines = @(& $SourcePy -m pip freeze --exclude-editable)
    if ($LASTEXITCODE -ne 0) { throw 'pip freeze failed in source project.' }
    [System.IO.File]::WriteAllLines($Frozen, [string[]]$FreezeLines, [System.Text.UTF8Encoding]::new($false))
    Write-Host '[3/5] Install snapshot into independent Benchmark2 .venv'
    & $TargetPy -m pip install -r $Frozen
    if ($LASTEXITCODE -ne 0) { throw 'Snapshot install failed. Original .venv has not been modified.' }
} else {
    Write-Host '[2-3/5] Source snapshot installation explicitly skipped.'
}
Write-Host '[4/5] Install Benchmark2-only OCR dependencies'
& $TargetPy -m pip install -r (Join-Path $Root 'requirements-extra.txt')
if ($LASTEXITCODE -ne 0) { throw 'Additional OCR dependency install failed.' }
Write-Host '[5/5] Verify package environment and Benchmark2 dataset definitions'
& $TargetPy -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflicts detected by pip check.' }
Push-Location $Root
try {
    & $TargetPy -m benchmark2 validate
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark2 validation failed.' }
} finally { Pop-Location }
[System.IO.File]::WriteAllText($Marker, $Root + "`n", [System.Text.UTF8Encoding]::new($false))
Write-Host "[READY] Separate venv installed at $Target" -ForegroundColor Green
Write-Host '[NEXT] Double-click run_benchmark2.bat. Only Benchmark2 outputs are written.'
