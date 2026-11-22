cd C:\Work\Projects\Prj_18_OCR_2\ 

@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python environment not found. Install dependencies first.
    pause
    exit /b 1
)

set "PROJECT18_PYTHON=%~dp0.venv\Scripts\python.exe"
echo Stopping Project 18 server...
powershell.exe -NoProfile -Command "$ErrorActionPreference = 'Stop'; try { $servers = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.ExecutablePath -ieq $env:PROJECT18_PYTHON -and $_.CommandLine -match '\s-m\s+App\s*$' }); $serverIds = @($servers.ProcessId); $roots = @($servers | Where-Object { $_.ParentProcessId -notin $serverIds }); foreach ($server in $roots) { & taskkill.exe /PID $server.ProcessId /T /F; if ($LASTEXITCODE -ne 0) { throw 'Failed to stop server.' } } } catch { Write-Host $_ -ForegroundColor Red; exit 1 }"
if errorlevel 1 (
    echo ERROR: Server could not be stopped.
    pause
    exit /b 1
)

echo Starting Project 18 server...
call "%~dp0start.bat"
set "SERVER_EXIT_CODE=%ERRORLEVEL%"
if not "%SERVER_EXIT_CODE%"=="0" pause
exit /b %SERVER_EXIT_CODE%
