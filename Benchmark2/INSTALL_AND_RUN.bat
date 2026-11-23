@echo off
setlocal EnableExtensions
rem Portable launcher: %~dp0 is the directory containing THIS file, on any drive.
pushd "%~dp0" || (echo [ERROR] Cannot access Benchmark2 directory. & pause & exit /b 1)
if not defined BENCHMARK2_BASE_PROJECT set "BENCHMARK2_BASE_PROJECT=C:\Work\Projects\Prj_18_OCR_2"

if not exist ".venv\.benchmark2-root.txt" (
    echo [SETUP] Installing isolated Benchmark2 environment in "%CD%\.venv"
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%CD%\scripts\setup_logged.ps1"
    if errorlevel 1 (
        echo [ERROR] Setup failed. Main project was not modified.
        pause
        popd
        exit /b 1
    )
)
call "%CD%\run_benchmark2.bat"
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
