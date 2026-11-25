@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "OUTPUT_ROOT=%~dp0data\output"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv\Scripts\python.exe not found.
    echo Install project dependencies first.
    pause
    exit /b 1
)

if not exist "%OUTPUT_ROOT%" (
    echo [ERROR] %OUTPUT_ROOT% not found.
    pause
    exit /b 1
)

set "PYTHONIOENCODING=utf-8"
if /i "%~1"=="--check" (
    ".venv\Scripts\python.exe" "scripts\clean_output.py" --mode duplicates --root "%OUTPUT_ROOT%"
) else (
    ".venv\Scripts\python.exe" "scripts\clean_output.py" --mode duplicates --apply --yes --root "%OUTPUT_ROOT%" %*
)
set "RC=%ERRORLEVEL%"

if "%~1"=="" pause
exit /b %RC%
