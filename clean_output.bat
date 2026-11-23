@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv\Scripts\python.exe not found.
    echo Install project dependencies first.
    pause
    exit /b 1
)

set "PYTHONIOENCODING=utf-8"
".venv\Scripts\python.exe" "scripts\clean_output.py" %*
set "RC=%ERRORLEVEL%"

if "%~1"=="" pause
exit /b %RC%
