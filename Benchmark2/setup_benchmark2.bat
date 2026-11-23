@echo off
setlocal EnableExtensions
pushd "%~dp0" || (echo [ERROR] Cannot access Benchmark2 directory. & pause & exit /b 1)
if not defined BENCHMARK2_BASE_PROJECT set "BENCHMARK2_BASE_PROJECT=C:\Work\Projects\Prj_18_OCR_2"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%CD%\scripts\setup_logged.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo [ERROR] Setup failed. Main project was not modified.
pause
popd
exit /b %EXIT_CODE%
