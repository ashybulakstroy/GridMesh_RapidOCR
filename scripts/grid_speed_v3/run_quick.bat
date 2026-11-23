@echo off
setlocal EnableExtensions
chcp 65001 >nul
if not defined PROJECT_ROOT set "PROJECT_ROOT=C:\Work\Projects\Prj_18_OCR_2"
set "PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo ERROR: not found "%PY%"
  echo Set PROJECT_ROOT to the existing project with its .venv
  pause
  exit /b 2
)
cd /d "%~dp0"
"%PY%" grid_speed_experiment.py --project "%PROJECT_ROOT%" --repeats 1 --warmups 0 %*
if errorlevel 1 (
  echo FAILED - check paths and dependencies.
  pause
  exit /b 1
)
echo.
echo Experiment completed. Open the newest results\...\summary.html
pause
