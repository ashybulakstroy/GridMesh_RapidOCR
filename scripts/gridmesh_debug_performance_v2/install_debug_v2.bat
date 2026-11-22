@echo off
setlocal
set "PROJECT=C:\Work\Projects\Prj_18_OCR_2"
set "PY=%PROJECT%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Python .venv was not found: "%PY%"
  echo Edit PROJECT in this file if your project uses another path.
  exit /b 2
)
echo Checking source revision before any changes...
"%PY%" "%~dp0apply_experiment.py" --project "%PROJECT%" --dry-run || exit /b 1
echo Applying DEBUG Performance V2...
"%PY%" "%~dp0apply_experiment.py" --project "%PROJECT%" || exit /b 1
echo Installed. Set LOG_LEVEL=DEBUG in %PROJECT%\.env and restart the app.
endlocal
