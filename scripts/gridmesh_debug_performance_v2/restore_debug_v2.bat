@echo off
setlocal
set "PROJECT=C:\Work\Projects\Prj_18_OCR_2"
set "PY=%PROJECT%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Python .venv not found: "%PY%"
  exit /b 2
)
echo Checking whether source files are unchanged since installation...
"%PY%" "%~dp0restore_experiment.py" --project "%PROJECT%" --dry-run || exit /b 1
echo Restoring project source from backup...
"%PY%" "%~dp0restore_experiment.py" --project "%PROJECT%" || exit /b 1
echo Restored. Revert LOG_LEVEL in .env manually if needed.
endlocal
