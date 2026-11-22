@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0..\numeric_only_crop_v2\.venv_numbers\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0..\numeric_only_crop_v1\.venv_numbers\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0.venv_numbers\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%PY%" (echo Install dependencies first: install_cnocr.bat & exit /b 1)
"%PY%" runner.py --latest --preprocess-only --save-crops %*
