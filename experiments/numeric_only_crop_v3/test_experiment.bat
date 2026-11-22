@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0..\numeric_only_crop_v1\.venv_numbers\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0.venv_numbers\Scripts\python.exe"
if not exist "%PY%" (echo First run install_cnocr.bat & exit /b 1)
"%PY%" test_preprocess.py
