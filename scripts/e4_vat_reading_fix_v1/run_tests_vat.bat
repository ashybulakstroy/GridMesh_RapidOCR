@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (echo ERROR: project .venv not found & exit /b 1)
".venv\Scripts\python.exe" "scripts\e4_vat_reading_fix_v1\test_vat_fix.py"
exit /b %errorlevel%
