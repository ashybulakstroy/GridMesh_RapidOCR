@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (echo ERROR: .venv\Scripts\python.exe not found & exit /b 1)
".venv\Scripts\python.exe" "scripts\e4_vat_reading_fix_v1\apply_patch.py" restore
exit /b %errorlevel%
