@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (echo ERROR: .venv not found & exit /b 1)
".venv\Scripts\python.exe" "scripts\footer_line_guided_v1\patch.py" restore
exit /b %errorlevel%
