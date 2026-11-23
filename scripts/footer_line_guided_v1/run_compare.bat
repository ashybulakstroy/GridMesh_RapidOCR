@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (echo ERROR: .venv not found & exit /b 1)
".venv\Scripts\python.exe" scripts\footer_line_guided_v1\compare.py --rounds 3 --expected-total 90665
exit /b %errorlevel%
