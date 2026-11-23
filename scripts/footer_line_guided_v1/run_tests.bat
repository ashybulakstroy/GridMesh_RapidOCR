@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (echo ERROR: .venv not found & exit /b 1)
".venv\Scripts\python.exe" -m pytest scripts\footer_line_guided_v1\tests\test_focus.py -q
exit /b %errorlevel%
