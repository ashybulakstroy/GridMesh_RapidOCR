@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (echo ERROR: .venv not found & exit /b 1)
set "OCR_FOOTER_FOCUS_ENABLE=0"
".venv\Scripts\python.exe" -m App
exit /b %errorlevel%
