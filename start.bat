@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Install dependencies first. & exit /b 1)
".venv\Scripts\python.exe" -m App
