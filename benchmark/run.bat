@echo off
cd /d "%~dp0.."
python -m benchmark plan
python -m benchmark validate
pause
