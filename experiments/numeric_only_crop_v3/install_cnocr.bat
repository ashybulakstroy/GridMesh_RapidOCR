@echo off
setlocal
cd /d "%~dp0"
rem Reuse V2/V1 isolated Python 3.12 environment if present.
set "PY=%~dp0..\numeric_only_crop_v2\.venv_numbers\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0..\numeric_only_crop_v1\.venv_numbers\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0.venv_numbers\Scripts\python.exe"
if not exist "%PY%" (
    py -3.12 -m venv ".venv_numbers"
    if errorlevel 1 (echo Python 3.12 not available. & exit /b 1)
)
if not exist "%PY%" (echo Python environment not found. & exit /b 1)
"%PY%" -m pip install "cnocr[ort-cpu]==2.3.3" "numpy<3" opencv-python-headless Pillow
if errorlevel 1 (echo CnOCR installation failed; main GridMesh Python was unchanged. & exit /b 1)
"%PY%" -c "from cnocr import CnOcr; print('CnOCR import OK')"
if errorlevel 1 exit /b 1
echo Setup complete. Model may be downloaded on first use.
