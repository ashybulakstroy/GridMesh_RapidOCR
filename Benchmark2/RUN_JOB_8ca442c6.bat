@echo off
setlocal
cd /d "%~dp0"
if not defined BENCHMARK2_BASE_PROJECT set "BENCHMARK2_BASE_PROJECT=C:\Work\Prj_36_OCR2"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Benchmark2\.venv not installed. Run normal Benchmark2 setup first.
  pause
  exit /b 1
)
echo [IMPORT + BENCHMARK] Reconstructing + verifying GridMesh ROIs; then REC speed test.
".venv\Scripts\python.exe" "scripts\rebench_job.py" --job-id 8ca442c659ab434ca9fd6f7166f01cfa --runs 3
if errorlevel 1 (
  echo ERROR: ROI export or benchmark failed; see console.
  pause
  exit /b 2
)
echo [DONE] Results: Benchmark2\results\job_rebench\8ca442c659ab434ca9fd6f7166f01cfa
pause
