@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0" || (echo [ERROR] Cannot access Benchmark2 directory. & pause & exit /b 1)
if not defined BENCHMARK2_BASE_PROJECT set "BENCHMARK2_BASE_PROJECT=C:\Work\Projects\Prj_18_OCR_2"
if not defined BENCHMARK2_PORT set "BENCHMARK2_PORT=18018"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not installed. Run INSTALL_AND_RUN.bat or setup_benchmark2.bat.
    pause
    popd
    exit /b 1
)
if not exist ".venv\.benchmark2-root.txt" (
    echo [ERROR] Installation not finished. Run setup_benchmark2.bat again.
    pause
    popd
    exit /b 1
)
set /p "INSTALLED_AT=" < ".venv\.benchmark2-root.txt"
if /I not "!INSTALLED_AT!"=="%CD%" (
    echo [ERROR] The .venv was installed for a different folder: "!INSTALLED_AT!"
    echo [ACTION] Delete Benchmark2\.venv and run INSTALL_AND_RUN.bat in the NEW location.
    pause
    popd
    exit /b 1
)
echo [BENCHMARK2] Location: %CD%
echo [SOURCE]     %BENCHMARK2_BASE_PROJECT%
".venv\Scripts\python.exe" -m benchmark2 run
set "EXIT_CODE=!ERRORLEVEL!"
if not "!EXIT_CODE!"=="0" echo [ERROR] Check the terminal and results\latest.txt for diagnostics.
pause
popd
exit /b %EXIT_CODE%
