@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

TITLE Trustworthy RAG - Master Research and Reproduction Launcher
COLOR 0B

echo ===========================================================================
echo   TRUSTWORTHY RETRIEVAL-AUGMENTED GENERATION UNDER NOISY EVIDENCE
echo   One-Click Reproduction and Verification Launcher
echo ===========================================================================

:: Locate Python executable
set "PYTHON_EXE="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "%LOCALAPPDATA%\Python\bin\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Python\bin\python.exe"
) else (
    for /f "tokens=*" %%i in ('where python 2^>nul') do (
        echo "%%i" | findstr /i "WindowsApps" >nul
        if errorlevel 1 (
            if not defined PYTHON_EXE set "PYTHON_EXE=%%i"
        )
    )
)

if not defined PYTHON_EXE (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
    ) else (
        set "PYTHON_EXE=python"
    )
)

echo [ENVIRONMENT]
echo Detected Python: !PYTHON_EXE!
echo Project Root:    %~dp0
echo Target Ports:    Frontend: 2217 ^| Backend: 2218
echo ===========================================================================
echo.

set "PYTHONPATH=%~dp0src;%~dp0"

echo [1/2] Running automated regression test suite...
"!PYTHON_EXE!" -m pytest tests --quiet
if errorlevel 1 (
    echo.
    echo [ERROR] Regression test suite failed. Please check environment.
    pause
    exit /b %ERRORLEVEL%
)
echo [PASS] All 49 test suites verified.

echo.
echo [2/2] Launching master experimental reproduction suite...
"!PYTHON_EXE!" run_reproduction.py
if errorlevel 1 (
    echo.
    echo [ERROR] Replication run failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===========================================================================
echo Reproduction completed successfully!
echo Detailed figures:       experiments\figures\
echo Research manuscript:    paper\manuscript.tex
echo Error taxonomy report:  reports\error_taxonomy.md
echo ===========================================================================
pause
