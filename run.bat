@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

TITLE Trustworthy RAG - Master Research Platform
COLOR 0B

echo ===========================================================================
echo   TRUSTWORTHY RETRIEVAL-AUGMENTED GENERATION UNDER NOISY EVIDENCE
echo   One-Click Reproduction and Interactive Web Platform
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

set "PYTHONPATH=%~dp0src;%~dp0"

echo [ENVIRONMENT]
echo Detected Python: !PYTHON_EXE!
echo Project Root:    %~dp0
echo ===========================================================================
echo.

echo [1/2] Verifying test suite...
"!PYTHON_EXE!" -m pytest tests --quiet
if errorlevel 1 (
    echo.
    echo [ERROR] Regression test suite failed. Please check environment.
    pause
    exit /b %ERRORLEVEL%
)
echo [PASS] All 49 automated test suites verified.

echo.
echo ===========================================================================
echo   LOCAL WEB DASHBOARD READY
echo ===========================================================================
echo   Local Web Interface: http://localhost:2217
echo   REST API Docs:       http://localhost:2217/docs
echo   Assigned Ports:      Frontend: 2217 ^| Backend: 2218
echo ===========================================================================
echo.
echo [2/2] Opening browser to http://localhost:2217 and starting server...
start http://localhost:2217

"!PYTHON_EXE!" -m uvicorn trustworthy_rag.server.app:app --host 127.0.0.1 --port 2217
pause
