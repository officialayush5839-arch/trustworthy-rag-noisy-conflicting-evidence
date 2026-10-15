@echo off
TITLE Trustworthy RAG - Master Research & Reproduction Launcher
COLOR 0B

echo ===========================================================================
echo   TRUSTWORTHY RETRIEVAL-AUGMENTED GENERATION UNDER NOISY EVIDENCE
echo   One-Click Reproduction & Verification Launcher
echo ===========================================================================
echo [ENVIRONMENT]
echo Python Environment: Active System Python
echo Target Frontend Port: 2217
echo Target Backend Port:  2218
echo ===========================================================================
echo.

set PYTHONPATH=src;.

echo [1/2] Running automated regression test suite...
python -m pytest tests --quiet
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Regression test suite failed. Please check environment.
    pause
    exit /b %ERRORLEVEL%
)
echo [PASS] All test suites verified.

echo.
echo [2/2] Launching master experimental reproduction suite...
python run_reproduction.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Replication run failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===========================================================================
echo Reproduction completed successfully!
echo Detailed figures: experiments/figures/
echo Research manuscript: paper/manuscript.tex
echo Error taxonomy report: reports/error_taxonomy.md
echo ===========================================================================
pause
