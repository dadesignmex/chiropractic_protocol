@echo off
echo ========================================
echo  New Chiropractic Protocol - Full Pipeline
echo ========================================
echo.

cd /d "%~dp0code"

echo [1/7] Importing raw data...
python 00_import_raw.py
if %errorlevel% neq 0 (
    echo ERROR in 00_import_raw.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [2/7] Preprocessing...
python 01_preprocess.py
if %errorlevel% neq 0 (
    echo ERROR in 01_preprocess.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [3/7] Descriptive statistics...
python 02_descriptive_stats.py
if %errorlevel% neq 0 (
    echo ERROR in 02_descriptive_stats.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [4/7] RQ1 - VAS change analysis...
python 03_rq1_vas_change.py
if %errorlevel% neq 0 (
    echo ERROR in 03_rq1_vas_change.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [5/7] RQ2 - Success rate vs benchmark...
python 04_rq2_success_benchmark.py
if %errorlevel% neq 0 (
    echo ERROR in 04_rq2_success_benchmark.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [6/7] RQ3 - Logistic regression...
python 05_rq3_logistic_regression.py
if %errorlevel% neq 0 (
    echo ERROR in 05_rq3_logistic_regression.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [7/7] RQ4 - Machine learning model...
python 06_rq4_ml_model.py
if %errorlevel% neq 0 (
    echo ERROR in 06_rq4_ml_model.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================
echo  Pipeline completed successfully!
echo  Check processLog.txt for full output.
echo ========================================
pause