@echo off
echo ========================================
echo  New Chiropractic Protocol - Full Pipeline
echo ========================================
echo.

cd /d "%~dp0code"

echo [1/11] Importing raw data...
python 00_import_raw.py
if %errorlevel% neq 0 (
    echo ERROR in 00_import_raw.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [2/11] Preprocessing...
python 01_preprocess.py
if %errorlevel% neq 0 (
    echo ERROR in 01_preprocess.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [3/11] Descriptive statistics...
python 02_descriptive_stats.py
if %errorlevel% neq 0 (
    echo ERROR in 02_descriptive_stats.py - Aborting.
    pause
    exit /b %errorlevel%
)
echo [4/11] EDA ...
python 02_01_eda.py
if %errorlevel% neq 0 (
    echo ERROR in 02_01_eda.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [5/11] RQ1 - VAS change analysis...
python 03_rq1_vas_change.py
if %errorlevel% neq 0 (
    echo ERROR in 03_rq1_vas_change.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [6/11] RQ2 - Success rate vs benchmark...
python 04_rq2_success_benchmark.py
if %errorlevel% neq 0 (
    echo ERROR in 04_rq2_success_benchmark.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [7/11] RQ3 - Logistic regression...
python 05_rq3_logistic_regression.py
if %errorlevel% neq 0 (
    echo ERROR in 05_rq3_logistic_regression.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [8/11] RQ4 - Machine learning model...
python 06_rq4_ml_model.py
if %errorlevel% neq 0 (
    echo ERROR in 06_rq4_ml_model.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [9/11] RQ4 - LightGBM model...
python 08_rq4_advanced.py
if %errorlevel% neq 0 (
    echo ERROR in 08_rq4_advanced.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo [10/11] RQ4 - Diagnose predictions...
python 09_diagnose_predictions.py
if %errorlevel% neq 0 (
    echo ERROR in 09_diagnose_predictions.py - Aborting.
    pause
    exit /b %errorlevel%
)
 
echo [11/11] RQ4 - SHAP interpretability...
python 10_rq4_shap.py
if %errorlevel% neq 0 (
    echo ERROR in 10_rq4_shap.py - Aborting.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================
echo  Pipeline completed successfully!
echo  Check processLog.txt for full output.
echo ========================================
pause
