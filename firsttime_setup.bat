@echo off
setlocal
cd /d "%~dp0"

if not exist "%USERPROFILE%\tmp" mkdir "%USERPROFILE%\tmp"
set "TMPDIR=%USERPROFILE%\tmp"
set "TMP=%USERPROFILE%\tmp"
set "TEMP=%USERPROFILE%\tmp"
set "PIP_NO_CACHE_DIR=1"
set "PIP_TIMEOUT=120"
set "PIP_RETRIES=10"
set "PIP_RESUME_RETRIES=20"

if not defined SKIP_RUSTFS_SETUP set "SKIP_RUSTFS_SETUP=false"
if not defined SKIP_TEMPORAL_SETUP set "SKIP_TEMPORAL_SETUP=false"
if not defined SKIP_DUCKDB_SETUP set "SKIP_DUCKDB_SETUP=false"
if not defined SKIP_LANCEDB_SETUP set "SKIP_LANCEDB_SETUP=false"

echo.
echo ============================================================
echo Creating Python 3.12 virtual environment
echo ============================================================
python -c "import sys; print('Python:', sys.version.split()[0]); raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
if errorlevel 1 (
    echo Python 3.12 or newer is required. Install it or make it available as "python" on PATH.
    exit /b 1
)
set "PYTHON=%~dp0venv312\Scripts\python.exe"
if exist "%PYTHON%" (
    echo Existing virtual environment found; reusing it.
    "%PYTHON%" -c "import sys; print('Environment Python:', sys.version.split()[0]); raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
    if errorlevel 1 (
        echo Existing venv312 must use Python 3.12 or newer.
        exit /b 1
    )
) else (
    python -m venv venv312
    if errorlevel 1 goto :failed
)

echo.
echo ============================================================
echo Installing PySpark
echo ============================================================
"%PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed
"%PYTHON%" -m pip install pyspark==3.5.9
if errorlevel 1 goto :failed
"%PYTHON%" -c "from pyspark.sql import SparkSession; s=SparkSession.builder.master('local[*]').getOrCreate(); print('Spark:',s.version); print('Python:',__import__('sys').version); print('Java:',s.sparkContext._jvm.java.lang.System.getProperty('java.version')); print('Hadoop:',s.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion()); s.stop()"
if errorlevel 1 goto :failed
echo.
echo ============================================================
echo PyTorch - Explicit install - Installing PyTorch with CUDA 13.0 support
echo ============================================================
"%PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto :failed
echo.
echo ============================================================
echo Installing packages mentioned in requirements.txt
echo ============================================================
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
"%PYTHON%" -m pip install -e .
if errorlevel 1 goto :failed
echo.
echo ============================================================
echo Setting up DuckDB
echo ============================================================
if /I "%SKIP_DUCKDB_SETUP%"=="true" (
    echo Skipping DuckDB setup.
) else (
    "%PYTHON%" -m loghawk.admin.setup_duckdb
    if errorlevel 1 goto :failed
)
echo.
echo ============================================================
echo Setting up LanceDB
echo ============================================================
if /I "%SKIP_LANCEDB_SETUP%"=="true" (
    echo Skipping LanceDB setup.
) else (
    "%PYTHON%" -m loghawk.admin.setup_lancedb2
    if errorlevel 1 goto :failed
)

if /I "%SKIP_RUSTFS_SETUP%"=="true" (
    echo Skipping RustFS startup, bucket setup, and test-data upload.
) else (
echo.
echo ============================================================
echo Starting RustFS S3 server
echo ============================================================
if not exist "C:\rustfs\data" mkdir "C:\rustfs\data"
if not exist "C:\rustfs\logs" mkdir "C:\rustfs\logs"
docker container inspect loghawk-rustfs >nul 2>&1
if errorlevel 1 (
    docker run -d --name loghawk-rustfs -p 9000:9000 -p 9001:9001 -v "C:\rustfs\data:/data" -v "C:\rustfs\logs:/logs" rustfs/rustfs:latest
) else (
    docker start loghawk-rustfs
)
if errorlevel 1 goto :failed

set "AWS_ACCESS_KEY_ID=rustfsadmin"
set "AWS_SECRET_ACCESS_KEY=rustfsadmin"
if not defined AWS_DEFAULT_REGION set "AWS_DEFAULT_REGION=us-east-1"
echo.
echo ============================================================
echo Creating the loghawk-data S3 bucket
echo ============================================================
call :wait_for_rustfs
if errorlevel 1 goto :failed
aws --endpoint-url http://localhost:9000 s3api head-bucket --bucket loghawk-data >nul 2>&1
if errorlevel 1 (
    aws --endpoint-url http://localhost:9000 s3 mb s3://loghawk-data
    if errorlevel 1 goto :failed
) else (
    echo Bucket loghawk-data already exists.
)

echo.
echo ============================================================
echo Generating and uploading test data to S3
echo ============================================================
"%PYTHON%" "%~dp0src\loghawk\test\testdata.py"
if errorlevel 1 goto :failed
)

if /I "%SKIP_TEMPORAL_SETUP%"=="true" (
    echo Skipping Temporal server startup.
) else (
echo.
echo ============================================================
echo Starting Temporal server
echo ============================================================
docker-compose -f src\loghawk\admin\temporal-docker-compose.yml up -d
if errorlevel 1 goto :failed
)

echo Setup complete.
echo Activate the environment:
echo   PowerShell: .\venv312\Scripts\Activate.ps1
echo   Command Prompt: venv312\Scripts\activate.bat
echo RustFS console: http://localhost:9001
echo Temporal UI: http://localhost:8233
echo Start the worker and workflow using the firsttime_start_workflow scripts.
exit /b 0

:failed
echo Setup failed. Review the error above.
exit /b 1

:wait_for_rustfs
set "RUSTFS_ATTEMPT=0"
:wait_for_rustfs_retry
aws --endpoint-url http://localhost:9000 s3api list-buckets >nul 2>&1
if not errorlevel 1 exit /b 0
set /a RUSTFS_ATTEMPT+=1
if %RUSTFS_ATTEMPT% GEQ 30 goto :wait_for_rustfs_timeout
timeout /t 2 /nobreak >nul
goto :wait_for_rustfs_retry

:wait_for_rustfs_timeout
echo RustFS did not become ready within 60 seconds.
docker logs --tail 50 loghawk-rustfs
exit /b 1
