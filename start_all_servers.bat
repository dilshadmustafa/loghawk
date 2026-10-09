@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "LOGHAWK_ROOT=%CD%"
set "PYTHON=%LOGHAWK_ROOT%\venv312\Scripts\python.exe"

echo ============================================================
echo Starting LogHawk services
echo ============================================================

if not exist "%PYTHON%" (
    echo Run firsttime_setup.bat before starting LogHawk.
    exit /b 1
)
where docker >nul 2>nul || (echo Docker is required and must be running.& exit /b 1)
where docker-compose >nul 2>nul || (echo docker-compose is required.& exit /b 1)
where aws >nul 2>nul || (echo AWS CLI is required for the RustFS readiness check.& exit /b 1)

echo.
echo ============================================================
echo Checking RustFS
echo ============================================================
docker container inspect loghawk-rustfs >nul 2>&1
if errorlevel 1 (
    if not exist "C:\rustfs\data" mkdir "C:\rustfs\data"
    if not exist "C:\rustfs\logs" mkdir "C:\rustfs\logs"
    docker run -d --name loghawk-rustfs -p 9000:9000 -p 9001:9001 -v "C:\rustfs\data:/data" -v "C:\rustfs\logs:/logs" rustfs/rustfs:latest
    if errorlevel 1 exit /b 1
) else (
    for /f "delims=" %%R in ('docker inspect -f "{{.State.Running}}" loghawk-rustfs') do set "RUSTFS_RUNNING=%%R"
    if /I not "!RUSTFS_RUNNING!"=="true" docker start loghawk-rustfs
    if errorlevel 1 exit /b 1
)

set "AWS_ACCESS_KEY_ID=rustfsadmin"
set "AWS_SECRET_ACCESS_KEY=rustfsadmin"
if not defined AWS_DEFAULT_REGION set "AWS_DEFAULT_REGION=us-east-1"
set "RUSTFS_READY=false"
for /L %%I in (1,1,30) do (
    aws --endpoint-url http://localhost:9000 s3api list-buckets >nul 2>&1
    if not errorlevel 1 (
        set "RUSTFS_READY=true"
        goto :rustfs_ready
    )
    timeout /t 2 /nobreak >nul
)
:rustfs_ready
if /I not "%RUSTFS_READY%"=="true" (
    echo RustFS did not become ready within 60 seconds.
    docker logs --tail 50 loghawk-rustfs
    exit /b 1
)
echo RustFS is ready at http://localhost:9000

echo.
echo ============================================================
echo Checking Temporal
echo ============================================================
docker container inspect temporal >nul 2>&1
if errorlevel 1 (
    docker-compose -f src\loghawk\admin\temporal-docker-compose.yml up -d
    if errorlevel 1 exit /b 1
) else (
    set "TEMPORAL_RUNNING=false"
    for /f "delims=" %%R in ('docker inspect -f "{{.State.Running}}" temporal') do set "TEMPORAL_RUNNING=%%R"
    if /I not "!TEMPORAL_RUNNING!"=="true" docker start temporal
    if errorlevel 1 exit /b 1
)

set "TEMPORAL_READY=false"
for /L %%I in (1,1,30) do (
    powershell -NoProfile -Command "$c=New-Object Net.Sockets.TcpClient; try {$c.Connect('127.0.0.1',7233); exit 0} catch {exit 1} finally {$c.Dispose()}" >nul 2>&1
    if not errorlevel 1 (
        set "TEMPORAL_READY=true"
        goto :temporal_ready
    )
    timeout /t 2 /nobreak >nul
)
:temporal_ready
if /I not "%TEMPORAL_READY%"=="true" (
    echo Temporal did not become ready on port 7233 within 60 seconds.
    docker logs --tail 50 temporal
    exit /b 1
)
echo Temporal is ready at http://localhost:8233

echo.
echo ============================================================
echo Checking Node.js and npm
echo ============================================================
where node >nul 2>nul
set "NODE_MISSING=!errorlevel!"
where npm >nul 2>nul
if not "!NODE_MISSING!"=="0" goto :install_node
if errorlevel 1 goto :install_node
goto :check_node_version

:install_node
where winget >nul 2>nul
if errorlevel 1 (
    echo winget is unavailable. Install Node.js LTS from https://nodejs.org/en/download/ and rerun this script.
    exit /b 1
)
winget install --id OpenJS.NodeJS.LTS --exact --accept-source-agreements --accept-package-agreements
if errorlevel 1 exit /b 1
set "PATH=%ProgramFiles%\nodejs;%LOCALAPPDATA%\Programs\nodejs;%PATH%"
where node >nul 2>nul || (echo Node.js installation completed, but node is not available on PATH.& exit /b 1)
where npm >nul 2>nul || (echo Node.js installation completed, but npm is not available on PATH.& exit /b 1)

:check_node_version
node -e "const [major,minor]=process.versions.node.split('.').map(Number);process.exit((major===20&&minor>=19)||(major>=22&&(major>22||minor>=12))?0:1)"
if errorlevel 1 (
    echo This UI requires Node.js ^20.19.0 or ^22.12.0 and newer. Update Node.js, then rerun this script.
    exit /b 1
)

echo.
echo ============================================================
echo Installing LogHawk Web UI packages
echo ============================================================
pushd "%LOGHAWK_ROOT%\frontend\loghawk-ui"
call npm install
if errorlevel 1 (popd& exit /b 1)
popd

echo.
echo ============================================================
echo Starting LogHawk worker, API, and Vite
echo ============================================================
start "LogHawk Temporal Worker" /D "%LOGHAWK_ROOT%" "%PYTHON%" -m loghawk.workflows.temporal.worker
start "LogHawk Web UI API" /D "%LOGHAWK_ROOT%" "%PYTHON%" -m uvicorn loghawk.api.main:app --app-dir src --host 127.0.0.1 --port 8000
start "LogHawk Web UI - Vite" /D "%LOGHAWK_ROOT%\frontend\loghawk-ui" cmd /k "npm run dev"

echo Worker, API, and Vite were started in separate windows.
echo Web UI: http://localhost:5173
echo Web UI API: http://localhost:8000
echo RustFS console: http://localhost:9001
echo Temporal UI: http://localhost:8233
exit /b 0
