@echo off
setlocal
cd /d "%~dp0frontend\loghawk-ui"

where node >nul 2>nul
if errorlevel 1 (
    echo Node.js is not installed or is not on PATH.
    echo Install Node.js, then open a new terminal and retry.
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo npm is not installed or is not on PATH.
    echo Install Node.js, then open a new terminal and retry.
    exit /b 1
)

echo Installing LogHawk Web UI packages...
call npm install
if errorlevel 1 exit /b 1

echo Starting LogHawk Web UI with Vite...
call npm run dev
exit /b %errorlevel%
