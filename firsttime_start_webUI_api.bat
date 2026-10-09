@echo off
setlocal
cd /d "%~dp0"

if not exist "venv312\Scripts\python.exe" (
    echo Run firsttime_setup.bat before starting the LogHawk API.
    exit /b 1
)

venv312\Scripts\python.exe -m uvicorn loghawk.api.main:app --app-dir src --host 127.0.0.1 --port 8000
