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

echo.
echo ============================================================
echo Setting up LogHawk tools Python environment
echo ============================================================
python -c "import sys; print('Python:', sys.version.split()[0]); raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
if errorlevel 1 (
    echo Python 3.12 or newer must be available as "python" on PATH.
    exit /b 1
)
set "PYTHON=%~dp0venv_tools\Scripts\python.exe"
if exist "%PYTHON%" (
    echo Reusing existing venv_tools environment.
    "%PYTHON%" -c "import sys; print('Environment Python:', sys.version.split()[0]); raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
    if errorlevel 1 (
        echo Existing venv_tools must use Python 3.12 or newer.
        exit /b 1
    )
) else (
    python -m venv venv_tools
    if errorlevel 1 goto :failed
)

"%PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed
echo.
echo ============================================================
echo Installing optional packages from requirements_tools.txt
echo ============================================================
"%PYTHON%" -m pip install -r requirements_tools.txt
if errorlevel 1 goto :failed
"%PYTHON%" -m pip install -e . --no-deps
if errorlevel 1 goto :failed
if not exist ".env_tools" if exist ".env_tools.example" copy ".env_tools.example" ".env_tools" >nul
echo.
echo Tools environment setup complete.
echo Activate it with: .\venv_tools\Scripts\Activate.ps1
exit /b 0

:failed
echo Tools setup failed. Review the error above.
exit /b 1
