@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=%~dp0venv_hawkconsole"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

echo.
echo ============================================================
echo Setting up Hawk Console Python environment
echo ============================================================
set "BOOTSTRAP_PYTHON="
for /f "delims=" %%P in ('where.exe python 2^>nul') do (
    "%%P" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
    if not errorlevel 1 set "BOOTSTRAP_PYTHON=%%P"
)
if not defined BOOTSTRAP_PYTHON (
    echo Python 3.12 or newer is required and must be available on PATH.
    exit /b 1
)
"%BOOTSTRAP_PYTHON%" -c "import sys; print('Python:', sys.version.split()[0])"

if not exist "%PYTHON%" (
    "%BOOTSTRAP_PYTHON%" -m venv "%VENV_DIR%"
    if errorlevel 1 goto :failed
) else (
    "%PYTHON%" -c "import sys; print('Environment Python:', sys.version.split()[0]); raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
    if errorlevel 1 (
        echo Existing venv_hawkconsole must use Python 3.12 or newer.
        exit /b 1
    )
)

echo.
echo ============================================================
echo Installing Hawk Console packages
echo ============================================================
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%PYTHON%" -m pip install -r "%~dp0requirements-hawkconsole.txt"
if errorlevel 1 goto :failed
"%PYTHON%" -m pip install --no-deps -e .
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo Setting up LanceDB
echo ============================================================
"%PYTHON%" -m loghawk.admin.setup_lancedb2
if errorlevel 1 goto :failed

echo.
echo Hawk Console setup complete.
echo Activate with:
echo   PowerShell: .\venv_hawkconsole\Scripts\Activate.ps1
echo   Command Prompt: venv_hawkconsole\Scripts\activate.bat
echo Then run: hawk
exit /b 0

:failed
echo Setup failed. Review the error above.
exit /b 1
