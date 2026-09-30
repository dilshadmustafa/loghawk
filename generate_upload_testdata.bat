@echo off
setlocal
cd /d "%~dp0"

if not exist "venv312\Scripts\python.exe" (
    echo Run firsttime_setup_terminal_1.bat first.
    exit /b 1
)

"venv312\Scripts\python.exe" "%~dp0src\loghawk\test\testdata.py"
exit /b %ERRORLEVEL%
