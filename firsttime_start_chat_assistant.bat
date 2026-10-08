@echo off
setlocal
cd /d "%~dp0"

if not exist "venv312\Scripts\python.exe" (
    echo Run firsttime_setup.bat before starting the chat assistant.
    exit /b 1
)

venv312\Scripts\python.exe src\loghawk\chat_assistant\chat_assistant5.py
