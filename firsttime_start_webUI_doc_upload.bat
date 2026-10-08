@echo off
setlocal
cd /d "%~dp0"

if not exist "venv312\Scripts\streamlit.exe" (
    echo Run firsttime_setup.bat before starting the document upload UI.
    exit /b 1
)

venv312\Scripts\streamlit.exe run src\loghawk\docs_to_ragvectordb\upload_docs2.py
