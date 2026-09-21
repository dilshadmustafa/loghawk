
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e .
python -m loghawk.admin.setup_duckdb
python -m loghawk.admin.setup_lancedb
streamlit run .\src\loghawk\docs_to_ragvectordb\upload_docs.py
python .\src\loghawk\anomaly_detection\chat_assistant.py



