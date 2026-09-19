
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e .
python -m loghawk.admin.setup_duckdb
python -m loghawk.admin.setup_lancedb

