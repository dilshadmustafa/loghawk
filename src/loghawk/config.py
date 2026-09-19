from pathlib import Path
import os
from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

LH_DUCKDB_FILE_PATH = Path(os.getenv("LH_DUCKDB_FILE_PATH"))

if not LH_DUCKDB_FILE_PATH.is_absolute():
    LH_DUCKDB_FILE_PATH = PROJECT_ROOT / LH_DUCKDB_FILE_PATH

LH_DUCKDB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

LH_DATA_DIR = Path(os.getenv("LH_DATA_DIR"))

if not LH_DATA_DIR.is_absolute():
    LH_DATA_DIR = PROJECT_ROOT / LH_DATA_DIR

LH_DATA_DIR.parent.mkdir(parents=True, exist_ok=True)

LH_DUCKDB_TABLE_NAME = os.getenv(
    "LH_DUCKDB_TABLE_NAME",
    "convo"
)

def main():
    print("LH DUCKDB FILE PATH : ", LH_DUCKDB_FILE_PATH)
    print("LH DUCKDB TABLE NAME : ", LH_DUCKDB_TABLE_NAME)

if __name__ == '__main__':
    main()


