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

LH_LANCEDB_FILE_PATH = Path(os.getenv("LH_LANCEDB_FILE_PATH"))

if not LH_LANCEDB_FILE_PATH.is_absolute():
    LH_LANCEDB_FILE_PATH = PROJECT_ROOT / LH_LANCEDB_FILE_PATH

LH_LANCEDB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

LH_LANCEDB_TABLE_NAME = os.getenv(
    "LH_LANCEDB_TABLE_NAME",
    "loghawk"
)

LH_DOCS_STORAGE_DIR_PATH = Path(os.getenv("LH_DOCS_STORAGE_DIR_PATH"))

if not LH_DOCS_STORAGE_DIR_PATH.is_absolute():
    LH_DOCS_STORAGE_DIR_PATH = PROJECT_ROOT / LH_DOCS_STORAGE_DIR_PATH

LH_DOCS_STORAGE_DIR_PATH.parent.mkdir(parents=True, exist_ok=True)

LH_EMBEDDING_MODEL = os.getenv(
    "LH_EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

LH_LLM_MODEL = os.getenv(
    "LH_LLM_MODEL",
    "gemma2:latest"
)

LH_LOG_DIR = Path(os.getenv("LH_LOG_DIR"))

if not LH_LOG_DIR.is_absolute():
    LH_LOG_DIR = PROJECT_ROOT / LH_LOG_DIR

LH_LOG_DIR.parent.mkdir(parents=True, exist_ok=True)

LH_FEATURE_DIR = Path(os.getenv("LH_FEATURE_DIR"))

if not LH_FEATURE_DIR.is_absolute():
    LH_FEATURE_DIR = PROJECT_ROOT / LH_FEATURE_DIR

LH_FEATURE_DIR.parent.mkdir(parents=True, exist_ok=True)

LH_ANOMALY_DETECTION_DIR = Path(os.getenv("LH_ANOMALY_DETECTION_DIR"))

if not LH_ANOMALY_DETECTION_DIR.is_absolute():
    LH_ANOMALY_DETECTION_DIR = PROJECT_ROOT / LH_ANOMALY_DETECTION_DIR

LH_ANOMALY_DETECTION_DIR.parent.mkdir(parents=True, exist_ok=True)

LH_MODEL_DIR = Path(os.getenv("LH_MODEL_DIR"))

if not LH_MODEL_DIR.is_absolute():
    LH_MODEL_DIR = PROJECT_ROOT / LH_MODEL_DIR

LH_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

LH_S3_ENDPOINT = os.getenv(
    "LH_S3_ENDPOINT",
    "http://localhost:9000"
)

LH_S3_BUCKET= os.getenv(
    "LH_S3_BUCKET",
    "loghawk-data"

LH_S3_ACCESS_KEY_ID= os.getenv(
    "AWS_ACCESS_KEY_ID",
    "rustfsadmin"
)

LH_S3_SECRET_ACCESS_KEY= os.getenv(
    "AWS_SECRET_ACCESS_KEY",
    "rustfsadmin"
)

LH_S3_REGION= os.getenv(
    "AWS_REGION",
    "us-east-1"
)

LH_S3_SELECT_RECORD_FILTER = os.getenv(
    "LH_S3_SELECT_RECORD_FILTER",
    "WARN,ERROR"
)

LH_S3_SELECT_RECORD_FILTER_LIST = [x.strip() for x in LH_S3_SELECT_RECORD_FILTER.split(",")]


def main():
    print("LH DUCKDB FILE PATH : ", LH_DUCKDB_FILE_PATH)
    print("LH DUCKDB TABLE NAME : ", LH_DUCKDB_TABLE_NAME)
    print("LH S3 SELECT RECORD FILTER : ", LH_S3_SELECT_RECORD_FILTER)

if __name__ == '__main__':
    main()


