from pathlib import Path
import os
import platform
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

LH_CORRELATION_WINDOW_MINUTES = int(
    os.getenv("LH_CORRELATION_WINDOW_MINUTES", "5")
)
if LH_CORRELATION_WINDOW_MINUTES < 1:
    raise ValueError("LH_CORRELATION_WINDOW_MINUTES must be at least 1")

LH_ANOMALY_BACKEND = os.getenv(
    "LH_ANOMALY_BACKEND",
    "sklearn",
).strip().lower()
if LH_ANOMALY_BACKEND not in {"sklearn", "cuml"}:
    raise ValueError("LH_ANOMALY_BACKEND must be 'sklearn' or 'cuml'")

LH_ANOMALY_ALGORITHM = os.getenv(
    "LH_ANOMALY_ALGORITHM", "isolation_forest"
).strip().lower()
if LH_ANOMALY_ALGORITHM not in {"isolation_forest"}:
    raise ValueError("LH_ANOMALY_ALGORITHM must currently be 'isolation_forest'")

LH_ANOMALY_DEVICE = os.getenv(
    "LH_ANOMALY_DEVICE",
    "gpu" if LH_ANOMALY_BACKEND == "cuml" else "cpu",
).strip().lower()
if LH_ANOMALY_DEVICE not in {"cpu", "gpu"}:
    raise ValueError("LH_ANOMALY_DEVICE must be 'cpu' or 'gpu'")

LH_TEMPORAL_ADDRESS = os.getenv(
    "LH_TEMPORAL_ADDRESS",
    "localhost:7233",
)

LH_OLLAMA_URL = os.getenv(
    "LH_OLLAMA_URL",
    "http://localhost:11434/api/chat",
)

LH_IDENTITY_MAPPING_SAMPLE_SIZE = int(
    os.getenv("LH_IDENTITY_MAPPING_SAMPLE_SIZE", "10")
)
if LH_IDENTITY_MAPPING_SAMPLE_SIZE < 1:
    raise ValueError("LH_IDENTITY_MAPPING_SAMPLE_SIZE must be at least 1")

LH_IDENTITY_MAPPING_SAMPLE_SEED = int(
    os.getenv("LH_IDENTITY_MAPPING_SAMPLE_SEED", "42")
)

LH_IDENTITY_MAPPING_SAMPLE_STRATEGY = os.getenv(
    "LH_IDENTITY_MAPPING_SAMPLE_STRATEGY",
    "reservoir",
).strip().lower()
if LH_IDENTITY_MAPPING_SAMPLE_STRATEGY not in {"reservoir", "first"}:
    raise ValueError(
        "LH_IDENTITY_MAPPING_SAMPLE_STRATEGY must be "
        "'reservoir' or 'first'"
    )

_skip_existing = os.getenv(
    "LH_IDENTITY_MAPPING_SKIP_EXISTING",
    "true",
).strip().lower()
if _skip_existing not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
    raise ValueError(
        "LH_IDENTITY_MAPPING_SKIP_EXISTING must be a boolean value"
    )
LH_IDENTITY_MAPPING_SKIP_EXISTING = _skip_existing in {
    "true", "1", "yes", "on"
}

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
)

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

def _env_bool(name: str, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    if value not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
        raise ValueError(f"{name} must be a boolean value")
    return value in {"true", "1", "yes", "on"}


LH_S3_SELECT_SUPPORTED = _env_bool(
    "LH_S3_SELECT_SUPPORTED",
    "false",
)
# Keep the legacy setting available for the existing v4/v6 modules.
LH_S3_SELECT_USE = _env_bool(
    "LH_S3_SELECT_USE",
    "false",
)
LH_S3_SELECT_USE_TRAIN_PHASE = _env_bool(
    "LH_S3_SELECT_USE_TRAIN_PHASE",
    "false",
)
LH_S3_SELECT_USE_DETECT_PHASE = _env_bool(
    "LH_S3_SELECT_USE_DETECT_PHASE",
    "false",
)

LH_S3_BATCH_FOLDER = os.getenv(
    "LH_S3_BATCH_FOLDER",
    "2026-09-28",
).strip().strip("/")
if not LH_S3_BATCH_FOLDER or "/" in LH_S3_BATCH_FOLDER or "\\" in LH_S3_BATCH_FOLDER:
    raise ValueError("LH_S3_BATCH_FOLDER must be one non-empty folder name")

LH_TRAIN_PHASE = _env_bool("LH_TRAIN_PHASE", "false")
LH_DETECT_PHASE = _env_bool("LH_DETECT_PHASE", "false")
if not LH_TRAIN_PHASE and not LH_DETECT_PHASE:
    raise ValueError(
        "At least one of LH_TRAIN_PHASE or LH_DETECT_PHASE must be true"
    )

LH_S3_SELECT_RECORD_FILTER_LIST = (
    []
    if LH_S3_SELECT_RECORD_FILTER.strip().upper() == "ALL"
    else [
        value.strip().upper()
        for value in LH_S3_SELECT_RECORD_FILTER.split(",")
        if value.strip()
    ]
)

_SYSTEM_TO_OS_FAMILY = {
    "Windows": "WINDOWS",
    "Linux": "LINUX",
    "Darwin": "MAC",
}

try:
    LH_OS_FAMILY = _SYSTEM_TO_OS_FAMILY[platform.system()]
except KeyError as exc:
    raise RuntimeError(
        f"Unsupported operating system: {platform.system()}"
    ) from exc

LH_JAVA_HOME_WINDOWS = os.getenv(
    "JAVA_HOME_WINDOWS",
    r"C:\jdk-17",
).strip()

LH_HADOOP_HOME_WINDOWS = os.getenv(
    "HADOOP_HOME_WINDOWS",
    r"C:\hadoop",
).strip()

# Shared Linux/Mac Java path, read from the .env variable JAVA_HOME.
LH_JAVA_HOME = os.getenv("JAVA_HOME", "").strip()




def main():
    print("LH DUCKDB FILE PATH : ", LH_DUCKDB_FILE_PATH)
    print("LH DUCKDB TABLE NAME : ", LH_DUCKDB_TABLE_NAME)
    print("LH S3 SELECT RECORD FILTER : ", LH_S3_SELECT_RECORD_FILTER)

if __name__ == '__main__':
    main()


