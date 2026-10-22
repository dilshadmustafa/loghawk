from pathlib import Path
import fnmatch
import json
import os
import platform
from urllib.parse import urlparse
from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

LH_API_SERVER_BIND_URL = os.getenv(
    "LH_API_SERVER_BIND_URL", "http://127.0.0.1:8000"
).strip().rstrip("/")
LH_API_SERVER_URL = os.getenv(
    "LH_API_SERVER_URL", "http://127.0.0.1:8000"
).strip().rstrip("/")

for _api_url_name, _api_url in (
    ("LH_API_SERVER_BIND_URL", LH_API_SERVER_BIND_URL),
    ("LH_API_SERVER_URL", LH_API_SERVER_URL),
):
    _parsed_api_url = urlparse(_api_url)
    try:
        _api_url_port = _parsed_api_url.port
    except ValueError as exc:
        raise ValueError(f"{_api_url_name} must contain a valid port") from exc
    if (
        _parsed_api_url.scheme not in {"http", "https"}
        or not _parsed_api_url.hostname
        or _api_url_port is None
        or _parsed_api_url.path not in {"", "/"}
        or _parsed_api_url.query
        or _parsed_api_url.fragment
    ):
        raise ValueError(
            f"{_api_url_name} must be an HTTP(S) URL with host and port, "
            "for example http://127.0.0.1:8000"
        )

_duckdb_file_path = os.getenv("LH_DUCKDB_FILE_PATH", "").strip()
if _duckdb_file_path:
    LH_DUCKDB_FILE_PATH = Path(_duckdb_file_path)
    if not LH_DUCKDB_FILE_PATH.is_absolute():
        LH_DUCKDB_FILE_PATH = PROJECT_ROOT / LH_DUCKDB_FILE_PATH
    LH_DUCKDB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
else:
    LH_DUCKDB_FILE_PATH = Path(":memory:")

LH_DATA_DIR = Path(os.getenv("LH_DATA_DIR"))

if not LH_DATA_DIR.is_absolute():
    LH_DATA_DIR = PROJECT_ROOT / LH_DATA_DIR

LH_DATA_DIR.parent.mkdir(parents=True, exist_ok=True)

LH_CONSOLE_DATABASE_URL = os.getenv(
    "LH_CONSOLE_DATABASE_URL",
    "sqlite:///data/db/loghawk_local_hawkconsole.sqlite3",
).strip()
LH_SERVER_DATABASE_URL = os.getenv(
    "LH_SERVER_DATABASE_URL",
    "sqlite:///data/db/loghawk_server.sqlite3",
).strip()
if not LH_CONSOLE_DATABASE_URL:
    raise ValueError("LH_CONSOLE_DATABASE_URL must not be blank")
if not LH_SERVER_DATABASE_URL:
    raise ValueError("LH_SERVER_DATABASE_URL must not be blank")

LH_DUCKDB_TABLE_NAME = os.getenv(
    "LH_DUCKDB_TABLE_NAME",
    "convo"
)

LH_SERVER_LANCEDB_FILE_PATH = Path(
    os.getenv("LH_SERVER_LANCEDB_FILE_PATH", "data/db/serverlancedb").strip()
)

if not LH_SERVER_LANCEDB_FILE_PATH.is_absolute():
    LH_SERVER_LANCEDB_FILE_PATH = PROJECT_ROOT / LH_SERVER_LANCEDB_FILE_PATH

LH_SERVER_LANCEDB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

LH_SERVER_LANCEDB_TABLE_NAME = os.getenv(
    "LH_SERVER_LANCEDB_TABLE_NAME", "loghawk"
).strip()

# Compatibility for older shared/server scripts.
LH_LANCEDB_FILE_PATH = LH_SERVER_LANCEDB_FILE_PATH
LH_LANCEDB_TABLE_NAME = LH_SERVER_LANCEDB_TABLE_NAME

LH_CONSOLE_LANCEDB_FILE_PATH = Path(
    os.getenv("LH_CONSOLE_LANCEDB_FILE_PATH", "data/db/locallancedb")
)
if not LH_CONSOLE_LANCEDB_FILE_PATH.is_absolute():
    LH_CONSOLE_LANCEDB_FILE_PATH = PROJECT_ROOT / LH_CONSOLE_LANCEDB_FILE_PATH
LH_CONSOLE_LANCEDB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

LH_CONSOLE_LANCEDB_TABLE_NAME = os.getenv(
    "LH_CONSOLE_LANCEDB_TABLE_NAME",
    "loghawk"
)

LH_DOCS_STORAGE_DIR_PATH = Path(os.getenv("LH_DOCS_STORAGE_DIR_PATH"))

if not LH_DOCS_STORAGE_DIR_PATH.is_absolute():
    LH_DOCS_STORAGE_DIR_PATH = PROJECT_ROOT / LH_DOCS_STORAGE_DIR_PATH

LH_DOCS_STORAGE_DIR_PATH.parent.mkdir(parents=True, exist_ok=True)

# RAG Web UI upload and bounded website crawl limits.
LH_RAG_MAX_UPLOAD_MB = int(os.getenv("LH_RAG_MAX_UPLOAD_MB", "100"))
LH_RAG_CRAWL_MAX_PAGES = int(os.getenv("LH_RAG_CRAWL_MAX_PAGES", "50"))
LH_RAG_CRAWL_MAX_DEPTH = int(os.getenv("LH_RAG_CRAWL_MAX_DEPTH", "2"))
LH_RAG_CRAWL_DELAY_SECONDS = float(os.getenv("LH_RAG_CRAWL_DELAY_SECONDS", "1"))
LH_RAG_CRAWL_MAX_FILE_MB = int(os.getenv("LH_RAG_CRAWL_MAX_FILE_MB", "25"))
LH_RAG_CRAWL_MAX_TOTAL_MB = int(os.getenv("LH_RAG_CRAWL_MAX_TOTAL_MB", "100"))
if LH_RAG_MAX_UPLOAD_MB < 1:
    raise ValueError("LH_RAG_MAX_UPLOAD_MB must be at least 1")
if LH_RAG_CRAWL_MAX_PAGES < 1 or LH_RAG_CRAWL_MAX_DEPTH < 0:
    raise ValueError("RAG crawl page limit must be positive and depth non-negative")
if LH_RAG_CRAWL_DELAY_SECONDS < 0:
    raise ValueError("LH_RAG_CRAWL_DELAY_SECONDS must be zero or greater")
if LH_RAG_CRAWL_MAX_FILE_MB < 1 or LH_RAG_CRAWL_MAX_TOTAL_MB < 1:
    raise ValueError("RAG crawl file and total size limits must be at least 1 MB")

LH_EMBEDDING_MODEL = os.getenv(
    "LH_EMBEDDING_MODEL",
    "ollama/nomic-embed-text"
)

LH_LLM_MODEL = os.getenv(
    "LH_LLM_MODEL",
    "gemma2:latest"
)

LH_LLM_CODEASSIST_MODEL = (
    os.getenv("LH_LLM_CODEASSIST_MODEL", "").strip()
    or LH_LLM_MODEL
)
LH_LLM_CODEASSIST_MAX_TOKENS = int(
    os.getenv("LH_LLM_CODEASSIST_MAX_TOKENS", "4096")
)
if LH_LLM_CODEASSIST_MAX_TOKENS < 1:
    raise ValueError("LH_LLM_CODEASSIST_MAX_TOKENS must be at least 1")

# Number of recent LLM Terminal messages included when drafting commands.
LH_CONSOLE_COMMAND_CONTEXT_MESSAGES = int(
    os.getenv("LH_CONSOLE_COMMAND_CONTEXT_MESSAGES", "10")
)
if LH_CONSOLE_COMMAND_CONTEXT_MESSAGES < 0:
    raise ValueError("LH_CONSOLE_COMMAND_CONTEXT_MESSAGES must be zero or greater")

LH_LLM_PROVIDER = os.getenv("LH_LLM_PROVIDER", "ollama").strip().lower()
if LH_LLM_PROVIDER == "microsoft_foundry":
    LH_LLM_PROVIDER = "azure_ai"
if LH_LLM_PROVIDER not in {"ollama", "bedrock", "azure_ai"}:
    raise ValueError(
        "LH_LLM_PROVIDER must be one of: ollama, bedrock, azure_ai "
        "(microsoft_foundry is an alias for azure_ai)"
    )

LH_LLM_BASE_URL = os.getenv(
    "LH_LLM_BASE_URL",
    "http://localhost:11434" if LH_LLM_PROVIDER == "ollama" else "",
).strip()
LH_LLM_API_KEY = os.getenv("LH_LLM_API_KEY", "").strip()
LH_LLM_TIMEOUT_SECONDS = int(os.getenv("LH_LLM_TIMEOUT_SECONDS", "300"))
if LH_LLM_TIMEOUT_SECONDS < 1:
    raise ValueError("LH_LLM_TIMEOUT_SECONDS must be at least 1")

_raw_llm_fallback_models = os.getenv("LH_LLM_FALLBACK_MODELS", "").strip()
if _raw_llm_fallback_models:
    LH_LLM_FALLBACK_MODELS = tuple(
        model.strip()
        for model in _raw_llm_fallback_models.split(",")
        if model.strip()
    )
elif LH_LLM_PROVIDER == "ollama":
    LH_LLM_FALLBACK_MODELS = (
        "llama3.2:1b",
        "qwen2.5:0.5b",
        "tinyllama",
    )
else:
    LH_LLM_FALLBACK_MODELS = ()

LH_EMBEDDING_BASE_URL = os.getenv(
    "LH_EMBEDDING_BASE_URL",
    LH_LLM_BASE_URL,
).strip()
LH_EMBEDDING_API_KEY = os.getenv(
    "LH_EMBEDDING_API_KEY",
    LH_LLM_API_KEY,
).strip()
LH_EMBEDDING_BATCH_SIZE = int(os.getenv("LH_EMBEDDING_BATCH_SIZE", "32"))
if LH_EMBEDDING_BATCH_SIZE < 1:
    raise ValueError("LH_EMBEDDING_BATCH_SIZE must be at least 1")

LH_CORRELATION_WINDOW_MINUTES = int(
    os.getenv("LH_CORRELATION_WINDOW_MINUTES", "5")
)
if LH_CORRELATION_WINDOW_MINUTES < 1:
    raise ValueError("LH_CORRELATION_WINDOW_MINUTES must be at least 1")

LH_ANOMALY_BACKEND = os.getenv(
    "LH_ANOMALY_BACKEND",
    "sklearn",
).strip().lower()
if (
    LH_ANOMALY_BACKEND not in {"sklearn", "cuml", "pyod"}
    and not os.getenv("LH_ANOMALY_ALGORITHMS", "").strip()
):
    raise ValueError("LH_ANOMALY_BACKEND must be 'sklearn', 'cuml', or 'pyod'")

_ANOMALY_ALGORITHM_RUNTIME = {
    "sklearn-isolationforest": ("sklearn", "cpu"),
    "cuml-isolationforest": ("cuml", "gpu"),
    "pyod-isolationforest": ("pyod", "cpu"),
    "pyod-copod": ("pyod", "cpu"),
    "pyod-ecod": ("pyod", "cpu"),
    "pyod-hbos": ("pyod", "cpu"),
    "pyod-knn": ("pyod", "cpu"),
    "pyod-lof": ("pyod", "cpu"),
    "pyod-ocsvm": ("pyod", "cpu"),
    "pyod-pca": ("pyod", "cpu"),
    "pyod-autoencoder": ("pyod", "cpu"),
    "pyod-autoencoder-gpu": ("pyod", "gpu"),
}
_raw_anomaly_algorithms = os.getenv("LH_ANOMALY_ALGORITHMS", "").strip()
if _raw_anomaly_algorithms:
    LH_ANOMALY_ALGORITHMS = tuple(
        item.strip().lower() for item in _raw_anomaly_algorithms.split(",")
    )
    if any(not item for item in LH_ANOMALY_ALGORITHMS):
        raise ValueError("LH_ANOMALY_ALGORITHMS contains an empty item")
    if len(set(LH_ANOMALY_ALGORITHMS)) != len(LH_ANOMALY_ALGORITHMS):
        raise ValueError("LH_ANOMALY_ALGORITHMS contains duplicate IDs")
    unknown_algorithms = sorted(
        set(LH_ANOMALY_ALGORITHMS) - set(_ANOMALY_ALGORITHM_RUNTIME)
    )
    if unknown_algorithms:
        raise ValueError(
            "Unsupported LH_ANOMALY_ALGORITHMS ID(s): "
            + ", ".join(unknown_algorithms)
        )
else:
    LH_ANOMALY_ALGORITHMS = ()

_DEFAULT_ANOMALY_ALGORITHM = {
    "sklearn": "sklearn-isolationforest",
    "cuml": "cuml-isolationforest",
    "pyod": "pyod-copod",
}.get(LH_ANOMALY_BACKEND, "sklearn-isolationforest")
LH_ANOMALY_ALGORITHM = os.getenv(
    "LH_ANOMALY_ALGORITHM", _DEFAULT_ANOMALY_ALGORITHM
).strip().lower()
_ALGORITHMS_BY_BACKEND = {
    "sklearn": {"sklearn-isolationforest"},
    "cuml": {"cuml-isolationforest"},
    "pyod": {
        "pyod-isolationforest",
        "pyod-copod",
        "pyod-ecod",
        "pyod-hbos",
        "pyod-knn",
        "pyod-lof",
        "pyod-ocsvm",
        "pyod-pca",
        "pyod-autoencoder",
        "pyod-autoencoder-gpu",
    },
}
if (
    not LH_ANOMALY_ALGORITHMS
    and LH_ANOMALY_ALGORITHM not in _ALGORITHMS_BY_BACKEND[LH_ANOMALY_BACKEND]
):
    valid_algorithms = ", ".join(
        sorted(_ALGORITHMS_BY_BACKEND[LH_ANOMALY_BACKEND])
    )
    raise ValueError(
        f"LH_ANOMALY_ALGORITHM must be one of {valid_algorithms} "
        f"when LH_ANOMALY_BACKEND={LH_ANOMALY_BACKEND!r}"
    )

LH_ANOMALY_DEVICE = os.getenv(
    "LH_ANOMALY_DEVICE",
    _ANOMALY_ALGORITHM_RUNTIME.get(
        LH_ANOMALY_ALGORITHM,
        ("", "cpu"),
    )[1],
).strip().lower()
if not LH_ANOMALY_ALGORITHMS:
    if LH_ANOMALY_DEVICE not in {"cpu", "gpu"}:
        raise ValueError("LH_ANOMALY_DEVICE must be 'cpu' or 'gpu'")
    expected_backend, expected_device = _ANOMALY_ALGORITHM_RUNTIME[
        LH_ANOMALY_ALGORITHM
    ]
    if LH_ANOMALY_BACKEND != expected_backend:
        raise ValueError(
            f"LH_ANOMALY_ALGORITHM={LH_ANOMALY_ALGORITHM!r} requires "
            f"LH_ANOMALY_BACKEND={expected_backend!r}"
        )
    if LH_ANOMALY_DEVICE != expected_device:
        raise ValueError(
            f"LH_ANOMALY_ALGORITHM={LH_ANOMALY_ALGORITHM!r} requires "
            f"LH_ANOMALY_DEVICE={expected_device!r}"
        )

LH_EFFECTIVE_ANOMALY_ALGORITHMS = (
    LH_ANOMALY_ALGORITHMS or (LH_ANOMALY_ALGORITHM,)
)
LH_ENSEMBLE_THRESHOLD = float(os.getenv("LH_ENSEMBLE_THRESHOLD", "0.95"))
if not 0.0 < LH_ENSEMBLE_THRESHOLD < 1.0:
    raise ValueError("LH_ENSEMBLE_THRESHOLD must be between 0 and 1")


def anomaly_algorithm_backend_device(algorithm_id: str) -> tuple[str, str]:
    """Return the backend and expected device encoded by an algorithm ID."""
    try:
        return _ANOMALY_ALGORITHM_RUNTIME[algorithm_id.strip().lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"Unsupported anomaly algorithm ID: {algorithm_id!r}") from exc

LH_TEMPORAL_ADDRESS = os.getenv(
    "LH_TEMPORAL_ADDRESS",
    "localhost:7233",
)

LH_N8N_BASE_URL = os.getenv(
    "LH_N8N_BASE_URL",
    "http://localhost:5678",
).rstrip("/")
LH_N8N_API_KEY = os.getenv("LH_N8N_API_KEY", "").strip()

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
    "false",
).strip().lower()
if _skip_existing not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
    raise ValueError(
        "LH_IDENTITY_MAPPING_SKIP_EXISTING must be a boolean value"
    )
LH_IDENTITY_MAPPING_SKIP_EXISTING = _skip_existing in {
    "true", "1", "yes", "on"
}

LH_IDENTITY_MAPPING_USE_CACHE_FOR_GROUPS = tuple(
    dict.fromkeys(
        group.strip().lower()
        for group in os.getenv(
            "LH_IDENTITY_MAPPING_USE_CACHE_FOR_GROUPS", ""
        ).split(",")
        if group.strip()
    )
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
)

LH_S3_ACCESS_KEY_ID = (
    os.getenv("LH_S3_ACCESS_KEY_ID", "").strip()
    or os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    or "rustfsadmin"
)

LH_S3_SECRET_ACCESS_KEY = (
    os.getenv("LH_S3_SECRET_ACCESS_KEY", "").strip()
    or os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    or "rustfsadmin"
)

LH_S3_REGION = (
    os.getenv("LH_S3_REGION", "").strip()
    or os.getenv("AWS_REGION", "").strip()
    or os.getenv("AWS_DEFAULT_REGION", "").strip()
    or "us-east-1"
)

LH_S3_SELECT_RECORD_FILTER_JSON = os.getenv(
    "LH_S3_SELECT_RECORD_FILTER_JSON",
    "WARN,ERROR"
)
LH_S3_SELECT_RECORD_FILTER_CSV = os.getenv(
    "LH_S3_SELECT_RECORD_FILTER_CSV",
    "WARN,ERROR"
)
LH_S3_SELECT_RECORD_FILTER = os.getenv(
    "LH_S3_SELECT_RECORD_FILTER",
    "",
).strip()

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

LH_EXTERNAL_DATA_USE = _env_bool("LH_EXTERNAL_DATA_USE", "false")
LH_EXTERNAL_S3SELECT_SUPPORT = _env_bool(
    "LH_EXTERNAL_S3SELECT_SUPPORT", "false"
)


def _parse_csv_setting(name: str) -> list[str]:
    return [value.strip() for value in os.getenv(name, "").split(",") if value.strip()]


LH_S3_INPUT_FILENAMES = tuple(
    _parse_csv_setting("LH_S3_INPUT_FILENAMES")
)
LH_S3_SELECT_SKIP_FILENAMES = tuple(
    _parse_csv_setting("LH_S3_SELECT_SKIP_FILENAMES")
)


def matches_s3_filename(filename: str, patterns: tuple[str, ...]) -> bool:
    """Match an S3 object's basename against case-insensitive wildcards."""
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return any(
        fnmatch.fnmatchcase(basename, pattern.casefold())
        for pattern in patterns
    )


def _parse_s3_folder_setting(name: str) -> list[str]:
    folders = _parse_csv_setting(name)
    normalized = []
    for folder in folders:
        parsed = urlparse(folder)
        if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError(f"{name} entries must be s3://bucket[/prefix/] URLs: {folder}")
        normalized.append(folder.rstrip("/") + "/")
    return normalized


LH_EXTERNAL_DATA_TRAIN = _parse_s3_folder_setting("LH_EXTERNAL_DATA_TRAIN")
LH_EXTERNAL_S3BUCKET_REGIONS_TRAIN = _parse_csv_setting(
    "LH_EXTERNAL_S3BUCKET_REGIONS_TRAIN"
)
LH_EXTERNAL_DATA_RAW = _parse_s3_folder_setting("LH_EXTERNAL_DATA_RAW")
LH_EXTERNAL_S3BUCKET_REGIONS_RAW = _parse_csv_setting(
    "LH_EXTERNAL_S3BUCKET_REGIONS_RAW"
)
LH_EXTERNAL_S3_ENDPOINT = os.getenv("LH_EXTERNAL_S3_ENDPOINT", "").strip()
LH_EXTERNAL_S3_ACCESS_KEY_ID = os.getenv(
    "LH_EXTERNAL_S3_ACCESS_KEY_ID", ""
).strip()
LH_EXTERNAL_S3_SECRET_ACCESS_KEY = os.getenv(
    "LH_EXTERNAL_S3_SECRET_ACCESS_KEY", ""
).strip()

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


def _validate_external_sources(
    phase: str,
    enabled: bool,
    folders: list[str],
    regions: list[str],
) -> None:
    if not LH_EXTERNAL_DATA_USE or not enabled:
        return
    if not folders:
        raise ValueError(
            f"LH_EXTERNAL_DATA_USE is enabled and {phase} is enabled, "
            f"so LH_EXTERNAL_DATA_{phase} must contain at least one S3 URL"
        )
    if len(folders) != len(regions):
        raise ValueError(
            f"LH_EXTERNAL_DATA_{phase} and "
            f"LH_EXTERNAL_S3BUCKET_REGIONS_{phase} must contain the same "
            f"number of comma-separated entries ({len(folders)} URLs, "
            f"{len(regions)} regions)"
        )


def _validate_unique_external_bucket_regions() -> None:
    if not LH_EXTERNAL_DATA_USE:
        return
    bucket_regions: dict[str, str] = {}
    phase_sources = (
        (LH_EXTERNAL_DATA_TRAIN, LH_EXTERNAL_S3BUCKET_REGIONS_TRAIN),
        (LH_EXTERNAL_DATA_RAW, LH_EXTERNAL_S3BUCKET_REGIONS_RAW),
    )
    for folders, regions in phase_sources:
        for folder, region in zip(folders, regions):
            bucket = urlparse(folder).netloc
            if bucket == LH_S3_BUCKET:
                raise ValueError(
                    "External input bucket names must differ from "
                    "LH_S3_BUCKET because Spark uses per-bucket endpoint "
                    "settings for sources and RustFS output."
                )
            previous = bucket_regions.setdefault(bucket, region)
            if previous != region:
                raise ValueError(
                    f"External bucket {bucket!r} has conflicting regions: "
                    f"{previous!r} and {region!r}"
                )
_validate_external_sources(
    "TRAIN", LH_TRAIN_PHASE,
    LH_EXTERNAL_DATA_TRAIN, LH_EXTERNAL_S3BUCKET_REGIONS_TRAIN,
)
_validate_external_sources(
    "RAW", LH_DETECT_PHASE,
    LH_EXTERNAL_DATA_RAW, LH_EXTERNAL_S3BUCKET_REGIONS_RAW,
)
_validate_unique_external_bucket_regions()

if bool(LH_EXTERNAL_S3_ACCESS_KEY_ID) != bool(LH_EXTERNAL_S3_SECRET_ACCESS_KEY):
    raise ValueError(
        "Set both LH_EXTERNAL_S3_ACCESS_KEY_ID and "
        "LH_EXTERNAL_S3_SECRET_ACCESS_KEY, or leave both blank."
    )

def _parse_s3_select_record_filter(value: str) -> list[str]:
    if value.strip().upper() == "ALL":
        return []
    return [
        item.strip().upper()
        for item in value.split(",")
        if item.strip()
    ]


LH_S3_SELECT_RECORD_FILTER_JSON_LIST = _parse_s3_select_record_filter(
    LH_S3_SELECT_RECORD_FILTER_JSON
)
LH_S3_SELECT_RECORD_FILTER_CSV_LIST = _parse_s3_select_record_filter(
    LH_S3_SELECT_RECORD_FILTER_CSV
)


def _parse_s3_select_filter_rules(value: str) -> dict[str, dict[str, list[str]]]:
    if not value.strip():
        return {}

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "LH_S3_SELECT_RECORD_FILTER must contain a valid JSON object"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("LH_S3_SELECT_RECORD_FILTER must be a JSON object")

    rules: dict[str, dict[str, list[str]]] = {}
    normalized_patterns: set[str] = set()
    for pattern, columns in parsed.items():
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError(
                "LH_S3_SELECT_RECORD_FILTER rule names must be non-empty strings"
            )
        pattern = pattern.strip()
        if pattern.casefold() in normalized_patterns:
            raise ValueError(
                "LH_S3_SELECT_RECORD_FILTER contains duplicate patterns "
                f"that differ only by case: {pattern!r}"
            )
        normalized_patterns.add(pattern.casefold())
        if not isinstance(columns, dict):
            raise ValueError(
                f"S3 Select filter rule {pattern!r} must map columns to value lists"
            )

        normalized_columns: dict[str, list[str]] = {}
        for column, values in columns.items():
            if not isinstance(column, str) or not column.strip():
                raise ValueError(
                    f"S3 Select filter rule {pattern!r} has an empty column name"
                )
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(item, str) for item in values)
            ):
                raise ValueError(
                    f"S3 Select filter values for {column!r} in rule "
                    f"{pattern!r} must be a non-empty list of strings"
                )
            normalized_columns[column] = values
        rules[pattern] = normalized_columns
    return rules


LH_S3_SELECT_FILTER_RULES = _parse_s3_select_filter_rules(
    LH_S3_SELECT_RECORD_FILTER
)


def s3_select_filter_rule_for_path(
    path: str,
) -> tuple[str | None, dict[str, list[str]]]:
    """Resolve an exact filename or wildcard rule, then fall back to default."""
    filename = urlparse(path).path.rsplit("/", 1)[-1].casefold()
    matching_patterns = [
        pattern
        for pattern in LH_S3_SELECT_FILTER_RULES
        if pattern.casefold() != "default"
        and fnmatch.fnmatchcase(filename, pattern.casefold())
    ]
    if matching_patterns:
        selected_pattern = max(
            enumerate(matching_patterns),
            key=lambda item: (
                not any(char in item[1] for char in "*?[]"),
                sum(char not in "*?[]" for char in item[1]),
                -item[0],
            ),
        )[1]
        return selected_pattern, LH_S3_SELECT_FILTER_RULES[selected_pattern]

    if "default" in (key.casefold() for key in LH_S3_SELECT_FILTER_RULES):
        default_key = next(
            key
            for key in LH_S3_SELECT_FILTER_RULES
            if key.casefold() == "default"
        )
        return default_key, LH_S3_SELECT_FILTER_RULES[default_key]
    return None, {}

# Compatibility alias for older versioned modules that still expect a list
# of severity levels. Active Stage A and Identity Mapping use filter rules.
LH_S3_SELECT_RECORD_FILTER_LIST = LH_S3_SELECT_RECORD_FILTER_JSON_LIST


def s3_select_record_filter_for_path(path: str) -> tuple[str, str, list[str]]:
    """Return the setting name, configured value, and parsed filter for a path."""
    filename = urlparse(path).path.rsplit("/", 1)[-1].lower()
    if filename.endswith(".csv"):
        return (
            "LH_S3_SELECT_RECORD_FILTER_CSV",
            LH_S3_SELECT_RECORD_FILTER_CSV,
            LH_S3_SELECT_RECORD_FILTER_CSV_LIST,
        )
    return (
        "LH_S3_SELECT_RECORD_FILTER_JSON",
        LH_S3_SELECT_RECORD_FILTER_JSON,
        LH_S3_SELECT_RECORD_FILTER_JSON_LIST,
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
    print("LH S3 SELECT RECORD FILTER RULES : ", LH_S3_SELECT_FILTER_RULES)

if __name__ == '__main__':
    main()


