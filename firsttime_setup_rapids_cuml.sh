#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"
export TMP="$HOME/tmp"
export TEMP="$HOME/tmp"
export PIP_NO_CACHE_DIR=1
export PIP_TIMEOUT=120
export PIP_RETRIES=10
export PIP_RESUME_RETRIES=20

SKIP_RUSTFS_SETUP="${SKIP_RUSTFS_SETUP:-false}"
SKIP_TEMPORAL_SETUP="${SKIP_TEMPORAL_SETUP:-false}"
SKIP_DUCKDB_SETUP="${SKIP_DUCKDB_SETUP:-false}"
SKIP_LANCEDB_SETUP="${SKIP_LANCEDB_SETUP:-false}"
RAPIDS_ENV_NAME="${RAPIDS_ENV_NAME:-loghawk-rapids}"
RAPIDS_CUDA_VERSION="${RAPIDS_CUDA_VERSION:-13.2}"
MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-$HOME/miniforge3}"

as_bool() {
    case "${1,,}" in
        true|1|yes|on)  echo true ;;
        false|0|no|off) echo false ;;
        *)
            echo "Invalid boolean value: $1 (use true or false)." >&2
            exit 2
            ;;
    esac
}

SKIP_RUSTFS_SETUP="$(as_bool "$SKIP_RUSTFS_SETUP")"
SKIP_TEMPORAL_SETUP="$(as_bool "$SKIP_TEMPORAL_SETUP")"
SKIP_DUCKDB_SETUP="$(as_bool "$SKIP_DUCKDB_SETUP")"
SKIP_LANCEDB_SETUP="$(as_bool "$SKIP_LANCEDB_SETUP")"

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Required command not found: $1" >&2
        exit 1
    }
}

wait_for_rustfs() {
    local attempt
    echo "Waiting for RustFS S3 API (up to 60 seconds)..."
    for ((attempt = 1; attempt <= 30; attempt++)); do
        if aws --endpoint-url http://localhost:9000 \
            s3api list-buckets >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    echo "RustFS did not become ready within 60 seconds." >&2
    docker logs --tail 50 loghawk-rustfs >&2 || true
    return 1
}

echo
echo "============================================================"
echo "Checking WSL/Linux and NVIDIA GPU prerequisites"
echo "============================================================"

case "$(uname -s)" in
    Linux) ;;
    *)
        echo "Run this script inside Ubuntu on WSL2 or a supported Linux host."
        exit 1
        ;;
esac

require_command nvidia-smi
nvidia-smi

require_command java
JAVA_VERSION="$(java -version 2>&1 | head -n 1)"
if [[ "$JAVA_VERSION" != *'17.'* && "$JAVA_VERSION" != *'"17'* ]]; then
    echo "Java 17 is required by this LogHawk setup. Found: $JAVA_VERSION"
    exit 1
fi

if [[ -z "${JAVA_HOME:-}" ]]; then
    JAVA_BIN="$(readlink -f "$(command -v java)")"
    JAVA_HOME="${JAVA_BIN%/bin/java}"
fi
export JAVA_HOME
export HADOOP_HOME="${HADOOP_HOME:-$JAVA_HOME}"

if [[ "$SKIP_RUSTFS_SETUP" == false ]]; then
    require_command docker
    require_command aws
fi
if [[ "$SKIP_TEMPORAL_SETUP" == false ]]; then
    require_command docker
    docker compose version >/dev/null
fi

echo
echo "============================================================"
echo "Installing Miniforge / Conda if needed"
echo "============================================================"

require_command curl
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|aarch64) ;;
    *)
        echo "Unsupported Miniforge architecture: $ARCH"
        exit 1
        ;;
esac

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    INSTALLER="$(mktemp "${TMPDIR:-/tmp}/loghawk-miniforge.XXXXXX")"
    trap 'rm -f "$INSTALLER"' EXIT
    curl -fsSL \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh" \
        -o "$INSTALLER"
    bash "$INSTALLER" -b -p "$MINIFORGE_PREFIX"
    CONDA_BASE="$MINIFORGE_PREFIX"
fi

# Make conda activate available in this non-interactive setup shell.
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo
echo "============================================================"
echo "Creating or updating the RAPIDS cuML environment"
echo "============================================================"

if conda env list | awk '{print $1}' | grep -Fxq "$RAPIDS_ENV_NAME"; then
    conda install --yes --name "$RAPIDS_ENV_NAME" \
        --channel rapidsai --channel conda-forge \
        "python=3.12" "cuda-version=$RAPIDS_CUDA_VERSION" cuml nvforest
else
    conda create --yes --name "$RAPIDS_ENV_NAME" \
        --channel rapidsai --channel conda-forge \
        "python=3.12" "cuda-version=$RAPIDS_CUDA_VERSION" cuml nvforest
fi

conda activate "$RAPIDS_ENV_NAME"

echo
echo "============================================================"
echo "Verifying Python 3.12 and cuML GPU access"
echo "============================================================"
python --version
python -c 'import cupy, cuml, nvforest, treelite; from importlib.metadata import version; print("cuML:", cuml.__version__); print("nvForest:", version("nvforest")); print("Treelite:", treelite.__version__); print("GPU:", cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())'

echo
echo "============================================================"
python -m pip install --upgrade pip setuptools wheel
echo
echo "============================================================"
echo "PyTorch - Explicit install - Installing PyTorch with CUDA 13.2 support"
echo "============================================================"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
echo
echo "============================================================"
echo "Installing packages mentioned in requirements.txt"
echo "============================================================"
python -m pip install -r requirements.txt
python -m pip install -e .

echo
echo "============================================================"
echo "Checking PySpark and Java 17"
echo "============================================================"
python -c 'from pyspark.sql import SparkSession; s=SparkSession.builder.master("local[*]").getOrCreate(); print("Spark:",s.version); print("Python:",__import__("sys").version); print("Java:",s.sparkContext._jvm.java.lang.System.getProperty("java.version")); s.stop()'

echo
echo "============================================================"
echo "Setting up DuckDB"
echo "============================================================"
if [[ "$SKIP_DUCKDB_SETUP" == false ]]; then
    python -m loghawk.admin.setup_duckdb
else
    echo "Skipping DuckDB setup."
fi

echo
echo "============================================================"
echo "Setting up LanceDB"
echo "============================================================"
if [[ "$SKIP_LANCEDB_SETUP" == false ]]; then
    python -m loghawk.admin.setup_lancedb2
else
    echo "Skipping LanceDB setup."
fi

if [[ "$SKIP_RUSTFS_SETUP" == false ]]; then
    echo
    echo "============================================================"
    echo "Starting RustFS S3 server"
    echo "============================================================"
    mkdir -p "$HOME/rustfs/data" "$HOME/rustfs/logs"
    if docker container inspect loghawk-rustfs >/dev/null 2>&1; then
        docker start loghawk-rustfs
    else
        docker run -d --name loghawk-rustfs \
            -p 9000:9000 -p 9001:9001 \
            -v "$HOME/rustfs/data:/data" \
            -v "$HOME/rustfs/logs:/logs" \
            rustfs/rustfs:latest
    fi

    export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-rustfsadmin}"
    export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-rustfsadmin}"
    export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

    echo
    echo "============================================================"
    echo "Creating the loghawk-data S3 bucket"
    echo "============================================================"
    wait_for_rustfs
    if aws --endpoint-url http://localhost:9000 \
        s3api head-bucket --bucket loghawk-data >/dev/null 2>&1; then
        echo "Bucket loghawk-data already exists."
    else
        aws --endpoint-url http://localhost:9000 s3 mb s3://loghawk-data
    fi

    echo
    echo "============================================================"
    echo "Generating and uploading test data to S3"
    echo "============================================================"
    python src/loghawk/test/testdata.py
else
    echo "Skipping RustFS startup, bucket setup, and test-data upload."
fi

if [[ "$SKIP_TEMPORAL_SETUP" == false ]]; then
    echo
    echo "============================================================"
    echo "Starting Temporal server"
    echo "============================================================"
    docker compose -f src/loghawk/admin/temporal-docker-compose.yml up -d
else
    echo "Skipping Temporal server startup."
fi

conda deactivate

echo
echo "NVIDIA GPU setup complete."
echo "Activate the environment in each new terminal with:"
echo "  source \"$MINIFORGE_PREFIX/etc/profile.d/conda.sh\""
echo "  conda activate $RAPIDS_ENV_NAME"
echo "RustFS console: http://localhost:9001"
echo "Temporal UI: http://localhost:8233"
echo "Start the worker and workflow using the firsttime_start_workflow scripts."

