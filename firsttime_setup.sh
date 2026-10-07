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
MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-$HOME/miniforge3}"
CONDA_ENV_NAME="venv312"

as_bool() {
    case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
        true|1|yes|on) echo true ;;
        false|0|no|off) echo false ;;
        *) echo "Invalid boolean value: $1 (use true or false)." >&2; exit 2 ;;
    esac
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

SKIP_RUSTFS_SETUP="$(as_bool "$SKIP_RUSTFS_SETUP")"
SKIP_TEMPORAL_SETUP="$(as_bool "$SKIP_TEMPORAL_SETUP")"
SKIP_DUCKDB_SETUP="$(as_bool "$SKIP_DUCKDB_SETUP")"
SKIP_LANCEDB_SETUP="$(as_bool "$SKIP_LANCEDB_SETUP")"

command -v curl >/dev/null || {
    echo "curl is required to install Miniforge."
    exit 1
}
command -v java >/dev/null || {
    echo "Java 17 is required."
    exit 1
}
if [[ "$SKIP_RUSTFS_SETUP" == false ]]; then
    command -v docker >/dev/null || {
        echo "Docker is required."
        exit 1
    }
    command -v aws >/dev/null || {
        echo "AWS CLI is required."
        exit 1
    }
fi
if [[ "$SKIP_TEMPORAL_SETUP" == false ]]; then
    command -v docker >/dev/null || {
        echo "Docker is required."
        exit 1
    }
fi

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -f "$MINIFORGE_PREFIX/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="$MINIFORGE_PREFIX"
else
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Linux)
            case "$ARCH" in
                x86_64|aarch64) ;;
                *)
                    echo "Unsupported Linux architecture for Miniforge: $ARCH" >&2
                    exit 1
                    ;;
            esac
            INSTALLER_NAME="Miniforge3-Linux-${ARCH}.sh"
            ;;
        Darwin)
            case "$ARCH" in
                arm64) INSTALLER_NAME="Miniforge3-MacOSX-arm64.sh" ;;
                x86_64) INSTALLER_NAME="Miniforge3-MacOSX-x86_64.sh" ;;
                *)
                    echo "Unsupported macOS architecture for Miniforge: $ARCH" >&2
                    exit 1
                    ;;
            esac
            ;;
        *)
            echo "Unsupported operating system for Miniforge: $OS" >&2
            exit 1
            ;;
    esac

    echo
    echo "============================================================"
    echo "Installing Miniforge / Conda if needed"
    echo "============================================================"
    INSTALLER="$(mktemp "${TMPDIR:-/tmp}/loghawk-miniforge.XXXXXX")"
    trap 'rm -f "$INSTALLER"' EXIT
    curl -fsSL \
        "https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER_NAME}" \
        -o "$INSTALLER"
    bash "$INSTALLER" -b -p "$MINIFORGE_PREFIX"
    rm -f "$INSTALLER"
    trap - EXIT
    CONDA_BASE="$MINIFORGE_PREFIX"
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo
echo "============================================================"
echo "Creating Python 3.12 Conda environment: $CONDA_ENV_NAME"
echo "============================================================"
if conda env list | awk 'NR > 2 {print $1}' | grep -Fxq "$CONDA_ENV_NAME"; then
    conda install --yes --name "$CONDA_ENV_NAME" \
        --channel conda-forge python=3.12
else
    conda create --yes --name "$CONDA_ENV_NAME" \
        --channel conda-forge python=3.12
fi
conda activate "$CONDA_ENV_NAME"
python -c 'import sys; print("Python:", sys.version); assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'

if [[ -z "${JAVA_HOME:-}" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        JAVA_HOME="$(/usr/libexec/java_home -v 17)"
    else
        JAVA_BIN="$(readlink -f "$(command -v java)")"
        JAVA_HOME="${JAVA_BIN%/bin/java}"
    fi
fi
export JAVA_HOME
export HADOOP_HOME="${HADOOP_HOME:-$JAVA_HOME}"

echo
echo "============================================================"
echo "Using Python 3.12 Conda environment"
echo "============================================================"
echo
echo "============================================================"
echo "Installing PySpark"
echo "============================================================"
python -m pip install --upgrade pip setuptools wheel
python -m pip install pyspark==3.5.9
python -c 'from pyspark.sql import SparkSession; s=SparkSession.builder.master("local[*]").getOrCreate(); print("Spark:",s.version); print("Python:",__import__("sys").version); print("Java:",s.sparkContext._jvm.java.lang.System.getProperty("java.version")); print("Hadoop:",s.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion()); s.stop()'
echo
echo "============================================================"
echo "PyTorch - Explicit install"
echo "============================================================"
OS="$(uname -s)"
if [[ "$OS" == "Linux" ]]; then
    echo "Installing PyTorch with CUDA 13.0 support"
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
elif [[ "$OS" == "Darwin" ]]; then
    echo "Installing standard PyTorch for macOS"
    python -m pip install torch torchvision
else
    echo "Unsupported operating system for PyTorch: $OS" >&2
    exit 1
fi
echo
echo "============================================================"
echo "Installing packages mentioned in requirements.txt"
echo "============================================================"
python -m pip install -r requirements.txt
python -m pip install -e .
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
    python -m loghawk.admin.setup_lancedb
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

echo "Setup complete."
echo "RustFS console: http://localhost:9001"
echo "Temporal UI: http://localhost:8233"
echo "Start the worker and workflow using the firsttime_start_workflow scripts."
conda deactivate
echo "Activate the environment in a new terminal with:"
echo "  source \"$CONDA_BASE/etc/profile.d/conda.sh\""
echo "  conda activate $CONDA_ENV_NAME"
