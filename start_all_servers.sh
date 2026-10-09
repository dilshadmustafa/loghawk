#!/usr/bin/env bash
set -euo pipefail
unset NPM_CONFIG_PREFIX npm_config_prefix

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
LOG_DIR="${HOME}/.loghawk/logs"
CONDA_ENV_NAME="venv312"
MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-$HOME/miniforge3}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -f "$MINIFORGE_PREFIX/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="$MINIFORGE_PREFIX"
else
    echo "Run firsttime_setup.sh before starting LogHawk."
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
PYTHON="$(command -v python)"
if [[ -z "$PYTHON" ]]; then
    echo "Python was not found in the venv312 Conda environment."
    exit 1
fi
command -v docker >/dev/null || { echo "Docker is required and must be running."; exit 1; }
command -v aws >/dev/null || { echo "AWS CLI is required for the RustFS readiness check."; exit 1; }

echo "============================================================"
echo "Checking RustFS"
echo "============================================================"
RUSTFS_RUNNING="$(docker inspect -f '{{.State.Running}}' loghawk-rustfs 2>/dev/null || true)"
if [[ "$RUSTFS_RUNNING" != true ]]; then
    if docker container inspect loghawk-rustfs >/dev/null 2>&1; then
        docker start loghawk-rustfs
    else
        mkdir -p "$HOME/rustfs/data" "$HOME/rustfs/logs"
        docker run -d --name loghawk-rustfs \
            -p 9000:9000 -p 9001:9001 \
            -v "$HOME/rustfs/data:/data" \
            -v "$HOME/rustfs/logs:/logs" \
            rustfs/rustfs:latest
    fi
fi

export AWS_ACCESS_KEY_ID=rustfsadmin
export AWS_SECRET_ACCESS_KEY=rustfsadmin
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
RUSTFS_READY=false
for _ in {1..30}; do
    if aws --endpoint-url http://localhost:9000 s3api list-buckets >/dev/null 2>&1; then
        RUSTFS_READY=true
        break
    fi
    sleep 2
done
if [[ "$RUSTFS_READY" != true ]]; then
    echo "RustFS did not become ready within 60 seconds." >&2
    docker logs --tail 50 loghawk-rustfs >&2 || true
    exit 1
fi
echo "RustFS is ready at http://localhost:9000"

echo "============================================================"
echo "Checking Temporal"
echo "============================================================"
TEMPORAL_RUNNING="$(docker inspect -f '{{.State.Running}}' temporal 2>/dev/null || true)"
if [[ "$TEMPORAL_RUNNING" != true ]]; then
    if docker container inspect temporal >/dev/null 2>&1; then
        docker start temporal
    else
        docker compose -f src/loghawk/admin/temporal-docker-compose.yml up -d
    fi
fi

TEMPORAL_READY=false
for _ in {1..30}; do
    if (echo >/dev/tcp/127.0.0.1/7233) >/dev/null 2>&1; then
        TEMPORAL_READY=true
        break
    fi
    sleep 2
done
if [[ "$TEMPORAL_READY" != true ]]; then
    echo "Temporal did not become ready on port 7233 within 60 seconds." >&2
    docker logs --tail 50 temporal >&2 || true
    exit 1
fi
echo "Temporal is ready at http://localhost:8233"

node_compatible() {
    command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 && \
        node -e 'const [major,minor]=process.versions.node.split(".").map(Number);process.exit((major===20&&minor>=19)||(major>=22&&(major>22||minor>=12))?0:1)' >/dev/null 2>&1
}

load_nvm() {
    export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
    if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
        command -v curl >/dev/null || { echo "curl is required to install Node.js with nvm."; exit 1; }
        echo "Installing nvm for this user..."
        curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.8/install.sh | PROFILE=/dev/null bash
    fi
    # shellcheck disable=SC1091
    source "$NVM_DIR/nvm.sh"
}

if ! node_compatible; then
    echo "Installing a compatible Node.js LTS release with nvm..."
    set +u
    load_nvm
    nvm install --lts
    nvm use --lts
    set -u
fi
if ! node_compatible; then
    echo "Node.js/npm did not meet the frontend requirement after installation." >&2
    exit 1
fi

echo "============================================================"
echo "Installing LogHawk Web UI packages"
echo "============================================================"
cd "$ROOT_DIR/frontend/loghawk-ui"
npm install
cd "$ROOT_DIR"

if [[ -z "${JAVA_HOME:-}" ]]; then
    if [[ "$(uname -s)" == Darwin ]]; then
        JAVA_HOME="$(/usr/libexec/java_home -v 17)"
    else
        JAVA_BIN="$(readlink -f "$(command -v java)")"
        JAVA_HOME="${JAVA_BIN%/bin/java}"
    fi
fi
export JAVA_HOME
export HADOOP_HOME="${HADOOP_HOME:-$JAVA_HOME}"

mkdir -p "$LOG_DIR"
WORKER_LOG="$LOG_DIR/temporal-worker.log"
API_LOG="$LOG_DIR/webui-api.log"
"$PYTHON" -m loghawk.workflows.temporal.worker >"$WORKER_LOG" 2>&1 &
WORKER_PID=$!
"$PYTHON" -m uvicorn loghawk.api.main:app --app-dir src --host 127.0.0.1 --port 8000 >"$API_LOG" 2>&1 &
API_PID=$!

cleanup() {
    kill "$WORKER_PID" "$API_PID" 2>/dev/null || true
    wait "$WORKER_PID" "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Worker log: $WORKER_LOG"
echo "API log: $API_LOG"
echo "Web UI: http://localhost:5173"
echo "Web UI API: http://localhost:8000"
echo "RustFS console: http://localhost:9001"
echo "Temporal UI: http://localhost:8233"
echo "Starting Vite in this terminal. Press Ctrl+C to stop the local worker and API too."
cd "$ROOT_DIR/frontend/loghawk-ui"
npm run dev
