#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOGHAWK_CONDA_ENV_NAME="${RAPIDS_ENV_NAME:-loghawk-rapids}"
export LOGHAWK_SETUP_SCRIPT="firsttime_setup_rapids_cuml.sh"

exec "$ROOT_DIR/start_all_servers.sh"
