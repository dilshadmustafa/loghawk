#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "venv312/bin/python" ]]; then
    echo "Run firsttime_setup.sh before starting the LogHawk API."
    exit 1
fi

exec venv312/bin/python -m uvicorn loghawk.api.main:app --app-dir src --host 127.0.0.1 --port 8000
