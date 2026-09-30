#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "venv312/bin/python" ]]; then
    echo "Run firsttime_setup_terminal_1.sh first."
    exit 1
fi

exec venv312/bin/python src/loghawk/chat_assistant/chat_assistant4.py
