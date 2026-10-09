#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/frontend/loghawk-ui"

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo "Node.js and npm must be installed and available on PATH."
    exit 1
fi

echo "Installing LogHawk Web UI packages..."
npm install

echo "Starting LogHawk Web UI with Vite..."
npm run dev
