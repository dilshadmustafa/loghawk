#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$ROOT_DIR/venv_hawkconsole"
VENV_PYTHON="$VENV_DIR/bin/python"

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.12)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "Python 3.12 or newer is required."
    exit 1
fi

"$PYTHON_BIN" -c 'import sys; print("Python:", sys.version.split()[0]); assert sys.version_info >= (3, 12), "Python 3.12 or newer is required"'

echo
echo "============================================================"
echo "Setting up Hawk Console Python environment"
echo "============================================================"
if [[ ! -x "$VENV_PYTHON" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    "$VENV_PYTHON" -c 'import sys; print("Environment Python:", sys.version.split()[0]); assert sys.version_info >= (3, 12), "Existing venv_hawkconsole must use Python 3.12 or newer"'
fi

echo
echo "============================================================"
echo "Installing Hawk Console packages"
echo "============================================================"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements-hawkconsole.txt"
"$VENV_PYTHON" -m pip install --no-deps -e "$ROOT_DIR"

echo
echo "Hawk Console setup complete."
echo "Activate with:"
echo "  source venv_hawkconsole/bin/activate"
echo "Then run: hawk"
