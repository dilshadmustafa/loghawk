#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-$HOME/miniforge3}"
CONDA_ENV_NAME="venv_hawkconsole"

echo
echo "============================================================"
echo "Installing Miniforge / Conda if needed"
echo "============================================================"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -f "$MINIFORGE_PREFIX/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="$MINIFORGE_PREFIX"
else
    command -v curl >/dev/null || {
        echo "curl is required to install Miniforge."
        exit 1
    }

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

    INSTALLER_DIR="$(mktemp -d "${TMPDIR:-/tmp}/loghawk-miniforge.XXXXXX")"
    INSTALLER="$INSTALLER_DIR/Miniforge3.sh"
    trap 'rm -f "$INSTALLER"; rmdir "$INSTALLER_DIR" 2>/dev/null || true' EXIT
    curl -fsSL \
        "https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER_NAME}" \
        -o "$INSTALLER"
    bash "$INSTALLER" -b -p "$MINIFORGE_PREFIX"
    rm -f "$INSTALLER"
    rmdir "$INSTALLER_DIR"
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

echo
echo "============================================================"
echo "Installing Hawk Console packages"
echo "============================================================"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements-hawkconsole.txt"
python -m pip install --no-deps -e "$ROOT_DIR"

echo
echo "============================================================"
echo "Setting up LanceDB"
echo "============================================================"
python -m loghawk.admin.setup_console_lancedb

echo
echo "Hawk Console setup complete."
echo "Activate with:"
echo "  source \"$CONDA_BASE/etc/profile.d/conda.sh\""
echo "  conda activate $CONDA_ENV_NAME"
echo "Then run: hawk"
conda deactivate
