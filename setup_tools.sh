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

MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-$HOME/miniforge3}"
ENV_NAME="venv_tools"

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo 'Please run with "bash setup_tools.sh", not "source".' >&2
    return 2
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
            case "$ARCH" in x86_64|aarch64) ;; *) echo "Unsupported Linux architecture: $ARCH" >&2; exit 1 ;; esac
            INSTALLER_NAME="Miniforge3-Linux-${ARCH}.sh"
            ;;
        Darwin)
            case "$ARCH" in
                arm64) INSTALLER_NAME="Miniforge3-MacOSX-arm64.sh" ;;
                x86_64) INSTALLER_NAME="Miniforge3-MacOSX-x86_64.sh" ;;
                *) echo "Unsupported macOS architecture: $ARCH" >&2; exit 1 ;;
            esac
            ;;
        *) echo "Unsupported operating system: $OS" >&2; exit 1 ;;
    esac
    command -v curl >/dev/null || { echo "curl is required to install Miniforge." >&2; exit 1; }
    echo "Installing Miniforge under $MINIFORGE_PREFIX..."
    installer_dir="$(mktemp -d "${TMPDIR:-/tmp}/loghawk-miniforge.XXXXXX")"
    installer="$installer_dir/Miniforge3.sh"
    trap 'rm -f "$installer"; rmdir "$installer_dir" 2>/dev/null || true' EXIT
    curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER_NAME}" -o "$installer"
    bash "$installer" -b -p "$MINIFORGE_PREFIX"
    rm -f "$installer"
    rmdir "$installer_dir"
    trap - EXIT
    CONDA_BASE="$MINIFORGE_PREFIX"
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
echo
echo "============================================================"
echo "Creating or updating Conda environment: $ENV_NAME"
echo "============================================================"
if conda env list | awk 'NR > 2 {print $1}' | grep -Fxq "$ENV_NAME"; then
    conda install --yes --name "$ENV_NAME" --channel conda-forge python=3.12 pip
else
    conda create --yes --name "$ENV_NAME" --channel conda-forge python=3.12 pip
fi
conda activate "$ENV_NAME"
python -c 'import sys; print("Tools Python:", sys.version); assert sys.version_info >= (3, 12), "Python 3.12 or newer is required"'
python -m pip install --upgrade pip setuptools wheel
echo
echo "============================================================"
echo "Installing optional packages from requirements_tools.txt"
echo "============================================================"
python -m pip install -r requirements_tools.txt
python -m pip install -e . --no-deps
if [[ ! -f .env_tools && -f .env_tools.example ]]; then
    cp .env_tools.example .env_tools
fi
echo
echo "Tools environment setup complete."
echo "Activate it in a new terminal with:"
echo "  source \"$CONDA_BASE/etc/profile.d/conda.sh\""
echo "  conda activate $ENV_NAME"
