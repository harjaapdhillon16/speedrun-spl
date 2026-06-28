#!/usr/bin/env bash
set -euo pipefail

# RunPod bootstrap for the standalone SPL Midcap speed-run.
#
# Usage inside a RunPod terminal:
#   export url='https://YOUR_PROJECT.supabase.co'
#   export secret_key='YOUR_SUPABASE_SERVICE_ROLE_KEY'
#   bash runpod_bootstrap.sh --start 2023-04-01
#
# This script expects spl_midcap_speedrun.py to be in the same directory.

if [[ -z "${url:-${SUPABASE_URL:-}}" ]]; then
  echo "Missing Supabase URL. Set either url or SUPABASE_URL." >&2
  exit 1
fi

if [[ -z "${secret_key:-${SUPABASE_SERVICE_ROLE_KEY:-${SUPABASE_SECRET_KEY:-}}}" ]]; then
  echo "Missing Supabase service role key. Set secret_key or SUPABASE_SERVICE_ROLE_KEY." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/spl_midcap_speedrun.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "Cannot find ${PYTHON_SCRIPT}. Put this bootstrap next to spl_midcap_speedrun.py." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-hin \
    python3-pip \
    python3-venv \
    ca-certificates
  if apt-cache show python3.10 >/dev/null 2>&1; then
    apt-get install -y --no-install-recommends python3.10 python3.10-venv
    if apt-cache show python3.10-distutils >/dev/null 2>&1; then
      apt-get install -y --no-install-recommends python3.10-distutils
    fi
  fi
else
  echo "apt-get not found. Install ffmpeg, tesseract-ocr, tesseract-ocr-hin, and python3-pip manually." >&2
fi

PYTHON_BIN="${RUNPOD_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  else
    PYTHON_BIN="python3"
  fi
fi

VENV_DIR="${RUNPOD_VENV:-/workspace/spl_speedrun_venv}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

DEFAULT_CONCURRENCY="${RUNPOD_CONCURRENCY:-16}"
DEFAULT_WORKDIR="${RUNPOD_WORKDIR:-/workspace/spl_speedrun_work}"

python "${PYTHON_SCRIPT}" \
  --concurrency "${DEFAULT_CONCURRENCY}" \
  --workdir "${DEFAULT_WORKDIR}" \
  "$@"
