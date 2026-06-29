#!/usr/bin/env bash
set -euo pipefail

# Background launcher for the EasyOCR re-fill of stock_analysis_calls.
# Reads the already-uploaded card frames from Supabase Storage, re-OCRs them with
# the layout-aware EasyOCR parser, market-enriches, and upserts clean rows into
# the stock_analysis_calls table. No video download / ffmpeg needed.
#
# Usage inside a RunPod terminal:
#   export url='https://YOUR_PROJECT.supabase.co'
#   export secret_key='YOUR_SUPABASE_SERVICE_ROLE_KEY'
#   bash run_ocr_fill.sh                 # uses all CPUs
#   bash run_ocr_fill.sh --workers 24    # or pin worker count
#   tail -f ./ocr_fill.log
# Stop:  kill "$(cat ./ocr_fill.pid)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  source ".env"
fi

if [[ -z "${url:-${SUPABASE_URL:-}}" || -z "${secret_key:-${SUPABASE_SERVICE_ROLE_KEY:-${SUPABASE_SECRET_KEY:-}}}" ]]; then
  echo "Set url/SUPABASE_URL and secret_key/SUPABASE_SERVICE_ROLE_KEY first." >&2
  exit 1
fi

# EasyOCR/torch need a few system libs; opencv-python-headless avoids libGL.
export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq || true
  apt-get install -y --no-install-recommends python3-pip python3-venv ca-certificates libglib2.0-0 >/dev/null 2>&1 || true
fi

VENV_DIR="${RUNPOD_VENV:-/workspace/spl_speedrun_venv}"
if [[ ! -d "${VENV_DIR}" ]]; then
  "${RUNPOD_PYTHON:-python3}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip >/dev/null

WORKERS="${OCR_WORKERS:-${RUNPOD_CONCURRENCY:-$(nproc 2>/dev/null || echo 8)}}"
LOG_FILE="${OCR_LOG_FILE:-./ocr_fill.log}"
PID_FILE="${OCR_PID_FILE:-./ocr_fill.pid}"

# This is a one-off batch job: if a previous run is still alive, replace it
# (otherwise pulling new code and re-running would keep the OLD process going).
if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Stopping previous OCR fill: pid=${old_pid}"
    kill "${old_pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${old_pid}" 2>/dev/null || true
  fi
fi
# Also clear any stray fill processes from earlier attempts.
pkill -f ocr_stock_analysis_fill.py 2>/dev/null || true
sleep 1
: > "${LOG_FILE}"

# First invocation pip-installs easyocr+torch (large) and downloads the model
# weights; this can take a few minutes before rows start appearing.
nohup python -u ocr_stock_analysis_fill.py --workers "${WORKERS}" "$@" \
  </dev/null >>"${LOG_FILE}" 2>&1 &
pid="$!"
echo "${pid}" > "${PID_FILE}"
echo "Started OCR fill: pid=${pid} workers=${WORKERS}"
echo "Monitor: tail -f ${LOG_FILE}"
echo "Stop:    kill ${pid}"
