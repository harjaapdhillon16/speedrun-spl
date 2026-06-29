#!/usr/bin/env bash
set -euo pipefail

# GPU vision-language re-fill of stock_analysis_calls (RunPod / Ubuntu, CUDA).
# Reads card frames from Supabase Storage, runs Qwen2.5-VL on the GPU to extract
# the call fields as JSON, and upserts clean rows. Replaces the slow/inaccurate
# EasyOCR CPU fill.
#
# Usage in a RunPod GPU pod terminal (PyTorch/CUDA base image):
#   export url='https://YOUR_PROJECT.supabase.co'
#   export secret_key='YOUR_SUPABASE_SERVICE_ROLE_KEY'
#   bash run_vlm_fill.sh                 # batch=3, 7B model
#   bash run_vlm_fill.sh --batch 4
#   VLM_MODEL='Qwen/Qwen2.5-VL-3B-Instruct' bash run_vlm_fill.sh   # smaller GPU
#   tail -f ./vlm_fill.log
# Stop:  kill "$(cat ./vlm_fill.pid)"

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

# RunPod CUDA images already ship torch; we only add the model stack.
export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq || true
  apt-get install -y --no-install-recommends python3-pip python3-venv ca-certificates >/dev/null 2>&1 || true
fi

# Reuse the pod's system Python so we inherit the pre-installed CUDA torch.
# (A fresh venv would re-pull a multi-GB torch wheel for no reason.)
PYTHON="${RUNPOD_PYTHON:-python3}"
"${PYTHON}" -m pip install --upgrade pip >/dev/null 2>&1 || true

LOG_FILE="${VLM_LOG_FILE:-./vlm_fill.log}"
PID_FILE="${VLM_PID_FILE:-./vlm_fill.pid}"

# One-off batch job: replace any previous run so re-launching uses the new code.
if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Stopping previous VLM fill: pid=${old_pid}"
    kill "${old_pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${old_pid}" 2>/dev/null || true
  fi
fi
pkill -f ocr_vlm_fill.py 2>/dev/null || true
sleep 1
: > "${LOG_FILE}"

# First run pip-installs transformers/accelerate and downloads the model weights
# (~16GB for 7B) once; then inference is fast.
nohup "${PYTHON}" -u ocr_vlm_fill.py "$@" \
  </dev/null >>"${LOG_FILE}" 2>&1 &
pid="$!"
echo "${pid}" > "${PID_FILE}"
echo "Started VLM fill: pid=${pid}"
echo "Monitor: tail -f ${LOG_FILE}"
echo "Stop:    kill ${pid}"
