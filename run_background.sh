#!/usr/bin/env bash
set -euo pipefail

# Start the SPL Midcap runner in the background.
# Logs and PID are written in this current directory only.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  source ".env"
fi

LOG_FILE="${RUNPOD_LOG_FILE:-./spl_midcap_run.log}"
PID_FILE="${RUNPOD_PID_FILE:-./spl_midcap_run.pid}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Runner already active: pid=${old_pid}"
    echo "Monitor: tail -f ${LOG_FILE}"
    exit 0
  fi
fi

touch "${LOG_FILE}"

nohup bash ./runpod_bootstrap.sh \
  --existing-video-jobs \
  --skip-discovery \
  --start "${RUNPOD_START_DATE:-2023-04-01}" \
  --concurrency "${RUNPOD_CONCURRENCY:-16}" \
  "$@" >> "${LOG_FILE}" 2>&1 &

pid="$!"
echo "${pid}" > "${PID_FILE}"

echo "Started SPL Midcap runner: pid=${pid}"
echo "Log: ${LOG_FILE}"
echo "Monitor: tail -f ${LOG_FILE}"
