#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="qwen36-35b-a3b-nvfp4-vllm"
PID_FILE=".vllm.pid"
LOG_FILE=".vllm.log"

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping vLLM process ${pid}"
    kill "${pid}" 2>/dev/null || true
    for _ in {1..30}; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Process did not exit cleanly; sending SIGKILL"
      kill -9 "${pid}" 2>/dev/null || true
    fi
  else
    echo "No running process for PID ${pid}"
  fi
else
  echo "No PID file found"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Cleaning up stale container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

rm -f "${PID_FILE}"
echo "Stopped. Log preserved at ${LOG_FILE}"
