#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT_DIR/.dev_pids"

if [ -f "$PID_FILE" ]; then
  while IFS='=' read -r key value; do
    if [ "$key" = "BACKEND_PID" ] || [ "$key" = "FRONTEND_PID" ]; then
      kill "$value" >/dev/null 2>&1 || true
    fi
  done < "$PID_FILE"
  rm -f "$PID_FILE"
fi

pkill -f "uvicorn main:app --host 127.0.0.1 --port 8000" >/dev/null 2>&1 || true
pkill -f "vite --host 127.0.0.1 --port 5173" >/dev/null 2>&1 || true

echo "Stopped backend/frontend services."
