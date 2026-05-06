#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
PID_FILE="$ROOT_DIR/.dev_pids"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: $1"
    exit 1
  fi
}

setup_node_fallback() {
  if command -v npm >/dev/null 2>&1; then
    return
  fi

  local bundled_node_bin="$ROOT_DIR/../.tools/node-v24.14.1-darwin-arm64/bin"
  if [ -x "$bundled_node_bin/npm" ]; then
    export PATH="$bundled_node_bin:$PATH"
    echo "[SETUP] Using bundled Node.js toolchain from .tools/"
  fi
}

cleanup_old() {
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
}

setup_backend() {
  cd "$BACKEND_DIR"
  if [ ! -d "venv" ]; then
    echo "[SETUP] Creating Python virtual environment..."
    python3 -m venv venv
  fi

  # shellcheck disable=SC1091
  source venv/bin/activate
  python -m pip install --upgrade pip >/dev/null

  if [ -f "requirements-lock.txt" ]; then
    echo "[SETUP] Installing locked Python dependencies..."
    pip install -r requirements-lock.txt >/dev/null
  else
    echo "[SETUP] Installing Python dependencies..."
    pip install -r requirements.txt >/dev/null
  fi
}

setup_frontend() {
  cd "$FRONTEND_DIR"
  if [ ! -d "node_modules" ]; then
    echo "[SETUP] Installing frontend dependencies..."
    npm ci >/dev/null 2>&1 || npm install >/dev/null
  fi
}

start_services() {
  cd "$BACKEND_DIR"
  # shellcheck disable=SC1091
  source venv/bin/activate
  nohup python -m uvicorn main:app --host 127.0.0.1 --port 8000 > backend_run.log 2>&1 &
  BACKEND_PID=$!

  cd "$FRONTEND_DIR"
  nohup npm run dev -- --host 127.0.0.1 --port 5173 > frontend_run.log 2>&1 &
  FRONTEND_PID=$!

  {
    echo "BACKEND_PID=$BACKEND_PID"
    echo "FRONTEND_PID=$FRONTEND_PID"
  } > "$PID_FILE"

  sleep 2

  echo "[OK] Backend PID: $BACKEND_PID"
  echo "[OK] Frontend PID: $FRONTEND_PID"
  echo "[OPEN] Dashboard: http://localhost:5173"
  echo "[OPEN] Model Lab:  http://localhost:5173/models"
  echo "[STOP] ./stop_everything.sh"
}

main() {
  setup_node_fallback
  require_cmd python3
  require_cmd npm

  cleanup_old
  setup_backend
  setup_frontend
  start_services
}

main "$@"
