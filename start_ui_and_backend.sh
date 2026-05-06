#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v npm >/dev/null 2>&1; then
  NODE_HOME="$ROOT_DIR/../.tools/node-v24.14.1-darwin-arm64"
  if [ -x "$NODE_HOME/bin/npm" ]; then
    export PATH="$NODE_HOME/bin:$PATH"
    echo "[SETUP] Using bundled Node.js from .tools/"
  else
    echo "[ERROR] npm not found. Install Node.js or provide .tools/node-v24.14.1-darwin-arm64."
    exit 1
  fi
fi

cd "$ROOT_DIR/backend"
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > backend_run.log 2>&1 &
BACK_PID=$!

cd "$ROOT_DIR/frontend"
nohup npm run dev -- --host 0.0.0.0 --port 5173 > frontend_run.log 2>&1 &
FRONT_PID=$!

echo "Backend PID: $BACK_PID"
echo "Frontend PID: $FRONT_PID"
echo "UI: http://localhost:5173"
