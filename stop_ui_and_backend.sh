#!/usr/bin/env bash
set -euo pipefail
pkill -f "uvicorn main:app" || true
pkill -f "vite --host 0.0.0.0 --port 5173" || true
echo "Stopped backend/frontend dev servers."
