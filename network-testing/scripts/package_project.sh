#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PARENT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_PATH="${1:-${PARENT_DIR}/trading-bot-complete-${STAMP}.zip}"

cd "${PARENT_DIR}"

if ! command -v zip >/dev/null 2>&1; then
  echo "[error] zip command not found."
  exit 1
fi

zip -r "${OUT_PATH}" trading-bot

echo "[ok] Handoff package created:"
echo "${OUT_PATH}"
