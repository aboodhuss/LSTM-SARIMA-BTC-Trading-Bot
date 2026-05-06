#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./network-testing/scripts/run_profile_session.sh [--no-shaping] <iface> <profile> <duration_minutes> [base_url] [output_dir]

Example:
  ./network-testing/scripts/run_profile_session.sh eth0 latency_300_jitter_50 90
  ./network-testing/scripts/run_profile_session.sh --no-shaping lo0 baseline_smoke 0.05 http://127.0.0.1:8000
EOF
}

NO_SHAPING=0
if [[ "${1:-}" == "--no-shaping" ]]; then
  NO_SHAPING=1
  shift
fi

if [[ $# -lt 3 ]]; then
  usage
  exit 1
fi

IFACE="$1"
PROFILE="$2"
DURATION_MINUTES="$3"
BASE_URL="${4:-http://127.0.0.1:8000}"
OUTPUT_DIR="${5:-network-testing/results}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
APPLY_SCRIPT="${SCRIPT_DIR}/apply_tc_profile.sh"
CAPTURE_SCRIPT="${SCRIPT_DIR}/capture_metrics.py"

cleanup() {
  if [[ "${NO_SHAPING}" -eq 1 ]]; then
    return
  fi
  echo "[info] Clearing qdisc on ${IFACE}..."
  sudo "${APPLY_SCRIPT}" "${IFACE}" clear >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if [[ "${NO_SHAPING}" -eq 1 ]]; then
  echo "[info] --no-shaping enabled. Skipping tc profile apply."
else
  echo "[info] Applying profile '${PROFILE}' on '${IFACE}'..."
  sudo "${APPLY_SCRIPT}" "${IFACE}" "${PROFILE}"
fi

echo "[info] Capturing metrics for ${DURATION_MINUTES} minute(s) from ${BASE_URL}..."
python3 "${CAPTURE_SCRIPT}" \
  --profile-id "${PROFILE}" \
  --base-url "${BASE_URL}" \
  --duration-minutes "${DURATION_MINUTES}" \
  --output-dir "${OUTPUT_DIR}"

echo "[ok] Session complete for profile '${PROFILE}'. Outputs are in ${PROJECT_DIR}/${OUTPUT_DIR}."
