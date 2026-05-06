#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sudo ./network-testing/scripts/apply_tc_profile.sh <iface> <profile>

Profiles:
  baseline | clear
  show
  latency_150
  latency_300_jitter_50
  latency_500_jitter_100
  loss_1
  loss_3
  loss_5
  bandwidth_5mbit
  bandwidth_2mbit
  bandwidth_1mbit
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "[error] Run as root (use sudo)."
  exit 1
fi

if ! command -v tc >/dev/null 2>&1; then
  echo "[error] tc command not found."
  exit 1
fi

if ! command -v ip >/dev/null 2>&1; then
  echo "[error] ip command not found."
  exit 1
fi

IFACE="$1"
PROFILE="$2"

if ! ip link show "${IFACE}" >/dev/null 2>&1; then
  echo "[error] Interface '${IFACE}' does not exist."
  exit 1
fi

clear_root_qdisc() {
  tc qdisc del dev "${IFACE}" root 2>/dev/null || true
}

show_qdisc() {
  tc -s qdisc show dev "${IFACE}"
}

case "${PROFILE}" in
  baseline|clear)
    clear_root_qdisc
    echo "[ok] Cleared qdisc on ${IFACE}."
    ;;
  show)
    show_qdisc
    exit 0
    ;;
  latency_150)
    clear_root_qdisc
    tc qdisc replace dev "${IFACE}" root netem delay 150ms
    ;;
  latency_300_jitter_50)
    clear_root_qdisc
    tc qdisc replace dev "${IFACE}" root netem delay 300ms 50ms distribution normal
    ;;
  latency_500_jitter_100)
    clear_root_qdisc
    tc qdisc replace dev "${IFACE}" root netem delay 500ms 100ms distribution normal
    ;;
  loss_1)
    clear_root_qdisc
    tc qdisc replace dev "${IFACE}" root netem loss 1%
    ;;
  loss_3)
    clear_root_qdisc
    tc qdisc replace dev "${IFACE}" root netem loss 3%
    ;;
  loss_5)
    clear_root_qdisc
    tc qdisc replace dev "${IFACE}" root netem loss 5%
    ;;
  bandwidth_5mbit)
    clear_root_qdisc
    tc qdisc replace dev "${IFACE}" root tbf rate 5mbit burst 32kbit latency 400ms
    ;;
  bandwidth_2mbit)
    clear_root_qdisc
    tc qdisc replace dev "${IFACE}" root tbf rate 2mbit burst 32kbit latency 400ms
    ;;
  bandwidth_1mbit)
    clear_root_qdisc
    tc qdisc replace dev "${IFACE}" root tbf rate 1mbit burst 32kbit latency 500ms
    ;;
  *)
    echo "[error] Unknown profile '${PROFILE}'."
    usage
    exit 1
    ;;
esac

if [[ "${PROFILE}" != "baseline" && "${PROFILE}" != "clear" ]]; then
  echo "[ok] Applied profile '${PROFILE}' on ${IFACE}."
fi

show_qdisc
