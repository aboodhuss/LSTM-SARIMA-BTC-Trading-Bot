#!/usr/bin/env python3
"""Capture backend network-testing metrics from FastAPI /health."""

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (p / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_values[low])
    fraction = rank - low
    return float(sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction)


def mean_or_none(values):
    return statistics.fmean(values) if values else None


def stdev_or_none(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None


def fetch_health(base_url: str, timeout: float):
    url = f"{base_url.rstrip('/')}/health"
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload


def create_row(profile_id: str, sample_time: datetime, payload: dict):
    latest_state = payload.get("latest_state") or {}
    telemetry = latest_state.get("telemetry") or {}
    prediction = latest_state.get("prediction") or {}
    portfolio = latest_state.get("portfolio") or {}
    network_cfg = latest_state.get("networkTest") or {}
    live_training = latest_state.get("liveTraining") or {}

    return {
        "timestamp_utc": sample_time.isoformat(timespec="seconds"),
        "profile_id": profile_id,
        "status": payload.get("status", ""),
        "prediction_action": str(prediction.get("action", "")),
        "prediction_confidence": safe_float(prediction.get("confidence")),
        "telemetry_delta_ms": safe_float(telemetry.get("deltaMs")),
        "telemetry_backend_latency_ms": safe_float(telemetry.get("latencyMs")),
        "telemetry_simulated_delay_ms": safe_float(telemetry.get("simulatedDelayMs")),
        "telemetry_updates_per_minute": safe_float(telemetry.get("updatesPerMinute")),
        "telemetry_packet_loss_pct": safe_float(telemetry.get("packetLossPct")),
        "telemetry_processed_candles": safe_int(telemetry.get("processedCandles")),
        "telemetry_dropped_candles": safe_int(telemetry.get("droppedCandles")),
        "portfolio_trade_count": safe_int(portfolio.get("tradeCount")),
        "portfolio_total_pnl": safe_float(portfolio.get("totalPnl")),
        "portfolio_signal_accuracy": safe_float(portfolio.get("signalAccuracy")),
        "portfolio_win_rate": safe_float(portfolio.get("winRate")),
        "portfolio_position": str(portfolio.get("position", "")),
        "network_enabled": bool(network_cfg.get("enabled", False)),
        "network_latency_ms": safe_float(network_cfg.get("latencyMs")),
        "network_jitter_ms": safe_float(network_cfg.get("jitterMs")),
        "network_packet_loss_pct": safe_float(network_cfg.get("packetLossPct")),
        "live_feedback_count": safe_int(live_training.get("tradeFeedbackCount")),
    }


def compute_summary(profile_id, rows, poll_failures, low_conf_threshold, gap_events, max_gap_sec):
    delays = sorted([v for v in (row["telemetry_delta_ms"] for row in rows) if v is not None])
    backend_latencies = sorted([v for v in (row["telemetry_backend_latency_ms"] for row in rows) if v is not None])
    packet_loss = [v for v in (row["telemetry_packet_loss_pct"] for row in rows) if v is not None]
    updates = [v for v in (row["telemetry_updates_per_minute"] for row in rows) if v is not None]
    confidences = [v for v in (row["prediction_confidence"] for row in rows) if v is not None]
    actions = [row["prediction_action"] for row in rows if row["prediction_action"]]

    processed_values = [v for v in (row["telemetry_processed_candles"] for row in rows) if v is not None]
    dropped_values = [v for v in (row["telemetry_dropped_candles"] for row in rows) if v is not None]
    trade_values = [v for v in (row["portfolio_trade_count"] for row in rows) if v is not None]
    pnl_values = [v for v in (row["portfolio_total_pnl"] for row in rows) if v is not None]

    action_flip_count = 0
    for prev, cur in zip(actions, actions[1:]):
        if prev != cur:
            action_flip_count += 1
    action_flip_rate = action_flip_count / max(1, len(actions) - 1)

    low_confidence_count = len([v for v in confidences if v < low_conf_threshold])
    low_confidence_rate = low_confidence_count / max(1, len(confidences))

    processed_delta = (processed_values[-1] - processed_values[0]) if len(processed_values) >= 2 else 0
    dropped_delta = (dropped_values[-1] - dropped_values[0]) if len(dropped_values) >= 2 else 0
    trade_delta = (trade_values[-1] - trade_values[0]) if len(trade_values) >= 2 else 0
    pnl_delta = (pnl_values[-1] - pnl_values[0]) if len(pnl_values) >= 2 else 0.0

    drop_rate_from_counter = (dropped_delta / max(1, processed_delta + dropped_delta)) * 100.0

    start_ts = rows[0]["timestamp_utc"]
    end_ts = rows[-1]["timestamp_utc"]
    start_dt = datetime.fromisoformat(start_ts)
    end_dt = datetime.fromisoformat(end_ts)

    summary = {
        "profile_id": profile_id,
        "start_utc": start_ts,
        "end_utc": end_ts,
        "duration_seconds": round((end_dt - start_dt).total_seconds(), 2),
        "samples_collected": len(rows),
        "poll_failures": poll_failures,
        "decision_delay_mean_ms": mean_or_none(delays),
        "decision_delay_p50_ms": percentile(delays, 50),
        "decision_delay_p95_ms": percentile(delays, 95),
        "decision_delay_p99_ms": percentile(delays, 99),
        "decision_delay_max_ms": max(delays) if delays else None,
        "backend_latency_mean_ms": mean_or_none(backend_latencies),
        "backend_latency_p95_ms": percentile(backend_latencies, 95),
        "backend_latency_max_ms": max(backend_latencies) if backend_latencies else None,
        "packet_loss_mean_pct": mean_or_none(packet_loss),
        "packet_loss_max_pct": max(packet_loss) if packet_loss else None,
        "updates_per_minute_mean": mean_or_none(updates),
        "processed_candles_delta": processed_delta,
        "dropped_candles_delta": dropped_delta,
        "drop_rate_pct_from_counter": drop_rate_from_counter,
        "trade_count_delta": trade_delta,
        "total_pnl_start": pnl_values[0] if pnl_values else None,
        "total_pnl_end": pnl_values[-1] if pnl_values else None,
        "total_pnl_delta": pnl_delta,
        "confidence_mean": mean_or_none(confidences),
        "confidence_std": stdev_or_none(confidences),
        "confidence_min": min(confidences) if confidences else None,
        "confidence_max": max(confidences) if confidences else None,
        "confidence_start": confidences[0] if confidences else None,
        "confidence_end": confidences[-1] if confidences else None,
        "confidence_drift": (confidences[-1] - confidences[0]) if len(confidences) >= 2 else 0.0,
        "low_confidence_threshold": low_conf_threshold,
        "low_confidence_count": low_confidence_count,
        "low_confidence_rate": low_confidence_rate,
        "action_flip_count": action_flip_count,
        "action_flip_rate": action_flip_rate,
        "sample_gap_events": gap_events,
        "max_sample_gap_seconds": max_gap_sec,
    }
    return summary


def write_csv(path: Path, rows):
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_summary_csv(path: Path, summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not existing:
            writer.writeheader()
        writer.writerow(summary)


def parse_args():
    parser = argparse.ArgumentParser(description="Capture backend network-testing telemetry and summary metrics.")
    parser.add_argument("--profile-id", required=True, help="Profile label (for example baseline or loss_3).")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL.")
    parser.add_argument("--duration-minutes", type=float, required=True, help="Session duration in minutes.")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Polling cadence in seconds.")
    parser.add_argument("--timeout-seconds", type=float, default=5.0, help="Request timeout in seconds.")
    parser.add_argument("--output-dir", default="network-testing/results", help="Directory for outputs.")
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional path for rolling summary CSV. Default: <output-dir>/summary_table.csv",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.55,
        help="Prediction confidence threshold for low-confidence rate.",
    )
    parser.add_argument(
        "--gap-threshold-seconds",
        type=float,
        default=3.0,
        help="If sample timestamp gaps exceed this value, count as continuity gap events.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.duration_minutes <= 0:
        print("[error] --duration-minutes must be > 0", file=sys.stderr)
        return 1
    if args.poll_seconds <= 0:
        print("[error] --poll-seconds must be > 0", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_prefix = f"{args.profile_id}_{session_stamp}"
    raw_csv_path = output_dir / f"{session_prefix}_raw.csv"
    summary_json_path = output_dir / f"{session_prefix}_summary.json"
    summary_csv_path = Path(args.summary_csv) if args.summary_csv else output_dir / "summary_table.csv"

    end_time = time.monotonic() + (args.duration_minutes * 60.0)
    rows = []
    poll_failures = 0
    gap_events = 0
    max_gap_sec = 0.0
    last_success_utc = None

    print(f"[info] Starting capture for profile={args.profile_id} at {now_utc_iso()}")
    print(f"[info] Writing raw samples to: {raw_csv_path}")

    while time.monotonic() < end_time:
        loop_started = time.monotonic()
        sample_time = datetime.now(timezone.utc)
        try:
            payload = fetch_health(args.base_url, args.timeout_seconds)
            row = create_row(args.profile_id, sample_time, payload)
            rows.append(row)

            if last_success_utc is not None:
                delta = (sample_time - last_success_utc).total_seconds()
                if delta > args.gap_threshold_seconds:
                    gap_events += 1
                    max_gap_sec = max(max_gap_sec, delta)
            last_success_utc = sample_time
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            poll_failures += 1
            print(f"[warn] Poll failure ({poll_failures}): {exc}")

        elapsed = time.monotonic() - loop_started
        sleep_for = args.poll_seconds - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    if not rows:
        print("[error] No successful samples captured. Check backend URL/status.", file=sys.stderr)
        return 2

    write_csv(raw_csv_path, rows)
    summary = compute_summary(
        profile_id=args.profile_id,
        rows=rows,
        poll_failures=poll_failures,
        low_conf_threshold=args.low_confidence_threshold,
        gap_events=gap_events,
        max_gap_sec=round(max_gap_sec, 4),
    )

    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_summary_csv(summary_csv_path, summary)

    print(f"[ok] Capture complete at {now_utc_iso()}")
    print(f"[ok] Raw CSV: {raw_csv_path}")
    print(f"[ok] Summary JSON: {summary_json_path}")
    print(f"[ok] Summary table row appended to: {summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
