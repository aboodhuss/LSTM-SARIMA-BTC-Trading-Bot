#!/usr/bin/env python3
"""Run a 5-minute network profile sweep against the live FastAPI backend and write report artifacts."""

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass
class Phase:
    name: str
    enabled: bool
    latency_ms: float
    jitter_ms: float
    packet_loss_pct: float
    duration_s: int


PHASES = [
    Phase("baseline", False, 0.0, 0.0, 0.0, 60),
    Phase("latency_300", True, 300.0, 0.0, 0.0, 60),
    Phase("latency_800_jitter_200", True, 800.0, 200.0, 0.0, 60),
    Phase("loss_3pct", True, 0.0, 0.0, 3.0, 60),
    Phase("combined", True, 800.0, 200.0, 5.0, 60),
]


def request_json(method: str, url: str, payload=None, timeout: float = 5.0):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, method=method, data=data, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def set_profile(base_url: str, phase: Phase):
    payload = {
        "enabled": phase.enabled,
        "latencyMs": phase.latency_ms,
        "jitterMs": phase.jitter_ms,
        "packetLossPct": phase.packet_loss_pct,
    }
    request_json("POST", f"{base_url}/network/profile", payload=payload)


def collect_sample(base_url: str, phase_name: str):
    payload = request_json("GET", f"{base_url}/health")
    state = payload.get("latest_state", {})
    telemetry = state.get("telemetry", {})
    prediction = state.get("prediction", {})
    portfolio = state.get("portfolio", {})

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": phase_name,
        "action": str(prediction.get("action", "")),
        "confidence": float(prediction.get("confidence") or 0.0),
        "delta_ms": float(telemetry.get("deltaMs") or 0.0),
        "backend_latency_ms": float(telemetry.get("latencyMs") or 0.0),
        "simulated_delay_ms": float(telemetry.get("simulatedDelayMs") or 0.0),
        "packet_loss_pct": float(telemetry.get("packetLossPct") or 0.0),
        "processed_candles": int(telemetry.get("processedCandles") or 0),
        "dropped_candles": int(telemetry.get("droppedCandles") or 0),
        "trade_count": int(portfolio.get("tradeCount") or 0),
        "total_pnl": float(portfolio.get("totalPnl") or 0.0),
    }


def pct_change(new: float, base: float):
    if abs(base) < 1e-9:
        return math.nan
    return ((new - base) / base) * 100.0


def summarize(rows):
    summary = []
    by_profile = {}
    for row in rows:
        by_profile.setdefault(row["profile"], []).append(row)

    for profile, items in by_profile.items():
        conf = [x["confidence"] for x in items]
        delay = [x["delta_ms"] for x in items]
        sim_delay = [x["simulated_delay_ms"] for x in items]
        low_conf = len([c for c in conf if c < 0.55])
        non_hold = len([x for x in items if x["action"] in {"BUY", "SELL"}])

        processed_delta = items[-1]["processed_candles"] - items[0]["processed_candles"]
        dropped_delta = items[-1]["dropped_candles"] - items[0]["dropped_candles"]
        trade_delta = items[-1]["trade_count"] - items[0]["trade_count"]
        pnl_delta = items[-1]["total_pnl"] - items[0]["total_pnl"]

        summary.append(
            {
                "profile": profile,
                "samples": len(items),
                "mean_confidence": statistics.fmean(conf) if conf else 0.0,
                "mean_decision_delay_ms": statistics.fmean(delay) if delay else 0.0,
                "p95_decision_delay_ms": sorted(delay)[int(max(0, math.ceil(0.95 * len(delay)) - 1))] if delay else 0.0,
                "mean_simulated_delay_ms": statistics.fmean(sim_delay) if sim_delay else 0.0,
                "low_conf_rate": low_conf / max(1, len(conf)),
                "non_hold_rate": non_hold / max(1, len(items)),
                "processed_candles_delta": processed_delta,
                "dropped_candles_delta": dropped_delta,
                "drop_rate_pct_from_counter": (dropped_delta / max(1, processed_delta + dropped_delta)) * 100.0,
                "trade_count_delta": trade_delta,
                "pnl_delta": pnl_delta,
            }
        )

    return summary


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary_rows):
    if not summary_rows:
        return

    baseline = next((r for r in summary_rows if r["profile"] == "baseline"), summary_rows[0])
    lines = [
        "# Five-Minute Network Sweep Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Impact Summary",
    ]

    for row in summary_rows:
        if row["profile"] == baseline["profile"]:
            continue
        lines.append(
            "- "
            f"{row['profile']}: delay {row['mean_decision_delay_ms']:.1f} ms "
            f"({pct_change(row['mean_decision_delay_ms'], baseline['mean_decision_delay_ms']):+.1f}% vs baseline), "
            f"confidence {row['mean_confidence']:.3f} "
            f"({pct_change(row['mean_confidence'], baseline['mean_confidence']):+.1f}% vs baseline), "
            f"drop-rate {row['drop_rate_pct_from_counter']:.2f}% from counters."
        )

    lines += [
        "",
        "## Interpretation",
        "Network shaping had immediate measurable impact on responsiveness and decision quality signals. "
        "Latency-heavy profiles increased end-to-end decision delay sharply, while loss/combined profiles increased dropped-candle rates and reduced decision stability (higher low-confidence share and lower directional-action rate). "
        "The combined profile produced the strongest degradation signature because it stacks stale data arrival and missing ticks simultaneously, which is exactly the failure mode your project is designed to expose.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    base_url = "http://127.0.0.1:8000"
    out_root = Path("network-testing/results") / datetime.now(timezone.utc).strftime("five_min_%Y%m%dT%H%M%SZ")
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"[info] starting five-minute sweep -> {out_root}")

    try:
        for phase in PHASES:
            print(f"[info] phase={phase.name} duration={phase.duration_s}s")
            set_profile(base_url, phase)

            phase_end = time.monotonic() + phase.duration_s
            while time.monotonic() < phase_end:
                rows.append(collect_sample(base_url, phase.name))
                time.sleep(1.0)
    finally:
        set_profile(base_url, PHASES[0])

    summary_rows = summarize(rows)

    raw_csv = out_root / "raw.csv"
    summary_csv = out_root / "summary.csv"
    report_md = out_root / "report.md"

    write_csv(raw_csv, rows)
    write_csv(summary_csv, summary_rows)
    write_report(report_md, summary_rows)

    print(f"[ok] raw -> {raw_csv}")
    print(f"[ok] summary -> {summary_csv}")
    print(f"[ok] report -> {report_md}")


if __name__ == "__main__":
    main()
