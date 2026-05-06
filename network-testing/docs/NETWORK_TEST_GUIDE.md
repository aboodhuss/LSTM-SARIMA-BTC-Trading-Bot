# Network Testing Execution Guide

This document is the execution guide for network testing and reporting the bot evaluation.

## 1. System Scope Under Test

### 1.1 Current Architecture

- Backend: Python + FastAPI.
- Data source: Binance WebSocket (`btcusdt@kline_1m`).
- Core loop: ingestion -> feature extraction -> PyTorch inference -> BUY/SELL/HOLD paper-trade action.
- Telemetry: per-cycle timing and decision metadata are exposed through `/health` and `/ws`.
- Frontend: React (Vite) dashboard with live candles, model outputs, telemetry, and network profile controls.

### 1.2 Feature Pipeline

Features currently exercised in real time:

- Higher High / Lower Low structural flags.
- Fair Value Gap (FVG) detection.
- Confluence scoring across signal alignment.

### 1.3 Adaptive Learning Loop

- Live inference is active.
- Feedback and retrain-related artifacts are logged in backend files (`trade_feedback.jsonl`, `training_report.json`, `model_history.jsonl`).
- A core objective is learning from failure cases, especially confidence mismatches against realized outcomes.

## 2. Stage Gates Before Network Shaping

Complete these gates once under normal connectivity:

1. Ingestion and feature extraction stable for a continuous 60+ minute window.
2. Decision loop timestamps are present and non-null (`tickTime`, `receivedTime`, `signalTime`, `deltaMs`).
3. Frontend and backend integration verified (`/ws` stream active, dashboard updating).
4. Baseline metric window captured and stored in `network-testing/results/`.
5. Baseline artifacts signed off before shaped tests begin.

## 3. Test Profiles

Use one profile at a time, with identical runtime windows.

- Baseline: no shaping.
- Latency injection: fixed and variable delay bands.
- Packet loss: controlled drop rates (1%, 3%, 5%).
- Bandwidth throttling: reduced throughput to force stale/queued decisions.

Profile IDs and commands are in `profiles/network_profile_matrix.csv`.

## 4. Tooling

### 4.1 Linux (Primary)

- `tc` + `netem` + `tbf` via:
  - `scripts/apply_tc_profile.sh`
  - `scripts/run_profile_session.sh`

### 4.2 Windows Fallback

- Clumsy profile mapping in `profiles/windows_clumsy_profiles.md`.

## 5. Measurement Plan

For every profile session, collect:

1. Decision timing:
   - Mean, p50, p95, p99 of `telemetry.deltaMs`.
2. Tick processing continuity:
   - `processedCandles` delta.
   - `droppedCandles` delta.
   - Sample gap events in polling stream.
3. Trading behavior:
   - `tradeCount` delta.
   - `totalPnl` delta.
4. Confidence and model behavior:
   - Mean/stdev/min/max of prediction confidence.
   - Low-confidence rate (default threshold: 0.55).
   - Action flip frequency as a disagreement proxy.

`scripts/capture_metrics.py` outputs:

- Raw sample CSV.
- Session summary JSON.
- Rolling summary table CSV for profile-to-profile comparison.

## 6. Standard Experimental Procedure

1. Start backend/frontend.
2. Run baseline for 180 minutes.
3. Apply exactly one shaping profile.
4. Run for fixed window (recommended 90 minutes each profile).
5. Remove shaping rules and verify recovery with a post-profile baseline (30 minutes minimum).
6. Repeat for each profile.
7. Compare all sessions against baseline using the same summary fields.

Recommended command pattern:

```bash
cd trading-bot

# Baseline
./network-testing/scripts/run_profile_session.sh <iface> baseline 180

# Shaped profiles
./network-testing/scripts/run_profile_session.sh <iface> latency_150 90
./network-testing/scripts/run_profile_session.sh <iface> latency_300_jitter_50 90
./network-testing/scripts/run_profile_session.sh <iface> loss_1 90
./network-testing/scripts/run_profile_session.sh <iface> loss_3 90
./network-testing/scripts/run_profile_session.sh <iface> loss_5 90
./network-testing/scripts/run_profile_session.sh <iface> bandwidth_5mbit 90
./network-testing/scripts/run_profile_session.sh <iface> bandwidth_2mbit 90
./network-testing/scripts/run_profile_session.sh <iface> bandwidth_1mbit 90

# Recovery verification
./network-testing/scripts/run_profile_session.sh <iface> baseline_recovery 30
```

## 7. Interpretation Rules

Do not use PnL alone to conclude behavior changes.

For every profile where `totalPnl` drops, inspect:

1. Whether `deltaMs` and p95/p99 timing increased.
2. Whether dropped candle count or sample gaps increased.
3. Whether low-confidence rate rose.
4. Whether action-flip rate increased.
5. Whether behavior recovered after profile removal.

A valid finding ties outcome changes to at least one observed timing or continuity shift.

## 8. Deliverables Required From Network Team

Each test run submission must include:

1. Raw metrics CSV(s) from `network-testing/results/`.
2. Session summary JSON(s).
3. Updated summary table CSV.
4. Notes describing anomalies and recovery behavior.
5. One comparison table against baseline:
   - decision delay
   - trade count
   - simulated PnL
   - retrain/promotion-related events observed in logs

## 9. Recovery and Safety

- Always clear shaping at end:
  - `sudo ./network-testing/scripts/apply_tc_profile.sh <iface> clear`
- Validate clear state:
  - `sudo tc -s qdisc show dev <iface>`
- If any profile hangs data flow, clear immediately and record timestamp in session notes.
