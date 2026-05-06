"""
model_comparison.py – Multi-model network-resilience comparison engine.

This module powers the "Start Network Test" button in the dashboard.  It
replays a window of BTC candles through two model families under six
simulated network conditions, producing the side-by-side comparison table.

Model families
--------------
  1. LSTM baseline   – The recurrent neural network from ai_core.py.
  2. LSTM + SARIMA   – Same LSTM, but with SARIMA-reconstructed missing
                       candles (sarima_preprocessor.py).

Simulated network phases
------------------------
  • baseline              – No degradation
  • latency_300           – 300 ms fixed delay
  • latency_800_jitter_200 – 800 ms ± 200 ms jitter
  • loss_3pct             – 3% random packet loss
  • combined              – 800 ms + 200 ms jitter + 5% loss
  • sarima_stress_loss_15pct – 15% packet loss stress case for SARIMA

Interpreting results
--------------------
Each phase runs 15 live candles (1 candle per minute), so each profile has a
15-minute window.  At this sample size, PnL differences between
phases are dominated by *stochastic noise* — i.e. which specific 15-minute window of BTC price action the
phase happened to land on.  A positive PnL under degraded conditions does
NOT mean degradation helps; it means the market moved favourably during
that specific time window.

For the presentation, focus on:
  • Decision delay increases across phases (measurable, reproducible).
  • Data quality drops and imputed-rate increases under loss.
  • Signal accuracy / flip-rate changes as input quality degrades.
  • PnL as a secondary, illustrative metric — not a definitive ranking.

Data sources
------------
  • Live candles      : Binance WebSocket / REST API (BTCUSDT 1m)
  • SARIMA profile    : Kaggle BTC minute dataset
      https://www.kaggle.com/datasets/swaptr/bitcoin-historical-data
"""

from __future__ import annotations

import random
import statistics
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from ai_core import AdaptiveTradingLSTM, FEATURE_COLUMNS, predict_action
from data_ingestion import CANDLE_INTERVAL_MS, get_live_closed_candles
from feature_engineering import PatternRecognizer
from sarima_preprocessor import PROFILE_PATH, load_profile, reconstruct_missing_rows
from train import fetch_historical_data


DEFAULT_FETCH_LOOPS = 1
DEFAULT_LIVE_PHASE_CANDLES = 15
LIVE_HISTORY_LIMIT = 300
MAX_REASONABLE_PRICE = 1_000_000.0

MODEL_FAMILIES = [
    # These keys are shared with the UI so each replay row can be grouped back
    # into the exact model family discussed in the presentation.
    {"key": "lstm_baseline", "label": "LSTM"},
    {"key": "sarima_lstm", "label": "LSTM + SARIMA"},
]

HYBRID_REPAIR_MIN_CONFIDENCE = 0.50
HYBRID_REPAIR_MAX_CONFIDENCE_PENALTY = 0.30
HYBRID_REPAIR_STRONG_CONFLUENCE = 0.08


def network_phases():
    return [
        {"name": "baseline", "enabled": False, "latencyMs": 0.0, "jitterMs": 0.0, "packetLossPct": 0.0},
        {"name": "latency_300", "enabled": True, "latencyMs": 300.0, "jitterMs": 0.0, "packetLossPct": 0.0},
        {"name": "latency_800_jitter_200", "enabled": True, "latencyMs": 800.0, "jitterMs": 200.0, "packetLossPct": 0.0},
        {"name": "loss_3pct", "enabled": True, "latencyMs": 0.0, "jitterMs": 0.0, "packetLossPct": 3.0},
        {"name": "combined", "enabled": True, "latencyMs": 800.0, "jitterMs": 200.0, "packetLossPct": 5.0},
        {"name": "sarima_stress_loss_15pct", "enabled": True, "latencyMs": 0.0, "jitterMs": 0.0, "packetLossPct": 15.0},
    ]


def load_lstm_model():
    from pathlib import Path

    best_path = Path("adaptive_weights.best.pth")
    model_path = best_path if best_path.exists() else Path("adaptive_weights.pth")
    model = AdaptiveTradingLSTM(input_size=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def new_state():
    return {
        "cash": 100000.0,
        "position_side": "FLAT",
        "position_size": 0.0,
        "entry_price": None,
        "realized_pnl": 0.0,
        "trade_count": 0,
        "wins": 0,
        "losses": 0,
        "pending_signal": None,
        "resolved": 0,
        "correct": 0,
        "confidences": [],
        "actions": [],
        "quality": [],
        "imputed_rates": [],
        "decision_delay_ms": [],
        "high_delay_count": 0,
        "position_notional": 0.0,
        "equity": 100000.0,
        "peak_equity": 100000.0,
        "max_drawdown_pct": 0.0,
    }


def update_equity(state, close_price):
    unrealized = 0.0
    if state["entry_price"] is not None and state["position_size"] > 0:
        if state["position_side"] == "LONG":
            unrealized = (close_price - state["entry_price"]) * state["position_size"]
        elif state["position_side"] == "SHORT":
            unrealized = (state["entry_price"] - close_price) * state["position_size"]
    state["equity"] = state["cash"] + unrealized
    state["peak_equity"] = max(state["peak_equity"], state["equity"])
    if state["peak_equity"] > 0:
        drawdown = ((state["peak_equity"] - state["equity"]) / state["peak_equity"]) * 100.0
        state["max_drawdown_pct"] = max(state["max_drawdown_pct"], drawdown)


def close_position(state, close_price):
    if state["position_side"] == "FLAT" or state["entry_price"] is None or state["position_size"] <= 0:
        return
    if state["position_side"] == "LONG":
        pnl = (close_price - state["entry_price"]) * state["position_size"]
    else:
        pnl = (state["entry_price"] - close_price) * state["position_size"]
    state["cash"] += pnl
    state["realized_pnl"] += pnl
    state["trade_count"] += 1
    if pnl >= 0:
        state["wins"] += 1
    else:
        state["losses"] += 1
    state["position_side"] = "FLAT"
    state["position_size"] = 0.0
    state["entry_price"] = None
    state["position_notional"] = 0.0


def open_position(state, action, close_price, confidence):
    desired_side = "LONG" if action == "BUY" else "SHORT"
    if state["position_side"] == desired_side:
        return
    if state["position_side"] != "FLAT":
        close_position(state, close_price)
    allocation_notional = max(2000.0, state["equity"] * min(0.2, 0.04 + confidence * 0.12))
    size = allocation_notional / max(close_price, 1e-9)
    state["position_side"] = desired_side
    state["position_size"] = size
    state["entry_price"] = close_price
    state["position_notional"] = allocation_notional


def resolve_signal(state, close_price):
    pending = state["pending_signal"]
    if pending is None:
        return
    action = pending["action"]
    previous_close = pending["close"]
    if action == "BUY":
        correct = close_price > previous_close
    elif action == "SELL":
        correct = close_price < previous_close
    else:
        correct = abs(close_price - previous_close) / max(previous_close, 1e-9) < 0.001
    state["resolved"] += 1
    state["correct"] += int(correct)
    state["pending_signal"] = None


def predict_for_family(family_key, lstm_model, df_features):
    return predict_action(lstm_model, df_features)


def apply_hybrid_repair_gate(family_key, action, confidence, df_features):
    if family_key != "sarima_lstm" or action not in {"BUY", "SELL"} or df_features.empty:
        return action, confidence, False
    if "Is_Imputed" not in df_features.columns:
        return action, confidence, False

    recent_imputed_rate = float(df_features["Is_Imputed"].astype(float).tail(20).mean())
    if recent_imputed_rate <= 0:
        return action, confidence, False

    latest_row = df_features.iloc[-1]
    latest_is_imputed = bool(latest_row.get("Is_Imputed", False))
    latest_quality = float(latest_row.get("Data_Quality", 1.0))
    confluence = float(latest_row.get("Confluence_Score", 0.0))
    confluence_aligned = (
        confluence >= HYBRID_REPAIR_STRONG_CONFLUENCE
        if action == "BUY"
        else confluence <= -HYBRID_REPAIR_STRONG_CONFLUENCE
    )

    # Do not open or flip a position on the synthetic candle itself. Let the
    # repaired sequence stabilize and wait for a real candle to confirm it.
    if latest_is_imputed:
        return "HOLD", 0.0, True

    quality_gap = max(0.0, 1.0 - latest_quality)
    confidence_penalty = min(
        HYBRID_REPAIR_MAX_CONFIDENCE_PENALTY,
        (recent_imputed_rate * 1.1) + (quality_gap * 0.35),
    )
    adjusted_confidence = float(max(0.0, confidence * (1.0 - confidence_penalty)))
    min_confidence = HYBRID_REPAIR_MIN_CONFIDENCE + min(0.06, recent_imputed_rate * 0.15)

    if adjusted_confidence < min_confidence or not confluence_aligned:
        return "HOLD", adjusted_confidence, True
    return action, adjusted_confidence, False


def build_naive_row(history_df, open_ms, receive_timestamp_ms, gap_minutes):
    last_close = float(history_df.iloc[-1]["Close"])
    recent = history_df.tail(30)
    median_range = (
        ((recent["High"] - recent["Low"]) / recent["Close"].replace(0, pd.NA)).dropna().median()
        if not recent.empty
        else 0.001
    )
    range_pct = float(max(0.0004, min(0.02, median_range if pd.notna(median_range) else 0.001)))
    high_price = last_close * (1.0 + range_pct * 0.4)
    low_price = last_close * (1.0 - range_pct * 0.4)
    close_tick_ms = open_ms + 60_000 - 1
    return {
        "Timestamp": pd.to_datetime(open_ms, unit="ms"),
        "Open": last_close,
        "High": high_price,
        "Low": low_price,
        "Close": last_close,
        "Volume": 0.0,
        "Is_Imputed": True,
        "Gap_Minutes": float(max(1.0, gap_minutes)),
        "Receive_Lag_Ms": float(max(0.0, receive_timestamp_ms - close_tick_ms)),
    }


def family_fill_rows(family_key, history_df, open_times_ms, receive_timestamp_ms, sarima_profile):
    # SARIMA is only a missing-candle repair layer for the hybrid path.
    # It must not smooth or rewrite complete real-candle sequences.
    if family_key == "sarima_lstm":
        if sarima_profile is not None and len(history_df) >= 80:
            try:
                return reconstruct_missing_rows(
                    history_df,
                    open_times_ms=open_times_ms,
                    receive_timestamp_ms=receive_timestamp_ms,
                    profile=sarima_profile,
                )
            except Exception:
                pass
    rows = []
    temp = history_df.copy()
    for step, open_ms in enumerate(open_times_ms):
        gap_minutes = max(1.0, len(open_times_ms) - step)
        row = build_naive_row(temp, open_ms, receive_timestamp_ms, gap_minutes)
        rows.append(row)
        temp = pd.concat([temp, pd.DataFrame([row])], ignore_index=True)
    return rows


def append_row(history_df, row):
    next_df = pd.concat([history_df, pd.DataFrame([row])], ignore_index=True)
    if "Is_Imputed" not in next_df.columns:
        next_df["Is_Imputed"] = False
    next_df["_real_candle_priority"] = next_df["Is_Imputed"].astype(bool).map({False: 1, True: 0})
    next_df = (
        next_df.sort_values(["Timestamp", "_real_candle_priority"])
        .drop_duplicates(subset=["Timestamp"], keep="last")
        .drop(columns=["_real_candle_priority"])
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    return next_df


def valid_price(value):
    return np.isfinite(value) and 0.0 < float(value) < MAX_REASONABLE_PRICE


def sanitize_market_frame(raw_df):
    frame = raw_df.copy()
    if frame.empty:
        return frame
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"])
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["Open"].map(valid_price)
        & frame["High"].map(valid_price)
        & frame["Low"].map(valid_price)
        & frame["Close"].map(valid_price)
        & (frame["High"] >= frame["Low"])
    ].copy()
    frame["Volume"] = frame["Volume"].where(np.isfinite(frame["Volume"]), 0.0).fillna(0.0)
    if "Is_Imputed" not in frame.columns:
        frame["Is_Imputed"] = False
    if "Gap_Minutes" not in frame.columns:
        frame["Gap_Minutes"] = 0.0
    if "Receive_Lag_Ms" not in frame.columns:
        frame["Receive_Lag_Ms"] = 0.0
    return frame.drop_duplicates(subset=["Timestamp"], keep="last").sort_values("Timestamp").reset_index(drop=True)


def bootstrap_live_history(fetch_loops=DEFAULT_FETCH_LOOPS):
    raw_df = fetch_historical_data(symbol="BTCUSDT", loops=max(1, int(fetch_loops)))
    frame = sanitize_market_frame(raw_df).tail(LIVE_HISTORY_LIMIT).reset_index(drop=True)
    if len(frame) < 80:
        raise ValueError("Not enough clean BTC candles available to seed the live comparison run.")
    return frame


def build_actual_market_row(candle):
    open_ms = int(candle["open_ms"])
    receive_timestamp_ms = int(candle["receive_timestamp_ms"])
    close_tick_ms = open_ms + CANDLE_INTERVAL_MS - 1
    return {
        "Timestamp": pd.to_datetime(open_ms, unit="ms"),
        "Open": float(candle["Open"]),
        "High": float(candle["High"]),
        "Low": float(candle["Low"]),
        "Close": float(candle["Close"]),
        "Volume": float(candle["Volume"]),
        "Is_Imputed": False,
        "Gap_Minutes": 0.0,
        "Receive_Lag_Ms": float(max(0, receive_timestamp_ms - close_tick_ms)),
    }


def build_live_phase_history(bootstrap_df, upto_seq):
    """
    Build a clean history ending at the phase start.

    The live comparison phases run sequentially. Reusing only the original
    REST bootstrap for later phases creates a fake timeline gap, which then
    inflates imputed-rate even for latency-only phases. Appending the real
    closed candles already observed keeps each phase aligned with live time.
    """
    history_df = bootstrap_df.copy()
    closed_items, _ = get_live_closed_candles(after_seq=0, limit=None)
    for item in closed_items:
        if int(item["seq"]) > int(upto_seq):
            continue
        history_df = append_row(history_df, build_actual_market_row(item["row"]))
    return sanitize_market_frame(history_df).tail(LIVE_HISTORY_LIMIT).reset_index(drop=True)


def simulate_family_on_phase(
    raw_df,
    family_key,
    phase,
    lstm_model,
    sarima_profile,
    progress_callback=None,
    should_continue=None,
):
    # Replay the same candle window under one network condition for one model
    # family so the final table compares like-for-like runs.
    rng = random.Random(f"{phase['name']}::{family_key}")
    history_df = raw_df.iloc[:60].copy().reset_index(drop=True)
    state = new_state()
    engine = PatternRecognizer()
    dropped_count = 0
    processed = 0
    estimated_samples = max(0, len(raw_df) - 60)

    for row_index in range(60, len(raw_df)):
        if should_continue is not None and not should_continue():
            break
        raw_row = raw_df.iloc[row_index].copy()
        open_ms = int(pd.Timestamp(raw_row["Timestamp"]).timestamp() * 1000)

        if history_df.empty:
            previous_open_ms = None
        else:
            previous_open_ms = int(pd.Timestamp(history_df.iloc[-1]["Timestamp"]).timestamp() * 1000)

        if phase["enabled"]:
            applied_delay = phase["latencyMs"]
            if phase["jitterMs"] > 0:
                applied_delay = max(0.0, applied_delay + rng.uniform(-phase["jitterMs"], phase["jitterMs"]))
        else:
            applied_delay = 0.0

        should_drop = phase["packetLossPct"] > 0 and rng.random() < (phase["packetLossPct"] / 100.0)
        if should_drop:
            dropped_count += 1
            continue

        receive_timestamp_ms = int(open_ms + 60_000 - 1 + applied_delay)
        if previous_open_ms is not None:
            missing = int((open_ms - previous_open_ms) / 60_000) - 1
            if missing > 0:
                open_times_ms = [previous_open_ms + ((step + 1) * 60_000) for step in range(missing)]
                reconstructed_rows = family_fill_rows(
                    family_key,
                    history_df,
                    open_times_ms,
                    receive_timestamp_ms,
                    sarima_profile,
                )
                for fill_row in reconstructed_rows:
                    history_df = append_row(history_df, fill_row)

        close_tick_ms = open_ms + 60_000 - 1
        actual_row = {
            "Timestamp": pd.to_datetime(raw_row["Timestamp"]),
            "Open": float(raw_row["Open"]),
            "High": float(raw_row["High"]),
            "Low": float(raw_row["Low"]),
            "Close": float(raw_row["Close"]),
            "Volume": float(raw_row["Volume"]),
            "Is_Imputed": False,
            "Gap_Minutes": 0.0,
            "Receive_Lag_Ms": float(max(0.0, receive_timestamp_ms - close_tick_ms)),
        }
        history_df = append_row(history_df, actual_row)
        features = engine.extract_features(history_df).reset_index(drop=True)
        latest_features = features.iloc[-1]
        close_price = float(latest_features["Close"])
        resolve_signal(state, close_price)
        action, confidence = predict_for_family(family_key, lstm_model, features)
        action, confidence, repair_blocked = apply_hybrid_repair_gate(family_key, action, confidence, features)
        if action in {"BUY", "SELL"}:
            open_position(state, action, close_price, confidence)
        elif action == "HOLD" and state["position_side"] != "FLAT" and not repair_blocked:
            close_position(state, close_price)
        state["pending_signal"] = {"action": action, "close": close_price}
        state["actions"].append(action)
        state["confidences"].append(float(confidence))
        state["quality"].append(float(latest_features.get("Data_Quality", 1.0)))
        state["imputed_rates"].append(float(features["Is_Imputed"].astype(float).tail(20).mean() * 100.0))
        state["decision_delay_ms"].append(float(latest_features.get("Receive_Lag_Ms", 0.0)))
        if float(latest_features.get("Receive_Lag_Ms", 0.0)) >= 500.0:
            state["high_delay_count"] += 1
        processed += 1
        update_equity(state, close_price)
        if progress_callback is not None and (processed == 1 or processed % 25 == 0 or row_index == len(raw_df) - 1):
            progress_callback(
                {
                    "currentSamples": processed,
                    "estimatedSamples": estimated_samples,
                }
            )

    if state["position_side"] != "FLAT" and not history_df.empty:
        close_position(state, float(history_df.iloc[-1]["Close"]))
        update_equity(state, float(history_df.iloc[-1]["Close"]))

    actions = state["actions"]
    action_flips = sum(1 for prev, cur in zip(actions, actions[1:]) if prev != cur)
    sorted_delays = sorted(state["decision_delay_ms"]) if state["decision_delay_ms"] else [0.0]
    p95_index = max(0, int(np.ceil(0.95 * len(sorted_delays))) - 1)
    return {
        "modelKey": family_key,
        "modelLabel": next(item["label"] for item in MODEL_FAMILIES if item["key"] == family_key),
        "phase": phase["name"],
        "samples": processed,
        "mean_confidence": statistics.fmean(state["confidences"]) if state["confidences"] else 0.0,
        "mean_decision_delay_ms": statistics.fmean(state["decision_delay_ms"]) if state["decision_delay_ms"] else 0.0,
        "p95_decision_delay_ms": sorted_delays[p95_index],
        "mean_data_quality": statistics.fmean(state["quality"]) if state["quality"] else 1.0,
        "mean_imputed_rate_pct": statistics.fmean(state["imputed_rates"]) if state["imputed_rates"] else 0.0,
        "high_delay_rate": state["high_delay_count"] / max(1, processed),
        "action_flip_rate": action_flips / max(1, len(actions) - 1),
        "trade_count_delta": int(state["trade_count"]),
        "pnl_delta": float(round(state["realized_pnl"], 2)),
        "drop_rate_pct_from_counter": (dropped_count / max(1, dropped_count + processed)) * 100.0,
        "non_hold_rate": len([action for action in actions if action != "HOLD"]) / max(1, len(actions)),
        "signal_accuracy": state["correct"] / max(1, state["resolved"]),
        "max_drawdown_pct": float(state["max_drawdown_pct"]),
    }


def build_paragraph(rows):
    if not rows:
        return "No model comparison rows were generated."

    def report_number(value):
        numeric = float(value)
        formatted = f"{abs(numeric):.2f}"
        return f"({formatted})" if numeric < 0 else f"+{formatted}"

    best_row = max(rows, key=lambda row: (row["pnl_delta"], row["signal_accuracy"]))
    strongest_quality = max(rows, key=lambda row: row["mean_data_quality"])
    weakest_network = min(rows, key=lambda row: row["mean_data_quality"])
    return (
        f"The strongest overall result came from {best_row['modelLabel']} during the {best_row['phase']} phase, "
        f"with PnL {report_number(best_row['pnl_delta'])} and signal accuracy {best_row['signal_accuracy']:.3f}. "
        f"The cleanest data path was {strongest_quality['modelLabel']} in {strongest_quality['phase']} "
        f"(quality {strongest_quality['mean_data_quality']:.3f}), while the most degraded path was "
        f"{weakest_network['modelLabel']} in {weakest_network['phase']} "
        f"(quality {weakest_network['mean_data_quality']:.3f})."
    )


def build_report(rows, raw_df, fetch_loops, total_runs, completed):
    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "fetch_loops_used": int(fetch_loops),
        "replay_candles": int(len(raw_df)),
        "summary_rows": deepcopy(rows),
        "paragraph": build_paragraph(rows),
        "models": deepcopy(MODEL_FAMILIES),
        "phases": [phase["name"] for phase in network_phases()],
        "completed_runs": int(len(rows)),
        "total_runs": int(total_runs),
        "completed": bool(completed),
        "run_mode": "completion-based",
        "estimated_samples_per_run": max(0, len(raw_df) - 60),
    }


def generate_model_comparison_report(
    fetch_loops: int = DEFAULT_FETCH_LOOPS,
    progress_callback=None,
    should_continue=None,
):
    # This powers the Start button in the dashboard's multi-model network test.
    # A single run replays one BTC window through both model families so
    # the output table is a fair side-by-side network-resilience comparison.
    loops = max(1, int(fetch_loops))
    raw_df = fetch_historical_data(symbol="BTCUSDT", loops=loops).reset_index(drop=True)
    lstm_model = load_lstm_model()
    try:
        sarima_profile = load_profile(PROFILE_PATH)
    except Exception:
        sarima_profile = None

    phases = network_phases()
    total_runs = len(phases) * len(MODEL_FAMILIES)
    rows = []
    for phase in phases:
        if should_continue is not None and not should_continue():
            break
        phase_rows = []
        for family in MODEL_FAMILIES:
            if should_continue is not None and not should_continue():
                break
            if progress_callback is not None:
                progress_callback(
                    {
                        "currentPhase": phase["name"],
                        "currentModel": family["label"],
                        "completedRuns": len(rows),
                        "totalRuns": total_runs,
                        "rows": deepcopy(rows),
                        "report": build_report(rows, raw_df, loops, total_runs, completed=False),
                    }
                )
            def run_progress(progress_payload):
                if progress_callback is not None:
                    progress_callback(
                        {
                            "currentPhase": phase["name"],
                            "currentModel": family["label"],
                            "completedRuns": len(rows),
                            "totalRuns": total_runs,
                            "rows": deepcopy(rows),
                            "currentSamples": int(progress_payload.get("currentSamples") or 0),
                            "estimatedSamples": int(progress_payload.get("estimatedSamples") or max(0, len(raw_df) - 60)),
                            "report": build_report(rows, raw_df, loops, total_runs, completed=False),
                        }
                    )

            result = simulate_family_on_phase(
                raw_df,
                family["key"],
                phase,
                lstm_model,
                sarima_profile,
                run_progress,
                should_continue,
            )
            phase_rows.append(result)

        if not phase_rows:
            break
        baseline_lstm = next((row for row in phase_rows if row["modelKey"] == "lstm_baseline"), None)
        baseline_pnl = baseline_lstm["pnl_delta"] if baseline_lstm else 0.0
        for row in phase_rows:
            row["pnl_impact_vs_lstm"] = row["pnl_delta"] - baseline_pnl
            rows.append(row)

        if progress_callback is not None:
            progress_callback(
                {
                    "currentPhase": phase["name"],
                    "currentModel": None,
                    "completedRuns": len(rows),
                    "totalRuns": total_runs,
                    "rows": deepcopy(rows),
                    "report": build_report(rows, raw_df, loops, total_runs, completed=False),
                }
            )

    completed = len(rows) == total_runs
    return build_report(rows, raw_df, loops, total_runs, completed)


def new_live_runtime(family_key, phase, bootstrap_df, sarima_profile):
    return {
        "family_key": family_key,
        "phase": phase,
        "history_df": bootstrap_df.copy(),
        "state": new_state(),
        "engine": PatternRecognizer(),
        "rng": random.Random(f"live::{phase['name']}::{family_key}"),
        "sarima_profile": sarima_profile,
        "processed": 0,
        "dropped_count": 0,
        "last_actual_close": None,
    }


def simulate_live_step(runtime, actual_candle, lstm_model):
    phase = runtime["phase"]
    history_df = runtime["history_df"]
    state = runtime["state"]
    rng = runtime["rng"]
    open_ms = int(actual_candle["open_ms"])
    receive_timestamp_ms = int(actual_candle["receive_timestamp_ms"])

    applied_delay = float(phase["latencyMs"]) if phase["enabled"] else 0.0
    if phase["enabled"] and float(phase["jitterMs"]) > 0:
        applied_delay = max(0.0, applied_delay + rng.uniform(-float(phase["jitterMs"]), float(phase["jitterMs"])))

    previous_open_ms = None
    if not history_df.empty:
        previous_open_ms = int(pd.Timestamp(history_df.iloc[-1]["Timestamp"]).timestamp() * 1000)

    perceived_rows = []
    if previous_open_ms is not None:
        missing = int((open_ms - previous_open_ms) / CANDLE_INTERVAL_MS) - 1
        if missing > 0:
            gap_open_times = [previous_open_ms + ((step + 1) * CANDLE_INTERVAL_MS) for step in range(missing)]
            perceived_rows.extend(
                family_fill_rows(
                    runtime["family_key"],
                    history_df,
                    gap_open_times,
                    receive_timestamp_ms,
                    runtime["sarima_profile"],
                )
            )

    loss_pct = float(phase["packetLossPct"])
    should_drop = bool(loss_pct > 0 and rng.random() < (loss_pct / 100.0))
    if should_drop:
        runtime["dropped_count"] += 1
        perceived_rows.extend(
            family_fill_rows(
                runtime["family_key"],
                history_df if not perceived_rows else sanitize_market_frame(pd.concat([history_df, pd.DataFrame(perceived_rows)], ignore_index=True)),
                [open_ms],
                receive_timestamp_ms,
                runtime["sarima_profile"],
            )
        )
    else:
        perceived = build_actual_market_row(actual_candle)
        perceived["Receive_Lag_Ms"] = float(applied_delay)
        perceived_rows.append(perceived)

    for row in perceived_rows:
        history_df = append_row(history_df, row)
    history_df = sanitize_market_frame(history_df).tail(LIVE_HISTORY_LIMIT).reset_index(drop=True)
    runtime["history_df"] = history_df

    actual_close = float(actual_candle["Close"])
    actual_high = float(actual_candle["High"])
    actual_low = float(actual_candle["Low"])
    runtime["last_actual_close"] = actual_close
    resolve_signal(state, actual_close)

    features = runtime["engine"].extract_features(history_df).reset_index(drop=True)
    action, confidence = predict_for_family(runtime["family_key"], lstm_model, features)
    action, confidence, repair_blocked = apply_hybrid_repair_gate(runtime["family_key"], action, confidence, features)
    if action in {"BUY", "SELL"}:
        open_position(state, action, actual_close, confidence)
    elif action == "HOLD" and state["position_side"] != "FLAT" and not repair_blocked:
        close_position(state, actual_close)

    state["pending_signal"] = {"action": action, "close": actual_close}
    state["actions"].append(action)
    state["confidences"].append(float(confidence))
    latest_features = features.iloc[-1]
    state["quality"].append(float(np.clip(float(latest_features.get("Data_Quality", 1.0)), 0.0, 1.0)))
    state["imputed_rates"].append(float(features["Is_Imputed"].astype(float).tail(20).mean() * 100.0))
    decision_delay_ms = float(perceived_rows[-1].get("Receive_Lag_Ms", applied_delay))
    state["decision_delay_ms"].append(decision_delay_ms)
    if decision_delay_ms >= 500.0:
        state["high_delay_count"] += 1
    runtime["processed"] += 1

    if state["position_side"] == "LONG":
        if actual_low <= (state["entry_price"] or actual_close):
            update_equity(state, actual_close)
        else:
            update_equity(state, actual_close)
    else:
        update_equity(state, actual_close)


def finalize_live_runtime(runtime):
    state = runtime["state"]
    history_df = runtime["history_df"]
    if state["position_side"] != "FLAT":
        close_price = runtime["last_actual_close"]
        if close_price is None and not history_df.empty:
            close_price = float(history_df.iloc[-1]["Close"])
        if close_price is not None:
            close_position(state, float(close_price))
            update_equity(state, float(close_price))

    actions = state["actions"]
    action_flips = sum(1 for prev, cur in zip(actions, actions[1:]) if prev != cur)
    sorted_delays = sorted(state["decision_delay_ms"]) if state["decision_delay_ms"] else [0.0]
    p95_index = max(0, int(np.ceil(0.95 * len(sorted_delays))) - 1)
    return {
        "modelKey": runtime["family_key"],
        "modelLabel": next(item["label"] for item in MODEL_FAMILIES if item["key"] == runtime["family_key"]),
        "phase": runtime["phase"]["name"],
        "samples": runtime["processed"],
        "mean_confidence": statistics.fmean(state["confidences"]) if state["confidences"] else 0.0,
        "mean_decision_delay_ms": statistics.fmean(state["decision_delay_ms"]) if state["decision_delay_ms"] else 0.0,
        "p95_decision_delay_ms": sorted_delays[p95_index],
        "mean_data_quality": statistics.fmean(state["quality"]) if state["quality"] else 1.0,
        "mean_imputed_rate_pct": statistics.fmean(state["imputed_rates"]) if state["imputed_rates"] else 0.0,
        "high_delay_rate": state["high_delay_count"] / max(1, runtime["processed"]),
        "action_flip_rate": action_flips / max(1, len(actions) - 1),
        "trade_count_delta": int(state["trade_count"]),
        "pnl_delta": float(round(state["realized_pnl"], 2)),
        "drop_rate_pct_from_counter": (runtime["dropped_count"] / max(1, runtime["processed"] + runtime["dropped_count"])) * 100.0,
        "non_hold_rate": len([action for action in actions if action != "HOLD"]) / max(1, len(actions)),
        "signal_accuracy": state["correct"] / max(1, state["resolved"]),
        "max_drawdown_pct": float(state["max_drawdown_pct"]),
    }


def build_live_report(rows, phase_candles, total_runs, completed):
    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "phase_candles": int(phase_candles),
        "summary_rows": deepcopy(rows),
        "paragraph": build_paragraph(rows),
        "models": deepcopy(MODEL_FAMILIES),
        "phases": [phase["name"] for phase in network_phases()],
        "completed_runs": int(len(rows)),
        "total_runs": int(total_runs),
        "completed": bool(completed),
        "run_mode": "live-future-evaluation",
        "estimated_samples_per_run": int(phase_candles),
    }


def generate_live_model_comparison_report(
    *,
    fetch_loops: int = DEFAULT_FETCH_LOOPS,
    phase_candles: int = DEFAULT_LIVE_PHASE_CANDLES,
    progress_callback=None,
    should_continue=None,
):
    bootstrap_df = bootstrap_live_history(fetch_loops=fetch_loops)
    lstm_model = load_lstm_model()
    try:
        sarima_profile = load_profile(PROFILE_PATH)
    except Exception:
        sarima_profile = None

    rows = []
    phases = network_phases()
    total_runs = len(phases) * len(MODEL_FAMILIES)
    _, last_seq = get_live_closed_candles(after_seq=0, limit=0)

    for phase in phases:
        if should_continue is not None and not should_continue():
            break
        phase_bootstrap_df = build_live_phase_history(bootstrap_df, last_seq)
        runtimes = [
            new_live_runtime(family["key"], phase, phase_bootstrap_df, sarima_profile)
            for family in MODEL_FAMILIES
        ]
        completed_samples = 0

        while completed_samples < phase_candles:
            if should_continue is not None and not should_continue():
                break
            fresh_rows, latest_seq = get_live_closed_candles(after_seq=last_seq, limit=20)
            if not fresh_rows:
                time.sleep(1.0)
                continue
            for item in fresh_rows:
                last_seq = int(item["seq"])
                actual_candle = item["row"]
                for runtime in runtimes:
                    simulate_live_step(runtime, actual_candle, lstm_model)
                completed_samples += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "currentPhase": phase["name"],
                            "currentModel": "All models",
                            "completedRuns": len(rows),
                            "totalRuns": total_runs,
                            "rows": deepcopy(rows),
                            "currentSamples": completed_samples,
                            "estimatedSamples": phase_candles,
                            "report": build_live_report(rows, phase_candles, total_runs, completed=False),
                        }
                    )
                if completed_samples >= phase_candles:
                    break
            if should_continue is not None and not should_continue():
                break

        if completed_samples <= 0:
            break

        phase_rows = [finalize_live_runtime(runtime) for runtime in runtimes]
        baseline_lstm = next((row for row in phase_rows if row["modelKey"] == "lstm_baseline"), None)
        baseline_pnl = baseline_lstm["pnl_delta"] if baseline_lstm else 0.0
        for row in phase_rows:
            row["pnl_impact_vs_lstm"] = row["pnl_delta"] - baseline_pnl
            rows.append(row)

        if progress_callback is not None:
            progress_callback(
                {
                    "currentPhase": phase["name"],
                    "currentModel": "All models",
                    "completedRuns": len(rows),
                    "totalRuns": total_runs,
                    "rows": deepcopy(rows),
                    "currentSamples": completed_samples,
                    "estimatedSamples": phase_candles,
                    "report": build_live_report(rows, phase_candles, total_runs, completed=False),
                }
            )

    completed = len(rows) == total_runs
    return build_live_report(rows, phase_candles, total_runs, completed)
