"""
data_ingestion.py – Core real-time data pipeline and paper-trading engine.

This is the largest backend module.  It is responsible for:

  1. **Binance WebSocket ingestion** – Connects to the live BTC/USDT
     1-minute kline stream and maintains a rolling DataFrame of candles.

  2. **SARIMA gap filling** – When candles are missing (due to network
     issues or simulated packet loss), the SARIMA pre-processor from
     sarima_preprocessor.py reconstructs them using a Kaggle-calibrated
     profile.

  3. **Feature engineering** – Passes each new candle through
     PatternRecognizer.extract_features() to compute FVGs, HH/LL,
     confluence scores, and data quality metrics.

  4. **LSTM inference** – Runs the AdaptiveTradingLSTM on the latest
     feature window and produces BUY / SELL / HOLD predictions.

  5. **Paper-trading simulation** – Manages a virtual portfolio with
     position sizing, fee handling, stop-loss / take-profit levels,
     and PnL tracking.  No real money is involved.

  6. **Live training loop** – Optionally retrains the LSTM periodically
     using fresh market data and trade feedback (controlled by env vars).

  7. **State broadcasting** – Packages all state into a JSON snapshot
     that the React dashboard polls via WebSocket.

Data sources
------------
  • Live candles     : Binance WebSocket  (wss://stream.binance.com)
  • Historical data  : Binance REST API   (BTCUSDT 1m klines)
  • SARIMA profile   : Calibrated on Kaggle BTC minute data
      https://www.kaggle.com/datasets/swaptr/bitcoin-historical-data
  • Kaggle daily BTC :
      https://www.kaggle.com/datasets/hasanyiitakbulut/bitcoin-btc-historical-price-data-2020-2026
"""

import asyncio
import json
import logging
import os
import random
import threading
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import numpy as np
import torch
import websockets

from ai_core import AdaptiveTradingLSTM, FEATURE_COLUMNS, predict_action
from feature_engineering import PatternRecognizer
from sarima_preprocessor import PROFILE_PATH, load_profile, reconstruct_missing_rows
from train import fetch_historical_data, train_adaptive_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataIngestion")

MARKET_LOG_TABLE_BORDER = "+------------------+-------------+--------+------------+"
MARKET_LOG_TABLE_HEADER = "| Time             | Close       | Action | Confidence |"
_market_log_table_started = False


def log_market_table_row(timestamp, close_price, action, confidence):
    global _market_log_table_started

    if not _market_log_table_started:
        logger.info(
            "\n%s\n%s\n%s",
            MARKET_LOG_TABLE_BORDER,
            MARKET_LOG_TABLE_HEADER,
            MARKET_LOG_TABLE_BORDER,
        )
        _market_log_table_started = True

    logger.info(
        "| %-16s | $%10.2f | %-6s | %10.2f |",
        timestamp,
        close_price,
        action,
        confidence,
    )


MAX_CANDLES = 400
CANDLE_INTERVAL_MS = 60_000
MAX_SYNTHETIC_GAP_FILL = 30
DEFAULT_INITIAL_EQUITY = float(os.getenv("SIM_INITIAL_EQUITY", "100000"))
DEFAULT_MIN_ALLOCATION_PCT = float(os.getenv("SIM_MIN_ALLOCATION_PCT", "2.0"))
DEFAULT_MAX_ALLOCATION_PCT = float(os.getenv("SIM_MAX_ALLOCATION_PCT", "12.0"))
DEFAULT_MIN_CONFIDENCE_PCT = float(os.getenv("SIM_MIN_CONFIDENCE_PCT", "45.0"))
DEFAULT_FEE_RATE_PCT = float(os.getenv("SIM_FEE_RATE_PCT", "0.10"))
DEFAULT_MIN_TRADE_NOTIONAL = float(os.getenv("SIM_MIN_TRADE_NOTIONAL", "1000"))
DEFAULT_SLIPPAGE_BUFFER_PCT = float(os.getenv("SIM_SLIPPAGE_BUFFER_PCT", "0.03"))
DEFAULT_EDGE_SAFETY_MULTIPLIER = float(os.getenv("SIM_EDGE_SAFETY_MULTIPLIER", "1.05"))
LIVE_TRAINING_ENABLED = os.getenv("LIVE_TRAINING_ENABLED", "0") != "0"
LIVE_TRAINING_INTERVAL_SEC = int(os.getenv("LIVE_TRAINING_INTERVAL_SEC", "300"))
LIVE_TRAINING_WARMUP_SEC = int(os.getenv("LIVE_TRAINING_WARMUP_SEC", "30"))
LIVE_TRAINING_MIN_FEEDBACK = int(os.getenv("LIVE_TRAINING_MIN_FEEDBACK", "5"))
LIVE_TRAINING_LOOPS = int(os.getenv("LIVE_TRAINING_LOOPS", "4"))
LIVE_TRAINING_EPOCHS = int(os.getenv("LIVE_TRAINING_EPOCHS", "20"))
LIVE_TRAINING_BATCH_SIZE = int(os.getenv("LIVE_TRAINING_BATCH_SIZE", "128"))
LIVE_TRAINING_FORECAST_HORIZON = int(os.getenv("LIVE_TRAINING_FORECAST_HORIZON", "3"))
LIVE_TRAINING_MIN_IMPROVEMENT = float(os.getenv("LIVE_TRAINING_MIN_IMPROVEMENT", "0.01"))
MODEL_EVAL_WINDOW_CANDLES = int(os.getenv("MODEL_EVAL_WINDOW_CANDLES", "180"))
MODEL_EVAL_MIN_TRADES = int(os.getenv("MODEL_EVAL_MIN_TRADES", "8"))
MODEL_EVAL_MIN_SCORE_MARGIN = float(os.getenv("MODEL_EVAL_MIN_SCORE_MARGIN", "1.0"))
SHADOW_EVAL_MIN_CANDLES = int(os.getenv("SHADOW_EVAL_MIN_CANDLES", "120"))
SHADOW_EVAL_MIN_TRADES = int(os.getenv("SHADOW_EVAL_MIN_TRADES", "6"))
SHADOW_EVAL_MIN_SCORE_MARGIN = float(os.getenv("SHADOW_EVAL_MIN_SCORE_MARGIN", "1.0"))
TRAINING_REPORT_PATH = Path("training_report.json")
TRADE_FEEDBACK_PATH = Path("trade_feedback.jsonl")
MODEL_HISTORY_PATH = Path("model_history.jsonl")
LIVE_CANDIDATE_MODEL_PATH = Path("adaptive_weights.live_candidate.pth")
LIVE_CANDIDATE_BEST_PATH = Path("adaptive_weights.live_candidate.best.pth")
LIVE_CANDIDATE_REPORT_PATH = Path("training_report.live_candidate.json")
MODEL_CHECKPOINT_DIR = Path("model_checkpoints")
MAX_MODEL_TRACE_POINTS = int(os.getenv("MAX_MODEL_TRACE_POINTS", "240"))

simulation_config = {
    "initialEquity": DEFAULT_INITIAL_EQUITY,
    "minAllocationPct": DEFAULT_MIN_ALLOCATION_PCT,
    "maxAllocationPct": DEFAULT_MAX_ALLOCATION_PCT,
    "minConfidencePct": DEFAULT_MIN_CONFIDENCE_PCT,
    "feeRatePct": DEFAULT_FEE_RATE_PCT,
    "ignoreFees": False,
    "minTradeNotional": DEFAULT_MIN_TRADE_NOTIONAL,
    "dynamicThresholds": True,
    "allowLong": True,
    "allowShort": True,
}
network_test_config = {
    "enabled": False,
    "latencyMs": 0.0,
    "jitterMs": 0.0,
    "packetLossPct": 0.0,
}
network_test_stats = {
    "receivedClosedCandles": 0,
    "processedClosedCandles": 0,
    "droppedClosedCandles": 0,
    "lastAppliedDelayMs": 0.0,
    "syntheticFilledCandles": 0,
    "consecutiveSyntheticCandles": 0,
    "maxConsecutiveSyntheticCandles": 0,
    "lastGapMinutes": 0.0,
}
sarima_profile = None
dynamic_threshold_state = {
    "enabled": True,
    "minConfidencePct": DEFAULT_MIN_CONFIDENCE_PCT,
    "minTradeNotional": DEFAULT_MIN_TRADE_NOTIONAL,
}


def effective_fee_rate_pct():
    return 0.0 if simulation_config.get("ignoreFees") else simulation_config["feeRatePct"]


def simulation_summary():
    round_trip_cost_pct = effective_fee_rate_pct() * 2
    fee_aware_trade_gate_pct = (
        0.0
        if simulation_config.get("ignoreFees")
        else round_trip_cost_pct * DEFAULT_EDGE_SAFETY_MULTIPLIER + DEFAULT_SLIPPAGE_BUFFER_PCT
    )
    return {
        "budget": simulation_config["initialEquity"],
        "aiPolicy": {
            "allocationRangePct": [
                simulation_config["minAllocationPct"],
                simulation_config["maxAllocationPct"],
            ],
            "minConfidencePct": simulation_config["minConfidencePct"],
            "dynamicMinConfidencePct": round(dynamic_threshold_state["minConfidencePct"], 2),
            "feeRatePct": round(effective_fee_rate_pct(), 4),
            "configuredFeeRatePct": round(simulation_config["feeRatePct"], 4),
            "ignoreFees": bool(simulation_config.get("ignoreFees")),
            "roundTripCostPct": round_trip_cost_pct,
            "feeAwareTradeGatePct": round(fee_aware_trade_gate_pct, 4),
            "minTradeNotional": simulation_config["minTradeNotional"],
            "dynamicMinTradeNotional": round(dynamic_threshold_state["minTradeNotional"], 2),
            "dynamicThresholds": bool(simulation_config.get("dynamicThresholds", True)),
            "allowLong": simulation_config["allowLong"],
            "allowShort": simulation_config["allowShort"],
        },
    }

df = pd.DataFrame(
    columns=[
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Timestamp",
        "Is_Imputed",
        "Gap_Minutes",
        "Receive_Lag_Ms",
    ]
)
pattern_engine = PatternRecognizer()
ai_model = AdaptiveTradingLSTM(input_size=len(FEATURE_COLUMNS))
ai_model_lock = threading.RLock()
update_timestamps = deque(maxlen=240)
pending_signal = None
shadow_models = {
    "champion": None,
    "candidate": None,
}
shadow_model_pool = {}
live_training_status = {
    "enabled": LIVE_TRAINING_ENABLED,
    "running": False,
    "intervalSec": LIVE_TRAINING_INTERVAL_SEC,
    "warmupSec": LIVE_TRAINING_WARMUP_SEC,
    "lastStartedAt": None,
    "lastCompletedAt": None,
    "lastPromotedAt": None,
    "nextRunAt": None,
    "activeModelPath": None,
    "activeModelVersion": None,
    "championModelLabel": None,
    "championModelPath": None,
    "championScore": None,
    "championMetrics": None,
    "promotionCount": 0,
    "tradeFeedbackCount": 0,
    "lastCandidate": None,
    "lastPromotionDecision": "pending",
    "lastError": None,
    "shadowEvaluation": None,
}

paper_state = {
    "cash": DEFAULT_INITIAL_EQUITY,
    "reserved_margin": 0.0,
    "position_side": "FLAT",
    "position_size": 0.0,
    "position_notional": 0.0,
    "allocation_pct": 0.0,
    "entry_price": None,
    "entry_confidence": None,
    "realized_pnl": 0.0,
    "unrealized_pnl": 0.0,
    "fees_paid": 0.0,
    "equity": DEFAULT_INITIAL_EQUITY,
    "trade_count": 0,
    "wins": 0,
    "losses": 0,
    "deployed_notional_total": 0.0,
}

latest_state = {
    "history": [],
    "candle": None,
    "patterns": {
        "Bullish_FVG": False,
        "Bearish_FVG": False,
        "HH": False,
        "LL": False,
        "HL": False,
        "LH": False,
    },
    "prediction": {
        "action": "HOLD",
        "confidence": 0.0,
        "rawConfidence": 0.0,
        "targetPrice": None,
        "projectedMovePct": 0.0,
    },
    "portfolio": {
        "initialEquity": DEFAULT_INITIAL_EQUITY,
        "equity": DEFAULT_INITIAL_EQUITY,
        "cash": DEFAULT_INITIAL_EQUITY,
        "availableCash": DEFAULT_INITIAL_EQUITY,
        "reservedMargin": 0.0,
        "realizedPnl": 0.0,
        "unrealizedPnl": 0.0,
        "totalPnl": 0.0,
        "feesPaid": 0.0,
        "position": "FLAT",
        "positionSize": 0.0,
        "positionNotional": 0.0,
        "deployedCapital": 0.0,
        "allocationPct": 0.0,
        "entryPrice": None,
        "entryConfidencePct": None,
        "tradeCount": 0,
        "winRate": 0.0,
        "signalAccuracy": 0.0,
    },
    "telemetry": {
        "tickTime": None,
        "receivedTime": None,
        "signalTime": None,
        "deltaMs": 0,
        "updatesPerMinute": 0.0,
        "packetLossPct": 0.0,
        "latencyMs": 0.0,
        "simulatedDelayMs": 0.0,
        "processedCandles": 0,
        "droppedCandles": 0,
        "syntheticFilledCandles": 0,
        "imputedRatePct": 0.0,
        "maxConsecutiveImputed": 0,
        "lastGapMinutes": 0.0,
        "volatilityPct": 0.0,
        "rangePct": 0.0,
        "trendBiasPct": 0.0,
        "confluenceScore": 0.0,
        "dataQuality": 1.0,
    },
    "simulation": simulation_config.copy(),
    "networkTest": network_test_config.copy(),
    "training": {},
    "liveTraining": deepcopy(live_training_status),
    "modelHistory": [],
    "modelTraces": [],
    "activity": [],
    "blotter": [],
    "logs": [],
}

signal_scoreboard = {
    "resolved": 0,
    "correct": 0,
}
trade_blotter = deque(maxlen=100)
trade_sequence = 0
model_performance_store = {}
live_closed_candle_lock = threading.RLock()
live_closed_candles = deque(maxlen=5000)
live_closed_candle_seq = 0


def empty_shadow_state():
    return {
        "candidateLabel": None,
        "candidatePath": None,
        "startedAt": None,
        "startedCandleTime": None,
        "candlesObserved": 0,
        "minCandles": SHADOW_EVAL_MIN_CANDLES,
        "minTrades": SHADOW_EVAL_MIN_TRADES,
        "status": "idle",
        "decision": "no shadow evaluation running",
        "candidateReplay": None,
        "championReplay": None,
        "candidateMetrics": None,
        "championMetrics": None,
        "activeReport": None,
        "candidateReport": None,
    }


def publish_live_closed_candle(
    *,
    open_ms,
    open_price,
    high_price,
    low_price,
    close_price,
    volume,
    receive_timestamp_ms,
):
    global live_closed_candle_seq
    payload = {
        "open_ms": int(open_ms),
        "Timestamp": pd.to_datetime(open_ms, unit="ms"),
        "Open": float(open_price),
        "High": float(high_price),
        "Low": float(low_price),
        "Close": float(close_price),
        "Volume": float(volume),
        "receive_timestamp_ms": int(receive_timestamp_ms),
    }
    with live_closed_candle_lock:
        live_closed_candle_seq += 1
        live_closed_candles.append({"seq": int(live_closed_candle_seq), "row": payload})


def get_live_closed_candles(after_seq=0, limit=5000):
    with live_closed_candle_lock:
        rows = [item for item in live_closed_candles if int(item["seq"]) > int(after_seq)]
        if limit is not None:
            rows = rows[: int(limit)]
        latest_seq = int(live_closed_candle_seq)
    return rows, latest_seq


def new_virtual_account_state():
    return {
        "cash": simulation_config["initialEquity"],
        "equity": simulation_config["initialEquity"],
        "reserved_margin": 0.0,
        "position_side": "FLAT",
        "position_size": 0.0,
        "position_notional": 0.0,
        "allocation_pct": 0.0,
        "entry_price": None,
        "entry_confidence": None,
        "entry_fee": 0.0,
        "stop_loss": None,
        "take_profit": None,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "fees_paid": 0.0,
        "trade_count": 0,
        "wins": 0,
        "losses": 0,
        "resolved": 0,
        "correct": 0,
        "pending_signal": None,
        "peak_equity": simulation_config["initialEquity"],
        "max_drawdown_pct": 0.0,
        "lastAction": "HOLD",
        "lastConfidence": 0.0,
    }


def reset_runtime_state():
    global pending_signal, update_timestamps, trade_sequence
    pending_signal = None
    update_timestamps.clear()
    trade_sequence = 0
    initial_equity = simulation_config["initialEquity"]

    paper_state.update(
        {
            "cash": initial_equity,
            "reserved_margin": 0.0,
            "position_side": "FLAT",
            "position_size": 0.0,
            "position_notional": 0.0,
            "allocation_pct": 0.0,
            "entry_price": None,
            "entry_confidence": None,
            "entry_fee": 0.0,
            "opened_at": None,
            "opened_candle_time": None,
            "entry_action_label": None,
            "stop_loss": None,
            "take_profit": None,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "fees_paid": 0.0,
            "equity": initial_equity,
            "trade_count": 0,
            "wins": 0,
            "losses": 0,
            "deployed_notional_total": 0.0,
        }
    )
    signal_scoreboard["resolved"] = 0
    signal_scoreboard["correct"] = 0
    network_test_stats["receivedClosedCandles"] = 0
    network_test_stats["processedClosedCandles"] = 0
    network_test_stats["droppedClosedCandles"] = 0
    network_test_stats["lastAppliedDelayMs"] = 0.0
    network_test_stats["syntheticFilledCandles"] = 0
    network_test_stats["consecutiveSyntheticCandles"] = 0
    network_test_stats["maxConsecutiveSyntheticCandles"] = 0
    network_test_stats["lastGapMinutes"] = 0.0
    latest_state["logs"] = []
    latest_state["activity"] = []
    trade_blotter.clear()
    latest_state["simulation"] = simulation_config.copy()
    dynamic_threshold_state["enabled"] = bool(simulation_config.get("dynamicThresholds", True))
    dynamic_threshold_state["minConfidencePct"] = simulation_config["minConfidencePct"]
    dynamic_threshold_state["minTradeNotional"] = simulation_config["minTradeNotional"]
    latest_state["networkTest"] = network_test_config.copy()
    latest_state["simulationSummary"] = simulation_summary()
    sync_live_training_state()


def reset_shadow_evaluation(reason="no shadow evaluation running"):
    live_training_status["shadowEvaluation"] = empty_shadow_state()
    live_training_status["shadowEvaluation"]["decision"] = reason
    shadow_models["champion"] = None
    shadow_models["candidate"] = None


def model_trace_snapshot():
    traces = []
    for label, trace in model_performance_store.items():
        points = trace.get("points", [])
        traces.append(
            {
                "modelLabel": label,
                "status": trace.get("status", "archived"),
                "role": trace.get("role", "historical"),
                "lastUpdated": trace.get("lastUpdated"),
                "points": points[-MAX_MODEL_TRACE_POINTS:],
                "latestPnl": points[-1]["pnl"] if points else 0.0,
                "tradeCount": trace.get("tradeCount", 0),
            }
        )
    traces.sort(key=lambda item: item.get("lastUpdated") or item.get("modelLabel"))
    return traces


def ensure_model_trace(model_label, role="historical", status="archived"):
    if not model_label:
        return None
    trace = model_performance_store.setdefault(
        model_label,
        {
            "modelLabel": model_label,
            "role": role,
            "status": status,
            "lastUpdated": None,
            "tradeCount": 0,
            "points": [],
        },
    )
    trace["role"] = role
    trace["status"] = status
    return trace


def append_model_trace_point(model_label, timestamp_ms, pnl_value, trade_count=0, role="historical", status="archived"):
    trace = ensure_model_trace(model_label, role=role, status=status)
    if trace is None:
        return
    timestamp_ms = int(timestamp_ms)
    point = {
        "time": timestamp_ms,
        "pnl": round(float(pnl_value), 2),
    }
    if trace["points"] and trace["points"][-1]["time"] == timestamp_ms:
        trace["points"][-1] = point
    else:
        trace["points"].append(point)
        if len(trace["points"]) > MAX_MODEL_TRACE_POINTS:
            trace["points"] = trace["points"][-MAX_MODEL_TRACE_POINTS:]
    trace["tradeCount"] = int(trade_count)
    trace["lastUpdated"] = timestamp_ms


def checkpoint_path_for_label(model_label):
    return MODEL_CHECKPOINT_DIR / f"{model_label}.pth"


def snapshot_model_checkpoint(model_label, source_path):
    if not model_label or not source_path:
        return None
    source = Path(source_path)
    if not source.exists():
        return None
    MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    target = checkpoint_path_for_label(model_label)
    if not target.exists():
        target.write_bytes(source.read_bytes())
    return str(target.resolve())


def rebuild_model_traces_from_history():
    model_performance_store.clear()
    for item in load_model_history():
        label = item.get("modelLabel")
        snapshot = item.get("portfolioAtDecision", {})
        timestamp_raw = item.get("timestamp")
        try:
            timestamp_ms = int(pd.Timestamp(timestamp_raw).timestamp() * 1000)
        except Exception:
            timestamp_ms = int(time.time() * 1000)
        pnl_value = snapshot.get("totalPnl", 0.0)
        trace = ensure_model_trace(label, role="historical", status="promoted" if item.get("promoted") else "rejected")
        append_model_trace_point(
            label,
            timestamp_ms,
            pnl_value,
            trade_count=snapshot.get("tradeCount", 0),
            role=trace["role"],
            status=trace["status"],
        )


def rebuild_shadow_model_pool():
    shadow_model_pool.clear()
    history = load_model_history()
    labels_loaded = set()
    for item in history:
        label = item.get("modelLabel")
        if not label or label in labels_loaded:
            continue
        candidate = item.get("candidate", {})
        source_path = candidate.get("checkpointPath")
        if not source_path and item.get("promoted"):
            source_path = item.get("activeAfter", {}).get("path") or candidate.get("path")
        checkpoint = checkpoint_path_for_label(label)
        if not checkpoint.exists() and source_path:
            snapshot_model_checkpoint(label, source_path)
        if not checkpoint.exists():
            continue
        if live_training_status.get("championModelLabel") == label:
            continue
        try:
            shadow_model_pool[label] = {
                "model": build_model_from_path(checkpoint),
                "state": new_virtual_account_state(),
                "path": str(checkpoint.resolve()),
                "status": "shadow-active",
                "role": "historical",
            }
            ensure_model_trace(label, role="historical", status="shadow-active")
            labels_loaded.add(label)
        except Exception:
            continue


def virtual_refresh_account(state, last_price):
    unrealized = 0.0
    if state["entry_price"] and state["position_size"]:
        if state["position_side"] == "LONG":
            unrealized = (last_price - state["entry_price"]) * state["position_size"]
        elif state["position_side"] == "SHORT":
            unrealized = (state["entry_price"] - last_price) * state["position_size"]

    state["unrealized_pnl"] = unrealized
    state["equity"] = state["cash"] + unrealized
    state["peak_equity"] = max(state["peak_equity"], state["equity"])
    if state["peak_equity"] > 0:
        drawdown_pct = ((state["peak_equity"] - state["equity"]) / state["peak_equity"]) * 100.0
        state["max_drawdown_pct"] = max(state["max_drawdown_pct"], drawdown_pct)


def virtual_summary(state):
    win_rate = state["wins"] / max(1, state["wins"] + state["losses"])
    accuracy = state["correct"] / max(1, state["resolved"])
    score = (
        state["realized_pnl"]
        + (accuracy * 120.0)
        + (win_rate * 80.0)
        - (state["max_drawdown_pct"] * 8.0)
        + min(state["trade_count"], 20) * 1.5
    )
    return {
        "score": round(score, 4),
        "netPnl": round(state["realized_pnl"], 2),
        "tradeCount": int(state["trade_count"]),
        "winRate": round(win_rate, 4),
        "signalAccuracy": round(accuracy, 4),
        "maxDrawdownPct": round(state["max_drawdown_pct"], 4),
        "equity": round(state["equity"], 2),
        "feesPaid": round(state["fees_paid"], 2),
        "position": state["position_side"],
        "candlesObserved": int(state.get("candles_observed", 0)),
        "lastAction": state.get("lastAction", "HOLD"),
        "lastConfidence": round(float(state.get("lastConfidence", 0.0)), 4),
    }


def projected_move_and_target(close_price, action, confidence, confluence_score):
    directional_bias = 0.0
    if action == "BUY":
        directional_bias = max(confidence, 0.2)
    elif action == "SELL":
        directional_bias = -max(confidence, 0.2)

    projected_move_pct = directional_bias * (0.002 + abs(confluence_score) * 0.0025)
    target_price = close_price * (1.0 + projected_move_pct)
    return projected_move_pct, target_price


def compute_dynamic_entry_thresholds(volatility_pct, confluence_score, trend_bias_pct, equity_base):
    base_min_conf = float(simulation_config["minConfidencePct"])
    base_min_notional = float(simulation_config["minTradeNotional"])

    if not simulation_config.get("dynamicThresholds", True):
        dynamic_threshold_state["enabled"] = False
        dynamic_threshold_state["minConfidencePct"] = base_min_conf
        dynamic_threshold_state["minTradeNotional"] = base_min_notional
        return base_min_conf, base_min_notional

    equity_base = max(1.0, float(equity_base))
    vol_norm = min(1.0, max(0.0, volatility_pct / 0.25))
    confluence_norm = min(1.0, abs(confluence_score))
    trend_norm = min(1.0, abs(trend_bias_pct) / 0.5)
    signal_strength = min(1.0, (confluence_norm + trend_norm) / 2.0)
    budget_scale = min(1.35, max(0.75, (equity_base / 10000.0) ** 0.18))

    dynamic_min_conf = (base_min_conf + (vol_norm * 6.0) - (signal_strength * 10.0)) * (
        0.9 + (budget_scale - 1.0) * 0.6
    )
    dynamic_min_conf = max(30.0, min(85.0, dynamic_min_conf))

    notional_multiplier = 0.7 + (signal_strength * 0.7) + ((1.0 - vol_norm) * 0.3)
    dynamic_min_notional = base_min_notional * notional_multiplier * budget_scale
    cash_floor = max(25.0, equity_base * 0.008)
    cash_ceiling = max(cash_floor, equity_base * 0.2)
    dynamic_min_notional = max(cash_floor, min(dynamic_min_notional, cash_ceiling))

    dynamic_threshold_state["enabled"] = True
    dynamic_threshold_state["minConfidencePct"] = dynamic_min_conf
    dynamic_threshold_state["minTradeNotional"] = dynamic_min_notional
    return dynamic_min_conf, dynamic_min_notional


def calculate_position_size_for_state(state, price, confidence, confluence_score, min_confidence_pct, min_trade_notional):
    available_cash = max(0.0, state["cash"] - state["reserved_margin"])
    equity_base = max(0.0, state["equity"])
    min_confidence = min_confidence_pct / 100.0

    if confidence < min_confidence or available_cash <= 0:
        return 0.0, 0.0, 0.0

    confidence_span = max(1e-9, 1.0 - min_confidence)
    confidence_factor = min(1.0, max(0.0, (confidence - min_confidence) / confidence_span))
    confluence_factor = 0.8 + min(1.0, abs(confluence_score)) * 0.4
    allocation_pct = simulation_config["minAllocationPct"] + (
        simulation_config["maxAllocationPct"] - simulation_config["minAllocationPct"]
    ) * confidence_factor * confluence_factor
    allocation_pct = max(simulation_config["minAllocationPct"], min(simulation_config["maxAllocationPct"], allocation_pct))
    target_notional = min(available_cash, equity_base * allocation_pct / 100.0)

    if target_notional < min_trade_notional:
        return 0.0, 0.0, 0.0

    size = target_notional / max(price, 1e-9)
    return round(size, 6), round(target_notional, 2), round(allocation_pct, 4)


def close_virtual_position(state, price):
    if state["position_side"] == "FLAT" or not state["position_size"] or state["entry_price"] is None:
        return

    if state["position_side"] == "LONG":
        gross_pnl = (price - state["entry_price"]) * state["position_size"]
    else:
        gross_pnl = (state["entry_price"] - price) * state["position_size"]

    fee = price * state["position_size"] * (effective_fee_rate_pct() / 100.0)
    pnl = gross_pnl - fee
    state["cash"] += pnl
    state["realized_pnl"] += pnl
    state["fees_paid"] += fee
    state["trade_count"] += 1
    if pnl >= 0:
        state["wins"] += 1
    else:
        state["losses"] += 1

    state["position_side"] = "FLAT"
    state["position_size"] = 0.0
    state["position_notional"] = 0.0
    state["allocation_pct"] = 0.0
    state["reserved_margin"] = 0.0
    state["entry_price"] = None
    state["entry_confidence"] = None
    state["entry_fee"] = 0.0
    state["stop_loss"] = None
    state["take_profit"] = None


def open_virtual_position(
    state,
    action,
    price,
    confidence,
    confluence_score,
    projected_move_pct,
    volatility_pct,
    trend_bias_pct,
):
    if action not in {"BUY", "SELL"}:
        return

    desired_side = "LONG" if action == "BUY" else "SHORT"
    if action == "BUY" and not simulation_config["allowLong"]:
        return
    if action == "SELL" and not simulation_config["allowShort"]:
        return
    if state["position_side"] == desired_side:
        return
    if state["position_side"] != "FLAT":
        close_virtual_position(state, price)

    expected_move_pct, required_edge_pct = expected_edge_pct(projected_move_pct)
    if expected_move_pct < required_edge_pct:
        return

    dynamic_min_conf, dynamic_min_notional = compute_dynamic_entry_thresholds(
        volatility_pct,
        confluence_score,
        trend_bias_pct,
        state["equity"],
    )
    size, target_notional, allocation_pct = calculate_position_size_for_state(
        state,
        price,
        confidence,
        confluence_score,
        dynamic_min_conf,
        dynamic_min_notional,
    )
    if size <= 0 or target_notional <= 0:
        return

    fee = target_notional * (effective_fee_rate_pct() / 100.0)
    state["cash"] -= fee
    state["realized_pnl"] -= fee
    state["fees_paid"] += fee
    state["reserved_margin"] = target_notional
    state["position_side"] = desired_side
    state["position_size"] = size
    state["position_notional"] = target_notional
    state["allocation_pct"] = allocation_pct
    state["entry_price"] = price
    state["entry_confidence"] = confidence
    state["entry_fee"] = fee
    state["stop_loss"], state["take_profit"] = compute_exit_levels(
        price, action, confidence, volatility_pct, projected_move_pct
    )


def maybe_close_virtual_on_risk_levels(state, latest_features):
    if state["position_side"] == "FLAT":
        return

    high_price = float(latest_features["High"])
    low_price = float(latest_features["Low"])

    if state["position_side"] == "LONG":
        if low_price <= (state["stop_loss"] or 0):
            close_virtual_position(state, float(state["stop_loss"]))
            return
        if high_price >= (state["take_profit"] or float("inf")):
            close_virtual_position(state, float(state["take_profit"]))
            return
    else:
        if high_price >= (state["stop_loss"] or float("inf")):
            close_virtual_position(state, float(state["stop_loss"]))
            return
        if low_price <= (state["take_profit"] or 0):
            close_virtual_position(state, float(state["take_profit"]))
            return


def resolve_virtual_signal(state, latest_close):
    pending = state.get("pending_signal")
    if pending is None:
        return

    action = pending["action"]
    previous_close = pending["close"]
    if action == "BUY":
        correct = latest_close > previous_close
    elif action == "SELL":
        correct = latest_close < previous_close
    else:
        correct = abs(latest_close - previous_close) / max(previous_close, 1e-9) < 0.001

    state["resolved"] += 1
    state["correct"] += int(correct)
    state["pending_signal"] = None


def predict_with_model(model, df_features):
    return predict_action(model, df_features)


def step_virtual_model(state, model, df_features):
    latest_features = df_features.iloc[-1]
    close_price = float(latest_features["Close"])
    confluence_score = float(latest_features.get("Confluence_Score", 0.0))
    volatility_pct = float(latest_features.get("Volatility_20", 0.0) * 100.0)
    trend_bias_pct = float(latest_features.get("Trend_Bias", 0.0) * 100.0)

    maybe_close_virtual_on_risk_levels(state, latest_features)
    resolve_virtual_signal(state, close_price)

    action, confidence = predict_with_model(model, df_features)
    projected_move, _ = projected_move_and_target(close_price, action, confidence, confluence_score)

    if action in {"BUY", "SELL"}:
        open_virtual_position(
            state,
            action,
            close_price,
            confidence,
            confluence_score,
            projected_move * 100.0,
            volatility_pct,
            trend_bias_pct,
        )
    elif action == "HOLD" and state["position_side"] != "FLAT":
        close_virtual_position(state, close_price)

    state["pending_signal"] = {
        "action": action,
        "close": close_price,
    }
    state["lastAction"] = action
    state["lastConfidence"] = confidence
    state["candles_observed"] = int(state.get("candles_observed", 0)) + 1
    virtual_refresh_account(state, close_price)


def load_training_report():
    if not TRAINING_REPORT_PATH.exists():
        return {}

    try:
        return json.loads(TRAINING_REPORT_PATH.read_text())
    except Exception:
        return {}


def count_trade_feedback():
    if not TRADE_FEEDBACK_PATH.exists():
        return 0

    try:
        with TRADE_FEEDBACK_PATH.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception:
        return 0


def sync_live_training_state():
    live_training_status["tradeFeedbackCount"] = count_trade_feedback()
    latest_state["liveTraining"] = deepcopy(live_training_status)
    latest_state["modelHistory"] = load_model_history(limit=40)
    latest_state["modelTraces"] = model_trace_snapshot()


def append_trade_feedback(entry):
    with TRADE_FEEDBACK_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    live_training_status["tradeFeedbackCount"] = count_trade_feedback()


def load_model_history(limit=None):
    if not MODEL_HISTORY_PATH.exists():
        return []

    items = []
    with MODEL_HISTORY_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    items.sort(key=lambda item: item.get("sequence", 0))
    if limit is not None:
        return items[-limit:]
    return items


def best_champion_event():
    promoted = [item for item in load_model_history() if item.get("promoted")]
    if not promoted:
        return None
    scored = [
        item
        for item in promoted
        if (
            item.get("candidate", {}).get("shadowEvaluation", {}).get("score") is not None
            or item.get("candidate", {}).get("evaluation", {}).get("score") is not None
        )
    ]
    if scored:
        return max(
            scored,
            key=lambda item: float(
                item.get("candidate", {}).get("shadowEvaluation", {}).get("score")
                or item.get("candidate", {}).get("evaluation", {}).get("score", float("-inf"))
            ),
        )
    return promoted[-1]


def next_model_sequence():
    history = load_model_history(limit=1)
    return int(history[-1]["sequence"]) + 1 if history else 1


def append_model_history(entry):
    with MODEL_HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def current_portfolio_snapshot():
    portfolio = latest_state.get("portfolio", {})
    return {
        "equity": float(portfolio.get("equity", 0.0)),
        "totalPnl": float(portfolio.get("totalPnl", 0.0)),
        "realizedPnl": float(portfolio.get("realizedPnl", 0.0)),
        "unrealizedPnl": float(portfolio.get("unrealizedPnl", 0.0)),
        "feesPaid": float(portfolio.get("feesPaid", 0.0)),
        "tradeCount": int(portfolio.get("tradeCount", 0)),
        "winRate": float(portfolio.get("winRate", 0.0)),
        "signalAccuracy": float(portfolio.get("signalAccuracy", 0.0)),
    }


def record_model_event(candidate_report, promoted, decision_reason, active_report_before, active_version_before):
    sequence = next_model_sequence()
    label = f"M{sequence:03d}"
    checkpoint_path = snapshot_model_checkpoint(
        label,
        candidate_report.get("candidateModelPath") or candidate_report.get("best_model_path"),
    )
    event = {
        "sequence": sequence,
        "modelLabel": label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "promoted": bool(promoted),
        "decision": decision_reason,
        "candidate": {
            "valAccuracy": float(candidate_report.get("val_accuracy", 0.0)),
            "bestValLoss": float(candidate_report.get("best_val_loss", float("inf"))),
            "valLoss": float(candidate_report.get("val_loss", float("inf"))),
            "feedbackMatches": int(candidate_report.get("feedback_matches", 0)),
            "path": candidate_report.get("candidateModelPath") or candidate_report.get("best_model_path"),
            "checkpointPath": checkpoint_path,
            "evaluation": candidate_report.get("evaluation", {}),
            "shadowEvaluation": candidate_report.get("shadowEvaluation", {}),
        },
        "activeBefore": {
            "version": active_version_before,
            "bestValLoss": float((active_report_before or {}).get("best_val_loss", float("inf"))),
            "valAccuracy": float((active_report_before or {}).get("val_accuracy", 0.0)),
            "evaluation": (active_report_before or {}).get("evaluation", {}),
            "shadowEvaluation": (active_report_before or {}).get("shadowEvaluation", {}),
        },
        "activeAfter": {
            "version": live_training_status.get("activeModelVersion"),
            "path": live_training_status.get("activeModelPath"),
        },
        "portfolioAtDecision": current_portfolio_snapshot(),
    }
    append_model_history(event)
    try:
        event_time_ms = int(pd.Timestamp(event["timestamp"]).timestamp() * 1000)
    except Exception:
        event_time_ms = int(time.time() * 1000)
    append_model_trace_point(
        label,
        event_time_ms,
        event["portfolioAtDecision"].get("totalPnl", 0.0),
        trade_count=event["portfolioAtDecision"].get("tradeCount", 0),
        role="champion" if promoted else "historical",
        status="promoted" if promoted else "rejected",
    )
    if checkpoint_path and not promoted:
        try:
            shadow_model_pool[label] = {
                "model": build_model_from_path(checkpoint_path),
                "state": new_virtual_account_state(),
                "path": checkpoint_path,
                "status": "shadow-active",
                "role": "historical",
            }
            ensure_model_trace(label, role="historical", status="shadow-active")
        except Exception:
            pass
    return event


def current_active_weights_path():
    return Path(BEST_WEIGHTS_PATH if os.path.exists(BEST_WEIGHTS_PATH) else WEIGHTS_PATH)


def load_active_model_from_path(weights_path):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        return False

    with ai_model_lock:
        ai_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        ai_model.eval()

    live_training_status["activeModelPath"] = str(weights_path.resolve())
    live_training_status["activeModelVersion"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(weights_path.stat().st_mtime)
    )
    sync_live_training_state()
    return True


def ensure_model_history_seeded():
    history = load_model_history(limit=1)
    current_version = live_training_status.get("activeModelVersion")
    if history and history[-1].get("activeAfter", {}).get("version") == current_version:
        return

    sequence = next_model_sequence()
    active_report = load_training_report()
    event = {
        "sequence": sequence,
        "modelLabel": f"M{sequence:03d}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "promoted": True,
        "decision": "initial active model loaded",
        "candidate": {
            "valAccuracy": float(active_report.get("val_accuracy", 0.0)),
            "bestValLoss": float(active_report.get("best_val_loss", float("inf"))),
            "valLoss": float(active_report.get("val_loss", float("inf"))),
            "feedbackMatches": 0,
            "path": live_training_status.get("activeModelPath"),
            "checkpointPath": snapshot_model_checkpoint(f"M{sequence:03d}", live_training_status.get("activeModelPath")),
        },
        "activeBefore": {
            "version": None,
            "bestValLoss": 0.0,
            "valAccuracy": 0.0,
        },
        "activeAfter": {
            "version": live_training_status.get("activeModelVersion"),
            "path": live_training_status.get("activeModelPath"),
        },
        "portfolioAtDecision": current_portfolio_snapshot(),
    }
    append_model_history(event)
    try:
        event_time_ms = int(pd.Timestamp(event["timestamp"]).timestamp() * 1000)
    except Exception:
        event_time_ms = int(time.time() * 1000)
    append_model_trace_point(
        event["modelLabel"],
        event_time_ms,
        event["portfolioAtDecision"].get("totalPnl", 0.0),
        trade_count=event["portfolioAtDecision"].get("tradeCount", 0),
        role="champion",
        status="promoted",
    )
    live_training_status["championModelLabel"] = event["modelLabel"]
    live_training_status["championModelPath"] = live_training_status.get("activeModelPath")
    live_training_status["championScore"] = None
    live_training_status["championMetrics"] = None
    sync_live_training_state()


def restore_champion_from_history():
    champion_event = best_champion_event()
    if not champion_event:
        return

    champion_path = champion_event.get("activeAfter", {}).get("path") or champion_event.get("candidate", {}).get("path")
    live_training_status["championModelLabel"] = champion_event.get("modelLabel")
    live_training_status["championModelPath"] = champion_path
    champion_metrics = champion_event.get("candidate", {}).get("shadowEvaluation") or champion_event.get("candidate", {}).get("evaluation", {})
    live_training_status["championScore"] = champion_metrics.get("score")
    live_training_status["championMetrics"] = champion_metrics

    if champion_path and live_training_status.get("activeModelPath") != champion_path and Path(champion_path).exists():
        load_active_model_from_path(champion_path)
        append_log("Restored champion model from history on startup.", "success")
    rebuild_model_traces_from_history()
    rebuild_shadow_model_pool()


def build_model_from_path(weights_path):
    model = AdaptiveTradingLSTM(input_size=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def simulate_model_performance(weights_path, raw_df):
    weights_path = Path(weights_path)
    if not weights_path.exists() or raw_df.empty or len(raw_df) < 40:
        return {
            "score": float("-inf"),
            "netPnl": 0.0,
            "tradeCount": 0,
            "winRate": 0.0,
            "signalAccuracy": 0.0,
            "maxDrawdownPct": 0.0,
            "evaluationCandles": 0,
        }

    model = build_model_from_path(weights_path)
    features = pattern_engine.extract_features(raw_df.reset_index(drop=True))
    start_index = max(20, len(features) - MODEL_EVAL_WINDOW_CANDLES)
    eval_state = {
        "cash": simulation_config["initialEquity"],
        "equity": simulation_config["initialEquity"],
        "reserved_margin": 0.0,
        "position_side": "FLAT",
        "position_size": 0.0,
        "position_notional": 0.0,
        "entry_price": None,
        "entry_confidence": 0.0,
        "stop_loss": None,
        "take_profit": None,
        "wins": 0,
        "losses": 0,
        "trade_count": 0,
        "realized_pnl": 0.0,
        "fees_paid": 0.0,
        "allocation_pct": 0.0,
    }
    resolved = 0
    correct = 0
    pending = None
    peak_equity = eval_state["equity"]
    max_drawdown_pct = 0.0

    def update_equity(price):
        unrealized = 0.0
        if eval_state["entry_price"] and eval_state["position_size"]:
            if eval_state["position_side"] == "LONG":
                unrealized = (price - eval_state["entry_price"]) * eval_state["position_size"]
            elif eval_state["position_side"] == "SHORT":
                unrealized = (eval_state["entry_price"] - price) * eval_state["position_size"]
        eval_state["equity"] = eval_state["cash"] + unrealized
        return unrealized

    def close_eval_position(price):
        if eval_state["position_side"] == "FLAT" or not eval_state["position_size"] or eval_state["entry_price"] is None:
            return
        gross = (
            (price - eval_state["entry_price"]) * eval_state["position_size"]
            if eval_state["position_side"] == "LONG"
            else (eval_state["entry_price"] - price) * eval_state["position_size"]
        )
        fee = price * eval_state["position_size"] * (effective_fee_rate_pct() / 100.0)
        pnl = gross - fee
        eval_state["cash"] += pnl
        eval_state["realized_pnl"] += pnl
        eval_state["fees_paid"] += fee
        eval_state["trade_count"] += 1
        if pnl >= 0:
            eval_state["wins"] += 1
        else:
            eval_state["losses"] += 1
        eval_state["position_side"] = "FLAT"
        eval_state["position_size"] = 0.0
        eval_state["position_notional"] = 0.0
        eval_state["entry_price"] = None
        eval_state["stop_loss"] = None
        eval_state["take_profit"] = None
        eval_state["reserved_margin"] = 0.0
        eval_state["allocation_pct"] = 0.0

    for idx in range(start_index, len(features)):
        partial = features.iloc[: idx + 1]
        current = partial.iloc[-1]
        price = float(current["Close"])
        high = float(current["High"])
        low = float(current["Low"])
        confluence = float(current.get("Confluence_Score", 0.0))
        volatility_pct = float(current.get("Volatility_20", 0.0) * 100.0)

        if eval_state["position_side"] == "LONG":
            if low <= (eval_state["stop_loss"] or 0):
                close_eval_position(float(eval_state["stop_loss"]))
            elif high >= (eval_state["take_profit"] or float("inf")):
                close_eval_position(float(eval_state["take_profit"]))
        elif eval_state["position_side"] == "SHORT":
            if high >= (eval_state["stop_loss"] or float("inf")):
                close_eval_position(float(eval_state["stop_loss"]))
            elif low <= (eval_state["take_profit"] or 0):
                close_eval_position(float(eval_state["take_profit"]))

        if pending is not None:
            resolved += 1
            if pending["action"] == "BUY":
                correct += int(price > pending["close"])
            elif pending["action"] == "SELL":
                correct += int(price < pending["close"])
            else:
                correct += int(abs(price - pending["close"]) / max(pending["close"], 1e-9) < 0.001)
            pending = None

        action, confidence = predict_action(model, partial)
        projected_move_pct = abs((max(confidence, 0.2) * (0.002 + abs(confluence) * 0.0025)) * 100.0)

        if action in {"BUY", "SELL"}:
            desired_side = "LONG" if action == "BUY" else "SHORT"
            expected_move_pct, required_edge_pct = expected_edge_pct(projected_move_pct)
            if expected_move_pct >= required_edge_pct and confidence >= simulation_config["minConfidencePct"] / 100.0:
                if eval_state["position_side"] != desired_side:
                    if eval_state["position_side"] != "FLAT":
                        close_eval_position(price)
                    available_cash = max(0.0, eval_state["cash"] - eval_state["reserved_margin"])
                    confidence_span = max(1e-9, 1.0 - simulation_config["minConfidencePct"] / 100.0)
                    confidence_factor = min(
                        1.0,
                        max(
                            0.0,
                            (confidence - simulation_config["minConfidencePct"] / 100.0) / confidence_span,
                        ),
                    )
                    confluence_factor = 0.8 + min(1.0, abs(confluence)) * 0.4
                    allocation_pct = simulation_config["minAllocationPct"] + (
                        simulation_config["maxAllocationPct"] - simulation_config["minAllocationPct"]
                    ) * confidence_factor * confluence_factor
                    allocation_pct = max(
                        simulation_config["minAllocationPct"],
                        min(simulation_config["maxAllocationPct"], allocation_pct),
                    )
                    target_notional = min(available_cash, eval_state["equity"] * allocation_pct / 100.0)
                    if target_notional >= simulation_config["minTradeNotional"]:
                        fee = target_notional * (effective_fee_rate_pct() / 100.0)
                        eval_state["cash"] -= fee
                        eval_state["realized_pnl"] -= fee
                        eval_state["fees_paid"] += fee
                        eval_state["reserved_margin"] = target_notional
                        eval_state["position_side"] = desired_side
                        eval_state["position_size"] = round(target_notional / max(price, 1e-9), 6)
                        eval_state["position_notional"] = target_notional
                        eval_state["allocation_pct"] = allocation_pct
                        eval_state["entry_price"] = price
                        eval_state["entry_confidence"] = confidence
                        eval_state["stop_loss"], eval_state["take_profit"] = compute_exit_levels(
                            price, action, confidence, volatility_pct, projected_move_pct
                        )
        elif action == "HOLD" and eval_state["position_side"] != "FLAT":
            close_eval_position(price)

        update_equity(price)
        peak_equity = max(peak_equity, eval_state["equity"])
        drawdown_pct = ((peak_equity - eval_state["equity"]) / max(peak_equity, 1e-9)) * 100.0
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        pending = {"action": action, "close": price}

    if eval_state["position_side"] != "FLAT":
        close_eval_position(float(features.iloc[-1]["Close"]))
        update_equity(float(features.iloc[-1]["Close"]))

    settled = eval_state["wins"] + eval_state["losses"]
    win_rate = eval_state["wins"] / settled if settled else 0.0
    accuracy = correct / resolved if resolved else 0.0
    score = (
        eval_state["realized_pnl"]
        + (accuracy * 120.0)
        + (win_rate * 80.0)
        - (max_drawdown_pct * 8.0)
        + min(eval_state["trade_count"], 20) * 1.5
    )
    return {
        "score": round(score, 4),
        "netPnl": round(eval_state["realized_pnl"], 2),
        "tradeCount": int(eval_state["trade_count"]),
        "winRate": round(win_rate, 4),
        "signalAccuracy": round(accuracy, 4),
        "maxDrawdownPct": round(max_drawdown_pct, 4),
        "evaluationCandles": int(len(features) - start_index),
    }


def predict_with_active_model(df_features):
    with ai_model_lock:
        return predict_action(ai_model, df_features)


def start_shadow_evaluation(candidate_report, active_report, champion_path):
    champion_model = build_model_from_path(champion_path)
    candidate_model = build_model_from_path(LIVE_CANDIDATE_BEST_PATH)
    shadow_models["champion"] = champion_model
    shadow_models["candidate"] = candidate_model

    candidate_label = f"M{next_model_sequence():03d}"
    champion_label = live_training_status.get("championModelLabel") or "Champion"
    candidate_checkpoint = snapshot_model_checkpoint(candidate_label, LIVE_CANDIDATE_BEST_PATH)
    snapshot_model_checkpoint(champion_label, champion_path)
    ensure_model_trace(champion_label, role="champion", status="active")
    ensure_model_trace(candidate_label, role="challenger", status="shadow")
    live_training_status["shadowEvaluation"] = {
        "candidateLabel": candidate_label,
        "championLabel": champion_label,
        "candidatePath": candidate_checkpoint or str(LIVE_CANDIDATE_BEST_PATH.resolve()),
        "startedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "startedCandleTime": None,
        "candlesObserved": 0,
        "minCandles": SHADOW_EVAL_MIN_CANDLES,
        "minTrades": SHADOW_EVAL_MIN_TRADES,
        "status": "running",
        "decision": "shadow evaluation running on equal live candles",
        "candidateReplay": candidate_report.get("evaluation", {}),
        "championReplay": (active_report or {}).get("evaluation", {}),
        "candidateMetrics": new_virtual_account_state(),
        "championMetrics": new_virtual_account_state(),
        "candidateReport": deepcopy(candidate_report),
        "activeReport": deepcopy(active_report or {}),
    }
    live_training_status["lastCandidate"] = candidate_report
    live_training_status["lastPromotionDecision"] = (
        f"shadow evaluation started for {candidate_label}: "
        f"need {SHADOW_EVAL_MIN_CANDLES} live candles and {SHADOW_EVAL_MIN_TRADES} shadow trades"
    )


def shadow_summary_ready(shadow):
    if not shadow or shadow.get("status") != "running":
        return False, "no live shadow evaluation"

    candidate_metrics = virtual_summary(shadow["candidateMetrics"])
    champion_metrics = virtual_summary(shadow["championMetrics"])
    candles_observed = int(shadow.get("candlesObserved", 0))
    min_candles = int(shadow.get("minCandles", SHADOW_EVAL_MIN_CANDLES))
    min_trades = int(shadow.get("minTrades", SHADOW_EVAL_MIN_TRADES))

    if candles_observed < min_candles:
        return False, f"candidate needs more equal live candles ({candles_observed}/{min_candles})"
    if candidate_metrics["tradeCount"] < min_trades or champion_metrics["tradeCount"] < min_trades:
        return (
            False,
            (
                "candidate needs more equal live shadow trades "
                f"({candidate_metrics['tradeCount']}/{min_trades} vs champion {champion_metrics['tradeCount']}/{min_trades})"
            ),
        )
    return True, "ready"


def decide_shadow_promotion(candidate_report, active_report, shadow):
    ready, gate_reason = shadow_summary_ready(shadow)
    candidate_shadow = virtual_summary(shadow["candidateMetrics"])
    champion_shadow = virtual_summary(shadow["championMetrics"])
    candidate_report["shadowEvaluation"] = candidate_shadow
    active_report["shadowEvaluation"] = champion_shadow

    if not ready:
        return False, gate_reason

    candidate_score = float(candidate_shadow.get("score", float("-inf")))
    champion_score = float(champion_shadow.get("score", float("-inf")))
    candidate_pnl = float(candidate_shadow.get("netPnl", 0.0))
    champion_pnl = float(champion_shadow.get("netPnl", 0.0))
    candidate_drawdown = float(candidate_shadow.get("maxDrawdownPct", 0.0))
    champion_drawdown = float(champion_shadow.get("maxDrawdownPct", 0.0))

    beats_shadow = candidate_score > champion_score + SHADOW_EVAL_MIN_SCORE_MARGIN
    better_shadow_profile = candidate_pnl >= champion_pnl and candidate_drawdown <= champion_drawdown + 0.5
    beats_replay, _ = score_candidate_report(candidate_report, active_report)

    if beats_shadow and better_shadow_profile and beats_replay:
        return (
            True,
            (
                "candidate beat champion on equal live shadow and replay "
                f"(shadow score {candidate_score:.2f} > {champion_score:.2f}, "
                f"net {candidate_pnl:.2f} vs {champion_pnl:.2f})"
            ),
        )

    return (
        False,
        (
            "candidate did not beat champion after equal live shadow "
            f"(shadow score {candidate_score:.2f} vs {champion_score:.2f}, "
            f"net {candidate_pnl:.2f} vs {champion_pnl:.2f}, "
            f"drawdown {candidate_drawdown:.2f}% vs {champion_drawdown:.2f}%)"
        ),
    )


def score_candidate_report(candidate_report, champion_report):
    if not candidate_report:
        return False, "candidate report missing"
    if not champion_report:
        return True, "no active baseline report"

    candidate_best = float(candidate_report.get("best_val_loss", float("inf")))
    champion_best = float(champion_report.get("best_val_loss", float("inf")))
    candidate_acc = float(candidate_report.get("val_accuracy", 0.0))
    champion_acc = float(champion_report.get("val_accuracy", 0.0))
    candidate_eval = candidate_report.get("evaluation", {})
    champion_eval = champion_report.get("evaluation", {})

    candidate_score = float(candidate_eval.get("score", float("-inf")))
    champion_score = float(champion_eval.get("score", float("-inf")))
    candidate_trades = int(candidate_eval.get("tradeCount", 0))
    champion_trades = int(champion_eval.get("tradeCount", 0))
    candidate_pnl = float(candidate_eval.get("netPnl", 0.0))
    champion_pnl = float(champion_eval.get("netPnl", 0.0))
    candidate_drawdown = float(candidate_eval.get("maxDrawdownPct", 0.0))
    champion_drawdown = float(champion_eval.get("maxDrawdownPct", 0.0))

    if candidate_trades < MODEL_EVAL_MIN_TRADES:
        return False, f"candidate needs a longer fair replay ({candidate_trades}/{MODEL_EVAL_MIN_TRADES} trades)"

    beats_replay = candidate_score > champion_score + MODEL_EVAL_MIN_SCORE_MARGIN
    beats_validation = (
        candidate_best < champion_best - LIVE_TRAINING_MIN_IMPROVEMENT and candidate_acc >= champion_acc - 0.03
    ) or (
        candidate_acc > champion_acc + 0.01 and candidate_best <= champion_best * 1.03
    )
    better_profit_profile = candidate_pnl >= champion_pnl and candidate_drawdown <= champion_drawdown + 0.5

    if beats_replay and beats_validation and better_profit_profile:
        return (
            True,
            (
                f"candidate beat champion on equal replay "
                f"(score {candidate_score:.2f} > {champion_score:.2f}, "
                f"net {candidate_pnl:.2f} vs {champion_pnl:.2f})"
            ),
        )

    return (
        False,
        (
            f"candidate did not beat champion fairly "
            f"(replay score {candidate_score:.2f} vs {champion_score:.2f}, "
            f"net {candidate_pnl:.2f} vs {champion_pnl:.2f}, "
            f"drawdown {candidate_drawdown:.2f}% vs {champion_drawdown:.2f}%)"
        ),
    )


def run_live_training_cycle():
    if not LIVE_TRAINING_ENABLED:
        return
    if live_training_status["running"]:
        return
    if live_training_status.get("shadowEvaluation", {}).get("status") == "running":
        shadow = live_training_status["shadowEvaluation"]
        live_training_status["lastPromotionDecision"] = (
            f"shadow evaluation in progress for {shadow.get('candidateLabel')} "
            f"({shadow.get('candlesObserved', 0)}/{shadow.get('minCandles', SHADOW_EVAL_MIN_CANDLES)} candles)"
        )
        sync_live_training_state()
        return

    live_training_status["running"] = True
    live_training_status["lastStartedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
    live_training_status["lastError"] = None
    sync_live_training_state()

    try:
        active_report = load_training_report()
        champion_path = Path(live_training_status.get("championModelPath") or current_active_weights_path())
        candidate_report = train_adaptive_model(
            symbol="BTCUSDT",
            loops=LIVE_TRAINING_LOOPS,
            epochs=LIVE_TRAINING_EPOCHS,
            batch_size=LIVE_TRAINING_BATCH_SIZE,
            forecast_horizon=LIVE_TRAINING_FORECAST_HORIZON,
            cycles=1,
            model_path=LIVE_CANDIDATE_MODEL_PATH,
            best_model_path=LIVE_CANDIDATE_BEST_PATH,
            report_path=LIVE_CANDIDATE_REPORT_PATH,
            initial_weights_path=current_active_weights_path(),
            feedback_path=TRADE_FEEDBACK_PATH,
            promote_best_to_model=False,
        )
        candidate_report["candidateModelPath"] = str(LIVE_CANDIDATE_BEST_PATH.resolve())
        evaluation_df = fetch_historical_data(symbol="BTCUSDT", loops=max(2, min(LIVE_TRAINING_LOOPS, 4)))
        champion_evaluation = simulate_model_performance(champion_path, evaluation_df)
        candidate_evaluation = simulate_model_performance(LIVE_CANDIDATE_BEST_PATH, evaluation_df)
        active_report["evaluation"] = champion_evaluation
        candidate_report["evaluation"] = candidate_evaluation
        should_promote, reason = score_candidate_report(candidate_report, active_report)
        candidate_report["promotionDecision"] = reason
        candidate_report["promoted"] = False
        live_training_status["lastCandidate"] = candidate_report
        active_version_before = live_training_status.get("activeModelVersion")
        live_training_status["championModelPath"] = str(champion_path.resolve())
        live_training_status["championScore"] = champion_evaluation["score"]
        live_training_status["championMetrics"] = champion_evaluation

        if should_promote and LIVE_CANDIDATE_BEST_PATH.exists():
            start_shadow_evaluation(candidate_report, active_report, champion_path)
            live_training_status["shadowEvaluation"]["activeVersionBefore"] = active_version_before
            append_log(
                f"Replay winner staged for live shadow evaluation: {live_training_status['shadowEvaluation']['candidateLabel']}.",
                "info",
            )
        else:
            live_training_status["lastPromotionDecision"] = reason
            record_model_event(candidate_report, False, reason, active_report, active_version_before)
            append_log(f"Live training kept current model: {reason}.", "info")

        live_training_status["lastCompletedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        live_training_status["lastError"] = str(exc)
        live_training_status["lastCompletedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
        append_log(f"Live training error: {exc}", "error")
    finally:
        live_training_status["running"] = False
        live_training_status["nextRunAt"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + LIVE_TRAINING_INTERVAL_SEC)
        )
        sync_live_training_state()


async def live_training_loop():
    if not LIVE_TRAINING_ENABLED:
        sync_live_training_state()
        return

    live_training_status["nextRunAt"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + LIVE_TRAINING_WARMUP_SEC)
    )
    sync_live_training_state()
    await asyncio.sleep(LIVE_TRAINING_WARMUP_SEC)

    while True:
        if count_trade_feedback() >= LIVE_TRAINING_MIN_FEEDBACK:
            await asyncio.to_thread(run_live_training_cycle)
        else:
            live_training_status["lastPromotionDecision"] = (
                f"waiting for trade feedback ({count_trade_feedback()}/{LIVE_TRAINING_MIN_FEEDBACK})"
            )
            live_training_status["nextRunAt"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + LIVE_TRAINING_INTERVAL_SEC)
            )
            sync_live_training_state()
        await asyncio.sleep(LIVE_TRAINING_INTERVAL_SEC)


def append_log(message, level="info"):
    latest_state["logs"] = (
        latest_state["logs"][-39:]
        + [{"time": time.strftime("%H:%M:%S"), "message": message, "level": level}]
    )


def append_activity(text):
    latest_state["activity"] = (
        latest_state["activity"][-19:]
        + [{"time": time.strftime("%H:%M:%S"), "text": text}]
    )


def history_payload():
    if df.empty:
        return []

    candles = []
    ordered = df.drop_duplicates(subset=["Timestamp"], keep="last").sort_values("Timestamp")
    for _, row in ordered.tail(240).iterrows():
        candles.append(
            {
                "time": int(pd.Timestamp(row["Timestamp"]).timestamp()),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            }
        )
    return candles


def _latest_candle_open_ms():
    if df.empty:
        return None
    return int(pd.Timestamp(df.iloc[-1]["Timestamp"]).timestamp() * 1000)


def _row_exists_for_open_ms(open_ms):
    if df.empty:
        return False
    ts = pd.to_datetime(open_ms, unit="ms")
    return bool((df["Timestamp"] == ts).any())


def _estimate_recent_volatility(window=30):
    if df.empty or len(df) < 3:
        return 0.0005
    closes = df["Close"].astype(float).tail(window + 1).to_numpy()
    rets = np.diff(np.log(np.clip(closes, 1e-9, None)))
    if len(rets) == 0:
        return 0.0005
    return float(max(0.0002, np.nanstd(rets)))


def _build_candle_row(
    *,
    open_ms,
    open_price,
    high_price,
    low_price,
    close_price,
    volume,
    receive_timestamp_ms,
    is_imputed,
    gap_minutes,
):
    close_tick_ms = open_ms + CANDLE_INTERVAL_MS - 1
    receive_lag_ms = max(0.0, float(receive_timestamp_ms - close_tick_ms))
    return {
        "Timestamp": pd.to_datetime(open_ms, unit="ms"),
        "Open": float(open_price),
        "High": float(high_price),
        "Low": float(low_price),
        "Close": float(close_price),
        "Volume": float(max(0.0, volume)),
        "Is_Imputed": bool(is_imputed),
        "Gap_Minutes": float(max(0.0, gap_minutes)),
        "Receive_Lag_Ms": float(receive_lag_ms),
    }


def _append_row_dict(row_dict):
    global df
    new_df = pd.DataFrame([row_dict], columns=df.columns)
    if df.empty:
        df = new_df
    else:
        df = pd.concat([df, new_df], ignore_index=True)
    if "Is_Imputed" not in df.columns:
        df["Is_Imputed"] = False
    df["_real_candle_priority"] = df["Is_Imputed"].astype(bool).map({False: 1, True: 0})
    df = (
        df.sort_values(["Timestamp", "_real_candle_priority"])
        .drop_duplicates(subset=["Timestamp"], keep="last")
        .drop(columns=["_real_candle_priority"])
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    if len(df) > MAX_CANDLES:
        df = df.iloc[-MAX_CANDLES:].reset_index(drop=True)


def _generate_synthetic_row(open_ms, receive_timestamp_ms, gap_minutes):
    if df.empty:
        return None

    last_close = float(df.iloc[-1]["Close"])
    vol = _estimate_recent_volatility()
    gap_factor = min(2.0, 1.0 + (float(gap_minutes) * 0.25))
    shock = random.gauss(0.0, min(0.012, max(0.0005, vol * 1.4 * gap_factor)))
    synthetic_close = max(1e-9, last_close * (1.0 + shock))
    wick = min(0.02, max(0.001, abs(shock) * 2.2 + vol * 1.2))
    high = max(last_close, synthetic_close) * (1.0 + random.uniform(0.0, wick))
    low = min(last_close, synthetic_close) * (1.0 - random.uniform(0.0, wick))
    return _build_candle_row(
        open_ms=open_ms,
        open_price=last_close,
        high_price=high,
        low_price=low,
        close_price=synthetic_close,
        volume=0.0,
        receive_timestamp_ms=receive_timestamp_ms,
        is_imputed=True,
        gap_minutes=gap_minutes,
    )


def _load_sarima_profile():
    global sarima_profile
    if sarima_profile is not None:
        return sarima_profile
    try:
        sarima_profile = load_profile(PROFILE_PATH)
    except Exception:
        sarima_profile = None
    return sarima_profile


def _reconstruct_missing_rows(open_times_ms, receive_timestamp_ms):
    # This helper is the presentation-aligned entry point for network resilience:
    # instead of always inventing synthetic candles, try the SARIMA repair layer
    # first so downstream AI sees reconstructed market structure.
    profile = _load_sarima_profile()
    if profile is None or df.empty or len(df) < 80:
        return []

    try:
        return reconstruct_missing_rows(
            df,
            open_times_ms=open_times_ms,
            receive_timestamp_ms=receive_timestamp_ms,
            profile=profile,
        )
    except Exception:
        return []


def _insert_synthetic_candle(open_ms, receive_timestamp_ms, gap_minutes=1.0):
    if _row_exists_for_open_ms(open_ms):
        return None
    row = _generate_synthetic_row(open_ms, receive_timestamp_ms, gap_minutes)
    if row is None:
        return None
    _append_row_dict(row)
    network_test_stats["syntheticFilledCandles"] += 1
    network_test_stats["consecutiveSyntheticCandles"] += 1
    network_test_stats["maxConsecutiveSyntheticCandles"] = max(
        network_test_stats["maxConsecutiveSyntheticCandles"],
        network_test_stats["consecutiveSyntheticCandles"],
    )
    network_test_stats["lastGapMinutes"] = float(max(network_test_stats["lastGapMinutes"], gap_minutes))
    return {"tick_timestamp_ms": int(open_ms + CANDLE_INTERVAL_MS - 1), "receive_timestamp_ms": receive_timestamp_ms}


def fill_missing_candles_until(target_open_ms, receive_timestamp_ms):
    """
    Backfills timeline gaps before target_open_ms.

    Final project flow:
    1. Reconstruct missing candles with SARIMA when possible.
    2. Fall back to the old synthetic method if that repair is unavailable.
    3. Pass the repaired stream into feature engineering and then the LSTM.

    This keeps the implementation aligned with the presentation claim that
    network damage is handled first at the candle-repair layer.
    """
    inserted = []
    last_open_ms = _latest_candle_open_ms()
    if last_open_ms is None:
        return inserted

    missing = int((target_open_ms - last_open_ms) / CANDLE_INTERVAL_MS) - 1
    if missing <= 0:
        return inserted

    missing = min(missing, MAX_SYNTHETIC_GAP_FILL)
    open_times_ms = [last_open_ms + (step * CANDLE_INTERVAL_MS) for step in range(1, missing + 1)]
    reconstructed_rows = _reconstruct_missing_rows(open_times_ms, receive_timestamp_ms)

    if reconstructed_rows:
        # These inserted rows are now statistical reconstructions rather than
        # pure noise, which is the key network-to-AI link from the presentation.
        for row in reconstructed_rows:
            open_ms = int(pd.Timestamp(row["Timestamp"]).timestamp() * 1000)
            if _row_exists_for_open_ms(open_ms):
                continue
            _append_row_dict(row)
            network_test_stats["syntheticFilledCandles"] += 1
            network_test_stats["consecutiveSyntheticCandles"] += 1
            network_test_stats["maxConsecutiveSyntheticCandles"] = max(
                network_test_stats["maxConsecutiveSyntheticCandles"],
                network_test_stats["consecutiveSyntheticCandles"],
            )
            network_test_stats["lastGapMinutes"] = float(
                max(network_test_stats["lastGapMinutes"], float(row.get("Gap_Minutes", 1.0)))
            )
            inserted.append(
                {
                    "tick_timestamp_ms": int(open_ms + CANDLE_INTERVAL_MS - 1),
                    "receive_timestamp_ms": receive_timestamp_ms,
                }
            )
        return inserted

    for step in range(1, missing + 1):
        # Preserve the legacy fallback so the live app remains usable even if
        # the SARIMA profile has not been trained yet.
        open_ms = open_times_ms[step - 1]
        gap_minutes = (target_open_ms - open_ms) / CANDLE_INTERVAL_MS
        entry = _insert_synthetic_candle(open_ms, receive_timestamp_ms, gap_minutes=max(1.0, gap_minutes))
        if entry:
            inserted.append(entry)
    return inserted


def process_latest_candle_state(tick_timestamp_ms, receive_timestamp_ms):
    df_features = pattern_engine.extract_features(df)
    apply_closed_candle_logic(df_features, tick_timestamp_ms, receive_timestamp_ms)


def refresh_portfolio_snapshot(last_price):
    entry_price = paper_state["entry_price"]
    position_size = paper_state["position_size"]
    position_side = paper_state["position_side"]
    reserved_margin = paper_state["reserved_margin"]

    unrealized = 0.0
    if entry_price and position_size:
        if position_side == "LONG":
            unrealized = (last_price - entry_price) * position_size
        elif position_side == "SHORT":
            unrealized = (entry_price - last_price) * position_size

    paper_state["unrealized_pnl"] = unrealized
    paper_state["equity"] = paper_state["cash"] + unrealized
    available_cash = max(0.0, paper_state["cash"] - reserved_margin)

    resolved = signal_scoreboard["resolved"]
    accuracy = signal_scoreboard["correct"] / resolved if resolved else 0.0
    settled_trades = paper_state["wins"] + paper_state["losses"]
    win_rate = paper_state["wins"] / settled_trades if settled_trades else 0.0

    latest_state["portfolio"] = {
        "initialEquity": round(simulation_config["initialEquity"], 2),
        "equity": round(paper_state["equity"], 2),
        "cash": round(paper_state["cash"], 2),
        "availableCash": round(available_cash, 2),
        "reservedMargin": round(reserved_margin, 2),
        "realizedPnl": round(paper_state["realized_pnl"], 2),
        "unrealizedPnl": round(unrealized, 2),
        "totalPnl": round(paper_state["realized_pnl"] + unrealized, 2),
        "feesPaid": round(paper_state["fees_paid"], 2),
        "position": position_side,
        "positionSize": position_size,
        "positionNotional": round(paper_state["position_notional"], 2),
        "deployedCapital": round(paper_state["deployed_notional_total"], 2),
        "allocationPct": round(paper_state["allocation_pct"], 4),
        "entryPrice": round(entry_price, 2) if entry_price else None,
        "entryConfidencePct": round(paper_state["entry_confidence"] * 100.0, 2)
        if paper_state["entry_confidence"] is not None
        else None,
        "stopLoss": round(paper_state["stop_loss"], 2) if paper_state["stop_loss"] else None,
        "takeProfit": round(paper_state["take_profit"], 2) if paper_state["take_profit"] else None,
        "tradeCount": paper_state["trade_count"],
        "winRate": round(win_rate, 4),
        "signalAccuracy": round(accuracy, 4),
    }
    latest_state["simulation"] = simulation_config.copy()
    latest_state["networkTest"] = network_test_config.copy()
    latest_state["simulationSummary"] = simulation_summary()
    latest_state["training"] = load_training_report()
    latest_state["blotter"] = list(trade_blotter)
    champion_label = live_training_status.get("championModelLabel")
    if champion_label:
        append_model_trace_point(
            champion_label,
            int(time.time() * 1000),
            latest_state["portfolio"]["totalPnl"],
            trade_count=latest_state["portfolio"]["tradeCount"],
            role="champion",
            status="active",
        )
    sync_live_training_state()


def close_position(price, timestamp_label, reason="signal", candle_time=None):
    position_side = paper_state["position_side"]
    position_size = paper_state["position_size"]
    entry_price = paper_state["entry_price"]

    if position_side == "FLAT" or not position_size or entry_price is None:
        return

    if position_side == "LONG":
        gross_pnl = (price - entry_price) * position_size
    else:
        gross_pnl = (entry_price - price) * position_size

    fee = price * position_size * (effective_fee_rate_pct() / 100.0)
    pnl = gross_pnl - fee

    paper_state["cash"] += pnl
    paper_state["realized_pnl"] += pnl
    paper_state["fees_paid"] += fee
    paper_state["trade_count"] += 1

    if pnl >= 0:
        paper_state["wins"] += 1
    else:
        paper_state["losses"] += 1

    append_log(
        f"{timestamp_label} Closed {position_side} at ${price:,.2f} | Net PnL {pnl:+.2f} | {reason}",
        "success" if pnl >= 0 else "error",
    )
    append_activity(
        f"{'Sold' if position_side == 'LONG' else 'Covered'} {position_size:.4f} BTC at ${price:,.2f} ({reason})"
    )
    trade_blotter.appendleft(
        {
            "id": paper_state.get("trade_id"),
            "openedAt": paper_state.get("opened_at"),
            "closedAt": time.strftime("%H:%M:%S"),
            "side": position_side,
            "size": round(position_size, 6),
            "entryPrice": round(entry_price, 2),
            "exitPrice": round(price, 2),
            "notional": round(paper_state["position_notional"], 2),
            "allocationPct": round(paper_state["allocation_pct"], 4),
            "entryConfidencePct": round((paper_state["entry_confidence"] or 0.0) * 100.0, 2),
            "grossPnl": round(gross_pnl, 2),
            "fees": round((paper_state.get("entry_fee") or 0.0) + fee, 2),
            "netPnl": round((gross_pnl - (paper_state.get("entry_fee") or 0.0) - fee), 2),
            "reason": reason,
        }
    )
    append_trade_feedback(
        {
            "tradeId": paper_state.get("trade_id"),
            "openedAt": paper_state.get("opened_at"),
            "closedAt": time.strftime("%H:%M:%S"),
            "openedCandleTime": paper_state.get("opened_candle_time"),
            "closedCandleTime": candle_time,
            "actionLabel": 1 if position_side == "LONG" else 2,
            "profitable": bool(pnl >= 0),
            "netPnl": round(pnl, 2),
            "grossPnl": round(gross_pnl, 2),
            "fees": round((paper_state.get("entry_fee") or 0.0) + fee, 2),
            "reason": reason,
            "entryConfidencePct": round((paper_state.get("entry_confidence") or 0.0) * 100.0, 2),
        }
    )

    paper_state["position_side"] = "FLAT"
    paper_state["position_size"] = 0.0
    paper_state["position_notional"] = 0.0
    paper_state["allocation_pct"] = 0.0
    paper_state["reserved_margin"] = 0.0
    paper_state["entry_price"] = None
    paper_state["entry_confidence"] = None
    paper_state["entry_fee"] = 0.0
    paper_state["opened_at"] = None
    paper_state["stop_loss"] = None
    paper_state["take_profit"] = None
    paper_state["trade_id"] = None
    paper_state["opened_candle_time"] = None
    paper_state["entry_action_label"] = None


def calculate_position_size(price, confidence, confluence_score, min_confidence_pct, min_trade_notional):
    available_cash = max(0.0, paper_state["cash"] - paper_state["reserved_margin"])
    equity_base = max(0.0, paper_state["equity"])
    min_confidence = min_confidence_pct / 100.0

    if confidence < min_confidence or available_cash <= 0:
        return 0.0, 0.0, 0.0

    confidence_span = max(1e-9, 1.0 - min_confidence)
    confidence_factor = min(1.0, max(0.0, (confidence - min_confidence) / confidence_span))
    confluence_factor = 0.8 + min(1.0, abs(confluence_score)) * 0.4
    allocation_pct = simulation_config["minAllocationPct"] + (
        simulation_config["maxAllocationPct"] - simulation_config["minAllocationPct"]
    ) * confidence_factor * confluence_factor
    allocation_pct = max(simulation_config["minAllocationPct"], min(simulation_config["maxAllocationPct"], allocation_pct))

    target_notional = min(available_cash, equity_base * allocation_pct / 100.0)
    if target_notional < min_trade_notional:
        return 0.0, 0.0, 0.0

    size = target_notional / max(price, 1e-9)
    return round(size, 6), round(target_notional, 2), round(allocation_pct, 4)


def expected_edge_pct(projected_move_pct):
    if simulation_config.get("ignoreFees"):
        return abs(projected_move_pct), 0.0

    round_trip_cost_pct = effective_fee_rate_pct() * 2
    required_edge_pct = round_trip_cost_pct * DEFAULT_EDGE_SAFETY_MULTIPLIER + DEFAULT_SLIPPAGE_BUFFER_PCT
    expected_move_pct = abs(projected_move_pct)
    return expected_move_pct, required_edge_pct


def compute_exit_levels(price, action, confidence, volatility_pct, projected_move_pct):
    expected_move_pct = max(abs(projected_move_pct), volatility_pct, 0.20)
    confidence_boost = 0.8 + confidence
    stop_distance_pct = max(0.18, expected_move_pct * 0.65)
    take_distance_pct = max(stop_distance_pct * 1.6, expected_move_pct * confidence_boost)

    if action == "BUY":
        stop_loss = price * (1.0 - stop_distance_pct / 100.0)
        take_profit = price * (1.0 + take_distance_pct / 100.0)
    else:
        stop_loss = price * (1.0 + stop_distance_pct / 100.0)
        take_profit = price * (1.0 - take_distance_pct / 100.0)

    return round(stop_loss, 2), round(take_profit, 2)


def open_position(
    action,
    price,
    confidence,
    timestamp_label,
    confluence_score,
    projected_move_pct,
    volatility_pct,
    trend_bias_pct,
    candle_time,
):
    global trade_sequence
    if action not in {"BUY", "SELL"}:
        return

    if action == "BUY" and not simulation_config["allowLong"]:
        return
    if action == "SELL" and not simulation_config["allowShort"]:
        return

    desired_side = "LONG" if action == "BUY" else "SHORT"
    if paper_state["position_side"] == desired_side:
        return

    if paper_state["position_side"] != "FLAT":
        close_position(price, timestamp_label, candle_time=candle_time)

    expected_move_pct, required_edge_pct = expected_edge_pct(projected_move_pct)
    if expected_move_pct < required_edge_pct:
        append_log(
            (
                f"{timestamp_label} Skipped {desired_side}: expected edge {expected_move_pct:.3f}% "
                f"below {'signal' if simulation_config.get('ignoreFees') else 'fee-aware'} threshold {required_edge_pct:.3f}%"
            ),
            "info",
        )
        return

    dynamic_min_conf, dynamic_min_notional = compute_dynamic_entry_thresholds(
        volatility_pct,
        confluence_score,
        trend_bias_pct,
        paper_state["equity"],
    )
    size, target_notional, allocation_pct = calculate_position_size(
        price,
        confidence,
        confluence_score,
        dynamic_min_conf,
        dynamic_min_notional,
    )
    if size <= 0 or target_notional <= 0:
        return

    fee_rate = effective_fee_rate_pct() / 100.0
    fee = target_notional * fee_rate
    paper_state["cash"] -= fee
    paper_state["realized_pnl"] -= fee
    paper_state["fees_paid"] += fee
    paper_state["reserved_margin"] = target_notional
    paper_state["position_side"] = desired_side
    paper_state["position_size"] = size
    paper_state["position_notional"] = target_notional
    paper_state["deployed_notional_total"] += target_notional
    paper_state["allocation_pct"] = allocation_pct
    paper_state["entry_price"] = price
    paper_state["entry_confidence"] = confidence
    paper_state["entry_fee"] = fee
    paper_state["opened_at"] = time.strftime("%H:%M:%S")
    paper_state["opened_candle_time"] = candle_time
    paper_state["entry_action_label"] = 1 if desired_side == "LONG" else 2
    paper_state["stop_loss"], paper_state["take_profit"] = compute_exit_levels(
        price, action, confidence, volatility_pct, projected_move_pct
    )
    trade_sequence += 1
    paper_state["trade_id"] = trade_sequence
    append_log(
        (
            f"{timestamp_label} Opened {desired_side} {paper_state['position_size']:.4f} BTC "
            f"(${target_notional:,.2f}, {allocation_pct:.2f}% of equity) at ${price:,.2f} "
            f"| SL ${paper_state['stop_loss']:,.2f} | TP ${paper_state['take_profit']:,.2f}"
        ),
        "info",
    )
    append_activity(
        f"{'Bought' if desired_side == 'LONG' else 'Shorted'} {paper_state['position_size']:.4f} BTC at ${price:,.2f}"
    )


def maybe_close_on_risk_levels(latest_features, timestamp_label, candle_time):
    if paper_state["position_side"] == "FLAT":
        return

    high_price = float(latest_features["High"])
    low_price = float(latest_features["Low"])

    if paper_state["position_side"] == "LONG":
        if low_price <= (paper_state["stop_loss"] or 0):
            close_position(float(paper_state["stop_loss"]), timestamp_label, "stop-loss", candle_time)
            return
        if high_price >= (paper_state["take_profit"] or float("inf")):
            close_position(float(paper_state["take_profit"]), timestamp_label, "take-profit", candle_time)
            return
    else:
        if high_price >= (paper_state["stop_loss"] or float("inf")):
            close_position(float(paper_state["stop_loss"]), timestamp_label, "stop-loss", candle_time)
            return
        if low_price <= (paper_state["take_profit"] or 0):
            close_position(float(paper_state["take_profit"]), timestamp_label, "take-profit", candle_time)
            return


def resolve_previous_signal(latest_close):
    global pending_signal
    if pending_signal is None:
        return

    action = pending_signal["action"]
    previous_close = pending_signal["close"]

    if action == "BUY":
        correct = latest_close > previous_close
    elif action == "SELL":
        correct = latest_close < previous_close
    else:
        correct = abs(latest_close - previous_close) / max(previous_close, 1e-9) < 0.001

    signal_scoreboard["resolved"] += 1
    signal_scoreboard["correct"] += int(correct)
    pending_signal = None


def update_prediction_targets(close_price, action, confidence, confluence_score, raw_confidence=None):
    projected_move_pct, target_price = projected_move_and_target(close_price, action, confidence, confluence_score)

    latest_state["prediction"] = {
        "action": action,
        "confidence": round(float(confidence), 4),
        "rawConfidence": round(float(raw_confidence if raw_confidence is not None else confidence), 4),
        "targetPrice": round(float(target_price), 2),
        "projectedMovePct": round(float(projected_move_pct * 100.0), 4),
    }


def update_telemetry(latest_features, tick_timestamp_ms, receive_timestamp_ms, signal_timestamp_ms):
    update_timestamps.append(receive_timestamp_ms)

    if len(update_timestamps) > 1:
        window_seconds = (update_timestamps[-1] - update_timestamps[0]) / 1000
        updates_per_minute = ((len(update_timestamps) - 1) / window_seconds * 60) if window_seconds > 0 else 0.0
    else:
        updates_per_minute = 0.0

    received = max(1, network_test_stats["receivedClosedCandles"])
    packet_loss_estimate = (network_test_stats["droppedClosedCandles"] / received) * 100.0
    synthetic_total = max(0, int(network_test_stats["syntheticFilledCandles"]))
    real_processed = max(0, int(network_test_stats["processedClosedCandles"]))
    imputed_rate = (synthetic_total / max(1, synthetic_total + real_processed)) * 100.0

    latest_state["telemetry"] = {
        "tickTime": tick_timestamp_ms,
        "receivedTime": receive_timestamp_ms,
        "signalTime": signal_timestamp_ms,
        "deltaMs": int(signal_timestamp_ms - tick_timestamp_ms),
        "updatesPerMinute": round(updates_per_minute, 2),
        "packetLossPct": round(packet_loss_estimate, 2),
        "latencyMs": round((signal_timestamp_ms - receive_timestamp_ms), 2),
        "simulatedDelayMs": round(network_test_stats["lastAppliedDelayMs"], 2),
        "processedCandles": int(network_test_stats["processedClosedCandles"]),
        "droppedCandles": int(network_test_stats["droppedClosedCandles"]),
        "syntheticFilledCandles": synthetic_total,
        "imputedRatePct": round(imputed_rate, 2),
        "maxConsecutiveImputed": int(network_test_stats["maxConsecutiveSyntheticCandles"]),
        "lastGapMinutes": round(float(network_test_stats["lastGapMinutes"]), 2),
        "volatilityPct": round(float(latest_features.get("Volatility_20", 0.0) * 100.0), 4),
        "rangePct": round(float(latest_features.get("Range_Pct", 0.0) * 100.0), 4),
        "trendBiasPct": round(float(latest_features.get("Trend_Bias", 0.0) * 100.0), 4),
        "confluenceScore": round(float(latest_features.get("Confluence_Score", 0.0)), 4),
        "dataQuality": round(float(latest_features.get("Data_Quality", 1.0)), 4),
    }


def maybe_advance_shadow_evaluation(df_features):
    shadow = live_training_status.get("shadowEvaluation")
    if not shadow or shadow.get("status") != "running":
        return
    if shadow_models["champion"] is None or shadow_models["candidate"] is None:
        reset_shadow_evaluation("shadow models were unavailable and the probation run was cleared")
        return

    latest_features = df_features.iloc[-1]
    candle_time = pd.Timestamp(latest_features["Timestamp"]).strftime("%Y-%m-%d %H:%M:%S")

    if shadow.get("startedCandleTime") is None:
        shadow["startedCandleTime"] = candle_time

    step_virtual_model(shadow["championMetrics"], shadow_models["champion"], df_features)
    step_virtual_model(shadow["candidateMetrics"], shadow_models["candidate"], df_features)
    shadow["candlesObserved"] = int(shadow.get("candlesObserved", 0)) + 1
    shadow["candidateMetrics"] = shadow["candidateMetrics"]
    shadow["championMetrics"] = shadow["championMetrics"]
    timestamp_ms = int(pd.Timestamp(latest_features["Timestamp"]).timestamp() * 1000)
    candidate_shadow = virtual_summary(shadow["candidateMetrics"])
    champion_shadow = virtual_summary(shadow["championMetrics"])
    shadow["candidateShadow"] = candidate_shadow
    shadow["championShadow"] = champion_shadow
    append_model_trace_point(
        shadow.get("championLabel") or live_training_status.get("championModelLabel"),
        timestamp_ms,
        champion_shadow["netPnl"],
        trade_count=champion_shadow["tradeCount"],
        role="champion",
        status="active-shadow",
    )
    append_model_trace_point(
        shadow.get("candidateLabel"),
        timestamp_ms,
        candidate_shadow["netPnl"],
        trade_count=candidate_shadow["tradeCount"],
        role="challenger",
        status="shadow",
    )

    active_report = deepcopy(shadow.get("activeReport") or load_training_report())
    candidate_report = deepcopy(shadow.get("candidateReport") or {})
    ready, gate_reason = shadow_summary_ready(shadow)

    if not ready:
        shadow["decision"] = gate_reason
        live_training_status["lastPromotionDecision"] = gate_reason
        sync_live_training_state()
        return

    should_promote, reason = decide_shadow_promotion(candidate_report, active_report, shadow)
    candidate_report["promotionDecision"] = reason
    candidate_report["promoted"] = should_promote
    live_training_status["lastCandidate"] = candidate_report
    live_training_status["lastPromotionDecision"] = reason
    active_version_before = shadow.get("activeVersionBefore") or live_training_status.get("activeModelVersion")

    if should_promote and LIVE_CANDIDATE_BEST_PATH.exists():
        target_best = Path(BEST_WEIGHTS_PATH)
        target_model = Path(WEIGHTS_PATH)
        target_best.write_bytes(LIVE_CANDIDATE_BEST_PATH.read_bytes())
        target_model.write_bytes(LIVE_CANDIDATE_BEST_PATH.read_bytes())
        load_active_model_from_path(target_best)
        TRAINING_REPORT_PATH.write_text(json.dumps(candidate_report, indent=2))
        live_training_status["lastPromotedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
        live_training_status["promotionCount"] += 1
        live_training_status["championModelPath"] = str(target_best.resolve())
        live_training_status["championScore"] = candidate_shadow["score"]
        live_training_status["championMetrics"] = candidate_shadow
        event = record_model_event(candidate_report, True, reason, active_report, active_version_before)
        live_training_status["championModelLabel"] = event["modelLabel"]
        ensure_model_trace(event["modelLabel"], role="champion", status="promoted")
        append_log(f"Shadow evaluation promoted a new champion: {reason}.", "success")
    else:
        record_model_event(candidate_report, False, reason, active_report, active_version_before)
        if shadow.get("candidateLabel") in model_performance_store:
            model_performance_store[shadow["candidateLabel"]]["status"] = "rejected"
            model_performance_store[shadow["candidateLabel"]]["role"] = "historical"
        append_log(f"Shadow evaluation rejected the challenger: {reason}.", "info")

    reset_shadow_evaluation(reason)
    sync_live_training_state()


def advance_shadow_model_pool(df_features):
    if not shadow_model_pool:
        return

    latest_features = df_features.iloc[-1]
    timestamp_ms = int(pd.Timestamp(latest_features["Timestamp"]).timestamp() * 1000)
    labels_to_remove = []

    for label, entry in shadow_model_pool.items():
        model = entry.get("model")
        state = entry.get("state")
        if model is None or state is None:
            labels_to_remove.append(label)
            continue
        step_virtual_model(state, model, df_features)
        summary = virtual_summary(state)
        append_model_trace_point(
            label,
            timestamp_ms,
            summary["netPnl"],
            trade_count=summary["tradeCount"],
            role=entry.get("role", "historical"),
            status=entry.get("status", "shadow-active"),
        )

    for label in labels_to_remove:
        shadow_model_pool.pop(label, None)


def apply_closed_candle_logic(df_features, tick_timestamp_ms, receive_timestamp_ms, signal_timestamp_ms=None):
    global pending_signal

    latest_features = df_features.iloc[-1]
    close_price = float(latest_features["Close"])
    candle_time = int(pd.Timestamp(latest_features["Timestamp"]).timestamp())
    timestamp_label = pd.Timestamp(latest_features["Timestamp"]).strftime("%H:%M")

    latest_state["patterns"] = {
        "Bullish_FVG": bool(latest_features.get("Bullish_FVG", False)),
        "Bearish_FVG": bool(latest_features.get("Bearish_FVG", False)),
        "HH": bool(latest_features.get("HH", False)),
        "LL": bool(latest_features.get("LL", False)),
        "HL": bool(latest_features.get("HL", False)),
        "LH": bool(latest_features.get("LH", False)),
    }

    maybe_close_on_risk_levels(latest_features, timestamp_label, candle_time)
    resolve_previous_signal(close_price)

    signal_timestamp_ms = int(signal_timestamp_ms if signal_timestamp_ms is not None else time.time() * 1000)
    action, raw_confidence = predict_with_active_model(df_features)
    data_quality = float(latest_features.get("Data_Quality", 1.0))
    confidence = float(
        max(
            0.0,
            min(1.0, raw_confidence * (0.35 + 0.65 * max(0.0, min(1.0, data_quality)))),
        )
    )
    confluence_score = float(latest_features.get("Confluence_Score", 0.0))
    volatility_pct = float(latest_features.get("Volatility_20", 0.0) * 100.0)
    trend_bias_pct = float(latest_features.get("Trend_Bias", 0.0) * 100.0)
    compute_dynamic_entry_thresholds(
        volatility_pct,
        confluence_score,
        trend_bias_pct,
        paper_state["equity"],
    )
    maybe_advance_shadow_evaluation(df_features)
    advance_shadow_model_pool(df_features)
    update_prediction_targets(close_price, action, confidence, confluence_score, raw_confidence=raw_confidence)
    update_telemetry(latest_features, tick_timestamp_ms, receive_timestamp_ms, signal_timestamp_ms)

    if action in {"BUY", "SELL"}:
        open_position(
            action,
            close_price,
            confidence,
            timestamp_label,
            confluence_score,
            latest_state["prediction"]["projectedMovePct"],
            volatility_pct,
            trend_bias_pct,
            candle_time,
        )
    elif action == "HOLD" and paper_state["position_side"] != "FLAT":
        close_position(close_price, timestamp_label, "signal-flatten", candle_time)

    refresh_portfolio_snapshot(close_price)

    pending_signal = {
        "action": action,
        "close": close_price,
    }

    latest_state["history"] = history_payload()
    append_log(
        (
            f"{timestamp_label} {action} {confidence * 100:0.1f}% | "
            f"Δt {latest_state['telemetry']['deltaMs']}ms | "
            f"Conf {confluence_score:+0.2f}"
        ),
        "success" if action == "BUY" else "error" if action == "SELL" else "info",
    )


def replay_seeded_history(df_features, replay_window=180):
    reset_runtime_state()
    start_index = max(10, len(df_features) - replay_window)

    for idx in range(start_index, len(df_features)):
        partial = df_features.iloc[: idx + 1]
        candle_timestamp_ms = int(pd.Timestamp(partial.iloc[-1]["Timestamp"]).timestamp() * 1000)
        apply_closed_candle_logic(partial, candle_timestamp_ms, candle_timestamp_ms, signal_timestamp_ms=candle_timestamp_ms)

    latest_state["history"] = history_payload()


def _request_json(url):
    request = Request(url, headers={"User-Agent": "BTC-Trading-Bot/1.0"})
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _binance_history_rows(symbol, limit):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval=1m&limit={limit}"
    raw = _request_json(url)
    rows = []
    for candle in raw:
        rows.append(
            {
                "Timestamp": pd.to_datetime(candle[0], unit="ms"),
                "Open": float(candle[1]),
                "High": float(candle[2]),
                "Low": float(candle[3]),
                "Close": float(candle[4]),
                "Volume": float(candle[5]),
                "Is_Imputed": False,
                "Gap_Minutes": 0.0,
                "Receive_Lag_Ms": 0.0,
            }
        )
    return rows


def _coinbase_history_rows(limit):
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60"
    raw = _request_json(url)
    rows = []
    for candle in raw[:limit]:
        timestamp_s, low, high, open_price, close_price, volume = candle
        rows.append(
            {
                "Timestamp": pd.to_datetime(int(timestamp_s), unit="s"),
                "Open": float(open_price),
                "High": float(high),
                "Low": float(low),
                "Close": float(close_price),
                "Volume": float(volume),
                "Is_Imputed": False,
                "Gap_Minutes": 0.0,
                "Receive_Lag_Ms": 0.0,
            }
        )
    return rows


def _current_minute_open_ms():
    return int(time.time() // 60 * CANDLE_INTERVAL_MS)


def _publish_rest_market_rows(rows, provider):
    global df
    if not rows:
        return

    ordered_rows = sorted(rows, key=lambda row: row["Timestamp"])
    latest_row = ordered_rows[-1]
    receive_timestamp_ms = int(time.time() * 1000)

    latest_state["candle"] = {
        "time": int(pd.Timestamp(latest_row["Timestamp"]).timestamp()),
        "open": float(latest_row["Open"]),
        "high": float(latest_row["High"]),
        "low": float(latest_row["Low"]),
        "close": float(latest_row["Close"]),
    }
    refresh_portfolio_snapshot(float(latest_row["Close"]))

    current_minute_open_ms = _current_minute_open_ms()
    closed_rows = [
        row for row in ordered_rows
        if int(pd.Timestamp(row["Timestamp"]).timestamp() * 1000) < current_minute_open_ms
    ]

    for row in closed_rows:
        open_ms = int(pd.Timestamp(row["Timestamp"]).timestamp() * 1000)
        if _row_exists_for_open_ms(open_ms):
            continue

        backfilled = fill_missing_candles_until(open_ms, receive_timestamp_ms)
        for item in backfilled:
            process_latest_candle_state(item["tick_timestamp_ms"], item["receive_timestamp_ms"])

        _append_row_dict(row)
        network_test_stats["receivedClosedCandles"] += 1
        network_test_stats["processedClosedCandles"] += 1
        network_test_stats["lastAppliedDelayMs"] = 0.0
        network_test_stats["consecutiveSyntheticCandles"] = 0
        process_latest_candle_state(open_ms + CANDLE_INTERVAL_MS - 1, receive_timestamp_ms)
        append_log(f"Processed fallback market candle from {provider}.", "info")


async def poll_rest_market_data(symbol="BTCUSDT", interval_seconds=10):
    provider = "Coinbase REST fallback"
    while True:
        try:
            rows = await asyncio.to_thread(_coinbase_history_rows, 4)
            _publish_rest_market_rows(rows, provider)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("REST market fallback failed: %s", exc)
            append_log(f"REST market fallback failed: {exc}", "error")

        await asyncio.sleep(interval_seconds)


def bootstrap_historical_data(symbol="BTCUSDT", limit=200):
    global df
    rows = []
    provider = "Binance REST"
    try:
        rows = _binance_history_rows(symbol, limit)
    except Exception as exc:
        logger.warning("Binance historical seed failed: %s", exc)
        try:
            provider = "Coinbase REST fallback"
            rows = _coinbase_history_rows(limit)
        except Exception as fallback_exc:
            logger.warning("Coinbase historical seed failed: %s", fallback_exc)
            append_log("Historical seed unavailable. Waiting for live stream.", "error")
            return

    try:
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["Timestamp"], keep="last").sort_values("Timestamp").reset_index(drop=True)
        latest_state["history"] = history_payload()
        if not df.empty:
            last = df.iloc[-1]
            latest_state["candle"] = {
                "time": int(pd.Timestamp(last["Timestamp"]).timestamp()),
                "open": float(last["Open"]),
                "high": float(last["High"]),
                "low": float(last["Low"]),
                "close": float(last["Close"]),
            }
            seeded_features = pattern_engine.extract_features(df)
            replay_seeded_history(seeded_features)
            refresh_portfolio_snapshot(float(last["Close"]))
        append_log(f"Seeded {len(df)} historical candles from {provider}.", "success")
    except Exception as exc:
        logger.warning("Failed to seed historical candles: %s", exc)
        append_log("Historical seed unavailable. Waiting for live stream.", "error")


def get_simulation_config():
    return simulation_config.copy()


def get_network_config():
    return network_test_config.copy()


def set_network_config(updates):
    changed = False
    numeric_fields = {
        "latencyMs": (0.0, 4000.0),
        "jitterMs": (0.0, 2000.0),
        "packetLossPct": (0.0, 90.0),
    }

    for key, value in updates.items():
        if key == "enabled" and value is not None:
            network_test_config["enabled"] = bool(value)
            changed = True
        elif key in numeric_fields and value is not None:
            lower, upper = numeric_fields[key]
            network_test_config[key] = max(lower, min(upper, float(value)))
            changed = True

    if changed:
        latest_state["networkTest"] = network_test_config.copy()
        append_log(
            (
                "Network test profile updated: "
                f"{'enabled' if network_test_config['enabled'] else 'disabled'}, "
                f"latency={network_test_config['latencyMs']:.0f}ms, "
                f"jitter={network_test_config['jitterMs']:.0f}ms, "
                f"loss={network_test_config['packetLossPct']:.1f}%"
            ),
            "info",
        )
    return network_test_config.copy()


def start_simulation_with_budget(budget):
    return start_live_simulation({"initialEquity": budget})


def set_simulation_config(updates, replay_history=True):
    changed = False
    numeric_fields = {
        "initialEquity": (1000.0, 1_000_000_000.0),
        "minAllocationPct": (0.1, 100.0),
        "maxAllocationPct": (0.1, 100.0),
        "minConfidencePct": (1.0, 99.9),
        "feeRatePct": (0.0, 2.0),
        "minTradeNotional": (10.0, 10_000_000.0),
    }
    boolean_fields = {"allowLong", "allowShort", "ignoreFees", "dynamicThresholds"}

    for key, value in updates.items():
        if key in numeric_fields and value is not None:
            lower, upper = numeric_fields[key]
            simulation_config[key] = max(lower, min(upper, float(value)))
            changed = True
        elif key in boolean_fields and value is not None:
            simulation_config[key] = bool(value)
            changed = True

    if simulation_config["minAllocationPct"] > simulation_config["maxAllocationPct"]:
        simulation_config["minAllocationPct"], simulation_config["maxAllocationPct"] = (
            simulation_config["maxAllocationPct"],
            simulation_config["minAllocationPct"],
        )

    latest_state["simulation"] = simulation_config.copy()
    dynamic_threshold_state["enabled"] = bool(simulation_config.get("dynamicThresholds", True))
    dynamic_threshold_state["minConfidencePct"] = simulation_config["minConfidencePct"]
    dynamic_threshold_state["minTradeNotional"] = simulation_config["minTradeNotional"]
    latest_state["simulationSummary"] = simulation_summary()
    if changed and replay_history and not df.empty:
        if live_training_status.get("shadowEvaluation", {}).get("status") == "running":
            reset_shadow_evaluation("shadow evaluation reset because the simulation assumptions changed")
        seeded_features = pattern_engine.extract_features(df)
        replay_seeded_history(seeded_features)
        latest_state["candle"] = {
            "time": int(pd.Timestamp(df.iloc[-1]["Timestamp"]).timestamp()),
            "open": float(df.iloc[-1]["Open"]),
            "high": float(df.iloc[-1]["High"]),
            "low": float(df.iloc[-1]["Low"]),
            "close": float(df.iloc[-1]["Close"]),
        }
        refresh_portfolio_snapshot(float(df.iloc[-1]["Close"]))
        append_log("Simulation settings updated and paper-trading state replayed.", "success")

    return simulation_config.copy()


def start_live_simulation(updates):
    updated = set_simulation_config(updates, replay_history=False)
    reset_runtime_state()
    if not df.empty:
        latest_state["history"] = history_payload()
        latest_state["candle"] = {
            "time": int(pd.Timestamp(df.iloc[-1]["Timestamp"]).timestamp()),
            "open": float(df.iloc[-1]["Open"]),
            "high": float(df.iloc[-1]["High"]),
            "low": float(df.iloc[-1]["Low"]),
            "close": float(df.iloc[-1]["Close"]),
        }
        refresh_portfolio_snapshot(float(df.iloc[-1]["Close"]))
    append_log("Started a fresh forward-live simulation from the current market state.", "success")
    return updated


BEST_WEIGHTS_PATH = "adaptive_weights.best.pth"
WEIGHTS_PATH = "adaptive_weights.pth"
reset_shadow_evaluation()
weights_to_load = BEST_WEIGHTS_PATH if os.path.exists(BEST_WEIGHTS_PATH) else WEIGHTS_PATH
if os.path.exists(weights_to_load):
    logger.info("Loading trained AI weights from %s...", weights_to_load)
    load_active_model_from_path(weights_to_load)
    ensure_model_history_seeded()
    restore_champion_from_history()
else:
    logger.info("No trained weights found. Running with untrained baseline model.")
    with ai_model_lock:
        ai_model.eval()
sync_live_training_state()


async def ingest_kraken_data():
    """Connect to Kraken v2 WebSocket for real-time 1-minute BTC/USD OHLC data.

    Kraken is used instead of Binance because Binance blocks cloud-server IPs.
    The message format differs but the downstream candle processing is identical.
    """
    global df
    if df.empty:
        await asyncio.to_thread(bootstrap_historical_data)

    ws_url = "wss://ws.kraken.com/v2"
    logger.info("Connecting to Kraken 1m OHLC stream for BTC/USD...")

    async for websocket in websockets.connect(ws_url, ping_interval=20, ping_timeout=20):
        try:
            await websocket.send(json.dumps({
                "method": "subscribe",
                "params": {"channel": "ohlc", "symbol": ["BTC/USD"], "interval": 1},
            }))

            last_closed_open_ms = None

            async for raw_message in websocket:
                receive_timestamp_ms = int(time.time() * 1000)
                try:
                    message = json.loads(raw_message)
                except Exception:
                    continue

                if message.get("channel") != "ohlc":
                    continue
                if message.get("type") not in ("snapshot", "update"):
                    continue

                items = message.get("data", [])
                if not items:
                    continue

                # Always update the live display with the latest item in this batch.
                live_item = items[-1]
                live_begin_s = pd.Timestamp(live_item["interval_begin"]).timestamp()
                live_close = float(live_item["close"])
                latest_state["candle"] = {
                    "time": int(live_begin_s),
                    "open": float(live_item["open"]),
                    "high": float(live_item["high"]),
                    "low": float(live_item["low"]),
                    "close": live_close,
                }
                refresh_portfolio_snapshot(live_close)

                # Process every closed candle in this batch (snapshot has many; update has one).
                for kline in items:
                    kline_begin_s = pd.Timestamp(kline["interval_begin"]).timestamp()
                    kline_open_ms = int(kline_begin_s * 1000)

                    if time.time() < kline_begin_s + 60:
                        continue  # Candle still open
                    if kline_open_ms == last_closed_open_ms:
                        continue  # Already processed this close
                    if _row_exists_for_open_ms(kline_open_ms):
                        last_closed_open_ms = kline_open_ms
                        continue  # Already in dataframe

                    last_closed_open_ms = kline_open_ms
                    tick_timestamp_ms = kline_open_ms + CANDLE_INTERVAL_MS - 1

                    publish_live_closed_candle(
                        open_ms=kline_open_ms,
                        open_price=float(kline["open"]),
                        high_price=float(kline["high"]),
                        low_price=float(kline["low"]),
                        close_price=float(kline["close"]),
                        volume=float(kline.get("volume", 0.0)),
                        receive_timestamp_ms=receive_timestamp_ms,
                    )
                    network_test_stats["receivedClosedCandles"] += 1
                    applied_delay_ms = 0.0

                    if network_test_config["enabled"]:
                        base_latency = network_test_config["latencyMs"]
                        jitter = network_test_config["jitterMs"]
                        if jitter > 0:
                            applied_delay_ms = max(0.0, base_latency + random.uniform(-jitter, jitter))
                        else:
                            applied_delay_ms = max(0.0, base_latency)

                        packet_loss_chance = network_test_config["packetLossPct"] / 100.0
                        if packet_loss_chance > 0 and random.random() < packet_loss_chance:
                            network_test_stats["droppedClosedCandles"] += 1
                            network_test_stats["lastAppliedDelayMs"] = applied_delay_ms
                            latest_state["networkTest"] = network_test_config.copy()
                            backfilled = fill_missing_candles_until(kline_open_ms, receive_timestamp_ms)
                            dropped_entry = _insert_synthetic_candle(kline_open_ms, receive_timestamp_ms, gap_minutes=2.0)
                            if dropped_entry:
                                backfilled.append(dropped_entry)
                            for bfitem in backfilled:
                                process_latest_candle_state(bfitem["tick_timestamp_ms"], bfitem["receive_timestamp_ms"])
                            append_log(
                                f"Simulated packet loss dropped candle "
                                f"({network_test_stats['droppedClosedCandles']}/{network_test_stats['receivedClosedCandles']}) "
                                "and inserted imputed candle.",
                                "error",
                            )
                            continue

                        if applied_delay_ms > 0:
                            await asyncio.sleep(applied_delay_ms / 1000.0)

                    network_test_stats["lastAppliedDelayMs"] = applied_delay_ms
                    network_test_stats["processedClosedCandles"] += 1
                    backfilled = fill_missing_candles_until(kline_open_ms, receive_timestamp_ms)
                    for bfitem in backfilled:
                        process_latest_candle_state(bfitem["tick_timestamp_ms"], bfitem["receive_timestamp_ms"])

                    row = _build_candle_row(
                        open_ms=kline_open_ms,
                        open_price=float(kline["open"]),
                        high_price=float(kline["high"]),
                        low_price=float(kline["low"]),
                        close_price=float(kline["close"]),
                        volume=float(kline.get("volume", 0.0)),
                        receive_timestamp_ms=receive_timestamp_ms,
                        is_imputed=False,
                        gap_minutes=0.0,
                    )
                    _append_row_dict(row)
                    network_test_stats["consecutiveSyntheticCandles"] = 0
                    process_latest_candle_state(tick_timestamp_ms, receive_timestamp_ms)

                    log_market_table_row(
                        pd.Timestamp(row["Timestamp"]).strftime("%Y-%m-%d %H:%M"),
                        float(kline["close"]),
                        latest_state["prediction"]["action"],
                        latest_state["prediction"]["confidence"],
                    )

        except websockets.ConnectionClosed:
            logger.warning("Kraken connection closed. Reconnecting...")
            append_log("Kraken stream disconnected. Reconnecting...", "error")
            continue
        except Exception as exc:
            logger.error("Kraken ingestion error: %s", exc)
            append_log(f"Kraken backend error: {exc}", "error")
            await asyncio.sleep(2)


async def ingest_binance_data(symbol="btcusdt"):
    global df
    if df.empty:
        await asyncio.to_thread(bootstrap_historical_data, symbol.upper())

    ws_url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_1m"
    logger.info("Connecting to Binance 1m kline stream for %s...", symbol.upper())

    async for websocket in websockets.connect(ws_url, ping_interval=20, ping_timeout=20):
        try:
            async for message in websocket:
                receive_timestamp_ms = int(time.time() * 1000)
                data = json.loads(message)
                kline = data["k"]
                tick_timestamp_ms = int(kline["T"])

                current_candle = {
                    "time": int(kline["t"] / 1000),
                    "open": float(kline["o"]),
                    "high": float(kline["h"]),
                    "low": float(kline["l"]),
                    "close": float(kline["c"]),
                }
                latest_state["candle"] = current_candle
                refresh_portfolio_snapshot(current_candle["close"])

                if not kline["x"]:
                    continue

                publish_live_closed_candle(
                    open_ms=int(kline["t"]),
                    open_price=float(kline["o"]),
                    high_price=float(kline["h"]),
                    low_price=float(kline["l"]),
                    close_price=float(kline["c"]),
                    volume=float(kline["v"]),
                    receive_timestamp_ms=receive_timestamp_ms,
                )
                network_test_stats["receivedClosedCandles"] += 1
                applied_delay_ms = 0.0
                candle_open_ms = int(kline["t"])
                if network_test_config["enabled"]:
                    base_latency = network_test_config["latencyMs"]
                    jitter = network_test_config["jitterMs"]
                    if jitter > 0:
                        applied_delay_ms = max(0.0, base_latency + random.uniform(-jitter, jitter))
                    else:
                        applied_delay_ms = max(0.0, base_latency)

                    packet_loss_chance = network_test_config["packetLossPct"] / 100.0
                    if packet_loss_chance > 0 and random.random() < packet_loss_chance:
                        network_test_stats["droppedClosedCandles"] += 1
                        network_test_stats["lastAppliedDelayMs"] = applied_delay_ms
                        latest_state["networkTest"] = network_test_config.copy()
                        backfilled = fill_missing_candles_until(candle_open_ms, receive_timestamp_ms)
                        dropped_entry = _insert_synthetic_candle(
                            candle_open_ms,
                            receive_timestamp_ms,
                            gap_minutes=2.0,
                        )
                        if dropped_entry:
                            backfilled.append(dropped_entry)
                        for item in backfilled:
                            process_latest_candle_state(item["tick_timestamp_ms"], item["receive_timestamp_ms"])
                        append_log(
                            (
                                f"Simulated packet loss dropped candle "
                                f"({network_test_stats['droppedClosedCandles']}/{network_test_stats['receivedClosedCandles']}) "
                                "and inserted imputed candle."
                            ),
                            "error",
                        )
                        continue

                    if applied_delay_ms > 0:
                        await asyncio.sleep(applied_delay_ms / 1000.0)

                network_test_stats["lastAppliedDelayMs"] = applied_delay_ms
                network_test_stats["processedClosedCandles"] += 1
                backfilled = fill_missing_candles_until(candle_open_ms, receive_timestamp_ms)
                for item in backfilled:
                    process_latest_candle_state(item["tick_timestamp_ms"], item["receive_timestamp_ms"])

                row = _build_candle_row(
                    open_ms=candle_open_ms,
                    open_price=float(kline["o"]),
                    high_price=float(kline["h"]),
                    low_price=float(kline["l"]),
                    close_price=float(kline["c"]),
                    volume=float(kline["v"]),
                    receive_timestamp_ms=receive_timestamp_ms,
                    is_imputed=False,
                    gap_minutes=0.0,
                )
                _append_row_dict(row)
                network_test_stats["consecutiveSyntheticCandles"] = 0
                process_latest_candle_state(tick_timestamp_ms, receive_timestamp_ms)

                log_market_table_row(
                    pd.Timestamp(row["Timestamp"]).strftime("%Y-%m-%d %H:%M"),
                    current_candle["close"],
                    latest_state["prediction"]["action"],
                    latest_state["prediction"]["confidence"],
                )

        except websockets.ConnectionClosed:
            logger.warning("Binance connection closed. Reconnecting...")
            append_log("Market stream disconnected. Reconnecting...", "error")
            continue
        except Exception as exc:
            logger.error("Ingestion error: %s", exc)
            append_log(f"Backend error: {exc}", "error")
            await asyncio.sleep(2)


def get_latest_state():
    latest_state["training"] = load_training_report()
    latest_state["simulation"] = simulation_config.copy()
    latest_state["networkTest"] = network_test_config.copy()
    latest_state["simulationSummary"] = simulation_summary()
    sync_live_training_state()
    return latest_state


def get_model_history():
    return load_model_history()


if __name__ == "__main__":
    try:
        asyncio.run(ingest_binance_data())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
