"""
train.py – Offline training pipeline for the Adaptive LSTM model.

This script downloads historical BTC/USDT 1-minute candles from the Binance
REST API, engineers SMC features via feature_engineering.PatternRecognizer,
and trains the AdaptiveTradingLSTM defined in ai_core.py.

Key design decisions
--------------------
  • Labels are generated from *future* price movement over a short horizon
    (default 3 candles), net of a round-trip fee estimate.
  • The custom adaptive_loss_function penalises confident-but-wrong
    predictions more heavily to discourage overconfident signals.
  • Class weights are computed per-cycle to counteract the typical HOLD
    majority in the label distribution.
  • Trade feedback from the live simulation (trade_feedback.jsonl) is
    optionally incorporated to refine pattern confidence scores.

Data sources
------------
  • Live / historical candles:  Binance REST API  (BTCUSDT 1m klines)
  • SARIMA calibration data  :  Kaggle minute-level BTC dataset
      https://www.kaggle.com/datasets/swaptr/bitcoin-historical-data
  • Kaggle daily BTC prices  :
      https://www.kaggle.com/datasets/hasanyiitakbulut/bitcoin-btc-historical-price-data-2020-2026

Usage
-----
    cd backend
    python train.py --symbol BTCUSDT --loops 12 --epochs 120

Output files
------------
  • adaptive_weights.pth       – Latest model checkpoint
  • adaptive_weights.best.pth  – Best validation-loss checkpoint
  • training_report.json       – Training metrics summary
"""

import argparse
import shutil
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

from ai_core import AdaptiveTradingLSTM, FEATURE_COLUMNS, adaptive_loss_function
from feature_engineering import PatternRecognizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Training_Pipeline")

MODEL_PATH = Path("adaptive_weights.pth")
BEST_MODEL_PATH = Path("adaptive_weights.best.pth")
REPORT_PATH = Path("training_report.json")
TRADE_FEEDBACK_PATH = Path("trade_feedback.jsonl")


def fetch_historical_data(symbol="BTCUSDT", interval="1m", limit=1000, loops=12, pause_seconds=0.15):
    """
    Fetches multiple batches of historical klines from the Binance REST API.
    """
    logger.info("Fetching up to %s candles of %s %s data...", limit * loops, symbol, interval)
    all_klines = []
    end_time = None

    for loop_idx in range(loops):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        if end_time is not None:
            url += f"&endTime={end_time}"

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data:
            logger.info("No more data returned after %s loops.", loop_idx)
            break

        all_klines = data + all_klines
        end_time = data[0][0] - 1
        time.sleep(pause_seconds)

    df = pd.DataFrame(
        all_klines,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close_Time",
            "Quote_Asset_Volume",
            "Trades",
            "TBB",
            "TBQ",
            "Ignore",
        ],
    )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = df[column].astype(float)

    df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
    df = df.drop_duplicates(subset=["Timestamp"], keep="last").sort_values("Timestamp").reset_index(drop=True)
    return df


def load_trade_feedback(feedback_path=TRADE_FEEDBACK_PATH):
    if not feedback_path or not Path(feedback_path).exists():
        return {}

    feedback_by_timestamp = {}
    with Path(feedback_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            candle_time = entry.get("openedCandleTime")
            if candle_time is None:
                continue
            feedback_by_timestamp.setdefault(int(candle_time), []).append(entry)

    return feedback_by_timestamp


def prepare_tensors(df_features, seq_length=20, forecast_horizon=3, trade_feedback=None):
    """
    Converts the features DataFrame into tensors for sequence classification.
    """
    logger.info("Preparing tensors from %s engineered candles...", len(df_features))

    df_norm = df_features.copy()
    for column in ["Bullish_FVG", "Bearish_FVG", "HH", "LL"]:
        df_norm[column] = df_norm[column].astype(float)

    close_reference = df_norm["Close"].replace(0, pd.NA).ffill().bfill()
    df_norm["Open"] = (df_norm["Open"] / close_reference).fillna(0.0)
    df_norm["High"] = (df_norm["High"] / close_reference).fillna(0.0)
    df_norm["Low"] = (df_norm["Low"] / close_reference).fillna(0.0)
    df_norm["Close"] = (df_norm["Close"] / close_reference).fillna(0.0)
    df_norm["Volume"] = df_norm["Volume"] / (df_norm["Volume"].rolling(window=50, min_periods=1).max() + 1e-9)
    df_norm["Confluence_Score"] = df_norm["Confluence_Score"].clip(-1.0, 1.0)
    df_norm["Return_1"] = df_norm["Return_1"].clip(-0.05, 0.05)

    X = []
    y = []
    pattern_confidences = []
    feedback_hits = 0
    round_trip_cost = 0.20
    edge_buffer = 0.05

    for i in range(seq_length, len(df_norm) - forecast_horizon):
        seq = df_norm[FEATURE_COLUMNS].iloc[i - seq_length:i].values
        current_close = float(df_features["Close"].iloc[i - 1])
        current_timestamp = int(pd.Timestamp(df_features["Timestamp"].iloc[i - 1]).timestamp())
        future_window = df_features.iloc[i : i + forecast_horizon]
        future_high = float(future_window["High"].max())
        future_low = float(future_window["Low"].min())

        price_upside = ((future_high - current_close) / max(current_close, 1e-9)) * 100.0
        price_downside = ((current_close - future_low) / max(current_close, 1e-9)) * 100.0
        net_upside = price_upside - round_trip_cost
        net_downside = price_downside - round_trip_cost

        action = 0
        if net_upside >= edge_buffer and net_upside > net_downside:
            action = 1
        elif net_downside >= edge_buffer and net_downside > net_upside:
            action = 2

        X.append(seq)
        y.append(action)

        last_row = df_features.iloc[i - 1]
        pattern_strength = abs(float(last_row.get("Confluence_Score", 0.0)))
        has_pattern = any(
            bool(last_row.get(name, False))
            for name in ["Bullish_FVG", "Bearish_FVG", "HH", "LL", "HL", "LH"]
        )
        projected_edge = max(net_upside, net_downside, 0.0)
        confidence = min(1.0, 0.15 + pattern_strength + (0.2 if has_pattern else 0.0) + min(0.4, projected_edge / 2.0))

        for feedback in (trade_feedback or {}).get(current_timestamp, []):
            feedback_hits += 1
            action_label = int(feedback.get("actionLabel", 0))
            profitable = bool(feedback.get("profitable", False))
            confidence += 0.18 if profitable and action == action_label and action != 0 else 0.0
            confidence -= 0.18 if (not profitable and action == action_label and action != 0) else 0.0
            confidence += 0.08 if (not profitable and action != action_label and action != 0) else 0.0

        pattern_confidences.append(min(1.25, max(0.05, confidence)))

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    confidence_tensor = torch.tensor(pattern_confidences, dtype=torch.float32)
    return X_tensor, y_tensor, confidence_tensor, feedback_hits


def compute_class_weights(y_tensor):
    counts = Counter(y_tensor.tolist())
    total = sum(counts.values())
    num_classes = 3
    weights = []
    for class_idx in range(num_classes):
        count = counts.get(class_idx, 1)
        weights.append(total / (num_classes * count))

    # Keep HOLD underweighted relative to directional classes so the model does not collapse to HOLD.
    weights[0] *= 0.65
    return torch.tensor(weights, dtype=torch.float32), counts


def split_dataset(X, y, pattern_confidences, validation_ratio=0.2):
    if len(X) < 100:
        raise ValueError("Not enough training samples generated for a meaningful run.")

    split_index = max(1, int(len(X) * (1 - validation_ratio)))
    train_data = (X[:split_index], y[:split_index], pattern_confidences[:split_index])
    val_data = (X[split_index:], y[split_index:], pattern_confidences[split_index:])
    return train_data, val_data


def evaluate(model, X, y, pattern_confidences, batch_size, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch_X = X[start : start + batch_size].to(device)
            batch_y = y[start : start + batch_size].to(device)
            batch_cf = pattern_confidences[start : start + batch_size].to(device)

            logits = model(batch_X)
            batch_loss = adaptive_loss_function(logits, batch_y, batch_cf, criterion)
            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            batch_count = len(batch_X)
            total_loss += float(batch_loss.mean().item()) * batch_count
            total_samples += batch_count
            correct += int((predictions == batch_y).sum().item())

    return total_loss / max(total_samples, 1), correct / max(total_samples, 1)


def save_report(report):
    REPORT_PATH.write_text(json.dumps(report, indent=2))


def train_adaptive_model(
    symbol="BTCUSDT",
    loops=12,
    epochs=120,
    batch_size=128,
    learning_rate=0.001,
    seq_length=20,
    forecast_horizon=3,
    cycles=1,
    device=None,
    model_path=MODEL_PATH,
    best_model_path=BEST_MODEL_PATH,
    report_path=REPORT_PATH,
    initial_weights_path=None,
    feedback_path=TRADE_FEEDBACK_PATH,
    promote_best_to_model=True,
):
    logger.info("Starting Adaptive AI Training Pipeline")
    runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", runtime_device)

    model_path = Path(model_path)
    best_model_path = Path(best_model_path)
    report_path = Path(report_path)
    initial_weights_path = Path(initial_weights_path) if initial_weights_path else model_path
    feedback_path = Path(feedback_path) if feedback_path else None

    best_val_loss = math.inf
    last_report = {}

    for cycle in range(1, cycles + 1):
        logger.info("Cycle %s/%s", cycle, cycles)
        df_raw = fetch_historical_data(symbol=symbol, loops=loops)
        logger.info("Downloaded %s candles.", len(df_raw))

        engine = PatternRecognizer()
        df_features = engine.extract_features(df_raw)
        trade_feedback = load_trade_feedback(feedback_path)
        X, y, pattern_confidences, feedback_hits = prepare_tensors(
            df_features,
            seq_length=seq_length,
            forecast_horizon=forecast_horizon,
            trade_feedback=trade_feedback,
        )
        logger.info("Generated %s training sequences.", len(X))
        class_weights, class_counts = compute_class_weights(y)
        logger.info("Label distribution: %s", dict(class_counts))
        logger.info("Matched %s trade-feedback samples into this training set.", feedback_hits)

        (train_X, train_y, train_cf), (val_X, val_y, val_cf) = split_dataset(X, y, pattern_confidences)

        model = AdaptiveTradingLSTM(input_size=len(FEATURE_COLUMNS)).to(runtime_device)
        if initial_weights_path.exists():
            model.load_state_dict(torch.load(initial_weights_path, map_location=runtime_device))
            logger.info("Loaded existing weights from %s", initial_weights_path)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(reduction="none", weight=class_weights.to(runtime_device))

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_samples = 0

            permutation = torch.randperm(len(train_X))
            train_X = train_X[permutation]
            train_y = train_y[permutation]
            train_cf = train_cf[permutation]

            for start in range(0, len(train_X), batch_size):
                batch_X = train_X[start : start + batch_size].to(runtime_device)
                batch_y = train_y[start : start + batch_size].to(runtime_device)
                batch_cf = train_cf[start : start + batch_size].to(runtime_device)

                optimizer.zero_grad()
                logits = model(batch_X)
                batch_loss = adaptive_loss_function(logits, batch_y, batch_cf, criterion)
                loss = batch_loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_count = len(batch_X)
                total_loss += float(loss.item()) * batch_count
                total_samples += batch_count

            train_loss = total_loss / max(total_samples, 1)
            val_loss, val_accuracy = evaluate(
                model, val_X, val_y, val_cf, batch_size=batch_size, criterion=criterion, device=runtime_device
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                logger.info(
                    "Epoch %s/%s improved validation loss to %.6f with accuracy %.4f",
                    epoch,
                    epochs,
                    val_loss,
                    val_accuracy,
                )

            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                logger.info(
                    "Epoch %s/%s | train_loss=%.6f | val_loss=%.6f | val_acc=%.4f",
                    epoch,
                    epochs,
                    train_loss,
                    val_loss,
                    val_accuracy,
                )

            torch.save(model.state_dict(), model_path)
            last_report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "cycles_completed": cycle,
                "epochs_per_cycle": epochs,
                "train_sequences": int(len(train_X)),
                "validation_sequences": int(len(val_X)),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "best_val_loss": best_val_loss,
                "device": runtime_device,
                "label_counts": {str(k): int(v) for k, v in class_counts.items()},
                "feedback_matches": int(feedback_hits),
                "model_path": str(model_path.resolve()),
                "best_model_path": str(best_model_path.resolve()),
            }
            report_path.write_text(json.dumps(last_report, indent=2))

    logger.info("Training complete. Final report written to %s", report_path)
    if promote_best_to_model and best_model_path.exists():
        shutil.copyfile(best_model_path, model_path)
        logger.info("Promoted best checkpoint to %s", model_path)
    return last_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train the adaptive trading model.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--loops", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seq-length", type=int, default=20)
    parser.add_argument("--forecast-horizon", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_adaptive_model(
        symbol=args.symbol,
        loops=args.loops,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seq_length=args.seq_length,
        forecast_horizon=args.forecast_horizon,
        cycles=args.cycles,
    )
