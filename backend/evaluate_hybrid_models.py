"""
evaluate_hybrid_models.py – Offline comparison of LSTM, SARIMA, and LSTM+SARIMA.

This script runs a rolling validation across a holdout BTC price window to
compare three model paths:

  1. LSTM alone         – The baseline neural network prediction.
  2. SARIMA alone       – Pure statistical time-series forecast.
  3. LSTM + SARIMA      – The blended hybrid from hybrid_models.blend_signals().

It produces both a JSON report and a human-readable Markdown report.

This is separate from the *network* comparison table (model_comparison.py)
which tests LSTM and LSTM + SARIMA under simulated latency / packet loss.

References & data sources
-------------------------
  • statsmodels SARIMAX API:
      https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
  • Live BTC candles: Binance REST API (BTCUSDT 1m klines)
  • Kaggle BTC minute data:
      https://www.kaggle.com/datasets/swaptr/bitcoin-historical-data

Usage
-----
    cd backend
    python evaluate_hybrid_models.py --symbol BTCUSDT --loops 12 --forecast-horizon 3
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from statistics import fmean

import pandas as pd
import torch

from ai_core import AdaptiveTradingLSTM, FEATURE_COLUMNS, predict_action
from feature_engineering import PatternRecognizer
from hybrid_models import (
    blend_signals,
    build_close_series,
    default_order_grid,
    default_seasonal_grid,
    fit_best_sarima,
    sarima_signal_from_forecast,
)
from train import fetch_historical_data


BEST_WEIGHTS_PATH = Path("adaptive_weights.best.pth")
WEIGHTS_PATH = Path("adaptive_weights.pth")
DEFAULT_REPORT_PATH = Path("hybrid_model_report.json")
DEFAULT_MARKDOWN_PATH = Path("hybrid_model_report.md")


def load_lstm_model(weights_path: Path | None = None) -> AdaptiveTradingLSTM:
    chosen_path = weights_path or (BEST_WEIGHTS_PATH if BEST_WEIGHTS_PATH.exists() else WEIGHTS_PATH)
    if not chosen_path.exists():
        raise FileNotFoundError(
            f"No LSTM weights found at {chosen_path}. Train the current model first or point to a weights file."
        )

    model = AdaptiveTradingLSTM(input_size=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(chosen_path, map_location="cpu"))
    model.eval()
    return model


def actual_action_for_index(
    df_features: pd.DataFrame,
    index_position: int,
    forecast_horizon: int,
    round_trip_cost: float = 0.20,
    edge_buffer: float = 0.05,
) -> str:
    current_close = float(df_features["Close"].iloc[index_position - 1])
    future_window = df_features.iloc[index_position : index_position + forecast_horizon]
    future_high = float(future_window["High"].max())
    future_low = float(future_window["Low"].min())

    price_upside = ((future_high - current_close) / max(current_close, 1e-9)) * 100.0
    price_downside = ((current_close - future_low) / max(current_close, 1e-9)) * 100.0
    net_upside = price_upside - round_trip_cost
    net_downside = price_downside - round_trip_cost

    if net_upside >= edge_buffer and net_upside > net_downside:
        return "BUY"
    if net_downside >= edge_buffer and net_downside > net_upside:
        return "SELL"
    return "HOLD"


def summarize_predictions(rows: list[dict[str, object]], key_prefix: str) -> dict[str, object]:
    actions = [str(row[f"{key_prefix}_action"]) for row in rows]
    actual_actions = [str(row["actual_action"]) for row in rows]
    confidences = [float(row[f"{key_prefix}_confidence"]) for row in rows]

    correct = sum(1 for row in rows if row[f"{key_prefix}_action"] == row["actual_action"])
    predicted_directional = [row for row in rows if row[f"{key_prefix}_action"] != "HOLD"]
    actual_directional = [row for row in rows if row["actual_action"] != "HOLD"]
    directional_hits = sum(
        1
        for row in rows
        if row[f"{key_prefix}_action"] != "HOLD" and row[f"{key_prefix}_action"] == row["actual_action"]
    )
    directional_precision = directional_hits / max(1, len(predicted_directional))
    directional_recall = directional_hits / max(1, len(actual_directional))

    confusion = Counter((actual, predicted) for actual, predicted in zip(actual_actions, actions))

    return {
        "samples": len(rows),
        "accuracy": correct / max(1, len(rows)),
        "directionalPrecision": directional_precision,
        "directionalRecall": directional_recall,
        "signalRate": len(predicted_directional) / max(1, len(rows)),
        "avgConfidence": fmean(confidences) if confidences else 0.0,
        "actionCounts": dict(Counter(actions)),
        "confusion": {f"{actual}->{predicted}": count for (actual, predicted), count in sorted(confusion.items())},
    }


def build_markdown_report(report: dict[str, object]) -> str:
    dataset = report["dataset"]
    sarima = report["sarima"]
    models = report["models"]

    lines = [
        "# Hybrid Model Report",
        "",
        f"Generated: {report['generatedAt']}",
        "",
        "## Dataset",
        "",
        f"- Symbol: `{dataset['symbol']}`",
        f"- Candles fetched: `{dataset['candles']}`",
        f"- Validation samples: `{dataset['validationSamples']}`",
        f"- Sequence length: `{dataset['sequenceLength']}`",
        f"- Forecast horizon: `{dataset['forecastHorizon']}`",
        "",
        "## Best SARIMA",
        "",
        f"- Order: `{tuple(sarima['bestOrder'])}`",
        f"- Seasonal order: `{tuple(sarima['bestSeasonalOrder'])}`",
        f"- AIC: `{sarima['aic']:.4f}`",
        f"- BIC: `{sarima['bic']:.4f}`",
        "",
        "## Metrics",
        "",
        f"- LSTM accuracy: `{models['lstm']['accuracy']:.4f}` | signal rate `{models['lstm']['signalRate']:.4f}`",
        f"- SARIMA accuracy: `{models['sarima']['accuracy']:.4f}` | signal rate `{models['sarima']['signalRate']:.4f}`",
        f"- LSTM + SARIMA accuracy: `{models['hybrid']['accuracy']:.4f}` | signal rate `{models['hybrid']['signalRate']:.4f}`",
        "",
        "## Recommendation",
        "",
        report["recommendation"],
        "",
        "## Sources",
        "",
    ]
    for source in report["sources"]:
        lines.append(f"- [{source['label']}]({source['url']})")
    return "\n".join(lines)


def run_evaluation(
    *,
    symbol: str,
    loops: int,
    seq_length: int,
    forecast_horizon: int,
    validation_ratio: float,
    seasonal_periods: list[int],
    report_path: Path,
    markdown_path: Path,
) -> dict[str, object]:
    raw_df = fetch_historical_data(symbol=symbol, loops=loops)
    engine = PatternRecognizer()
    df_features = engine.extract_features(raw_df).reset_index(drop=True)

    if len(df_features) < (seq_length + forecast_horizon + 120):
        raise ValueError("Not enough candles were fetched to run the hybrid evaluation safely.")

    split_index = max(seq_length + 1, int(len(df_features) * (1.0 - validation_ratio)))
    split_index = min(split_index, len(df_features) - forecast_horizon - 1)

    lstm_model = load_lstm_model()
    train_close = build_close_series(df_features.iloc[:split_index])
    sarima_result, sarima_candidate = fit_best_sarima(
        train_close,
        order_grid=default_order_grid(),
        seasonal_grid=default_seasonal_grid(seasonal_periods),
    )

    rows: list[dict[str, object]] = []
    rolling_result = sarima_result
    full_close_series = build_close_series(df_features)

    for index_position in range(split_index, len(df_features) - forecast_horizon):
        observed_features = df_features.iloc[:index_position]
        last_close = float(df_features["Close"].iloc[index_position - 1])
        actual_action = actual_action_for_index(df_features, index_position, forecast_horizon)

        lstm_action, lstm_confidence = predict_action(lstm_model, observed_features)

        sarima_forecast = rolling_result.forecast(steps=forecast_horizon)
        forecast_price = float(sarima_forecast.iloc[-1] if hasattr(sarima_forecast, "iloc") else sarima_forecast[-1])
        sarima_signal = sarima_signal_from_forecast(last_close, forecast_price)
        hybrid_action, hybrid_confidence = blend_signals(
            lstm_action,
            lstm_confidence,
            sarima_signal.action,
            sarima_signal.confidence,
        )

        rows.append(
            {
                "timestamp": str(df_features["Timestamp"].iloc[index_position - 1]),
                "actual_action": actual_action,
                "lstm_action": lstm_action,
                "lstm_confidence": float(lstm_confidence),
                "sarima_action": sarima_signal.action,
                "sarima_confidence": float(sarima_signal.confidence),
                "hybrid_action": hybrid_action,
                "hybrid_confidence": float(hybrid_confidence),
                "sarima_forecast_price": float(sarima_signal.forecast_price),
                "sarima_expected_move_pct": float(sarima_signal.expected_move_pct),
            }
        )

        next_observation = full_close_series.iloc[index_position : index_position + 1]
        rolling_result = rolling_result.extend(next_observation)

    report = {
        "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "symbol": symbol,
            "candles": int(len(df_features)),
            "validationSamples": int(len(rows)),
            "sequenceLength": int(seq_length),
            "forecastHorizon": int(forecast_horizon),
            "validationRatio": float(validation_ratio),
        },
        "sarima": {
            "bestOrder": list(sarima_candidate.order),
            "bestSeasonalOrder": list(sarima_candidate.seasonal_order),
            "aic": float(sarima_candidate.aic),
            "bic": float(sarima_candidate.bic),
            "seasonalPeriodsTried": seasonal_periods,
        },
        "models": {
            "lstm": summarize_predictions(rows, "lstm"),
            "sarima": summarize_predictions(rows, "sarima"),
            "hybrid": summarize_predictions(rows, "hybrid"),
        },
        "recommendation": (
            "Use the hybrid path for the Zoom demo if its validation accuracy or directional precision is "
            "better than the baseline LSTM. If the hybrid only improves signal quality while trading less often, "
            "frame it as a robustness trade-off rather than a pure accuracy win."
        ),
        "sources": [
            {
                "label": "statsmodels SARIMAX API",
                "url": "https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html",
            },
            {
                "label": "statsmodels SARIMAX example",
                "url": "https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html",
            },
            {
                "label": "statsmodels forecasting notebook",
                "url": "https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_forecasting.html",
            },
        ],
        "samples": rows,
    }

    report_path.write_text(json.dumps(report, indent=2))
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the current LSTM against SARIMA and a blended hybrid.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--loops", type=int, default=12)
    parser.add_argument("--seq-length", type=int, default=20)
    parser.add_argument("--forecast-horizon", type=int, default=3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seasonal-periods", default="5,15,60")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--markdown-path", default=str(DEFAULT_MARKDOWN_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seasonal_periods = [int(item.strip()) for item in args.seasonal_periods.split(",") if item.strip()]
    report = run_evaluation(
        symbol=args.symbol,
        loops=args.loops,
        seq_length=args.seq_length,
        forecast_horizon=args.forecast_horizon,
        validation_ratio=args.validation_ratio,
        seasonal_periods=seasonal_periods,
        report_path=Path(args.report_path),
        markdown_path=Path(args.markdown_path),
    )
    print(json.dumps(report["models"], indent=2))
