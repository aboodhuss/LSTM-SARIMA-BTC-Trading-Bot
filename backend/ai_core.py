"""
ai_core.py – Central AI model definition and inference logic.

This module defines the PyTorch LSTM neural network that powers the trading
bot's BUY / SELL / HOLD predictions.  It also contains the custom adaptive
loss function used during offline and live training.

Data source context
-------------------
The model is trained on BTC/USDT 1-minute OHLCV candles.  Live data comes
from the Binance WebSocket feed.  Offline calibration of the SARIMA
pre-processor uses the following Kaggle datasets:

  • Minute-level BTC data:
      https://www.kaggle.com/datasets/swaptr/bitcoin-historical-data

  • Daily BTC price data (2020–2026):
      https://www.kaggle.com/datasets/hasanyiitakbulut/bitcoin-btc-historical-price-data-2020-2026

See also: backend/sarima_preprocessor.py for how the Kaggle data feeds into
the SARIMA missing-candle reconstruction step.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature columns that the LSTM expects as input.
# These are produced by feature_engineering.PatternRecognizer.extract_features()
# and must stay in sync with that pipeline.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Open",               # Opening price of the 1-min candle
    "High",               # Highest price during the candle
    "Low",                # Lowest price during the candle
    "Close",              # Closing price of the candle
    "Volume",             # Traded volume in that minute
    "Bullish_FVG",        # True if a bullish Fair Value Gap was detected
    "Bearish_FVG",        # True if a bearish Fair Value Gap was detected
    "HH",                 # True if this candle marks a Higher High
    "LL",                 # True if this candle marks a Lower Low
    "Confluence_Score",   # Weighted composite of pattern strength (-1 to +1)
    "Return_1",           # 1-candle percentage return, quality-adjusted
]


class AdaptiveTradingLSTM(nn.Module):
    """
    A compact LSTM classifier that reads a sliding window of engineered
    features and outputs a probability distribution over three actions:

        Index 0 → HOLD   (do nothing)
        Index 1 → BUY    (go long)
        Index 2 → SELL   (go short)

    Architecture
    ------------
    • One LSTM layer  (input_size → hidden_layer_size)
    • One Linear head (hidden_layer_size → 3 output logits)

    The model is intentionally small so that:
      1. It trains quickly on a single laptop CPU/GPU.
      2. It can be fully explained during a viva or tutor demo.
      3. It is easy to instrument for network-degradation experiments.
    """

    def __init__(self, input_size, hidden_layer_size=50, output_size=3):
        """
        Parameters
        ----------
        input_size : int
            Number of features per time-step (len(FEATURE_COLUMNS) == 11).
        hidden_layer_size : int
            Dimensionality of the LSTM hidden state.  50 is a reasonable
            default that balances capacity and training speed.
        output_size : int
            Number of action classes (always 3: HOLD, BUY, SELL).
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        """
        Forward pass.

        Parameters
        ----------
        input_seq : Tensor of shape (batch, seq_len, input_size)
            The most recent `seq_len` rows of normalised feature data.

        Returns
        -------
        Tensor of shape (batch, 3)
            Raw logits for [HOLD, BUY, SELL].  Apply softmax externally
            if you need probabilities.
        """
        lstm_out, _ = self.lstm(input_seq)
        # We only care about the output at the LAST time-step because the
        # trading decision applies to the current moment, not past candles.
        last_timestep_out = lstm_out[:, -1, :]
        return self.linear(last_timestep_out)


def predict_action(model, df_features: pd.DataFrame):
    """
    Run a single forward pass on the most recent 10 candles and return the
    model's recommendation.

    Parameters
    ----------
    model : AdaptiveTradingLSTM
        A loaded (and eval-mode) LSTM model.
    df_features : pd.DataFrame
        The full engineered-feature DataFrame.  Only the last 10 rows are
        actually used.

    Returns
    -------
    (action: str, confidence: float)
        action is one of "HOLD", "BUY", "SELL".
        confidence is the softmax probability of the chosen action (0–1).

    Notes
    -----
    • If fewer than 10 candles are available, the function defaults to
      ("HOLD", 0.0) because the LSTM needs a full sequence window.
    • Price columns are normalised relative to the latest Close so the model
      sees ratio-scale inputs rather than raw dollar values.
    """
    if len(df_features) < 10:
        return "HOLD", 0.0

    # --- Prepare the 10-step input sequence ---
    seq_data = df_features[FEATURE_COLUMNS].iloc[-10:].copy()

    # Boolean pattern flags must be cast to float for the tensor.
    bool_columns = ["Bullish_FVG", "Bearish_FVG", "HH", "LL"]
    for column in bool_columns:
        seq_data[column] = seq_data[column].astype(float)

    # Normalise OHLC relative to the latest close so the model is
    # price-level-invariant (it sees ratios ≈1.0 instead of raw $90k etc.).
    close_reference = seq_data["Close"].iloc[-1] or 1.0
    seq_data["Open"] = seq_data["Open"] / close_reference
    seq_data["High"] = seq_data["High"] / close_reference
    seq_data["Low"] = seq_data["Low"] / close_reference
    seq_data["Close"] = seq_data["Close"] / close_reference

    # Normalise volume to [0, 1] within the window.
    seq_data["Volume"] = seq_data["Volume"] / (seq_data["Volume"].max() + 1e-9)

    # Clip derived features to keep inputs within reasonable bounds.
    seq_data["Confluence_Score"] = seq_data["Confluence_Score"].clip(-1.0, 1.0)
    seq_data["Return_1"] = seq_data["Return_1"].clip(-0.05, 0.05)

    # Convert to a (1, 10, 11) tensor and run inference.
    tensor_data = torch.tensor(seq_data.values, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor_data)
        output = torch.softmax(logits, dim=1)

    # Guard against NaN / Inf outputs that can occur on degenerate input.
    probabilities = np.nan_to_num(output[0].numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(probabilities.sum())
    if total <= 0:
        return "HOLD", 0.0
    probabilities = probabilities / total

    action_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[action_idx])

    actions = ["HOLD", "BUY", "SELL"]
    return actions[action_idx], confidence


def adaptive_loss_function(prediction, actual_outcome, pattern_confidence, base_loss_fn=nn.CrossEntropyLoss()):
    """
    Custom loss that penalises *confident wrong predictions* more heavily.

    The idea: if the feature pipeline is highly confident in a pattern
    (e.g. a clear FVG + HH confluence) but the model still gets it wrong,
    the penalty should be amplified.  This nudges the network to be cautious
    when patterns are ambiguous and decisive when patterns are strong.

    Parameters
    ----------
    prediction : Tensor (batch, 3)
        Raw logits from the LSTM.
    actual_outcome : Tensor (batch,)
        Ground-truth class labels (0=HOLD, 1=BUY, 2=SELL).
    pattern_confidence : Tensor (batch,)
        Per-sample confidence score derived from the feature pipeline.
    base_loss_fn : nn.Module
        The underlying classification loss (default: cross-entropy).

    Returns
    -------
    Tensor
        Scalar (or per-sample) loss value.  Higher values → stronger
        gradient signal for the optimiser.
    """
    standard_loss = base_loss_fn(prediction, actual_outcome)
    # Scale the loss by (1 + pattern_confidence).
    # When pattern_confidence ≈ 0 the penalty is just standard_loss.
    # When pattern_confidence ≈ 1 the penalty doubles.
    adaptive_penalty = standard_loss * (1.0 + pattern_confidence)
    return adaptive_penalty
