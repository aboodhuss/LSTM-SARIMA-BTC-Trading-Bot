"""
feature_engineering.py – Smart Money Concept (SMC) feature extraction pipeline.

This module takes raw OHLCV candle data and engineers the features that the
LSTM and LSTM + SARIMA paths consume.  The key features are:

  • Fair Value Gaps (FVGs) – 3-candle price imbalances used in ICT/SMC theory.
  • Market structure labels  – Higher Highs (HH), Lower Lows (LL), etc.
  • Confluence Score        – A weighted composite of all pattern signals.
  • Data Quality            – A decay-based metric that penalises imputed or
                              stale candles so the models can down-weight
                              unreliable inputs.

Data source
-----------
Features are computed on BTC/USDT 1-minute candles received from Binance.
When candles are missing (due to network issues), the SARIMA pre-processor
reconstructs them using a profile calibrated on Kaggle BTC data:

  • Kaggle minute dataset : https://www.kaggle.com/datasets/swaptr/bitcoin-historical-data
  • Kaggle daily dataset  : https://www.kaggle.com/datasets/hasanyiitakbulut/bitcoin-btc-historical-price-data-2020-2026

See sarima_preprocessor.py for the reconstruction logic.
"""

import numpy as np
import pandas as pd


class PatternRecognizer:
    """
    Extracts simple market-structure features from rolling OHLCV candles.

    Usage
    -----
    >>> engine = PatternRecognizer(window_size=50)
    >>> df_with_features = engine.extract_features(raw_ohlcv_df)

    The returned DataFrame has all the original columns plus the engineered
    features listed in ai_core.FEATURE_COLUMNS.
    """

    def __init__(self, window_size=50):
        """
        Parameters
        ----------
        window_size : int
            Not used directly in feature extraction today, but reserved for
            future rolling-window calculations.
        """
        self.window_size = window_size

    # ------------------------------------------------------------------
    # 1. Fair Value Gap detection
    # ------------------------------------------------------------------
    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps (FVGs) using a 3-candle pattern.

        A *Bullish FVG* exists when candle 3's Low is above candle 1's High,
        indicating an upward price imbalance that may act as future support.

        A *Bearish FVG* exists when candle 3's High is below candle 1's Low,
        indicating a downward imbalance that may act as future resistance.

        These are core signals in Smart Money Concepts (SMC) / ICT
        methodology and are one of the main reasons we chose this approach
        for the project.
        """
        df = df.copy()
        df["Bullish_FVG"] = False
        df["Bearish_FVG"] = False

        if len(df) < 3:
            return df

        for i in range(2, len(df)):
            candle_1 = df.iloc[i - 2]   # Two candles ago
            candle_3 = df.iloc[i]        # Current candle

            # Bullish gap: current Low is above the High from 2 candles ago
            if candle_3["Low"] > candle_1["High"]:
                df.at[df.index[i], "Bullish_FVG"] = True
            # Bearish gap: current High is below the Low from 2 candles ago
            elif candle_3["High"] < candle_1["Low"]:
                df.at[df.index[i], "Bearish_FVG"] = True

        return df

    # ------------------------------------------------------------------
    # 2. Market structure (swing highs / swing lows)
    # ------------------------------------------------------------------
    def detect_market_structure(self, df: pd.DataFrame, lookback=5) -> pd.DataFrame:
        """
        Label each candle with market-structure flags:

          HH = Higher High  (bullish continuation)
          HL = Higher Low   (bullish pullback)
          LL = Lower Low    (bearish continuation)
          LH = Lower High   (bearish pullback)

        The algorithm finds local maxima / minima using a centered rolling
        window of `lookback` candles, then compares consecutive swing
        points to classify the structure.
        """
        df = df.copy()
        df["HH"] = False
        df["LL"] = False
        df["HL"] = False
        df["LH"] = False

        # Identify local swing points via a centered rolling max / min.
        rolling_high = df["High"].rolling(window=lookback, center=True).max()
        rolling_low = df["Low"].rolling(window=lookback, center=True).min()
        df["Local_Max"] = df["High"] == rolling_high
        df["Local_Min"] = df["Low"] == rolling_low

        previous_max = None  # Track the last swing high
        previous_min = None  # Track the last swing low

        for i in range(len(df)):
            # Compare consecutive swing highs
            if df["Local_Max"].iloc[i]:
                current_high = df["High"].iloc[i]
                if previous_max is not None:
                    df.at[df.index[i], "HH"] = current_high > previous_max
                    df.at[df.index[i], "LH"] = current_high < previous_max
                previous_max = current_high

            # Compare consecutive swing lows
            if df["Local_Min"].iloc[i]:
                current_low = df["Low"].iloc[i]
                if previous_min is not None:
                    df.at[df.index[i], "HL"] = current_low > previous_min
                    df.at[df.index[i], "LL"] = current_low < previous_min
                previous_min = current_low

        return df

    # ------------------------------------------------------------------
    # 3. Derived / secondary features
    # ------------------------------------------------------------------
    def add_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute secondary features on top of the raw OHLCV and pattern flags.

        Added columns:
          Return_1         – Quality-adjusted 1-candle return
          Range_Pct        – Candle range as a fraction of close
          SMA_20           – 20-period simple moving average of close
          Trend_Bias       – Distance from SMA_20, quality-adjusted
          Volume_Z         – Volume z-score over 20 candles
          Confluence_Score – Weighted sum of all pattern signals (-1 to +1)
          Volatility_20    – 20-period return std, inflated for imputed data
          Data_Quality     – 0-to-1 score penalising imputed / stale candles

        Data quality
        ------------
        When candles are imputed by the SARIMA pre-processor (due to dropped
        or delayed data), the Data_Quality score decays.  This is important
        because the network-testing module deliberately introduces latency and
        packet loss, and we need the AI to know its inputs may be unreliable.
        """
        df = df.copy()

        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        # Is_Imputed flag: 1.0 for SARIMA-reconstructed candles, 0.0 for real.
        imputed = (
            df["Is_Imputed"].astype(float)
            if "Is_Imputed" in df.columns
            else pd.Series(0.0, index=df.index, dtype=float)
        )
        # Gap_Minutes: how many minutes of data were missing before this candle.
        gap_minutes = (
            df["Gap_Minutes"].astype(float).clip(lower=0.0)
            if "Gap_Minutes" in df.columns
            else pd.Series(0.0, index=df.index, dtype=float)
        )
        # Receive_Lag_Ms: how late the candle arrived from Binance.
        receive_lag_ms = (
            df["Receive_Lag_Ms"].astype(float).clip(lower=0.0)
            if "Receive_Lag_Ms" in df.columns
            else pd.Series(0.0, index=df.index, dtype=float)
        )

        # --- Data Quality calculation ---
        # Missing or stale candles naturally degrade feature reliability.
        # We model this as an exponential decay so that quality drops
        # quickly with gaps but recovers once real data resumes.
        stale_minutes = (receive_lag_ms / 60_000.0).clip(lower=0.0, upper=10.0)
        quality = (
            np.exp(-0.22 * gap_minutes)       # Penalty for gaps
            * np.exp(-0.08 * stale_minutes)    # Penalty for staleness
            * (1.0 - 0.55 * imputed)           # Penalty for imputed candles
        )
        quality = pd.Series(np.clip(quality, 0.05, 1.0), index=df.index)

        # If many recent candles are imputed, reduce quality further.
        recent_imputed = imputed.rolling(window=6, min_periods=1).mean().fillna(0.0)
        effective_quality = pd.Series(
            np.clip(quality * (1.0 - 0.70 * recent_imputed), 0.02, 1.0),
            index=df.index,
        )

        # --- Suppress pattern flags on imputed candles ---
        # Patterns detected on synthetic data are unreliable.
        if "Bullish_FVG" in df.columns:
            df.loc[imputed > 0.5, "Bullish_FVG"] = False
        if "Bearish_FVG" in df.columns:
            df.loc[imputed > 0.5, "Bearish_FVG"] = False
        for col in ["HH", "HL", "LL", "LH"]:
            if col in df.columns:
                df.loc[imputed > 0.5, col] = False

        # --- Return_1: quality-adjusted 1-candle percentage return ---
        raw_return = close.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df["Return_1"] = raw_return * (0.35 + 0.65 * effective_quality)

        # --- Range_Pct: intrabar range as a fraction of close price ---
        df["Range_Pct"] = ((high - low) / close.replace(0, np.nan)).fillna(0.0)

        # --- SMA_20: simple moving average used for trend detection ---
        df["SMA_20"] = close.rolling(window=20, min_periods=1).mean()

        # --- Trend_Bias: how far the price is from the SMA, quality-weighted ---
        df["Trend_Bias"] = ((close - df["SMA_20"]) / df["SMA_20"].replace(0, np.nan)).fillna(0.0)
        df["Trend_Bias"] = df["Trend_Bias"] * (0.45 + 0.55 * effective_quality)

        # --- Volume_Z: volume z-score to identify unusual activity ---
        df["Volume_Z"] = (
            (volume - volume.rolling(window=20, min_periods=5).mean())
            / volume.rolling(window=20, min_periods=5).std().replace(0, np.nan)
        ).fillna(0.0)

        # --- Confluence Score: weighted blend of all pattern signals ---
        # Positive values indicate bullish confluence, negative → bearish.
        confluence_score = np.zeros(len(df), dtype=float)
        confluence_score += df["Bullish_FVG"].astype(float) * 0.4     # Bullish FVG → +0.4
        confluence_score += df["Bearish_FVG"].astype(float) * -0.4    # Bearish FVG → -0.4
        confluence_score += df["HH"].astype(float) * 0.3              # Higher High → +0.3
        confluence_score += df["HL"].astype(float) * 0.2              # Higher Low  → +0.2
        confluence_score += df["LL"].astype(float) * -0.3             # Lower Low   → -0.3
        confluence_score += df["LH"].astype(float) * -0.2             # Lower High  → -0.2
        confluence_score += np.clip(df["Trend_Bias"], -0.5, 0.5) * 0.2  # Trend bias

        # Dampen confluence when data quality is poor.
        adjusted_confluence = confluence_score * effective_quality.to_numpy()
        imputed_ratio = imputed.rolling(window=20, min_periods=1).mean().fillna(0.0)
        base_volatility = raw_return.rolling(window=20, min_periods=5).std().fillna(0.0)

        df["Confluence_Score"] = np.clip(adjusted_confluence, -1.0, 1.0)

        # --- Volatility_20: inflated when imputed data is present ---
        df["Volatility_20"] = (
            base_volatility * (1.0 + 0.95 * imputed_ratio + 0.60 * recent_imputed)
        ).fillna(0.0)

        df["Data_Quality"] = effective_quality.astype(float)

        return df

    # ------------------------------------------------------------------
    # 4. Public entry point
    # ------------------------------------------------------------------
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature-extraction pipeline.

        Runs FVG detection → market structure → derived metrics in sequence.
        The output DataFrame is ready to be consumed by ai_core.predict_action()
        or used to build training labels in train.py.
        """
        df = self.detect_fvg(df)
        df = self.detect_market_structure(df)
        df = self.add_derived_metrics(df)
        return df
