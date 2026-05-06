"""
sarima_preprocessor.py – SARIMA-based missing-candle reconstruction.

This is the "SARIMA before LSTM" bridge described in the presentation:
when the Binance WebSocket drops or delays candles (or when network testing
deliberately introduces packet loss), this module fills the gaps with
SARIMA-forecasted synthetic candles so that the downstream feature
engineering and LSTM inference pipelines receive a continuous stream.

Workflow
--------
1. **Offline profiling** (run once via train_kaggle_sarima_profile.py):
   - Download BTC/USD minute-level reference data from Kaggle.
   - Fit a grid of SARIMA(p,d,q)×(P,D,Q,s) models.
   - Save the best configuration to `sarima_kaggle_profile.json`.

2. **Runtime reconstruction** (called automatically by data_ingestion.py):
   - Load the saved profile.
   - Refit the SARIMA on the most recent local candles (not the full Kaggle
     history) so the model adapts to the current regime.
   - Forecast synthetic OHLCV rows for every missing minute.
   - Tag each reconstructed row with Is_Imputed=True so downstream
     modules can down-weight them.

Kaggle datasets used
--------------------
  Minute-level BTC/USD data:
    https://www.kaggle.com/datasets/swaptr/bitcoin-historical-data
    File: data.csv (columns: Date, Symbol, Open, High, Low, Close, Volume BTC, Volume USD)

  Daily BTC price data (2020–2026):
    https://www.kaggle.com/datasets/hasanyiitakbulut/bitcoin-btc-historical-price-data-2020-2026
    File: BTC_prices.csv (columns: Date, Price)

References
----------
  statsmodels SARIMAX API:
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

# Path to the saved SARIMA profile (produced by train_kaggle_sarima_profile.py).
PROFILE_PATH = BASE_DIR / "sarima_kaggle_profile.json"

# Kaggle dataset identifiers — used by kagglehub.dataset_download().
KAGGLE_DATASET = "swaptr/bitcoin-historical-data"
KAGGLE_MINUTE_FILE = "data.csv"
KAGGLE_LATEST_DAILY_DATASET = "hasanyiitakbulut/bitcoin-btc-historical-price-data-2020-2026"
KAGGLE_LATEST_DAILY_FILE = "BTC_prices.csv"


# ---------------------------------------------------------------------------
# Lazy imports for optional heavy dependencies
# ---------------------------------------------------------------------------

def _statsmodels():
    """Import statsmodels lazily so the rest of the app can load without it."""
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for the SARIMA candle reconstruction preprocessor."
        ) from exc
    return sm


def _kagglehub():
    """Import kagglehub lazily (only needed when downloading Kaggle data)."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download the Kaggle BTC-USD dataset."
        ) from exc
    return kagglehub


# ---------------------------------------------------------------------------
# SARIMA parameter grids (for the offline profile selection)
# ---------------------------------------------------------------------------

def candidate_orders() -> list[tuple[int, int, int]]:
    """Small ARIMA (p,d,q) grid for the offline profile search."""
    return [
        (0, 1, 1),   # IMA(1,1)
        (1, 1, 1),   # ARIMA(1,1,1)
    ]


def candidate_seasonal_orders(periods: Iterable[int] = (15, 60)) -> list[tuple[int, int, int, int]]:
    """Seasonal grid.  Period 15 = quarter-hour, period 60 = hourly."""
    grid = [(0, 0, 0, 0)]  # Non-seasonal baseline
    for period in periods:
        grid.append((1, 0, 1, int(period)))
    return grid


# ---------------------------------------------------------------------------
# Kaggle dataset helpers
# ---------------------------------------------------------------------------

def ensure_kaggle_dataset(dataset: str = KAGGLE_DATASET) -> Path:
    """Download the Kaggle dataset (if not already cached) and return
    the local directory path."""
    kagglehub = _kagglehub()
    return Path(kagglehub.dataset_download(dataset))


def load_kaggle_minute_frame(dataset_dir: Path, filename: str = KAGGLE_MINUTE_FILE) -> pd.DataFrame:
    """
    Load the Kaggle BTC/USD minute-level CSV into a clean DataFrame.

    Handles two timestamp formats:
      • 'Date' column  → parsed as datetime strings
      • 'Timestamp' column → parsed as millisecond Unix timestamps

    Returns a DataFrame sorted by Timestamp with columns:
      Timestamp, Open, High, Low, Close, VolumeBTC, VolumeUSD
    """
    csv_path = dataset_dir / filename
    df = pd.read_csv(csv_path)
    # Normalise column names across CSV versions.
    df = df.rename(columns={"Symbol": "symbol", "Volume BTC": "VolumeBTC", "Volume USD": "VolumeUSD"})

    if "Date" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Date"], utc=True)
    elif "Timestamp" in df.columns:
        numeric_ts = pd.to_numeric(df["Timestamp"], errors="coerce")
        df["Timestamp"] = pd.to_datetime(numeric_ts, unit="ms", utc=True)
    else:
        raise ValueError(f"No recognizable timestamp column found in {csv_path}")

    for column in ["Open", "High", "Low", "Close", "VolumeBTC", "VolumeUSD"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Keep only BTC/USD rows and drop any with missing price data.
    df = (
        df[df.get("symbol", "BTC/USD") == "BTC/USD"]
        .dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Profile persistence
# ---------------------------------------------------------------------------

def load_profile(profile_path: Path = PROFILE_PATH) -> dict[str, object]:
    """Load a saved SARIMA profile from JSON."""
    if profile_path.exists():
        return json.loads(profile_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"SARIMA profile not found at {profile_path}")


def save_profile(profile: dict[str, object], profile_path: Path = PROFILE_PATH) -> None:
    """Persist a SARIMA profile to JSON."""
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Offline profile selection (the core Kaggle-backed calibration step)
# ---------------------------------------------------------------------------

def select_best_profile(
    minute_df: pd.DataFrame,
    *,
    sample_size: int = 480,
    seasonal_periods: Iterable[int] = (15, 60),
) -> dict[str, object]:
    """
    Choose the best SARIMA configuration from the Kaggle BTC minute data.

    This matches the presentation plan:
      1. Fit candidates on a tail slice of Kaggle history.
      2. Select by AIC.
      3. Persist the structure so the runtime pre-processor can reuse it.

    The selected profile contains:
      • source   – Provenance metadata (Kaggle dataset IDs, file paths).
      • model    – SARIMA order, seasonal order, AIC, BIC, and calibration
                   statistics (median range, median volume).
      • purpose  – Human-readable description.
    """
    sm = _statsmodels()
    candidate_frame = minute_df.tail(sample_size).copy()

    # Build a close-price Series with DatetimeIndex for SARIMAX.
    close_series = pd.Series(
        candidate_frame["Close"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(candidate_frame["Timestamp"]),
        name="Close",
    )

    best_result = None
    best_choice = None
    failures: list[str] = []

    # Grid search: try every (order × seasonal_order) combination.
    for order in candidate_orders():
        for seasonal_order in candidate_seasonal_orders(seasonal_periods):
            try:
                model = sm.tsa.statespace.SARIMAX(
                    close_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend="c",
                    simple_differencing=False,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    concentrate_scale=True,
                )
                result = model.fit(disp=False)
                aic = float(result.aic)
                bic = float(result.bic)
                if not math.isfinite(aic):
                    continue
                if best_choice is None or aic < float(best_choice["aic"]):
                    best_result = result
                    best_choice = {
                        "order": list(order),
                        "seasonal_order": list(seasonal_order),
                        "aic": aic,
                        "bic": bic,
                    }
            except Exception as exc:
                failures.append(f"{order} x {seasonal_order}: {exc}")

    if best_choice is None or best_result is None:
        message = failures[0] if failures else "no model fit succeeded"
        raise RuntimeError(f"Unable to calibrate SARIMA from Kaggle BTC-USD minute data: {message}")

    # Compute calibration statistics from the Kaggle data to use as
    # defaults during runtime candle reconstruction.
    recent_range_pct = (
        ((candidate_frame["High"] - candidate_frame["Low"]) / candidate_frame["Close"].replace(0, pd.NA))
        .dropna()
        .tail(120)
        .median()
    )
    recent_volume = candidate_frame["VolumeUSD"].dropna().tail(240).median()

    # Also download the latest daily BTC dataset for extended reference.
    daily_dir = ensure_kaggle_dataset(KAGGLE_LATEST_DAILY_DATASET)
    latest_daily = pd.read_csv(daily_dir / KAGGLE_LATEST_DAILY_FILE, sep=";")
    latest_daily.columns = ["Date", "Price"]
    latest_daily["Date"] = pd.to_datetime(latest_daily["Date"])
    latest_daily["Price"] = (
        latest_daily["Price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    latest_daily = latest_daily.sort_values("Date").reset_index(drop=True)

    return {
        "source": {
            "provider": "Kaggle",
            "minute_dataset": KAGGLE_DATASET,
            "minute_file": KAGGLE_MINUTE_FILE,
            "latest_daily_dataset": KAGGLE_LATEST_DAILY_DATASET,
            "latest_daily_file": KAGGLE_LATEST_DAILY_FILE,
            "symbol": "BTC/USD",
            "minuteDatasetPath": str((ensure_kaggle_dataset() / KAGGLE_MINUTE_FILE).resolve()),
            "latestDailyEnd": str(latest_daily["Date"].iloc[-1].date()),
            "latestDailyStart": str(latest_daily["Date"].iloc[0].date()),
        },
        "model": {
            **best_choice,
            "sample_size": int(len(candidate_frame)),
            "training_start": str(candidate_frame["Timestamp"].iloc[0]),
            "training_end": str(candidate_frame["Timestamp"].iloc[-1]),
            "lookback_candles": 240,
            "median_range_pct": float(max(0.0004, recent_range_pct if pd.notna(recent_range_pct) else 0.0015)),
            "median_volume_usd": float(max(0.0, recent_volume if pd.notna(recent_volume) else 0.0)),
        },
        "purpose": (
            "Offline-trained BTC/USD SARIMA profile used as the missing-candle reconstruction "
            "pre-processor before feature engineering and LSTM inference."
        ),
    }


# ---------------------------------------------------------------------------
# Runtime model fitting (called during live operation)
# ---------------------------------------------------------------------------

def fit_runtime_model(history_df: pd.DataFrame, profile: dict[str, object]):
    """
    Refit the SARIMA model on the most recent local candles.

    Unlike the offline profile which is trained on Kaggle data, this step
    uses the *current* live data so the pre-processor adapts to the current
    market regime rather than replaying stale historical dynamics.

    Parameters
    ----------
    history_df : DataFrame
        Recent OHLCV candle history from data_ingestion.
    profile : dict
        The saved SARIMA profile (provides the order and lookback size).

    Returns
    -------
    SARIMAXResultsWrapper
        The fitted model, ready for .forecast().
    """
    sm = _statsmodels()
    lookback = int(profile["model"].get("lookback_candles", 240))
    frame = history_df.tail(lookback).copy()

    # Build a close-price Series for SARIMAX.
    close_series = pd.Series(
        frame["Close"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(frame["Timestamp"])),
        name="Close",
    )
    model = sm.tsa.statespace.SARIMAX(
        close_series,
        order=tuple(profile["model"]["order"]),
        seasonal_order=tuple(profile["model"]["seasonal_order"]),
        trend="c",
        simple_differencing=False,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


# ---------------------------------------------------------------------------
# Missing-candle reconstruction (the "SARIMA before LSTM" bridge)
# ---------------------------------------------------------------------------

def reconstruct_missing_rows(
    history_df: pd.DataFrame,
    *,
    open_times_ms: list[int],
    receive_timestamp_ms: int,
    profile: dict[str, object],
) -> list[dict[str, float | bool | pd.Timestamp]]:
    """
    Rebuild missing BTC candles using the SARIMA model, then let the
    feature/LSTM pipeline consume the repaired stream.

    For each missing minute in `open_times_ms`, produces a synthetic
    candle with:
      • Close = SARIMA forecast (clamped to a reasonable range)
      • Open  = previous candle's close
      • High / Low = close ± a small wick derived from recent volatility
      • Volume = median USD volume from the Kaggle profile
      • Is_Imputed = True  (so downstream quality scoring knows)

    Parameters
    ----------
    history_df : DataFrame
        The candles we already have (used to fit the runtime SARIMA).
    open_times_ms : list[int]
        Unix-ms timestamps for each missing candle's open time.
    receive_timestamp_ms : int
        When we noticed the data was missing (used for lag calculation).
    profile : dict
        The SARIMA profile (order, seasonal_order, median statistics).

    Returns
    -------
    list[dict]
        One dict per reconstructed candle, ready to be appended to
        the history DataFrame.
    """
    if not open_times_ms or history_df.empty:
        return []

    # Fit the SARIMA on the recent local history.
    result = fit_runtime_model(history_df, profile)
    forecast = result.forecast(steps=len(open_times_ms))
    if not hasattr(forecast, "iloc"):
        forecast_values = list(forecast)
    else:
        forecast_values = [float(value) for value in forecast.iloc[: len(open_times_ms)]]

    # Use observed volatility to size the synthetic wicks.
    recent = history_df.tail(120).copy()
    range_pct = float(profile["model"].get("median_range_pct", 0.0015))
    if not recent.empty:
        observed_range_pct = (
            ((recent["High"] - recent["Low"]) / recent["Close"].replace(0, pd.NA)).dropna().median()
        )
        if pd.notna(observed_range_pct):
            range_pct = float(max(0.0004, min(0.02, observed_range_pct)))

    volume_usd = float(profile["model"].get("median_volume_usd", 0.0))
    last_close = float(history_df.iloc[-1]["Close"])

    rows = []
    for step, open_ms in enumerate(open_times_ms):
        raw_forecast_close = float(forecast_values[step])

        # Clamp the forecast to a reasonable band around the last close
        # to prevent wild SARIMA forecasts from corrupting the pipeline.
        max_step_move_pct = max(0.003, min(0.03, range_pct * 8.0))
        lower_bound = last_close * (1.0 - max_step_move_pct)
        upper_bound = last_close * (1.0 + max_step_move_pct)
        if not math.isfinite(raw_forecast_close):
            forecast_close = float(last_close)
        else:
            forecast_close = float(min(max(raw_forecast_close, lower_bound), upper_bound))
        forecast_close = float(max(1e-9, forecast_close))

        # Synthetic OHLC: Open = last close, Close = forecast, with
        # small wicks on both sides.
        open_price = float(last_close)
        wick_padding = max(0.0002, min(0.012, range_pct * 0.45))
        high_price = max(open_price, forecast_close) * (1.0 + wick_padding)
        low_price = min(open_price, forecast_close) * (1.0 - wick_padding)

        gap_minutes = float(max(1.0, len(open_times_ms) - step))
        close_tick_ms = int(open_ms + 60_000 - 1)
        receive_lag_ms = max(0.0, float(receive_timestamp_ms - close_tick_ms))

        rows.append(
            {
                "Timestamp": pd.to_datetime(open_ms, unit="ms"),
                "Open": open_price,
                "High": float(high_price),
                "Low": float(low_price),
                "Close": forecast_close,
                "Volume": float(max(0.0, volume_usd)),
                "Is_Imputed": True,        # Flag: this candle is synthetic
                "Gap_Minutes": gap_minutes,
                "Receive_Lag_Ms": receive_lag_ms,
            }
        )
        last_close = forecast_close

    return rows
