"""
hybrid_models.py – SARIMA model fitting and LSTM+SARIMA signal blending.

This module provides three capabilities:

  1. **SARIMA order search** – Grid-searches ARIMA(p,d,q) × seasonal(P,D,Q,s)
     configurations to find the best-fitting model by AIC.

  2. **SARIMA signal generation** – Converts a SARIMA forecast into the same
     action/confidence format used by the LSTM, so both can be compared or
     blended on equal footing.

  3. **Signal blending** – Combines the LSTM and SARIMA signals into a single
     hybrid recommendation.  The hybrid path is the project's main
     "robustness argument": the statistical model anchors short-horizon
     structure while the neural model captures nonlinear context.

References
----------
  • statsmodels SARIMAX API:
      https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
  • statsmodels SARIMAX example:
      https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


# The three possible trading actions, shared with ai_core and the UI.
ACTIONS = ("HOLD", "BUY", "SELL")


# ---------------------------------------------------------------------------
# Data classes for structured SARIMA outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SarimaCandidate:
    """Stores the best SARIMA configuration found during the grid search."""
    order: tuple[int, int, int]              # ARIMA (p, d, q)
    seasonal_order: tuple[int, int, int, int] # Seasonal (P, D, Q, s)
    aic: float                                # Akaike Information Criterion
    bic: float                                # Bayesian Information Criterion


@dataclass(frozen=True)
class SarimaSignal:
    """A single SARIMA recommendation, analogous to the LSTM's output."""
    action: str              # "BUY", "SELL", or "HOLD"
    confidence: float        # 0–1 confidence in the action
    expected_move_pct: float # Forecast price change as a percentage
    forecast_price: float    # The SARIMA-predicted close price


# ---------------------------------------------------------------------------
# Internal: lazy import of statsmodels (optional heavy dependency)
# ---------------------------------------------------------------------------

def _statsmodels():
    """Import statsmodels lazily so the rest of the codebase can load
    without it installed (e.g. for frontend-only development)."""
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for the SARIMA hybrid path. "
            "Install it with `pip install statsmodels`."
        ) from exc
    return sm


# ---------------------------------------------------------------------------
# SARIMA parameter grids
# ---------------------------------------------------------------------------

def default_order_grid() -> list[tuple[int, int, int]]:
    """
    ARIMA (p, d, q) candidates to try during the grid search.
    We keep this small to avoid excessive fitting time on 1-minute data.
    """
    return [
        (1, 0, 0),  # AR(1) – simple autoregressive
        (0, 1, 1),  # IMA(1,1) – integrated moving average
        (1, 1, 0),  # ARI(1,1) – autoregressive integrated
        (1, 1, 1),  # ARIMA(1,1,1) – general baseline
        (2, 1, 0),  # AR(2) integrated – captures slightly longer memory
    ]


def default_seasonal_grid(periods: Iterable[int] = (5, 15, 60)) -> list[tuple[int, int, int, int]]:
    """
    Seasonal (P, D, Q, s) candidates.

    The period `s` controls the seasonality length:
      • s=5   → 5-minute micro-cycles
      • s=15  → 15-minute quarter-hour patterns
      • s=60  → hourly seasonality

    A (0,0,0,0) entry is always included to test the non-seasonal case.
    """
    grid = [(0, 0, 0, 0)]  # No seasonality
    for period in periods:
        grid.extend(
            [
                (1, 0, 0, int(period)),  # Seasonal AR
                (0, 1, 1, int(period)),  # Seasonal IMA
                (1, 0, 1, int(period)),  # Seasonal ARMA
            ]
        )
    return grid


# ---------------------------------------------------------------------------
# Close-price series builder
# ---------------------------------------------------------------------------

def build_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Convert a DataFrame with 'Close' and 'Timestamp' columns into a
    DatetimeIndex-backed Series suitable for statsmodels SARIMAX.
    """
    series = pd.Series(
        df["Close"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(df["Timestamp"])),
        name="Close",
    )
    return series


# ---------------------------------------------------------------------------
# SARIMA fitting (grid search)
# ---------------------------------------------------------------------------

def fit_best_sarima(
    close_series: pd.Series,
    order_grid: Sequence[tuple[int, int, int]] | None = None,
    seasonal_grid: Sequence[tuple[int, int, int, int]] | None = None,
):
    """
    Fit every combination in (order_grid × seasonal_grid) and return the
    model with the lowest AIC.

    Returns
    -------
    (result, candidate)
        result     – The fitted SARIMAXResultsWrapper from statsmodels.
        candidate  – A SarimaCandidate recording the chosen hyper-parameters.

    Raises
    ------
    RuntimeError
        If no model in the grid converges.
    """
    sm = _statsmodels()
    order_grid = list(order_grid or default_order_grid())
    seasonal_grid = list(seasonal_grid or default_seasonal_grid())

    best_result = None
    best_candidate = None
    failures: list[str] = []

    for order in order_grid:
        for seasonal_order in seasonal_grid:
            try:
                model = sm.tsa.statespace.SARIMAX(
                    close_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend="c",                    # Constant trend term
                    simple_differencing=False,
                    enforce_stationarity=False,    # Allow non-stationary fits
                    enforce_invertibility=False,   # Allow non-invertible MA
                )
                result = model.fit(disp=False)
                candidate = SarimaCandidate(
                    order=order,
                    seasonal_order=seasonal_order,
                    aic=float(result.aic),
                    bic=float(result.bic),
                )
                # Skip candidates with non-finite AIC (degenerate fits).
                if not math.isfinite(candidate.aic):
                    continue
                # Keep the candidate with the lowest AIC.
                if best_candidate is None or candidate.aic < best_candidate.aic:
                    best_candidate = candidate
                    best_result = result
            except Exception as exc:
                failures.append(f"{order} x {seasonal_order}: {exc}")

    if best_result is None or best_candidate is None:
        message = failures[0] if failures else "no SARIMA candidate could be fit"
        raise RuntimeError(f"Unable to fit any SARIMA model: {message}")

    return best_result, best_candidate


# ---------------------------------------------------------------------------
# SARIMA forecast → trading signal
# ---------------------------------------------------------------------------

def sarima_signal_from_forecast(
    last_close: float,
    forecast_price: float,
    trade_gate_pct: float = 0.05,
) -> SarimaSignal:
    """
    Convert a raw SARIMA price forecast into a trading signal.

    Parameters
    ----------
    last_close : float
        The latest observed close price.
    forecast_price : float
        The SARIMA-predicted future close price.
    trade_gate_pct : float
        Minimum expected move (%) to trigger a BUY or SELL instead of HOLD.
        This acts as a noise filter to prevent over-trading on tiny moves.

    Returns
    -------
    SarimaSignal
        Action, confidence, expected move %, and forecast price.
    """
    expected_move_pct = ((forecast_price - last_close) / max(last_close, 1e-9)) * 100.0
    magnitude = abs(expected_move_pct)

    # Bullish forecast above the gate → BUY
    if expected_move_pct >= trade_gate_pct:
        confidence = min(0.99, 0.52 + (magnitude / max(trade_gate_pct * 5.0, 0.20)))
        return SarimaSignal("BUY", float(confidence), float(expected_move_pct), float(forecast_price))

    # Bearish forecast below the gate → SELL
    if expected_move_pct <= -trade_gate_pct:
        confidence = min(0.99, 0.52 + (magnitude / max(trade_gate_pct * 5.0, 0.20)))
        return SarimaSignal("SELL", float(confidence), float(expected_move_pct), float(forecast_price))

    # Forecast too small to trade → HOLD
    hold_confidence = max(0.40, 0.92 - (magnitude / max(trade_gate_pct * 4.0, 0.20)))
    return SarimaSignal("HOLD", float(min(0.95, hold_confidence)), float(expected_move_pct), float(forecast_price))


# ---------------------------------------------------------------------------
# LSTM + SARIMA signal blender
# ---------------------------------------------------------------------------

def blend_signals(
    lstm_action: str,
    lstm_confidence: float,
    sarima_action: str,
    sarima_confidence: float,
    *,
    lstm_weight: float = 0.65,
    sarima_weight: float = 0.35,
    hold_band: float = 0.18,
) -> tuple[str, float]:
    """
    Blend the LSTM and SARIMA signals into a single hybrid recommendation.

    The blending logic works as follows:

    1. Convert each action into a signed score:
         BUY = +confidence,  SELL = -confidence,  HOLD = 0

    2. Compute a weighted sum:
         signed_score = lstm_weight × lstm_signed + sarima_weight × sarima_signed

    3. Apply agreement / disagreement adjustments:
       • If both models agree on a directional trade → boost the score (+10%)
       • If they disagree (one BUY, one SELL) → dampen the score (× 0.65)

    4. If |signed_score| < hold_band → output HOLD (not enough conviction).
       Otherwise output BUY (positive) or SELL (negative).

    Parameters
    ----------
    lstm_weight : float
        Weight given to the LSTM signal.  Default 0.65 (primary model).
    sarima_weight : float
        Weight given to the SARIMA signal.  Default 0.35 (supporting model).
    hold_band : float
        Dead-zone around zero.  Scores with |value| < hold_band → HOLD.

    Returns
    -------
    (action, confidence)
        The merged recommendation in the same format as predict_action().
    """
    # Map actions to signed directions.
    direction_map = {"SELL": -1.0, "HOLD": 0.0, "BUY": 1.0}
    lstm_signed = direction_map[lstm_action] * float(lstm_confidence)
    sarima_signed = direction_map[sarima_action] * float(sarima_confidence)
    signed_score = (lstm_weight * lstm_signed) + (sarima_weight * sarima_signed)

    # Agreement bonus: both models agree on a directional trade.
    if lstm_action == sarima_action and lstm_action != "HOLD":
        signed_score += 0.10 * direction_map[lstm_action] * min(lstm_confidence, sarima_confidence)
    # Disagreement penalty: one says BUY and the other says SELL.
    elif lstm_action != "HOLD" and sarima_action != "HOLD" and lstm_action != sarima_action:
        signed_score *= 0.65

    # Dead zone: if the blended score is too small, default to HOLD.
    if abs(signed_score) < hold_band:
        hold_confidence = max(
            0.45,
            (lstm_weight * (1.0 - abs(lstm_signed))) + (sarima_weight * (1.0 - abs(sarima_signed))),
        )
        return "HOLD", float(min(0.95, hold_confidence))

    # Final directional output.
    action = "BUY" if signed_score > 0 else "SELL"
    confidence = max(abs(signed_score), 0.35)
    return action, float(min(0.99, confidence))
