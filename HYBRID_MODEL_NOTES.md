# Hybrid Model Notes

## Why this SARIMA source

The best implementation source for this project is `statsmodels` rather than a Kaggle notebook.

- Official SARIMAX API: <https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html>
- Official SARIMAX example: <https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html>
- Official rolling forecast example using `append` and `extend`: <https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_forecasting.html>

For the actual system integration, the official `statsmodels` route is safer because it gives us:

- a stable Python package instead of notebook-only code,
- documented `seasonal_order` handling,
- rolling forecast updates through `append` and `extend`,
- a clean path to compare SARIMA, LSTM, and LSTM + SARIMA outputs on the same candle stream.

## What was added

- `backend/hybrid_models.py`
  - SARIMA order search
  - SARIMA signal generation
  - LSTM + SARIMA blending logic

- `backend/evaluate_hybrid_models.py`
  - loads the current trained LSTM
  - fits the best SARIMA candidate on the training slice
  - runs rolling validation across the holdout window
  - compares `LSTM`, `SARIMA`, and `LSTM + SARIMA`
  - writes:
    - `backend/hybrid_model_report.json`
    - `backend/hybrid_model_report.md`

## How to run it

From `backend/`:

```bash
pip install -r requirements.txt
python evaluate_hybrid_models.py --symbol BTCUSDT --loops 12 --forecast-horizon 3
```

## What to show on the Zoom call

Use the markdown report and focus on:

- baseline LSTM accuracy and signal rate,
- standalone SARIMA accuracy and signal rate,
- hybrid accuracy and directional precision,
- the selected SARIMA order and seasonal order,
- whether the hybrid trades less often but with cleaner signals.

## Network-behaviour talking points

These are the key hypotheses to discuss when comparing model behaviour under network constraints:

- SARIMA should be easier to explain because it reacts mostly to recent linear and seasonal structure.
- LSTM may be more flexible, but it can also be more sensitive to stale or imputed sequences because the hidden state depends on recent order and continuity.
- A hybrid can be framed as a robustness play: the statistical model anchors short-horizon structure while the neural model captures nonlinear context.
