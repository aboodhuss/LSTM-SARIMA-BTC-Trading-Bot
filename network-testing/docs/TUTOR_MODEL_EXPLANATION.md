# AI Trading Bot Model Explanation (Tutor Brief)

## 1) Are we building from scratch?

Yes, mostly.

- The model architecture is a custom PyTorch LSTM classifier (`HOLD`, `BUY`, `SELL`).
- We are not using a pre-trained financial foundation model.
- This is acceptable for this assignment because the main project objective is **system + network behavior evaluation**, not state-of-the-art trading alpha.

## 2) Should we have used an existing model instead?

For this project scope, building a small custom model is reasonable.

- Pros for assignment: easier to explain, fully controllable, easier to instrument for network testing.
- Cons: lower predictive sophistication than advanced finance models.

Recommended framing to tutor:

> We intentionally used a compact custom model because the assignment focus is internetworking impact on a real-time AI pipeline. A simpler model improves observability and reproducibility under latency/loss tests.

## 3) What data is fed into the model?

Data source:

- Binance BTCUSDT 1-minute OHLCV candles (Open, High, Low, Close, Volume).

Feature-engineered inputs (11 columns):

1. Open
2. High
3. Low
4. Close
5. Volume
6. Bullish_FVG
7. Bearish_FVG
8. HH (Higher High)
9. LL (Lower Low)
10. Confluence_Score
11. Return_1

So each model input is a **time sequence of engineered candle features**, not raw tick text and not a static CSV file.

## 4) Is this using a CSV with 200+ rows?

Not by default.

- Training data is fetched live from Binance REST into a pandas DataFrame.
- Inference data is streamed from Binance WebSocket.
- You can export to CSV, but the pipeline itself is DataFrame + JSON log based.

## 5) How many entries are used?

From current training report:

- Train sequences: **6381**
- Validation sequences: **1596**
- Total sequences: **7977**

Default training data request can pull up to:

- `1000 candles x 12 loops = 12,000 candles` (1-minute candles, ~8.3 days max).

In your recorded run, effective usable data was about ~8,000 candles (after sequencing rules).

## 6) What horizon is predicted?

Training label is based on a short future window:

- Forecast horizon: **3 candles** (3 minutes).
- Output classes:
  - `0 = HOLD`
  - `1 = BUY`
  - `2 = SELL`

So this is a **short-horizon directional classifier**.

## 7) What is the model input shape?

Conceptually:

- `batch_size x sequence_length x feature_count`
- Training default sequence length: **20**
- Feature count: **11**
- Example single sample: `1 x 20 x 11`

Note:

- Live inference currently uses last **10** steps.
- Training uses **20** steps.
- This mismatch should be documented as a limitation.

## 8) Why seasonality is not central here

Your tutor concern is valid: seasonality is not the core of this assignment.

- This project is focused on:
  1. real-time ingestion,
  2. feature extraction,
  3. AI decision latency,
  4. behavior under network degradation.

Seasonality can be a future model-improvement topic, but not required to justify current system design.

## 9) What to say in viva/presentation (short script)

> Our model is a custom LSTM classifier in PyTorch for BTCUSDT 1-minute candles.  
> We engineer 11 market-structure features including FVG and HH/LL signals, then classify each short horizon into HOLD, BUY, or SELL.  
> In the latest run, we trained on 6,381 sequences and validated on 1,596 sequences.  
> We chose a compact model intentionally because our assignment’s primary goal is to measure how network conditions affect end-to-end AI trading behavior, not to optimize a production-grade quant model.
