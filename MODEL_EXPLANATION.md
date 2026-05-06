# Plain-English Model Explanation (For Tutor)

## What this project is

This is a live AI trading simulation system with two parts:
- A backend AI engine (Python) that reads market data and makes decisions.
- A frontend dashboard (React) that shows those decisions and performance in real time.

It is not placing real-money trades. It runs as a paper-trading simulation.

## What the AI actually predicts

Every minute, the model chooses one of three actions:
- `HOLD` (do nothing)
- `BUY`
- `SELL`

It also gives a confidence score (how sure it is).

## Where the data comes from

The backend receives live BTC/USDT candle data from Binance:
- Open, High, Low, Close, Volume (OHLCV)

So the model is continuously fed current market data, not a static spreadsheet.

## How the model works (simple view)

1. Data arrives from Binance.
2. The system converts raw prices into engineered features (for example trend and structure signals such as FVG/HH/LL).
3. The LSTM model reads the recent time window of those features.
4. The model outputs `HOLD/BUY/SELL` + confidence.
5. A paper-trading rule engine uses that output to simulate position sizing, fees, PnL, and risk controls.
6. The frontend updates charts and performance cards live through a WebSocket connection.

## Why this model choice is reasonable for the assignment

The model is intentionally compact (custom LSTM) so the team can:
- explain it clearly,
- instrument it for network tests,
- show end-to-end behaviour under latency and packet loss.

So the focus is on system behaviour and networking impact, not on building a hedge-fund-grade predictor.

## What the UI shows

Main dashboard (`/`):
- live candlestick chart,
- current model action and confidence,
- simulated portfolio performance,
- network-quality and telemetry metrics.

Model lab (`/models`):
- model version timeline,
- candidate vs champion comparison,
- promotion/rejection history,
- score and PnL traces per model version.

## Key limitations to acknowledge

- It is a simulation, not a live exchange execution engine.
- Exchange feed quality and internet conditions can affect behaviour.
- A short-horizon LSTM can be useful for demonstration, but it is not guaranteed profitable in real markets.

## One-sentence summary for viva

"Our system continuously ingests live crypto candles, engineers market-structure features, uses an LSTM to classify BUY/SELL/HOLD each minute, and then measures how network conditions affect decision quality and simulated trading outcomes in real time."
