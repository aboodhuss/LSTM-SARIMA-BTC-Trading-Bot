# Slide Deck: Model + Data Pipeline (Paste Into PowerPoint)

## Slide 1 - Title

**AI Trading Bot: Model and Data Pipeline Overview**

- Course project focus: AI + internetworking evaluation
- Team objective: measure model behavior under network degradation

## Slide 2 - Why This Model

**Why we used a custom LSTM instead of a prebuilt finance model**

- Assignment priority is observability and reproducibility
- Small model is easier to instrument and explain
- Good fit for latency/loss/bandwidth experiments

## Slide 3 - Data Source

**What data enters the pipeline**

- Binance BTCUSDT 1-minute OHLCV candles
- Backend ingests via WebSocket (live) + REST (training batches)
- Stored in rolling pandas DataFrame

## Slide 4 - Feature Engineering

**Engineered features used by model**

- Market structure: HH, LL
- Fair Value Gaps: Bullish_FVG, Bearish_FVG
- Confluence_Score and Return_1
- Total feature count = 11

## Slide 5 - Model Architecture

**Model design**

- PyTorch LSTM classifier
- Output classes: HOLD, BUY, SELL
- Input shape: batch x sequence_length x 11 features
- Sequence context: 20 in training (10 currently in live inference)

## Slide 6 - Training Labels and Horizon

**What the model predicts**

- Predicts short-horizon direction (next ~3 minutes)
- Labels generated from future high/low movement with cost-aware thresholds
- Classes:
  - 0 HOLD
  - 1 BUY
  - 2 SELL

## Slide 7 - Actual Current Training Size

**Latest run metrics**

- Train sequences: 6,381
- Validation sequences: 1,596
- Total: 7,977 sequences
- Data cadence: 1-minute candles

## Slide 8 - Why This Is Enough for Assignment

**Project alignment**

- Model is a realistic, testable signal generator
- Main research question is network impact on AI decisions
- We evaluate delay, dropped candles, confidence drift, and simulated PnL

## Slide 9 - Known Limitations

**Current constraints**

- Training sequence length (20) differs from live inference (10)
- Limited macro context (no long-term seasonal features)
- Not designed as production trading alpha model

## Slide 10 - Next Improvements

**Future extensions**

- Align train/inference sequence length
- Add richer features only if required by assignment scope
- Compare custom model against baseline ML model as ablation
