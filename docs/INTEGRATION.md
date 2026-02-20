# Integration Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Actions                          │
├─────────────────────────────────────────────────────────────┤
│  Daily Training (4 AM UTC)                                  │
│  ├── app.strategies.optimizer (find best params)           │
│  ├── app.data.fetcher (get Bybit data)                     │
│  ├── app.features.calculator (engineer features)           │
│  ├── app.ml.lstm (train model)                             │
│  └── Cache model for trading bot                           │
├─────────────────────────────────────────────────────────────┤
│  Trading Bot (Every 15 min)                                 │
│  ├── Load cached LSTM model                                 │
│  ├── app.data.fetcher (latest candles)                     │
│  ├── app.features.calculator (current features)            │
│  ├── app.ml.signals (generate signals)                     │
│  └── Execute via LumiBot + Bybit                           │
└─────────────────────────────────────────────────────────────┘
```

## GitHub Actions Workflows

### 1. CI - Tests & Linting

**File**: `.github/workflows/ci.yml`

**Steps**:
1. Run pytest unit tests
2. Generate coverage report
3. Run Ruff linter
4. Upload test reports as artifacts

**Triggers**:
- On push to main/develop
- On pull requests to main
- Manual: workflow_dispatch

### 2. Daily Training (4:00 AM UTC)

**File**: `.github/workflows/lstm_training.yml`

**Steps**:
1. Run parameter optimization (`app.strategies.optimizer`)
2. Fetch training data from Bybit
3. Train LSTM model with best parameters
4. Run backtest and generate HTML report
5. Cache model + best_configs.json
6. Send Telegram notification

**Triggers**:
- Scheduled: Daily at 4 AM UTC
- Manual: workflow_dispatch

### 3. Trading Bot (Every 15 Minutes)

**File**: `.github/workflows/trading_bot.yml`

**Steps**:
1. Restore cached LSTM model
2. Load best configuration
3. Run trading iteration (`app.strategies.live`)
4. Upload logs
5. Send notifications on failure

**Triggers**:
- Scheduled: Every 15 minutes
- Manual: workflow_dispatch

### 4. Backtest & Generate Reports (On Demand)

**File**: `.github/workflows/backtest.yml`

**Steps**:
1. Fetch historical data from Bybit
2. Run backtest with specified parameters
3. Generate interactive HTML report with buy/sell signals
4. Upload report as downloadable artifact
5. Send Telegram notification with link

**Triggers**:
- Manual: workflow_dispatch (with inputs for ticker, dates, strategy)

## Environment Variables (GitHub Secrets)

```bash
# Required
BYBIT_API_KEY           # Bybit API key
BYBIT_API_SECRET        # Bybit API secret
TELEGRAM_BOT_TOKEN      # Telegram bot token
TELEGRAM_CHAT_ID        # Telegram chat ID

# Optional
TICKER                  # Trading pair (default: BTCUSDT) - Supported: BTCUSDT, ETHUSDT, SOLUSDT
TIMEFRAME              # Candle interval (default: 15)
```

## Quick Start

### 1. Run Optimization

```bash
python -m app.strategies.optimizer
```

### 2. Train LSTM

```bash
python -m app.ml.lstm \
  --features ".tmp/features/BTCUSDT_15m_features.csv" \
  --epochs 50
```

### 3. Test Live Trading

```bash
python -m app.strategies.live
```

### 4. Generate HTML Report

```bash
# Run backtest with JSON output
python -m app.strategies.backtest_ml \
  --ticker "BTC-USD" \
  --start "2026-01-01" \
  --end "2026-02-14" \
  --output-json ".tmp/backtest_results.json"

# Generate HTML report
python -m app.reporting.html \
  --backtest-results ".tmp/backtest_results.json" \
  --output "reports/backtest_report.html" \
  --title "My Backtest Report"
```

### 5. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app

# Run only unit tests (skip integration)
pytest tests/ -v -m "not integration"
```

---

**Last updated**: 2026-02-15
**System version**: v2.0
