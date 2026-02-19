# HedgeMe - ML-Powered Crypto Trading Bot

An automated trading system using LSTM neural networks for volume prediction on cryptocurrency markets.

## Features

- **LSTM Volume Prediction**: Predicts volume acceleration to identify trading opportunities
- **Multi-Signal Strategy**: Combines traditional indicators with ML predictions
- **Bybit Integration**: Real-time trading via ccxt
- **Automated Training**: Daily model retraining via GitHub Actions
- **Interactive Reports**: HTML reports with buy/sell signal visualization

## Supported Assets

- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- SOLUSDT (Solana)

## Project Structure

```
hedgeme/
├── app/                    # Main application
│   ├── config/             # Trading configuration
│   ├── data/               # Data fetching (Bybit)
│   ├── features/           # Feature engineering
│   ├── ml/                 # LSTM model & signals
│   ├── strategies/         # Trading strategies
│   └── reporting/          # HTML reports
├── scripts/                # Pipeline scripts
├── tests/                  # Test suite
├── models/                 # Trained models
├── docs/                   # Documentation
└── .github/workflows/      # CI/CD pipelines
```

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/hedgeme.git
cd hedgeme
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your Bybit API keys

# 3. Run full pipeline (fetch, train, backtest)
python scripts/run_pipeline.py --ticker BTCUSDT --days 30 --epochs 50
```

### Manual Steps (alternative)

```bash
# Fetch data
python -m app.data.fetcher --ticker "BTCUSDT" --timeframe "15" --limit 200

# Calculate features
python -m app.features.calculator --input ".tmp/market_data/BTCUSDT_15m.csv"

# Train model
python -m app.ml.lstm --features ".tmp/features/BTCUSDT_15m_features.csv" --epochs 50

# Run backtest
python -m app.strategies.backtest
```

## Running Tests

```bash
pytest tests/ -v
```

## Documentation

- [Setup Guide](docs/SETUP.md)
- [Integration Guide](docs/INTEGRATION.md)

## License

MIT
