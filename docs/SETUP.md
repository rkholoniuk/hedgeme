# Setup Guide

## Prerequisites

- **Python**: 3.12
- **OS**: macOS, Linux, or WSL2 (Windows)
- **System dependencies**: ta-lib

## Quick Setup

### 1. Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### 2. Install System Dependencies

**macOS:**
```bash
brew install ta-lib
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

**Windows (WSL2):**
Use Linux instructions above

### 3. Install Python Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**If you get ta-lib errors:**
```bash
# Try installing without ta-lib first
pip install --no-deps TA-Lib
# Or skip it - the 'ta' library provides similar functionality as fallback
```

### 4. Configure Environment

```bash
# Copy example
cp .env.example .env

# Edit with your API keys
nano .env
```

**Required variables:**
```bash
BYBIT_API_KEY=your_key_here
BYBIT_API_SECRET=your_secret_here
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### 5. Test Installation

```bash
# Test data fetching
python -m app.data.fetcher --ticker "BTCUSDT" --timeframe "15" --limit 10

# Should output:
# ✓ Successfully fetched 10 candles
```

## Common Issues

### Issue: "No module named 'talib'"

**Solution 1**: Use the `ta` library fallback (already in requirements)
```python
# In calculator.py, it falls back to the 'ta' library automatically
```

**Solution 2**: Install system ta-lib first
```bash
# macOS
brew install ta-lib

# Then reinstall Python package
pip install TA-Lib
```

### Issue: Python version conflicts

**Solution**: Use Python 3.11 or 3.12
```bash
# Check version
python --version

# If wrong version, create venv with specific version
python3.11 -m venv venv
source venv/bin/activate
```

### Issue: "tensorflow not found on M1/M2 Mac"

**Solution**: Use tensorflow-macos
```bash
pip install tensorflow-macos tensorflow-metal
```

## Verify Setup

```bash
# Run full pipeline test
python -m app.data.fetcher --ticker "BTCUSDT" --timeframe "15" --limit 200
python -m app.features.calculator --input ".tmp/market_data/BTCUSDT_15m.csv"
python -m app.ml.lstm --features ".tmp/features/BTCUSDT_15m_features.csv" --epochs 10

# If all succeed, you're ready!
```

## Next Steps

1. **Get Bybit API keys**: [Bybit.com](https://www.bybit.com) → API Management
2. **Run optimization**: `python -m app.strategies.optimizer`
3. **Backtest**: `python -m app.strategies.backtest`
4. **Deploy to GitHub Actions**: See [INTEGRATION.md](INTEGRATION.md)
