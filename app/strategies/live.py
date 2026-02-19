#!/usr/bin/env python3
"""
Live Trading Integration for GitHub Actions
Loads optimized parameters and runs LumiBot strategy.

This script:
1. Loads best config from optimization
2. Trains/loads LSTM model
3. Runs live trading via LumiBot

Usage:
    python -m app.strategies.live
"""

import json
from pathlib import Path
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.strategies.lumibot_ml import MLVolumeStrategy
from app.ml.lstm import VolumeLSTM
from lumibot.brokers import Alpaca
from lumibot.traders import Trader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_optimized_config():
    """Load best configuration from optimization results"""

    config_file = Path('best_configs.json')

    if config_file.exists():
        with open(config_file, 'r') as f:
            configs = json.load(f)

        if configs:
            best_config = configs[0]  # First is best (sorted by Sharpe)
            logger.info(f"✅ Loaded optimized config (Sharpe: {best_config.get('sharpe_ratio', 'N/A')})")
            return best_config

    logger.warning("⚠️  No optimized config found, using defaults")
    return None

def sync_config_to_yaml(optimized_config):
    """Sync optimized parameters to args/trading_config.yaml"""

    import yaml

    config_path = Path('args/trading_config.yaml')

    # Load existing config
    if config_path.exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
    else:
        yaml_config = {}

    # Update with optimized parameters
    if optimized_config:
        # Map optimization params to yaml config
        param_mapping = {
            'VOLUME_SMOOTH_PERIODS': 'VOLUME_SMOOTH_PERIODS',
            'ACCEL_BARS_REQUIRED': 'ACCEL_BARS_REQUIRED',
            'ADX_THRESHOLD': 'ADX_THRESHOLD',
            'RISK_PER_TRADE': 'POSITION_SIZE',  # Map to POSITION_SIZE
            'ATR_STOP_MULTIPLIER': 'ATR_STOP_MULTIPLIER',
            'TP_POINTS': 'TP_POINTS',
            'TRAILING_START': 'TRAILING_START',
            'TRAILING_STEP': 'TRAILING_STEP',
            'MIN_CONFIRMATIONS_RATIO': 'MIN_CONFIRMATIONS_RATIO',
            'USE_ADX': 'USE_ADX',
            'USE_OBV': 'USE_OBV',
            'USE_TRAILING_STOP': 'USE_TRAILING_STOP',
        }

        for opt_key, yaml_key in param_mapping.items():
            if opt_key in optimized_config:
                yaml_config[yaml_key] = optimized_config[opt_key]

        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)

        logger.info(f"✅ Synced optimized config to {config_path}")

# ============================================================================
# LSTM TRAINING/LOADING
# ============================================================================

def ensure_lstm_model():
    """Ensure LSTM model exists, train if needed"""

    model_path = Path('models/lstm_volume_predictor.keras')

    if model_path.exists():
        logger.info(f"✅ LSTM model found at {model_path}")
        return str(model_path)

    logger.info("🔄 LSTM model not found, training new model...")

    # Train model using existing modules
    from app.data.fetcher import BybitDataFetcher
    from app.features.calculator import FeatureCalculator
    from app.ml.lstm import VolumeLSTM

    # Fetch data
    ticker = os.getenv('TICKER', 'BTCUSDT')
    fetcher = BybitDataFetcher()
    df = fetcher.fetch_ohlcv(ticker, timeframe='15', limit=1000)

    # Calculate features
    calculator = FeatureCalculator(df)
    df_features = calculator.calculate_all_features()

    # Train LSTM
    lstm = VolumeLSTM(lookback=10)
    lstm.train(df_features['volume'].values, epochs=50)

    # Save model
    scaler_path = str(model_path).replace('.keras', '.scaler.pkl')
    lstm.save(str(model_path), scaler_path)

    logger.info(f"✅ LSTM model trained and saved to {model_path}")

    return str(model_path)

# ============================================================================
# BROKER SETUP
# ============================================================================

def setup_broker():
    """Setup broker connection (Bybit via ccxt)"""

    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')

    if not api_key or not api_secret:
        raise ValueError("Missing BYBIT_API_KEY or BYBIT_API_SECRET in environment")

    # Use ccxt for Bybit integration
    import ccxt

    broker = ccxt.bybit({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',  # or 'linear' for USDT perpetuals
        }
    })

    logger.info("✅ Connected to Bybit")

    return broker

# ============================================================================
# TELEGRAM NOTIFICATIONS
# ============================================================================

def send_telegram(message):
    """Send Telegram notification"""

    try:
        import requests

        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')

        if not bot_token or not chat_id:
            logger.warning("Telegram credentials not configured")
            return

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }

        response = requests.post(url, data=data)

        if response.status_code == 200:
            logger.info("✅ Telegram notification sent")
        else:
            logger.error(f"❌ Telegram error: {response.text}")

    except Exception as e:
        logger.error(f"❌ Telegram failed: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution for GitHub Actions"""

    logger.info("="*80)
    logger.info("  LIVE TRADING WITH LSTM")
    logger.info("="*80)
    logger.info(f"  Started: {datetime.now()}")
    logger.info("="*80)

    try:
        # 1. Load optimized configuration
        optimized_config = load_optimized_config()

        if optimized_config:
            sync_config_to_yaml(optimized_config)

        # 2. Ensure LSTM model exists
        model_path = ensure_lstm_model()

        # 3. Setup broker
        broker = setup_broker()

        # 4. Create strategy
        ticker = os.getenv('TICKER', 'BTC/USD')

        strategy = MLVolumeStrategy(
            broker=broker,
            parameters={
                'ticker': ticker,
                'timeframe': '15m',
                'model_path': model_path
            }
        )

        # 5. Run strategy (single iteration for GitHub Actions)
        trader = Trader()
        trader.add_strategy(strategy)

        # For GitHub Actions: run one iteration
        logger.info("🚀 Running single trading iteration...")

        strategy.initialize(parameters={
            'ticker': ticker,
            'timeframe': '15m',
            'model_path': model_path
        })

        strategy.on_trading_iteration()

        logger.info("✅ Trading iteration complete")

        # Send notification
        send_telegram(
            f"🤖 *Trading Bot Execution*\n"
            f"Ticker: {ticker}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Status: ✅ Complete"
        )

    except Exception as e:
        logger.error(f"❌ Error: {e}")

        send_telegram(
            f"🚨 *Trading Bot Error*\n"
            f"```\n{str(e)}\n```"
        )

        raise

if __name__ == '__main__':
    main()
