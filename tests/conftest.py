"""
Shared test fixtures for the trading system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_candles = 200

    dates = pd.date_range(
        start=datetime.now() - timedelta(days=n_candles),
        periods=n_candles,
        freq='15min'
    )

    # Generate realistic price data
    base_price = 50000.0  # BTC-like price
    returns = np.random.normal(0, 0.02, n_candles)
    close_prices = base_price * np.cumprod(1 + returns)
    open_prices = close_prices * (1 + np.random.uniform(-0.005, 0.005, n_candles))

    # Ensure proper OHLC relationships: high >= max(open,close), low <= min(open,close)
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.uniform(0, 0.01, n_candles))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.uniform(0, 0.01, n_candles))
    volumes = np.random.uniform(1000000, 10000000, n_candles)

    # Use capitalized names for backtesting.py compatibility + lowercase for tools
    df = pd.DataFrame({
        'timestamp': dates,
        'Open': open_prices, 'High': high_prices, 'Low': low_prices, 'Close': close_prices, 'Volume': volumes,
        'open': open_prices, 'high': high_prices, 'low': low_prices, 'close': close_prices, 'volume': volumes
    })

    df.set_index('timestamp', inplace=True)
    df['timestamp'] = df.index  # Keep timestamp as column for FeatureCalculator
    return df


@pytest.fixture
def sample_features_data(sample_ohlcv_data):
    """Generate sample data with calculated features."""
    df = sample_ohlcv_data.copy()

    # Add volume derivatives (use capitalized Volume for backtesting compatibility)
    df['Volume_Smoothed'] = df['Volume'].rolling(window=4).mean()
    df['Vol_1st_Der'] = df['Volume_Smoothed'].diff()
    df['Vol_2nd_Der'] = df['Vol_1st_Der'].diff()

    # Normalize
    df['Vol_1st_Der_Norm'] = (df['Vol_1st_Der'] - df['Vol_1st_Der'].mean()) / (df['Vol_1st_Der'].std() + 1e-10)
    df['Vol_2nd_Der_Norm'] = (df['Vol_2nd_Der'] - df['Vol_2nd_Der'].mean()) / (df['Vol_2nd_Der'].std() + 1e-10)

    # Acceleration signals
    df['Accel_Positive'] = ((df['Vol_1st_Der_Norm'] > 0.1) & (df['Vol_2nd_Der_Norm'] > 0.1)).astype(int)
    df['Accel_Negative'] = ((df['Vol_1st_Der_Norm'] < -0.1) & (df['Vol_2nd_Der_Norm'] < -0.1)).astype(int)

    # Consecutive acceleration
    df['Consecutive_Accel'] = 0
    consec = 0
    for i in range(1, len(df)):
        if df['Accel_Positive'].iloc[i]:
            consec = max(0, consec) + 1
        elif df['Accel_Negative'].iloc[i]:
            consec = min(0, consec) - 1
        else:
            consec = 0
        df.iloc[i, df.columns.get_loc('Consecutive_Accel')] = consec

    # Technical indicators (simplified)
    df['RSI'] = 50 + np.random.uniform(-20, 20, len(df))
    df['ADX'] = 20 + np.random.uniform(0, 30, len(df))
    df['ATR'] = df['High'] - df['Low']

    return df.dropna()


@pytest.fixture
def trading_config():
    """Sample trading configuration."""
    return {
        'ACCEL_BARS_REQUIRED': 2,
        'USE_LSTM': False,  # Disable for unit tests
        'LSTM_WEIGHT': 0.5,
        'LSTM_CONFIRMATION_REQUIRED': False,
        'LSTM_LOOKBACK': 10,
        'POSITION_SIZE': 0.05,
        'STOP_LOSS_PCT': 0.02,
        'TAKE_PROFIT_PCT': 0.04,
        'LEVERAGE': 1,
        'INTERVAL_MINUTES': 15,
        'LOOKBACK_CANDLES': 200,
        'USE_TRAILING_STOP': True,
    }


@pytest.fixture
def mock_lstm_model():
    """Mock LSTM model for testing without TensorFlow."""
    class MockLSTM:
        def __init__(self):
            self.lookback = 10

        def predict(self, data):
            # Return random predictions
            return np.random.uniform(-1, 1, (len(data), 3))

        def predict_derivatives(self, features):
            """Mock prediction of volume derivatives - returns dict like real model."""
            v1_pred = np.random.uniform(-1, 1)
            v2_pred = np.random.uniform(-1, 1)
            return {
                'v1_predicted': v1_pred,
                'v2_predicted': v2_pred,
                'is_accelerating_positive': v1_pred > 0.1 and v2_pred > 0.1,
                'is_accelerating_negative': v1_pred < -0.1 and v2_pred < -0.1
            }

        def save(self, model_path, scaler_path):
            pass

        @classmethod
        def load(cls, model_path, scaler_path):
            return cls()

    return MockLSTM()


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
