#!/usr/bin/env python3
"""
Calculate volume derivatives and technical indicators for trading bot.

This tool engineers features from raw OHLCV data:
- Volume derivatives (v1: velocity, v2: acceleration)
- Traditional indicators (RSI, ADX, OBV)
- Consecutive acceleration counting

Usage:
    python -m app.features.calculator --input ".tmp/market_data/BTCUSDT_15m.csv"
    python -m app.features.calculator --input data.csv --output features.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Tuple

# Try importing ta-lib, fall back to pandas-ta
try:
    import talib
    USE_TALIB = True
except ImportError:
    import pandas_ta as ta
    USE_TALIB = False
    logging.warning("ta-lib not found, using pandas-ta (slower). Install: brew install ta-lib && pip install TA-Lib")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureCalculator:
    """Calculate volume derivatives and technical indicators."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV data.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        self.df = df.copy()
        self._validate_input()

    def _validate_input(self) -> None:
        """Validate input DataFrame has required columns."""
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in self.df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if self.df.empty:
            raise ValueError("DataFrame is empty")

        logger.info(f"Loaded {len(self.df)} candles")

    def calculate_volume_derivatives(self) -> pd.DataFrame:
        """
        Calculate first and second derivatives of volume.

        Returns:
            DataFrame with added columns: [v1, v2, v1_normalized, v2_normalized]
        """
        logger.info("Calculating volume derivatives...")

        df = self.df.copy()

        # First derivative (velocity): change in volume
        df['v1'] = df['volume'].diff()

        # Second derivative (acceleration): change in velocity
        df['v2'] = df['v1'].diff()

        # Normalize derivatives using rolling statistics
        # This makes patterns comparable across different absolute volume levels
        window = 20  # 20 periods for normalization

        # Calculate rolling mean and std for v1
        v1_rolling_mean = df['v1'].rolling(window=window, min_periods=1).mean()
        v1_rolling_std = df['v1'].rolling(window=window, min_periods=1).std()

        # Calculate rolling mean and std for v2
        v2_rolling_mean = df['v2'].rolling(window=window, min_periods=1).mean()
        v2_rolling_std = df['v2'].rolling(window=window, min_periods=1).std()

        # Normalized derivatives (z-score)
        df['v1_normalized'] = (df['v1'] - v1_rolling_mean) / (v1_rolling_std + 1e-8)
        df['v2_normalized'] = (df['v2'] - v2_rolling_mean) / (v2_rolling_std + 1e-8)

        # Fill NaN values (from diff and rolling operations)
        df['v1'] = df['v1'].fillna(0)
        df['v2'] = df['v2'].fillna(0)
        df['v1_normalized'] = df['v1_normalized'].fillna(0)
        df['v2_normalized'] = df['v2_normalized'].fillna(0)

        logger.info("Volume derivatives calculated")
        return df

    def calculate_consecutive_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Count consecutive candles with positive/negative acceleration.

        This is the core traditional signal: if 2+ consecutive candles
        have positive v2, it's a buy signal.

        Args:
            df: DataFrame with v2 column

        Returns:
            DataFrame with added column: [Consecutive_Accel]
        """
        logger.info("Calculating consecutive acceleration...")

        # Determine if acceleration is positive or negative
        df['accel_sign'] = np.sign(df['v2'])

        # Count consecutive occurrences
        consecutive = []
        count = 0
        prev_sign = 0

        for sign in df['accel_sign']:
            if sign == prev_sign and sign != 0:
                count += 1
            else:
                count = 1 if sign != 0 else 0

            consecutive.append(count if sign > 0 else -count if sign < 0 else 0)
            prev_sign = sign

        df['Consecutive_Accel'] = consecutive

        logger.info("Consecutive acceleration calculated")
        return df

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            df: DataFrame with close prices
            period: RSI period (default: 14)

        Returns:
            DataFrame with added column: [RSI]
        """
        logger.info(f"Calculating RSI (period={period})...")

        if USE_TALIB:
            df['RSI'] = talib.RSI(df['close'], timeperiod=period)
        else:
            rsi_indicator = ta.rsi(df['close'], length=period)
            df['RSI'] = rsi_indicator

        df['RSI'] = df['RSI'].fillna(50)  # Neutral value for NaN

        logger.info("RSI calculated")
        return df

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX).

        ADX measures trend strength (not direction).
        Values > 25 indicate strong trend.

        Args:
            df: DataFrame with high, low, close
            period: ADX period (default: 14)

        Returns:
            DataFrame with added column: [ADX]
        """
        logger.info(f"Calculating ADX (period={period})...")

        if USE_TALIB:
            df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)
        else:
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=period)
            df['ADX'] = adx_data[f'ADX_{period}'] if f'ADX_{period}' in adx_data.columns else adx_data.iloc[:, -1]

        df['ADX'] = df['ADX'].fillna(0)

        logger.info("ADX calculated")
        return df

    def calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate On-Balance Volume (OBV).

        OBV is a cumulative indicator that adds/subtracts volume
        based on price direction.

        Args:
            df: DataFrame with close and volume

        Returns:
            DataFrame with added column: [OBV]
        """
        logger.info("Calculating OBV...")

        if USE_TALIB:
            df['OBV'] = talib.OBV(df['close'], df['volume'])
        else:
            obv_indicator = ta.obv(df['close'], df['volume'])
            df['OBV'] = obv_indicator

        df['OBV'] = df['OBV'].fillna(0)

        logger.info("OBV calculated")
        return df

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).

        ATR measures volatility and is used for:
        - Stop loss placement
        - Position sizing adjustments

        Args:
            df: DataFrame with high, low, close
            period: ATR period (default: 14)

        Returns:
            DataFrame with added column: [ATR]
        """
        logger.info(f"Calculating ATR (period={period})...")

        if USE_TALIB:
            df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        else:
            atr_indicator = ta.atr(df['high'], df['low'], df['close'], length=period)
            df['ATR'] = atr_indicator

        df['ATR'] = df['ATR'].fillna(df['ATR'].mean())

        logger.info("ATR calculated")
        return df

    def calculate_all_features(self) -> pd.DataFrame:
        """
        Calculate all features: derivatives + indicators.

        Returns:
            DataFrame with all features
        """
        logger.info("Calculating all features...")

        df = self.calculate_volume_derivatives()
        df = self.calculate_consecutive_acceleration(df)
        df = self.calculate_rsi(df)
        df = self.calculate_adx(df)
        df = self.calculate_obv(df)
        df = self.calculate_atr(df)

        # Drop intermediate columns
        df = df.drop(columns=['accel_sign'], errors='ignore')

        logger.info(f"All features calculated. Total columns: {len(df.columns)}")

        return df


def save_features(df: pd.DataFrame, output_path: str) -> None:
    """
    Save feature DataFrame to CSV.

    Args:
        df: DataFrame with features
        output_path: Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    logger.info(f"Saved features to {output_file}")


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of calculated features."""
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)

    print(f"\nTotal candles: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    print("\n--- Volume Derivatives ---")
    print(f"  v1 (velocity) mean: {df['v1'].mean():.2f}")
    print(f"  v2 (acceleration) mean: {df['v2'].mean():.2f}")
    print(f"  Consecutive_Accel range: [{df['Consecutive_Accel'].min()}, {df['Consecutive_Accel'].max()}]")

    print("\n--- Traditional Indicators ---")
    print(f"  RSI mean: {df['RSI'].mean():.2f}")
    print(f"  ADX mean: {df['ADX'].mean():.2f}")
    print(f"  OBV final: {df['OBV'].iloc[-1]:.0f}")
    print(f"  ATR mean: {df['ATR'].mean():.4f}")

    # Signal conditions
    accel_buys = len(df[df['Consecutive_Accel'] >= 2])
    accel_sells = len(df[df['Consecutive_Accel'] <= -2])
    rsi_oversold = len(df[df['RSI'] < 30])
    rsi_overbought = len(df[df['RSI'] > 70])

    print("\n--- Signal Conditions Met ---")
    print(f"  Consecutive Accel ≥2 (buy): {accel_buys} candles ({accel_buys/len(df)*100:.1f}%)")
    print(f"  Consecutive Accel ≤-2 (sell): {accel_sells} candles ({accel_sells/len(df)*100:.1f}%)")
    print(f"  RSI < 30 (oversold): {rsi_oversold} candles ({rsi_oversold/len(df)*100:.1f}%)")
    print(f"  RSI > 70 (overbought): {rsi_overbought} candles ({rsi_overbought/len(df)*100:.1f}%)")

    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate volume derivatives and technical indicators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate features from market data (Supported: BTCUSDT, ETHUSDT, SOLUSDT)
  python tools/trading/calculate_features.py --input ".tmp/market_data/BTCUSDT_15m.csv"

  # Custom output path
  python tools/trading/calculate_features.py --input data.csv --output features.csv

  # Show summary only (no save)
  python tools/trading/calculate_features.py --input data.csv --no-save
        """
    )

    parser.add_argument('--input', type=str, required=True, help='Input CSV file with OHLCV data')
    parser.add_argument('--output', type=str, help='Output CSV path (default: .tmp/features/{input_name}_features.csv)')
    parser.add_argument('--no-save', action='store_true', help='Do not save output, just print summary')

    args = parser.parse_args()

    # Create output path
    if args.output:
        output_path = args.output
    elif not args.no_save:
        input_name = Path(args.input).stem
        output_path = f".tmp/features/{input_name}_features.csv"
    else:
        output_path = None

    try:
        # Load data
        df = pd.read_csv(args.input)
        logger.info(f"Loaded data from {args.input}")

        # Calculate features
        calculator = FeatureCalculator(df)
        df_features = calculator.calculate_all_features()

        # Save features
        if output_path:
            save_features(df_features, output_path)

        # Print summary
        print_feature_summary(df_features)

        if output_path:
            print(f"✓ Features saved to {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Failed to calculate features: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
