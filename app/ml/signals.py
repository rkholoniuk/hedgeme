#!/usr/bin/env python3
"""
Generate trading signals by combining traditional indicators with LSTM predictions.

This implements the signal combination logic from the article:
- Traditional signals (volume acceleration)
- LSTM signals (predicted acceleration)
- Combined signals (50/50 weighting or conservative mode)

Usage:
    python -m app.ml.signals --features ".tmp/features/BTCUSDT_15m_features.csv" --model "models/lstm_volume_predictor.keras"
    python -m app.ml.signals --features data.csv --model model.keras --mode conservative
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from typing import Dict, Optional, Tuple

from app.ml.lstm import VolumeLSTM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate trading signals from features and LSTM predictions."""

    def __init__(
        self,
        df: pd.DataFrame,
        lstm_model: Optional[VolumeLSTM] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize signal generator.

        Args:
            df: DataFrame with features (must include Consecutive_Accel, volume)
            lstm_model: Trained LSTM model (optional)
            config: Configuration dict from args/trading_config.yaml
        """
        self.df = df.copy()
        self.lstm_model = lstm_model
        self.config = config or self._load_default_config()

        self._validate_input()

    def _validate_input(self) -> None:
        """Validate input DataFrame has required columns."""
        required_cols = ['volume', 'Consecutive_Accel']
        missing = [col for col in required_cols if col not in self.df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Loaded {len(self.df)} candles for signal generation")

    def _load_default_config(self) -> Dict:
        """Load default configuration."""
        return {
            'ACCEL_BARS_REQUIRED': 2,
            'USE_LSTM': True,
            'LSTM_WEIGHT': 0.5,
            'LSTM_CONFIRMATION_REQUIRED': False,
            'LSTM_LOOKBACK': 10,
        }

    def generate_traditional_signal(self, row: pd.Series) -> int:
        """
        Generate traditional signal based on consecutive acceleration.

        Logic:
        - If Consecutive_Accel >= 2: Buy (+1)
        - If Consecutive_Accel <= -2: Sell (-1)
        - Otherwise: No signal (0)

        Args:
            row: DataFrame row with Consecutive_Accel

        Returns:
            Signal: +1 (buy), -1 (sell), 0 (neutral)
        """
        accel = row['Consecutive_Accel']
        threshold = self.config['ACCEL_BARS_REQUIRED']

        if accel >= threshold:
            return 1  # Buy signal
        elif accel <= -threshold:
            return -1  # Sell signal
        else:
            return 0  # No signal

    def generate_lstm_signal(self, recent_volumes: np.ndarray) -> Tuple[int, Optional[Dict]]:
        """
        Generate LSTM signal based on predicted acceleration.

        Logic:
        - If is_accelerating_positive: Buy (+1)
        - If is_accelerating_negative: Sell (-1)
        - Otherwise: No signal (0)

        Args:
            recent_volumes: Array of last N volumes for LSTM input

        Returns:
            Tuple of (signal, prediction_dict)
        """
        if not self.config['USE_LSTM'] or self.lstm_model is None:
            return 0, None

        try:
            # Get LSTM prediction
            prediction = self.lstm_model.predict_derivatives(recent_volumes)

            # Generate signal from prediction
            if prediction['is_accelerating_positive']:
                signal = 1
            elif prediction['is_accelerating_negative']:
                signal = -1
            else:
                signal = 0

            return signal, prediction

        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return 0, None

    def combine_signals(
        self,
        traditional_signal: int,
        lstm_signal: int
    ) -> int:
        """
        Combine traditional and LSTM signals.

        Two modes:
        1. Conservative (LSTM_CONFIRMATION_REQUIRED=True): Both must agree
        2. Balanced (default): Weighted average using LSTM_WEIGHT

        Args:
            traditional_signal: Traditional signal (+1, -1, or 0)
            lstm_signal: LSTM signal (+1, -1, or 0)

        Returns:
            Combined signal (+1, -1, or 0)
        """
        if not self.config['USE_LSTM'] or lstm_signal == 0:
            # No LSTM signal, use traditional only
            return traditional_signal

        if self.config['LSTM_CONFIRMATION_REQUIRED']:
            # Conservative mode: both must agree
            if traditional_signal == lstm_signal and traditional_signal != 0:
                return traditional_signal
            else:
                return 0
        else:
            # Balanced mode: weighted average
            lstm_weight = self.config['LSTM_WEIGHT']
            combined = (
                traditional_signal * (1 - lstm_weight) +
                lstm_signal * lstm_weight
            )

            # Threshold for signal
            if combined > 0.5:
                return 1
            elif combined < -0.5:
                return -1
            else:
                return 0

    def generate_all_signals(self) -> pd.DataFrame:
        """
        Generate signals for all candles in DataFrame.

        Returns:
            DataFrame with added columns:
            - traditional_signal
            - lstm_signal
            - combined_signal
            - lstm_prediction (dict as string)
        """
        logger.info("Generating signals for all candles...")

        signals_data = []
        lookback = self.config['LSTM_LOOKBACK']

        for idx, row in self.df.iterrows():
            # Traditional signal
            trad_signal = self.generate_traditional_signal(row)

            # LSTM signal (only if we have enough history)
            lstm_signal = 0
            lstm_prediction = None

            if idx >= lookback:
                recent_volumes = self.df['volume'].iloc[idx - lookback:idx + 1].values
                lstm_signal, lstm_prediction = self.generate_lstm_signal(recent_volumes)

            # Combined signal
            combined_signal = self.combine_signals(trad_signal, lstm_signal)

            signals_data.append({
                'traditional_signal': trad_signal,
                'lstm_signal': lstm_signal,
                'combined_signal': combined_signal,
                'lstm_prediction': str(lstm_prediction) if lstm_prediction else None
            })

        # Add signals to DataFrame
        signals_df = pd.DataFrame(signals_data, index=self.df.index)
        result_df = pd.concat([self.df, signals_df], axis=1)

        logger.info("Signal generation complete")

        return result_df


def print_signal_summary(df: pd.DataFrame) -> None:
    """Print summary of generated signals."""
    print("\n" + "="*60)
    print("SIGNAL SUMMARY")
    print("="*60)

    print(f"\nTotal candles: {len(df)}")

    # Traditional signals
    trad_buys = len(df[df['traditional_signal'] == 1])
    trad_sells = len(df[df['traditional_signal'] == -1])
    trad_neutral = len(df[df['traditional_signal'] == 0])

    print("\n--- Traditional Signals ---")
    print(f"  Buy (+1): {trad_buys} ({trad_buys/len(df)*100:.1f}%)")
    print(f"  Sell (-1): {trad_sells} ({trad_sells/len(df)*100:.1f}%)")
    print(f"  Neutral (0): {trad_neutral} ({trad_neutral/len(df)*100:.1f}%)")

    # LSTM signals
    if 'lstm_signal' in df.columns:
        lstm_buys = len(df[df['lstm_signal'] == 1])
        lstm_sells = len(df[df['lstm_signal'] == -1])
        lstm_neutral = len(df[df['lstm_signal'] == 0])

        print("\n--- LSTM Signals ---")
        print(f"  Buy (+1): {lstm_buys} ({lstm_buys/len(df)*100:.1f}%)")
        print(f"  Sell (-1): {lstm_sells} ({lstm_sells/len(df)*100:.1f}%)")
        print(f"  Neutral (0): {lstm_neutral} ({lstm_neutral/len(df)*100:.1f}%)")

    # Combined signals
    if 'combined_signal' in df.columns:
        combined_buys = len(df[df['combined_signal'] == 1])
        combined_sells = len(df[df['combined_signal'] == -1])
        combined_neutral = len(df[df['combined_signal'] == 0])

        print("\n--- Combined Signals ---")
        print(f"  Buy (+1): {combined_buys} ({combined_buys/len(df)*100:.1f}%)")
        print(f"  Sell (-1): {combined_sells} ({combined_sells/len(df)*100:.1f}%)")
        print(f"  Neutral (0): {combined_neutral} ({combined_neutral/len(df)*100:.1f}%)")

        # Agreement analysis
        if 'lstm_signal' in df.columns:
            agreement = len(df[
                (df['traditional_signal'] == df['lstm_signal']) &
                (df['traditional_signal'] != 0)
            ])
            print(f"\n  Signal agreement (when both non-zero): {agreement} candles")

    print("\n" + "="*60 + "\n")


def load_config(config_path: str = "args/trading_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Generate trading signals from features and LSTM predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate signals with LSTM model (Supported: BTCUSDT, ETHUSDT, SOLUSDT)
  python tools/trading/generate_signals.py --features ".tmp/features/BTCUSDT_15m_features.csv" --model "models/lstm_volume_predictor.keras"

  # Traditional signals only (no LSTM)
  python tools/trading/generate_signals.py --features data.csv --no-lstm

  # Conservative mode (both must agree)
  python tools/trading/generate_signals.py --features data.csv --model model.keras --mode conservative

  # Custom output path
  python tools/trading/generate_signals.py --features data.csv --model model.keras --output signals.csv
        """
    )

    parser.add_argument('--features', type=str, required=True, help='Input CSV file with features')
    parser.add_argument('--model', type=str, help='LSTM model path (.keras)')
    parser.add_argument('--scaler', type=str, help='Scaler path (.pkl, auto-detected if not specified)')
    parser.add_argument('--config', type=str, default='args/trading_config.yaml', help='Config file path')
    parser.add_argument('--no-lstm', action='store_true', help='Generate traditional signals only')
    parser.add_argument('--mode', type=str, choices=['balanced', 'conservative'], help='Signal combination mode')
    parser.add_argument('--output', type=str, help='Output CSV path (default: same as features with _signals suffix)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.no_lstm:
        config['USE_LSTM'] = False
    if args.mode == 'conservative':
        config['LSTM_CONFIRMATION_REQUIRED'] = True
    elif args.mode == 'balanced':
        config['LSTM_CONFIRMATION_REQUIRED'] = False

    # Output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.features)
        output_path = str(input_path.parent / f"{input_path.stem}_signals.csv")

    try:
        # Load features
        df = pd.read_csv(args.features)
        logger.info(f"Loaded features from {args.features}")

        # Load LSTM model if specified
        lstm_model = None
        if args.model and not args.no_lstm:
            scaler_path = args.scaler or str(Path(args.model).with_suffix('.scaler.pkl'))
            lstm_model = VolumeLSTM.load(args.model, scaler_path)
            logger.info("LSTM model loaded")

        # Generate signals
        generator = SignalGenerator(df, lstm_model, config)
        df_signals = generator.generate_all_signals()

        # Save signals
        df_signals.to_csv(output_path, index=False)
        logger.info(f"Saved signals to {output_path}")

        # Print summary
        print_signal_summary(df_signals)

        print(f"✓ Signals saved to {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Failed to generate signals: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
