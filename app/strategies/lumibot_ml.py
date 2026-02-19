#!/usr/bin/env python3
"""
LumiBot ML Volume Prediction Trading Strategy.

This is the main strategy class that implements the complete trading system:
- Fetches latest OHLCV data every 15 minutes
- Calculates volume derivatives and indicators
- Gets LSTM predictions
- Combines signals (traditional + LSTM)
- Executes trades with risk management
- Sends Telegram notifications

Usage:
    # For backtesting - see backtest_ml_strategy.py
    # For live trading - see deploy_paper_trading.py

    # Direct usage (advanced):
    from lumibot.strategies import Strategy
    from lumibot_ml_strategy import MLVolumeStrategy

    strategy = MLVolumeStrategy()
    strategy.run_backtest(...)
"""

from lumibot.strategies import Strategy
from lumibot.brokers import Alpaca  # Uses Bybit via ccxt for crypto
from lumibot.entities import Asset
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional

from app.ml.lstm import VolumeLSTM
from app.features.calculator import FeatureCalculator
from app.ml.signals import SignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiskManager:
    """Risk management for position sizing and stop loss/take profit."""

    def __init__(self, config: Dict):
        """
        Initialize risk manager.

        Args:
            config: Configuration dict from args/trading_config.yaml
        """
        self.config = config

    def calculate_position_size(
        self,
        account_balance: float,
        current_price: float,
        atr: float
    ) -> Dict:
        """
        Calculate position size based on risk parameters.

        From use case:
        - Account balance: $1,000
        - Risk per trade: 5% = $50
        - ATR stop loss: 150 points
        - Position size: calculated based on risk
        - Leverage: 3x
        - Margin required: calculated based on position

        Args:
            account_balance: Current account balance
            current_price: Current asset price
            atr: Average True Range (volatility measure)

        Returns:
            Dict with position sizing details
        """
        position_fraction = self.config.get('POSITION_SIZE', 0.05)
        leverage = self.config.get('LEVERAGE', 1)  # Default no leverage
        stop_loss_pct = self.config.get('STOP_LOSS_PCT', 0.02)

        # Risk capital (fraction of account)
        risk_capital = account_balance * position_fraction

        # Calculate position size
        if leverage > 1:
            # With leverage
            margin_required = risk_capital
            position_value = margin_required * leverage
            quantity = position_value / current_price
        else:
            # No leverage
            position_value = risk_capital
            quantity = position_value / current_price

        # Stop loss distance (ATR-based)
        stop_loss_distance = atr * 1.5  # 1.5x ATR is common
        stop_loss_price = current_price - stop_loss_distance

        # Take profit distance
        take_profit_pct = self.config.get('TAKE_PROFIT_PCT', 0.04)
        take_profit_price = current_price * (1 + take_profit_pct)

        return {
            'quantity': quantity,
            'position_value': position_value,
            'margin_required': risk_capital if leverage > 1 else position_value,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'stop_loss_distance': stop_loss_distance,
            'leverage': leverage
        }

    def adjust_for_volatility(
        self,
        quantity: float,
        current_atr: float,
        normal_atr: float
    ) -> float:
        """
        Reduce position size if volatility is extreme.

        Args:
            quantity: Original position size
            current_atr: Current ATR
            normal_atr: Normal/average ATR

        Returns:
            Adjusted quantity
        """
        volatility_multiplier = self.config.get('VOLATILITY_MULTIPLIER', 3.0)

        if current_atr > normal_atr * volatility_multiplier:
            # Extreme volatility - reduce position by 50%
            adjusted_quantity = quantity * 0.5
            logger.warning(
                f"Extreme volatility detected (ATR: {current_atr:.4f} vs normal: {normal_atr:.4f}). "
                f"Reducing position size from {quantity:.2f} to {adjusted_quantity:.2f}"
            )
            return adjusted_quantity

        return quantity


class MLVolumeStrategy(Strategy):
    """
    ML-powered volume prediction trading strategy.

    This strategy combines traditional volume analysis with LSTM predictions
    to generate trading signals with sophisticated risk management.
    """

    def initialize(self, parameters: Optional[Dict] = None):
        """
        Initialize strategy. Called once at start.

        Args:
            parameters: Optional strategy parameters
        """
        logger.info(f"Initialize called with parameters argument: {parameters}")
        logger.info(f"Checking self.parameters attribute: {getattr(self, 'parameters', None)}")

        # LumiBot stores parameters as self.parameters, not passed as argument
        params = getattr(self, 'parameters', parameters) or {}
        logger.info(f"Using params: {params}")

        # Load configuration
        self.config = self._load_config()

        # Strategy parameters
        self.ticker = params.get('ticker', 'BTC/USD')
        logger.info(f"Ticker set to: {self.ticker}")
        self.timeframe = params.get('timeframe', '15m')
        logger.info(f"Timeframe set to: {self.timeframe}")
        self.lookback_candles = self.config.get('LOOKBACK_CANDLES', 200)

        # Set trading frequency (15 minutes from use case)
        self.sleeptime = f"{self.config.get('INTERVAL_MINUTES', 15)}M"

        # Load LSTM model
        model_path = params.get('model_path', 'models/lstm_volume_predictor.keras')
        scaler_path = str(Path(model_path).with_suffix('.scaler.pkl'))

        try:
            self.lstm_model = VolumeLSTM.load(model_path, scaler_path)
            logger.info(f"LSTM model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load LSTM model: {e}. Running with traditional signals only.")
            self.lstm_model = None
            self.config['USE_LSTM'] = False

        # Risk manager
        self.risk_manager = RiskManager(self.config)

        # Track position state
        self.current_position = None

        # Telegram notifications (if configured)
        self.telegram_enabled = self._setup_telegram()

        logger.info(f"Strategy initialized: {self.ticker} @ {self.timeframe}")
        logger.info(f"Config: {self.config}")

    def _load_config(self) -> Dict:
        """Load configuration from args/trading_config.yaml."""
        config_path = "args/trading_config.yaml"

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {
                'ACCEL_BARS_REQUIRED': 2,
                'USE_LSTM': True,
                'LSTM_WEIGHT': 0.5,
                'POSITION_SIZE': 0.05,
                'STOP_LOSS_PCT': 0.02,
                'TAKE_PROFIT_PCT': 0.04,
                'LEVERAGE': 1,
                'INTERVAL_MINUTES': 15,
                'LOOKBACK_CANDLES': 200,
            }

    def _setup_telegram(self) -> bool:
        """Set up Telegram notifications."""
        # TODO: Implement Telegram bot integration
        # For now, just return False
        return False

    def _get_base_asset(self) -> Asset:
        """
        Parse ticker and return base Asset object.

        Handles both "BTC/USD" and "BTC-USDT" formats.
        Returns only the base symbol - LumiBot should match this to the data source.

        Returns:
            Asset object for the base currency
        """
        if '/' in self.ticker:
            base_symbol = self.ticker.split('/')[0]
        elif '-' in self.ticker:
            base_symbol = self.ticker.split('-')[0]
        else:
            base_symbol = self.ticker

        # Just use base symbol - LumiBot should find the matching data
        return Asset(symbol=base_symbol, asset_type='crypto')

    def on_trading_iteration(self):
        """
        Main trading logic. Called every 15 minutes.

        This implements the use case workflow:
        1. Download latest OHLCV data
        2. Calculate volume derivatives
        3. Get LSTM prediction
        4. Combine signals
        5. Risk management calculation
        6. Execute order
        7. Send Telegram notification
        """
        logger.info("=" * 60)
        logger.info(f"Trading iteration at {datetime.now()}")
        logger.info("=" * 60)

        try:
            # Step 1: Get historical data
            df = self._fetch_historical_data()

            if df is None or len(df) < self.config.get('LSTM_LOOKBACK', 10) + 10:
                logger.warning("Insufficient data, skipping iteration")
                return

            # Step 2: Calculate features
            df_features = self._calculate_features(df)

            # Step 3 & 4: Generate signals (traditional + LSTM)
            signal, signal_details = self._generate_signal(df_features)

            logger.info(f"Signal: {signal} (1=BUY, -1=SELL, 0=NEUTRAL)")
            logger.info(f"Details: {signal_details}")

            # Step 5 & 6: Execute trades based on signal
            if signal == 1 and self.current_position is None:
                # Buy signal + no position → Enter long
                self._enter_long(df_features)

            elif signal == -1 and self.current_position is not None:
                # Sell signal + have position → Exit long
                current_price = df_features['close'].iloc[-1]
                self._exit_position("Sell signal", current_price)

            elif signal == -1 and self.current_position is None:
                # Sell signal + no position → Enter short (if shorting enabled)
                # For now, we only do longs
                logger.info("Sell signal but no short trading enabled")

            # Check stop loss / take profit if we have a position
            if self.current_position is not None:
                self._check_exit_conditions(df_features)

        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")
            import traceback
            traceback.print_exc()

    def _fetch_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data.

        For PandasDataBacktesting: Access pre-loaded pandas_data directly
        For live trading: Use get_historical_prices

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check if we're in PandasDataBacktesting mode
            if hasattr(self, 'parameters') and 'pandas_data' in self.parameters:
                logger.info("Using pandas_data directly (PandasDataBacktesting mode)")
                pandas_data_list = self.parameters['pandas_data']
                if pandas_data_list and len(pandas_data_list) > 0:
                    data_obj = pandas_data_list[0]
                    df = data_obj.df.copy()

                    # Get current backtest datetime
                    current_dt = self.get_datetime()
                    logger.info(f"Current backtest datetime: {current_dt}")

                    # Filter data up to current time and get last N bars
                    df_until_now = df[df.index <= current_dt]
                    df_recent = df_until_now.tail(self.lookback_candles)

                    logger.info(f"Fetched {len(df_recent)} candles from pandas_data")

                    # Ensure required columns are lowercase
                    df_recent.columns = [col.lower() for col in df_recent.columns]

                    # Add timestamp column if missing
                    if 'timestamp' not in df_recent.columns:
                        df_recent['timestamp'] = df_recent.index

                    return df_recent
                else:
                    logger.error("pandas_data list is empty")
                    return None

            # Fall back to get_historical_prices for live trading
            asset = self._get_base_asset()
            logger.info(f"Fetching data using get_historical_prices (live mode)")

            bars = self.get_historical_prices(asset, self.lookback_candles)

            if bars is None:
                logger.error("get_historical_prices returned None")
                return None

            # Convert to DataFrame
            df = bars.df

            # Ensure required columns are lowercase
            df.columns = [col.lower() for col in df.columns]

            # Add timestamp column if missing
            if 'timestamp' not in df.columns and df.index.name == 'date':
                df['timestamp'] = df.index

            logger.info(f"Fetched {len(df)} candles")

            return df

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return None

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume derivatives and technical indicators.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with features
        """
        try:
            calculator = FeatureCalculator(df)
            df_features = calculator.calculate_all_features()

            logger.info(f"Features calculated: {len(df_features.columns)} columns")

            return df_features

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            raise

    def _generate_signal(self, df: pd.DataFrame) -> tuple:
        """
        Generate trading signal from features and LSTM.

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (signal, details_dict)
        """
        try:
            generator = SignalGenerator(df, self.lstm_model, self.config)

            # Get signal for latest candle
            latest_row = df.iloc[-1]

            traditional_signal = generator.generate_traditional_signal(latest_row)

            lstm_signal = 0
            lstm_prediction = None

            if self.config['USE_LSTM'] and self.lstm_model is not None:
                lookback = self.config.get('LSTM_LOOKBACK', 10)
                recent_volumes = df['volume'].iloc[-lookback:].values
                lstm_signal, lstm_prediction = generator.generate_lstm_signal(recent_volumes)

            combined_signal = generator.combine_signals(traditional_signal, lstm_signal)

            details = {
                'traditional_signal': traditional_signal,
                'lstm_signal': lstm_signal,
                'combined_signal': combined_signal,
                'lstm_prediction': lstm_prediction,
                'consecutive_accel': latest_row['Consecutive_Accel'],
                'rsi': latest_row.get('RSI'),
                'adx': latest_row.get('ADX'),
                'volume': latest_row['volume'],
            }

            return combined_signal, details

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return 0, {}

    def _enter_long(self, df: pd.DataFrame):
        """
        Enter long position with risk management.

        Args:
            df: Features DataFrame
        """
        try:
            # Get current price and account info
            current_price = df['close'].iloc[-1]
            atr = df['ATR'].iloc[-1]
            account_balance = self.get_cash()

            # Calculate position size
            position_details = self.risk_manager.calculate_position_size(
                account_balance,
                current_price,
                atr
            )

            # Adjust for volatility
            normal_atr = df['ATR'].mean()
            quantity = self.risk_manager.adjust_for_volatility(
                position_details['quantity'],
                atr,
                normal_atr
            )

            # Create order using base asset
            asset = self._get_base_asset()

            logger.info(f"Creating BUY order: asset={asset.symbol}, quantity={quantity}")
            order = self.create_order(
                asset,
                quantity,
                "buy"
            )

            logger.info(f"Order created: {order}")
            logger.info(f"Order type: {type(order)}, filled: {hasattr(order, 'status')}")

            result = self.submit_order(order)
            logger.info(f"Order submitted, result: {result}")

            # Track position
            self.current_position = {
                'entry_price': current_price,
                'quantity': quantity,
                'stop_loss': position_details['stop_loss_price'],
                'take_profit': position_details['take_profit_price'],
                'entry_time': datetime.now()
            }

            logger.info(f"🟢 LONG POSITION OPENED")
            logger.info(f"  Quantity: {quantity:.2f} @ ${current_price:.4f}")
            logger.info(f"  Stop Loss: ${position_details['stop_loss_price']:.4f}")
            logger.info(f"  Take Profit: ${position_details['take_profit_price']:.4f}")
            logger.info(f"  Position Value: ${position_details['position_value']:.2f}")

            # Send Telegram notification
            if self.telegram_enabled:
                self._send_telegram_notification(
                    f"🟢 POSITION OPENED\n"
                    f"Buy {quantity:.2f} {self.ticker} @ ${current_price:.4f}\n"
                    f"Stop: ${position_details['stop_loss_price']:.4f} | "
                    f"Target: ${position_details['take_profit_price']:.4f}"
                )

        except Exception as e:
            logger.error(f"Error entering long position: {e}")
            import traceback
            traceback.print_exc()

    def _exit_position(self, reason: str, current_price: float = None):
        """
        Exit current position.

        Args:
            reason: Reason for exit
            current_price: Current price (optional, will fetch if not provided)
        """
        if self.current_position is None:
            return

        try:
            # Get base asset
            asset = self._get_base_asset()

            order = self.create_order(
                asset,
                self.current_position['quantity'],
                "sell"
            )

            self.submit_order(order)

            # Calculate PnL
            if current_price is None:
                # Try to get from pandas_data if available
                if hasattr(self, 'parameters') and 'pandas_data' in self.parameters:
                    pandas_data_list = self.parameters['pandas_data']
                    if pandas_data_list and len(pandas_data_list) > 0:
                        df = pandas_data_list[0].df
                        current_dt = self.get_datetime()
                        df_until_now = df[df.index <= current_dt]
                        if len(df_until_now) > 0:
                            current_price = df_until_now['close'].iloc[-1]

                # Fall back to get_last_price
                if current_price is None:
                    current_price = self.get_last_price(asset)

            if current_price is None:
                logger.error("Could not get current price for PnL calculation")
                current_price = self.current_position['entry_price']  # Assume breakeven

            pnl = (current_price - self.current_position['entry_price']) * self.current_position['quantity']
            pnl_pct = (current_price / self.current_position['entry_price'] - 1) * 100

            logger.info(f"🔴 POSITION CLOSED ({reason})")
            logger.info(f"  Entry: ${self.current_position['entry_price']:.4f}")
            logger.info(f"  Exit: ${current_price:.4f}")
            logger.info(f"  PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")

            # Send Telegram notification
            if self.telegram_enabled:
                self._send_telegram_notification(
                    f"🔴 POSITION CLOSED ({reason})\n"
                    f"Sell {self.current_position['quantity']:.2f} {self.ticker} @ ${current_price:.4f}\n"
                    f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)"
                )

            self.current_position = None

        except Exception as e:
            logger.error(f"Error exiting position: {e}")
            import traceback
            traceback.print_exc()

    def _check_exit_conditions(self, df: pd.DataFrame):
        """
        Check if stop loss or take profit hit.

        Args:
            df: Features DataFrame
        """
        if self.current_position is None:
            return

        current_price = df['close'].iloc[-1]

        # Check stop loss
        if current_price <= self.current_position['stop_loss']:
            self._exit_position("Stop loss hit", current_price)
            return

        # Check take profit
        if current_price >= self.current_position['take_profit']:
            self._exit_position("Take profit hit", current_price)
            return

        # Update trailing stop if enabled
        if self.config.get('USE_TRAILING_STOP', True):
            self._update_trailing_stop(current_price)

    def _update_trailing_stop(self, current_price: float):
        """
        Update trailing stop loss.

        Args:
            current_price: Current asset price
        """
        if self.current_position is None:
            return

        # Calculate new trailing stop
        stop_loss_pct = self.config.get('STOP_LOSS_PCT', 0.02)
        new_stop = current_price * (1 - stop_loss_pct)

        # Only update if new stop is higher than current
        if new_stop > self.current_position['stop_loss']:
            logger.info(f"Trailing stop updated: ${self.current_position['stop_loss']:.4f} → ${new_stop:.4f}")
            self.current_position['stop_loss'] = new_stop

    def _send_telegram_notification(self, message: str):
        """
        Send Telegram notification.

        Args:
            message: Message to send
        """
        # TODO: Implement Telegram bot integration
        logger.info(f"Telegram notification: {message}")

    def on_abrupt_closing(self):
        """Emergency shutdown handler."""
        logger.warning("Strategy shutting down abruptly")

        if self.current_position is not None:
            logger.warning("Emergency: Closing open position")
            self._exit_position("Emergency shutdown")  # Price will be fetched in method


# For testing
if __name__ == '__main__':
    print("MLVolumeStrategy class loaded successfully")
    print("Use backtest_ml_strategy.py or deploy_paper_trading.py to run the strategy")
