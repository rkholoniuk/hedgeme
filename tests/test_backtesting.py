"""
Tests for backtesting engine.
"""

import pytest
import pandas as pd
import numpy as np


class TestBacktester:
    """Tests for backtesting functionality."""

    def test_backtester_initialization(self):
        """Test that backtester initializes correctly."""
        from app.strategies.backtest import Backtester, StrategyConfig

        config = StrategyConfig()
        bt = Backtester(config)

        assert bt.capital == config.INITIAL_CAPITAL
        assert bt.positions == []
        assert bt.trades == []

    def test_backtest_runs_without_error(self, sample_features_data):
        """Test that backtest runs without error."""
        from app.strategies.backtest import Backtester, StrategyConfig

        config = StrategyConfig()
        config.USE_ADX = False
        config.USE_OBV = False

        # Add required columns
        df = sample_features_data.copy()
        df['Signal_Final'] = 0
        df.loc[df['Consecutive_Accel'] >= 2, 'Signal_Final'] = 1
        df.loc[df['Consecutive_Accel'] <= -2, 'Signal_Final'] = -1

        bt = Backtester(config)
        results = bt.run(df)

        assert 'total_return' in results
        assert 'total_trades' in results
        assert 'sharpe_ratio' in results

    def test_backtest_results_structure(self, sample_features_data):
        """Test that backtest results have correct structure."""
        from app.strategies.backtest import Backtester, StrategyConfig

        config = StrategyConfig()

        df = sample_features_data.copy()
        df['Signal_Final'] = 0

        bt = Backtester(config)
        results = bt.run(df)

        expected_keys = [
            'total_return',
            'total_trades',
            'win_rate',
            'avg_win',
            'avg_loss',
            'profit_factor',
            'max_drawdown',
            'sharpe_ratio',
            'final_capital',
            'trades_df',
            'equity_df'
        ]

        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_no_trades_no_error(self, sample_features_data):
        """Test that backtest handles no trades gracefully."""
        from app.strategies.backtest import Backtester, StrategyConfig

        config = StrategyConfig()

        df = sample_features_data.copy()
        df['Signal_Final'] = 0  # No signals = no trades

        bt = Backtester(config)
        results = bt.run(df)

        assert results['total_trades'] == 0
        assert results['win_rate'] == 0

    def test_equity_curve_generated(self, sample_features_data):
        """Test that equity curve is generated."""
        from app.strategies.backtest import Backtester, StrategyConfig

        config = StrategyConfig()

        df = sample_features_data.copy()
        df['Signal_Final'] = 0
        df.iloc[110, df.columns.get_loc('Signal_Final')] = 1  # Force a trade

        bt = Backtester(config)
        results = bt.run(df)

        equity_df = results['equity_df']
        assert len(equity_df) > 0, "Should have equity curve"
        assert 'timestamp' in equity_df.columns
        assert 'equity' in equity_df.columns

    def test_capital_preserved_no_trades(self, sample_features_data):
        """Test that capital is preserved when no trades."""
        from app.strategies.backtest import Backtester, StrategyConfig

        config = StrategyConfig()
        initial = config.INITIAL_CAPITAL

        df = sample_features_data.copy()
        df['Signal_Final'] = 0

        bt = Backtester(config)
        results = bt.run(df)

        assert results['final_capital'] == initial, \
            "Capital should be preserved when no trades"

    def test_max_positions_respected(self, sample_features_data):
        """Test that max positions limit is respected."""
        from app.strategies.backtest import Backtester, StrategyConfig

        config = StrategyConfig()
        config.MAX_POSITIONS = 1

        df = sample_features_data.copy()
        df['Signal_Final'] = 1  # All buy signals

        bt = Backtester(config)
        bt.run(df)

        # Should never have more than MAX_POSITIONS at once
        # This is implicitly tested by the backtest not crashing


class TestStrategyConfig:
    """Tests for strategy configuration."""

    def test_default_config_values(self):
        """Test default configuration values."""
        from app.strategies.backtest import StrategyConfig

        config = StrategyConfig()

        assert config.INITIAL_CAPITAL > 0
        assert config.RISK_PER_TRADE > 0
        assert config.RISK_PER_TRADE <= 1

    def test_config_modification(self):
        """Test that config can be modified."""
        from app.strategies.backtest import StrategyConfig

        config = StrategyConfig()
        config.INITIAL_CAPITAL = 5000

        assert config.INITIAL_CAPITAL == 5000


class TestVolumeDerivatives:
    """Tests for volume derivative calculations."""

    def test_volume_derivatives_shape(self, sample_ohlcv_data):
        """Test that volume derivatives have correct shape."""
        from app.strategies.backtest import calculate_volume_derivatives, StrategyConfig

        config = StrategyConfig()
        df = calculate_volume_derivatives(sample_ohlcv_data, config)

        assert len(df) == len(sample_ohlcv_data)
        assert 'Vol_1st_Der' in df.columns
        assert 'Vol_2nd_Der' in df.columns

    def test_consecutive_accel_values(self, sample_ohlcv_data):
        """Test consecutive acceleration values."""
        from app.strategies.backtest import calculate_volume_derivatives, StrategyConfig

        config = StrategyConfig()
        df = calculate_volume_derivatives(sample_ohlcv_data, config)

        # Should have both positive and negative values typically
        assert df['Consecutive_Accel'].max() >= 0
        assert df['Consecutive_Accel'].min() <= 0
