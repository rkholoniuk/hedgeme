"""
Tests for signal generation.
"""

import pytest
import pandas as pd
import numpy as np


class TestSignalGenerator:
    """Tests for signal generation functionality."""

    def test_generator_initialization(self, sample_features_data, trading_config):
        """Test that signal generator initializes correctly."""
        from app.ml.signals import SignalGenerator

        gen = SignalGenerator(sample_features_data, None, trading_config)
        assert gen.df is not None
        assert gen.config == trading_config

    def test_traditional_signal_values(self, sample_features_data, trading_config):
        """Test that traditional signals are -1, 0, or 1."""
        from app.ml.signals import SignalGenerator

        gen = SignalGenerator(sample_features_data, None, trading_config)

        for i in range(len(sample_features_data)):
            row = sample_features_data.iloc[i]
            signal = gen.generate_traditional_signal(row)
            assert signal in [-1, 0, 1], f"Invalid signal: {signal}"

    def test_buy_signal_on_positive_acceleration(self, sample_features_data, trading_config):
        """Test that positive consecutive acceleration generates buy signal."""
        from app.ml.signals import SignalGenerator

        gen = SignalGenerator(sample_features_data, None, trading_config)

        # Create row with positive acceleration
        row = sample_features_data.iloc[-1].copy()
        row['Consecutive_Accel'] = trading_config['ACCEL_BARS_REQUIRED']

        signal = gen.generate_traditional_signal(row)
        assert signal == 1, "Should generate buy signal on positive acceleration"

    def test_sell_signal_on_negative_acceleration(self, sample_features_data, trading_config):
        """Test that negative consecutive acceleration generates sell signal."""
        from app.ml.signals import SignalGenerator

        gen = SignalGenerator(sample_features_data, None, trading_config)

        # Create row with negative acceleration
        row = sample_features_data.iloc[-1].copy()
        row['Consecutive_Accel'] = -trading_config['ACCEL_BARS_REQUIRED']

        signal = gen.generate_traditional_signal(row)
        assert signal == -1, "Should generate sell signal on negative acceleration"

    def test_neutral_signal_on_no_acceleration(self, sample_features_data, trading_config):
        """Test that no acceleration generates neutral signal."""
        from app.ml.signals import SignalGenerator

        gen = SignalGenerator(sample_features_data, None, trading_config)

        # Create row with no acceleration
        row = sample_features_data.iloc[-1].copy()
        row['Consecutive_Accel'] = 0

        signal = gen.generate_traditional_signal(row)
        assert signal == 0, "Should generate neutral signal on no acceleration"

    def test_combine_signals_agreement(self, sample_features_data, trading_config):
        """Test signal combination when both agree."""
        from app.ml.signals import SignalGenerator

        gen = SignalGenerator(sample_features_data, None, trading_config)

        # Both signals agree on buy
        combined = gen.combine_signals(1, 1)
        assert combined == 1, "Should return buy when both agree"

        # Both signals agree on sell
        combined = gen.combine_signals(-1, -1)
        assert combined == -1, "Should return sell when both agree"

    def test_combine_signals_disagreement(self, sample_features_data, trading_config):
        """Test signal combination when signals disagree."""
        from app.ml.signals import SignalGenerator

        # Enable LSTM for signal combination
        trading_config['USE_LSTM'] = True
        trading_config['LSTM_WEIGHT'] = 0.5
        gen = SignalGenerator(sample_features_data, None, trading_config)

        # Signals disagree - weighted avg of 1*0.5 + (-1)*0.5 = 0, within neutral threshold
        combined = gen.combine_signals(1, -1)
        assert combined == 0, "Should return neutral when signals conflict"

    def test_lstm_signal_generation(self, sample_features_data, trading_config, mock_lstm_model):
        """Test LSTM signal generation."""
        from app.ml.signals import SignalGenerator

        trading_config['USE_LSTM'] = True
        gen = SignalGenerator(sample_features_data, mock_lstm_model, trading_config)

        recent_volumes = sample_features_data['volume'].iloc[-10:].values
        signal, prediction = gen.generate_lstm_signal(recent_volumes)

        assert signal in [-1, 0, 1], f"Invalid LSTM signal: {signal}"
        assert prediction is not None, "Should return prediction"

    def test_signal_consistency(self, sample_features_data, trading_config):
        """Test that same input produces same signal."""
        from app.ml.signals import SignalGenerator

        gen = SignalGenerator(sample_features_data, None, trading_config)
        row = sample_features_data.iloc[-1]

        signal1 = gen.generate_traditional_signal(row)
        signal2 = gen.generate_traditional_signal(row)

        assert signal1 == signal2, "Same input should produce same signal"
