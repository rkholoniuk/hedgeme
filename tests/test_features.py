"""
Tests for feature calculation.
"""

import pytest
import pandas as pd
import numpy as np


class TestFeatureCalculator:
    """Tests for feature calculation functionality."""

    def test_calculator_initialization(self, sample_ohlcv_data):
        """Test that calculator initializes correctly."""
        from app.features.calculator import FeatureCalculator

        calc = FeatureCalculator(sample_ohlcv_data)
        assert calc.df is not None
        assert len(calc.df) == len(sample_ohlcv_data)

    def test_volume_derivatives_calculated(self, sample_ohlcv_data):
        """Test that volume derivatives are calculated."""
        from app.features.calculator import FeatureCalculator

        calc = FeatureCalculator(sample_ohlcv_data)
        df = calc.calculate_all_features()

        # FeatureCalculator uses 'v1', 'v2' naming convention
        assert 'v1' in df.columns
        assert 'v2' in df.columns
        assert 'v1_normalized' in df.columns
        assert 'v2_normalized' in df.columns

    def test_consecutive_acceleration(self, sample_ohlcv_data):
        """Test consecutive acceleration calculation."""
        from app.features.calculator import FeatureCalculator

        calc = FeatureCalculator(sample_ohlcv_data)
        df = calc.calculate_all_features()

        assert 'Consecutive_Accel' in df.columns
        # Values should be integers
        assert df['Consecutive_Accel'].dtype in [np.int64, np.int32, np.float64]

    def test_technical_indicators_calculated(self, sample_ohlcv_data):
        """Test that technical indicators are calculated."""
        from app.features.calculator import FeatureCalculator

        calc = FeatureCalculator(sample_ohlcv_data)
        df = calc.calculate_all_features()

        # Check for expected indicators
        expected_indicators = ['RSI', 'ADX', 'ATR']
        for indicator in expected_indicators:
            assert indicator in df.columns, f"Missing indicator: {indicator}"

    def test_rsi_bounds(self, sample_ohlcv_data):
        """Test that RSI is between 0 and 100."""
        from app.features.calculator import FeatureCalculator

        calc = FeatureCalculator(sample_ohlcv_data)
        df = calc.calculate_all_features()

        rsi = df['RSI'].dropna()
        assert (rsi >= 0).all(), "RSI should be >= 0"
        assert (rsi <= 100).all(), "RSI should be <= 100"

    def test_atr_positive(self, sample_ohlcv_data):
        """Test that ATR is positive."""
        from app.features.calculator import FeatureCalculator

        calc = FeatureCalculator(sample_ohlcv_data)
        df = calc.calculate_all_features()

        atr = df['ATR'].dropna()
        assert (atr >= 0).all(), "ATR should be >= 0"

    def test_adx_bounds(self, sample_ohlcv_data):
        """Test that ADX is between 0 and 100."""
        from app.features.calculator import FeatureCalculator

        calc = FeatureCalculator(sample_ohlcv_data)
        df = calc.calculate_all_features()

        adx = df['ADX'].dropna()
        assert (adx >= 0).all(), "ADX should be >= 0"
        assert (adx <= 100).all(), "ADX should be <= 100"

    def test_no_nan_in_output(self, sample_ohlcv_data):
        """Test that output has minimal NaN values after warmup period."""
        from app.features.calculator import FeatureCalculator

        calc = FeatureCalculator(sample_ohlcv_data)
        df = calc.calculate_all_features()

        # After warmup (first 50 rows), should have minimal NaN
        df_after_warmup = df.iloc[50:]
        nan_counts = df_after_warmup.isna().sum()

        # Allow some NaN but not excessive
        for col, count in nan_counts.items():
            assert count < len(df_after_warmup) * 0.1, f"Too many NaN in {col}: {count}"

    def test_feature_shapes_match(self, sample_ohlcv_data):
        """Test that all features have same length."""
        from app.features.calculator import FeatureCalculator

        calc = FeatureCalculator(sample_ohlcv_data)
        df = calc.calculate_all_features()

        # All columns should have same length
        lengths = [len(df[col].dropna()) for col in df.columns]
        # Allow some variance due to indicator warmup
        assert max(lengths) - min(lengths) < 50, "Feature lengths vary too much"
