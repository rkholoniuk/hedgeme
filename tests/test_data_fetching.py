"""
Tests for data fetching from Bybit API.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock


class TestBybitDataFetcher:
    """Tests for Bybit data fetching functionality."""

    def test_fetcher_initialization(self):
        """Test that fetcher initializes correctly."""
        from app.data.fetcher import BybitDataFetcher

        fetcher = BybitDataFetcher()
        assert fetcher.exchange is not None
        assert fetcher.exchange.id == 'bybit'

    def test_fetcher_rate_limit_enabled(self):
        """Test that rate limiting is enabled."""
        from app.data.fetcher import BybitDataFetcher

        fetcher = BybitDataFetcher()
        assert fetcher.exchange.enableRateLimit is True

    @pytest.mark.integration
    def test_fetch_ohlcv_live(self):
        """Integration test: fetch real data from Bybit."""
        from app.data.fetcher import BybitDataFetcher

        fetcher = BybitDataFetcher()
        df = fetcher.fetch_ohlcv(
            ticker='BTC/USDT',
            timeframe='15',
            limit=10
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def test_validate_data_detects_gaps(self, sample_ohlcv_data):
        """Test that data validation detects gaps."""
        from app.data.fetcher import BybitDataFetcher

        fetcher = BybitDataFetcher()

        # Create data with a gap - drop 'timestamp' column first if it exists as a column
        df = sample_ohlcv_data.copy()
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        df = df.reset_index()

        # Remove some rows to create a gap and reset index after concat
        df = pd.concat([df.iloc[:50], df.iloc[55:]]).reset_index(drop=True)

        # Should log warning but not raise
        fetcher._validate_data(df, '15')

    def test_validate_data_detects_zero_volume(self, sample_ohlcv_data):
        """Test that validation detects zero volume candles."""
        from app.data.fetcher import BybitDataFetcher

        fetcher = BybitDataFetcher()

        df = sample_ohlcv_data.copy()
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        df = df.reset_index()
        df.loc[10, 'volume'] = 0

        # Should log warning but not raise
        fetcher._validate_data(df, '15')

    def test_ohlcv_column_names(self, sample_ohlcv_data):
        """Test that OHLCV data has correct column names."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Just check columns directly, no need to reset_index
        for col in required_columns:
            assert col in sample_ohlcv_data.columns, f"Missing column: {col}"

    def test_price_relationships(self, sample_ohlcv_data):
        """Test that high >= low and high >= open/close."""
        df = sample_ohlcv_data

        assert (df['high'] >= df['low']).all(), "High should be >= Low"
        assert (df['high'] >= df['open']).all(), "High should be >= Open"
        assert (df['high'] >= df['close']).all(), "High should be >= Close"
        assert (df['low'] <= df['open']).all(), "Low should be <= Open"
        assert (df['low'] <= df['close']).all(), "Low should be <= Close"

    def test_volume_positive(self, sample_ohlcv_data):
        """Test that volume is positive."""
        df = sample_ohlcv_data
        assert (df['volume'] > 0).all(), "Volume should be positive"
