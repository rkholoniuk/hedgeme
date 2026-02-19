#!/usr/bin/env python3
"""
Fetch historical OHLCV data from Bybit API.

Bybit advantages over Kraken:
- Higher rate limits (10 requests/sec vs 1 request/3sec)
- Better WebSocket support
- More crypto pairs
- Lower fees

Usage:
    # Supported tickers: BTCUSDT, ETHUSDT, SOLUSDT
    python -m app.data.fetcher --ticker "BTCUSDT" --timeframe "15" --limit 200
    python -m app.data.fetcher --ticker "ETHUSDT" --start "2024-01-01" --end "2025-12-31"
"""

import argparse
import ccxt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging
from typing import Optional
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BybitDataFetcher:
    """Fetches OHLCV data from Bybit with rate limit handling."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Bybit connection.

        Args:
            api_key: Bybit API key (optional for public data)
            api_secret: Bybit API secret (optional for public data)
        """
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'rateLimit': 100,  # 10 requests per second (100ms between calls)
            'options': {
                'defaultType': 'spot',  # or 'linear' for USDT perpetuals
            }
        })

    def _normalize_ticker(self, ticker: str) -> str:
        """
        Normalize ticker format to CCXT standard (e.g., BTCUSDT -> BTC/USDT).

        Args:
            ticker: Ticker in any format (BTCUSDT, BTC/USDT, etc.)

        Returns:
            Normalized ticker (e.g., BTC/USDT)
        """
        # Already in correct format
        if '/' in ticker:
            return ticker

        # Common quote currencies
        quotes = ['USDT', 'USD', 'BTC', 'ETH', 'USDC']

        for quote in quotes:
            if ticker.endswith(quote):
                base = ticker[:-len(quote)]
                return f"{base}/{quote}"

        # If no match, return as-is
        return ticker

    def fetch_ohlcv(
        self,
        ticker: str,
        timeframe: str = '15',
        limit: int = 200,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Bybit.

        Args:
            ticker: Trading pair (e.g., "BTCUSDT", "ETHUSDT", "SOLUSDT")
            timeframe: Candle interval in minutes (1, 5, 15, 60, 240, D)
            limit: Number of candles to fetch (default: 200)
            start_date: Start date string (YYYY-MM-DD) for historical fetch
            end_date: End date string (YYYY-MM-DD) for historical fetch

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        logger.info(f"Fetching {ticker} {timeframe}m data from Bybit")

        try:
            # Load markets to validate ticker
            self.exchange.load_markets()

            # Normalize ticker format (BTCUSDT -> BTC/USDT)
            normalized_ticker = self._normalize_ticker(ticker)

            if normalized_ticker not in self.exchange.markets:
                raise ValueError(f"Invalid ticker: {ticker}. Available: {list(self.exchange.markets.keys())[:10]}...")

            ticker = normalized_ticker

            all_candles = []

            if start_date and end_date:
                # Historical fetch with date range
                all_candles = self._fetch_historical(ticker, timeframe, start_date, end_date)
            else:
                # Recent fetch (last N candles)
                candles = self.exchange.fetch_ohlcv(ticker, f"{timeframe}m", limit=limit)
                all_candles = candles

            # Convert to DataFrame
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Validate data
            self._validate_data(df, timeframe)

            logger.info(f"Fetched {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")

            return df

        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def _fetch_historical(
        self,
        ticker: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> list:
        """
        Fetch historical data in chunks (Bybit limits to 200 candles per call).

        Args:
            ticker: Trading pair
            timeframe: Candle interval
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of OHLCV candles
        """
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Convert to milliseconds
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        # Timeframe to milliseconds
        timeframe_map = {
            '1': 60 * 1000,
            '5': 5 * 60 * 1000,
            '15': 15 * 60 * 1000,
            '60': 60 * 60 * 1000,
            '240': 4 * 60 * 60 * 1000,
            'D': 24 * 60 * 60 * 1000,
        }

        if timeframe not in timeframe_map:
            raise ValueError(f"Invalid timeframe: {timeframe}. Valid: {list(timeframe_map.keys())}")

        tf_ms = timeframe_map[timeframe]

        # Fetch in chunks of 200 candles (Bybit limit)
        all_candles = []
        current_ms = start_ms

        while current_ms < end_ms:
            try:
                logger.info(f"Fetching from {datetime.fromtimestamp(current_ms/1000)}")

                candles = self.exchange.fetch_ohlcv(
                    ticker,
                    f"{timeframe}m",
                    since=current_ms,
                    limit=200  # Bybit max
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # Move to next chunk
                current_ms = candles[-1][0] + tf_ms

                # Rate limiting (100ms between calls = 10 req/sec)
                time.sleep(0.1)

            except ccxt.RateLimitExceeded:
                logger.warning("Rate limit hit, waiting 1 second...")
                time.sleep(1)
                continue

        # Remove duplicates and filter by end date
        all_candles = [c for c in all_candles if c[0] <= end_ms]
        all_candles = list({c[0]: c for c in all_candles}.values())  # Deduplicate by timestamp
        all_candles.sort(key=lambda x: x[0])  # Sort by timestamp

        return all_candles

    def _validate_data(self, df: pd.DataFrame, timeframe: str) -> None:
        """
        Validate fetched data for missing candles and anomalies.

        Args:
            df: DataFrame with OHLCV data
            timeframe: Candle interval
        """
        if df.empty:
            raise ValueError("No data fetched")

        # Check for missing candles
        expected_interval = pd.Timedelta(f"{timeframe}min")
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff()

        # Find gaps larger than 2x expected interval
        large_gaps = time_diffs[time_diffs > expected_interval * 2]

        if not large_gaps.empty:
            logger.warning(f"Found {len(large_gaps)} gaps in data larger than 2x interval")
            for idx, gap in large_gaps.items():
                gap_start = df_sorted.loc[idx-1, 'timestamp']
                gap_end = df_sorted.loc[idx, 'timestamp']
                logger.warning(f"  Gap: {gap_start} -> {gap_end} ({gap})")

        # Check for zero volumes
        zero_volumes = df[df['volume'] == 0]
        if not zero_volumes.empty:
            logger.warning(f"Found {len(zero_volumes)} candles with zero volume")

        logger.info("Data validation complete")


def save_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} candles to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch OHLCV data from Bybit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch last 200 candles (Supported: BTCUSDT, ETHUSDT, SOLUSDT)
  python tools/trading/fetch_bybit_data.py --ticker "BTCUSDT" --timeframe "15" --limit 200

  # Fetch historical data range
  python tools/trading/fetch_bybit_data.py --ticker "ETHUSDT" --start "2024-01-01" --end "2025-12-31"

  # Custom output path
  python tools/trading/fetch_bybit_data.py --ticker "SOLUSDT" --output ".tmp/market_data/sol_custom.csv"
        """
    )

    parser.add_argument('--ticker', type=str, required=True, help='Trading pair (e.g., BTCUSDT, ETHUSDT, SOLUSDT)')
    parser.add_argument('--timeframe', type=str, default='15', help='Candle interval in minutes (default: 15)')
    parser.add_argument('--limit', type=int, default=200, help='Number of recent candles (default: 200)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD) for historical fetch')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD) for historical fetch')
    parser.add_argument('--output', type=str, help='Output CSV path (default: .tmp/market_data/{ticker}_{tf}m.csv)')

    args = parser.parse_args()

    # Create output path
    if args.output:
        output_path = args.output
    else:
        ticker_clean = args.ticker.replace('/', '_')
        output_path = f".tmp/market_data/{ticker_clean}_{args.timeframe}m.csv"

    try:
        # Fetch data (no API keys needed for public OHLCV data)
        fetcher = BybitDataFetcher()

        df = fetcher.fetch_ohlcv(
            ticker=args.ticker,
            timeframe=args.timeframe,
            limit=args.limit,
            start_date=args.start,
            end_date=args.end
        )

        # Save to CSV
        save_to_csv(df, output_path)

        # Print summary
        print(f"\n✓ Successfully fetched {len(df)} candles")
        print(f"  Ticker: {args.ticker}")
        print(f"  Timeframe: {args.timeframe}m")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Output: {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
