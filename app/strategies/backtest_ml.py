#!/usr/bin/env python3
"""
Backtest the ML Volume Prediction Trading Strategy.

This script runs the MLVolumeStrategy using LumiBot's backtesting engine
to evaluate performance on historical data.

Usage:
    # Default backtest (last 30 days)
    python -m app.strategies.backtest_ml

    # Custom date range
    python -m app.strategies.backtest_ml \
        --ticker "BTC/USDT" \
        --start "2026-01-01" \
        --end "2026-02-07" \
        --capital 10000

    # With custom model
    python -m app.strategies.backtest_ml \
        --model "models/lstm_volume_predictor.keras" \
        --output ".tmp/backtest_results/"
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt

from lumibot.strategies import Strategy
from lumibot.backtesting import YahooDataBacktesting, PandasDataBacktesting
from lumibot.entities import Data, Asset

from app.strategies.lumibot_ml import MLVolumeStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_local_data(csv_path: str, ticker: str) -> Data:
    """
    Load local CSV data and convert to LumiBot Data format.

    Args:
        csv_path: Path to CSV file with OHLCV data
        ticker: Ticker symbol

    Returns:
        LumiBot Data object
    """
    logger.info(f"Loading local data from {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Ensure timestamp column is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Rename columns to LumiBot format (lowercase)
    df.columns = [col.lower() for col in df.columns]

    # Parse ticker to get base and quote (e.g., "BTC-USDT" -> "BTC", "USDT")
    if '-' in ticker:
        base, quote_symbol = ticker.split('-')
    elif '/' in ticker:
        base, quote_symbol = ticker.split('/')
    else:
        base = ticker
        quote_symbol = 'USD'

    # Create Assets
    asset = Asset(symbol=base, asset_type='crypto')
    quote_asset = Asset(symbol=quote_symbol, asset_type='crypto')

    # Create LumiBot Data object
    # Note: timestep must be 'minute', 'hour', or 'day' (not '15Min')
    data = Data(
        asset=asset,
        df=df,
        timestep='minute',  # For 15-minute candles, use 'minute'
        quote=quote_asset
    )

    logger.info(f"Loaded {len(df)} candles from {csv_path}")

    return data


def run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
    model_path: str,
    output_dir: str,
    use_local_data: bool = True,
    local_data_path: str = None
) -> dict:
    """
    Run backtest for the ML volume strategy.

    Args:
        ticker: Trading pair (e.g., "BTC-USD" for Yahoo Finance)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital in USD
        model_path: Path to trained LSTM model
        output_dir: Directory to save results

    Returns:
        Backtest results dictionary
    """
    logger.info(f"Starting backtest for {ticker}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial capital: ${initial_capital:,.2f}")

    # Convert ticker format for Yahoo Finance (BTC/USDT -> BTC-USD)
    # Yahoo Finance uses "-" instead of "/"
    yahoo_ticker = ticker.replace("/", "-")

    # Convert date strings to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Run backtest
    try:
        logger.info("Running backtest...")

        if use_local_data and local_data_path:
            # Use local CSV data
            logger.info(f"Using local data source from {local_data_path}")
            data = load_local_data(local_data_path, yahoo_ticker)

            # Strategy parameters for PandasDataBacktesting
            # Must match the timestep used when creating Data object ('minute')
            strategy_params = {
                'ticker': yahoo_ticker,
                'timeframe': 'minute',  # Must match Data.timestep
                'model_path': model_path,
            }

            results = MLVolumeStrategy.backtest(
                PandasDataBacktesting,
                start_dt,
                end_dt,
                parameters=strategy_params,
                pandas_data=[data],
                budget=initial_capital,
                show_plot=False,
                show_tearsheet=True,
                save_tearsheet=True,
                tearsheet_file=str(Path(output_dir) / "tearsheet.html")
            )
        else:
            # Use Yahoo Finance
            logger.info("Using Yahoo Finance data source")

            # Strategy parameters for YahooDataBacktesting
            # Yahoo Finance uses different format
            strategy_params = {
                'ticker': yahoo_ticker,
                'timeframe': '15Min',  # Yahoo Finance format
                'model_path': model_path,
            }

            results = MLVolumeStrategy.backtest(
                YahooDataBacktesting,
                start_dt,
                end_dt,
                parameters=strategy_params,
                budget=initial_capital,
                show_plot=False,
                show_tearsheet=True,
                save_tearsheet=True,
                tearsheet_file=str(Path(output_dir) / "tearsheet.html")
            )

        logger.info("Backtest complete!")

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Print summary
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"\nPeriod: {start_date} to {end_date}")
        print(f"Ticker: {ticker}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"\nResults object type: {type(results)}")
        print(f"Results keys: {results.keys() if hasattr(results, 'keys') else 'N/A'}")
        print("\n" + "="*60)
        print(f"\nResults saved to: {output_dir}")
        print(f"Tearsheet: {output_path / 'tearsheet.html'}")
        print("="*60 + "\n")

        # Save full results for debugging
        import json
        results_file = output_path / "backtest_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"Backtest Results\n")
            f.write(f"================\n\n")
            f.write(f"Type: {type(results)}\n\n")
            if hasattr(results, '__dict__'):
                f.write(f"Attributes:\n{results.__dict__}\n\n")
            f.write(f"String representation:\n{results}\n")
        logger.info(f"Raw results saved to {results_file}")

        return results

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def export_results_json(results, ticker: str, start_date: str, end_date: str, output_path: str):
    """
    Export backtest results to JSON format for HTML report generator.

    Args:
        results: Backtest results from LumiBot
        ticker: Trading pair
        start_date: Start date
        end_date: End date
        output_path: Output JSON file path
    """
    import json

    # Extract data from results
    export_data = {
        'ticker': ticker,
        'start_date': start_date,
        'end_date': end_date,
        'results': {},
        'trades': [],
        'price_data': []
    }

    # Try to extract performance metrics
    if hasattr(results, 'keys'):
        for key in results.keys():
            try:
                val = results[key]
                if isinstance(val, (int, float, str, bool)):
                    export_data['results'][key] = val
            except Exception:
                pass

    # Try to extract trades
    if hasattr(results, 'trades_df'):
        trades_df = results.trades_df
        if not trades_df.empty:
            export_data['trades'] = trades_df.to_dict(orient='records')

    # Try to extract equity curve
    if hasattr(results, 'equity_df'):
        equity_df = results.equity_df
        if not equity_df.empty:
            export_data['results']['equity_df'] = equity_df.to_dict(orient='records')

    # Write JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)

    logger.info(f"Exported results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest ML Volume Trading Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default backtest (last 30 days)
  python tools/trading/backtest_ml_strategy.py

  # Custom date range
  python tools/trading/backtest_ml_strategy.py --ticker "BTC-USD" --start "2026-01-01" --end "2026-02-07"

  # With custom initial capital
  python tools/trading/backtest_ml_strategy.py --capital 50000

  # Specify custom model
  python tools/trading/backtest_ml_strategy.py --model "models/my_model.keras"
        """
    )

    # Default dates: last 30 days
    default_end = datetime.now()
    default_start = default_end - timedelta(days=30)

    parser.add_argument(
        '--ticker',
        type=str,
        default='BTC-USD',
        help='Trading pair in Yahoo Finance format (e.g., BTC-USD, ETH-USD). Default: BTC-USD'
    )
    parser.add_argument(
        '--start',
        type=str,
        default=default_start.strftime('%Y-%m-%d'),
        help='Start date (YYYY-MM-DD). Default: 30 days ago'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=default_end.strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Initial capital in USD. Default: 10000'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/lstm_volume_predictor.keras',
        help='Path to trained LSTM model. Default: models/lstm_volume_predictor.keras'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='.tmp/backtest_results',
        help='Output directory for results. Default: .tmp/backtest_results'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to local CSV data file. If not specified, will use Yahoo Finance.'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        help='Path to output JSON file for HTML report generation.'
    )

    args = parser.parse_args()

    try:
        # Validate model exists
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"Model not found: {args.model}")
            logger.error("Train a model first using: python tools/trading/train_lstm_model.py")
            return 1

        # Run backtest
        results = run_backtest(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
            model_path=args.model,
            output_dir=args.output,
            use_local_data=args.data is not None,
            local_data_path=args.data
        )

        # Export results to JSON for HTML report generator
        if args.output_json:
            export_results_json(results, args.ticker, args.start, args.end, args.output_json)

        return 0

    except Exception as e:
        logger.error(f"Failed to run backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
