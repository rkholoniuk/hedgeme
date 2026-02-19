#!/usr/bin/env python3
"""
Full Training & Backtesting Pipeline

Runs the complete workflow:
1. Fetch historical data from Bybit
2. Calculate features (volume derivatives, indicators)
3. Train LSTM model
4. Run backtest
5. Generate HTML report

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --ticker ETHUSDT --days 60
    python scripts/run_pipeline.py --skip-training  # Use existing model
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"$ {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        return False

    print(f"\n✓ {description} completed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run full training and backtesting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --ticker ETHUSDT
    python scripts/run_pipeline.py --days 90 --epochs 100
    python scripts/run_pipeline.py --skip-training
        """
    )

    parser.add_argument(
        "--ticker",
        default="BTCUSDT",
        choices=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="Trading pair (default: BTCUSDT)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of historical data (default: 30)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="LSTM training epochs (default: 50)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, use existing model"
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip backtesting"
    )

    args = parser.parse_args()

    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    print("\n" + "="*60)
    print("  HEDGEME PIPELINE")
    print("="*60)
    print(f"  Ticker:     {args.ticker}")
    print(f"  Period:     {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Training:   {'Skip' if args.skip_training else 'Yes'}")
    print(f"  Backtest:   {'Skip' if args.skip_backtest else 'Yes'}")
    print("="*60)

    # Create directories
    Path(".tmp/market_data").mkdir(parents=True, exist_ok=True)
    Path(".tmp/features").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # Step 1: Fetch data
    data_file = f".tmp/market_data/{args.ticker}_15m.csv"
    if not run_command([
        sys.executable, "-m", "app.data.fetcher",
        "--ticker", args.ticker,
        "--timeframe", "15",
        "--start", start_date.strftime("%Y-%m-%d"),
        "--end", end_date.strftime("%Y-%m-%d")
    ], "Step 1: Fetch historical data"):
        return 1

    # Step 2: Calculate features
    features_file = f".tmp/features/{args.ticker}_15m_features.csv"
    if not run_command([
        sys.executable, "-m", "app.features.calculator",
        "--input", data_file
    ], "Step 2: Calculate features"):
        return 1

    # Step 3: Train LSTM (optional)
    model_path = "models/lstm_volume_predictor.keras"
    if not args.skip_training:
        if not run_command([
            sys.executable, "-m", "app.ml.lstm",
            "--features", features_file,
            "--epochs", str(args.epochs),
            "--output", model_path
        ], "Step 3: Train LSTM model"):
            return 1
    else:
        print(f"\n⏭️  Skipping training, using existing model: {model_path}")

    # Step 4: Run backtest (optional)
    if not args.skip_backtest:
        if not run_command([
            sys.executable, "-m", "app.strategies.backtest"
        ], "Step 4: Run backtest"):
            return 1

    # Step 5: Generate report
    report_file = f"reports/pipeline_{args.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    # Create simple results JSON for report
    results = {
        "ticker": args.ticker,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "epochs": args.epochs,
        "timestamp": datetime.now().isoformat()
    }

    results_file = ".tmp/pipeline_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"  Data:     {data_file}")
    print(f"  Features: {features_file}")
    print(f"  Model:    {model_path}")
    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
