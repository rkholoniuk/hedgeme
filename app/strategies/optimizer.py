"""
AUTOMATIC STRATEGY OPTIMIZATION SYSTEM
Tests multiple parameter combinations and ranks by Sharpe ratio.

Usage:
    python -m app.strategies.optimizer
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path

from app.strategies.backtest import (
    StrategyConfig, download_forex_data, calculate_volume_derivatives,
    add_technical_indicators, generate_signals, Backtester
)

# ============================================================================
# OPTIMIZATION CONFIGURATION
# ============================================================================

class OptimizationConfig:
    """Define parameters to optimize with their ranges"""

    # Parameters to optimize [min, max, step]
    PARAMS_TO_OPTIMIZE = {
        'VOLUME_SMOOTH_PERIODS': [3, 5, 1],
        'ACCEL_BARS_REQUIRED': [2, 4, 1],
        'ADX_THRESHOLD': [20, 30, 5],
        'RISK_PER_TRADE': [0.03, 0.07, 0.02],
        'ATR_STOP_MULTIPLIER': [1.0, 2.0, 0.5],
        'TP_POINTS': [50, 150, 50],
        'TRAILING_START': [15, 35, 10],
        'TRAILING_STEP': [10, 20, 5],
        'MIN_CONFIRMATIONS_RATIO': [0.25, 0.75, 0.25],
    }

    # Boolean parameters
    BOOLEAN_PARAMS = {
        'USE_ADX': [True, False],
        'USE_OBV': [True, False],
        'USE_TRAILING_STOP': [True, False],
    }

    # Fixed parameters (not optimized)
    # Supported symbols: BTC-USD, ETH-USD, SOL-USD
    FIXED_PARAMS = {
        'INITIAL_CAPITAL': 10000,
        'SYMBOL': 'BTC-USD',
        'PERIOD': '2y',
        'INTERVAL': '1h',
        'COMMISSION': 0.0002,
        'USE_PRICE_MA': False,
        'USE_RSI_FILTER': False,
        'USE_BB_FILTER': False,
        'OBV_USE_TREND': False,
        'PROFIT_CLOSE': 50,
        'MAX_DAILY_LOSS': -50,
        'MAX_POSITIONS': 1,
        'SAME_DIRECTION_ONLY': False,
        'MAX_BARS_IN_TRADE': 3,
        'USE_TRADING_HOURS': True,
        'TRADE_ASIAN_SESSION': False,
        'TRADE_EUROPEAN_SESSION': True,
        'TRADE_AMERICAN_SESSION': True,
    }

    # Optimization settings
    TOP_N_RESULTS = 10
    MIN_TRADES_REQUIRED = 10
    OUTPUT_FILE = 'optimization_results.csv'
    BEST_CONFIGS_FILE = 'best_configs.json'

# ============================================================================
# PARAMETER GENERATOR
# ============================================================================

def generate_parameter_combinations():
    """Generate all parameter combinations to test"""

    # Generate ranges for numeric parameters
    numeric_params = {}
    for param, (min_val, max_val, step) in OptimizationConfig.PARAMS_TO_OPTIMIZE.items():
        if step == 0:
            numeric_params[param] = [min_val]
        else:
            numeric_params[param] = np.arange(min_val, max_val + step, step)

    # Get all combinations
    param_names = list(numeric_params.keys())
    param_values = [numeric_params[name] for name in param_names]

    # Add boolean parameters
    bool_names = list(OptimizationConfig.BOOLEAN_PARAMS.keys())
    bool_values = [OptimizationConfig.BOOLEAN_PARAMS[name] for name in bool_names]

    # Combine numeric and boolean
    all_names = param_names + bool_names
    all_values = param_values + bool_values

    combinations = list(product(*all_values))

    print(f"📊 Generated {len(combinations):,} parameter combinations")
    print(f"   Numeric params: {len(param_names)}")
    print(f"   Boolean params: {len(bool_names)}")

    return all_names, combinations

# ============================================================================
# OPTIMIZER
# ============================================================================

class StrategyOptimizer:
    """Optimize strategy parameters"""

    def __init__(self, opt_config=None):
        self.opt_config = opt_config or OptimizationConfig
        self.results = []

    def run_optimization(self, df):
        """Run optimization on all parameter combinations"""

        print("\n" + "="*80)
        print("  STRATEGY OPTIMIZATION")
        print("="*80)

        param_names, combinations = generate_parameter_combinations()

        print(f"\n🚀 Starting optimization on {len(combinations):,} combinations...")
        print(f"   This may take a while...\n")

        for combo in tqdm(combinations, desc="Optimizing"):
            try:
                # Create config with this parameter combination
                config = self._create_config(param_names, combo)

                # Run backtest
                df_processed = calculate_volume_derivatives(df.copy(), config)
                df_processed = add_technical_indicators(df_processed)
                df_processed = generate_signals(df_processed, config)

                backtester = Backtester(config)
                results = backtester.run(df_processed)

                # Check minimum trades requirement
                if results['total_trades'] < self.opt_config.MIN_TRADES_REQUIRED:
                    continue

                # Store results
                result_row = {
                    **{param_names[i]: combo[i] for i in range(len(param_names))},
                    'total_return': results['total_return'],
                    'sharpe_ratio': results['sharpe_ratio'],
                    'total_trades': results['total_trades'],
                    'win_rate': results['win_rate'],
                    'profit_factor': results['profit_factor'],
                    'max_drawdown': results['max_drawdown'],
                    'final_capital': results['final_capital']
                }

                self.results.append(result_row)

            except Exception as e:
                # Skip invalid configurations
                continue

        print(f"\n✅ Optimization complete!")
        print(f"   Valid configurations: {len(self.results)}")

        return self._analyze_results()

    def _create_config(self, param_names, combo):
        """Create a StrategyConfig with specific parameters"""

        config = StrategyConfig()

        # Apply fixed parameters
        for key, value in self.opt_config.FIXED_PARAMS.items():
            setattr(config, key, value)

        # Apply optimized parameters
        for i, name in enumerate(param_names):
            setattr(config, name, combo[i])

        return config

    def _analyze_results(self):
        """Analyze and rank results"""

        if not self.results:
            print("❌ No valid results found!")
            return None

        results_df = pd.DataFrame(self.results)

        # Rank by Sharpe ratio
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        # Get top N
        top_results = results_df.head(self.opt_config.TOP_N_RESULTS)

        # Save to CSV
        output_path = self.opt_config.OUTPUT_FILE
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Saved all results to {output_path}")

        # Save best configs to JSON
        best_configs = []
        for idx, row in top_results.iterrows():
            config_dict = row.to_dict()
            best_configs.append(config_dict)

        with open(self.opt_config.BEST_CONFIGS_FILE, 'w') as f:
            json.dump(best_configs, f, indent=2)

        print(f"💾 Saved top {self.opt_config.TOP_N_RESULTS} configs to {self.opt_config.BEST_CONFIGS_FILE}")

        return top_results

    def print_top_results(self, top_results):
        """Print top results"""

        print("\n" + "="*80)
        print(f"  TOP {len(top_results)} CONFIGURATIONS (by Sharpe Ratio)")
        print("="*80)

        for i, (idx, row) in enumerate(top_results.iterrows(), 1):
            print(f"\n{'='*80}")
            print(f"  RANK #{i}")
            print(f"{'='*80}")

            print(f"\n  📊 Performance:")
            print(f"     Sharpe Ratio:      {row['sharpe_ratio']:>10.2f}")
            print(f"     Total Return:      {row['total_return']:>10.2f}%")
            print(f"     Win Rate:          {row['win_rate']:>10.2f}%")
            print(f"     Profit Factor:     {row['profit_factor']:>10.2f}")
            print(f"     Max Drawdown:      {row['max_drawdown']:>10.2f}%")
            print(f"     Total Trades:      {row['total_trades']:>10.0f}")

            print(f"\n  ⚙️  Parameters:")
            param_cols = [col for col in row.index if col not in [
                'total_return', 'sharpe_ratio', 'total_trades', 'win_rate',
                'profit_factor', 'max_drawdown', 'final_capital'
            ]]

            for param in param_cols:
                value = row[param]
                if isinstance(value, bool):
                    print(f"     {param:<25} {value}")
                elif isinstance(value, float):
                    print(f"     {param:<25} {value:.2f}")
                else:
                    print(f"     {param:<25} {value}")

        print("\n" + "="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_optimization():
    """Run full optimization"""

    print("\n" + "="*80)
    print("  STRATEGY PARAMETER OPTIMIZATION")
    print("="*80)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Download data once
    print("\n📥 Downloading data...")
    df = download_forex_data(
        symbol=OptimizationConfig.FIXED_PARAMS['SYMBOL'],
        period=OptimizationConfig.FIXED_PARAMS['PERIOD'],
        interval=OptimizationConfig.FIXED_PARAMS['INTERVAL']
    )

    # Run optimization
    optimizer = StrategyOptimizer()
    top_results = optimizer.run_optimization(df)

    if top_results is not None:
        optimizer.print_top_results(top_results)

        print(f"\n✅ Optimization complete!")
        print(f"   Best Sharpe Ratio: {top_results.iloc[0]['sharpe_ratio']:.2f}")
        print(f"   Best Return: {top_results.iloc[0]['total_return']:.2f}%")

        return top_results
    else:
        print("\n❌ Optimization failed - no valid results")
        return None


if __name__ == "__main__":
    top_results = run_optimization()
