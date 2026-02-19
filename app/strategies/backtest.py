"""
COMPLETE VOLUME + OHLC TRADING STRATEGY
Backtesting framework with volume derivatives, technical indicators, and risk management.

Usage:
    python -m app.strategies.backtest
"""

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, time
from sklearn.preprocessing import StandardScaler
import ta

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# STRATEGY CONFIGURATION
# ============================================================================

class StrategyConfig:
    """Complete configuration with all features"""

    # Volume derivatives (core)
    VOLUME_SMOOTH_PERIODS = 4
    ACCEL_BARS_REQUIRED = 2

    # Confirmations
    USE_ADX = False
    ADX_THRESHOLD = 20
    USE_OBV = False
    OBV_USE_TREND = False
    USE_PRICE_MA = False
    USE_RSI_FILTER = False
    RSI_OVERSOLD = 35
    RSI_OVERBOUGHT = 65
    USE_BB_FILTER = False
    MIN_CONFIRMATIONS_RATIO = 0.25

    # Risk management
    INITIAL_CAPITAL = 200
    RISK_PER_TRADE = 0.05
    TP_POINTS = 100
    ATR_STOP_MULTIPLIER = 1.0

    # Trailing stop
    USE_TRAILING_STOP = True
    TRAILING_START = 25
    TRAILING_STEP = 15

    # Profit & loss limits
    PROFIT_CLOSE = 50
    MAX_DAILY_LOSS = -20

    # Position management
    MAX_POSITIONS = 15
    SAME_DIRECTION_ONLY = False
    MAX_BARS_IN_TRADE = 2

    # Trading hours
    USE_TRADING_HOURS = True
    TRADE_ASIAN_SESSION = False
    TRADE_EUROPEAN_SESSION = True
    TRADE_AMERICAN_SESSION = True
    SESSION_BUFFER = 15
    USE_CUSTOM_HOURS = False
    START_HOUR = 0
    START_MINUTE = 30
    END_HOUR = 23
    END_MINUTE = 59
    TRADE_MONDAY = True
    TRADE_TUESDAY = True
    TRADE_WEDNESDAY = True
    TRADE_THURSDAY = True
    TRADE_FRIDAY = True

    # Data settings (Supported: BTC-USD, ETH-USD, SOL-USD)
    SYMBOL = 'BTC-USD'
    PERIOD = '2y'
    INTERVAL = '1h'
    COMMISSION = 0.0002

# ============================================================================
# DATA ACQUISITION
# ============================================================================

def download_forex_data(symbol=None, period=None, interval=None):
    """Download and synthesize forex data"""
    symbol = symbol or StrategyConfig.SYMBOL
    period = period or StrategyConfig.PERIOD
    interval = interval or StrategyConfig.INTERVAL

    print(f"📥 Downloading {symbol} data...")
    print(f"   Period: {period}, Interval: {interval}")

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data downloaded for {symbol}")

    # Synthesize volume for Forex
    if df['Volume'].sum() == 0 or df['Volume'].mean() < 1:
        print("   Synthesizing volume from price action...")

        price_range = df['High'] - df['Low']
        price_change = abs(df['Close'].diff())

        price_range_norm = (price_range - price_range.min()) / (price_range.max() - price_range.min() + 1e-10)
        price_change_norm = (price_change - price_change.min()) / (price_change.max() - price_change.min() + 1e-10)

        base_volume = (price_range_norm + price_change_norm) / 2

        hour_of_day = df.index.hour
        time_factor = np.ones(len(df))
        time_factor[(hour_of_day >= 8) & (hour_of_day < 12)] = 1.5
        time_factor[(hour_of_day >= 13) & (hour_of_day < 17)] = 2.0
        time_factor[(hour_of_day >= 17) & (hour_of_day < 22)] = 1.3

        noise = np.random.normal(1, 0.2, len(df))
        noise = np.clip(noise, 0.5, 1.5)

        synthetic_volume = base_volume * time_factor * noise * 100000
        df['Volume'] = synthetic_volume

    print(f"   ✅ Downloaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    return df

# ============================================================================
# VOLUME DERIVATIVES
# ============================================================================

def calculate_volume_derivatives(df, config=None):
    """Calculate volume derivatives"""
    if config is None:
        config = StrategyConfig

    df = df.copy()

    df['Volume_Smoothed'] = df['Volume'].rolling(
        window=config.VOLUME_SMOOTH_PERIODS,
        min_periods=1
    ).mean()

    df['Vol_1st_Der'] = df['Volume_Smoothed'].diff()
    df['Vol_2nd_Der'] = df['Vol_1st_Der'].diff()

    df['Vol_1st_Der_Norm'] = (df['Vol_1st_Der'] - df['Vol_1st_Der'].mean()) / (df['Vol_1st_Der'].std() + 1e-10)
    df['Vol_2nd_Der_Norm'] = (df['Vol_2nd_Der'] - df['Vol_2nd_Der'].mean()) / (df['Vol_2nd_Der'].std() + 1e-10)

    df['Accel_Positive'] = (
        (df['Vol_1st_Der_Norm'] > 0.1) &
        (df['Vol_2nd_Der_Norm'] > 0.1)
    ).astype(int)

    df['Accel_Negative'] = (
        (df['Vol_1st_Der_Norm'] < -0.1) &
        (df['Vol_2nd_Der_Norm'] < -0.1)
    ).astype(int)

    df['Consecutive_Accel'] = 0
    consec = 0
    for i in range(1, len(df)):
        if df['Accel_Positive'].iloc[i]:
            consec = max(0, consec) + 1
        elif df['Accel_Negative'].iloc[i]:
            consec = min(0, consec) - 1
        else:
            consec = 0
        df.loc[df.index[i], 'Consecutive_Accel'] = consec

    print(f"📊 Volume derivatives:")
    print(f"   Max positive accel: {df['Consecutive_Accel'].max():.0f}")
    print(f"   Max negative accel: {df['Consecutive_Accel'].min():.0f}")

    return df

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def add_technical_indicators(df):
    """Add technical indicators"""
    df = df.copy()

    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    df['ADX_pos'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=14)
    df['ADX_neg'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=14)

    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['OBV_MA'] = df['OBV'].rolling(window=3).mean()

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_mid'] = bb.bollinger_mavg()

    df['Price_Momentum'] = df['Close'].diff(5)

    return df

# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_signals(df, config=None):
    """Generate trading signals"""
    if config is None:
        config = StrategyConfig

    df = df.copy()

    df['Signal_Volume'] = 0
    df['Signal_ADX'] = 0
    df['Signal_OBV'] = 0
    df['Signal_Price'] = 0
    df['Signal_RSI'] = 0
    df['Signal_BB'] = 0
    df['Signal_Final'] = 0

    # Volume signals
    df.loc[df['Consecutive_Accel'] >= config.ACCEL_BARS_REQUIRED, 'Signal_Volume'] = 1
    df.loc[df['Consecutive_Accel'] <= -config.ACCEL_BARS_REQUIRED, 'Signal_Volume'] = -1

    # ADX
    if config.USE_ADX:
        df.loc[
            (df['ADX'] > config.ADX_THRESHOLD) &
            (df['ADX_pos'] > df['ADX_neg']),
            'Signal_ADX'
        ] = 1
        df.loc[
            (df['ADX'] > config.ADX_THRESHOLD) &
            (df['ADX_neg'] > df['ADX_pos']),
            'Signal_ADX'
        ] = -1

    # OBV
    if config.USE_OBV:
        if config.OBV_USE_TREND:
            df.loc[
                (df['OBV'] > df['OBV_MA']) &
                (df['OBV'] > df['OBV'].shift(1)),
                'Signal_OBV'
            ] = 1
            df.loc[
                (df['OBV'] < df['OBV_MA']) &
                (df['OBV'] < df['OBV'].shift(1)),
                'Signal_OBV'
            ] = -1
        else:
            df.loc[df['OBV'] > df['OBV'].shift(1), 'Signal_OBV'] = 1
            df.loc[df['OBV'] < df['OBV'].shift(1), 'Signal_OBV'] = -1

    # Price MA
    if config.USE_PRICE_MA:
        df.loc[
            (df['Close'] > df['SMA_20']) &
            (df['Price_Momentum'] > 0),
            'Signal_Price'
        ] = 1
        df.loc[
            (df['Close'] < df['SMA_20']) &
            (df['Price_Momentum'] < 0),
            'Signal_Price'
        ] = -1

    # RSI filter
    if config.USE_RSI_FILTER:
        df.loc[df['RSI'] < config.RSI_OVERSOLD, 'Signal_RSI'] = 1
        df.loc[df['RSI'] > config.RSI_OVERBOUGHT, 'Signal_RSI'] = -1

    # BB filter
    if config.USE_BB_FILTER:
        df.loc[df['Close'] < df['BB_lower'], 'Signal_BB'] = 1
        df.loc[df['Close'] > df['BB_upper'], 'Signal_BB'] = -1

    # Combine signals
    for i in range(len(df)):
        vol_signal = df['Signal_Volume'].iloc[i]

        if vol_signal == 0:
            continue

        confirmations = 0
        required_confirmations = 0

        if config.USE_ADX:
            required_confirmations += 1
            if df['Signal_ADX'].iloc[i] == vol_signal:
                confirmations += 1

        if config.USE_OBV:
            required_confirmations += 1
            if df['Signal_OBV'].iloc[i] == vol_signal:
                confirmations += 1

        if config.USE_PRICE_MA:
            required_confirmations += 1
            if df['Signal_Price'].iloc[i] == vol_signal:
                confirmations += 1

        if config.USE_BB_FILTER:
            required_confirmations += 1
            if df['Signal_BB'].iloc[i] == vol_signal:
                confirmations += 1

        # RSI filter
        if config.USE_RSI_FILTER:
            rsi_signal = df['Signal_RSI'].iloc[i]
            if vol_signal == 1 and rsi_signal == -1:
                continue
            if vol_signal == -1 and rsi_signal == 1:
                continue
            if rsi_signal == vol_signal:
                confirmations += 0.5

        if required_confirmations == 0:
            df.loc[df.index[i], 'Signal_Final'] = vol_signal
        elif confirmations >= required_confirmations * config.MIN_CONFIRMATIONS_RATIO:
            df.loc[df.index[i], 'Signal_Final'] = vol_signal

    print(f"\n🎯 Signal Diagnostics:")
    print(f"   Volume signals:     {(df['Signal_Volume'] != 0).sum():>5}")
    if config.USE_ADX:
        print(f"   ADX confirmations:  {(df['Signal_ADX'] != 0).sum():>5}")
    if config.USE_OBV:
        print(f"   OBV confirmations:  {(df['Signal_OBV'] != 0).sum():>5}")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   FINAL SIGNALS:      {(df['Signal_Final'] != 0).sum():>5}")

    return df

# ============================================================================
# TRADING HOURS CHECKER
# ============================================================================

def can_trade(timestamp, config):
    """Check if trading is allowed"""
    if not config.USE_TRADING_HOURS:
        return True

    hour = timestamp.hour
    minute = timestamp.minute
    day_of_week = timestamp.weekday()

    day_allowed = [
        config.TRADE_MONDAY,
        config.TRADE_TUESDAY,
        config.TRADE_WEDNESDAY,
        config.TRADE_THURSDAY,
        config.TRADE_FRIDAY,
        False,
        False
    ]

    if not day_allowed[day_of_week]:
        return False

    if config.USE_CUSTOM_HOURS:
        current_minutes = hour * 60 + minute
        start_minutes = config.START_HOUR * 60 + config.START_MINUTE
        end_minutes = config.END_HOUR * 60 + config.END_MINUTE
        return start_minutes <= current_minutes <= end_minutes
    else:
        time_in_minutes = hour * 60 + minute

        if config.TRADE_ASIAN_SESSION:
            if config.SESSION_BUFFER <= time_in_minutes <= (8 * 60 - config.SESSION_BUFFER):
                return True

        if config.TRADE_EUROPEAN_SESSION:
            if (7 * 60 + config.SESSION_BUFFER) <= time_in_minutes <= (16 * 60 - config.SESSION_BUFFER):
                return True

        if config.TRADE_AMERICAN_SESSION:
            if (13 * 60 + config.SESSION_BUFFER) <= time_in_minutes <= (22 * 60 - config.SESSION_BUFFER):
                return True

        return False

# ============================================================================
# BACKTESTER
# ============================================================================

class Backtester:
    """Complete backtesting engine"""

    def __init__(self, config=None):
        self.config = config or StrategyConfig
        self.initial_capital = self.config.INITIAL_CAPITAL
        self.capital = self.config.INITIAL_CAPITAL
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.max_drawdown = 0
        self.peak_capital = self.config.INITIAL_CAPITAL
        self.current_day = None
        self.daily_profit = 0
        self.daily_loss_hit = False

    def run(self, df):
        """Run backtest"""
        print(f"\n🚀 Running backtest...")

        for i in range(100, len(df)):
            timestamp = df.index[i]
            current_price = df['Close'].iloc[i]
            high = df['High'].iloc[i]
            low = df['Low'].iloc[i]
            atr = df['ATR'].iloc[i]
            signal = df['Signal_Final'].iloc[i]

            # Reset daily profit tracking
            current_day = timestamp.date()
            if self.current_day != current_day:
                self.current_day = current_day
                self.daily_profit = 0
                self.daily_loss_hit = False

            # Check daily loss limit
            if self.daily_profit <= self.config.MAX_DAILY_LOSS and not self.daily_loss_hit:
                self._close_all_positions(timestamp, current_price, 'daily_loss')
                self.daily_loss_hit = True

            if self.daily_loss_hit:
                continue

            if not can_trade(timestamp, self.config):
                continue

            self._update_positions(timestamp, current_price, high, low, atr)

            if signal != 0 and len(self.positions) < self.config.MAX_POSITIONS:
                if self.config.SAME_DIRECTION_ONLY and len(self.positions) > 0:
                    existing_direction = self.positions[0]['direction']
                    new_direction = 'long' if signal > 0 else 'short'
                    if existing_direction != new_direction:
                        continue

                self._open_position(timestamp, current_price, signal, atr)

            floating_pnl = sum([
                (current_price - p['entry_price']) * p['size'] if p['direction'] == 'long'
                else (p['entry_price'] - current_price) * p['size']
                for p in self.positions
            ])

            current_equity = self.capital + floating_pnl
            self.equity_curve.append({'timestamp': timestamp, 'equity': current_equity})

            if current_equity > self.peak_capital:
                self.peak_capital = current_equity

            drawdown = (self.peak_capital - current_equity) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, drawdown)

        self._close_all_positions(df.index[-1], df['Close'].iloc[-1], 'end')

        print(f"   ✅ Backtest complete. Total trades: {len(self.trades)}")

        return self._get_results()

    def _open_position(self, timestamp, price, signal, atr):
        """Open position"""
        risk_amount = self.capital * self.config.RISK_PER_TRADE
        position_size = risk_amount / (self.config.ATR_STOP_MULTIPLIER * atr)
        position_size = min(position_size, 0.1 * 100000)

        position = {
            'id': len(self.trades) + len(self.positions),
            'timestamp': timestamp,
            'entry_price': price,
            'direction': 'long' if signal > 0 else 'short',
            'size': position_size,
            'bars_open': 0,
            'stop_loss': price - self.config.ATR_STOP_MULTIPLIER * atr if signal > 0
                        else price + self.config.ATR_STOP_MULTIPLIER * atr,
            'take_profit': price + self.config.TP_POINTS * 0.0001 if signal > 0
                          else price - self.config.TP_POINTS * 0.0001,
            'trailing_activated': False,
            'highest_profit': 0
        }

        commission = position_size * price * self.config.COMMISSION
        self.capital -= commission
        self.positions.append(position)

    def _close_position(self, position, timestamp, price, reason):
        """Close position"""
        if position not in self.positions:
            return

        if position['direction'] == 'long':
            pnl = (price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - price) * position['size']

        commission = position['size'] * price * self.config.COMMISSION
        net_pnl = pnl - commission

        self.capital += net_pnl
        self.daily_profit += net_pnl

        self.trades.append({
            'entry_time': position['timestamp'],
            'exit_time': timestamp,
            'entry_price': position['entry_price'],
            'exit_price': price,
            'direction': position['direction'],
            'pnl': net_pnl,
            'bars_held': position['bars_open'],
            'exit_reason': reason
        })

        try:
            self.positions.remove(position)
        except ValueError:
            pass

    def _close_all_positions(self, timestamp, price, reason):
        """Close all positions"""
        for position in list(self.positions):
            self._close_position(position, timestamp, price, reason)

    def _update_positions(self, timestamp, current_price, high, low, atr):
        """Update positions with trailing stop, profit close, etc."""
        point = 0.0001

        for position in list(self.positions):
            if position not in self.positions:
                continue

            position['bars_open'] += 1

            if position['direction'] == 'long':
                profit_points = (current_price - position['entry_price']) / point
            else:
                profit_points = (position['entry_price'] - current_price) / point

            # Check profit close
            if profit_points >= self.config.PROFIT_CLOSE:
                self._close_position(position, timestamp, current_price, 'profit_target')
                continue

            # Trailing stop
            if self.config.USE_TRAILING_STOP and profit_points >= self.config.TRAILING_START:
                if not position['trailing_activated']:
                    position['trailing_activated'] = True
                    position['highest_profit'] = profit_points
                else:
                    if profit_points > position['highest_profit']:
                        position['highest_profit'] = profit_points
                        trail_level = position['highest_profit'] - self.config.TRAILING_STEP

                        if position['direction'] == 'long':
                            new_sl = position['entry_price'] + trail_level * point
                            if new_sl > position['stop_loss']:
                                position['stop_loss'] = new_sl
                        else:
                            new_sl = position['entry_price'] - trail_level * point
                            if new_sl < position['stop_loss']:
                                position['stop_loss'] = new_sl

            # Check SL/TP
            if position['direction'] == 'long':
                if low <= position['stop_loss']:
                    self._close_position(position, timestamp, position['stop_loss'], 'stop_loss')
                    continue
                elif high >= position['take_profit']:
                    self._close_position(position, timestamp, position['take_profit'], 'take_profit')
                    continue
            else:
                if high >= position['stop_loss']:
                    self._close_position(position, timestamp, position['stop_loss'], 'stop_loss')
                    continue
                elif low <= position['take_profit']:
                    self._close_position(position, timestamp, position['take_profit'], 'take_profit')
                    continue

            # Time limit
            if position['bars_open'] >= self.config.MAX_BARS_IN_TRADE:
                self._close_position(position, timestamp, current_price, 'time_limit')
                continue

    def _get_results(self):
        """Calculate results"""
        equity_df = pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame({
            'timestamp': [datetime.now()],
            'equity': [self.initial_capital]
        })

        if len(self.trades) == 0:
            return {
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': self.max_drawdown * 100,
                'sharpe_ratio': 0,
                'final_capital': self.capital,
                'trades_df': pd.DataFrame(),
                'equity_df': equity_df
            }

        trades_df = pd.DataFrame(self.trades)
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]

        gross_profit = winning['pnl'].sum() if len(winning) > 0 else 0
        gross_loss = abs(losing['pnl'].sum()) if len(losing) > 0 else 1

        equity_returns = equity_df['equity'].pct_change().dropna()
        sharpe = (equity_returns.mean() / equity_returns.std()) * np.sqrt(252) if len(equity_returns) > 1 and equity_returns.std() > 0 else 0

        return {
            'total_return': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'total_trades': len(trades_df),
            'win_rate': (len(winning) / len(trades_df)) * 100,
            'avg_win': winning['pnl'].mean() if len(winning) > 0 else 0,
            'avg_loss': losing['pnl'].mean() if len(losing) > 0 else 0,
            'profit_factor': gross_profit / gross_loss,
            'max_drawdown': self.max_drawdown * 100,
            'sharpe_ratio': sharpe,
            'final_capital': self.capital,
            'trades_df': trades_df,
            'equity_df': equity_df
        }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(df, results, config):
    """Create visualization"""
    fig = plt.figure(figsize=(20, 12))

    # Price and signals
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(df.index, df['Close'], label='Close', linewidth=1, alpha=0.7)
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', linewidth=0.8, alpha=0.5)

    buy_signals = df[df['Signal_Final'] == 1]
    sell_signals = df[df['Signal_Final'] == -1]

    if len(buy_signals) > 0:
        ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy', zorder=5)
    if len(sell_signals) > 0:
        ax1.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=100, label='Sell', zorder=5)

    ax1.set_title('Price & Signals', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Volume
    ax2 = plt.subplot(3, 2, 2)
    ax2_twin = ax2.twinx()
    ax2.bar(df.index, df['Volume'], alpha=0.3, color='blue')
    ax2_twin.plot(df.index, df['Consecutive_Accel'], color='red', linewidth=1, label='Accel')
    ax2.set_title('Volume & Acceleration', fontsize=12, fontweight='bold')
    ax2_twin.legend()
    ax2.grid(alpha=0.3)

    # Equity
    ax3 = plt.subplot(3, 2, 3)
    equity_df = results['equity_df']
    if len(equity_df) > 0:
        ax3.plot(equity_df['timestamp'], equity_df['equity'], color='darkgreen', linewidth=2)
        ax3.axhline(y=results['final_capital'], color='green', linestyle='--', alpha=0.5)
    ax3.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)

    # Drawdown
    ax4 = plt.subplot(3, 2, 4)
    if len(equity_df) > 1:
        equity_df['dd'] = (equity_df['equity'] / equity_df['equity'].cummax() - 1) * 100
        ax4.fill_between(equity_df['timestamp'], 0, equity_df['dd'], color='red', alpha=0.3)
    ax4.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)

    # Trade distribution
    ax5 = plt.subplot(3, 2, 5)
    trades_df = results['trades_df']
    if len(trades_df) > 0:
        ax5.hist(trades_df['pnl'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax5.axvline(x=trades_df['pnl'].mean(), color='green', linestyle='--', linewidth=2)
    ax5.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3)

    # Metrics
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')

    exit_reasons = ""
    if len(trades_df) > 0:
        reasons_count = trades_df['exit_reason'].value_counts()
        exit_reasons = "\n    ".join([f"{k}: {v}" for k, v in reasons_count.items()])

    metrics_text = f"""
    PERFORMANCE METRICS
    {'='*45}

    Total Return:      {results['total_return']:>10.2f}%
    Final Capital:     ${results['final_capital']:>10,.2f}

    Total Trades:      {results['total_trades']:>10}
    Win Rate:          {results['win_rate']:>10.2f}%

    Avg Win:           ${results['avg_win']:>10.2f}
    Avg Loss:          ${results['avg_loss']:>10.2f}

    Profit Factor:     {results['profit_factor']:>10.2f}
    Max Drawdown:      {results['max_drawdown']:>10.2f}%
    Sharpe Ratio:      {results['sharpe_ratio']:>10.2f}

    Exit Reasons:
    {exit_reasons}
    """

    ax6.text(0.05, 0.5, metrics_text, fontsize=9, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_strategy(config=None):
    """Run strategy"""
    if config is None:
        config = StrategyConfig

    print("\n" + "="*80)
    print("  COMPLETE VOLUME + OHLC TRADING STRATEGY")
    print("="*80)

    df = download_forex_data()
    df = calculate_volume_derivatives(df, config)
    df = add_technical_indicators(df)
    df = generate_signals(df, config)

    backtester = Backtester(config)
    results = backtester.run(df)

    print("\n" + "="*80)
    print("  RESULTS")
    print("="*80)
    print(f"  Total Return:      {results['total_return']:>10.2f}%")
    print(f"  Final Capital:     ${results['final_capital']:>10,.2f}")
    print(f"  Total Trades:      {results['total_trades']:>10}")
    print(f"  Win Rate:          {results['win_rate']:>10.2f}%")
    print(f"  Sharpe Ratio:      {results['sharpe_ratio']:>10.2f}")
    print("="*80)

    fig = plot_results(df, results, config)
    plt.savefig('.tmp/backtest_results.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: .tmp/backtest_results.png")
    plt.show()

    return df, results, config


if __name__ == "__main__":
    df, results, config = run_strategy()
