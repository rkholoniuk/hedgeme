#!/usr/bin/env python3
"""
HTML Report Generator for Trading Analysis.

Generates comprehensive HTML reports with:
- Performance metrics
- Interactive equity curve
- Buy/sell signal visualization on price chart
- Trade distribution analysis
- LSTM training metrics

Usage:
    python -m app.reporting.html --backtest-results results.json --output report.html
    python -m app.reporting.html --training-log training.csv --output training_report.html
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import base64
import io

# Try to import plotly for interactive charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Install with: pip install plotly")


class HTMLReportGenerator:
    """Generate comprehensive HTML reports for trading analysis."""

    def __init__(self):
        self.report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_backtest_report(
        self,
        df: pd.DataFrame,
        results: Dict,
        trades_df: pd.DataFrame,
        output_path: str,
        title: str = "Backtest Analysis Report"
    ) -> str:
        """
        Generate comprehensive backtest HTML report.

        Args:
            df: DataFrame with OHLCV and signals
            results: Dict with performance metrics
            trades_df: DataFrame with trade history
            output_path: Output HTML file path
            title: Report title

        Returns:
            Path to generated HTML file
        """
        # Generate charts
        price_chart = self._create_price_signal_chart(df, trades_df)
        equity_chart = self._create_equity_chart(results.get('equity_df', pd.DataFrame()))
        distribution_chart = self._create_trade_distribution_chart(trades_df)
        metrics_table = self._create_metrics_table(results)
        trades_table = self._create_trades_table(trades_df)

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --accent-green: #00d26a;
            --accent-red: #ff6b6b;
            --accent-blue: #4dabf7;
            --border-color: #2a2a4a;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
            border-radius: 15px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header .date {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: var(--bg-card);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid var(--border-color);
            transition: transform 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
        }}

        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .metric-value.positive {{
            color: var(--accent-green);
        }}

        .metric-value.negative {{
            color: var(--accent-red);
        }}

        .metric-label {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .chart-section {{
            background: var(--bg-secondary);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
        }}

        .chart-section h2 {{
            margin-bottom: 20px;
            color: var(--accent-blue);
            font-size: 1.3rem;
        }}

        .chart-container {{
            width: 100%;
            height: 500px;
        }}

        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}

        .trades-table th,
        .trades-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        .trades-table th {{
            background: var(--bg-card);
            color: var(--accent-blue);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 1px;
        }}

        .trades-table tr:hover {{
            background: rgba(77, 171, 247, 0.1);
        }}

        .pnl-positive {{
            color: var(--accent-green);
        }}

        .pnl-negative {{
            color: var(--accent-red);
        }}

        .signal-buy {{
            background: rgba(0, 210, 106, 0.2);
            color: var(--accent-green);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
        }}

        .signal-sell {{
            background: rgba(255, 107, 107, 0.2);
            color: var(--accent-red);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}

        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}

        @media (max-width: 768px) {{
            .two-column {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p class="date">Generated: {self.report_date}</p>
        </div>

        {metrics_table}

        <div class="chart-section">
            <h2>Price Chart with Buy/Sell Signals</h2>
            <div id="price-chart" class="chart-container"></div>
        </div>

        <div class="two-column">
            <div class="chart-section">
                <h2>Equity Curve</h2>
                <div id="equity-chart" class="chart-container" style="height: 350px;"></div>
            </div>
            <div class="chart-section">
                <h2>Trade P&L Distribution</h2>
                <div id="distribution-chart" class="chart-container" style="height: 350px;"></div>
            </div>
        </div>

        <div class="chart-section">
            <h2>Trade History</h2>
            {trades_table}
        </div>

        <div class="footer">
            <p>Generated by HedgeMe Trading System</p>
        </div>
    </div>

    <script>
        {price_chart}
        {equity_chart}
        {distribution_chart}
    </script>
</body>
</html>
"""

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html_content)

        return str(output_file)

    def _create_price_signal_chart(self, df: pd.DataFrame, trades_df: pd.DataFrame) -> str:
        """Create interactive price chart with buy/sell signals."""
        if not PLOTLY_AVAILABLE:
            return "console.log('Plotly not available');"

        # Prepare data
        df = df.reset_index()
        if 'timestamp' not in df.columns and 'index' in df.columns:
            df['timestamp'] = df['index']

        # Get buy and sell signals
        buy_signals = df[df.get('Signal_Final', df.get('signal', pd.Series([0]*len(df)))) == 1]
        sell_signals = df[df.get('Signal_Final', df.get('signal', pd.Series([0]*len(df)))) == -1]

        # Create candlestick chart
        fig_json = f"""
        var priceData = [
            {{
                type: 'candlestick',
                x: {df['timestamp'].astype(str).tolist()},
                open: {df['open'].tolist()},
                high: {df['high'].tolist()},
                low: {df['low'].tolist()},
                close: {df['close'].tolist()},
                name: 'Price',
                increasing: {{line: {{color: '#00d26a'}}}},
                decreasing: {{line: {{color: '#ff6b6b'}}}}
            }},
            {{
                type: 'scatter',
                mode: 'markers',
                x: {buy_signals['timestamp'].astype(str).tolist() if len(buy_signals) > 0 else []},
                y: {buy_signals['low'].tolist() if len(buy_signals) > 0 else []},
                marker: {{
                    symbol: 'triangle-up',
                    size: 15,
                    color: '#00d26a'
                }},
                name: 'Buy Signal'
            }},
            {{
                type: 'scatter',
                mode: 'markers',
                x: {sell_signals['timestamp'].astype(str).tolist() if len(sell_signals) > 0 else []},
                y: {sell_signals['high'].tolist() if len(sell_signals) > 0 else []},
                marker: {{
                    symbol: 'triangle-down',
                    size: 15,
                    color: '#ff6b6b'
                }},
                name: 'Sell Signal'
            }}
        ];

        var priceLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eaeaea'}},
            xaxis: {{
                gridcolor: '#2a2a4a',
                rangeslider: {{visible: false}}
            }},
            yaxis: {{
                gridcolor: '#2a2a4a',
                title: 'Price'
            }},
            legend: {{
                orientation: 'h',
                y: 1.1
            }},
            margin: {{t: 50, r: 50, b: 50, l: 50}}
        }};

        Plotly.newPlot('price-chart', priceData, priceLayout, {{responsive: true}});
        """

        return fig_json

    def _create_equity_chart(self, equity_df: pd.DataFrame) -> str:
        """Create equity curve chart."""
        if equity_df.empty:
            return "console.log('No equity data');"

        equity_df = equity_df.reset_index(drop=True)

        return f"""
        var equityData = [{{
            type: 'scatter',
            mode: 'lines',
            x: {list(range(len(equity_df)))},
            y: {equity_df['equity'].tolist()},
            fill: 'tozeroy',
            fillcolor: 'rgba(0, 210, 106, 0.1)',
            line: {{color: '#00d26a', width: 2}},
            name: 'Equity'
        }}];

        var equityLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eaeaea'}},
            xaxis: {{gridcolor: '#2a2a4a', title: 'Time'}},
            yaxis: {{gridcolor: '#2a2a4a', title: 'Equity ($)'}},
            margin: {{t: 20, r: 30, b: 50, l: 60}}
        }};

        Plotly.newPlot('equity-chart', equityData, equityLayout, {{responsive: true}});
        """

    def _create_trade_distribution_chart(self, trades_df: pd.DataFrame) -> str:
        """Create trade P&L distribution histogram."""
        if trades_df.empty:
            return "console.log('No trades data');"

        pnl_values = trades_df['pnl'].tolist()

        return f"""
        var distData = [{{
            type: 'histogram',
            x: {pnl_values},
            marker: {{
                color: {pnl_values}.map(v => v >= 0 ? '#00d26a' : '#ff6b6b')
            }},
            name: 'P&L Distribution'
        }}];

        var distLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eaeaea'}},
            xaxis: {{gridcolor: '#2a2a4a', title: 'P&L ($)'}},
            yaxis: {{gridcolor: '#2a2a4a', title: 'Count'}},
            bargap: 0.05,
            margin: {{t: 20, r: 30, b: 50, l: 60}},
            shapes: [{{
                type: 'line',
                x0: 0, x1: 0,
                y0: 0, y1: 1,
                yref: 'paper',
                line: {{color: '#eaeaea', width: 1, dash: 'dash'}}
            }}]
        }};

        Plotly.newPlot('distribution-chart', distData, distLayout, {{responsive: true}});
        """

    def _create_metrics_table(self, results: Dict) -> str:
        """Create metrics cards HTML."""
        def format_value(value, is_pct=False, is_money=False):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "N/A", ""
            if is_pct:
                cls = "positive" if value > 0 else "negative" if value < 0 else ""
                return f"{value:+.2f}%", cls
            if is_money:
                cls = "positive" if value > 0 else "negative" if value < 0 else ""
                return f"${value:,.2f}", cls
            if isinstance(value, float):
                return f"{value:.2f}", ""
            return str(value), ""

        metrics = [
            ("Total Return", results.get('total_return', 0), True, False),
            ("Sharpe Ratio", results.get('sharpe_ratio', 0), False, False),
            ("Max Drawdown", -abs(results.get('max_drawdown', 0)), True, False),
            ("Total Trades", results.get('total_trades', 0), False, False),
            ("Win Rate", results.get('win_rate', 0), True, False),
            ("Profit Factor", results.get('profit_factor', 0), False, False),
            ("Avg Win", results.get('avg_win', 0), False, True),
            ("Avg Loss", results.get('avg_loss', 0), False, True),
        ]

        cards_html = '<div class="metrics-grid">'
        for label, value, is_pct, is_money in metrics:
            formatted, cls = format_value(value, is_pct, is_money)
            cards_html += f'''
            <div class="metric-card">
                <div class="metric-value {cls}">{formatted}</div>
                <div class="metric-label">{label}</div>
            </div>
            '''
        cards_html += '</div>'

        return cards_html

    def _create_trades_table(self, trades_df: pd.DataFrame) -> str:
        """Create trades history table HTML."""
        if trades_df.empty:
            return '<p style="color: var(--text-secondary);">No trades executed.</p>'

        rows = []
        for _, row in trades_df.iterrows():
            pnl = row.get('pnl', 0)
            pnl_class = 'pnl-positive' if pnl > 0 else 'pnl-negative'
            direction = row.get('direction', 'long')
            signal_class = 'signal-buy' if direction == 'long' else 'signal-sell'

            entry_time = row.get('entry_time', '')
            if hasattr(entry_time, 'strftime'):
                entry_time = entry_time.strftime('%Y-%m-%d %H:%M')

            exit_time = row.get('exit_time', '')
            if hasattr(exit_time, 'strftime'):
                exit_time = exit_time.strftime('%Y-%m-%d %H:%M')

            rows.append(f'''
            <tr>
                <td>{entry_time}</td>
                <td>{exit_time}</td>
                <td><span class="{signal_class}">{direction.upper()}</span></td>
                <td>${row.get('entry_price', 0):.4f}</td>
                <td>${row.get('exit_price', 0):.4f}</td>
                <td class="{pnl_class}">${pnl:+.2f}</td>
                <td>{row.get('exit_reason', 'N/A')}</td>
            </tr>
            ''')

        return f'''
        <table class="trades-table">
            <thead>
                <tr>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Direction</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>P&L</th>
                    <th>Exit Reason</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows[:50])}
            </tbody>
        </table>
        <p style="color: var(--text-secondary); margin-top: 10px;">
            Showing {min(50, len(trades_df))} of {len(trades_df)} trades
        </p>
        '''

    def generate_training_report(
        self,
        training_history: Dict,
        predictions_df: pd.DataFrame,
        output_path: str,
        title: str = "LSTM Training Report"
    ) -> str:
        """
        Generate LSTM training analysis HTML report.

        Args:
            training_history: Dict with loss history
            predictions_df: DataFrame with actual vs predicted values
            output_path: Output HTML file path
            title: Report title

        Returns:
            Path to generated HTML file
        """
        loss_chart = self._create_loss_chart(training_history)
        prediction_chart = self._create_prediction_chart(predictions_df)

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --accent-green: #00d26a;
            --accent-red: #ff6b6b;
            --accent-blue: #4dabf7;
            --accent-purple: #9775fa;
            --border-color: #2a2a4a;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{ max-width: 1400px; margin: 0 auto; }}

        .header {{
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
            border-radius: 15px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, var(--accent-purple), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .chart-section {{
            background: var(--bg-secondary);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
        }}

        .chart-section h2 {{
            margin-bottom: 20px;
            color: var(--accent-blue);
        }}

        .chart-container {{ width: 100%; height: 400px; }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: var(--bg-card);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid var(--border-color);
        }}

        .metric-value {{ font-size: 2rem; font-weight: bold; color: var(--accent-blue); }}
        .metric-label {{ color: var(--text-secondary); font-size: 0.85rem; text-transform: uppercase; }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p style="color: var(--text-secondary);">Generated: {self.report_date}</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{len(training_history.get('loss', []))}</div>
                <div class="metric-label">Epochs Trained</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{training_history.get('loss', [0])[-1]:.4f}</div>
                <div class="metric-label">Final Loss</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{min(training_history.get('loss', [0])):.4f}</div>
                <div class="metric-label">Best Loss</div>
            </div>
        </div>

        <div class="chart-section">
            <h2>Training Loss Curve</h2>
            <div id="loss-chart" class="chart-container"></div>
        </div>

        <div class="chart-section">
            <h2>Predictions vs Actual</h2>
            <div id="prediction-chart" class="chart-container"></div>
        </div>

        <div class="footer">
            <p>Generated by HedgeMe Trading System</p>
        </div>
    </div>

    <script>
        {loss_chart}
        {prediction_chart}
    </script>
</body>
</html>
"""

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html_content)

        return str(output_file)

    def _create_loss_chart(self, training_history: Dict) -> str:
        """Create training loss chart."""
        loss = training_history.get('loss', [])
        val_loss = training_history.get('val_loss', [])

        return f"""
        var lossData = [
            {{
                type: 'scatter',
                mode: 'lines',
                y: {loss},
                name: 'Training Loss',
                line: {{color: '#4dabf7', width: 2}}
            }},
            {{
                type: 'scatter',
                mode: 'lines',
                y: {val_loss if val_loss else loss},
                name: 'Validation Loss',
                line: {{color: '#ff6b6b', width: 2}}
            }}
        ];

        var lossLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eaeaea'}},
            xaxis: {{gridcolor: '#2a2a4a', title: 'Epoch'}},
            yaxis: {{gridcolor: '#2a2a4a', title: 'Loss'}},
            legend: {{orientation: 'h', y: 1.1}},
            margin: {{t: 50, r: 30, b: 50, l: 60}}
        }};

        Plotly.newPlot('loss-chart', lossData, lossLayout, {{responsive: true}});
        """

    def _create_prediction_chart(self, predictions_df: pd.DataFrame) -> str:
        """Create predictions vs actual chart."""
        if predictions_df.empty:
            return "console.log('No prediction data');"

        actual = predictions_df.get('actual', predictions_df.iloc[:, 0]).tolist()
        predicted = predictions_df.get('predicted', predictions_df.iloc[:, 1] if len(predictions_df.columns) > 1 else predictions_df.iloc[:, 0]).tolist()

        return f"""
        var predData = [
            {{
                type: 'scatter',
                mode: 'lines',
                y: {actual[:200]},
                name: 'Actual',
                line: {{color: '#00d26a', width: 2}}
            }},
            {{
                type: 'scatter',
                mode: 'lines',
                y: {predicted[:200]},
                name: 'Predicted',
                line: {{color: '#9775fa', width: 2}}
            }}
        ];

        var predLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eaeaea'}},
            xaxis: {{gridcolor: '#2a2a4a', title: 'Sample'}},
            yaxis: {{gridcolor: '#2a2a4a', title: 'Value'}},
            legend: {{orientation: 'h', y: 1.1}},
            margin: {{t: 50, r: 30, b: 50, l: 60}}
        }};

        Plotly.newPlot('prediction-chart', predData, predLayout, {{responsive: true}});
        """


def main():
    parser = argparse.ArgumentParser(description="Generate HTML trading analysis reports")
    parser.add_argument('--backtest-results', type=str, help='Path to backtest results JSON')
    parser.add_argument('--training-log', type=str, help='Path to training history CSV')
    parser.add_argument('--output', type=str, default='reports/report.html', help='Output HTML path')
    parser.add_argument('--title', type=str, default='Trading Analysis Report', help='Report title')

    args = parser.parse_args()

    generator = HTMLReportGenerator()

    if args.backtest_results:
        with open(args.backtest_results, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data.get('price_data', []))
        results = data.get('results', {})
        trades_df = pd.DataFrame(data.get('trades', []))

        output = generator.generate_backtest_report(df, results, trades_df, args.output, args.title)
        print(f"Generated backtest report: {output}")

    elif args.training_log:
        training_df = pd.read_csv(args.training_log)
        training_history = {
            'loss': training_df.get('loss', []).tolist(),
            'val_loss': training_df.get('val_loss', []).tolist()
        }
        predictions_df = training_df[['actual', 'predicted']] if 'actual' in training_df.columns else pd.DataFrame()

        output = generator.generate_training_report(training_history, predictions_df, args.output, args.title)
        print(f"Generated training report: {output}")

    else:
        print("Please specify --backtest-results or --training-log")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
