"""
Tests for HTML report generation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestHTMLReportGenerator:
    """Tests for HTML report generation functionality."""

    def test_generator_initialization(self):
        """Test that generator initializes correctly."""
        from app.reporting.html import HTMLReportGenerator

        gen = HTMLReportGenerator()
        assert gen.report_date is not None

    def _prepare_df(self, sample_features_data):
        """Prepare DataFrame for report generation."""
        df = sample_features_data.copy()
        # Drop timestamp column if it exists (it's already the index)
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        return df

    def test_generate_backtest_report(self, sample_features_data, tmp_output_dir):
        """Test backtest report generation."""
        from app.reporting.html import HTMLReportGenerator

        gen = HTMLReportGenerator()

        # Prepare test data
        df = self._prepare_df(sample_features_data)
        df['Signal_Final'] = 0

        results = {
            'total_return': 0.15,
            'total_trades': 10,
            'win_rate': 0.6,
            'avg_win': 0.02,
            'avg_loss': -0.01,
            'profit_factor': 1.5,
            'max_drawdown': 0.05,
            'sharpe_ratio': 1.2,
            'final_capital': 11500,
            'equity_df': pd.DataFrame({
                'timestamp': df.index[:10],
                'equity': np.linspace(10000, 11500, 10)
            }),
            'trades_df': pd.DataFrame({
                'entry_time': df.index[:5],
                'exit_time': df.index[5:10],
                'side': ['long'] * 5,
                'pnl': [100, -50, 200, 150, -100],
                'pnl_pct': [0.01, -0.005, 0.02, 0.015, -0.01]
            })
        }

        trades_df = results['trades_df']
        output_path = tmp_output_dir / "test_report.html"

        result_path = gen.generate_backtest_report(
            df=df,
            results=results,
            trades_df=trades_df,
            output_path=str(output_path),
            title="Test Report"
        )

        assert Path(result_path).exists()
        # Check file has content
        content = Path(result_path).read_text()
        assert len(content) > 0
        assert "Test Report" in content

    def test_report_contains_metrics(self, sample_features_data, tmp_output_dir):
        """Test that report contains performance metrics."""
        from app.reporting.html import HTMLReportGenerator

        gen = HTMLReportGenerator()

        df = self._prepare_df(sample_features_data)
        df['Signal_Final'] = 0

        results = {
            'total_return': 0.15,
            'total_trades': 10,
            'win_rate': 0.6,
            'avg_win': 0.02,
            'avg_loss': -0.01,
            'profit_factor': 1.5,
            'max_drawdown': 0.05,
            'sharpe_ratio': 1.2,
            'final_capital': 11500,
            'equity_df': pd.DataFrame(),
            'trades_df': pd.DataFrame()
        }

        trades_df = pd.DataFrame()
        output_path = tmp_output_dir / "metrics_report.html"

        result_path = gen.generate_backtest_report(
            df=df,
            results=results,
            trades_df=trades_df,
            output_path=str(output_path)
        )

        content = Path(result_path).read_text()

        # Check metrics are present in report
        assert "15" in content or "0.15" in content  # total_return
        assert "10" in content  # total_trades
        assert "60" in content or "0.6" in content  # win_rate

    def test_report_handles_empty_trades(self, sample_features_data, tmp_output_dir):
        """Test report generation with no trades."""
        from app.reporting.html import HTMLReportGenerator

        gen = HTMLReportGenerator()

        df = self._prepare_df(sample_features_data)
        df['Signal_Final'] = 0

        results = {
            'total_return': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'final_capital': 10000,
            'equity_df': pd.DataFrame(),
            'trades_df': pd.DataFrame()
        }

        trades_df = pd.DataFrame()
        output_path = tmp_output_dir / "empty_trades_report.html"

        # Should not raise error
        result_path = gen.generate_backtest_report(
            df=df,
            results=results,
            trades_df=trades_df,
            output_path=str(output_path)
        )

        assert Path(result_path).exists()

    def test_report_is_valid_html(self, sample_features_data, tmp_output_dir):
        """Test that generated report is valid HTML."""
        from app.reporting.html import HTMLReportGenerator

        gen = HTMLReportGenerator()

        df = self._prepare_df(sample_features_data)
        df['Signal_Final'] = 0

        results = {
            'total_return': 0.1,
            'total_trades': 5,
            'win_rate': 0.5,
            'avg_win': 0.01,
            'avg_loss': -0.01,
            'profit_factor': 1.0,
            'max_drawdown': 0.03,
            'sharpe_ratio': 0.8,
            'final_capital': 10500,
            'equity_df': pd.DataFrame(),
            'trades_df': pd.DataFrame()
        }

        output_path = tmp_output_dir / "valid_html_report.html"

        result_path = gen.generate_backtest_report(
            df=df,
            results=results,
            trades_df=pd.DataFrame(),
            output_path=str(output_path)
        )

        content = Path(result_path).read_text()

        # Basic HTML structure checks
        assert "<!DOCTYPE html>" in content or "<!doctype html>" in content.lower()
        assert "<html" in content.lower()
        assert "</html>" in content.lower()
        assert "<head>" in content.lower()
        assert "</head>" in content.lower()
        assert "<body" in content.lower()
        assert "</body>" in content.lower()


class TestReportCharts:
    """Tests for chart generation in reports."""

    def _prepare_df(self, sample_features_data):
        """Prepare DataFrame for report generation."""
        df = sample_features_data.copy()
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        return df

    def test_metrics_table_created(self, sample_features_data, tmp_output_dir):
        """Test that metrics table is included in report."""
        from app.reporting.html import HTMLReportGenerator

        gen = HTMLReportGenerator()

        results = {
            'total_return': 0.25,
            'total_trades': 15,
            'win_rate': 0.65,
            'avg_win': 0.03,
            'avg_loss': -0.015,
            'profit_factor': 2.0,
            'max_drawdown': 0.08,
            'sharpe_ratio': 1.5,
            'final_capital': 12500,
            'equity_df': pd.DataFrame(),
            'trades_df': pd.DataFrame()
        }

        df = self._prepare_df(sample_features_data)
        df['Signal_Final'] = 0
        output_path = tmp_output_dir / "table_report.html"

        result_path = gen.generate_backtest_report(
            df=df,
            results=results,
            trades_df=pd.DataFrame(),
            output_path=str(output_path)
        )

        content = Path(result_path).read_text()

        # Should contain table elements
        assert "<table" in content.lower() or "metrics" in content.lower()
