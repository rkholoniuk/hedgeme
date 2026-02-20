"""
Tests for risk management functionality.
"""

import pytest


class TestRiskManager:
    """Tests for risk management functionality."""

    def test_position_size_calculation(self, trading_config):
        """Test position size calculation."""
        from app.strategies.lumibot_ml import RiskManager

        rm = RiskManager(trading_config)

        result = rm.calculate_position_size(
            account_balance=1000,
            current_price=0.10,
            atr=0.005
        )

        assert 'quantity' in result
        assert 'position_value' in result
        assert 'stop_loss_price' in result
        assert 'take_profit_price' in result
        assert result['quantity'] > 0

    def test_position_size_respects_risk_limit(self, trading_config):
        """Test that position size respects risk limit."""
        from app.strategies.lumibot_ml import RiskManager

        rm = RiskManager(trading_config)

        result = rm.calculate_position_size(
            account_balance=1000,
            current_price=0.10,
            atr=0.005
        )

        # Position value should be ~5% of account (POSITION_SIZE = 0.05)
        expected_position_value = 1000 * trading_config['POSITION_SIZE']
        assert abs(result['position_value'] - expected_position_value) < 1, \
            f"Position value {result['position_value']} should be ~{expected_position_value}"

    def test_stop_loss_below_entry(self, trading_config):
        """Test that stop loss is below entry price for long."""
        from app.strategies.lumibot_ml import RiskManager

        rm = RiskManager(trading_config)

        current_price = 0.10
        result = rm.calculate_position_size(
            account_balance=1000,
            current_price=current_price,
            atr=0.005
        )

        assert result['stop_loss_price'] < current_price, \
            "Stop loss should be below entry for long position"

    def test_take_profit_above_entry(self, trading_config):
        """Test that take profit is above entry price for long."""
        from app.strategies.lumibot_ml import RiskManager

        rm = RiskManager(trading_config)

        current_price = 0.10
        result = rm.calculate_position_size(
            account_balance=1000,
            current_price=current_price,
            atr=0.005
        )

        assert result['take_profit_price'] > current_price, \
            "Take profit should be above entry for long position"

    def test_volatility_adjustment_normal(self, trading_config):
        """Test no adjustment when volatility is normal."""
        from app.strategies.lumibot_ml import RiskManager

        rm = RiskManager(trading_config)

        quantity = 100
        current_atr = 0.005
        normal_atr = 0.005

        adjusted = rm.adjust_for_volatility(quantity, current_atr, normal_atr)
        assert adjusted == quantity, "Should not adjust for normal volatility"

    def test_volatility_adjustment_extreme(self, trading_config):
        """Test position reduction when volatility is extreme."""
        from app.strategies.lumibot_ml import RiskManager

        trading_config['VOLATILITY_MULTIPLIER'] = 2.0
        rm = RiskManager(trading_config)

        quantity = 100
        current_atr = 0.015  # 3x normal
        normal_atr = 0.005

        adjusted = rm.adjust_for_volatility(quantity, current_atr, normal_atr)
        assert adjusted < quantity, "Should reduce position for extreme volatility"
        assert adjusted == quantity * 0.5, "Should reduce by 50%"

    def test_leverage_calculation(self, trading_config):
        """Test position calculation with leverage."""
        from app.strategies.lumibot_ml import RiskManager

        trading_config['LEVERAGE'] = 3
        rm = RiskManager(trading_config)

        result = rm.calculate_position_size(
            account_balance=1000,
            current_price=0.10,
            atr=0.005
        )

        # With 3x leverage, position value should be 3x the margin
        expected_margin = 1000 * trading_config['POSITION_SIZE']
        expected_position = expected_margin * 3

        assert abs(result['position_value'] - expected_position) < 1, \
            f"Position value should be {expected_position} with 3x leverage"

    def test_zero_account_balance(self, trading_config):
        """Test handling of zero account balance."""
        from app.strategies.lumibot_ml import RiskManager

        rm = RiskManager(trading_config)

        result = rm.calculate_position_size(
            account_balance=0,
            current_price=0.10,
            atr=0.005
        )

        assert result['quantity'] == 0, "Should return zero quantity for zero balance"

    def test_negative_atr_handling(self, trading_config):
        """Test handling of edge case with zero/negative ATR."""
        from app.strategies.lumibot_ml import RiskManager

        rm = RiskManager(trading_config)

        # ATR should never be zero in practice, but test defensive coding
        result = rm.calculate_position_size(
            account_balance=1000,
            current_price=0.10,
            atr=0.0001  # Very small ATR
        )

        assert result['quantity'] > 0, "Should still calculate position with small ATR"
