"""
Tests for strategy optimization functionality.
"""

import pytest
import numpy as np


class TestOptimizationConfig:
    """Tests for optimization configuration."""

    def test_config_has_required_params(self):
        """Test that OptimizationConfig has required parameters."""
        from app.strategies.optimizer import OptimizationConfig

        assert hasattr(OptimizationConfig, 'PARAMS_TO_OPTIMIZE')
        assert hasattr(OptimizationConfig, 'BOOLEAN_PARAMS')
        assert hasattr(OptimizationConfig, 'FIXED_PARAMS')

    def test_params_to_optimize_format(self):
        """Test that PARAMS_TO_OPTIMIZE has correct format [min, max, step]."""
        from app.strategies.optimizer import OptimizationConfig

        for param, values in OptimizationConfig.PARAMS_TO_OPTIMIZE.items():
            assert len(values) == 3, f"{param} should have [min, max, step]"
            assert values[0] <= values[1], f"{param}: min should be <= max"
            assert values[2] > 0, f"{param}: step should be positive"

    def test_boolean_params_are_lists(self):
        """Test that BOOLEAN_PARAMS contains boolean lists."""
        from app.strategies.optimizer import OptimizationConfig

        for param, values in OptimizationConfig.BOOLEAN_PARAMS.items():
            assert isinstance(values, list), f"{param} should be a list"
            for v in values:
                assert isinstance(v, bool), f"{param} should contain booleans"

    def test_fixed_params_has_defaults(self):
        """Test that FIXED_PARAMS has essential defaults."""
        from app.strategies.optimizer import OptimizationConfig

        required = ['INITIAL_CAPITAL', 'SYMBOL']
        for param in required:
            assert param in OptimizationConfig.FIXED_PARAMS, \
                f"Missing required fixed param: {param}"

    def test_initial_capital_positive(self):
        """Test that initial capital is positive."""
        from app.strategies.optimizer import OptimizationConfig

        assert OptimizationConfig.FIXED_PARAMS['INITIAL_CAPITAL'] > 0

    def test_optimization_settings(self):
        """Test optimization settings are reasonable."""
        from app.strategies.optimizer import OptimizationConfig

        assert OptimizationConfig.TOP_N_RESULTS > 0
        assert OptimizationConfig.MIN_TRADES_REQUIRED > 0


class TestParameterGeneration:
    """Tests for parameter combination generation."""

    def test_parameter_range_generation(self):
        """Test that parameter ranges are generated correctly."""
        # Example: [3, 5, 1] should generate [3, 4, 5]
        min_val, max_val, step = 3, 5, 1
        values = list(np.arange(min_val, max_val + step, step))

        assert 3 in values
        assert 4 in values
        assert 5 in values

    def test_float_parameter_range(self):
        """Test float parameter range generation."""
        # Example: [0.03, 0.07, 0.02] should generate [0.03, 0.05, 0.07]
        min_val, max_val, step = 0.03, 0.07, 0.02
        values = list(np.arange(min_val, max_val + step/2, step))

        assert len(values) == 3
        assert abs(values[0] - 0.03) < 0.001
        assert abs(values[1] - 0.05) < 0.001
        assert abs(values[2] - 0.07) < 0.001


class TestOptimizerIntegration:
    """Integration tests for optimizer (marked for optional running)."""

    @pytest.mark.slow
    def test_generate_param_combinations(self):
        """Test that parameter combinations can be generated."""
        from itertools import product

        # Small test case
        params = {
            'A': [1, 3, 1],  # 3 values: 1, 2, 3
            'B': [0.1, 0.2, 0.1],  # 2 values: 0.1, 0.2
        }

        ranges = []
        for param, (min_val, max_val, step) in params.items():
            values = list(np.arange(min_val, max_val + step/2, step))
            ranges.append(values)

        combinations = list(product(*ranges))

        # Should have 3 * 2 = 6 combinations
        assert len(combinations) == 6

    @pytest.mark.slow
    def test_combination_count_reasonable(self):
        """Test that total combinations are reasonable for optimization."""
        from app.strategies.optimizer import OptimizationConfig
        from itertools import product
        import numpy as np

        total_combinations = 1

        # Count numeric params
        for param, (min_val, max_val, step) in OptimizationConfig.PARAMS_TO_OPTIMIZE.items():
            values = list(np.arange(min_val, max_val + step/2, step))
            total_combinations *= len(values)

        # Count boolean params
        for param, values in OptimizationConfig.BOOLEAN_PARAMS.items():
            total_combinations *= len(values)

        # Should be manageable (less than 1 million combinations typically)
        # This is a sanity check to ensure we don't accidentally create
        # an impossibly large search space
        assert total_combinations < 10_000_000, \
            f"Too many combinations: {total_combinations}"
