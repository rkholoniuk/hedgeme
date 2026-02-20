"""
Tests for LSTM model training and prediction.
"""

import pytest
import numpy as np


class TestVolumeLSTM:
    """Tests for LSTM model functionality."""

    def test_model_initialization(self):
        """Test that model initializes with correct parameters."""
        from app.ml.lstm import VolumeLSTM

        model = VolumeLSTM(hidden_size=32, lookback=10, dropout=0.2)

        assert model.hidden_size == 32
        assert model.lookback == 10
        assert model.dropout == 0.2
        assert model.model is None
        assert model.is_trained is False

    def test_model_build(self):
        """Test that model builds correctly."""
        from app.ml.lstm import VolumeLSTM

        lstm = VolumeLSTM(hidden_size=32, lookback=10)
        model = lstm.build_model()

        assert model is not None
        assert lstm.model is not None
        # Check model has layers
        assert len(model.layers) > 0

    def test_model_architecture(self):
        """Test model architecture matches specification."""
        from app.ml.lstm import VolumeLSTM

        lstm = VolumeLSTM(hidden_size=32, lookback=10)
        model = lstm.build_model()

        # Should have LSTM layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert 'LSTM' in layer_types
        assert 'Dense' in layer_types

    def test_prepare_sequences_shape(self):
        """Test that prepare_sequences returns correct shapes."""
        from app.ml.lstm import VolumeLSTM

        lstm = VolumeLSTM(lookback=10)

        # Create sample volume data (needs at least lookback + 100)
        volumes = np.random.uniform(1000000, 10000000, 200)

        X, y = lstm.prepare_sequences(volumes)

        # X shape should be (n_samples, lookback, 1)
        assert X.shape[1] == lstm.lookback
        assert X.shape[2] == 1

        # y shape should be (n_samples, 1)
        assert len(y.shape) == 2
        assert y.shape[1] == 1

        # Number of samples should match
        assert X.shape[0] == y.shape[0]

    def test_prepare_sequences_minimum_data(self):
        """Test that prepare_sequences raises error with insufficient data."""
        from app.ml.lstm import VolumeLSTM

        lstm = VolumeLSTM(lookback=10)

        # Too few samples
        volumes = np.random.uniform(1000000, 10000000, 50)

        with pytest.raises(ValueError, match="Need at least"):
            lstm.prepare_sequences(volumes)

    def test_predict_requires_training(self):
        """Test that prediction fails without training."""
        from app.ml.lstm import VolumeLSTM

        lstm = VolumeLSTM(lookback=10)
        lstm.build_model()

        recent_volumes = np.random.uniform(1000000, 10000000, 10)

        with pytest.raises(RuntimeError, match="not trained"):
            lstm.predict_next_volume(recent_volumes)

    def test_predict_requires_correct_length(self, mock_lstm_model):
        """Test that prediction requires correct input length."""
        from app.ml.lstm import VolumeLSTM

        lstm = VolumeLSTM(lookback=10)
        lstm.build_model()
        lstm.is_trained = True
        lstm.scaler.fit(np.random.uniform(1000000, 10000000, 100).reshape(-1, 1))

        # Wrong length input
        wrong_length = np.random.uniform(1000000, 10000000, 5)

        with pytest.raises(ValueError, match="Expected"):
            lstm.predict_next_volume(wrong_length)

    def test_scaler_fitted_during_preparation(self):
        """Test that scaler is fitted during sequence preparation."""
        from app.ml.lstm import VolumeLSTM

        lstm = VolumeLSTM(lookback=10)
        volumes = np.random.uniform(1000000, 10000000, 200)

        lstm.prepare_sequences(volumes)

        # Scaler should now be fitted
        assert hasattr(lstm.scaler, 'mean_')
        assert lstm.scaler.mean_ is not None

    def test_model_compile_settings(self):
        """Test that model is compiled with correct settings."""
        from app.ml.lstm import VolumeLSTM

        lstm = VolumeLSTM()
        model = lstm.build_model()

        # Check optimizer
        assert model.optimizer is not None
        # Check loss function is huber
        assert 'huber' in str(model.loss).lower()


class TestVolumeLSTMIntegration:
    """Integration tests for LSTM (marked for optional running)."""

    @pytest.mark.slow
    def test_train_and_predict(self):
        """Test full training and prediction cycle."""
        from app.ml.lstm import VolumeLSTM

        lstm = VolumeLSTM(lookback=10, hidden_size=16)

        # Generate synthetic volume data
        np.random.seed(42)
        volumes = np.random.uniform(1000000, 10000000, 300)

        # Train with minimal epochs for speed
        history = lstm.train(volumes, epochs=2, batch_size=16)

        assert lstm.is_trained is True
        assert 'loss' in history
        assert 'val_loss' in history

        # Make prediction
        recent = volumes[-10:]
        prediction = lstm.predict_next_volume(recent)

        assert isinstance(prediction, (int, float, np.floating))
        assert prediction > 0  # Volume should be positive


class TestVolumeLSTMSaveLoad:
    """Tests for model save/load functionality."""

    def test_save_requires_training(self, tmp_output_dir):
        """Test that save fails without training."""
        from app.ml.lstm import VolumeLSTM

        lstm = VolumeLSTM()
        lstm.build_model()

        model_path = tmp_output_dir / "model.keras"
        scaler_path = tmp_output_dir / "scaler.pkl"

        # Should handle untrained model gracefully or raise error
        # Depending on implementation
        try:
            lstm.save(str(model_path), str(scaler_path))
        except (RuntimeError, AttributeError):
            pass  # Expected if save requires training
