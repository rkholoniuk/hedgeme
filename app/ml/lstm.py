#!/usr/bin/env python3
"""
Train LSTM model for volume prediction.

This implements the two-layer LSTM architecture that predicts future volume
and volume derivatives (velocity and acceleration).

Usage:
    python -m app.ml.lstm --features ".tmp/features/BTCUSDT_15m_features.csv"
    python -m app.ml.lstm --features data.csv --lookback 20 --epochs 100
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import yaml
from typing import Tuple, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VolumeLSTM:
    """
    LSTM model for volume prediction.

    This network learns temporal patterns in volume data to predict
    future volume and its derivatives (velocity and acceleration).
    """

    def __init__(
        self,
        hidden_size: int = 32,
        lookback: int = 10,
        dropout: float = 0.2
    ):
        """
        Initialize the model.

        Args:
            hidden_size: Number of LSTM neurons (first layer)
            lookback: How many previous candles to consider
            dropout: Regularization to prevent overfitting
        """
        self.hidden_size = hidden_size
        self.lookback = lookback
        self.dropout = dropout

        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def build_model(self) -> keras.Model:
        """
        Construct the neural network architecture.

        Architecture (from article):
        - Layer 1: LSTM(32 units, return_sequences=True) - learns short-term patterns
        - Layer 2: LSTM(16 units, return_sequences=False) - learns meta-patterns
        - Dense: Dense(16, relu) + Dropout
        - Output: Dense(1, linear) - single volume prediction

        Returns:
            Compiled Keras model
        """
        logger.info("Building LSTM model...")

        model = keras.Sequential([
            # First LSTM layer - learns short-term patterns
            # (e.g., "volume has been rising for 3 candles")
            layers.LSTM(
                self.hidden_size,
                return_sequences=True,
                input_shape=(self.lookback, 1)
            ),

            # Second LSTM layer - learns meta-patterns
            # (e.g., "this type of volume rise usually leads to a spike")
            layers.LSTM(
                self.hidden_size // 2,  # 16 units
                return_sequences=False
            ),

            # Dense layer for final processing
            layers.Dense(16, activation='relu'),
            layers.Dropout(self.dropout),

            # Output layer - single volume prediction
            layers.Dense(1, activation='linear')
        ])

        # Use Huber loss instead of MSE
        # More robust to outliers (ignores crazy spikes instead of
        # letting them dominate the learning process)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']  # Mean Absolute Error for monitoring
        )

        self.model = model

        logger.info(f"Model built with {model.count_params():,} parameters")
        model.summary(print_fn=logger.info)

        return model

    def prepare_sequences(
        self,
        volumes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sliding window sequences for LSTM training.

        Args:
            volumes: Array of volume values

        Returns:
            Tuple of (X, y) where:
            - X: (n_samples, lookback, 1) - input sequences
            - y: (n_samples, 1) - target values (next volume)
        """
        if len(volumes) < self.lookback + 100:
            raise ValueError(
                f"Need at least {self.lookback + 100} historical candles. Got {len(volumes)}"
            )

        # Normalize volumes
        volumes_scaled = self.scaler.fit_transform(volumes.reshape(-1, 1))

        X, y = [], []

        # Create sliding windows
        for i in range(self.lookback, len(volumes_scaled)):
            # Input: last N volumes
            X.append(volumes_scaled[i - self.lookback:i])
            # Target: next volume
            y.append(volumes_scaled[i])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Prepared {len(X)} sequences (shape: {X.shape})")

        return X, y

    def train(
        self,
        volumes: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the model on historical volume data.

        Args:
            volumes: Array of volume values
            epochs: Maximum training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation

        Returns:
            Training history dict
        """
        logger.info("Training LSTM model...")

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Prepare sequences
        X, y = self.prepare_sequences(volumes)
        logger.info(f"Starting training with {len(X)} samples, batch_size={batch_size}")

        # Callbacks for smart training
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,  # Stop if no improvement for 10 epochs
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Cut learning rate in half
            patience=5,
            min_lr=0.00001,
            verbose=1
        )

        # Train the model
        logger.info("Calling model.fit() - this is where it might hang...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        self.is_trained = True

        logger.info("Training complete")

        return history.history

    def predict_next_volume(self, recent_volumes: np.ndarray) -> float:
        """
        Predict the next volume value.

        Args:
            recent_volumes: Array of last N volume values (length = lookback)

        Returns:
            Predicted next volume
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        if len(recent_volumes) != self.lookback:
            raise ValueError(f"Expected {self.lookback} volumes, got {len(recent_volumes)}")

        # Scale input
        volumes_scaled = self.scaler.transform(recent_volumes.reshape(-1, 1))

        # Reshape for LSTM: (1, lookback, 1)
        X = volumes_scaled.reshape(1, self.lookback, 1)

        # Predict
        pred_scaled = self.model.predict(X, verbose=0)

        # Inverse transform to original scale
        pred = self.scaler.inverse_transform(pred_scaled)[0, 0]

        return pred

    def predict_derivatives(self, recent_volumes: np.ndarray) -> Dict:
        """
        Predict not just volume, but its velocity and acceleration.

        This is the "secret sauce" - predicting derivatives gives us
        forward-looking acceleration signals.

        Args:
            recent_volumes: Array of last N volume values

        Returns:
            Dict with predictions and acceleration signals
        """
        if len(recent_volumes) < 3:
            raise ValueError("Need at least 3 recent volumes for derivative calculation")

        # Predict the next volume value
        predicted_volume = self.predict_next_volume(recent_volumes[-self.lookback:])

        # Calculate current state
        current_volume = recent_volumes[-1]
        prev_volume = recent_volumes[-2]
        current_first_der = current_volume - prev_volume

        # Predict first derivative (velocity)
        predicted_first_der = predicted_volume - current_volume

        # Calculate current second derivative (acceleration)
        if len(recent_volumes) >= 3:
            prev_first_der = prev_volume - recent_volumes[-3]
            current_second_der = current_first_der - prev_first_der
        else:
            current_second_der = 0

        # Predict second derivative (acceleration change)
        predicted_second_der = predicted_first_der - current_first_der

        # Generate signals based on predictions
        # These are the money signals - is volume about to accelerate?
        is_accelerating_positive = (
            predicted_first_der > current_first_der and
            predicted_second_der > current_second_der and
            predicted_first_der > 0
        )

        is_accelerating_negative = (
            predicted_first_der < current_first_der and
            predicted_second_der > current_second_der and
            predicted_first_der < 0
        )

        return {
            'predicted_volume': predicted_volume,
            'current_volume': current_volume,
            'predicted_first_derivative': predicted_first_der,
            'current_first_derivative': current_first_der,
            'predicted_second_derivative': predicted_second_der,
            'current_second_derivative': current_second_der,
            'is_accelerating_positive': is_accelerating_positive,
            'is_accelerating_negative': is_accelerating_negative
        }

    def save(self, model_path: str, scaler_path: str) -> None:
        """
        Save trained model and scaler.

        Args:
            model_path: Path to save model (.h5 or .keras)
            scaler_path: Path to save scaler (.pkl)
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        # Create directories
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    @classmethod
    def load(cls, model_path: str, scaler_path: str) -> 'VolumeLSTM':
        """
        Load trained model and scaler.

        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler

        Returns:
            Loaded VolumeLSTM instance
        """
        # Load model
        model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load scaler
        scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")

        # Create instance
        instance = cls()
        instance.model = model
        instance.scaler = scaler
        instance.is_trained = True

        return instance


def plot_training_history(history: Dict, output_path: str) -> None:
    """
    Plot training and validation loss curves.

    Args:
        history: Training history dict
        output_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Huber)')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # MAE
    ax2.plot(history['mae'], label='Training MAE')
    ax2.plot(history['val_mae'], label='Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Training & Validation MAE')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Training plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM model for volume prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default parameters (Supported: BTCUSDT, ETHUSDT, SOLUSDT)
  python tools/trading/train_lstm_model.py --features ".tmp/features/BTCUSDT_15m_features.csv"

  # Custom lookback and epochs
  python tools/trading/train_lstm_model.py --features data.csv --lookback 20 --epochs 100

  # Custom model output path
  python tools/trading/train_lstm_model.py --features data.csv --output "models/my_model.keras"
        """
    )

    parser.add_argument('--features', type=str, required=True, help='Input CSV file with features')
    parser.add_argument('--lookback', type=int, default=10, help='Lookback window size (default: 10)')
    parser.add_argument('--epochs', type=int, default=50, help='Max training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--hidden-size', type=int, default=32, help='LSTM hidden size (default: 32)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--output', type=str, help='Model output path (default: models/lstm_volume_predictor.keras)')

    args = parser.parse_args()

    # Default output paths
    if args.output:
        model_path = args.output
    else:
        model_path = "models/lstm_volume_predictor.keras"

    scaler_path = str(Path(model_path).with_suffix('.scaler.pkl'))
    plot_path = str(Path(model_path).with_suffix('.training.png'))

    try:
        # Load features
        df = pd.read_csv(args.features)
        logger.info(f"Loaded features from {args.features}")

        if 'volume' not in df.columns:
            raise ValueError("Features file must contain 'volume' column")

        # Drop rows with NaN values (from derivative calculations)
        df_clean = df.dropna(subset=['volume'])
        logger.info(f"Dropped {len(df) - len(df_clean)} rows with NaN values")

        if len(df_clean) < args.lookback + 100:
            raise ValueError(
                f"Insufficient data after cleaning. Need at least {args.lookback + 100} rows, got {len(df_clean)}. "
                f"Fetch more historical data with a longer date range."
            )

        volumes = df_clean['volume'].values

        # Create and train model
        lstm = VolumeLSTM(
            hidden_size=args.hidden_size,
            lookback=args.lookback,
            dropout=args.dropout
        )

        history = lstm.train(
            volumes=volumes,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        # Save model
        lstm.save(model_path, scaler_path)

        # Plot training history
        plot_training_history(history, plot_path)

        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"\nModel: {model_path}")
        print(f"Scaler: {scaler_path}")
        print(f"Plot: {plot_path}")
        print(f"\nFinal training loss: {history['loss'][-1]:.6f}")
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        print(f"Final validation MAE: {history['val_mae'][-1]:.2f}")
        print("\n" + "="*60 + "\n")

        # Test prediction
        logger.info("Testing prediction...")
        recent_volumes = volumes[-args.lookback:]
        pred_dict = lstm.predict_derivatives(volumes[-args.lookback:])

        print("Test Prediction:")
        print(f"  Current volume: {pred_dict['current_volume']:.0f}")
        print(f"  Predicted volume: {pred_dict['predicted_volume']:.0f}")
        print(f"  Acceleration signal: {'POSITIVE ⬆' if pred_dict['is_accelerating_positive'] else 'NEGATIVE ⬇' if pred_dict['is_accelerating_negative'] else 'NEUTRAL —'}")

        return 0

    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
