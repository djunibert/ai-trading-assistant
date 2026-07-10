"""
LSTM Trainer.

Gère l'entraînement du modèle LSTM.
"""

from __future__ import annotations

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class LSTMTrainer:
    def __init__(self, epochs: int = 10, batch_size: int = 64):
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, model, X_train, y_train, X_val, y_val):
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=0.00001,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        return history