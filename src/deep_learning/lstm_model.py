"""
LSTM Model.

Construit l'architecture du modèle LSTM.
"""

from __future__ import annotations

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    sequence_length: int,
    n_features: int,
    n_classes: int = 3,
):
    model = Sequential()

    model.add(
        LSTM(
            units=64,
            return_sequences=True,
            input_shape=(sequence_length, n_features),
        )
    )

    model.add(BatchNormalization())
    model.add(Dropout(0.30))

    model.add(
        LSTM(
            units=32,
            return_sequences=False,
        )
    )

    model.add(BatchNormalization())
    model.add(Dropout(0.30))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.20))

    model.add(Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model