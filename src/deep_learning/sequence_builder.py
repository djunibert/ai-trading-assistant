"""
Sequence Builder.

Transforme un dataset tabulaire en séquences pour LSTM.

Exemple :
sequence_length = 32

Le modèle reçoit 32 bougies passées
et prédit la classe de la bougie suivante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SequenceBuilder:
    def __init__(self, sequence_length: int = 32):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()

    def build(
        self,
        df: pd.DataFrame,
        target_column: str = "target",
        columns_to_exclude: list[str] | None = None,
    ):
        df = df.copy()

        if target_column not in df.columns:
            raise ValueError(f"Colonne target manquante : {target_column}")

        if columns_to_exclude is None:
            columns_to_exclude = []

        columns_to_drop = [target_column] + columns_to_exclude

        X_df = df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            errors="ignore",
        )

        y = df[target_column].copy()

        X_df = X_df.select_dtypes(include=["number"]).fillna(0)

        X_scaled = self.scaler.fit_transform(X_df)

        X_sequences = []
        y_sequences = []

        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i - self.sequence_length:i])
            y_sequences.append(y.iloc[i])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        return X_sequences, y_sequences, X_df.columns.tolist(), self.scaler