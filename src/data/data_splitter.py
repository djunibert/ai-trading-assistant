"""
Data Splitter.

Centralise la séparation train/test pour tous les modèles.
"""

from __future__ import annotations

import pandas as pd

from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42,
        stratify: bool = True,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split(
        self,
        df: pd.DataFrame,
        target_column: str = "target",
        columns_to_drop: list[str] | None = None,
    ):
        if target_column not in df.columns:
            raise ValueError(f"Colonne target manquante : {target_column}")

        if columns_to_drop is None:
            columns_to_drop = []

        X = df.drop(
            columns=[target_column] + columns_to_drop,
            errors="ignore",
        )

        y = df[target_column]

        stratify_values = y if self.stratify else None

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_values,
        )